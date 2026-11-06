//! Backend-aware source loading and complete native analysis artifacts.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

use volta_common::file_cache::FileCache;
use volta_common::report::{Report, locate_path, report_error};
use volta_expr::SymbolicOutputs;
use volta_graph::EventGraph;
use volta_graph_check::GraphReport;
use volta_pipeline::{
    Backend, KernelEvidence, KernelModule, SourceAnalysisParts, verify_source_kernel,
};
use volta_ptx_analysis::frontend::ascii::{AsAscii, AsciiChar};
use volta_ptx_analysis::frontend::ast::Module;
use volta_ptx_analysis::frontend::parse;
use volta_ptx_analysis::{AnalysisConfig, Stats};

pub(crate) struct KernelSide {
    pub(crate) label: &'static str,
    pub(crate) graph: EventGraph,
    pub(crate) report: GraphReport,
    pub(crate) early_readiness: Option<volta_pipeline::CertifiedScalarReadiness>,
    pub(crate) early_memory_conflict: Option<volta_pipeline::CertifiedMemoryConflict>,
    pub(crate) tensor_bounds: Option<volta_pipeline::CertifiedTensorBounds>,
}

impl KernelSide {
    pub(crate) fn has_certified_rejection(&self) -> bool {
        self.early_readiness.is_some()
            || self.early_memory_conflict.is_some()
            || self.tensor_bounds.is_some()
    }

    pub(crate) fn evidence(&self) -> KernelEvidence<'_> {
        KernelEvidence {
            label: self.label,
            graph: &self.report,
            assumptions: &self.graph.provenance().assumptions,
        }
    }
}

pub(crate) struct KernelRun {
    pub(crate) outputs: Result<SymbolicOutputs, String>,
    pub(crate) stats: Stats,
    pub(crate) op_counts: BTreeMap<&'static str, u64>,
}

/// Analysis stage plus graph branch for one kernel, split into its owned parts.
pub(crate) fn run_kernel(
    label: &'static str,
    module: &KernelModule,
    kernel: Option<&str>,
    config: AnalysisConfig,
) -> Result<(KernelRun, KernelSide), String> {
    let verified = verify_source_kernel(module, kernel, config)
        .map_err(|e| format!("{} kernel: {}", label, e))?;
    let SourceAnalysisParts {
        outputs,
        graph,
        stats,
        op_counts,
        ..
    } = verified.analysis;
    Ok((
        KernelRun {
            outputs,
            stats,
            op_counts,
        },
        KernelSide {
            label,
            graph,
            report: verified.graph,
            early_readiness: verified.early_readiness,
            early_memory_conflict: verified.early_memory_conflict,
            tensor_bounds: verified.tensor_bounds,
        },
    ))
}

pub(crate) fn load_module(
    file: &Path,
    backend: Backend,
) -> Result<(KernelModule, String), ExitCode> {
    if backend == Backend::Ptx {
        return load_ptx_module(file).map(|(module, source)| (KernelModule::Ptx(module), source));
    }
    let source = std::fs::read_to_string(file).map_err(|error| {
        eprintln!("error: failed to read {}: {error}", file.display());
        ExitCode::FAILURE
    })?;
    let module = volta_amdgpu_analysis::frontend::parse(&source)
        .map(KernelModule::Amdgpu)
        .map_err(|error| {
            eprintln!("error: {}: {error}", file.display());
            ExitCode::FAILURE
        })?;
    if let KernelModule::Amdgpu(module) = &module {
        backend
            .validate_amd_target(module.target.as_deref())
            .map_err(|error| {
                eprintln!("error: {}: {error}", file.display());
                ExitCode::FAILURE
            })?;
    }
    Ok((module, source))
}

fn load_ptx_module(file: &Path) -> Result<(Module, String), ExitCode> {
    let mut files = FileCache::new();
    let contents = match files.read(file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to read {}: {}", file.display(), e);
            return Err(ExitCode::FAILURE);
        }
    };
    let ascii_src: &[AsciiChar] = match contents.as_bytes().as_ascii_slice() {
        Some(src) => src,
        None => {
            eprintln!("error: file contains non-ASCII character");
            return Err(ExitCode::FAILURE);
        }
    };
    let mut parser = parse::Parser::new(ascii_src);
    match parser.parse_module().map_err(locate_path(file)) {
        Ok(module) => Ok((module, contents.to_string())),
        Err(e) => {
            let _ = report_error(
                &mut std::io::stderr(),
                &files,
                Report {
                    path: e.path.as_deref(),
                    span: e.span,
                    title: e.error.title(),
                    message: e.error.message().as_deref(),
                },
            );
            Err(ExitCode::FAILURE)
        }
    }
}

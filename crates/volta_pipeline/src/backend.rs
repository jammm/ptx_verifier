//! Thin ISA dispatch above the existing artifact-producing analysis stages.
use std::collections::BTreeMap;

use volta_expr::SymbolicOutputs;
use volta_graph::EventGraph;
use volta_graph_check::amdgpu::global_bounds::{
    GlobalBoundsCertificate, GlobalBoundsViolation, global_bounds_candidate,
    validate_global_bounds_prefix,
};
use volta_graph_check::amdgpu::tensor_bounds::{
    TensorBoundsCertificate, TensorBoundsViolation, tensor_bounds_candidate,
    validate_tensor_bounds_prefix,
};
use volta_graph_check::amdgpu::{
    AmdRegisterHazard,
    early_memory_conflicts::{
        MemoryConflictCertificate, MemoryConflictObserver, validate_memory_conflict_prefix,
    },
    early_readiness::{
        ScalarReadinessCertificate, ScalarReadinessObserver, validate_scalar_readiness_prefix,
    },
};
use volta_graph_check::memory_conflict::MemoryConflict;
use volta_graph_check::{GraphReport, check_graph};
use volta_ptx_analysis::{AnalysisConfig, Stats};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    Ptx,
    /// Native AMD gfx950 (the original AMD backend selection).
    Amdgpu,
    /// Native AMD gfx942; selection is checked against the source target.
    AmdgpuGfx942,
    /// Native AMD gfx1250; wave32 semantics are selected by the source target.
    AmdgpuGfx1250,
}

impl Backend {
    pub fn select(name: &str, target: Option<&str>) -> Result<Self, String> {
        match (name, target) {
            ("ptx", None) => Ok(Self::Ptx),
            ("ptx", Some(_)) => Err("--target is only supported with the amdgpu backend; PTX declares its target in source".into()),
            ("amdgpu", None | Some("gfx950")) => Ok(Self::Amdgpu),
            ("amdgpu", Some("gfx942")) => Ok(Self::AmdgpuGfx942),
            ("amdgpu", Some("gfx1250")) => Ok(Self::AmdgpuGfx1250),
            ("amdgpu", Some(target)) => Err(format!("unsupported AMD target '{target}'; expected gfx942, gfx950 or gfx1250")),
            (name, _) => Err(format!("unknown backend '{name}'; expected ptx or amdgpu")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Ptx => "ptx",
            Self::Amdgpu | Self::AmdgpuGfx942 | Self::AmdgpuGfx1250 => "amdgpu",
        }
    }

    pub fn target(self) -> Option<&'static str> {
        match self {
            Self::Ptx => None,
            Self::Amdgpu => Some("gfx950"),
            Self::AmdgpuGfx942 => Some("gfx942"),
            Self::AmdgpuGfx1250 => Some("gfx1250"),
        }
    }

    /// Keep explicit API/CLI selection and source architecture consistent. A
    /// missing source directive retains the original gfx950 default.
    pub fn validate_amd_target(self, source_target: Option<&str>) -> Result<(), String> {
        let selected = self
            .target()
            .ok_or("PTX selection cannot load AMD assembly")?;
        let declared = source_target.unwrap_or("gfx950").split(':').next().unwrap();
        if selected != declared {
            return Err(format!(
                "selected target {selected} conflicts with source target {declared}"
            ));
        }
        Ok(())
    }
}

pub enum KernelModule {
    Ptx(volta_ptx_analysis::frontend::ast::Module),
    Amdgpu(volta_amdgpu_analysis::frontend::Module),
}

/// Common owned artifacts. Normalization happens here, never in an analyzer.
pub struct SourceAnalysisParts {
    pub outputs: Result<SymbolicOutputs, String>,
    /// Every output word under the native raw-u32 contract, when selected.
    pub raw_outputs: Option<volta_amdgpu_analysis::RawOutputs>,
    pub graph: EventGraph,
    pub stats: Stats,
    pub op_counts: BTreeMap<&'static str, u64>,
}

pub struct SourceKernelVerification {
    pub analysis: SourceAnalysisParts,
    pub graph: GraphReport,
    /// A first finding certified against this exact incomplete graph prefix.
    /// This supplies no full-trace, output or equivalence evidence.
    pub early_readiness: Option<CertifiedScalarReadiness>,
    /// A same-instruction write/write finding certified against this prefix.
    /// This supplies no completed-trace or mathematical-output evidence.
    pub early_memory_conflict: Option<CertifiedMemoryConflict>,
    /// Independently certified invalid attempted access; outputs remain absent.
    pub tensor_bounds: Option<CertifiedTensorBounds>,
    /// Independently certified global access outside configured objects or
    /// into a guard (gfx1250 B0 m1 profile); outputs remain absent.
    pub global_bounds: Option<CertifiedGlobalBounds>,
}

/// Checker-validated memory-conflict rejection. Private fields prevent an
/// unvalidated candidate from becoming an authorized pipeline rejection.
/// Validation trusts the frontend's declared graph/provenance contract, rather
/// than independently decoding the original ISA bytes.
#[derive(Debug, Clone)]
pub struct CertifiedMemoryConflict {
    certificate: MemoryConflictCertificate,
    conflict: MemoryConflict,
}

impl CertifiedMemoryConflict {
    pub fn certificate(&self) -> &MemoryConflictCertificate {
        &self.certificate
    }

    pub fn conflict(&self) -> &MemoryConflict {
        &self.conflict
    }
}

/// Only this pipeline constructs the wrapper after replaying the exact prefix.
#[derive(Debug, Clone)]
pub struct CertifiedTensorBounds {
    certificate: TensorBoundsCertificate,
    violation: TensorBoundsViolation,
}

impl CertifiedTensorBounds {
    pub fn certificate(&self) -> &TensorBoundsCertificate {
        &self.certificate
    }
    pub fn violation(&self) -> &TensorBoundsViolation {
        &self.violation
    }
}

/// Only this pipeline constructs the wrapper after replaying the exact prefix.
#[derive(Debug, Clone)]
pub struct CertifiedGlobalBounds {
    certificate: GlobalBoundsCertificate,
    violation: GlobalBoundsViolation,
}

impl CertifiedGlobalBounds {
    pub fn certificate(&self) -> &GlobalBoundsCertificate {
        &self.certificate
    }
    pub fn violation(&self) -> &GlobalBoundsViolation {
        &self.violation
    }
}

/// Checker-validated scalar readiness rejection. Only the pipeline constructs it
/// after independently validating the final prefix. The checker trusts declared
/// graph/provenance contracts; this is not an independent raw ISA decoder.
#[derive(Debug, Clone)]
pub struct CertifiedScalarReadiness {
    certificate: ScalarReadinessCertificate,
    hazard: AmdRegisterHazard,
}

impl CertifiedScalarReadiness {
    pub fn certificate(&self) -> &ScalarReadinessCertificate {
        &self.certificate
    }

    pub fn hazard(&self) -> &AmdRegisterHazard {
        &self.hazard
    }
}

/// A candidate refusal consumes only that issue, not the observer. Keeping the
/// validator separate also permits an independent test of this control flow.
fn poll_memory_conflict(
    observer: &mut MemoryConflictObserver,
    observing: &mut bool,
    graph: &EventGraph,
    mut validate: impl FnMut(
        &EventGraph,
        &MemoryConflictCertificate,
    ) -> Result<MemoryConflict, Vec<volta_graph_check::Unresolved>>,
) -> Option<MemoryConflictCertificate> {
    if !*observing {
        return None;
    }
    match observer.observe(graph) {
        Ok(Some(certificate)) => validate(graph, &certificate).ok().map(|_| certificate),
        Err(_) => {
            *observing = false;
            None
        }
        Ok(None) => None,
    }
}

pub fn verify_source_kernel(
    module: &KernelModule,
    kernel: Option<&str>,
    config: AnalysisConfig,
) -> Result<SourceKernelVerification, String> {
    let mut early_readiness = None;
    let mut early_memory_conflict = None;
    let analysis = match module {
        KernelModule::Ptx(module) => {
            if config.native_profile != volta_amdgpu_analysis::NativeProfile::TargetDefault {
                return Err("native AMD profiles do not apply to PTX".into());
            }
            let parts = volta_ptx_analysis::analyze(module, kernel, config)
                .map_err(|error| error.to_string())?
                .into_parts();
            SourceAnalysisParts {
                outputs: parts.outputs.map_err(|error| error.to_string()),
                raw_outputs: None,
                graph: parts.graph,
                stats: parts.stats,
                op_counts: parts.op_counts,
            }
        }
        KernelModule::Amdgpu(module) => {
            let result = if config.early_readiness || config.early_memory_conflicts {
                let mut scalar_observer = ScalarReadinessObserver::new();
                let mut memory_observer = MemoryConflictObserver::new();
                let mut scalar_candidate = None;
                let mut memory_candidate = None;
                let mut observe_scalar = config.early_readiness;
                let mut observe_memory = config.early_memory_conflicts;
                let mut hook = |graph: &EventGraph| {
                    // Each observer owns its transition/discovery semantics. An
                    // observer error disables only that observer. Scalar readiness has
                    // deterministic priority when both find a valid witness.
                    if observe_scalar {
                        match scalar_observer.observe(graph) {
                            Ok(Some(certificate)) => {
                                if validate_scalar_readiness_prefix(graph, &certificate).is_ok() {
                                    scalar_candidate = Some(certificate);
                                    return std::ops::ControlFlow::Break(());
                                }
                                observe_scalar = false;
                            }
                            Err(_) => observe_scalar = false,
                            Ok(None) => {}
                        }
                    }
                    if let Some(certificate) = poll_memory_conflict(
                        &mut memory_observer,
                        &mut observe_memory,
                        graph,
                        validate_memory_conflict_prefix,
                    ) {
                        memory_candidate = Some(certificate);
                        return std::ops::ControlFlow::Break(());
                    }
                    std::ops::ControlFlow::Continue(())
                };
                let result =
                    volta_amdgpu_analysis::analyze_with_observer(module, kernel, config, &mut hook)
                        .map_err(|error| error.to_string())?;
                if let Some(certificate) = scalar_candidate {
                    let hazard = validate_scalar_readiness_prefix(result.graph(), &certificate)
                        .map_err(|reasons| {
                            format!(
                                "early readiness prefix certificate validation failed: {reasons:?}"
                            )
                        })?;
                    early_readiness = Some(CertifiedScalarReadiness {
                        certificate,
                        hazard,
                    });
                }
                if let Some(certificate) = memory_candidate {
                    let conflict = validate_memory_conflict_prefix(result.graph(), &certificate)
                        .map_err(|reasons| {
                            format!(
                                "early memory conflict prefix certificate validation failed: {reasons:?}"
                            )
                        })?;
                    early_memory_conflict = Some(CertifiedMemoryConflict {
                        certificate,
                        conflict,
                    });
                }
                result
            } else {
                volta_amdgpu_analysis::analyze(module, kernel, config)
                    .map_err(|error| error.to_string())?
            };
            let parts = result.into_parts();
            SourceAnalysisParts {
                outputs: parts.outputs.map_err(|error| error.to_string()),
                raw_outputs: parts.raw_outputs,
                graph: parts.graph,
                stats: Stats {
                    instructions: parts.stats.instructions,
                    block_syncs: parts.stats.block_syncs,
                    warp_syncs: parts.stats.warp_syncs,
                },
                op_counts: parts.op_counts,
            }
        }
    };
    let graph = check_graph(&analysis.graph);
    let tensor_bounds = tensor_bounds_candidate(&analysis.graph).and_then(|certificate| {
        validate_tensor_bounds_prefix(&analysis.graph, &certificate)
            .ok()
            .map(|violation| CertifiedTensorBounds {
                certificate,
                violation,
            })
    });
    let global_bounds = global_bounds_candidate(&analysis.graph).and_then(|certificate| {
        validate_global_bounds_prefix(&analysis.graph, &certificate)
            .ok()
            .map(|violation| CertifiedGlobalBounds {
                certificate,
                violation,
            })
    });
    Ok(SourceKernelVerification {
        analysis,
        graph,
        early_readiness,
        early_memory_conflict,
        tensor_bounds,
        global_bounds,
    })
}

#[cfg(test)]
mod tests {
    use super::Backend;

    #[test]
    fn a_refused_memory_candidate_does_not_disable_a_later_independent_issue() {
        use super::{
            MemoryConflictObserver, poll_memory_conflict, validate_memory_conflict_prefix,
        };
        use std::ops::ControlFlow;
        use volta_amdgpu_analysis::{AnalysisConfig, ArrayDef, ArrayKind, frontend};
        use volta_graph_check::Unresolved;

        let source = r#"
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.type k,@function
k:
s_mov_b32 s4,0x20000
s_mov_b32 s5,0
v_mov_b32 v2,0
v_mov_b32 v1,1.0
global_store_dword v2,v1,s[4:5]
global_store_dword v2,v1,s[4:5]
s_waitcnt vmcnt(0)
s_endpgm
.size k,.-k
.amdhsa_kernel k
.end_amdhsa_kernel
"#;
        let module = frontend::parse(source).unwrap();
        let mut config = AnalysisConfig::new((2, 1, 1));
        config.early_memory_conflicts = true;
        config.arrays.push(ArrayDef {
            name: "out".into(),
            base: 0x20000,
            elem_width: 4,
            len: 2,
            kind: ArrayKind::Output,
        });
        let mut observer = MemoryConflictObserver::new();
        let mut observing = true;
        let mut issues = Vec::new();
        let mut accepted = None;
        let mut hook = |graph: &volta_graph::EventGraph| {
            if let Some(certificate) = poll_memory_conflict(
                &mut observer,
                &mut observing,
                graph,
                |graph, certificate| {
                    // Both are genuine independently valid native candidates.
                    // Inject only the first validation refusal to exercise the
                    // retry policy without inventing transient ISA premises.
                    let conflict = validate_memory_conflict_prefix(graph, certificate)?;
                    issues.push(certificate.issue);
                    if issues.len() == 1 {
                        Err(vec![Unresolved::global(
                            "test-only first-candidate refusal",
                        )])
                    } else {
                        Ok(conflict)
                    }
                },
            ) {
                accepted = Some(certificate);
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let result =
            volta_amdgpu_analysis::analyze_with_observer(&module, Some("k"), config, &mut hook)
                .unwrap();
        assert!(observing);
        assert_eq!(issues.len(), 2, "no repeated first-pair replay");
        assert!(issues[0] < issues[1]);
        let certificate = accepted.expect("later independently valid issue");
        assert_eq!(certificate.issue, issues[1]);
        assert!(validate_memory_conflict_prefix(result.graph(), &certificate).is_ok());
        assert!(result.symbolic_outputs().is_err());
    }

    #[test]
    fn selection_is_explicit_and_target_specific() {
        assert_eq!(Backend::select("ptx", None), Ok(Backend::Ptx));
        assert_eq!(
            Backend::select("amdgpu", Some("gfx950")),
            Ok(Backend::Amdgpu)
        );
        assert_eq!(Backend::select("amdgpu", None), Ok(Backend::Amdgpu));
        assert!(Backend::select("ptx", Some("gfx950")).is_err());
        assert_eq!(
            Backend::select("amdgpu", Some("gfx942")),
            Ok(Backend::AmdgpuGfx942)
        );
        assert_eq!(
            Backend::select("amdgpu", Some("gfx1250")),
            Ok(Backend::AmdgpuGfx1250)
        );
        assert_eq!(Backend::AmdgpuGfx1250.name(), "amdgpu");
        assert_eq!(Backend::AmdgpuGfx1250.target(), Some("gfx1250"));
        assert!(
            Backend::AmdgpuGfx1250
                .validate_amd_target(Some("gfx1250"))
                .is_ok()
        );
        for other in [
            None,
            Some("gfx942"),
            Some("gfx950"),
            Some("gfx1251"),
            Some("gfx1250-strict"),
        ] {
            assert!(Backend::AmdgpuGfx1250.validate_amd_target(other).is_err());
        }
        for target in ["gfx1251", "gfx1250-strict"] {
            assert!(Backend::select("amdgpu", Some(target)).is_err());
        }
        assert!(Backend::select("ptx", Some("gfx1250")).is_err());
        assert!(Backend::select("amdgpu", Some("gfx940")).is_err());
        assert!(
            Backend::AmdgpuGfx942
                .validate_amd_target(Some("gfx942:sramecc+:xnack-"))
                .is_ok()
        );
        assert!(
            Backend::AmdgpuGfx942
                .validate_amd_target(Some("gfx950"))
                .is_err()
        );
        assert!(Backend::Amdgpu.validate_amd_target(Some("gfx942")).is_err());
        assert!(Backend::select("other", None).is_err());
    }
}

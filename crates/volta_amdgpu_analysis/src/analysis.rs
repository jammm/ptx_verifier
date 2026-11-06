//! Candidate execution and the two existing downstream artifacts, without verdicts.
use crate::{
    config::AnalysisConfig,
    eval::{
        EvalError, InstructionObserver, Interpreter, OutputsUnavailable, RawOutputs, RunArtifacts,
        Stats,
    },
    frontend::Module,
    lower::{LowerError, lower_kernel_in_module},
};
use std::{collections::BTreeMap, error::Error, fmt};
use volta_expr::{ExprArena, SymbolicOutputs};
use volta_graph::{Assumption, EventGraph, InstructionInfo, Provenance};

#[derive(Debug)]
pub enum AnalysisError {
    KernelNotFound { name: Option<String> },
    Lower(LowerError),
    Eval(EvalError),
}
impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelNotFound { name } => write!(f, "no AMD kernel matching {name:?}"),
            Self::Lower(e) => write!(f, "{e}"),
            Self::Eval(e) => write!(f, "{e}"),
        }
    }
}
impl Error for AnalysisError {}

#[derive(Debug)]
pub struct AnalysisResult {
    symbolic: SymbolicOutputs,
    outputs: Result<(), OutputsUnavailable>,
    raw_outputs: Option<RawOutputs>,
    pub graph: EventGraph,
    pub stats: Stats,
    pub op_counts: BTreeMap<&'static str, u64>,
}
impl AnalysisResult {
    fn from_run(run: RunArtifacts) -> Self {
        let (outputs, status) = match run.outputs {
            Ok(outputs) => (outputs, Ok(())),
            Err(why) => (Vec::new(), Err(why)),
        };
        Self {
            symbolic: SymbolicOutputs {
                arena: run.arena,
                outputs,
            },
            outputs: status,
            raw_outputs: run.raw_outputs,
            graph: run.graph,
            stats: run.stats,
            op_counts: run.op_counts,
        }
    }
    pub fn symbolic_outputs(&self) -> Result<&SymbolicOutputs, &OutputsUnavailable> {
        self.outputs.as_ref().map(|()| &self.symbolic)
    }
    /// Every output word under the raw-u32 contract after completed execution.
    pub fn raw_outputs(&self) -> Option<&RawOutputs> {
        self.raw_outputs.as_ref()
    }
    pub fn arena(&self) -> &ExprArena {
        &self.symbolic.arena
    }
    pub fn graph(&self) -> &EventGraph {
        &self.graph
    }
    pub fn into_parts(self) -> AnalysisParts {
        AnalysisParts {
            outputs: self.outputs.map(|()| self.symbolic),
            raw_outputs: self.raw_outputs,
            graph: self.graph,
            stats: self.stats,
            op_counts: self.op_counts,
        }
    }
}
#[derive(Debug)]
pub struct AnalysisParts {
    pub outputs: Result<SymbolicOutputs, OutputsUnavailable>,
    pub raw_outputs: Option<RawOutputs>,
    pub graph: EventGraph,
    pub stats: Stats,
    pub op_counts: BTreeMap<&'static str, u64>,
}

pub fn analyze(
    module: &Module,
    name: Option<&str>,
    config: AnalysisConfig,
) -> Result<AnalysisResult, AnalysisError> {
    analyze_internal(module, name, config, None)
}

/// Execute with a callback after each successfully completed instruction batch.
/// This requires an enabled pipeline observer option (`early_readiness` or
/// `early_memory_conflicts`) and its supported target. The pipeline supplies the
/// checker-backed observer; a callback stop alone is never a verified finding.
pub fn analyze_with_observer(
    module: &Module,
    name: Option<&str>,
    config: AnalysisConfig,
    observer: &mut InstructionObserver<'_>,
) -> Result<AnalysisResult, AnalysisError> {
    analyze_internal(module, name, config, Some(observer))
}

fn analyze_internal(
    module: &Module,
    name: Option<&str>,
    config: AnalysisConfig,
    observer: Option<&mut InstructionObserver<'_>>,
) -> Result<AnalysisResult, AnalysisError> {
    if (config.early_readiness || config.early_memory_conflicts) != observer.is_some() {
        return Err(AnalysisError::Eval(EvalError::InvalidConfig(
            "early_readiness or early_memory_conflicts requires an explicit checker-backed instruction observer".into(),
        )));
    }
    if config.early_memory_conflicts
        && !matches!(
            crate::target::Architecture::from_target(module.target.as_deref().unwrap_or("gfx950")),
            Some(crate::target::Architecture::Gfx950 | crate::target::Architecture::Gfx1250)
        )
    {
        return Err(AnalysisError::Eval(EvalError::InvalidConfig(
            "early_memory_conflicts supports only native gfx950/gfx1250 memory conflicts".into(),
        )));
    }
    if config.early_readiness
        && crate::target::Architecture::from_target(module.target.as_deref().unwrap_or("gfx950"))
            != Some(crate::target::Architecture::Gfx1250)
    {
        return Err(AnalysisError::Eval(EvalError::InvalidConfig(
            "early_readiness currently supports only gfx1250 scalar-load readiness".into(),
        )));
    }
    let kernel = module
        .kernels
        .iter()
        .find(|kernel| name.is_none_or(|name| kernel.name == name))
        .ok_or_else(|| AnalysisError::KernelNotFound {
            name: name.map(str::to_string),
        })?;
    let program = lower_kernel_in_module(module, kernel).map_err(AnalysisError::Lower)?;
    let target = module.target.clone().unwrap_or_else(|| "gfx950".into());
    let replay_enabled = target.split(':').any(|feature| feature == "xnack+");
    let mut assumptions = vec![
        Assumption::StructuredCta,
        Assumption::SingleWorkgroupAnalyzed,
        Assumption::MathematicalRealArithmetic,
        Assumption::EagerAsyncData,
    ];
    if crate::target::Architecture::from_target(&target)
        == Some(crate::target::Architecture::Gfx1250)
    {
        assumptions.push(Assumption::UnclusteredWorkgroup);
        if program.instructions.iter().any(|instruction| {
            matches!(instruction.operation(), crate::ir::Operation::Memory(memory)
                if memory.kind == crate::ir::MemoryKind::AsyncLoadToLds)
        }) {
            assumptions.push(Assumption::AmdDefaultScheduling);
        }
        let named = kernel.descriptor.value("named_barrier_count", 0);
        if named != 0 {
            assumptions.push(Assumption::AllocatedNamedBarriers((named * 4) as u8));
        }
    } else if !replay_enabled {
        assumptions.push(Assumption::XnackReplayDisabled);
    }
    if crate::target::Architecture::from_target(&target)
        != Some(crate::target::Architecture::Gfx1250)
        && !program.descriptor.flag("tg_split")
    {
        assumptions.push(Assumption::SingleCuWorkgroup);
    }
    let provenance = Provenance {
        kernel: kernel.name.clone(),
        block_dim: config.block_dim,
        unit: volta_graph::ThreadUnit::CtaThread,
        grid_dim: config.grid_dim,
        workgroup_id: config.workgroup_id,
        targets: vec![target],
        assumptions,
        instructions: program
            .instructions
            .iter()
            .map(|instruction| InstructionInfo {
                kind: if instruction.mnemonic() == "global_atomic_add_f32" {
                    "atomic f32 add no return"
                } else if crate::target::Architecture::from_target(
                    module.target.as_deref().unwrap_or("gfx950"),
                ) == Some(crate::target::Architecture::Gfx950)
                    && matches!(instruction.mnemonic().as_str(), "v_readlane_b32" | "v_writelane_b32")
                {
                    if instruction.mnemonic() == "v_readlane_b32" { "readlane" } else { "writelane" }
                } else if matches!(instruction.operation(), crate::ir::Operation::Memory(memory)
                    if matches!(memory.kind, crate::ir::MemoryKind::ScratchLoad | crate::ir::MemoryKind::ScratchStore))
                {
                    if matches!(instruction.operation(), crate::ir::Operation::Memory(memory) if memory.kind == crate::ir::MemoryKind::ScratchLoad) { "scratch load" } else { "scratch store" }
                } else if module
                    .target
                    .as_deref()
                    .and_then(crate::target::Architecture::from_target)
                    == Some(crate::target::Architecture::Gfx1250)
                    && instruction.mnemonic().starts_with("flat_")
                {
                    "flat memory"
                } else if instruction.mnemonic().starts_with("tensor_") {
                    "tensor memory"
                } else if instruction
                    .mnemonic()
                    .starts_with("global_load_async_to_lds_")
                {
                    "async memory"
                } else if instruction.mnemonic() == "v_nop" {
                    "vector ALU"
                } else if instruction.mnemonic() == "s_getreg_b32"
                    || (instruction.mnemonic() == "s_sendmsg"
                        && module
                            .target
                            .as_deref()
                            .and_then(crate::target::Architecture::from_target)
                            == Some(crate::target::Architecture::Gfx1250))
                {
                    "xcnt drain"
                } else if instruction.mnemonic() == "global_prefetch_b8" {
                    "prefetch"
                } else if instruction.mnemonic() == "s_wait_alu" {
                    "alu wait"
                } else if instruction.mnemonic() == "s_delay_alu" {
                    "s_delay_alu"
                } else if instruction.mnemonic() == "s_barrier_signal" {
                    if matches!(instruction.operands(), [crate::ir::Operand::Integer(-1)]) {
                        "barrier signal"
                    } else {
                        "named barrier signal"
                    }
                } else if instruction.mnemonic() == "s_barrier_wait" {
                    if matches!(instruction.operands(), [crate::ir::Operand::Integer(-1)]) {
                        "barrier wait"
                    } else {
                        "named barrier wait"
                    }
                } else if instruction.mnemonic() == "s_barrier_init" {
                    "named barrier init"
                } else if instruction.mnemonic() == "s_barrier_join" {
                    "named barrier join"
                } else if instruction.mnemonic() == "s_barrier_leave" {
                    "named barrier leave"
                } else if instruction.mnemonic() == "s_nop" {
                    "s_nop"
                } else if instruction.mnemonic() == "s_endpgm" {
                    "s_endpgm"
                } else if matches!(
                    instruction.mnemonic().as_str(),
                    "v_fma_mixlo_f16" | "v_fma_mixhi_f16"
                ) {
                    "shifted ALU"
                } else if matches!(
                    instruction
                        .mnemonic()
                        .trim_end_matches("_e32")
                        .trim_end_matches("_e64"),
                    "v_s_exp_f32"
                        | "v_exp_f32"
                        | "v_log_f32"
                        | "v_rcp_f32"
                        | "v_rsq_f32"
                        | "v_sqrt_f32"
                        | "v_rcp_f64"
                        | "v_sqrt_f64"
                ) {
                    "trans ALU"
                } else {
                    super::eval::interp_kind(instruction.opcode(), instruction.mnemonic())
                },
                span: Some((instruction.span().start, instruction.span().end)),
            })
            .collect(),
    };
    let interpreter =
        Interpreter::new(&program, config, provenance).map_err(AnalysisError::Eval)?;
    Ok(AnalysisResult::from_run(interpreter.execute(observer)))
}

pub fn write_op_counts(
    out: &mut dyn std::io::Write,
    label: &str,
    counts: &BTreeMap<&'static str, u64>,
) -> std::io::Result<()> {
    if counts.is_empty() {
        return Ok(());
    }
    let total: u64 = counts.values().sum();
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    writeln!(out, "{label} profile:")?;
    for (kind, count) in entries {
        writeln!(
            out,
            "  {kind:<16} {count:>10}  ({:>5.1}%)",
            100.0 * *count as f64 / total as f64
        )?;
    }
    Ok(())
}

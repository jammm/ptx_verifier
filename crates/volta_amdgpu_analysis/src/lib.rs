//! Native gfx942/gfx950/gfx1250 assembly analysis using the existing expression and event artifacts.
//!
//! [`frontend::parse`] validates compiler-module directives, metadata and globals,
//! then parses every descriptor/metadata entry body. Explicitly declared and
//! bounded non-entry functions form a separate helper catalog. The selected
//! gfx950 call profile executes complete reachable helpers through checked
//! rel32 code pointers and returns. Branches stay inside their source function;
//! scratch execution, unavailable runtime ABI values and general computed jumps
//! remain unsupported. Modules without entry declarations retain strict parsing
//! of all handwritten function bodies.
mod analysis;
mod config;
mod eval;
pub mod frontend;
pub mod ir;
pub mod lower;
pub mod m1;
mod source_encoding;
pub mod target;

pub use analysis::{
    AnalysisError, AnalysisParts, AnalysisResult, analyze, analyze_with_observer, write_op_counts,
};
pub use config::{
    AnalysisConfig, ArrayDef, ArrayKind, GuardRegion, InitialPattern, InitialWords, NativeProfile,
    OutputContract, ParamValue, SharedAllocation,
};
pub use eval::{
    EvalError, InstructionObserver, OBSERVER_STOP_REASON, OutputsUnavailable, RawOutputs, RawWord,
    RawWordRefusal, Stats, ThreadId,
};

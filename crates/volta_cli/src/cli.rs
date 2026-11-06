//! Command tree and global options.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::commands::{
    AnalyzeArgs, CheckProgramArgs, CompareArgs, ExtractLeanArgs, SolveArgs, VerifyArgs, VizArgs,
};

/// Log level for controlling output verbosity
#[cfg(feature = "logging")]
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub(crate) enum LogLevel {
    /// Only show errors
    Error,
    /// Show warnings and errors
    #[default]
    Warn,
    /// Show info, warnings, and errors
    Info,
    /// Show debug output and above
    Debug,
    /// Show all log output including trace
    Trace,
}

#[cfg(feature = "logging")]
impl From<LogLevel> for log::LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Trace => log::LevelFilter::Trace,
        }
    }
}

#[derive(Parser)]
#[command(name = "volta")]
#[command(about = "The Volta GPU kernel analysis engine.")]
#[command(version)]
pub(crate) struct Cli {
    /// Instruction-set backend: ptx or amdgpu. Legacy commands default to ptx;
    /// solve and extract infer the frontend from the extension when omitted.
    #[arg(long, global = true)]
    pub(crate) backend: Option<String>,

    /// Native AMDGPU architecture: gfx942, gfx950 (default), or gfx1250.
    #[arg(long, global = true)]
    pub(crate) target: Option<String>,

    #[command(subcommand)]
    pub(crate) command: Commands,

    /// Log level for output verbosity
    #[cfg(feature = "logging")]
    #[arg(long, value_enum, default_value = "warn", global = true)]
    pub(crate) log_level: LogLevel,

    /// Directory for per-run log files
    #[arg(long, global = true, default_value = "volta-logs")]
    pub(crate) log_dir: PathBuf,

    /// Don't write a per-run log file
    #[arg(long, global = true)]
    pub(crate) no_log_file: bool,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Parse PTX or native AMD assembly and check for syntax errors
    Parse {
        /// PTX or native AMD assembly file to parse
        file: PathBuf,
    },

    /// Symbolically execute a kernel: detect races/deadlocks and print the
    /// symbolic output tensors
    Analyze(AnalyzeArgs),

    /// Check two kernels for semantic equivalence (each is also checked for
    /// data races/deadlocks). Arrays/params/globals are shared by both
    /// kernels unless a `--block2`/`--grid2` override is given.
    Compare(CompareArgs),

    /// Check one kernel against a math spec (and for races/deadlocks). SPEC is a `.spec` path, or
    /// inline spec source if no such file exists; every array with a spec output equation must be a
    /// declared output array of the kernel.
    Verify(VerifyArgs),

    /// Run a kernel and optionally check its outputs against a spec. Without --backend,
    /// infer PTX, experimental gfx950 (outputs only), or NKI from the extension.
    /// Use --backend amdgpu --target gfx* for native AMD graph checks.
    Solve(SolveArgs),

    /// Extract proof obligations after checking the kernel's available graph evidence
    Extract {
        #[command(subcommand)]
        format: ExtractFormat,
    },

    /// Build the event-graph viewer: one self-contained HTML page over a store of recorded runs
    /// (`just viz` records every volta_pipeline test into one, then builds the page)
    Viz(VizArgs),

    /// Check one gfx1250 kernel under the B0 profile its request names and write the
    /// matching JSON report. Exit status equals the report class: 0 PASS, 1 REJECT,
    /// 2 UNKNOWN_UNSUPPORTED, 3 ERROR, 4 TIMEOUT.
    CheckProgram(CheckProgramArgs),
}

#[derive(Subcommand)]
pub(crate) enum ExtractFormat {
    /// Write Lean 4 proof obligations; fail on detected races, deadlocks or incomplete checks
    Lean(ExtractLeanArgs),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_requires_at_least_one_check_array() {
        // The list is required, never derived - parity with
        // `paired_elements`' explicit-list interface.
        let without = ["volta", "compare", "a.ptx", "b.ptx"];
        assert!(Cli::try_parse_from(without).is_err());
        let with = ["volta", "compare", "a.ptx", "b.ptx", "--check-array", "out"];
        assert!(Cli::try_parse_from(with).is_ok());
    }

    #[test]
    fn verify_requires_a_spec_argument() {
        let without = ["volta", "verify", "a.ptx"];
        assert!(Cli::try_parse_from(without).is_err());
        let with = ["volta", "verify", "a.ptx", "C[i] = A[i];", "--dim", "N=4"];
        assert!(Cli::try_parse_from(with).is_ok());
    }
}

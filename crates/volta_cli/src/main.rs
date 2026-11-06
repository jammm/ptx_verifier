//! Volta CLI - GPU kernel analysis tool: `parse <file>` (syntax check), `analyze <file>` (symbolically
//! execute one kernel), `compare <file1> <file2>` (equivalence), `verify <file>` (against a spec),
//! `solve <file>` (any ISA's kernel, optionally checked against a spec),
//! `extract lean <file>` (graph checks followed by Lean proof obligations),
//! `viz <store>` (the event-graph viewer page over recorded runs),
//! `check-program --request <json> --report <json>` (one gfx1250 B0 profile request).

mod cli;
mod commands;
mod config;
mod output;
mod sha256;
mod source;
mod spec;

use std::process::ExitCode;

use clap::Parser;
use volta_common::run_log;
use volta_pipeline::Backend;

use cli::{Cli, Commands, ExtractFormat};
use commands::{
    cmd_analyze, cmd_check_program, cmd_compare, cmd_extract_lean, cmd_parse, cmd_solve,
    cmd_verify, cmd_viz,
};

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Commands::CheckProgram(args) = &cli.command {
        // The report, not a printed message, records every outcome.
        let backend = Backend::select(
            cli.backend.as_deref().unwrap_or("ptx"),
            cli.target.as_deref(),
        );
        let log = run_log::RunLog::open(&cli.log_dir, "check-program", cli.no_log_file);
        let code = cmd_check_program(args, backend);
        if let Some(path) = log.path() {
            eprintln!("log: {}", path.display());
        }
        return code;
    }
    let backend = match Backend::select(
        cli.backend.as_deref().unwrap_or("ptx"),
        cli.target.as_deref(),
    ) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    let explicit_backend = cli.backend.as_ref().map(|_| backend);

    let command_name = match &cli.command {
        Commands::Parse { .. } => "parse",
        Commands::Analyze(_) => "analyze",
        Commands::Compare(_) => "compare",
        Commands::Verify(_) => "verify",
        Commands::Solve(_) => "solve",
        Commands::Extract { .. } => "extract lean",
        Commands::Viz(_) => "viz",
        Commands::CheckProgram(_) => unreachable!("handled before logging"),
    };
    let mut log = run_log::RunLog::open(&cli.log_dir, command_name, cli.no_log_file);

    #[cfg(feature = "logging")]
    env_logger::Builder::new()
        .filter_level(cli.log_level.into())
        .format_timestamp(None)
        .format_target(false)
        .target(env_logger::Target::Pipe(log.tee(std::io::stderr())))
        .init();

    let code = match cli.command {
        Commands::Parse { file } => cmd_parse(&file, backend),
        Commands::Analyze(args) => cmd_analyze(args, &mut log, backend),
        Commands::Compare(args) => cmd_compare(args, &mut log, backend),
        Commands::Verify(args) => cmd_verify(args, &mut log, backend),
        Commands::Solve(args) => cmd_solve(args, &mut log, explicit_backend),
        Commands::Extract {
            format: ExtractFormat::Lean(args),
        } => cmd_extract_lean(args, &mut log, explicit_backend),
        Commands::Viz(args) => cmd_viz(&args),
        Commands::CheckProgram(_) => unreachable!("handled before logging"),
    };

    if let Some(path) = log.path() {
        eprintln!("log: {}", path.display());
    }
    code
}

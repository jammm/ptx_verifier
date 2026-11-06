//! Native gfx942/gfx950/gfx1250 assembly frontend. Parsing preserves typed operands and source
//! locations; lowering resolves local control flow before symbolic execution.
//!
//! The accepted descriptor/metadata dialect is LLVM AMDHSA code-object 4–6,
//! metadata 1.0–1.2. Unknown semantic directives and entry opcodes are errors.
//! Descriptors and metadata identify entry kernels. Explicit `.type ...,@function`
//! bodies with matching `.size` are cataloged separately from entries. Uncalled
//! unsupported bodies retain a refusal, never an executable parsed prefix.
//! Lowering checks every selected entry and referenced helper instruction and
//! supports only the finite gfx950 rel32 scalar call/return profile. Arbitrary
//! computed jumps and branches outside their source function remain unsupported.
//! With no entry declarations, all handwritten bodies are parsed strictly.

pub mod ast;
mod constant_expr;
pub(crate) mod gfx1250;
mod instr;
mod metadata;
mod parse;

pub use ast::*;
pub use parse::{ParseError, function_source_lines, parse, parse_module};

#[cfg(test)]
mod tests;

/// Revalidate caller-built syntax before it becomes an executable instruction.
/// Parsing and lowering share the same operand/encoding checks; this does not
/// turn a recognized but unimplemented operation into an execution claim.
pub(crate) fn validate_for_lowering(
    instruction: &Instruction,
    architecture: crate::target::Architecture,
) -> Result<(), ParseError> {
    instr::validate_for_lowering(instruction, architecture).map_err(|message| ParseError {
        message,
        span: instruction.span,
    })
}

/// Canonicalize implicit sources and FP64 constants exactly once, recursively
/// for VOPD components. Already parsed instructions remain unchanged.
pub(crate) fn normalize_for_lowering(instruction: &mut Instruction) -> Result<(), ParseError> {
    instr::normalize_for_lowering(instruction)
}

/// Descriptor structure and register-allocation constraints shared with parsing.
pub(crate) fn validate_descriptor_for_lowering(kernel: &Kernel) -> Result<(), ParseError> {
    parse::validate_descriptor(kernel)
}

/// Assembler integer literal (decimal, hex, octal or binary, optionally signed).
pub fn integer_literal(text: &str) -> Option<i64> {
    instr::integer(text)
}

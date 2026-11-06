//! Typed native AMD assembly syntax, source locations and kernel metadata.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SourceSpan {
    /// One-based source line and column, followed by byte offsets.
    pub line: usize,
    pub column: usize,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterClass {
    Scalar,
    Vector,
    Accumulator,
    /// Architected system-register inputs on gfx1250; not ordinary SGPRs.
    Temporary,
    Special,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialRegister {
    Exec,
    ExecLo,
    ExecHi,
    Vcc,
    VccLo,
    VccHi,
    Scc,
    M0,
    Null,
    FlatScratch,
    FlatScratchLo,
    FlatScratchHi,
    /// Opaque 64-bit LDS aperture base, only read by the pinned s_mov_b64 ABI.
    SharedBase,
    /// Opaque 64-bit private aperture base; gfx950 s_mov_b64 source only.
    PrivateBase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Register {
    pub class: RegisterClass,
    pub index: u16,
    /// Number of adjacent 32-bit registers; masks and pointers commonly use 2.
    pub count: u16,
    pub special: Option<SpecialRegister>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Register(Register),
    Integer(i64),
    Float(f64),
    /// A label, `off`, or an explicitly declared assembly constant.
    Symbol(String),
    /// Direct LLVM AMDGPU PC-relative relocation. Only a validated
    /// contiguous address-construction sequence may consume this operand.
    Relocation(Relocation),
    Modified {
        value: Box<Operand>,
        negate: bool,
        absolute: bool,
    },
    WaitCounter {
        name: String,
        value: u32,
    },
    /// Only present after lowering.
    BranchTarget(usize),
    HardwareRegister {
        name: String,
        offset: u8,
        width: u8,
    },
    /// Two vector operations issued together; both read the pre-issue state.
    Dual(Box<Instruction>, Box<Instruction>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocationKind {
    Rel32Lo,
    Rel32Hi,
    Rel64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relocation {
    pub symbol: String,
    pub kind: RelocationKind,
    pub addend: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Global {
    pub symbol: String,
    pub size: u64,
    pub alignment: u64,
    /// Exact assembler object bytes, including explicit BSS zero bytes.
    pub initial_bytes: Vec<u8>,
    pub read_only: bool,
    pub span: SourceSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Modifier {
    pub name: String,
    pub value: Option<String>,
}

/// Instruction family. The exact validated mnemonic remains on `Instruction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    ScalarAlu,
    VectorAlu,
    ScalarMemory,
    VectorMemory,
    Lds,
    Branch,
    Wait,
    Barrier,
    Matrix,
    Control,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    pub mnemonic: String,
    pub opcode: Opcode,
    pub operands: Vec<Operand>,
    pub modifiers: Vec<Modifier>,
    pub span: SourceSpan,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelArg {
    pub name: Option<String>,
    pub offset: u64,
    pub size: u64,
    pub value_kind: String,
    pub address_space: Option<String>,
    pub value_type: Option<String>,
}

/// Descriptor fields retain their ISA spelling without the `.amdhsa_` prefix.
/// Preserving explicit zeroes lets ABI initialization distinguish missing data.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelDescriptor {
    pub fields: BTreeMap<String, u64>,
    /// Explicit (descriptor, metadata) private fixed extents before merging.
    /// Absent if either declaration is missing; never inferred from a default.
    pub private_segment_fixed_size_sources: Option<(u64, u64)>,
}

impl KernelDescriptor {
    pub fn get(&self, name: &str) -> Option<u64> {
        self.fields.get(name).copied()
    }

    pub fn value(&self, name: &str, default: u64) -> u64 {
        self.get(name).unwrap_or(default)
    }

    pub fn flag(&self, name: &str) -> bool {
        self.get(name).is_some_and(|value| value != 0)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Kernel {
    pub name: String,
    pub instructions: Vec<Instruction>,
    pub labels: BTreeMap<String, usize>,
    pub descriptor: KernelDescriptor,
    pub args: Vec<KernelArg>,
    /// Explicit `.size` that closed this body, if any.
    pub size_directive: Option<SizeDirective>,
    /// Text-section alignment or data directives after the closing `.size` and
    /// before the next function or section. They are retained, never executed.
    pub trailing_text_directives: Vec<TextDirective>,
}

/// A function's `.size` directive with its unevaluated extent expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SizeDirective {
    pub expression: String,
    pub span: SourceSpan,
}

/// A raw `.p2align`/`.fill`-style directive outside any executable body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextDirective {
    pub name: String,
    pub arguments: String,
    pub span: SourceSpan,
}

/// A bounded non-entry function. Uncalled compiler helpers may contain forms
/// outside this frontend; their first refusal is retained rather than treating
/// a parsed prefix as an executable body. The complete bytes remain in source.
#[derive(Debug, Clone, Default)]
pub struct Function {
    pub name: String,
    pub instructions: Vec<Instruction>,
    pub labels: BTreeMap<String, usize>,
    pub span: SourceSpan,
    pub refusal: Option<(String, SourceSpan)>,
}

#[derive(Debug, Clone, Default)]
pub struct Module {
    pub target: Option<String>,
    /// Exact `.amdgcn_target` and `amdhsa.target` spellings before normalization.
    pub target_directive: Option<String>,
    pub metadata_target: Option<String>,
    pub code_object_version: Option<u32>,
    pub metadata_version: Option<Vec<u32>>,
    pub kernels: Vec<Kernel>,
    /// Explicit `.type`/`.size` function bodies, separate from kernel exports.
    pub functions: Vec<Function>,
    pub constants: BTreeMap<String, i64>,
    pub globals: Vec<Global>,
    /// Original source is retained for diagnostics and event-graph exports.
    pub source: String,
}

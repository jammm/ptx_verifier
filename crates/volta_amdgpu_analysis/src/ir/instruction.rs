//! Target-validated executable instructions. Syntax is consumed during lowering;
//! only source identity and modifier spelling remain alongside semantic operations.
use super::alu::AluOperation;
use super::ops::{MatrixOpcode, MemoryOpcode};
use crate::frontend::{self, Modifier, Opcode, Register, Relocation, SourceSpan};
use crate::source_encoding::{
    EncodedImmediate, ResolvedScalarSources, ScalarLayout, ScalarSourceType,
};
use crate::target::Architecture;
use std::borrow::Cow;
use volta_graph::AmdCounter;

/// A value operand cannot contain a label, a wait syntax tree, or a raw VOPD instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Register(Register),
    Integer(i64),
    Float(f64),
    Off,
    Relocation(Relocation),
    Modified {
        value: Box<Operand>,
        negate: bool,
        absolute: bool,
    },
    HardwareRegister {
        name: String,
        offset: u8,
        width: u8,
    },
    EncodedImmediate(EncodedImmediate),
}
impl Operand {
    fn lower(value: &frontend::Operand) -> Result<Self, String> {
        Ok(match value {
            frontend::Operand::Register(r) => Self::Register(*r),
            frontend::Operand::Integer(v) => Self::Integer(*v),
            frontend::Operand::Float(v) => Self::Float(*v),
            frontend::Operand::Symbol(name) if name == "off" => Self::Off,
            frontend::Operand::Relocation(v) => Self::Relocation(v.clone()),
            frontend::Operand::Modified {
                value,
                negate,
                absolute,
            } => Self::Modified {
                value: Box::new(Self::lower(value)?),
                negate: *negate,
                absolute: *absolute,
            },
            frontend::Operand::HardwareRegister {
                name,
                offset,
                width,
            } => Self::HardwareRegister {
                name: name.clone(),
                offset: *offset,
                width: *width,
            },
            _ => return Err("syntax-only operand cannot enter a lowered value operation".into()),
        })
    }
}

/// Closed arities retain contiguous constituent operands for dependency walks.
/// Construction is private; instruction-specific register/encoding checks precede it.
#[derive(Debug, Clone, PartialEq)]
enum Arguments {
    None,
    One([Operand; 1]),
    Two([Operand; 2]),
    Three([Operand; 3]),
    Four([Operand; 4]),
    Five([Operand; 5]),
    Six([Operand; 6]),
}
impl Arguments {
    fn lower(values: &[frontend::Operand]) -> Result<Self, String> {
        let values = values
            .iter()
            .map(Operand::lower)
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_values(values)
    }
    fn lower_scalar(
        values: &[frontend::Operand],
        resolved: &ResolvedScalarSources,
    ) -> Result<Self, String> {
        let values = values
            .iter()
            .enumerate()
            .map(|(position, operand)| match operand {
                frontend::Operand::Integer(_) | frontend::Operand::Float(_) => resolved
                    .immediate_at(position)
                    .map(Operand::EncodedImmediate)
                    .ok_or_else(|| {
                        "migrated scalar numeric source lacks a checked encoding".into()
                    }),
                _ => Operand::lower(operand),
            })
            .collect::<Result<Vec<_>, String>>()?;
        Self::from_values(values)
    }
    fn from_values(values: Vec<Operand>) -> Result<Self, String> {
        Ok(match values.len() {
            0 => Self::None,
            1 => Self::One(values.try_into().unwrap()),
            2 => Self::Two(values.try_into().unwrap()),
            3 => Self::Three(values.try_into().unwrap()),
            4 => Self::Four(values.try_into().unwrap()),
            5 => Self::Five(values.try_into().unwrap()),
            6 => Self::Six(values.try_into().unwrap()),
            _ => {
                return Err("unsupported lowered operand arity".into());
            }
        })
    }
    fn slice(&self) -> &[Operand] {
        match self {
            Self::None => &[],
            Self::One(v) => v,
            Self::Two(v) => v,
            Self::Three(v) => v,
            Self::Four(v) => v,
            Self::Five(v) => v,
            Self::Six(v) => v,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    Eq,
    Lg,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Ordered,
    Unordered,
    NotLt,
    NotLe,
    NotGt,
    NotGe,
    NotEq,
    NotLg,
    True,
    False,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareType {
    U16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
    F16,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Comparison {
    pub relation: Relation,
    pub ty: CompareType,
    pub writes_exec: bool,
}
impl Comparison {
    pub(super) fn parse(name: &str) -> Option<Self> {
        let writes_exec = name.starts_with("v_cmpx_");
        let comparison = name
            .strip_prefix("v_cmp_")
            .or_else(|| name.strip_prefix("v_cmpx_"))?;
        let (relation, ty) = comparison.rsplit_once('_')?;
        Some(Self {
            relation: match relation {
                "eq" => Relation::Eq,
                "lg" => Relation::Lg,
                "ne" => Relation::Ne,
                "lt" => Relation::Lt,
                "le" => Relation::Le,
                "gt" => Relation::Gt,
                "ge" => Relation::Ge,
                "o" => Relation::Ordered,
                "u" => Relation::Unordered,
                "nlt" => Relation::NotLt,
                "nle" => Relation::NotLe,
                "ngt" => Relation::NotGt,
                "nge" => Relation::NotGe,
                "neq" => Relation::NotEq,
                "nlg" => Relation::NotLg,
                "tru" => Relation::True,
                "f" => Relation::False,
                _ => return None,
            },
            ty: match ty {
                "u16" => CompareType::U16,
                "u32" => CompareType::U32,
                "i32" => CompareType::I32,
                "f32" => CompareType::F32,
                "u64" => CompareType::U64,
                "i64" => CompareType::I64,
                "f64" => CompareType::F64,
                "f16" => CompareType::F16,
                _ => return None,
            },
            writes_exec,
        })
    }
    pub(super) fn name(self) -> String {
        let relation = match self.relation {
            Relation::Eq => "eq",
            Relation::Lg => "lg",
            Relation::Ne => "ne",
            Relation::Lt => "lt",
            Relation::Le => "le",
            Relation::Gt => "gt",
            Relation::Ge => "ge",
            Relation::Ordered => "o",
            Relation::Unordered => "u",
            Relation::NotLt => "nlt",
            Relation::NotLe => "nle",
            Relation::NotGt => "ngt",
            Relation::NotGe => "nge",
            Relation::NotEq => "neq",
            Relation::NotLg => "nlg",
            Relation::True => "tru",
            Relation::False => "f",
        };
        let ty = match self.ty {
            CompareType::U16 => "u16",
            CompareType::U32 => "u32",
            CompareType::I32 => "i32",
            CompareType::F32 => "f32",
            CompareType::U64 => "u64",
            CompareType::I64 => "i64",
            CompareType::F64 => "f64",
            CompareType::F16 => "f16",
        };
        format!(
            "v_cmp{}_{}_{}",
            if self.writes_exec { "x" } else { "" },
            relation,
            ty
        )
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct AluInstruction {
    pub operation: AluOperation,
    arguments: Arguments,
}
impl AluInstruction {
    pub fn operands(&self) -> &[Operand] {
        self.arguments.slice()
    }
    pub(crate) fn scalar_sources(&self) -> Result<Option<ScalarSources<'_>>, String> {
        let Some(layout) = ScalarLayout::for_operation(self.operation) else {
            return Ok(None);
        };
        let operands = self.operands();
        if operands.len() != layout.arity() {
            return Err("invalid migrated scalar operand count".into());
        }
        let source64 = |position: usize| match operands.get(position) {
            Some(Operand::Register(register)) if register.count == 2 => {
                Ok(ScalarSource64::Register(*register))
            }
            Some(Operand::EncodedImmediate(immediate))
                if immediate.source_type() == ScalarSourceType::Binary64 =>
            {
                Ok(ScalarSource64::Immediate(immediate))
            }
            _ => Err(
                "migrated Binary64 source lacks a register pair or checked encoding".to_string(),
            ),
        };
        Ok(Some(match layout {
            ScalarLayout::Move64 => ScalarSources::Move64(source64(1)?),
            ScalarLayout::Binary64 => ScalarSources::Binary64 {
                left: source64(1)?,
                right: source64(2)?,
            },
            ScalarLayout::Shift64 => {
                let count = match operands.get(2) {
                    Some(Operand::Register(register)) if register.count == 1 => {
                        ScalarSource32::Register(*register)
                    }
                    Some(Operand::EncodedImmediate(immediate))
                        if immediate.source_type() == ScalarSourceType::Bits32 =>
                    {
                        ScalarSource32::Immediate(immediate)
                    }
                    _ => {
                        return Err(
                            "migrated Bits32 source lacks a word or checked encoding".into()
                        );
                    }
                };
                ScalarSources::Shift64 {
                    data: source64(1)?,
                    count,
                }
            }
        }))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ScalarSource64<'a> {
    Register(Register),
    Immediate(&'a EncodedImmediate),
}
#[derive(Debug, Clone, Copy)]
pub(crate) enum ScalarSource32<'a> {
    Register(Register),
    Immediate(&'a EncodedImmediate),
}
#[derive(Debug, Clone, Copy)]
pub(crate) enum ScalarSources<'a> {
    Move64(ScalarSource64<'a>),
    Binary64 {
        left: ScalarSource64<'a>,
        right: ScalarSource64<'a>,
    },
    Shift64 {
        data: ScalarSource64<'a>,
        count: ScalarSource32<'a>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchCondition {
    Always,
    SccZero,
    SccNonzero,
    VccZero,
    VccNonzero,
    ExecZero,
    ExecNonzero,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchInstruction {
    pub condition: BranchCondition,
    pub target: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlOperation {
    EndProgram,
    ScalarNop,
    VectorNop,
    Clause,
    DelayAlu,
    SendMessage,
    Sleep,
    SetPriority,
    Trap,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ControlInstruction {
    pub operation: ControlOperation,
    arguments: Arguments,
}
impl ControlInstruction {
    pub fn operands(&self) -> &[Operand] {
        self.arguments.slice()
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierOperation {
    Workgroup,
    Signal,
    Wait,
    Init,
    Join,
    Leave,
}
#[derive(Debug, Clone, PartialEq)]
pub struct BarrierInstruction {
    pub operation: BarrierOperation,
    arguments: Arguments,
}
impl BarrierInstruction {
    pub fn object(&self) -> Option<&Operand> {
        self.arguments.slice().first()
    }
}

/// Wait omission is preserved rather than interpreted as an implicit zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaitInstruction {
    /// gfx12 S_WAIT_ALU: ALU dependency counters only, never memory completion.
    Alu(volta_graph::AmdAluWait),
    Cdna {
        vmcnt: Option<u16>,
        lgkmcnt: Option<u16>,
        expcnt: Option<u16>,
    },
    Counter {
        counter: AmdCounter,
        threshold: i64,
    },
    Combined {
        data: AmdCounter,
        encoded: i64,
    },
    /// gfx1250 S_WAIT_IDLE: every memory counter (and every ALU dependency
    /// counter) reaches zero.
    Idle,
    /// Recognized syntax whose execution semantics remain outside the profile.
    UnsupportedStoreCount,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    /// GLOBAL prefetch: address operands only, no data and no destination.
    Prefetch,
    ScalarLoad,
    ScalarBufferLoad,
    GlobalLoad,
    GlobalStore,
    FlatLoad,
    FlatStore,
    LdsLoad,
    LdsStore,
    LdsReadPair,
    LdsWritePair,
    Permute,
    AsyncLoadToLds,
    TensorLoadToLds,
    TensorStoreFromLds,
    AtomicNoReturn,
    AtomicReturning,
    BufferLoad,
    BufferLoadToLds,
    BufferStore,
    Swizzle,
    ScratchLoad,
    ScratchStore,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferShape {
    Registers {
        bytes_per_register: u8,
        registers_per_part: u8,
        parts: u8,
    },
    /// Direct memory-to-memory payload, with no register data tuple.
    PerLaneBytes(u8),
    /// Extent and element width are read from captured descriptor registers.
    TensorDescriptor,
}
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryInstruction {
    pub operation: MemoryOpcode,
    pub kind: MemoryKind,
    pub shape: TransferShape,
    arguments: Arguments,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryAddress<'a> {
    Scalar {
        base: &'a Operand,
        offset: &'a Operand,
    },
    ScalarBuffer {
        descriptor: &'a Operand,
        offset: &'a Operand,
    },
    Global {
        vector: &'a Operand,
        scalar: &'a Operand,
    },
    GlobalToLds {
        global: &'a Operand,
        lds: &'a Operand,
    },
    Flat {
        vector: &'a Operand,
        scalar: Option<&'a Operand>,
    },
    Lds {
        offset: &'a Operand,
    },
    Buffer {
        vector: &'a Operand,
        descriptor: &'a Operand,
        offset: &'a Operand,
    },
    BufferToLds {
        vector: &'a Operand,
        descriptor: &'a Operand,
    },
    CrossLane {
        index: &'a Operand,
    },
    Scratch {
        scalar: &'a Operand,
    },
    Descriptor {
        operands: &'a [Operand],
    },
}
impl MemoryInstruction {
    pub fn operands(&self) -> &[Operand] {
        self.arguments.slice()
    }
    pub fn is_store(&self) -> bool {
        matches!(
            self.kind,
            MemoryKind::GlobalStore
                | MemoryKind::FlatStore
                | MemoryKind::LdsStore
                | MemoryKind::LdsWritePair
                | MemoryKind::TensorStoreFromLds
                | MemoryKind::AtomicNoReturn
                | MemoryKind::AtomicReturning
                | MemoryKind::BufferStore
                | MemoryKind::ScratchStore
        )
    }
    pub fn address(&self) -> MemoryAddress<'_> {
        let o = self.operands();
        match self.kind {
            MemoryKind::Prefetch => MemoryAddress::Global {
                vector: &o[0],
                scalar: &o[1],
            },
            MemoryKind::ScalarLoad => MemoryAddress::Scalar {
                base: &o[1],
                offset: &o[2],
            },
            MemoryKind::ScalarBufferLoad => MemoryAddress::ScalarBuffer {
                descriptor: &o[1],
                offset: &o[2],
            },
            MemoryKind::GlobalLoad => MemoryAddress::Global {
                vector: &o[1],
                scalar: &o[2],
            },
            MemoryKind::AsyncLoadToLds => MemoryAddress::GlobalToLds {
                global: &o[1],
                lds: &o[0],
            },
            MemoryKind::GlobalStore => MemoryAddress::Global {
                vector: &o[0],
                scalar: &o[2],
            },
            MemoryKind::AtomicReturning => MemoryAddress::Global {
                vector: &o[1],
                scalar: &o[3],
            },
            MemoryKind::AtomicNoReturn => MemoryAddress::Global {
                vector: &o[0],
                scalar: &o[2],
            },
            MemoryKind::FlatLoad => MemoryAddress::Flat {
                vector: &o[1],
                scalar: o.get(2),
            },
            MemoryKind::FlatStore => MemoryAddress::Flat {
                vector: &o[0],
                scalar: o.get(2),
            },
            MemoryKind::LdsLoad | MemoryKind::LdsReadPair => MemoryAddress::Lds { offset: &o[1] },
            MemoryKind::LdsStore | MemoryKind::LdsWritePair => MemoryAddress::Lds { offset: &o[0] },
            MemoryKind::BufferLoad | MemoryKind::BufferStore => MemoryAddress::Buffer {
                vector: &o[1],
                descriptor: &o[2],
                offset: &o[3],
            },
            MemoryKind::BufferLoadToLds => MemoryAddress::BufferToLds {
                vector: &o[0],
                descriptor: &o[1],
            },
            MemoryKind::Permute => MemoryAddress::CrossLane { index: &o[1] },
            MemoryKind::ScratchLoad | MemoryKind::ScratchStore => {
                MemoryAddress::Scratch { scalar: &o[2] }
            }
            _ => MemoryAddress::Descriptor { operands: o },
        }
    }
    /// Register data destinations. Address/descriptor inputs are separate;
    /// tensor payloads reside in memory and have no register data destination.
    pub fn data_destinations(&self) -> &[Operand] {
        match self.kind {
            MemoryKind::ScalarLoad
            | MemoryKind::ScalarBufferLoad
            | MemoryKind::GlobalLoad
            | MemoryKind::FlatLoad
            | MemoryKind::LdsLoad
            | MemoryKind::LdsReadPair
            | MemoryKind::BufferLoad
            | MemoryKind::Permute
            | MemoryKind::Swizzle
            | MemoryKind::AtomicReturning => &self.operands()[..1],
            MemoryKind::ScratchLoad => &self.operands()[..1],
            _ => &[],
        }
    }
    /// All register data sources, including both DS_WRITE2 sources. Tensor
    /// descriptor inputs are available through address()/operands(), not here.
    pub fn data_sources(&self) -> &[Operand] {
        let o = self.operands();
        match self.kind {
            MemoryKind::GlobalStore
            | MemoryKind::FlatStore
            | MemoryKind::LdsStore
            | MemoryKind::AtomicNoReturn
            | MemoryKind::Swizzle => &o[1..2],
            MemoryKind::ScratchStore => &o[1..2],
            MemoryKind::LdsWritePair => &o[1..3],
            MemoryKind::AtomicReturning | MemoryKind::Permute => &o[2..3],
            MemoryKind::BufferStore => &o[..1],
            _ => &[],
        }
    }
    fn lower(name: &str, operands: &[frontend::Operand]) -> Result<Self, String> {
        let operation = MemoryOpcode::parse(name).ok_or("unknown lowered memory operation")?;
        if operation == MemoryOpcode::GlobalPrefetchB8 {
            return Ok(Self {
                operation,
                kind: MemoryKind::Prefetch,
                shape: TransferShape::PerLaneBytes(1),
                arguments: Arguments::lower(operands)?,
            });
        }
        let kind = if name.starts_with("s_buffer_") {
            MemoryKind::ScalarBufferLoad
        } else if name.starts_with("s_") {
            MemoryKind::ScalarLoad
        } else if name == "tensor_load_to_lds" {
            MemoryKind::TensorLoadToLds
        } else if name == "tensor_store_from_lds" {
            MemoryKind::TensorStoreFromLds
        } else if name.starts_with("global_load_async_to_lds_") {
            MemoryKind::AsyncLoadToLds
        } else if name.starts_with("global_atomic_") {
            if operands.len() == 4 {
                MemoryKind::AtomicReturning
            } else {
                MemoryKind::AtomicNoReturn
            }
        } else if name.starts_with("global_load_") {
            MemoryKind::GlobalLoad
        } else if name.starts_with("global_store_") {
            MemoryKind::GlobalStore
        } else if name.starts_with("flat_load_") {
            MemoryKind::FlatLoad
        } else if name.starts_with("flat_store_") {
            MemoryKind::FlatStore
        } else if name.starts_with("scratch_load_") {
            MemoryKind::ScratchLoad
        } else if name.starts_with("scratch_store_") {
            MemoryKind::ScratchStore
        } else if name.starts_with("buffer_load_") && operands.len() == 3 {
            MemoryKind::BufferLoadToLds
        } else if name.starts_with("buffer_load_") {
            MemoryKind::BufferLoad
        } else if name.starts_with("buffer_store_") {
            MemoryKind::BufferStore
        } else if matches!(name, "ds_bpermute_b32" | "ds_permute_b32") {
            MemoryKind::Permute
        } else if name == "ds_swizzle_b32" {
            MemoryKind::Swizzle
        } else if name.starts_with("ds_read2") || name == "ds_load_2addr_b32" {
            MemoryKind::LdsReadPair
        } else if name.starts_with("ds_write2") {
            MemoryKind::LdsWritePair
        } else if name.starts_with("ds_read_") || name.starts_with("ds_load_") {
            MemoryKind::LdsLoad
        } else {
            MemoryKind::LdsStore
        };
        let registers_per_part = if name.contains("dwordx16") || name.ends_with("b512") {
            16
        } else if name.contains("dwordx8") || name.ends_with("b256") {
            8
        } else if name.contains("dwordx4") || name.ends_with("b128") {
            4
        } else if name.contains("dwordx3") || name.ends_with("b96") {
            3
        } else if name.contains("dwordx2") || name.ends_with("b64") {
            2
        } else {
            1
        };
        let bytes_per_register = if name.contains("short")
            || name.ends_with("u16")
            || name.ends_with("i16")
            || name.contains("_b16")
        {
            2
        } else if name.contains("byte")
            || name.ends_with("u8")
            || name.ends_with("i8")
            || name.contains("_b8")
        {
            1
        } else {
            4
        };
        Ok(Self {
            operation,
            kind,
            shape: if matches!(
                kind,
                MemoryKind::TensorLoadToLds | MemoryKind::TensorStoreFromLds
            ) {
                TransferShape::TensorDescriptor
            } else if matches!(
                kind,
                MemoryKind::AsyncLoadToLds | MemoryKind::BufferLoadToLds
            ) {
                TransferShape::PerLaneBytes(4 * registers_per_part)
            } else {
                TransferShape::Registers {
                    bytes_per_register,
                    registers_per_part,
                    parts: if matches!(kind, MemoryKind::LdsReadPair | MemoryKind::LdsWritePair) {
                        2
                    } else {
                        1
                    },
                }
            },
            arguments: Arguments::lower(operands)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixFamily {
    Mfma,
    Wmma,
}
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixInstruction {
    pub operation: MatrixOpcode,
    pub family: MatrixFamily,
    arguments: Arguments,
}
impl MatrixInstruction {
    pub fn destination(&self) -> &Operand {
        &self.arguments.slice()[0]
    }
    pub fn a(&self) -> &Operand {
        &self.arguments.slice()[1]
    }
    pub fn b(&self) -> &Operand {
        &self.arguments.slice()[2]
    }
    pub fn accumulator(&self) -> &Operand {
        &self.arguments.slice()[3]
    }
    pub fn scales(&self) -> &[Operand] {
        &self.arguments.slice()[4..]
    }
    pub fn operands(&self) -> &[Operand] {
        self.arguments.slice()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    ScalarAlu(AluInstruction),
    VectorAlu(AluInstruction),
    Dual {
        x: Box<Instruction>,
        y: Box<Instruction>,
    },
    Memory(MemoryInstruction),
    Matrix(MatrixInstruction),
    Wait(WaitInstruction),
    Branch(BranchInstruction),
    Barrier(BarrierInstruction),
    Control(ControlInstruction),
    Call(CallInstruction),
    Return(ReturnInstruction),
}

/// An actual scalar code transfer; arguments retain exact dependency roles.
#[derive(Debug, Clone, PartialEq)]
pub struct CallInstruction {
    pub return_address: Register,
    pub target: Register,
    arguments: Arguments,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnInstruction {
    pub target: Register,
    arguments: Arguments,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Default,
    E32,
    E64,
    Sdwa,
    Vopd,
}
#[derive(Debug, Clone, PartialEq)]
pub struct SourceInstruction {
    pub mnemonic: String,
    pub encoding: Encoding,
    pub modifiers: Vec<Modifier>,
    pub span: SourceSpan,
}

/// Only lowering constructs executable instructions. Read-only semantic fields
/// cannot drift from the validated source identity after construction.
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    source: SourceInstruction,
    operation: Operation,
}
impl Instruction {
    pub fn source(&self) -> &SourceInstruction {
        &self.source
    }
    pub fn operation(&self) -> &Operation {
        &self.operation
    }
    pub fn alu_operation(&self) -> Option<AluOperation> {
        match &self.operation {
            Operation::ScalarAlu(v) | Operation::VectorAlu(v) => Some(v.operation),
            _ => None,
        }
    }
    pub fn for_each_operand<'a>(&'a self, visit: &mut impl FnMut(&'a Operand)) {
        match &self.operation {
            Operation::Dual { x, y } => {
                x.for_each_operand(visit);
                y.for_each_operand(visit);
            }
            _ => self.operands().iter().for_each(visit),
        }
    }
    pub fn mnemonic(&self) -> &String {
        &self.source.mnemonic
    }
    pub fn modifiers(&self) -> &Vec<Modifier> {
        &self.source.modifiers
    }
    pub fn span(&self) -> SourceSpan {
        self.source.span
    }
    pub fn operands(&self) -> &[Operand] {
        match &self.operation {
            Operation::ScalarAlu(v) | Operation::VectorAlu(v) => v.operands(),
            Operation::Memory(v) => v.operands(),
            Operation::Matrix(v) => v.operands(),
            Operation::Barrier(v) => v.arguments.slice(),
            Operation::Control(v) => v.operands(),
            Operation::Call(v) => v.arguments.slice(),
            Operation::Return(v) => v.arguments.slice(),
            Operation::Branch(_) | Operation::Wait(_) | Operation::Dual { .. } => &[],
        }
    }
    pub fn opcode(&self) -> Opcode {
        match &self.operation {
            Operation::ScalarAlu(_) => Opcode::ScalarAlu,
            Operation::VectorAlu(_) | Operation::Dual { .. } => Opcode::VectorAlu,
            Operation::Memory(v) => match v.kind {
                MemoryKind::ScalarLoad | MemoryKind::ScalarBufferLoad => Opcode::ScalarMemory,
                MemoryKind::LdsLoad
                | MemoryKind::LdsStore
                | MemoryKind::LdsReadPair
                | MemoryKind::LdsWritePair
                | MemoryKind::Permute
                | MemoryKind::Swizzle => Opcode::Lds,
                _ => Opcode::VectorMemory,
            },
            Operation::Matrix(_) => Opcode::Matrix,
            Operation::Wait(_) => Opcode::Wait,
            Operation::Branch(_) => Opcode::Branch,
            Operation::Barrier(_) => Opcode::Barrier,
            Operation::Control(_) | Operation::Call(_) | Operation::Return(_) => Opcode::Control,
        }
    }
    pub fn semantic_name(&self) -> Cow<'_, str> {
        match &self.operation {
            Operation::ScalarAlu(v) | Operation::VectorAlu(v) => v.operation.name(),
            Operation::Memory(v) => Cow::Borrowed(v.operation.name()),
            Operation::Matrix(v) => Cow::Borrowed(v.operation.name()),
            _ => Cow::Borrowed(&self.source.mnemonic),
        }
    }
    pub(crate) fn lower(
        source: &frontend::Instruction,
        architecture: Architecture,
    ) -> Result<Self, String> {
        let stem = source
            .mnemonic
            .trim_end_matches("_e32")
            .trim_end_matches("_e64")
            .trim_end_matches("_sdwa");
        let arguments = || Arguments::lower(&source.operands);
        let operation = match source.opcode {
            Opcode::ScalarAlu | Opcode::VectorAlu => {
                if let [frontend::Operand::Dual(x, y)] = source.operands.as_slice() {
                    Operation::Dual {
                        x: Box::new(Self::lower(x, architecture)?),
                        y: Box::new(Self::lower(y, architecture)?),
                    }
                } else {
                    let operation = AluOperation::parse(stem)
                        .ok_or_else(|| format!("no typed ALU operation for {stem}"))?;
                    let arguments = if source.opcode == Opcode::ScalarAlu {
                        if let Some(resolved) = crate::source_encoding::resolve(
                            &source.mnemonic,
                            &source.operands,
                            architecture,
                            false,
                        )? {
                            Arguments::lower_scalar(&source.operands, &resolved)?
                        } else {
                            arguments()?
                        }
                    } else {
                        arguments()?
                    };
                    let alu = AluInstruction {
                        operation,
                        arguments,
                    };
                    // Construction must establish the typed view; execution
                    // never recovers a migrated source through raw numerics.
                    alu.scalar_sources()?;
                    if source.opcode == Opcode::ScalarAlu {
                        Operation::ScalarAlu(alu)
                    } else {
                        Operation::VectorAlu(alu)
                    }
                }
            }
            Opcode::ScalarMemory | Opcode::VectorMemory | Opcode::Lds => {
                Operation::Memory(MemoryInstruction::lower(stem, &source.operands)?)
            }
            Opcode::Matrix => Operation::Matrix(MatrixInstruction {
                operation: MatrixOpcode::parse(stem).ok_or("unknown lowered matrix operation")?,
                family: if stem.starts_with("v_wmma_") || stem.starts_with("v_swmmac_") {
                    MatrixFamily::Wmma
                } else {
                    MatrixFamily::Mfma
                },
                arguments: arguments()?,
            }),
            Opcode::Branch => {
                let [frontend::Operand::BranchTarget(target)] = source.operands.as_slice() else {
                    return Err("unresolved lowered branch target".into());
                };
                let condition = match stem {
                    "s_branch" => BranchCondition::Always,
                    "s_cbranch_scc0" => BranchCondition::SccZero,
                    "s_cbranch_scc1" => BranchCondition::SccNonzero,
                    "s_cbranch_vccz" => BranchCondition::VccZero,
                    "s_cbranch_vccnz" => BranchCondition::VccNonzero,
                    "s_cbranch_execz" => BranchCondition::ExecZero,
                    "s_cbranch_execnz" => BranchCondition::ExecNonzero,
                    _ => return Err("unsupported lowered branch condition".into()),
                };
                Operation::Branch(BranchInstruction {
                    condition,
                    target: *target,
                })
            }
            Opcode::Barrier => Operation::Barrier(BarrierInstruction {
                operation: match stem {
                    "s_barrier" => BarrierOperation::Workgroup,
                    "s_barrier_signal" => BarrierOperation::Signal,
                    "s_barrier_wait" => BarrierOperation::Wait,
                    "s_barrier_init" => BarrierOperation::Init,
                    "s_barrier_join" => BarrierOperation::Join,
                    "s_barrier_leave" => BarrierOperation::Leave,
                    _ => return Err("unsupported lowered barrier operation".into()),
                },
                arguments: arguments()?,
            }),
            Opcode::Control if stem == "s_swappc_b64" => {
                let [
                    frontend::Operand::Register(return_address),
                    frontend::Operand::Register(target),
                ] = source.operands.as_slice()
                else {
                    return Err("device call requires two SGPR pairs".into());
                };
                Operation::Call(CallInstruction {
                    return_address: *return_address,
                    target: *target,
                    arguments: arguments()?,
                })
            }
            Opcode::Control if stem == "s_setpc_b64" => {
                let [frontend::Operand::Register(target)] = source.operands.as_slice() else {
                    return Err("device return requires an SGPR pair".into());
                };
                Operation::Return(ReturnInstruction {
                    target: *target,
                    arguments: arguments()?,
                })
            }
            Opcode::Control => Operation::Control(ControlInstruction {
                operation: match stem {
                    "s_endpgm" => ControlOperation::EndProgram,
                    "s_nop" => ControlOperation::ScalarNop,
                    "v_nop" => ControlOperation::VectorNop,
                    "s_clause" => ControlOperation::Clause,
                    "s_delay_alu" => ControlOperation::DelayAlu,
                    "s_sendmsg" => ControlOperation::SendMessage,
                    "s_sleep" => ControlOperation::Sleep,
                    "s_setprio" => ControlOperation::SetPriority,
                    "s_trap" => ControlOperation::Trap,
                    _ => return Err("unsupported lowered control operation".into()),
                },
                arguments: arguments()?,
            }),
            Opcode::Wait => Operation::Wait(lower_wait(stem, &source.operands)?),
        };
        let encoding = if matches!(operation, Operation::Dual { .. }) {
            Encoding::Vopd
        } else if source.mnemonic.ends_with("_e32") {
            Encoding::E32
        } else if source.mnemonic.ends_with("_e64") {
            Encoding::E64
        } else if source.mnemonic.ends_with("_sdwa") {
            Encoding::Sdwa
        } else {
            Encoding::Default
        };
        Ok(Self {
            source: SourceInstruction {
                mnemonic: source.mnemonic.clone(),
                encoding,
                modifiers: source.modifiers.clone(),
                span: source.span,
            },
            operation,
        })
    }
}
fn lower_wait(name: &str, operands: &[frontend::Operand]) -> Result<WaitInstruction, String> {
    let immediate = || match operands {
        [frontend::Operand::Integer(v)] => Ok(*v),
        _ => Err("wait requires one encoded immediate".to_owned()),
    };
    Ok(match name {
        "s_waitcnt" => {
            let (mut vmcnt, mut lgkmcnt, mut expcnt) = (None, None, None);
            for operand in operands {
                match operand {
                    frontend::Operand::WaitCounter { name, value } => {
                        let value =
                            u16::try_from(*value).map_err(|_| "wait threshold exceeds range")?;
                        match name.as_str() {
                            "vmcnt" => vmcnt = Some(value),
                            "lgkmcnt" => lgkmcnt = Some(value),
                            "expcnt" => expcnt = Some(value),
                            _ => return Err("unsupported wait counter".into()),
                        }
                    }
                    frontend::Operand::Integer(value) => {
                        let bits = *value as u16;
                        vmcnt = Some((bits & 15) | ((bits >> 10) & 48));
                        expcnt = Some((bits >> 4) & 7);
                        lgkmcnt = Some((bits >> 8) & 15);
                    }
                    _ => return Err("unsupported wait operand".into()),
                }
            }
            WaitInstruction::Cdna {
                vmcnt,
                lgkmcnt,
                expcnt,
            }
        }
        "s_waitcnt_vscnt" => WaitInstruction::UnsupportedStoreCount,
        "s_wait_idle" if operands.is_empty() => WaitInstruction::Idle,
        "s_wait_alu" => WaitInstruction::Alu(volta_graph::AmdAluWait {
            encoded: u16::try_from(immediate()?).map_err(|_| "s_wait_alu requires unsigned16")?,
        }),
        "s_wait_loadcnt_dscnt" | "s_wait_storecnt_dscnt" => WaitInstruction::Combined {
            data: if name == "s_wait_loadcnt_dscnt" {
                AmdCounter::Load
            } else {
                AmdCounter::Store
            },
            encoded: immediate()?,
        },
        _ => WaitInstruction::Counter {
            counter: match name {
                "s_wait_loadcnt" => AmdCounter::Load,
                "s_wait_storecnt" => AmdCounter::Store,
                "s_wait_dscnt" => AmdCounter::Ds,
                "s_wait_kmcnt" => AmdCounter::Km,
                "s_wait_xcnt" => AmdCounter::X,
                "s_wait_asynccnt" => AmdCounter::Async,
                "s_wait_tensorcnt" => AmdCounter::Tensor,
                _ => return Err("unsupported lowered wait instruction".into()),
            },
            threshold: immediate()?,
        },
    })
}

#[cfg(test)]
mod scalar_source_tests {
    use super::*;
    fn pair() -> Operand {
        Operand::Register(Register {
            class: frontend::RegisterClass::Scalar,
            index: 0,
            count: 2,
            special: None,
        })
    }
    #[test]
    fn raw_numeric_operands_cannot_enter_a_migrated_source_view() {
        for raw in [Operand::Integer(1), Operand::Float(1.0)] {
            let instruction = AluInstruction {
                operation: AluOperation::SMovB64,
                arguments: Arguments::Two([pair(), raw]),
            };
            assert!(instruction.scalar_sources().is_err());
        }
        let missing = AluInstruction {
            operation: AluOperation::SLshlB64,
            arguments: Arguments::None,
        };
        assert!(missing.scalar_sources().is_err());
    }
    #[test]
    fn checked_count_immediate_cannot_masquerade_as_binary64_data() {
        let operands = [
            frontend::Operand::Register(Register {
                class: frontend::RegisterClass::Scalar,
                index: 0,
                count: 2,
                special: None,
            }),
            frontend::Operand::Integer(1),
            frontend::Operand::Float(0.5),
        ];
        let checked =
            crate::source_encoding::resolve("s_lshl_b64", &operands, Architecture::Gfx950, false)
                .unwrap()
                .unwrap();
        let instruction = AluInstruction {
            operation: AluOperation::SMovB64,
            arguments: Arguments::Two([
                pair(),
                Operand::EncodedImmediate(checked.immediate_at(2).unwrap()),
            ]),
        };
        assert!(instruction.scalar_sources().is_err());
    }
    #[test]
    fn a_deferred_plan_cannot_copy_raw_numeric_operands_into_ir() {
        let operands = [
            frontend::Operand::Register(Register {
                class: frontend::RegisterClass::Scalar,
                index: 0,
                count: 2,
                special: None,
            }),
            frontend::Operand::Symbol("CONSTANT".into()),
        ];
        let deferred =
            crate::source_encoding::resolve("s_mov_b64", &operands, Architecture::Gfx950, true)
                .unwrap()
                .unwrap();
        let numeric = [operands[0].clone(), frontend::Operand::Integer(1)];
        assert!(Arguments::lower_scalar(&numeric, &deferred).is_err());
    }

    #[test]
    fn private_construction_does_not_normalize_unselected_scalar_forms() {
        for target in [
            Architecture::Gfx942,
            Architecture::Gfx950,
            Architecture::Gfx1250,
        ] {
            for name in [
                "s_mov_b64",
                "s_or_b64",
                "s_xor_b64",
                "s_lshl_b64",
                "s_lshr_b64",
            ] {
                for suffix in ["_e32", "_e64", "_sdwa"] {
                    let mut operands = vec![
                        frontend::Operand::Register(Register {
                            class: frontend::RegisterClass::Scalar,
                            index: 0,
                            count: 2,
                            special: None,
                        }),
                        frontend::Operand::Float(1.0),
                    ];
                    if name != "s_mov_b64" {
                        operands.push(frontend::Operand::Integer(1));
                    }
                    let source = frontend::Instruction {
                        mnemonic: format!("{name}{suffix}"),
                        opcode: Opcode::ScalarAlu,
                        operands,
                        modifiers: vec![],
                        span: SourceSpan::default(),
                    };
                    let error = Instruction::lower(&source, target).unwrap_err();
                    assert!(
                        error.contains("scalar source encoding")
                            && error.contains(&source.mnemonic)
                            && error.contains("bare SOP1/SOP2"),
                        "{error}"
                    );
                }
            }
        }
    }
}

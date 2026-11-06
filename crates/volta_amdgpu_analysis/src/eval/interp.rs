use super::{
    inputs::InputBacking,
    memory::Memory,
    value::{Value, WideKind},
};
use crate::{
    config::{AnalysisConfig, ArrayKind, ParamValue},
    ir::*,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};
use volta_expr::{ExprArena, ExprId, ExprNode};
use volta_graph::*;

#[path = "interp_async_lds.rs"]
mod async_lds;
#[path = "interp_buffer_lds.rs"]
mod buffer_lds;
#[path = "interp_litmus.rs"]
mod litmus;
#[path = "interp_m1.rs"]
mod m1;
#[path = "named_barrier.rs"]
mod named_barrier;
#[path = "interp_private.rs"]
mod private;
pub use m1::{RawOutputs, RawWord, RawWordRefusal};

mod control;
mod dependencies;
mod matrix_ops;
mod memory_ops;
mod scalar_alu;
mod tensor_ops;
mod values;
mod vector_alu;

const KERNARG_BASE: u64 = 0xffff_0000_0000;
const DISPATCH_BASE: u64 = 0xffff_1000_0000;
#[cfg(test)]
#[path = "interp_address_tests.rs"]
mod address_tests;
#[cfg(test)]
#[path = "interp_input_tests.rs"]
mod input_tests;
#[cfg(test)]
#[path = "interp_legacy_flat_tests.rs"]
mod legacy_flat_tests;
#[cfg(test)]
#[path = "interp_scalar_address_tests.rs"]
mod scalar_address_tests;
// Synthetic module storage is disjoint from host-bound arrays and analyzer ABI
// packets. Match the existing PTX module-window convention without a dependency.
pub const MODULE_GLOBAL_BASE: u64 = 0x7000_0000_0000_0000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadId(pub u32);
#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    pub instructions: u64,
    pub block_syncs: u64,
    pub warp_syncs: u64,
}
/// The graph remains an incomplete prefix when an instruction observer stops it.
pub const OBSERVER_STOP_REASON: &str = "execution stopped by instruction observer";

/// Called only after a successfully executed, complete instruction batch.
/// Breaking stops execution without producing outputs or a completed trace.
/// The callback itself supplies no native safety or acceptance claim.
pub type InstructionObserver<'a> = dyn FnMut(&EventGraph) -> std::ops::ControlFlow<()> + 'a;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    InvalidConfig(String),
    Unsupported {
        pc: usize,
        what: String,
    },
    Invalid {
        wave: u32,
        lane: Option<u8>,
        pc: usize,
        what: String,
    },
    BudgetExceeded {
        limit: u64,
    },
    ObserverStopped,
    TensorBounds {
        wave: u32,
        pc: usize,
        what: &'static str,
    },
    /// Attempted global footprint outside configured objects or in a guard.
    GlobalBounds {
        wave: u32,
        pc: usize,
        what: &'static str,
    },
}
impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(what) => write!(f, "invalid AMD configuration: {what}"),
            Self::Unsupported { pc, what } => {
                write!(f, "unsupported AMD instruction at pc {pc}: {what}")
            }
            Self::Invalid {
                wave,
                lane,
                pc,
                what,
            } => write!(f, "wave {wave} lane {lane:?}, pc {pc}: {what}"),
            Self::BudgetExceeded { limit } => write!(
                f,
                "analysis instruction budget {limit} exhausted (inconclusive)"
            ),
            Self::ObserverStopped => f.write_str(OBSERVER_STOP_REASON),
            Self::TensorBounds { what, .. } | Self::GlobalBounds { what, .. } => f.write_str(what),
        }
    }
}
impl Error for EvalError {}
#[derive(Debug)]
pub enum OutputsUnavailable {
    Aborted(EvalError),
    Stuck { blocked: Vec<(ThreadId, usize)> },
    Extraction(EvalError),
}
impl fmt::Display for OutputsUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aborted(e) => write!(f, "execution stopped: {e}"),
            Self::Stuck { blocked } => {
                write!(f, "execution is stuck with {} blocked lanes", blocked.len())
            }
            Self::Extraction(e) => write!(f, "output extraction failed: {e}"),
        }
    }
}
type ExtractedOutputs = Vec<(String, Vec<(u64, ExprId)>)>;

pub(crate) struct RunArtifacts {
    pub arena: ExprArena,
    pub outputs: Result<ExtractedOutputs, OutputsUnavailable>,
    /// Present only under the raw-u32 contract after a completed execution.
    pub raw_outputs: Option<RawOutputs>,
    pub graph: EventGraph,
    pub stats: Stats,
    pub op_counts: BTreeMap<&'static str, u64>,
}

#[derive(Debug, Clone)]
struct CallFrame {
    call_pc: usize,
    target: usize,
    continuation: usize,
}

#[derive(Debug, Clone)]
struct Wave {
    id: u32,
    width: u32,
    pc: usize,
    valid: u64,
    exec: u64,
    vcc: u64,
    vcc_defined: u8,
    vcc_predicates: [Option<Value>; 2],
    scc: bool,
    scc_defined: bool,
    // A uniform symbolic condition can select data, never concrete control,
    // carry, addresses, or an architectural EXEC mask.
    scc_condition: Option<ExprId>,
    m0: u32,
    m0_defined: bool,
    named_join: Option<named_barrier::Joined>,
    named_wait: Option<named_barrier::Joined>,
    // Program points of the instruction being executed and the one this wave
    // executed before it (litmus/2 STATE_PRIV wait-idle rule).
    current_pc: Option<usize>,
    previous_pc: Option<usize>,
    scalar: Vec<Option<Value>>,
    temporary: [Option<Value>; 16],
    vector: BTreeMap<(u16, u8), Value>,
    accumulator: BTreeMap<(u16, u8), Value>,
    at_barrier: Option<u32>,
    generation: u32,
    signal_generation: Option<u32>,
    exited: bool,
    calls: Vec<CallFrame>,
}
impl Wave {
    fn new(id: u32, lanes: u32, width: u32) -> Self {
        let valid = if lanes == 64 {
            u64::MAX
        } else {
            (1u64 << lanes) - 1
        };
        Self {
            id,
            width,
            pc: 0,
            valid,
            exec: valid,
            vcc: 0,
            vcc_defined: 0,
            vcc_predicates: [None; 2],
            scc: false,
            scc_defined: false,
            scc_condition: None,
            m0: 0,
            m0_defined: false,
            named_join: None,
            named_wait: None,
            current_pc: None,
            previous_pc: None,
            scalar: vec![None; 128],
            temporary: [None; 16],
            vector: BTreeMap::new(),
            accumulator: BTreeMap::new(),
            at_barrier: None,
            generation: 0,
            signal_generation: None,
            exited: false,
            calls: Vec::new(),
        }
    }
    fn fail(&self, lane: Option<u8>, what: impl Into<String>) -> EvalError {
        EvalError::Invalid {
            wave: self.id,
            lane,
            pc: self.pc,
            what: what.into(),
        }
    }
    fn active(&self) -> impl Iterator<Item = u8> {
        let mask = self.exec;
        (0..self.width as u8).filter(move |lane| mask & (1u64 << lane) != 0)
    }
    fn read_vcc(&self) -> Result<u64, EvalError> {
        if self.vcc_defined & ((1 << (self.width / 32)) - 1) != ((1 << (self.width / 32)) - 1) {
            return Err(self.fail(None, "uninitialized VCC flag (both halves must be defined)"));
        }
        if self.vcc_predicates.iter().any(Option::is_some) {
            return Err(self.fail(
                None,
                "symbolic predicate mask used as concrete VCC/control bits",
            ));
        }
        Ok(self.vcc)
    }
    fn read_scc(&self) -> Result<bool, EvalError> {
        if self.scc_condition.is_some() {
            return Err(self.fail(
                None,
                "symbolic SCC condition used as concrete control or carry",
            ));
        }
        if !self.scc_defined {
            return Err(self.fail(None, "uninitialized SCC flag"));
        }
        Ok(self.scc)
    }
    fn set_scc(&mut self, value: bool) {
        self.scc = value;
        self.scc_defined = true;
        self.scc_condition = None;
    }
    fn set_partial_scc(&mut self, bits: u32, defined: u32) {
        if bits & defined != 0 {
            self.set_scc(true);
        } else if defined == u32::MAX {
            self.set_scc(false);
        } else {
            self.clear_scc();
        }
    }
    fn clear_scc(&mut self) {
        self.scc_defined = false;
        self.scc_condition = None;
    }
    fn reg(&self, reg: Register, lane: u8, component: u16) -> Result<Value, EvalError> {
        let index = reg.index + component;
        let value = match reg.class {
            RegisterClass::Scalar => self.scalar.get(index as usize).copied().flatten(),
            RegisterClass::Temporary => self.temporary.get(index as usize).copied().flatten(),
            RegisterClass::Vector => self.vector.get(&(index, lane)).copied(),
            RegisterClass::Accumulator => self.accumulator.get(&(index, lane)).copied(),
            RegisterClass::Special => {
                let val = match reg.special {
                    Some(SpecialRegister::SharedBase) => {
                        if reg.count != 2 {
                            return Err(self.fail(
                                Some(lane),
                                "shared aperture source requires its approved64-bit profile",
                            ));
                        }
                        return Ok(if component == 0 {
                            Value::Bits(0)
                        } else {
                            Value::SharedApertureHi
                        });
                    }
                    Some(SpecialRegister::PrivateBase) => {
                        if reg.count != 2 {
                            return Err(self.fail(
                                Some(lane),
                                "private aperture source requires a 64-bit register pair",
                            ));
                        }
                        return Ok(if component == 0 {
                            Value::Bits(0)
                        } else {
                            Value::PrivateApertureHi
                        });
                    }
                    Some(SpecialRegister::Exec) => self.exec,
                    Some(SpecialRegister::ExecLo) => self.exec as u32 as u64,
                    Some(SpecialRegister::ExecHi) => self.exec >> 32,
                    Some(SpecialRegister::Vcc) => {
                        if self.vcc_defined & (1 << component) == 0 {
                            return Err(self.fail(Some(lane), "uninitialized VCC flag component"));
                        }
                        if let Some(value) = self.vcc_predicates[component as usize] {
                            return Ok(value);
                        }
                        self.vcc
                    }
                    Some(SpecialRegister::VccLo) => {
                        if self.vcc_defined & 1 == 0 {
                            return Err(self.fail(Some(lane), "uninitialized VCC_LO flag"));
                        }
                        if let Some(value) = self.vcc_predicates[0] {
                            return Ok(value);
                        }
                        self.vcc as u32 as u64
                    }
                    Some(SpecialRegister::VccHi) => {
                        if self.vcc_defined & 2 == 0 {
                            return Err(self.fail(Some(lane), "uninitialized VCC_HI flag"));
                        }
                        if let Some(value) = self.vcc_predicates[1] {
                            return Ok(value);
                        }
                        self.vcc >> 32
                    }
                    Some(SpecialRegister::Scc) => self.read_scc()? as u64,
                    Some(SpecialRegister::M0) => {
                        if self.width == 32 && self.m0_defined {
                            return Ok(Value::Bits(self.m0));
                        }
                        return Err(EvalError::Unsupported {
                            pc: self.pc,
                            what: "initial M0 value is not part of the modeled kernel ABI".into(),
                        });
                    }
                    Some(SpecialRegister::Null) => 0,
                    _ => return Err(self.fail(Some(lane), "unsupported special register")),
                };
                Some(Value::Bits((val >> (32 * component)) as u32))
            }
        };
        value.ok_or_else(|| {
            self.fail(
                Some(lane),
                format!("uninitialized {:?} register {index}", reg.class),
            )
        })
    }
    fn put(
        &mut self,
        reg: Register,
        lane: u8,
        component: u16,
        value: Value,
    ) -> Result<(), EvalError> {
        let index = reg.index + component;
        match reg.class {
            RegisterClass::Temporary => {
                return Err(self.fail(Some(lane), "architected system inputs are read-only"));
            }
            RegisterClass::Scalar => {
                let error = self.fail(Some(lane), "scalar register index out of range");
                *self.scalar.get_mut(index as usize).ok_or(error)? = Some(value);
            }
            RegisterClass::Vector => {
                self.vector.insert((index, lane), value);
            }
            RegisterClass::Accumulator => {
                self.accumulator.insert((index, lane), value);
            }
            RegisterClass::Special => {
                let vcc_word = match reg.special {
                    Some(SpecialRegister::Vcc) => Some(component as usize),
                    Some(SpecialRegister::VccLo) => Some(0),
                    Some(SpecialRegister::VccHi) => Some(1),
                    _ => None,
                };
                if let Some(word) = vcc_word {
                    if matches!(value, Value::PredicateWord { .. }) {
                        self.vcc_predicates[word] = Some(value);
                        self.vcc_defined |= 1 << word;
                        return Ok(());
                    }
                    // Read/validate before clearing an existing predicate word.
                    value.bits().map_err(|e| self.fail(Some(lane), e))?;
                    self.vcc_predicates[word] = None;
                }
                let bits = value.bits().map_err(|e| self.fail(Some(lane), e))? as u64;
                match reg.special {
                    Some(SpecialRegister::Exec) => {
                        self.exec = (self.exec & !(0xffff_ffffu64 << (32 * component)))
                            | (bits << (32 * component));
                    }
                    Some(SpecialRegister::ExecLo) => {
                        self.exec = (self.exec & 0xffff_ffff_0000_0000) | bits
                    }
                    Some(SpecialRegister::ExecHi) => {
                        self.exec = (self.exec & 0xffff_ffff) | (bits << 32)
                    }
                    Some(SpecialRegister::Vcc) => {
                        self.vcc = (self.vcc & !(0xffff_ffffu64 << (32 * component)))
                            | (bits << (32 * component));
                        self.vcc_defined |= 1 << component;
                    }
                    Some(SpecialRegister::VccLo) => {
                        self.vcc = (self.vcc & 0xffff_ffff_0000_0000) | bits;
                        self.vcc_defined |= 1;
                    }
                    Some(SpecialRegister::VccHi) => {
                        self.vcc = (self.vcc & 0xffff_ffff) | (bits << 32);
                        self.vcc_defined |= 2;
                    }
                    Some(SpecialRegister::Scc) => self.set_scc(bits != 0),
                    Some(SpecialRegister::M0) => {
                        if self.width == 32 {
                            self.m0 = bits as u32;
                            self.m0_defined = true;
                            return Ok(());
                        }
                        return Err(EvalError::Unsupported {
                            pc: self.pc,
                            what: "explicit M0 writes need software dependency-spacing checks"
                                .into(),
                        });
                    }
                    Some(SpecialRegister::Null) => {}
                    _ => return Err(self.fail(Some(lane), "unsupported special destination")),
                }
            }
        }
        Ok(())
    }
    fn value(&self, operand: &Operand, lane: u8) -> Result<Value, EvalError> {
        match operand {
            Operand::Register(reg) => self.reg(*reg, lane, 0),
            Operand::Integer(v) => Ok(Value::Bits(*v as u32)),
            Operand::Float(v) => Ok(Value::Bits((*v as f32).to_bits())),
            Operand::Off => Ok(Value::Bits(0)),
            _ => Err(self.fail(Some(lane), format!("non-value operand {operand:?}"))),
        }
    }
    /// Pure private-call spill transport may copy an unspecified numbered
    /// register word. No bit of this value becomes defined by the copy.
    /// Shape/range validation precedes execution; special registers never use
    /// this path, and ordinary semantic reads keep `reg`'s definedness checks.
    fn spill_word(&self, register: Register, lane: u8) -> Result<Value, EvalError> {
        if register.count != 1 {
            return Err(self.fail(Some(lane), "private spill requires one register word"));
        }
        let value = match register.class {
            RegisterClass::Scalar => self.scalar.get(register.index as usize).copied().flatten(),
            RegisterClass::Vector => self.vector.get(&(register.index, lane)).copied(),
            _ => return Err(self.fail(Some(lane), "private spill requires a numbered SGPR/VGPR")),
        };
        Ok(value.unwrap_or_else(|| Value::partial_bits(0, 0)))
    }
    fn bits(&self, operand: &Operand, lane: u8) -> Result<u32, EvalError> {
        self.value(operand, lane)?
            .bits()
            .map_err(|e| self.fail(Some(lane), e))
    }
    fn wide(&self, operand: &Operand, lane: u8) -> Result<u64, EvalError> {
        match operand {
            Operand::Register(reg) if reg.count == 2 => {
                let lo = self
                    .reg(*reg, lane, 0)?
                    .bits()
                    .map_err(|e| self.fail(Some(lane), e))?;
                let hi = self
                    .reg(*reg, lane, 1)?
                    .bits()
                    .map_err(|e| self.fail(Some(lane), e))?;
                Ok(lo as u64 | ((hi as u64) << 32))
            }
            Operand::Integer(v) => Ok(*v as u64),
            Operand::Off => Ok(0),
            Operand::Float(value) => {
                super::float_bits::inline_double(*value).ok_or_else(|| EvalError::Unsupported {
                    pc: self.pc,
                    what: "64-bit source requires an exact floating inline constant".into(),
                })
            }
            _ => Ok(self.bits(operand, lane)? as u64),
        }
    }
    fn scalar_source32(&self, source: ScalarSource32<'_>) -> Result<u32, EvalError> {
        match source {
            ScalarSource32::Register(register) => self
                .reg(register, 0, 0)?
                .bits()
                .map_err(|error| self.fail(None, error)),
            ScalarSource32::Immediate(immediate) => Ok(immediate.bits() as u32),
        }
    }
    fn scalar_source64(&self, source: ScalarSource64<'_>) -> Result<u64, EvalError> {
        match source {
            ScalarSource64::Register(register) => {
                let low = self
                    .reg(register, 0, 0)?
                    .bits()
                    .map_err(|error| self.fail(None, error))?;
                let high = self
                    .reg(register, 0, 1)?
                    .bits()
                    .map_err(|error| self.fail(None, error))?;
                Ok(u64::from(low) | (u64::from(high) << 32))
            }
            ScalarSource64::Immediate(immediate) => Ok(immediate.bits()),
        }
    }
    fn move_scalar64(
        &mut self,
        destination: &Operand,
        source: ScalarSource64<'_>,
    ) -> Result<(), EvalError> {
        match source {
            ScalarSource64::Register(register) => {
                let values = [self.reg(register, 0, 0)?, self.reg(register, 0, 1)?];
                let destination = self.destination(destination)?;
                self.put(destination, 0, 0, values[0])?;
                self.put(destination, 0, 1, values[1])
            }
            ScalarSource64::Immediate(immediate) => self.put_wide(destination, 0, immediate.bits()),
        }
    }
    /// Defined bits for a gfx1250 scalar bitwise word. An unset plain SGPR
    /// remains unknown; opaque and mathematical values still require bits().
    /// Instruction register roles independently retain every source word.
    fn scalar_bitword(&self, operand: &Operand, word: u16) -> Result<(u32, u32), EvalError> {
        let value = match operand {
            Operand::Register(register) => {
                if register.class == RegisterClass::Scalar
                    && self
                        .scalar
                        .get((register.index + word) as usize)
                        .copied()
                        .flatten()
                        .is_none()
                {
                    return Ok((0, 0));
                }
                self.reg(*register, 0, word)?
            }
            Operand::Integer(value) => Value::Bits((*value as u64 >> (32 * word)) as u32),
            _ => self.value(operand, 0)?,
        };
        match value {
            Value::PartialBits { bits, defined } => Ok((bits, defined)),
            value => Ok((
                value.bits().map_err(|error| self.fail(None, error))?,
                u32::MAX,
            )),
        }
    }
    fn destination(&self, operand: &Operand) -> Result<Register, EvalError> {
        match operand {
            Operand::Register(reg) => Ok(*reg),
            _ => Err(self.fail(None, "destination is not a register")),
        }
    }
    fn put_value(&mut self, operand: &Operand, lane: u8, value: Value) -> Result<(), EvalError> {
        let reg = self.destination(operand)?;
        self.put(reg, lane, 0, value)
    }
    fn put_wide(&mut self, operand: &Operand, lane: u8, value: u64) -> Result<(), EvalError> {
        let reg = self.destination(operand)?;
        self.put(reg, lane, 0, Value::Bits(value as u32))?;
        if reg.count == 2 {
            self.put(reg, lane, 1, Value::Bits((value >> 32) as u32))?;
        }
        Ok(())
    }
    fn move_wide(&mut self, destination: &Operand, source: &Operand) -> Result<(), EvalError> {
        if let Operand::Register(source) = source
            && source.count == 2
        {
            let values = [self.reg(*source, 0, 0)?, self.reg(*source, 0, 1)?];
            let destination = self.destination(destination)?;
            self.put(destination, 0, 0, values[0])?;
            return self.put(destination, 0, 1, values[1]);
        }
        let bits = self.wide(source, 0)?;
        self.put_wide(destination, 0, bits)
    }
}

#[derive(Clone)]
struct FloatBitPattern {
    decoded: ExprId,
    input_words: Vec<ExprId>,
}

pub(crate) struct Interpreter<'a> {
    program: &'a LoweredKernel,
    config: AnalysisConfig,
    arena: ExprArena,
    global: Memory,
    inputs: InputBacking,
    lds: Memory,
    kernargs: Memory,
    dispatch: Memory,
    waves: Vec<Wave>,
    builder: GraphBuilder,
    stats: Stats,
    op_counts: BTreeMap<&'static str, u64>,
    lds_size: u64,
    architecture: AmdArchitecture,
    byte_formats: BTreeMap<volta_expr::StringId, ByteFormat>,
    word_formats: BTreeMap<volta_expr::StringId, WordFormat>,
    predicate_masks: Vec<[ExprId; 64]>,
    // Only exact F32 sign/magnitude transport can defer its interpretation.
    // Actual integer consumers and observable raw outputs commit Integer32.
    float_bit_patterns: BTreeMap<ExprId, FloatBitPattern>,
    xnack_replay_enabled: bool,
    module_globals: BTreeMap<String, GlobalAllocation>,
    named: named_barrier::State,
    private: private::State,
    /// gfx1250 B0 m2 profile: MODE FP_ROUND (index 0) and FP_DENORM (index 1)
    /// per wave; `None` is an undefined field.
    mode_fields: BTreeMap<u32, [Option<u8>; 2]>,
    /// gfx1250 B0 multiwave profile: fresh symbols of opaque per-wave
    /// evidence reads (shader cycles, HW_ID1); stored words report OPAQUE.
    opaque_symbols: BTreeSet<ExprId>,
}
#[derive(Debug, Clone, Copy)]
struct GlobalAllocation {
    base: u64,
    size: u64,
    read_only: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteFormat {
    OcpE4m3,
    SignedI8,
    UnsignedU8,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WordFormat {
    Float32,
    Integer32,
    Float64,
    Float16,
    Bfloat16,
}
impl<'a> Interpreter<'a> {
    pub fn new(
        program: &'a LoweredKernel,
        config: AnalysisConfig,
        provenance: Provenance,
    ) -> Result<Self, EvalError> {
        config.validate().map_err(EvalError::InvalidConfig)?;
        let xnack_replay_enabled = provenance
            .targets
            .iter()
            .any(|target| target.split(':').any(|feature| feature == "xnack+"));
        let mut module_globals = BTreeMap::new();
        let mut module_end = MODULE_GLOBAL_BASE;
        for object in &program.globals {
            if object.symbol.is_empty()
                || object.size == 0
                || !object.alignment.is_power_of_two()
                || object.initial_bytes.len() as u64 != object.size
            {
                return Err(EvalError::InvalidConfig(format!(
                    "invalid size/alignment/initializer for module global '{}'",
                    object.symbol
                )));
            }
            let base = module_end
                .checked_add(object.alignment - 1)
                .map(|address| address & !(object.alignment - 1))
                .ok_or_else(|| {
                    EvalError::InvalidConfig(
                        "module-global alignment overflows address space".into(),
                    )
                })?;
            module_end = base.checked_add(object.size).ok_or_else(|| {
                EvalError::InvalidConfig("module-global window overflows address space".into())
            })?;
            if module_globals
                .insert(
                    object.symbol.clone(),
                    GlobalAllocation {
                        base,
                        size: object.size,
                        read_only: object.read_only,
                    },
                )
                .is_some()
            {
                return Err(EvalError::InvalidConfig(format!(
                    "duplicate module global '{}'",
                    object.symbol
                )));
            }
        }
        let mut bound_globals = std::collections::BTreeSet::new();
        for (name, _) in &config.global_values {
            if !bound_globals.insert(name) {
                return Err(EvalError::InvalidConfig(format!(
                    "duplicate module-global binding '{name}'"
                )));
            }
            let object = module_globals.get(name).ok_or_else(|| {
                EvalError::InvalidConfig(format!("no module global named '{name}'"))
            })?;
            if object.read_only {
                return Err(EvalError::InvalidConfig(format!(
                    "read-only module global '{name}' cannot be rebound"
                )));
            }
            if ![1, 2, 4, 8].contains(&object.size) {
                return Err(EvalError::InvalidConfig(format!(
                    "module-global binding '{name}' requires a scalar object of 1,2,4 or8 bytes"
                )));
            }
        }
        if module_end > MODULE_GLOBAL_BASE {
            for array in &config.arrays {
                if array.base < module_end && MODULE_GLOBAL_BASE < array.base + array.size_bytes() {
                    return Err(EvalError::InvalidConfig(format!(
                        "array '{}' overlaps the reserved module-global window",
                        array.name
                    )));
                }
            }
        }
        // Common validation established a nonzero, representable product.
        let threads = config.num_threads();
        if threads > 1024 {
            return Err(EvalError::InvalidConfig(
                "native AMD workgroup must contain 1..=1024 workitems".into(),
            ));
        }
        for (groups, threads) in [
            (config.grid_dim.0, config.block_dim.0),
            (config.grid_dim.1, config.block_dim.1),
            (config.grid_dim.2, config.block_dim.2),
        ] {
            if groups.checked_mul(threads).is_none() {
                return Err(EvalError::InvalidConfig(
                    "HSA grid size exceeds the 32-bit dispatch-packet field".into(),
                ));
            }
        }
        for array in &config.arrays {
            if array.kind == ArrayKind::IndexInput
                && array.elem_width < 4
                && array.len > 1u64 << (8 * array.elem_width)
            {
                return Err(EvalError::InvalidConfig(
                    "index array element width cannot represent its indices".into(),
                ));
            }
            if array.base < DISPATCH_BASE + 64 && array.base + array.size_bytes() > KERNARG_BASE {
                return Err(EvalError::InvalidConfig(
                    "allocation overlaps analyzer ABI address window".into(),
                ));
            }
            if ![1, 2, 4, 8].contains(&array.elem_width) {
                return Err(EvalError::InvalidConfig(
                    "supported array element widths are 1,2,4,8 bytes".into(),
                ));
            }
        }
        let lds_size = program
            .descriptor
            .value("group_segment_fixed_size", 0)
            .checked_add(config.dynamic_shared_bytes)
            .ok_or_else(|| EvalError::InvalidConfig("LDS size overflow".into()))?;
        let architecture = AmdArchitecture::from_targets(&provenance.targets).ok_or_else(|| {
            EvalError::InvalidConfig("expected one supported native AMD target".into())
        })?;
        if architecture == AmdArchitecture::Gfx1250
            && (config.workgroup_id.1 > u16::MAX as u32 || config.workgroup_id.2 > u16::MAX as u32)
        {
            return Err(EvalError::InvalidConfig(
                "gfx1250 workgroup_id Y/Z must fit the 16-bit architected ABI fields".into(),
            ));
        }
        if lds_size > architecture.lds_capacity() {
            return Err(EvalError::InvalidConfig(format!(
                "requested LDS exceeds {} capacity of {} bytes",
                architecture.name(),
                architecture.lds_capacity()
            )));
        }
        for allocation in &config.shared_allocations {
            if allocation
                .offset
                .checked_add(allocation.size)
                .is_none_or(|end| end > lds_size)
            {
                return Err(EvalError::InvalidConfig(format!(
                    "shared allocation '{}' exceeds configured LDS capacity",
                    allocation.name
                )));
            }
        }
        if program.descriptor.flag("tg_split") {
            return Err(EvalError::InvalidConfig(
                "tg_split workgroups require a different LDS/barrier execution model".into(),
            ));
        }
        fn registers(operand: &Operand, result: &mut Vec<Register>) {
            match operand {
                Operand::Register(register) => result.push(*register),
                Operand::Modified { value, .. } => registers(value, result),
                _ => {}
            }
        }
        let mut referenced = Vec::new();
        for instruction in &program.instructions {
            instruction.for_each_operand(&mut |operand| registers(operand, &mut referenced));
        }
        if referenced
            .iter()
            .any(|register| register.class == RegisterClass::Accumulator)
        {
            let offset = program.descriptor.get("accum_offset").ok_or_else(|| {
                EvalError::InvalidConfig(
                    "AGPR use requires explicit accum_offset descriptor".into(),
                )
            })?;
            if offset < 4 || offset % 4 != 0 || offset > 256 {
                return Err(EvalError::InvalidConfig(
                    "invalid accumulator register offset".into(),
                ));
            }
            if referenced.iter().any(|register| {
                register.class == RegisterClass::Vector
                    && register.index as u64 + register.count as u64 > offset
            }) {
                return Err(EvalError::InvalidConfig(
                    "explicit VGPR range aliases the AGPR register window".into(),
                ));
            }
        }
        if program
            .descriptor
            .get("max_flat_workgroup_size")
            .is_some_and(|max| threads as u64 > max)
        {
            return Err(EvalError::InvalidConfig(
                "launch exceeds metadata max_flat_workgroup_size".into(),
            ));
        }
        for (field, actual) in [
            ("reqd_workgroup_size_x", config.block_dim.0),
            ("reqd_workgroup_size_y", config.block_dim.1),
            ("reqd_workgroup_size_z", config.block_dim.2),
        ] {
            if program
                .descriptor
                .get(field)
                .is_some_and(|required| required != actual as u64)
            {
                return Err(EvalError::InvalidConfig(format!(
                    "launch does not match metadata {field}"
                )));
            }
        }
        if program.descriptor.flag("wavefront_size32") != (architecture.wave_size() == 32) {
            return Err(EvalError::InvalidConfig(format!(
                "{} requires wave{}",
                architecture.name(),
                architecture.wave_size()
            )));
        }
        if program
            .descriptor
            .value("user_sgpr_kernarg_preload_length", 0)
            != 0
        {
            return Err(EvalError::InvalidConfig(
                "kernarg preloading not implemented".into(),
            ));
        }
        let mut result = Self {
            program,
            config,
            arena: ExprArena::new(),
            global: Memory::default(),
            inputs: InputBacking::default(),
            lds: Memory::default(),
            kernargs: Memory::default(),
            dispatch: Memory::default(),
            waves: Vec::new(),
            builder: GraphBuilder::new(provenance),
            stats: Stats::default(),
            op_counts: BTreeMap::new(),
            lds_size,
            architecture,
            byte_formats: BTreeMap::new(),
            word_formats: BTreeMap::new(),
            predicate_masks: Vec::new(),
            float_bit_patterns: BTreeMap::new(),
            xnack_replay_enabled,
            module_globals,
            named: named_barrier::State::default(),
            private: private::State::new(program, architecture)?,
            mode_fields: BTreeMap::new(),
            opaque_symbols: BTreeSet::new(),
        };
        result
            .builder
            .set_shared_allocation_layout(result.config.shared_allocations.clone());
        if let Some(bytes) = result.private.extent {
            result
                .builder
                .assume(Assumption::FixedPrivateAllocation(bytes));
        }
        result.configure_native_profile()?;
        if architecture == AmdArchitecture::Gfx1250
            || architecture == AmdArchitecture::Gfx950
                && program.instructions.iter().any(|instruction| {
                    matches!(instruction.operation(), Operation::Memory(memory) if memory.kind == MemoryKind::BufferLoadToLds)
                })
        {
            result
                .builder
                .assume(Assumption::ConfiguredLdsBytes(lds_size));
        }
        result.initialize_memory()?;
        for id in 0..threads.div_ceil(architecture.wave_size()) {
            let mut wave = Wave::new(
                id,
                (threads - id * architecture.wave_size()).min(architecture.wave_size()),
                architecture.wave_size(),
            );
            wave.m0 = lds_size as u32;
            result.initialize_wave(&mut wave)?;
            result.waves.push(wave);
        }
        Ok(result)
    }
    fn initialize_memory(&mut self) -> Result<(), EvalError> {
        for object in &self.program.globals {
            let allocation = self.module_globals[&object.symbol];
            if let Some((_, binding)) = self
                .config
                .global_values
                .iter()
                .find(|(name, _)| name == &object.symbol)
            {
                if object.size <= 4 {
                    let bits = (*binding as u64)
                        & if object.size == 4 {
                            u32::MAX as u64
                        } else {
                            (1u64 << (8 * object.size)) - 1
                        };
                    self.global.initialize(
                        allocation.base,
                        object.size,
                        Value::Integer(bits as u32),
                    );
                } else {
                    self.global
                        .initialize(allocation.base, 4, Value::Integer(*binding as u32));
                    self.global.initialize(
                        allocation.base + 4,
                        4,
                        Value::Integer((*binding as u64 >> 32) as u32),
                    );
                }
            } else {
                for (offset, &byte) in object.initial_bytes.iter().enumerate() {
                    self.global.initialize(
                        allocation.base + offset as u64,
                        1,
                        Value::Bits(byte as u32),
                    );
                }
            }
        }
        self.inputs = InputBacking::new(&self.config.arrays, &mut self.arena);
        let mut position = 0;
        for arg in &self.program.args {
            let value = if arg.value_kind.starts_with("hidden_") {
                self.hidden_arg(&arg.value_kind)?
            } else {
                let param = self.config.params.get(position).ok_or_else(|| {
                    EvalError::InvalidConfig(format!(
                        "missing argument {position} ({:?})",
                        arg.name
                    ))
                })?;
                position += 1;
                if let ParamValue::Bytes(bytes) = param {
                    if arg.value_kind != "by_value" || bytes.len() as u64 != arg.size {
                        return Err(EvalError::InvalidConfig(format!(
                            "byte payload for argument {:?} must match its by_value metadata size ({} bytes)",
                            arg.name, arg.size
                        )));
                    }
                    for (offset, byte) in bytes.iter().enumerate() {
                        self.kernargs.initialize(
                            KERNARG_BASE + arg.offset + offset as u64,
                            1,
                            Value::Bits(u32::from(*byte)),
                        );
                    }
                    continue;
                }
                if matches!(param, ParamValue::Float(_) | ParamValue::SymFloat(_)) {
                    let expected = match arg.size {
                        4 => "f32",
                        8 => "f64",
                        _ => {
                            return Err(EvalError::InvalidConfig(
                                "floating arguments require32-bit or64-bit storage".into(),
                            ));
                        }
                    };
                    if arg.value_kind != "by_value"
                        || arg.value_type.as_deref().is_some_and(|ty| ty != expected)
                    {
                        return Err(EvalError::InvalidConfig(format!(
                            "floating argument metadata must describe a by-value {expected}"
                        )));
                    }
                }
                match param {
                    ParamValue::Bytes(_) => unreachable!("native byte payload bound above"),
                    ParamValue::Struct(_) => {
                        return Err(EvalError::InvalidConfig(
                            "structured parameters require the PTX backend; use exact ABI bytes for native AMD aggregates".into(),
                        ));
                    }
                    ParamValue::Int(value) => Some((*value as u64, None)),
                    ParamValue::Float(value) => Some((
                        if arg.size == 8 {
                            value.to_bits()
                        } else {
                            (*value as f32).to_bits() as u64
                        },
                        None,
                    )),
                    ParamValue::SymFloat(name) => {
                        Some((0, Some(self.arena.param_symbol(name.clone()))))
                    }
                    ParamValue::ArrayPtr(name) => Some((
                        self.config
                            .array(name)
                            .ok_or_else(|| {
                                EvalError::InvalidConfig(format!("unknown pointer array {name}"))
                            })?
                            .base,
                        None,
                    )),
                }
            };
            if let Some((bits, expression)) = value {
                if let Some(expression) = expression {
                    if arg.size == 8 {
                        self.kernargs.initialize(
                            KERNARG_BASE + arg.offset,
                            4,
                            Value::WidePart {
                                expr: expression,
                                high: false,
                                kind: WideKind::Float,
                            },
                        );
                        self.kernargs.initialize(
                            KERNARG_BASE + arg.offset + 4,
                            4,
                            Value::WidePart {
                                expr: expression,
                                high: true,
                                kind: WideKind::Float,
                            },
                        );
                    } else {
                        self.kernargs.initialize(
                            KERNARG_BASE + arg.offset,
                            4,
                            Value::Real(expression),
                        );
                    }
                } else if arg.value_kind.starts_with("hidden_") && arg.size == 4 {
                    // These fields are numeric launch quantities, rather than
                    // pointers or arbitrary data bits. Preserve that distinction
                    // when native code copies grid dimensions directly to output.
                    self.kernargs.initialize(
                        KERNARG_BASE + arg.offset,
                        4,
                        Value::Integer(bits as u32),
                    );
                } else {
                    for byte in 0..arg.size {
                        if byte >= 8 {
                            return Err(EvalError::InvalidConfig(format!(
                                "argument {:?} wider than 64 bits",
                                arg.name
                            )));
                        }
                        self.kernargs.initialize(
                            KERNARG_BASE + arg.offset + byte,
                            1,
                            Value::Bits(((bits >> (8 * byte)) & 255) as u32),
                        );
                    }
                }
            }
        }
        if position != self.config.params.len() {
            return Err(EvalError::InvalidConfig(format!(
                "{} parameters provided but metadata declares {position} explicit arguments",
                self.config.params.len()
            )));
        }
        // HSA dispatch packet fields used by LLVM AMDGPU ABI.
        for (offset, width, bits) in [
            (4, 2, self.config.block_dim.0 as u64),
            (6, 2, self.config.block_dim.1 as u64),
            (8, 2, self.config.block_dim.2 as u64),
            (
                12,
                4,
                self.config.grid_dim.0 as u64 * self.config.block_dim.0 as u64,
            ),
            (
                16,
                4,
                self.config.grid_dim.1 as u64 * self.config.block_dim.1 as u64,
            ),
            (
                20,
                4,
                self.config.grid_dim.2 as u64 * self.config.block_dim.2 as u64,
            ),
            (28, 4, self.lds_size),
        ] {
            for byte in 0..width {
                self.dispatch.initialize(
                    DISPATCH_BASE + offset + byte,
                    1,
                    Value::Bits(((bits >> (8 * byte)) & 255) as u32),
                );
            }
        }
        Ok(())
    }
    fn hidden_arg(&self, kind: &str) -> Result<Option<(u64, Option<ExprId>)>, EvalError> {
        let value = match kind {
            "hidden_block_count_x" => self.config.grid_dim.0 as u64,
            "hidden_block_count_y" => self.config.grid_dim.1 as u64,
            "hidden_block_count_z" => self.config.grid_dim.2 as u64,
            "hidden_group_size_x" => self.config.block_dim.0 as u64,
            "hidden_group_size_y" => self.config.block_dim.1 as u64,
            "hidden_group_size_z" => self.config.block_dim.2 as u64,
            "hidden_remainder_x"
            | "hidden_remainder_y"
            | "hidden_remainder_z"
            | "hidden_global_offset_x"
            | "hidden_global_offset_y"
            | "hidden_global_offset_z" => 0,
            "hidden_grid_dims" => {
                if self.config.grid_dim.2 > 1 || self.config.block_dim.2 > 1 {
                    3
                } else if self.config.grid_dim.1 > 1 || self.config.block_dim.1 > 1 {
                    2
                } else {
                    1
                }
            }
            "hidden_dynamic_lds_size" => self.config.dynamic_shared_bytes,
            _ => return Ok(None), // Unknown runtime facilities remain uninitialized, never guessed.
        };
        Ok(Some((value, None)))
    }
    fn global_read(&mut self, addr: u64, width: u64) -> Result<Value, String> {
        self.inputs.ensure(
            &self.config.arrays,
            &mut self.global,
            &mut self.arena,
            addr,
            width,
        )?;
        self.global.read(addr, width)
    }
    fn global_write(&mut self, addr: u64, width: u64, value: Value) -> Result<Value, String> {
        self.inputs.ensure(
            &self.config.arrays,
            &mut self.global,
            &mut self.arena,
            addr,
            width,
        )?;
        self.global.write(addr, width, value)
    }
    fn initialize_wave(&self, wave: &mut Wave) -> Result<(), EvalError> {
        if self.architecture == AmdArchitecture::Gfx1250 {
            // CDNA5 ISA3.5.3.1: an unclustered launch does not initialize TTMP6.
            // PartialBits allows unused compiler fallback calculations to remain
            // unknown; it never supplies concrete address, control or output bits.
            wave.temporary[6] = Some(Value::PartialBits {
                bits: 0,
                defined: 0,
            });
            let yz_valid = self.program.descriptor.flag("system_sgpr_workgroup_id_y")
                || self.program.descriptor.flag("system_sgpr_workgroup_id_z");
            wave.temporary[7] = Some(if yz_valid {
                Value::Integer(
                    self.config.workgroup_id.1
                        | if self.program.descriptor.flag("system_sgpr_workgroup_id_z") {
                            self.config.workgroup_id.2 << 16
                        } else {
                            0
                        },
                )
            } else {
                Value::PartialBits {
                    bits: 0,
                    defined: 0,
                }
            });
            wave.temporary[9] = Some(Value::Integer(self.config.workgroup_id.0));
            wave.temporary[8] = Some(Value::PartialBits {
                bits: (wave.id << 25) | (u32::from(yz_valid) << 30),
                // DebugMark=0, GridYZvalid and waveID are known. The runtime
                // dispatch packet index in bits0..24 is not inferred.
                defined: 0xfe00_0000,
            });
        }
        let mut sgpr = 0usize;
        let descriptor = &self.program.descriptor;
        // Reserve every enabled ABI slot, leaving unavailable runtime inputs
        // undefined. Unused slots are allowed; consuming one fails at its read.
        // Workgroup zero does not determine Dispatch ID, and the static private
        // segment size does not determine the runtime allocation size.
        for (field, count, value) in [
            ("user_sgpr_private_segment_buffer", 4, None),
            ("user_sgpr_dispatch_ptr", 2, Some(DISPATCH_BASE)),
            ("user_sgpr_queue_ptr", 2, None),
            ("user_sgpr_kernarg_segment_ptr", 2, Some(KERNARG_BASE)),
            ("user_sgpr_dispatch_id", 2, None),
            ("user_sgpr_flat_scratch_init", 2, None),
            ("user_sgpr_private_segment_size", 1, None),
        ] {
            if descriptor.flag(field) {
                if let Some(value) = value {
                    for component in 0..count {
                        wave.scalar[sgpr + component] =
                            Some(Value::Bits((value >> (32 * component)) as u32));
                    }
                }
                sgpr += count;
            }
        }
        if descriptor
            .get("user_sgpr_count")
            .is_some_and(|count| count != sgpr as u64)
        {
            return Err(EvalError::InvalidConfig(format!(
                "descriptor user_sgpr_count disagrees with enabled ABI inputs: expected {sgpr}"
            )));
        }
        for (field, coordinate) in [
            ("system_sgpr_workgroup_id_x", self.config.workgroup_id.0),
            ("system_sgpr_workgroup_id_y", self.config.workgroup_id.1),
            ("system_sgpr_workgroup_id_z", self.config.workgroup_id.2),
        ] {
            if descriptor.flag(field) && self.architecture != AmdArchitecture::Gfx1250 {
                wave.scalar[sgpr] = Some(Value::Integer(coordinate));
                sgpr += 1;
            }
        }
        if descriptor.flag("system_sgpr_workgroup_info") {
            return Err(EvalError::InvalidConfig(
                "workgroup_info SGPR not implemented".into(),
            ));
        }
        let dimensions = descriptor.value("system_vgpr_workitem_id", 0);
        for lane in 0..wave.width as u8 {
            if wave.valid & (1u64 << lane) == 0 {
                continue;
            }
            let linear = wave.id * wave.width + lane as u32;
            let x = linear % self.config.block_dim.0;
            let y = (linear / self.config.block_dim.0) % self.config.block_dim.1;
            let z = linear / (self.config.block_dim.0 * self.config.block_dim.1);
            // gfx942/gfx950 use the packed-workitem-ID ABI: x in 9:0, enabled y in
            // 19:10 and enabled z in 29:20 of v0. v1/v2 are not initialized.
            let packed = x
                | if dimensions >= 1 { y << 10 } else { 0 }
                | if dimensions >= 2 { z << 20 } else { 0 };
            wave.vector.insert((0, lane), Value::Integer(packed));
        }
        Ok(())
    }
    pub fn execute(mut self, observer: Option<&mut InstructionObserver<'_>>) -> RunArtifacts {
        let end = self.run(observer);
        let mut raw_outputs = None;
        let (terminal, outputs) = match end {
            Ok(()) => {
                raw_outputs = self.extract_raw();
                let outputs = match &raw_outputs {
                    Some(raw) => Ok(self.exact_raw_outputs(raw)),
                    None => self.extract().map_err(OutputsUnavailable::Extraction),
                };
                (Terminal::Completed, outputs)
            }
            Err(EvalError::Invalid { ref what, .. })
                if what == "workgroup rendezvous made no progress"
                    && self.waves.iter().any(|w| w.named_wait.is_some())
                    && self
                        .waves
                        .iter()
                        .all(|w| w.exited || w.named_wait.is_some() || w.at_barrier.is_some()) =>
            {
                let mut blocked = Vec::new();
                for wave in self.waves.iter().filter(|w| !w.exited) {
                    let on = if let Some(wait) = wave.named_wait {
                        let (expected, arrived) = self.named.counts(wait);
                        BlockedOn::AmdNamedBarrier {
                            object: wait.object,
                            phase: wait.phase,
                            expected,
                            arrived,
                        }
                    } else {
                        let generation = wave.at_barrier.unwrap();
                        BlockedOn::AmdBarrier {
                            generation,
                            missing_waves: self
                                .waves
                                .iter()
                                .filter(|w| !w.exited && w.signal_generation != Some(generation))
                                .map(|w| w.id)
                                .collect(),
                        }
                    };
                    for lane in 0..wave.valid.count_ones() {
                        blocked.push(BlockedThread {
                            thread: ThreadIx(wave.id * wave.width + lane),
                            pc: ProgramPoint(wave.pc as u32),
                            on: on.clone(),
                        });
                    }
                }
                let locations = blocked
                    .iter()
                    .map(|b| (ThreadId(b.thread.0), b.pc.0 as usize))
                    .collect();
                (
                    Terminal::Stuck { blocked },
                    Err(OutputsUnavailable::Stuck { blocked: locations }),
                )
            }
            Err(error) => {
                let cause = if matches!(error, EvalError::Unsupported { .. }) {
                    AbortCause::Unsupported
                } else {
                    AbortCause::Error
                };
                let (thread, pc) = match &error {
                    EvalError::Invalid { wave, pc, .. }
                    | EvalError::TensorBounds { wave, pc, .. }
                    | EvalError::GlobalBounds { wave, pc, .. } => (
                        Some(ThreadIx(wave * self.architecture.wave_size())),
                        Some(ProgramPoint(*pc as u32)),
                    ),
                    EvalError::Unsupported { pc, .. } => (None, Some(ProgramPoint(*pc as u32))),
                    _ => (None, None),
                };
                (
                    Terminal::Aborted {
                        thread,
                        pc,
                        cause,
                        reason: error.to_string(),
                    },
                    Err(OutputsUnavailable::Aborted(error)),
                )
            }
        };
        RunArtifacts {
            arena: self.arena,
            outputs,
            raw_outputs,
            graph: self.builder.finish(terminal),
            stats: self.stats,
            op_counts: self.op_counts,
        }
    }
    fn extract(&mut self) -> Result<ExtractedOutputs, EvalError> {
        let mut outputs = Vec::new();
        for array in &self.config.arrays {
            if !array.kind.is_output() {
                continue;
            }
            let mut elements = Vec::new();
            for index in 0..array.len {
                let addr = array.base + array.elem_width * index;
                let fully_written = (0..array.elem_width).all(|byte| {
                    self.global
                        .written
                        .range(..=addr + byte)
                        .next_back()
                        .is_some_and(|(&start, &width)| addr + byte - start < width)
                });
                if fully_written {
                    elements.push((
                        index,
                        self.global
                            .expression(addr, array.elem_width, &mut self.arena)
                            .map_err(|what| EvalError::Invalid {
                                wave: 0,
                                lane: None,
                                pc: 0,
                                what,
                            })?,
                    ));
                }
            }
            outputs.push((array.name.clone(), elements));
        }
        if !self.float_bit_patterns.is_empty() {
            let wave = Wave::new(0, 1, self.architecture.wave_size());
            let mut pending = outputs
                .iter()
                .flat_map(|(_, xs)| xs.iter().map(|(_, id)| *id))
                .collect::<Vec<_>>();
            let mut visited = std::collections::BTreeSet::new();
            while let Some(id) = pending.pop() {
                if !visited.insert(id) {
                    continue;
                }
                if let Some(pattern) = self.float_bit_patterns.get(&id).cloned() {
                    for input in pattern.input_words {
                        self.claim_word_format(&wave, input, WordFormat::Integer32)?;
                    }
                }
                self.arena
                    .node(id)
                    .for_each_child(|child| pending.push(child));
            }
        }
        Ok(outputs)
    }
    fn run(&mut self, mut observer: Option<&mut InstructionObserver<'_>>) -> Result<(), EvalError> {
        private::admit_lane_program(self.program, self.architecture)?;
        loop {
            if self.waves.iter().all(|wave| wave.exited) {
                return Ok(());
            }
            let mut progress = false;
            for index in 0..self.waves.len() {
                if self.waves[index].exited
                    || self.waves[index].at_barrier.is_some()
                    || self.waves[index].named_wait.is_some()
                {
                    continue;
                }
                if self.stats.instructions >= self.config.max_instructions {
                    return Err(EvalError::BudgetExceeded {
                        limit: self.config.max_instructions,
                    });
                }
                let mut wave = std::mem::replace(
                    &mut self.waves[index],
                    Wave::new(index as u32, 1, self.architecture.wave_size()),
                );
                let step = self.step(&mut wave);
                self.waves[index] = wave;
                step?;
                if let Some(observer) = observer.as_mut()
                    && observer(self.builder.graph_so_far()).is_break()
                {
                    return Err(EvalError::ObserverStopped);
                }
                progress = true;
            }
            if self.architecture == AmdArchitecture::Gfx1250 {
                progress |= self.release_named_barriers();
                // SIGNAL snapshots one phase without blocking. A WAIT can
                // complete as soon as every surviving wave has signalled it,
                // even if another participant has not reached its WAIT yet.
                let ready: Vec<_> = self
                    .waves
                    .iter()
                    .filter_map(|wave| wave.at_barrier)
                    .filter(|generation| {
                        self.waves.iter().filter(|wave| !wave.exited).all(|wave| {
                            wave.signal_generation
                                .is_some_and(|signal| signal >= *generation)
                        })
                    })
                    .collect();
                for wave in self.waves.iter_mut().filter(|wave| !wave.exited) {
                    if let Some(generation) = wave
                        .at_barrier
                        .filter(|generation| ready.contains(generation))
                    {
                        self.builder.amd_event(
                            wave.id,
                            ProgramPoint(wave.pc as u32),
                            AmdEventKind::SplitBarrierRelease {
                                barrier: -1,
                                generation,
                            },
                        );
                        wave.at_barrier = None;
                        wave.generation += 1;
                        wave.pc += 1;
                        progress = true;
                    }
                }
            }
            if self.architecture != AmdArchitecture::Gfx1250
                && self.waves.iter().any(|wave| wave.at_barrier.is_some())
                && self
                    .waves
                    .iter()
                    .filter(|wave| !wave.exited)
                    .all(|wave| wave.at_barrier.is_some())
            {
                let generation = self
                    .waves
                    .iter()
                    .find_map(|wave| if wave.exited { None } else { wave.at_barrier })
                    .unwrap();
                if self
                    .waves
                    .iter()
                    .filter(|wave| !wave.exited)
                    .any(|wave| wave.at_barrier != Some(generation))
                {
                    return Err(EvalError::Invalid {
                        wave: 0,
                        lane: None,
                        pc: 0,
                        what: "barrier generation mismatch".into(),
                    });
                }
                for wave in self.waves.iter_mut().filter(|wave| !wave.exited) {
                    self.builder.amd_event(
                        wave.id,
                        ProgramPoint(wave.pc as u32),
                        AmdEventKind::BarrierRelease { generation },
                    );
                    wave.at_barrier = None;
                    wave.generation += 1;
                    wave.pc += 1;
                }
                progress = true;
            }
            if !progress {
                return Err(EvalError::Invalid {
                    wave: 0,
                    lane: None,
                    pc: 0,
                    what: "workgroup rendezvous made no progress".into(),
                });
            }
        }
    }
    fn step(&mut self, wave: &mut Wave) -> Result<(), EvalError> {
        let function_start = wave.calls.last().map_or(0, |frame| frame.target);
        if !self.program.functions.iter().any(|function| {
            function.start == function_start && function.start <= wave.pc && wave.pc < function.end
        }) {
            return Err(wave.fail(None, "execution escaped its active source function"));
        }
        let instruction = self
            .program
            .instructions
            .get(wave.pc)
            .ok_or_else(|| wave.fail(None, "fell off kernel without s_endpgm"))?;
        let instruction = instruction.clone();
        wave.previous_pc = wave.current_pc.replace(wave.pc);
        if self.architecture == AmdArchitecture::Gfx950
            && matches!(
                instruction.alu_operation(),
                Some(AluOperation::VReadlaneB32 | AluOperation::VWritelaneB32)
            )
        {
            private::lane(wave, &instruction)?;
        }
        let (mut reads, writes) = self.instruction_registers(&instruction, wave);
        self.private
            .before_instruction(wave, &instruction, &reads, &writes)?;
        if instruction.opcode() == Opcode::Matrix {
            reads.extend(
                (126..126 + (wave.width / 32) as u16).map(|index| AmdRegister {
                    file: AmdRegisterFile::Scalar,
                    index,
                    lanes: u64::MAX,
                }),
            );
        }
        self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::Instruction {
                exec_mask: wave.exec,
                reads,
                writes,
                scalar_mask_reads: self.scalar_mask_reads(&instruction, wave),
            },
        );
        self.stats.instructions += 1;
        *self
            .op_counts
            .entry(super::interp_kind(
                instruction.opcode(),
                instruction.mnemonic(),
            ))
            .or_default() += 1;
        let alu = instruction.alu_operation();
        // Mask flags used as ordinary vector data have additional software
        // dependency/alias requirements. Carry and cndmask selector roles are
        // separately supported; treating all such reads as scalar data would
        // bypass those obligations because the graph has no special-file slice.
        if matches!(
            instruction.opcode(),
            Opcode::VectorAlu | Opcode::VectorMemory | Opcode::Matrix
        ) {
            fn special(operand: &Operand) -> Option<SpecialRegister> {
                match operand {
                    Operand::Register(register) => register.special,
                    Operand::Modified { value, .. } => special(value),
                    _ => None,
                }
            }
            let dual_destination = matches!(
                alu,
                Some(
                    AluOperation::VAddCoU32
                        | AluOperation::VAddcCoU32
                        | AluOperation::VSubCoU32
                        | AluOperation::VSubbCoU32
                        | AluOperation::VSubrevCoU32
                        | AluOperation::VSubbrevCoU32
                        | AluOperation::VMadU64U32
                        | AluOperation::VMadI64I32
                )
            );
            for (index, operand) in instruction.operands().iter().enumerate() {
                if instruction.opcode() != Opcode::VectorMemory
                    && (index == 0 || (dual_destination && index == 1))
                {
                    continue;
                }
                let Some(register) = special(operand) else {
                    continue;
                };
                if register == SpecialRegister::Null {
                    continue;
                }
                if self.architecture == AmdArchitecture::Gfx1250
                    && alu == Some(AluOperation::VMbcntLoU32B32)
                    && index == 1
                    && register == SpecialRegister::ExecLo
                {
                    // Wave32 lane rank uses the architectural concrete EXEC
                    // mask; the default gfx1250 ALU pipeline is interlocked.
                    continue;
                }
                let selector = alu == Some(AluOperation::VCndmaskB32) && index == 3;
                let carry = matches!(
                    alu,
                    Some(
                        AluOperation::VAddcCoU32
                            | AluOperation::VSubbCoU32
                            | AluOperation::VSubbrevCoU32
                    )
                ) && index == 4;
                if (selector || carry)
                    && (register == SpecialRegister::Vcc
                        || (wave.width == 32 && register == SpecialRegister::VccLo))
                {
                    continue;
                }
                return Err(EvalError::Unsupported{pc:wave.pc,what:"special mask registers used as vector data require additional Table 11 dependency checks".into()});
            }
        }
        match instruction.operation() {
            Operation::Call(call) => {
                self.call(wave, call)?;
                return Ok(());
            }
            Operation::Return(ret) => {
                self.return_from_call(wave, ret)?;
                return Ok(());
            }
            Operation::Memory(_) => self.memory_instruction(wave, &instruction)?,
            Operation::Wait(_) => self.wait(wave, &instruction)?,
            Operation::Barrier(barrier) if self.architecture == AmdArchitecture::Gfx1250 => {
                if !matches!(instruction.operands(), [Operand::Integer(-1)]) {
                    if self.named_barrier(wave, &instruction)? {
                        return Ok(());
                    }
                } else {
                    match barrier.operation {
                        BarrierOperation::Signal => {
                            if wave.signal_generation == Some(wave.generation) {
                                return Err(wave.fail(
                                    None,
                                    "duplicate workgroup barrier signal before phase wait",
                                ));
                            }
                            self.builder.amd_event(
                                wave.id,
                                ProgramPoint(wave.pc as u32),
                                AmdEventKind::SplitBarrierSignal {
                                    barrier: -1,
                                    generation: wave.generation,
                                    exec_mask: wave.exec,
                                },
                            );
                            wave.signal_generation = Some(wave.generation);
                        }
                        BarrierOperation::Wait => {
                            if wave.signal_generation != Some(wave.generation) {
                                return Err(self.unsupported(wave, &instruction));
                            }
                            self.builder.amd_event(
                                wave.id,
                                ProgramPoint(wave.pc as u32),
                                AmdEventKind::SplitBarrierWait {
                                    barrier: -1,
                                    generation: wave.generation,
                                },
                            );
                            wave.at_barrier = Some(wave.generation);
                            self.stats.block_syncs += 1;
                            return Ok(());
                        }
                        _ => return Err(self.unsupported(wave, &instruction)),
                    }
                }
            }
            Operation::Barrier(_) => {
                self.builder.amd_event(
                    wave.id,
                    ProgramPoint(wave.pc as u32),
                    AmdEventKind::BarrierArrive {
                        generation: wave.generation,
                        exec_mask: wave.exec,
                    },
                );
                wave.at_barrier = Some(wave.generation);
                self.stats.block_syncs += 1;
                return Ok(());
            }
            Operation::Branch(branch) => {
                let take = match branch.condition {
                    BranchCondition::Always => true,
                    BranchCondition::SccZero => !wave.read_scc()?,
                    BranchCondition::SccNonzero => wave.read_scc()?,
                    BranchCondition::VccZero => wave.read_vcc()? == 0,
                    BranchCondition::VccNonzero => wave.read_vcc()? != 0,
                    BranchCondition::ExecZero => wave.exec == 0,
                    BranchCondition::ExecNonzero => wave.exec != 0,
                };
                if take {
                    wave.pc = branch.target;
                    return Ok(());
                }
            }
            Operation::Control(control) => match control.operation {
                ControlOperation::VectorNop => {}
                ControlOperation::SendMessage
                    if self.architecture == AmdArchitecture::Gfx1250
                        && instruction.operands() == [Operand::Integer(3)] => {}
                ControlOperation::EndProgram => {
                    if !wave.calls.is_empty() {
                        return Err(wave.fail(None, "device function cannot execute s_endpgm"));
                    }
                    // CDNA4 S_ENDPGM drains all memory counters before the wave
                    // terminates and ceases participating in workgroup barriers.
                    if self.architecture != AmdArchitecture::Gfx1250 {
                        self.builder.amd_event(
                            wave.id,
                            ProgramPoint(wave.pc as u32),
                            AmdEventKind::Wait {
                                vmcnt: Some(0),
                                lgkmcnt: Some(0),
                                expcnt: Some(0),
                            },
                        );
                    }
                    for lane in 0..wave.width {
                        if wave.valid & (1u64 << lane) != 0 {
                            self.builder.exit(
                                ThreadIx(wave.id * wave.width + lane),
                                ProgramPoint(wave.pc as u32),
                            );
                        }
                    }
                    wave.exited = true;
                }
                ControlOperation::Clause => {
                    // An encoding clause marker; per-operation X replay sources remain recorded.
                    if self.architecture != AmdArchitecture::Gfx1250
                        || !matches!(instruction.operands(), [Operand::Integer(0..=63)])
                    {
                        return Err(self.unsupported(wave, &instruction));
                    }
                }
                ControlOperation::DelayAlu => {
                    let Some(Operand::Integer(encoded)) = instruction.operands().first() else {
                        return Err(self.unsupported(wave, &instruction));
                    };
                    let instid0 = AmdAluDependency::from_encoding((encoded & 15) as u8)
                        .ok_or_else(|| self.unsupported(wave, &instruction))?;
                    let raw1 = ((encoded >> 7) & 15) as u8;
                    let instid1 = if raw1 == 0 {
                        None
                    } else {
                        Some(
                            AmdAluDependency::from_encoding(raw1)
                                .ok_or_else(|| self.unsupported(wave, &instruction))?,
                        )
                    };
                    self.builder.amd_event(
                        wave.id,
                        ProgramPoint(wave.pc as u32),
                        AmdEventKind::DelayAlu {
                            instid0,
                            instskip: ((encoded >> 4) & 7) as u8,
                            instid1,
                        },
                    );
                }
                ControlOperation::ScalarNop => {
                    let Some(Operand::Integer(count)) = instruction.operands().first() else {
                        return Err(self.unsupported(wave, &instruction));
                    };
                    if !(0..=15).contains(count) {
                        return Err(wave.fail(None, "s_nop wait count outside 0..15"));
                    }
                    self.builder.amd_event(
                        wave.id,
                        ProgramPoint(wave.pc as u32),
                        AmdEventKind::Delay {
                            cycles: *count as u16 + 1,
                        },
                    );
                }
                _ => return Err(self.unsupported(wave, &instruction)),
            },
            Operation::ScalarAlu(alu) => self.scalar(wave, &instruction, alu.operation)?,
            Operation::Dual { x, y } => {
                let original = wave.clone();
                let mut wx = original.clone();
                let mut wy = original;
                self.vector(
                    &mut wx,
                    x,
                    x.alu_operation().expect("typed vector VOPD component"),
                )?;
                self.vector(
                    &mut wy,
                    y,
                    y.alu_operation().expect("typed vector VOPD component"),
                )?;
                for (op, computed) in [(&**x, &wx), (&**y, &wy)] {
                    let register = wave.destination(&op.operands()[0])?;
                    for lane in wave.active().collect::<Vec<_>>() {
                        for component in 0..register.count {
                            wave.put(
                                register,
                                lane,
                                component,
                                computed.reg(register, lane, component)?,
                            )?;
                        }
                    }
                }
            }
            Operation::VectorAlu(alu) => self.vector(wave, &instruction, alu.operation)?,
            Operation::Matrix(_) => self.matrix(wave, &instruction)?,
        }
        if wave.exec & !wave.valid != 0 {
            return Err(EvalError::Unsupported {
                pc: wave.pc,
                what: "EXEC activates lanes outside the launched partial wave".into(),
            });
        }
        wave.pc += 1;
        Ok(())
    }
    fn unsupported(&self, wave: &Wave, instruction: &Instruction) -> EvalError {
        EvalError::Unsupported {
            pc: wave.pc,
            what: format!("{} {:?}", instruction.mnemonic(), instruction.modifiers()),
        }
    }
}

fn compare(condition: Relation, a: i64, b: i64) -> Option<bool> {
    Some(match condition {
        Relation::Eq => a == b,
        Relation::Ne | Relation::Lg | Relation::NotEq => a != b,
        Relation::Lt => a < b,
        Relation::Le => a <= b,
        Relation::Gt => a > b,
        Relation::Ge => a >= b,
        _ => return None,
    })
}
fn parse_u64(value: &str) -> Option<u64> {
    if let Some(hex) = value.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).ok()
    } else {
        value.parse().ok()
    }
}
fn parse_i64(value: &str) -> Option<i64> {
    if let Some(value) = value.strip_prefix('-') {
        i64::try_from(parse_u64(value)?).ok()?.checked_neg()
    } else {
        i64::try_from(parse_u64(value)?).ok()
    }
}
fn selectors(
    instruction: &Instruction,
    name: &str,
    count: usize,
    default: u16,
) -> Option<Vec<u16>> {
    match instruction
        .modifiers()
        .iter()
        .find(|modifier| modifier.name == name)
    {
        None => Some(vec![default; count]),
        Some(modifier) => {
            let value = modifier.value.as_deref()?;
            let values: Option<Vec<_>> = value
                .strip_prefix('[')?
                .strip_suffix(']')?
                .split(',')
                .map(|value| value.trim().parse::<u16>().ok().filter(|value| *value <= 1))
                .collect();
            values.filter(|values| values.len() == count)
        }
    }
}

/// A gfx1250-b0-wmma/3 WMMA form: graph kind (F8F6F4 kinds are completed from
/// the format modifiers), fixed A/B formats of the non-F8F6F4 forms, K and the
/// block-scale size (32 or 16 values per scale) of the scaled forms.
#[derive(Debug, Clone, Copy)]
struct SmallWmma {
    kind: AmdWmmaKind,
    a: AmdSmallFloat,
    b: AmdSmallFloat,
    f8f6f4: bool,
    k: usize,
    scale: Option<u8>,
}

impl SmallWmma {
    fn of(operation: MatrixOpcode) -> Option<Self> {
        use AmdSmallFloat::{Bf8, Fp4, Fp8};
        use MatrixOpcode as M;
        let fixed = |kind, a, b, k| SmallWmma {
            kind,
            a,
            b,
            f8f6f4: false,
            k,
            scale: None,
        };
        let f8f6f4 = |scale| SmallWmma {
            kind: AmdWmmaKind::F8f6f4_16x16x128 { a: Fp8, b: Fp8 },
            a: Fp8,
            b: Fp8,
            f8f6f4: true,
            k: 128,
            scale,
        };
        let f4 = |scale| SmallWmma {
            kind: AmdWmmaKind::F4_32x16x128,
            a: Fp4,
            b: Fp4,
            f8f6f4: false,
            k: 128,
            scale,
        };
        Some(match operation {
            M::VWmmaF3216x16x64Fp8Bf8 => fixed(AmdWmmaKind::Fp8Bf8_16x16x64, Fp8, Bf8, 64),
            M::VWmmaF3216x16x64Bf8Fp8 => fixed(AmdWmmaKind::Bf8Fp8_16x16x64, Bf8, Fp8, 64),
            M::VWmmaF3216x16x64Bf8Bf8 => fixed(AmdWmmaKind::Bf8Bf8_16x16x64, Bf8, Bf8, 64),
            M::VWmmaF3216x16x128Fp8Bf8 => fixed(AmdWmmaKind::Fp8Bf8_16x16x128, Fp8, Bf8, 128),
            M::VWmmaF3216x16x128Bf8Fp8 => fixed(AmdWmmaKind::Bf8Fp8_16x16x128, Bf8, Fp8, 128),
            M::VWmmaF3216x16x128Bf8Bf8 => fixed(AmdWmmaKind::Bf8Bf8_16x16x128, Bf8, Bf8, 128),
            M::VWmmaF3216x16x128F8f6f4 => f8f6f4(None),
            M::VWmmaScaleF3216x16x128F8f6f4 => f8f6f4(Some(32)),
            M::VWmmaScale16F3216x16x128F8f6f4 => f8f6f4(Some(16)),
            M::VWmmaF3232x16x128F4 => f4(None),
            M::VWmmaScaleF3232x16x128F4 => f4(Some(32)),
            M::VWmmaScale16F3232x16x128F4 => f4(Some(16)),
            _ => return None,
        })
    }
}

/// Block-scale formats of the scaled WMMA forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmallScale {
    /// Exponent only, bias 127, 0xFF NaN.
    E8m0,
    /// Unsigned, 5-bit exponent (bias 15), 3-bit mantissa, 0xFF NaN.
    E5m3,
    /// 7 bits: 4-bit exponent (bias 7), 3-bit mantissa, 0x7F NaN.
    E4m3,
}

/// 32 copies of a 6-bit code packed contiguously (K = 0 in bits [5:0])
/// over six 32-bit words.
fn pack6(code: u32) -> [u32; 6] {
    let mut words = [0u32; 6];
    for position in 0..192u32 {
        if code >> (position % 6) & 1 != 0 {
            words[(position / 32) as usize] |= 1 << (position % 32);
        }
    }
    words
}

/// Value of a small-float code and whether it is normal or zero in its own
/// format; `None` for NaN or Inf.
fn small_float(format: AmdSmallFloat, code: u32) -> Option<(f64, bool)> {
    let (exponent_bits, mantissa_bits, bias) = match format {
        AmdSmallFloat::Fp8 => (4, 3, 7),
        AmdSmallFloat::Bf8 => (5, 2, 15),
        AmdSmallFloat::Fp6 => (2, 3, 1),
        AmdSmallFloat::Bf6 => (3, 2, 3),
        AmdSmallFloat::Fp4 => (2, 1, 1),
    };
    let mantissa = code & ((1 << mantissa_bits) - 1);
    let exponent = (code >> mantissa_bits) & ((1 << exponent_bits) - 1);
    let negative = code & (1 << (exponent_bits + mantissa_bits)) != 0;
    match format {
        AmdSmallFloat::Fp8 if exponent == 15 && mantissa == 7 => return None,
        AmdSmallFloat::Bf8 if exponent == 31 => return None,
        _ => {}
    }
    let fraction = f64::from(mantissa) / f64::from(1u32 << mantissa_bits);
    let (magnitude, normal) = if exponent == 0 {
        (fraction * 2f64.powi(1 - bias), mantissa == 0)
    } else {
        ((1.0 + fraction) * 2f64.powi(exponent as i32 - bias), true)
    };
    Some((if negative { -magnitude } else { magnitude }, normal))
}

/// Value of a block-scale byte and whether it is a normal power of two;
/// `None` for NaN and for an E4M3 byte with bit 7 set (not evaluated).
fn small_scale(format: SmallScale, byte: u32) -> Option<(f64, bool)> {
    match format {
        SmallScale::E8m0 => (byte != 0xff).then(|| (2f64.powi(byte as i32 - 127), true)),
        SmallScale::E5m3 | SmallScale::E4m3 => {
            let (bits, bias) = if format == SmallScale::E5m3 {
                (5, 15)
            } else {
                (4, 7)
            };
            if byte >> (bits + 3) != 0 || byte == (1 << (bits + 3)) - 1 {
                return None;
            }
            let (exponent, mantissa) = (byte >> 3, byte & 7);
            let fraction = f64::from(mantissa) / 8.0;
            Some(if exponent == 0 {
                (fraction * 2f64.powi(1 - bias), false)
            } else {
                (
                    (1.0 + fraction) * 2f64.powi(exponent as i32 - bias),
                    mantissa == 0,
                )
            })
        }
    }
}

#[cfg(test)]
mod small_wmma_tests {
    use super::*;

    #[test]
    fn six_bit_packing_and_decoding() {
        assert_eq!(&pack6(0x18)[..3], &[0x1861_8618, 0x8618_6186, 0x6186_1861]);
        assert_eq!(&pack6(0x0c)[3..], &[0x0c30_c30c, 0xc30c_30c3, 0x30c3_0c30]);
        assert_eq!(small_float(AmdSmallFloat::Fp6, 0x18), Some((4.0, true)));
        assert_eq!(small_float(AmdSmallFloat::Bf6, 0x18), Some((8.0, true)));
        assert_eq!(small_float(AmdSmallFloat::Bf6, 0x0c), Some((1.0, true)));
        assert_eq!(small_float(AmdSmallFloat::Fp4, 0x2), Some((1.0, true)));
        assert_eq!(small_float(AmdSmallFloat::Fp4, 0x1), Some((0.5, false)));
        assert_eq!(small_float(AmdSmallFloat::Fp4, 0xf), Some((-6.0, true)));
        assert_eq!(small_float(AmdSmallFloat::Bf8, 0x3c), Some((1.0, true)));
        assert_eq!(small_float(AmdSmallFloat::Bf8, 0x7c), None);
        assert_eq!(small_float(AmdSmallFloat::Fp8, 0x7f), None);
        assert_eq!(small_float(AmdSmallFloat::Fp8, 0x3c), Some((1.5, true)));
    }

    #[test]
    fn scale_decoding() {
        assert_eq!(small_scale(SmallScale::E8m0, 0x7f), Some((1.0, true)));
        assert_eq!(small_scale(SmallScale::E8m0, 0xff), None);
        assert_eq!(small_scale(SmallScale::E5m3, 0x78), Some((1.0, true)));
        assert_eq!(small_scale(SmallScale::E5m3, 0x80), Some((2.0, true)));
        assert_eq!(small_scale(SmallScale::E5m3, 0x79), Some((1.125, false)));
        assert_eq!(small_scale(SmallScale::E5m3, 0xff), None);
        assert_eq!(small_scale(SmallScale::E4m3, 0x38), Some((1.0, true)));
        assert_eq!(small_scale(SmallScale::E4m3, 0x7f), None);
        assert_eq!(small_scale(SmallScale::E4m3, 0x80), None);
        assert_eq!(
            small_scale(SmallScale::E4m3, 0x07),
            Some((0.875 / 64.0, false))
        );
    }
}

/// Input element formats of the gfx1250-b0-wmma/4 forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HalfInput {
    F16,
    Bf16,
    Iu8,
    Small(AmdSmallFloat),
}

/// A gfx1250-b0-wmma/4 form: graph kind, dense or sparse, A/B formats, K
/// (expanded for SWMMAC) and the result format.
#[derive(Debug, Clone, Copy)]
struct HalfSparse {
    kind: AmdWmmaKind,
    sparse: bool,
    a: HalfInput,
    b: HalfInput,
    k: usize,
    output: AmdWmmaOutput,
}

impl HalfSparse {
    fn of(operation: MatrixOpcode) -> Option<Self> {
        use AmdSmallFloat::{Bf8, Fp8};
        use MatrixOpcode as M;
        let half = |input, output| HalfSparse {
            kind: AmdWmmaKind::Half16x16x32 { output },
            sparse: false,
            a: input,
            b: input,
            k: 32,
            output,
        };
        let f16out = |a, b, k128: bool| HalfSparse {
            kind: AmdWmmaKind::F16Out8 { a, b, k128 },
            sparse: false,
            a: HalfInput::Small(a),
            b: HalfInput::Small(b),
            k: if k128 { 128 } else { 64 },
            output: AmdWmmaOutput::F16,
        };
        let sparse = |input: AmdSwmmacInput, output| {
            let (a, b, k) = match input {
                AmdSwmmacInput::F16 => (HalfInput::F16, HalfInput::F16, 64),
                AmdSwmmacInput::Bf16 => (HalfInput::Bf16, HalfInput::Bf16, 64),
                AmdSwmmacInput::Iu8 => (HalfInput::Iu8, HalfInput::Iu8, 128),
                AmdSwmmacInput::Small { a, b } => (HalfInput::Small(a), HalfInput::Small(b), 128),
            };
            HalfSparse {
                kind: AmdWmmaKind::Swmmac { input, output },
                sparse: true,
                a,
                b,
                k,
                output,
            }
        };
        let small = |a, b| AmdSwmmacInput::Small { a, b };
        use AmdWmmaOutput as O;
        Some(match operation {
            M::VWmmaF1616x16x32F16 => half(HalfInput::F16, O::F16),
            M::VWmmaBf1616x16x32Bf16 => half(HalfInput::Bf16, O::Bf16),
            M::VWmmaBf16f3216x16x32Bf16 => half(HalfInput::Bf16, O::Bf16F32),
            M::VWmmaF1616x16x64Fp8Fp8 => f16out(Fp8, Fp8, false),
            M::VWmmaF1616x16x64Fp8Bf8 => f16out(Fp8, Bf8, false),
            M::VWmmaF1616x16x64Bf8Fp8 => f16out(Bf8, Fp8, false),
            M::VWmmaF1616x16x64Bf8Bf8 => f16out(Bf8, Bf8, false),
            M::VWmmaF1616x16x128Fp8Fp8 => f16out(Fp8, Fp8, true),
            M::VWmmaF1616x16x128Fp8Bf8 => f16out(Fp8, Bf8, true),
            M::VWmmaF1616x16x128Bf8Fp8 => f16out(Bf8, Fp8, true),
            M::VWmmaF1616x16x128Bf8Bf8 => f16out(Bf8, Bf8, true),
            M::VSwmmacF3216x16x64F16 => sparse(AmdSwmmacInput::F16, O::F32),
            M::VSwmmacF3216x16x64Bf16 => sparse(AmdSwmmacInput::Bf16, O::F32),
            M::VSwmmacF1616x16x64F16 => sparse(AmdSwmmacInput::F16, O::F16),
            M::VSwmmacBf1616x16x64Bf16 => sparse(AmdSwmmacInput::Bf16, O::Bf16),
            M::VSwmmacBf16f3216x16x64Bf16 => sparse(AmdSwmmacInput::Bf16, O::Bf16F32),
            M::VSwmmacF3216x16x128Fp8Fp8 => sparse(small(Fp8, Fp8), O::F32),
            M::VSwmmacF3216x16x128Fp8Bf8 => sparse(small(Fp8, Bf8), O::F32),
            M::VSwmmacF3216x16x128Bf8Fp8 => sparse(small(Bf8, Fp8), O::F32),
            M::VSwmmacF3216x16x128Bf8Bf8 => sparse(small(Bf8, Bf8), O::F32),
            M::VSwmmacF1616x16x128Fp8Fp8 => sparse(small(Fp8, Fp8), O::F16),
            M::VSwmmacF1616x16x128Fp8Bf8 => sparse(small(Fp8, Bf8), O::F16),
            M::VSwmmacF1616x16x128Bf8Fp8 => sparse(small(Bf8, Fp8), O::F16),
            M::VSwmmacF1616x16x128Bf8Bf8 => sparse(small(Bf8, Bf8), O::F16),
            M::VSwmmacI3216x16x128Iu8 => sparse(AmdSwmmacInput::Iu8, O::I32),
            _ => return None,
        })
    }
}

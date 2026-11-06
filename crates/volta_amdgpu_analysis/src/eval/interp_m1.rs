//! gfx1250 B0 m1 profile: GLOBAL prefetch, pre-issue global bounds
//! certificates against configured objects and guards, and raw-u32 outputs.
//!
//! Public CDNA5 ISA, GLOBAL_PREFETCH: no data return and no register result;
//! the address operands are retained for translation replay (X count). A scope
//! wider than the work-group with the default temporal hint is speculative:
//! a failed translation is dropped. Narrower scopes may fault.
use super::*;
use crate::config::OutputContract;

/// Exact stored word, or why a word of a raw-u32 output has no exact value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawWord {
    Exact(u32),
    Undefined(RawWordRefusal),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawWordRefusal {
    NotWritten,
    Symbolic,
    FloatingPointDerived,
    UndefinedBits,
    Opaque,
}

impl RawWordRefusal {
    pub const fn name(self) -> &'static str {
        match self {
            Self::NotWritten => "NOT_WRITTEN",
            Self::Symbolic => "SYMBOLIC",
            Self::FloatingPointDerived => "FLOATING_POINT_DERIVED",
            Self::UndefinedBits => "UNDEFINED_BITS",
            Self::Opaque => "OPAQUE",
        }
    }
}

/// Every word of each output array under the raw-u32 contract, in index order.
pub type RawOutputs = Vec<(String, Vec<RawWord>)>;

impl Interpreter<'_> {
    /// The B0 integer profiles (m1 and its m2 MODE superset) share every m1
    /// behaviour: strict S_MAX SCC, global bounds certificates, raw outputs.
    pub(super) fn m1_profile(&self) -> bool {
        self.config.native_profile.b0_integer()
    }

    /// m2 and its supersets (WMMA, multiwave): runtime MODE state.
    pub(super) fn m2_mode_profile(&self) -> bool {
        self.config.native_profile.b0_mode_state()
    }

    /// WMMA and multiwave profiles: exact-regime BF16 WMMA result bits.
    pub(super) fn wmma_profile(&self) -> bool {
        self.config.native_profile.b0_wmma()
    }

    pub(super) fn multiwave_profile(&self) -> bool {
        self.config.native_profile.b0_multiwave()
    }

    /// Every B0 profile with declared objects, guards and pre-issue global
    /// bounds certificates (m1, m2 and litmus).
    pub(super) fn bounded_profile(&self) -> bool {
        self.config.native_profile.b0_bounded()
    }

    /// Profile setup recorded in provenance before execution starts.
    pub(super) fn configure_native_profile(&mut self) -> Result<(), EvalError> {
        if !self.bounded_profile() {
            return Ok(());
        }
        if self.architecture != AmdArchitecture::Gfx1250 {
            return Err(EvalError::InvalidConfig(
                "the gfx1250 B0 m1 profile requires the gfx1250 target".into(),
            ));
        }
        let mut layout: Vec<_> = self
            .config
            .arrays
            .iter()
            .map(|array| AmdGlobalRegion {
                name: array.name.clone(),
                base: array.base,
                size: array.size_bytes(),
                guard: false,
            })
            .chain(
                self.module_globals
                    .iter()
                    .map(|(name, object)| AmdGlobalRegion {
                        name: name.clone(),
                        base: object.base,
                        size: object.size,
                        guard: false,
                    }),
            )
            .chain(
                self.config
                    .guard_regions
                    .iter()
                    .map(|guard| AmdGlobalRegion {
                        name: guard.name.clone(),
                        base: guard.base,
                        size: guard.bytes,
                        guard: true,
                    }),
            )
            .collect();
        layout.sort_by_key(|region| region.base);
        for pair in layout.windows(2) {
            if pair[0].base + pair[0].size > pair[1].base {
                return Err(EvalError::InvalidConfig(format!(
                    "global regions '{}' and '{}' overlap",
                    pair[0].name, pair[1].name
                )));
            }
        }
        if self.litmus_profile() {
            self.builder.assume(Assumption::Gfx1250B0LitmusProfile);
            for init in &self.config.initial_words {
                let array = self
                    .config
                    .array(&init.array)
                    .expect("validated initial-word array");
                for index in 0..array.len {
                    self.global.initialize(
                        array.base + 4 * index,
                        4,
                        Value::Integer(init.pattern.word(index)),
                    );
                }
            }
        } else {
            self.builder.assume(Assumption::Gfx1250B0M1Profile);
        }
        if self.m2_mode_profile() {
            self.builder.assume(Assumption::Gfx1250B0M2ModeProfile);
        }
        if self.wmma_profile() {
            self.builder.assume(Assumption::Gfx1250B0WmmaProfile);
        }
        if self.multiwave_profile() {
            self.builder.assume(Assumption::Gfx1250B0MultiwaveProfile);
        }
        if self.config.native_profile.b0_comms() {
            self.builder
                .assume(Assumption::Gfx1250B0WorkgroupCoherentProfile);
        }
        if self.litmus_barrier_profile() {
            self.builder
                .assume(Assumption::Gfx1250B0LitmusBarrierProfile);
        }
        self.builder.set_global_region_layout(layout);
        Ok(())
    }

    /// Entry MODE[3:0] and MODE[7:4] from the descriptor's 2-bit round and
    /// denorm controls (32-bit in the low pair, 16/64-bit in the high pair).
    /// A missing or out-of-range control leaves its whole field undefined.
    fn entry_mode_fields(&self) -> [Option<u8>; 2] {
        let field = |low: &str, high: &str| {
            let get = |key: &str| {
                self.program
                    .descriptor
                    .get(key)
                    .filter(|value| (0..=3).contains(value))
            };
            Some((get(low)? | get(high)? << 2) as u8)
        };
        [
            field("float_round_mode_32", "float_round_mode_16_64"),
            field("float_denorm_mode_32", "float_denorm_mode_16_64"),
        ]
    }

    /// Multiwave profile: `s_get_shader_cycles_u64` and the whole-register
    /// `s_getreg_b32` of HW_ID1 write fresh opaque symbols; no value is
    /// claimed. Returns false for every other instruction or profile.
    pub(super) fn evidence_instruction(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
        name: &str,
    ) -> Result<bool, EvalError> {
        if !self.multiwave_profile() {
            return Ok(false);
        }
        let operands = instruction.operands();
        let words = match name {
            "s_get_shader_cycles_u64" => match operands {
                [Operand::Register(r)] if r.count == 2 => 2,
                _ => return Err(self.unsupported(wave, instruction)),
            },
            // mwpred: the SE/XCC id message return (0x87). The profile requires
            // the next instruction to be `s_wait_kmcnt 0`, so no read can see
            // the destination before the return completes.
            "s_sendmsg_rtn_b32" if self.config.native_profile.b0_mwpred() => match operands {
                [Operand::Register(r), Operand::Integer(0x87)] if r.count == 1 => 1,
                _ => return Err(self.unsupported(wave, instruction)),
            },
            // Other GETREG forms keep the MODE (or default) handling.
            "s_getreg_b32" => match operands {
                [
                    Operand::Register(r),
                    Operand::HardwareRegister {
                        name,
                        offset: 0,
                        width: 32,
                    },
                ] if r.count == 1 && name == "HW_REG_WAVE_HW_ID1" => 1,
                _ => return Ok(false),
            },
            _ => return Ok(false),
        };
        if !instruction.modifiers().is_empty() {
            return Err(self.unsupported(wave, instruction));
        }
        let destination = wave.destination(&operands[0])?;
        for word in 0..words {
            let symbol = self.arena.symbol();
            self.opaque_symbols.insert(symbol);
            wave.put(destination, 0, word, Value::IntExpr(symbol))?;
        }
        Ok(true)
    }

    /// The selected MODE forms of the m2 profile. Returns false for every
    /// other instruction or profile, which keep their existing handling.
    pub(super) fn mode_instruction(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
        name: &str,
    ) -> Result<bool, EvalError> {
        if !self.m2_mode_profile() {
            return Ok(false);
        }
        let op = match name {
            "s_getreg_b32" => AmdModeOp::Getreg,
            "s_setreg_b32" => AmdModeOp::Setreg,
            "s_setreg_imm32_b32" => AmdModeOp::SetregImm32,
            "s_round_mode" => AmdModeOp::RoundMode,
            "s_denorm_mode" => AmdModeOp::DenormMode,
            _ => return Ok(false),
        };
        let operands = instruction.operands();
        let field = |operand: &Operand| match operand {
            Operand::HardwareRegister {
                name,
                offset: offset @ (0 | 4),
                width: 4,
            } if name == "HW_REG_WAVE_MODE" => Some(*offset),
            _ => None,
        };
        let nibble = |operand: &Operand| match operand {
            Operand::Integer(value @ 0..=15) => Some(*value as u8),
            _ => None,
        };
        let (offset, written) = match (op, operands) {
            (AmdModeOp::Getreg, [_, hwreg]) => (field(hwreg), None),
            (AmdModeOp::Setreg, [hwreg, source]) => {
                let Some(offset) = field(hwreg) else {
                    return Err(self.unsupported(wave, instruction));
                };
                (Some(offset), Some((wave.bits(source, 0)? & 0xf) as u8))
            }
            (AmdModeOp::SetregImm32, [hwreg, source]) => (field(hwreg), nibble(source)),
            (AmdModeOp::RoundMode, [source]) => (Some(0), nibble(source)),
            (AmdModeOp::DenormMode, [source]) => (Some(4), nibble(source)),
            _ => (None, None),
        };
        let Some(offset) = offset.filter(|_| op == AmdModeOp::Getreg || written.is_some()) else {
            return Err(self.unsupported(wave, instruction));
        };
        let index = usize::from(offset / 4);
        let entry = self.entry_mode_fields();
        let fields = self.mode_fields.entry(wave.id).or_insert(entry);
        let value = match written {
            Some(value) => {
                fields[index] = Some(value);
                value
            }
            None => {
                let value = fields[index]
                    .ok_or_else(|| wave.fail(None, "s_getreg_b32 reads an undefined MODE field"))?;
                wave.put_value(&operands[0], 0, Value::Integer(u32::from(value)))?;
                value
            }
        };
        self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::ModeAccess {
                op,
                offset,
                width: 4,
                value: u32::from(value),
            },
        );
        Ok(true)
    }

    /// True when every byte of `[address, address + bytes)` lies in one
    /// configured object and no byte lies in a guard.
    fn inside_object(&self, address: u64, bytes: u64) -> bool {
        let Some(end) = address.checked_add(bytes) else {
            return false;
        };
        let layout = self.builder.graph_so_far().global_region_layout();
        layout
            .iter()
            .any(|r| !r.guard && r.base <= address && end <= r.base + r.size)
            && !layout
                .iter()
                .any(|r| r.guard && address < r.base + r.size && r.base < end)
    }

    /// Records an attempted access and aborts before issue when any active
    /// lane leaves the configured objects. Only the m1 profile records a
    /// layout, so the default contract keeps its existing diagnostics.
    pub(super) fn precheck_global_footprint(
        &mut self,
        wave: &Wave,
        access: AmdGlobalAccess,
        bytes_per_lane: u64,
        lanes: Vec<(u8, u64)>,
    ) -> Result<(), EvalError> {
        if !self.bounded_profile() {
            return Ok(());
        }
        let speculative =
            matches!(access, AmdGlobalAccess::Prefetch(scope) if scope.speculative_prefetch());
        let layout = self.builder.graph_so_far().global_region_layout();
        let violates = |&(_, address): &(u8, u64)| {
            let end = address.checked_add(bytes_per_lane);
            let guard = layout.iter().any(|r| {
                r.guard && end.is_none_or(|end| address < r.base + r.size && r.base < end)
            });
            guard || !speculative && !self.inside_object(address, bytes_per_lane)
        };
        if !lanes.iter().any(violates) {
            return Ok(());
        }
        self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::GlobalAccessFault {
                access,
                bytes_per_lane: bytes_per_lane as u8,
                lanes,
            },
        );
        Err(EvalError::GlobalBounds {
            wave: wave.id,
            pc: wave.pc,
            what: GLOBAL_BOUNDS_ABORT_REASON,
        })
    }

    /// GLOBAL_PREFETCH_B8 in its scalar-base form. It issues an X-count lease
    /// over its address sources and a one-byte-per-lane footprint record; it
    /// writes no register and no memory and charges no load counter.
    pub(super) fn prefetch(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<(), EvalError> {
        if self.architecture != AmdArchitecture::Gfx1250 || self.xnack_replay_enabled {
            return Err(self.unsupported(wave, instruction));
        }
        if wave.exec == 0 {
            return Ok(());
        }
        let mut offset = 0i64;
        let mut scope = AmdCacheScope::Cu;
        for modifier in instruction.modifiers() {
            match (modifier.name.as_str(), modifier.value.as_deref()) {
                ("offset", Some(value)) => {
                    offset = parse_i64(value)
                        .filter(|value| (-8_388_608..=8_388_607).contains(value))
                        .ok_or_else(|| self.unsupported(wave, instruction))?;
                }
                ("scope", Some(value)) => {
                    scope = AmdCacheScope::parse(value)
                        .ok_or_else(|| self.unsupported(wave, instruction))?;
                }
                _ => return Err(self.unsupported(wave, instruction)),
            }
        }
        let Operation::Memory(memory) = instruction.operation() else {
            unreachable!("typed prefetch dispatch")
        };
        let MemoryAddress::Global { vector, scalar } = memory.address() else {
            unreachable!("typed prefetch address")
        };
        let lanes = wave
            .active()
            .map(|lane| {
                self.global_memory_address(wave, vector, Some(scalar), lane, offset, 1)
                    .map(|address| (lane, address))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !scope.speculative_prefetch() && !self.bounded_profile() {
            let configured = |address: u64| {
                self.config
                    .arrays
                    .iter()
                    .any(|a| a.base <= address && address < a.base + a.size_bytes())
                    || self
                        .module_globals
                        .values()
                        .any(|a| a.base <= address && address < a.base + a.size)
            };
            if lanes.iter().any(|&(_, address)| !configured(address)) {
                return Err(EvalError::Unsupported {
                    pc: wave.pc,
                    what: "non-speculative prefetch outside configured objects".into(),
                });
            }
        }
        self.precheck_global_footprint(wave, AmdGlobalAccess::Prefetch(scope), 1, lanes.clone())?;
        let (reads, _) = self.instruction_registers(instruction, wave);
        let issue = self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::MemoryIssue {
                class: AmdMemoryClass::Prefetch,
                counters: vec![(AmdCounter::X, 1)],
                destinations: Vec::new(),
                held_sources: reads,
            },
        );
        self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::PrefetchFootprint {
                issue,
                scope,
                lanes,
            },
        );
        Ok(())
    }

    /// Exact stored words of every output array. Untyped bits are exact here;
    /// floating, symbolic, partially defined and opaque words are refused.
    pub(super) fn extract_raw(&mut self) -> Option<RawOutputs> {
        if self.config.output_contract != OutputContract::RawU32 {
            return None;
        }
        let mut outputs = Vec::new();
        for array in self.config.arrays.iter().filter(|a| a.kind.is_output()) {
            let initialized = self
                .config
                .initial_words
                .iter()
                .any(|i| i.array == array.name);
            let words = (0..array.len)
                .map(|index| {
                    let address = array.base + 4 * index;
                    let written = (0..4).all(|byte| {
                        self.global
                            .written
                            .range(..=address + byte)
                            .next_back()
                            .is_some_and(|(&start, &width)| address + byte - start < width)
                    });
                    // Launcher-initialized words (litmus profile) read back
                    // their exact initial value when the kernel left them.
                    if !written && !initialized {
                        return RawWord::Undefined(RawWordRefusal::NotWritten);
                    }
                    match self.global.read(address, 4) {
                        Ok(value) => self.raw_word(value),
                        Err(_) => RawWord::Undefined(RawWordRefusal::Opaque),
                    }
                })
                .collect();
            outputs.push((array.name.clone(), words));
        }
        Some(outputs)
    }

    /// The exact words as integer expressions, for callers of the ordinary
    /// output interface; refused words are absent rather than coerced.
    pub(super) fn exact_raw_outputs(&mut self, raw: &RawOutputs) -> ExtractedOutputs {
        raw.iter()
            .map(|(name, words)| {
                let elements = words
                    .iter()
                    .enumerate()
                    .filter_map(|(index, word)| match word {
                        RawWord::Exact(bits) => {
                            Some((index as u64, self.arena.int(i64::from(*bits))))
                        }
                        RawWord::Undefined(_) => None,
                    })
                    .collect();
                (name.clone(), elements)
            })
            .collect()
    }

    fn raw_word(&self, value: Value) -> RawWord {
        use RawWordRefusal::*;
        match value {
            Value::Bits(bits) | Value::Integer(bits) => RawWord::Exact(bits),
            Value::SignedInteger(value) => RawWord::Exact(value as u32),
            Value::IntExpr(expr) if self.opaque_symbols.contains(&expr) => {
                RawWord::Undefined(Opaque)
            }
            Value::IntExpr(expr) => match self.arena.as_int_const(expr) {
                Some(value) if (-(1i64 << 31)..1i64 << 32).contains(&value) => {
                    RawWord::Exact(value as u32)
                }
                _ => RawWord::Undefined(Symbolic),
            },
            Value::PartialBits { .. } => RawWord::Undefined(UndefinedBits),
            Value::Real(_)
            | Value::Half(_)
            | Value::HighHalf(_)
            | Value::Halves(..)
            | Value::PartialHalves(..)
            | Value::WidePart {
                kind: WideKind::Float,
                ..
            } => RawWord::Undefined(FloatingPointDerived),
            Value::Input(_)
            | Value::NarrowInt { .. }
            | Value::Low16Integer(_)
            | Value::PredicateWord { .. }
            | Value::WidePart { .. }
            | Value::Byte(_)
            | Value::BytePair(..)
            | Value::PartialBytePair(..)
            | Value::Bytes(_) => RawWord::Undefined(Symbolic),
            Value::RelocationWord { .. }
            | Value::CodeWord { .. }
            | Value::SharedApertureHi
            | Value::PrivateApertureHi => RawWord::Undefined(Opaque),
        }
    }
}

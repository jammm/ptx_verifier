//! gfx1250 B0 litmus profile: untyped raw-buffer loads and stores with the
//! documented per-DWORD range check, and scalar-base async loads to LDS.
//!
//! Buffer resource fields, the linear raw-buffer address and the range rule
//! are the documented contract: DWORD `c` of a lane is out of range when
//! `offset + 4c + 4 > num_records` or, with OOB_SELECT, `> stride`. An
//! out-of-range load DWORD reads zero and an out-of-range store DWORD writes
//! nothing. Before issue, every lane's unclamped footprint (all DWORDs, as if
//! in range) must lie in one configured object, so no range-check outcome can
//! reach another object. Forms the contract does not settle are refused.
use super::*;
use crate::config::NativeProfile;

/// Async LDS displacements at or above this are outside the profile and
/// refused.
const ASYNC_LDS_MAX_OFFSET: i64 = 0x1_0000;

impl Interpreter<'_> {
    pub(super) fn litmus_profile(&self) -> bool {
        self.config.native_profile.b0_litmus()
    }

    /// litmus/2: the named-barrier JOIN read-back additions.
    pub(super) fn litmus_barrier_profile(&self) -> bool {
        self.config.native_profile == NativeProfile::Gfx1250B0Litmus2
    }

    fn litmus_refusal(&self, wave: &Wave, what: &str) -> EvalError {
        EvalError::Unsupported {
            pc: wave.pc,
            what: format!("litmus profile: {what}"),
        }
    }

    /// `buffer_load_b32..b128` / `buffer_store_b32..b128` with a VGPR offset
    /// (`offen`) or no VGPR address (`off`), a null SGPR offset and default
    /// policy, under the litmus profile.
    pub(super) fn litmus_buffer(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<(), EvalError> {
        let Operation::Memory(memory) = instruction.operation() else {
            unreachable!("typed buffer dispatch")
        };
        let TransferShape::Registers {
            registers_per_part: dwords,
            bytes_per_register: 4,
            parts: 1,
        } = memory.shape
        else {
            return Err(self.unsupported(wave, instruction));
        };
        let store = memory.kind == MemoryKind::BufferStore;
        if wave.exec == 0 {
            return Ok(());
        }
        let o = memory.operands();
        let mut offen = false;
        let mut immediate = 0u64;
        for modifier in instruction.modifiers() {
            match (modifier.name.as_str(), modifier.value.as_deref()) {
                ("offen", None) => offen = true,
                ("offset", Some(value)) => {
                    immediate = parse_u64(value)
                        .filter(|value| *value <= 0x7f_ffff)
                        .ok_or_else(|| self.unsupported(wave, instruction))?;
                }
                _ => return Err(self.litmus_refusal(wave, "buffer modifier outside the profile")),
            }
        }
        let vaddr = match &o[1] {
            Operand::Off => None,
            operand @ Operand::Register(_) => Some(operand),
            _ => return Err(self.unsupported(wave, instruction)),
        };
        if offen != vaddr.is_some() {
            return Err(self.litmus_refusal(wave, "buffer VGPR address requires OFFEN"));
        }
        // Descriptor: base 56:0, num_records 101:57, reserved 107:102,
        // stride 121:108, stride scale 123:122, swizzle 124, OOB_SELECT 125,
        // type 127:126.
        let descriptor = wave.destination(&o[2])?;
        let mut resource = 0u128;
        for component in 0..4 {
            let word = wave
                .reg(descriptor, 0, component)?
                .bits()
                .map_err(|error| wave.fail(None, error))?;
            resource |= u128::from(word) << (u32::from(component) * 32);
        }
        let field = |low: u32, bits: u32| ((resource >> low) & ((1u128 << bits) - 1)) as u64;
        let (base, num_records) = (field(0, 57), field(57, 45));
        let (reserved, stride, scale) = (field(102, 6), field(108, 14), field(122, 2));
        let (swizzle, oob_select, kind) = (field(124, 1), field(125, 1) == 1, field(126, 2));
        let refusal = if reserved != 0 || kind != 0 {
            Some("buffer resource reserved or type bits are nonzero")
        } else if swizzle != 0 {
            Some("swizzled buffer addressing is outside the profile")
        } else if scale != 0 {
            Some("buffer stride scale is outside the profile")
        } else if num_records == 0 {
            Some("num_records 0 is an unbound resource (UNBOUND_RESOURCE)")
        } else if num_records == (1u64 << 45) - 1 {
            Some("num_records all ones disables range checking")
        } else {
            None
        };
        if let Some(what) = refusal {
            return Err(self.litmus_refusal(wave, what));
        }
        let mut lanes = Vec::new();
        for lane in wave.active() {
            let vector = match vaddr {
                Some(operand) => u64::from(wave.bits(operand, lane)?),
                None => 0,
            };
            let offset = (vector + immediate) & ((1u64 << 45) - 1);
            if !offset.is_multiple_of(4) {
                return Err(self.litmus_refusal(
                    wave,
                    "buffer offset is not DWORD aligned (alignment mode is not a launch premise)",
                ));
            }
            let address = base
                .checked_add(offset)
                .filter(|address| {
                    address
                        .checked_add(4 * u64::from(dwords))
                        .is_some_and(|end| end <= 1u64 << 57)
                })
                .ok_or_else(|| {
                    self.litmus_refusal(wave, "buffer footprint escapes the 57-bit address domain")
                })?;
            lanes.push((lane, offset, address));
        }
        let check = AmdBufferRangeCheck {
            base,
            num_records,
            stride,
            oob_select,
            dwords,
            lanes: Vec::new(),
        };
        let check = AmdBufferRangeCheck {
            lanes: lanes
                .iter()
                .map(|&(lane, offset, _)| AmdBufferLane {
                    lane,
                    offset,
                    in_range: check.expected_in_range(offset),
                })
                .collect(),
            ..check
        };
        // Envelope: the unclamped footprint of every lane, before issue.
        let access = if store {
            AmdGlobalAccess::Store
        } else {
            AmdGlobalAccess::Load
        };
        self.precheck_global_footprint(
            wave,
            access,
            4 * u64::from(dwords),
            lanes
                .iter()
                .map(|&(lane, _, address)| (lane, address))
                .collect(),
        )?;
        let (reads, writes) = self.instruction_registers(instruction, wave);
        let issue = self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::MemoryIssue {
                class: if store {
                    AmdMemoryClass::VectorStore
                } else {
                    AmdMemoryClass::VectorLoad
                },
                counters: vec![
                    (
                        if store {
                            AmdCounter::Store
                        } else {
                            AmdCounter::Load
                        },
                        1,
                    ),
                    (AmdCounter::X, 1),
                ],
                destinations: if store { Vec::new() } else { writes },
                held_sources: reads,
            },
        );
        let data = wave.destination(&o[0])?;
        for (&(lane, _, address), result) in lanes.iter().zip(&check.lanes) {
            for component in 0..u16::from(dwords) {
                let addr = address + 4 * u64::from(component);
                if result.in_range & (1 << component) == 0 {
                    if !store {
                        wave.put(data, lane, component, Value::Integer(0))?;
                    }
                    continue;
                }
                self.validate_address(wave, lane, false, addr, 4, store)?;
                let value = if store {
                    let value = wave.reg(data, lane, component)?;
                    self.global_write(addr, 4, value)
                        .map_err(|e| wave.fail(Some(lane), e))?;
                    Some(value)
                } else {
                    let value = self
                        .global_read(addr, 4)
                        .map_err(|e| wave.fail(Some(lane), format!("global {addr:#x}: {e}")))?;
                    wave.put(data, lane, component, value)?;
                    None
                };
                let stored = value.map(|value| self.litmus_stored(value));
                let thread = ThreadIx(wave.id * wave.width + u32::from(lane));
                let effect = self.builder.thread_access(
                    thread,
                    ProgramPoint(wave.pc as u32),
                    MemAccess {
                        space: Space::Global,
                        mode: if store {
                            AccessMode::Write
                        } else {
                            AccessMode::Read
                        },
                        ordering: OrderingKind::Ordinary,
                        proxy: Proxy::Generic,
                        footprint: Footprint::Bytes { addr, width: 4 },
                        role: OperandRole::Plain,
                        attributed: thread,
                        value: stored,
                    },
                );
                self.builder
                    .amd_memory_effect_scoped(issue, effect, AmdMemoryScope::Unspecified);
            }
        }
        self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::BufferRangeCheck { issue, check },
        );
        Ok(())
    }

    fn litmus_stored(&mut self, value: Value) -> ValueId {
        self.builder.value(match value {
            Value::Input(e) | Value::Real(e) | Value::IntExpr(e) => StoredValue::Scalar(e),
            Value::Bits(bits) | Value::Integer(bits) => {
                StoredValue::Scalar(self.arena.int(bits as i64))
            }
            Value::SignedInteger(value) => StoredValue::Scalar(self.arena.int(value as i64)),
            Value::Halves(lo, hi) | Value::BytePair(lo, hi) => StoredValue::Pair(lo, hi),
            Value::Bytes(bytes) => StoredValue::Quad(bytes),
            _ => StoredValue::Opaque,
        })
    }

    /// `global_load_async_to_lds_b32/b64/b128 vLDS, vOFF, s[base:base+1]`
    /// (scalar-base GVS form, unscaled, default scope): global address
    /// `SADDR + VADDR[31:0] + IOFFSET`, LDS address `VDST + IOFFSET`.
    pub(super) fn litmus_async_load_to_lds(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<(), EvalError> {
        let Operation::Memory(memory) = instruction.operation() else {
            unreachable!("typed async memory dispatch")
        };
        let TransferShape::PerLaneBytes(bytes) = memory.shape else {
            unreachable!("async transfers carry a per-lane byte count")
        };
        let o = memory.operands();
        let Some(saddr @ Operand::Register(_)) = o.get(2) else {
            return Err(self.litmus_refusal(
                wave,
                "async load to LDS requires the scalar-base (GVS) form",
            ));
        };
        if wave.exec == 0 {
            return Ok(());
        }
        let offset = match instruction.modifiers().as_slice() {
            [] => 0,
            [modifier] if modifier.name == "offset" => modifier
                .value
                .as_deref()
                .and_then(parse_i64)
                .ok_or_else(|| self.unsupported(wave, instruction))?,
            _ => return Err(self.litmus_refusal(wave, "async modifier outside the profile")),
        };
        if !(0..ASYNC_LDS_MAX_OFFSET).contains(&offset) {
            return Err(self.litmus_refusal(
                wave,
                "async LDS displacement at or above 64 KiB or negative (ASYNC_LDS_OFFSET_OUT_OF_PROFILE)",
            ));
        }
        let width = u64::from(bytes);
        let lds_extent = self.lds_size;
        let mut transfers = Vec::new();
        for lane in wave.active() {
            let global = wave
                .wide(saddr, lane)?
                .checked_add(u64::from(wave.bits(&o[1], lane)?))
                .and_then(|address| address.checked_add(offset as u64))
                .ok_or_else(|| self.litmus_refusal(wave, "async global address overflow"))?;
            let shared = u64::from(wave.bits(&o[0], lane)?) + offset as u64;
            if !global.is_multiple_of(width) || !shared.is_multiple_of(width) {
                return Err(self.litmus_refusal(wave, "async footprints must be naturally aligned"));
            }
            if shared + width > lds_extent {
                return Err(self.litmus_refusal(
                    wave,
                    "async LDS footprint outside the allocation (dropped writes are not modeled)",
                ));
            }
            transfers.push((lane, global, shared));
        }
        self.precheck_global_footprint(
            wave,
            AmdGlobalAccess::Load,
            width,
            transfers
                .iter()
                .map(|&(lane, global, _)| (lane, global))
                .collect(),
        )?;
        let mut staged = Vec::new();
        for &(lane, global, shared) in &transfers {
            self.validate_memory_range(wave, lane, false, global, width, false)?;
            self.validate_memory_range(wave, lane, true, shared, width, true)?;
            let mut words = Vec::new();
            for component in 0..width / 4 {
                words.push(
                    self.global_read(global + 4 * component, 4)
                        .map_err(|error| wave.fail(Some(lane), error))?,
                );
            }
            staged.push((lane, global, shared, words));
        }
        let (_, destinations) = self.instruction_registers(instruction, wave);
        debug_assert!(destinations.is_empty());
        let issue = self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::MemoryIssue {
                class: AmdMemoryClass::AsyncLoadToLds,
                counters: vec![(AmdCounter::Async, 1)],
                destinations,
                held_sources: Vec::new(),
            },
        );
        for (lane, global, shared, words) in staged {
            let thread = ThreadIx(wave.id * wave.width + u32::from(lane));
            for (component, value) in words.into_iter().enumerate() {
                let displacement = component as u64 * 4;
                self.lds
                    .write(shared + displacement, 4, value)
                    .map_err(|error| wave.fail(Some(lane), error))?;
                let stored = self.litmus_stored(value);
                for (space, mode, addr, value) in [
                    (Space::Global, AccessMode::Read, global + displacement, None),
                    (
                        Space::Shared,
                        AccessMode::Write,
                        shared + displacement,
                        Some(stored),
                    ),
                ] {
                    let effect = self.builder.thread_access(
                        thread,
                        ProgramPoint(wave.pc as u32),
                        MemAccess {
                            space,
                            mode,
                            ordering: OrderingKind::Ordinary,
                            proxy: Proxy::Generic,
                            footprint: Footprint::Bytes { addr, width: 4 },
                            role: OperandRole::Plain,
                            attributed: thread,
                            value,
                        },
                    );
                    self.builder.amd_memory_effect(issue, effect);
                }
            }
        }
        Ok(())
    }
}

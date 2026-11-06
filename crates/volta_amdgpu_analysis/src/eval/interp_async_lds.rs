//! Per-lane GV async copies, distinct from uniform tensor-descriptor transfers.
//!
//! Public CDNA5 ISA, sections 10.8 and 15.18: the byte displacement applies to
//! both addresses, VDST is an address input, and ASYNCCNT tracks LDS completion.
//! LLVM 5fd50ec8e07abb94e203933017e8d6e40fb51824, FLATInstructions.td,
//! SIInstrInfo.cpp and AMDGPUHWEvents.cpp: IsAsync disables VM_CNT, so this
//! opcode uses ASYNC without the ordinary VMEM Load/X completion tokens.
use super::*;

impl Interpreter<'_> {
    pub(super) fn async_load_to_lds(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<(), EvalError> {
        // Outside the litmus profile only the compiler's B128 GV form runs.
        if self.architecture != AmdArchitecture::Gfx1250
            || self.xnack_replay_enabled
            || instruction.mnemonic() != "global_load_async_to_lds_b128"
            || !matches!(instruction.operands().get(2), Some(Operand::Off))
        {
            return Err(self.unsupported(wave, instruction));
        }
        if wave.exec == 0 {
            return Ok(());
        }
        let offset = match instruction.modifiers().as_slice() {
            [] => 0,
            [modifier] if modifier.name == "offset" => modifier
                .value
                .as_deref()
                .and_then(parse_i64)
                .filter(|value| (-8_388_608..=8_388_607).contains(value))
                .ok_or_else(|| self.unsupported(wave, instruction))?,
            _ => return Err(self.unsupported(wave, instruction)),
        };
        let Operation::Memory(memory) = instruction.operation() else {
            unreachable!("typed async memory dispatch")
        };
        let MemoryAddress::GlobalToLds { global, lds } = memory.address() else {
            unreachable!("typed async addresses")
        };
        // Stage complete footprints and payloads before any LDS write or issue.
        // Refuse wrapped/unaligned LDS ranges rather than guessing suppression
        // or alignment-mask behavior outside this bounded profile.
        let mut transfers = Vec::new();
        for lane in wave.active() {
            let global = wave
                .wide(global, lane)?
                .checked_add_signed(offset)
                .ok_or_else(|| self.unsupported(wave, instruction))?;
            let shared = i64::from(wave.bits(lds, lane)?)
                .checked_add(offset)
                .and_then(|address| u32::try_from(address).ok())
                .map(u64::from)
                .ok_or_else(|| self.unsupported(wave, instruction))?;
            if !global.is_multiple_of(16) || !shared.is_multiple_of(16) {
                return Err(EvalError::Unsupported {
                    pc: wave.pc,
                    what: "selected async B128 copy requires 16-byte aligned global and LDS footprints".into(),
                });
            }
            self.validate_memory_range(wave, lane, false, global, 16, false)?;
            self.validate_memory_range(wave, lane, true, shared, 16, true)?;
            if !self.config.shared_allocations.is_empty()
                && !self.config.shared_allocations.iter().any(|allocation| {
                    allocation.offset <= shared
                        && shared + 16 <= allocation.offset + allocation.size
                })
            {
                return Err(wave.fail(
                    Some(lane),
                    "async footprint crosses shared allocation boundaries",
                ));
            }
            let mut words = [Value::Integer(0); 4];
            for (component, value) in words.iter_mut().enumerate() {
                *value = self
                    .global_read(global + component as u64 * 4, 4)
                    .map_err(|error| wave.fail(Some(lane), error))?;
            }
            transfers.push((lane, global, shared, words));
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
                // Ordinary instruction reads still check address readiness.
                // The model snapshots sources at issue. Default scheduling
                // requires no software completion-held source leases; expert
                // mode changes remain outside the native profile.
                held_sources: Vec::new(),
            },
        );
        for (lane, global, shared, words) in transfers {
            let thread = ThreadIx(wave.id * wave.width + u32::from(lane));
            for (component, value) in words.into_iter().enumerate() {
                let displacement = component as u64 * 4;
                self.lds
                    .write(shared + displacement, 4, value)
                    .map_err(|error| wave.fail(Some(lane), error))?;
                let stored = self.builder.value(match value {
                    Value::Input(e) | Value::Real(e) | Value::IntExpr(e) => StoredValue::Scalar(e),
                    Value::Bits(bits) | Value::Integer(bits) => {
                        StoredValue::Scalar(self.arena.int(bits as i64))
                    }
                    Value::Halves(lo, hi) | Value::BytePair(lo, hi) => StoredValue::Pair(lo, hi),
                    Value::Bytes(bytes) => StoredValue::Quad(bytes),
                    _ => StoredValue::Opaque,
                });
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

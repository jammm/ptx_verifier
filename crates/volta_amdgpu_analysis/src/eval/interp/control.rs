//! Native AMD control execution.
use super::*;

impl Interpreter<'_> {
    /// Snapshot both source words before writing either saved-PC word. Physical
    /// register/EXEC/SCC/M0/counter state is shared; a call supplies no wait.
    pub(super) fn call(&self, wave: &mut Wave, call: &CallInstruction) -> Result<(), EvalError> {
        if self.architecture != AmdArchitecture::Gfx950 {
            return Err(wave.fail(None, "device calls require gfx950"));
        }
        let low = wave.reg(call.target, 0, 0)?;
        let high = wave.reg(call.target, 0, 1)?;
        let Value::CodeWord {
            target,
            origin,
            returning: false,
            high: false,
        } = low
        else {
            return Err(wave.fail(None, "call target is not an approved opaque code address"));
        };
        if high
            != (Value::CodeWord {
                target,
                origin,
                returning: false,
                high: true,
            })
            || self.program.call_targets.get(&wave.pc) != Some(&(target as usize))
            || !self
                .program
                .code_address_sequences
                .get(&(origin as usize))
                .is_some_and(|sequence| {
                    sequence.target_pc == target as usize && sequence.register_pair == call.target
                })
            || !self
                .program
                .functions
                .iter()
                .skip(1)
                .any(|function| function.start == target as usize)
            || wave.calls.len() >= self.program.functions.len()
        {
            return Err(wave.fail(None, "invalid code-address identity, target or call depth"));
        }
        let continuation = wave
            .pc
            .checked_add(1)
            .ok_or_else(|| wave.fail(None, "return PC overflow"))?;
        let frame = CallFrame {
            call_pc: wave.pc,
            target: target as usize,
            continuation,
        };
        let saved = |high| Value::CodeWord {
            target: continuation as u32,
            origin: wave.pc as u32,
            returning: true,
            high,
        };
        let lo = saved(false);
        let hi = saved(true);
        wave.put(call.return_address, 0, 0, lo)?;
        wave.put(call.return_address, 0, 1, hi)?;
        wave.calls.push(frame);
        wave.pc = target as usize;
        Ok(())
    }
    pub(super) fn return_from_call(
        &self,
        wave: &mut Wave,
        ret: &ReturnInstruction,
    ) -> Result<(), EvalError> {
        let frame = wave
            .calls
            .last()
            .ok_or_else(|| wave.fail(None, "return without a caller"))?;
        let saved = |high| Value::CodeWord {
            target: frame.continuation as u32,
            origin: frame.call_pc as u32,
            returning: true,
            high,
        };
        if wave.reg(ret.target, 0, 0)? != saved(false) || wave.reg(ret.target, 0, 1)? != saved(true)
        {
            return Err(wave.fail(
                None,
                "return address does not match the active caller continuation",
            ));
        }
        let continuation = frame.continuation;
        wave.calls.pop();
        wave.pc = continuation;
        Ok(())
    }
    pub(super) fn code_address_scalar(
        &self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<bool, EvalError> {
        let sequence = (0..=2).find_map(|stage| {
            wave.pc.checked_sub(stage).and_then(|anchor| {
                self.program
                    .code_address_sequences
                    .get(&anchor)
                    .map(|sequence| (stage, sequence))
            })
        });
        let Some((stage, sequence)) = sequence else {
            return Ok(false);
        };
        if instruction.mnemonic() != GlobalAddressForm::SplitRel32.mnemonics()[stage] {
            return Err(wave.fail(None, "invalid approved code-address stage"));
        }
        let anchor = sequence.anchor_pc as u32;
        let pair = sequence.register_pair;
        let anchor_word = |high| Value::RelocationWord { anchor, high };
        let code_word = |high| Value::CodeWord {
            target: sequence.target_pc as u32,
            origin: anchor,
            returning: false,
            high,
        };
        match stage {
            0 => {
                wave.put(pair, 0, 0, anchor_word(false))?;
                wave.put(pair, 0, 1, anchor_word(true))?;
            }
            1 => {
                if wave.reg(pair, 0, 0)? != anchor_word(false)
                    || wave.reg(pair, 0, 1)? != anchor_word(true)
                {
                    return Err(wave.fail(None, "code-address low stage lost its opaque anchor"));
                }
                wave.put(pair, 0, 0, code_word(false))?;
                wave.clear_scc();
            }
            2 => {
                if wave.reg(pair, 0, 0)? != code_word(false)
                    || wave.reg(pair, 0, 1)? != anchor_word(true)
                {
                    return Err(wave.fail(None, "code-address high stage lost its opaque anchor"));
                }
                wave.put(pair, 0, 1, code_word(true))?;
                wave.clear_scc();
            }
            _ => unreachable!(),
        }
        Ok(true)
    }
    pub(super) fn module_address_scalar(
        &self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<bool, EvalError> {
        let sequence = (0..=2).find_map(|stage| {
            wave.pc.checked_sub(stage).and_then(|anchor| {
                self.program
                    .global_address_sequences
                    .get(&anchor)
                    .filter(|sequence| stage < sequence.form.mnemonics().len())
                    .map(|sequence| (stage, sequence))
            })
        });
        let Some((stage, sequence)) = sequence else {
            return Ok(false);
        };
        if sequence.anchor_pc + stage != wave.pc
            || instruction.mnemonic() != sequence.form.mnemonics()[stage]
        {
            return Err(wave.fail(None, "invalid approved module-address sequence stage"));
        }
        let anchor = u32::try_from(sequence.anchor_pc)
            .map_err(|_| wave.fail(None, "module-address anchor exceeds program-point range"))?;
        let pair = sequence.register_pair;
        let address = self
            .module_globals
            .get(&sequence.symbol)
            .ok_or_else(|| wave.fail(None, "approved relocation has no allocated module object"))?
            .base;
        match stage {
            0 => {
                wave.put(
                    pair,
                    0,
                    0,
                    Value::RelocationWord {
                        anchor,
                        high: false,
                    },
                )?;
                wave.put(pair, 0, 1, Value::RelocationWord { anchor, high: true })?;
            }
            1 => {
                if wave.reg(pair, 0, 0)?
                    != (Value::RelocationWord {
                        anchor,
                        high: false,
                    })
                    || wave.reg(pair, 0, 1)? != (Value::RelocationWord { anchor, high: true })
                {
                    return Err(wave.fail(None, "PC-relative low stage lost its opaque anchor"));
                }
                wave.put(pair, 0, 0, Value::Bits(address as u32))?;
                if sequence.form == crate::ir::GlobalAddressForm::Rel64 {
                    wave.put(pair, 0, 1, Value::Bits((address >> 32) as u32))?;
                } else {
                    wave.clear_scc();
                }
            }
            2 => {
                if wave.reg(pair, 0, 0)?.bits().ok() != Some(address as u32)
                    || wave.reg(pair, 0, 1)? != (Value::RelocationWord { anchor, high: true })
                {
                    return Err(wave.fail(None, "PC-relative high stage lost its approved anchor"));
                }
                wave.put(pair, 0, 1, Value::Bits((address >> 32) as u32))?;
                // The chosen abstract storage address determines the pointer,
                // not the actual code address or linker carry/overflow bits.
                wave.clear_scc();
            }
            _ => unreachable!(),
        }
        Ok(true)
    }
    pub(super) fn wait(&mut self, wave: &Wave, instruction: &Instruction) -> Result<(), EvalError> {
        let Operation::Wait(wait) = instruction.operation() else {
            unreachable!("typed wait dispatch")
        };
        let event = match wait {
            WaitInstruction::Cdna {
                vmcnt,
                lgkmcnt,
                expcnt,
            } if self.architecture != AmdArchitecture::Gfx1250 => AmdEventKind::Wait {
                vmcnt: *vmcnt,
                lgkmcnt: *lgkmcnt,
                expcnt: *expcnt,
            },
            WaitInstruction::Combined { data, encoded }
                if self.architecture == AmdArchitecture::Gfx1250 =>
            {
                if *encoded < 0 || *encoded > 0x3f3f {
                    return Err(self.unsupported(wave, instruction));
                }
                AmdEventKind::WaitCounters {
                    thresholds: vec![
                        (*data, ((*encoded >> 8) & 63) as u16),
                        (AmdCounter::Ds, (*encoded & 63) as u16),
                    ],
                }
            }
            WaitInstruction::Alu(wait) if self.architecture == AmdArchitecture::Gfx1250 => {
                AmdEventKind::AluWait { wait: *wait }
            }
            // ALU dependency counters are hardware-interlocked under SCHED_MODE
            // 0, so the memory counters carry the whole S_WAIT_IDLE effect.
            WaitInstruction::Idle if self.architecture == AmdArchitecture::Gfx1250 => {
                AmdEventKind::WaitCounters {
                    thresholds: [
                        AmdCounter::Load,
                        AmdCounter::Store,
                        AmdCounter::Ds,
                        AmdCounter::Km,
                        AmdCounter::X,
                        AmdCounter::Async,
                        AmdCounter::Tensor,
                    ]
                    .into_iter()
                    .map(|counter| (counter, 0))
                    .collect(),
                }
            }
            WaitInstruction::Counter { counter, threshold }
                if self.architecture == AmdArchitecture::Gfx1250 =>
            {
                let value =
                    u16::try_from(*threshold).map_err(|_| self.unsupported(wave, instruction))?;
                AmdEventKind::WaitCounters {
                    thresholds: vec![(*counter, value)],
                }
            }
            _ => return Err(self.unsupported(wave, instruction)),
        };
        if let AmdEventKind::Wait { vmcnt, lgkmcnt, .. } = &event {
            self.private
                .complete(wave.id, *vmcnt == Some(0), *lgkmcnt == Some(0));
        }
        self.builder
            .amd_event(wave.id, ProgramPoint(wave.pc as u32), event);
        Ok(())
    }
}

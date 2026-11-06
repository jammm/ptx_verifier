//! Selected GFX1250 shader named barriers. Execution state only; the downstream
//! checker independently reconstructs counts, object lifetimes and publication.
use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) struct Joined {
    pub object: AmdNamedBarrier,
    pub phase: u32,
}
#[derive(Debug)]
struct Object {
    incarnation: u32,
    expected: u16,
    arrived: u16,
    phase: u32,
    last_arrive: BTreeMap<u32, u32>,
}
#[derive(Debug, Default)]
pub(super) struct State {
    objects: BTreeMap<u8, Object>,
}
impl State {
    pub fn counts(&self, joined: Joined) -> (u16, u16) {
        self.objects
            .get(&joined.object.id)
            .map_or((0, 0), |o| (o.expected, o.arrived))
    }
}

impl Interpreter<'_> {
    fn named_id(&self, wave: &Wave, instruction: &Instruction) -> Result<u8, EvalError> {
        let value = match instruction.operands() {
            [Operand::Integer(id)] if (0..=16).contains(id) => *id as u32,
            [Operand::Register(r)] if r.special == Some(SpecialRegister::M0) => {
                if !wave.m0_defined {
                    return Err(wave.fail(None, "named barrier uses uninitialized M0"));
                }
                if wave.m0 & 0xffff > 16 {
                    return Err(self.unsupported(wave, instruction));
                }
                if instruction.mnemonic() == "s_barrier_signal"
                    && wave.m0 & 0xffff != 0
                    && wave.m0 & 0xff80_0000 != 0
                {
                    return Err(EvalError::Unsupported { pc: wave.pc, what: "named signal has nonzero reserved M0 bits above the 7-bit expected count".into() });
                }
                wave.m0 & 0xffff
            }
            _ => return Err(self.unsupported(wave, instruction)),
        };
        let allocated = self.program.descriptor.value("named_barrier_count", 0) * 4;
        if instruction.mnemonic() != "s_barrier_wait" && value as u64 > allocated {
            return Err(wave.fail(None, "named barrier ID exceeds descriptor allocation"));
        }
        Ok(value as u8)
    }
    pub(super) fn named_barrier(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<bool, EvalError> {
        let name = instruction.mnemonic().as_str();
        let id = if name == "s_barrier_leave" {
            wave.named_join.map_or(0, |j| j.object.id)
        } else {
            self.named_id(wave, instruction)?
        };
        let object = AmdNamedBarrier {
            id,
            incarnation: self.named.objects.get(&id).map_or(0, |o| o.incarnation),
        };
        let phase = self.named.objects.get(&id).map_or(0, |o| o.phase);
        let pc = ProgramPoint(wave.pc as u32);
        match name {
            "s_barrier_init" => {
                if id == 0 {
                    self.builder.amd_event(
                        wave.id,
                        pc,
                        AmdEventKind::NamedBarrierInit {
                            object,
                            expected: 0,
                        },
                    );
                    return Ok(false);
                }
                if !wave.m0_defined {
                    return Err(wave.fail(None, "named init requires explicit M0 expected count"));
                }
                let expected = (wave.m0 >> 16) as u16;
                let object = AmdNamedBarrier {
                    id,
                    incarnation: match self.named.objects.get(&id) {
                        None => 0,
                        Some(previous) => previous
                            .incarnation
                            .checked_add(1)
                            .ok_or_else(|| wave.fail(None, "named barrier incarnation overflow"))?,
                    },
                };
                self.builder.amd_event(
                    wave.id,
                    pc,
                    AmdEventKind::NamedBarrierInit { object, expected },
                );
                if expected == 0 {
                    return Err(wave.fail(None, "named barrier expected count must be positive"));
                }
                if self
                    .waves
                    .iter()
                    .any(|other| other.named_join.is_some_and(|j| j.object.id == id))
                    || wave.named_join.is_some_and(|j| j.object.id == id)
                {
                    return Err(EvalError::Unsupported { pc: wave.pc, what: "reinitialization with outstanding named joins is outside the selected profile".into() });
                }
                self.named.objects.insert(
                    id,
                    Object {
                        incarnation: object.incarnation,
                        expected,
                        arrived: 0,
                        phase: 0,
                        last_arrive: BTreeMap::new(),
                    },
                );
            }
            "s_barrier_join" => {
                if (self.litmus_barrier_profile() || self.config.native_profile.b0_comms())
                    && self.unconsumed_completion(wave)
                {
                    return Err(EvalError::Unsupported { pc: wave.pc, what: "litmus profile: JOIN while the joined named barrier has a completion this wave has not waited on (completion report order not documented)".into() });
                }
                self.builder.amd_event(
                    wave.id,
                    pc,
                    AmdEventKind::NamedBarrierJoin { object, phase },
                );
                if id != 0 && !self.named.objects.contains_key(&id) {
                    return Err(wave.fail(None, "join of uninitialized named barrier"));
                }
                wave.named_join = Some(Joined { object, phase });
            }
            "s_barrier_signal" => {
                let expected = match instruction.operands() {
                    [Operand::Register(r)] if r.special == Some(SpecialRegister::M0) && id != 0 => {
                        ((wave.m0 >> 16) & 0x7f) as u16
                    }
                    _ => 0,
                };
                self.builder.amd_event(
                    wave.id,
                    pc,
                    if expected != 0 {
                        AmdEventKind::NamedBarrierSignalWithCount {
                            object,
                            phase,
                            exec_mask: wave.exec,
                            expected,
                        }
                    } else {
                        AmdEventKind::NamedBarrierSignal {
                            object,
                            phase,
                            exec_mask: wave.exec,
                        }
                    },
                );
                if id == 0 {
                    return Ok(false);
                }
                let state = self
                    .named
                    .objects
                    .get_mut(&id)
                    .ok_or_else(|| wave.fail(None, "signal of uninitialized named barrier"))?;
                if expected != 0 {
                    if expected < state.arrived {
                        return Err(wave.fail(
                            None,
                            "named signal expected count is less than the current arrival count",
                        ));
                    }
                    if expected == state.arrived {
                        return Err(EvalError::Unsupported { pc: wave.pc, what: "named signal expected count equals current arrivals: public formal and code-sequence boundary rules differ".into() });
                    }
                    state.expected = expected;
                }
                if state.expected == 0 {
                    return Err(wave.fail(None, "arrival on zero-count named barrier"));
                }
                state.arrived = state
                    .arrived
                    .checked_add(1)
                    .ok_or_else(|| wave.fail(None, "named arrival count overflow"))?;
                state.last_arrive.insert(wave.id, state.phase);
                if state.arrived == state.expected {
                    state.arrived = 0;
                    state.phase += 1;
                }
            }
            "s_barrier_wait" => {
                if wave.named_join.is_none() && id == 0 {
                    return Err(EvalError::Unsupported { pc: wave.pc, what: "NULL WAIT needs an explicit NULL join; the initial last-joined selector is not pinned".into() });
                }
                let joined = wave.named_join.unwrap_or(Joined { object, phase });
                self.builder.amd_event(
                    wave.id,
                    pc,
                    AmdEventKind::NamedBarrierWait {
                        encoded: id,
                        object: joined.object,
                        phase: joined.phase,
                    },
                );
                if joined.object.id == 0 {
                    self.builder.amd_event(
                        wave.id,
                        pc,
                        AmdEventKind::NamedBarrierRelease {
                            object: joined.object,
                            phase: joined.phase,
                        },
                    );
                    return Ok(false);
                }
                if wave.named_join.is_none() {
                    return Err(wave.fail(None, "named wait has no preceding join"));
                }
                if !self
                    .named
                    .objects
                    .get(&joined.object.id)
                    .is_some_and(|o| o.incarnation == joined.object.incarnation)
                {
                    return Err(
                        wave.fail(None, "named wait refers to a stale initialization lifetime")
                    );
                }
                wave.named_wait = Some(joined);
                self.stats.block_syncs += 1;
                return Ok(true);
            }
            "s_barrier_leave" => {
                let joined = wave.named_join.unwrap_or(Joined { object, phase });
                self.builder.amd_event(
                    wave.id,
                    pc,
                    AmdEventKind::NamedBarrierLeave {
                        object: joined.object,
                        phase,
                    },
                );
                if joined.object.id == 0 {
                    return Ok(false);
                }
                wave.named_join = None;
                let state = self
                    .named
                    .objects
                    .get_mut(&joined.object.id)
                    .ok_or_else(|| wave.fail(None, "leave of uninitialized named barrier"))?;
                if state.incarnation != joined.object.incarnation {
                    return Err(wave.fail(
                        None,
                        "named leave refers to a stale initialization lifetime",
                    ));
                }
                if state.last_arrive.get(&wave.id) == Some(&state.phase) {
                    return Err(wave.fail(None, "named arrive then drop before phase completion"));
                }
                state.expected = state
                    .expected
                    .checked_sub(1)
                    .ok_or_else(|| wave.fail(None, "named drop makes expected count negative"))?;
                if state.arrived == state.expected {
                    state.arrived = 0;
                    state.phase += 1;
                }
            }
            _ => return Err(self.unsupported(wave, instruction)),
        }
        Ok(false)
    }
    /// The wave's joined named barrier completed a phase the wave has not
    /// waited on (or was reinitialized since the join).
    fn unconsumed_completion(&self, wave: &Wave) -> bool {
        wave.named_join.is_some_and(|joined| {
            joined.object.id != 0
                && self
                    .named
                    .objects
                    .get(&joined.object.id)
                    .is_none_or(|object| {
                        object.incarnation != joined.object.incarnation
                            || object.phase > joined.phase
                    })
        })
    }

    /// litmus/2: `s_getreg_b32 sD, hwreg(HW_REG_WAVE_STATE_PRIV)` directly
    /// after `s_wait_idle`. Only NAMED_BARRIER_ID (8:4) and a clear
    /// NAMED_BARRIER_COMPLETE (3) are defined. Returns false for every other
    /// instruction or profile.
    pub(super) fn state_priv_instruction(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
        name: &str,
    ) -> Result<bool, EvalError> {
        const DEFINED: u32 = 0x1f8;
        if !self.litmus_barrier_profile() || name != "s_getreg_b32" {
            return Ok(false);
        }
        let operands = instruction.operands();
        let [
            Operand::Register(destination),
            Operand::HardwareRegister {
                name: register,
                offset: 0,
                width: 32,
            },
        ] = operands
        else {
            return Ok(false);
        };
        if register != "HW_REG_WAVE_STATE_PRIV" || destination.count != 1 {
            return Ok(false);
        }
        let refuse = |what: &str| EvalError::Unsupported {
            pc: wave.pc,
            what: format!("litmus profile: {what}"),
        };
        if !instruction.modifiers().is_empty() {
            return Err(refuse("STATE_PRIV read with modifiers"));
        }
        let idle = wave
            .previous_pc
            .and_then(|pc| self.program.instructions.get(pc))
            .is_some_and(|previous| previous.mnemonic() == "s_wait_idle");
        if !idle {
            return Err(refuse(
                "STATE_PRIV read not directly after s_wait_idle (documented wait-idle requirement)",
            ));
        }
        let Some(joined) = wave.named_join else {
            return Err(refuse(
                "STATE_PRIV read before any named JOIN (initial named barrier state not pinned)",
            ));
        };
        if self.unconsumed_completion(wave) {
            return Err(refuse(
                "STATE_PRIV read while the joined named barrier has an unconsumed completion",
            ));
        }
        let bits = u32::from(joined.object.id) << 4;
        wave.put_value(&operands[0], 0, Value::partial_bits(bits, DEFINED))?;
        Ok(true)
    }

    pub(super) fn release_named_barriers(&mut self) -> bool {
        let mut progress = false;
        for wave in &mut self.waves {
            if let Some(wait) = wave.named_wait
                && self
                    .named
                    .objects
                    .get(&wait.object.id)
                    .is_some_and(|object| {
                        object.incarnation == wait.object.incarnation && object.phase > wait.phase
                    })
            {
                self.builder.amd_event(
                    wave.id,
                    ProgramPoint(wave.pc as u32),
                    AmdEventKind::NamedBarrierRelease {
                        object: wait.object,
                        phase: wait.phase,
                    },
                );
                wave.named_wait = None;
                wave.named_join.as_mut().unwrap().phase = wait.phase + 1;
                wave.pc += 1;
                progress = true;
            }
        }
        progress
    }
}

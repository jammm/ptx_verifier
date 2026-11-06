//! Native AMD memory ops execution.
use super::*;

impl Interpreter<'_> {
    pub(super) fn atomic_add_f32_no_return(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<(), EvalError> {
        let operands = instruction.operands();
        if self.architecture != AmdArchitecture::Gfx1250
            || operands.len() != 3
            || instruction
                .modifiers()
                .iter()
                .filter(|modifier| {
                    modifier.name == "scope" && modifier.value.as_deref() == Some("SCOPE_DEV")
                })
                .count()
                != 1
            || instruction
                .modifiers()
                .iter()
                .any(|modifier| !matches!(modifier.name.as_str(), "scope" | "scale_offset"))
            || instruction
                .modifiers()
                .iter()
                .filter(|modifier| modifier.name == "scale_offset")
                .count()
                > 1
        {
            return Err(self.unsupported(wave, instruction));
        }
        if wave.exec == 0 {
            return Ok(());
        }
        if wave.exec.count_ones() != 1 {
            return Err(EvalError::Unsupported {
                pc: wave.pc,
                what: "native F32 atomic add currently requires one issuing lane per wave".into(),
            });
        }
        let lane = wave.exec.trailing_zeros() as u8;
        let scale = if instruction
            .modifiers()
            .iter()
            .any(|modifier| modifier.name == "scale_offset")
        {
            4
        } else {
            1
        };
        let address =
            self.global_memory_address(wave, &operands[0], Some(&operands[2]), lane, 0, scale)?;
        self.validate_address(wave, lane, false, address, 4, true)?;
        let previous = self
            .global_read(address, 4)
            .map_err(|error| wave.fail(Some(lane), error))?;
        let previous = self.real_value(wave, previous, lane)?;
        let addend = self.real(wave, &operands[1], lane)?;
        let next = self.arena.add(previous, addend);
        let (reads, writes) = self.instruction_registers(instruction, wave);
        debug_assert!(writes.is_empty());
        let issue = self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::MemoryIssue {
                class: AmdMemoryClass::GlobalAtomicAddF32NoReturn,
                counters: vec![(AmdCounter::Store, 1), (AmdCounter::X, 1)],
                destinations: Vec::new(),
                held_sources: reads,
            },
        );
        self.global_write(address, 4, Value::Real(next))
            .map_err(|error| wave.fail(Some(lane), error))?;
        let thread = ThreadIx(wave.id * wave.width + lane as u32);
        for mode in [AccessMode::Read, AccessMode::Write] {
            let value = if mode == AccessMode::Write {
                Some(self.builder.value(StoredValue::Scalar(next)))
            } else {
                None
            };
            let effect = self.builder.thread_access(
                thread,
                ProgramPoint(wave.pc as u32),
                MemAccess {
                    space: Space::Global,
                    mode,
                    ordering: OrderingKind::Atomic,
                    proxy: Proxy::Generic,
                    footprint: Footprint::Bytes {
                        addr: address,
                        width: 4,
                    },
                    role: OperandRole::Plain,
                    attributed: thread,
                    value,
                },
            );
            self.builder.amd_memory_effect(issue, effect);
        }
        Ok(())
    }
    /// Selected CDNA4 raw DWORD/DWORDX2 profile (ISA 9.1.5, 9.1.7, Table48).
    /// This descriptor is not the CDNA5 base57/records45 layout.
    pub(super) fn gfx950_raw_buffer_addresses(
        &self,
        wave: &Wave,
        instruction: &Instruction,
        immediate: u64,
        registers: u16,
        store: bool,
    ) -> Result<[u64; 64], EvalError> {
        let o = instruction.operands();
        let bytes = 4 * u64::from(registers);
        let descriptor = wave.destination(&o[2])?;
        let mut words = [0u32; 4];
        for (component, word) in words.iter_mut().enumerate() {
            *word = wave
                .reg(descriptor, 0, component as u16)?
                .bits()
                .map_err(|error| wave.fail(None, error))?;
        }
        // Cache/swizzle are zero; raw IDXEN=0 ignores the 14-bit stride.
        // The observed format word has DFMT=4 (32 bits), other fields zero.
        // Untyped DWORD transfers do not apply channel selection/conversion.
        if words[1] >> 30 != 0 || words[3] != 0x20000 {
            return Err(EvalError::Unsupported {
                pc: wave.pc,
                what: "selected gfx950 raw buffer requires linear/cache-swizzle0 and descriptor word3=0x20000".into(),
            });
        }
        let base = u64::from(words[0]) | (u64::from(words[1] & 0xffff) << 32);
        let records = u64::from(words[2]);
        if records == 0 || !records.is_multiple_of(4) {
            return Err(EvalError::Unsupported {
                pc: wave.pc,
                what: "selected gfx950 raw buffer requires a nonzero DWORD-aligned byte bound"
                    .into(),
            });
        }
        let mut addresses = [0u64; 64];
        for lane in wave.active() {
            // Both components are unsigned. Refuse a resource boundary rather
            // than inventing zero-fill/drop effects or wrapping a large offset.
            let offset = u64::from(wave.bits(&o[1], lane)?) + immediate;
            if offset + bytes > records {
                return Err(EvalError::Unsupported {
                    pc: wave.pc,
                    what: "resource-OOB buffer DWORD/DWORDX2 needs explicit suppression/zero-fill graph evidence".into(),
                });
            }
            let address = base
                .checked_add(offset)
                .filter(|address| {
                    address
                        .checked_add(bytes)
                        .is_some_and(|end| end <= 1u64 << 48)
                })
                .ok_or_else(|| EvalError::Unsupported {
                    pc: wave.pc,
                    what:
                        "selected gfx950 buffer footprint escapes the bounded48-bit address domain"
                            .into(),
                })?;
            // Hardware ignores the final address low2 bits; this finite profile
            // accepts already-aligned addresses, without importing SMEM masks.
            if !address.is_multiple_of(4) {
                return Err(EvalError::Unsupported {
                    pc: wave.pc,
                    what: "selected gfx950 raw buffer requires an aligned DWORD address".into(),
                });
            }
            self.validate_memory_range(wave, lane, false, address, bytes, store)?;
            if store {
                let data = wave.destination(&o[0])?;
                // Check every word before issuing the whole instruction: an
                // undefined high word must not leave low-word effects behind.
                for component in 0..registers {
                    if matches!(
                        wave.reg(data, lane, component)?,
                        Value::PartialBits { .. }
                            | Value::Low16Integer(_)
                            | Value::PartialBytePair(..)
                            | Value::PartialHalves(None, _)
                            | Value::PartialHalves(_, None)
                    ) {
                        return Err(wave.fail(
                            Some(lane),
                            "buffer DWORD/DWORDX2 store observes undefined data bits",
                        ));
                    }
                }
            }
            addresses[usize::from(lane)] = address;
        }
        Ok(addresses)
    }
    /// Public CDNA5 Tables54-57: raw linear VBUFFER, IDXEN=0/OOB_SELECT=0.
    pub(super) fn raw_buffer_addresses(
        &self,
        wave: &Wave,
        instruction: &Instruction,
        immediate: u64,
        bytes: u64,
    ) -> Result<[u64; 64], EvalError> {
        let o = instruction.operands();
        let descriptor = wave.destination(&o[2])?;
        let mut resource = 0u128;
        for component in 0..4 {
            let word = wave
                .reg(descriptor, 0, component)?
                .bits()
                .map_err(|error| wave.fail(None, error))?;
            resource |= u128::from(word) << (u32::from(component) * 32);
        }
        let base = (resource & ((1u128 << 57) - 1)) as u64;
        let records = ((resource >> 57) & ((1u128 << 45) - 1)) as u64;
        // Bits107:102 are reserved; bits124/125 select swizzle and the extra
        // stride bound, and bits127:126 select resource type. Stride/scale
        // themselves do not contribute when IDXEN=0 and OOB_SELECT=0.
        if ((resource >> 102) & 63) != 0 || resource >> 124 != 0 {
            return Err(EvalError::Unsupported {
                pc: wave.pc,
                what: "selected gfx1250 raw buffer requires reserved=0, linear OOB_SELECT=0 and buffer type0".into(),
            });
        }
        if records == 0 || records == (1u64 << 45) - 1 {
            return Err(EvalError::Unsupported {
                pc: wave.pc,
                what: "unbound or bounds-disabled buffer resource is outside the selected finite profile".into(),
            });
        }
        let mut addresses = [0u64; 64];
        for lane in wave.active() {
            let offset = (u64::from(wave.bits(&o[1], lane)?) + immediate) & ((1u64 << 45) - 1);
            if offset + bytes > records {
                return Err(EvalError::Unsupported {
                    pc: wave.pc,
                    what: "resource-OOB buffer components need explicit suppression/zero-fill graph evidence".into(),
                });
            }
            let address = base
                .checked_add(offset)
                .filter(|address| {
                    address
                        .checked_add(bytes)
                        .is_some_and(|end| end <= 1u64 << 57)
                })
                .ok_or_else(|| EvalError::Unsupported {
                    pc: wave.pc,
                    what:
                        "selected buffer address/footprint escapes the bounded57-bit address domain"
                            .into(),
                })?;
            addresses[usize::from(lane)] = address;
        }
        Ok(addresses)
    }
    pub(super) fn memory_instruction(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<(), EvalError> {
        let Operation::Memory(memory) = instruction.operation() else {
            unreachable!("typed memory dispatch")
        };
        // These source aliases select the same established execution profile;
        // lowering has already validated their target-specific operand encoding.
        let operation = match memory.operation {
            MemoryOpcode::DsStoreB16D16Hi => MemoryOpcode::DsWriteB16D16Hi,
            MemoryOpcode::DsLoad2addrB32 if self.architecture == AmdArchitecture::Gfx1250 => {
                MemoryOpcode::DsRead2B32
            }
            operation => operation,
        };
        let o = memory.operands();
        if memory.kind == MemoryKind::AsyncLoadToLds {
            return if self.litmus_profile() {
                self.litmus_async_load_to_lds(wave, instruction)
            } else {
                self.async_load_to_lds(wave, instruction)
            };
        }
        if self.litmus_profile()
            && matches!(
                memory.kind,
                MemoryKind::BufferLoad | MemoryKind::BufferStore
            )
        {
            return self.litmus_buffer(wave, instruction);
        }
        if memory.kind == MemoryKind::Prefetch {
            return self.prefetch(wave, instruction);
        }
        if memory.kind == MemoryKind::BufferLoadToLds {
            return self.buffer_load_to_lds(wave, instruction);
        }
        if matches!(
            memory.kind,
            MemoryKind::ScratchLoad | MemoryKind::ScratchStore
        ) {
            return self.scratch(wave, instruction, memory);
        }
        if operation == MemoryOpcode::GlobalAtomicAddF32 {
            return self.atomic_add_f32_no_return(wave, instruction);
        }
        if matches!(
            memory.kind,
            MemoryKind::TensorLoadToLds | MemoryKind::TensorStoreFromLds
        ) {
            return self.tensor(wave, instruction);
        }
        if memory.kind == MemoryKind::Permute {
            return self.permute(wave, instruction);
        }
        if matches!(memory.kind, MemoryKind::FlatLoad | MemoryKind::FlatStore)
            && self.flat_private(wave, instruction, memory)?
        {
            return Ok(());
        }
        let TransferShape::Registers {
            bytes_per_register,
            registers_per_part,
            parts,
        } = memory.shape
        else {
            unreachable!("tensor transfer handled before ordinary memory")
        };
        let store = memory.is_store();
        let scalar = matches!(
            memory.kind,
            MemoryKind::ScalarLoad | MemoryKind::ScalarBufferLoad
        );
        let lds = instruction.opcode() == Opcode::Lds;
        let dual = parts == 2;
        if operation == MemoryOpcode::FlatStoreD16HiB16
            && self.architecture != AmdArchitecture::Gfx1250
        {
            return Err(self.unsupported(wave, instruction));
        }
        if !scalar && wave.exec == 0 {
            return Ok(());
        }
        if self.xnack_replay_enabled && instruction.opcode() == Opcode::VectorMemory {
            return Err(EvalError::Unsupported {pc:wave.pc,what:"XNACK-enabled vector memory replay clauses and source retention are not modeled".into()});
        }
        let flat = matches!(memory.kind, MemoryKind::FlatLoad | MemoryKind::FlatStore);
        let gfx950_buffer = self.architecture == AmdArchitecture::Gfx950
            && matches!(
                memory.operation,
                MemoryOpcode::BufferLoadDword
                    | MemoryOpcode::BufferStoreDword
                    | MemoryOpcode::BufferLoadDwordx2
                    | MemoryOpcode::BufferStoreDwordx2
            );
        let buffer = gfx950_buffer
            || self.architecture == AmdArchitecture::Gfx1250
                && matches!(
                    memory.operation,
                    MemoryOpcode::BufferLoadB128 | MemoryOpcode::BufferStoreB64
                );
        let global = matches!(
            memory.kind,
            MemoryKind::GlobalLoad | MemoryKind::GlobalStore
        );
        if !(matches!(
            memory.kind,
            MemoryKind::ScalarLoad | MemoryKind::LdsLoad | MemoryKind::LdsStore
        ) || global
            || buffer
            || flat
            || dual)
        {
            return Err(self.unsupported(wave, instruction));
        }
        if scalar && self.xnack_replay_enabled {
            self.validate_smem_clause(wave)?;
        }
        let mut dual_offsets = [0u64; 2];
        let offset = instruction
            .modifiers()
            .iter()
            .try_fold(0i64, |offset, modifier| match modifier.name.as_str() {
                "offset" => modifier
                    .value
                    .as_ref()
                    .and_then(|value| parse_i64(value))
                    .ok_or_else(|| self.unsupported(wave, instruction)),
                "glc" | "slc" | "dlc" | "sc0" | "sc1" | "scale_offset" => Ok(offset),
                "offen" if buffer => Ok(offset),
                "scope"
                    if self.architecture == AmdArchitecture::Gfx1250
                        && flat
                        && modifier.value.as_deref() == Some("SCOPE_SYS") =>
                {
                    Ok(offset)
                }
                "offset0" | "offset1" if dual => {
                    let index = if modifier.name == "offset0" { 0 } else { 1 };
                    dual_offsets[index] = modifier
                        .value
                        .as_deref()
                        .and_then(parse_u64)
                        .ok_or_else(|| self.unsupported(wave, instruction))?;
                    Ok(offset)
                }
                _ => Err(self.unsupported(wave, instruction)),
            })?;
        let offset_valid = if gfx950_buffer {
            (0..=4095).contains(&offset)
        } else if buffer {
            (0..=0x7f_ffff).contains(&offset)
        } else if lds {
            (0..=65535).contains(&offset)
        } else if flat && self.architecture == AmdArchitecture::Gfx1250 {
            // Native VFLAT_Real carries a signed24 byte displacement in bits95:72,
            // including the generic-aperture encoding and the scalar-base form.
            (-8_388_608..=8_388_607).contains(&offset)
        } else if flat {
            (0..=4095).contains(&offset)
        } else if scalar {
            (0..=crate::target::scalar_memory_offset_max(self.architecture)).contains(&offset)
        } else if self.architecture == AmdArchitecture::Gfx1250 && global {
            // LLVM FeatureFlatOffsetBits24 and VFLAT_Real encode a signed24
            // immediate. scale_offset scales the VGPR index, not this byte offset.
            (-8_388_608..=8_388_607).contains(&offset)
        } else {
            (-4096..=4095).contains(&offset)
        };
        if !offset_valid || dual_offsets.iter().any(|&value| value > 255) {
            return Err(EvalError::Unsupported {
                pc: wave.pc,
                what: "memory offset is outside its native encoded range".into(),
            });
        }
        let registers_per_part = u16::from(registers_per_part);
        let registers = registers_per_part * u16::from(parts);
        if dual && dual_offsets[0] == dual_offsets[1] {
            return Err(EvalError::Unsupported {
                pc: wave.pc,
                what:
                    "equal-offset DS_READ2/DS_WRITE2 requires explicit duplicate-access semantics"
                        .into(),
            });
        }
        let width = u64::from(bytes_per_register);
        // Capture descriptor and every active lane's offset before any aliased
        // VGPR load result is written. Resource-OOB modes cannot fabricate
        // memory effects in the ordinary VectorLoad/Store graph contract.
        let buffer_addresses = if gfx950_buffer {
            Some(self.gfx950_raw_buffer_addresses(
                wave,
                instruction,
                offset as u64,
                registers,
                store,
            )?)
        } else if buffer {
            Some(self.raw_buffer_addresses(
                wave,
                instruction,
                offset as u64,
                width * u64::from(registers),
            )?)
        } else {
            None
        };
        let scalar_displacement = if scalar {
            wave.wide(&o[2], 0)?
                .checked_mul(
                    if instruction
                        .modifiers()
                        .iter()
                        .any(|m| m.name == "scale_offset")
                    {
                        width * registers as u64
                    } else {
                        1
                    },
                )
                .ok_or_else(|| wave.fail(None, "scaled scalar offset overflow"))?
        } else {
            0
        };
        let scalar_alignment = if self.architecture == AmdArchitecture::Gfx1250
            && matches!(operation, MemoryOpcode::SLoadU8 | MemoryOpcode::SLoadU16)
        {
            width - 1
        } else {
            3
        };
        // SMEM discards each component's low bits before addition. Scaling
        // applies to the SGPR displacement before those byte-address masks.
        let scalar_address = if scalar {
            Some(
                (wave.wide(&o[1], 0)? & !scalar_alignment)
                    .checked_add(scalar_displacement & !scalar_alignment)
                    .and_then(|address| address.checked_add(offset as u64 & !scalar_alignment))
                    .ok_or_else(|| wave.fail(None, "scalar load address overflow"))?,
            )
        } else {
            None
        };
        let flat_scale = if flat
            && instruction
                .modifiers()
                .iter()
                .any(|modifier| modifier.name == "scale_offset")
        {
            if self.architecture != AmdArchitecture::Gfx1250
                || o.len() != 3
                || !matches!(
                    operation,
                    MemoryOpcode::FlatLoadB32 | MemoryOpcode::FlatStoreB32
                )
            {
                return Err(self.unsupported(wave, instruction));
            }
            4
        } else {
            1
        };
        let native_flat_shared = if self.architecture == AmdArchitecture::Gfx1250 && flat {
            let mut shared = None;
            for lane in wave.active() {
                let vector = if store { &o[0] } else { &o[1] };
                let (_, is_shared) = if flat_scale == 1 {
                    self.flat_memory_address_with_saddr(wave, vector, o.get(2), lane, offset)?
                } else {
                    self.flat_memory_address_with_scale(
                        wave,
                        vector,
                        o.get(2),
                        lane,
                        offset,
                        flat_scale,
                    )?
                };
                if shared.is_some_and(|old| old != is_shared) {
                    return Err(wave.fail(
                        Some(lane),
                        "FLAT instruction mixes Global and Shared apertures",
                    ));
                }
                shared = Some(is_shared);
            }
            shared.unwrap_or(false)
        } else {
            false
        };
        let class = if self.architecture == AmdArchitecture::Gfx1250 {
            if flat {
                match (native_flat_shared, store) {
                    (false, false) => AmdMemoryClass::FlatGlobalLoad,
                    (false, true) => AmdMemoryClass::FlatGlobalStore,
                    (true, false) => AmdMemoryClass::FlatLdsLoad,
                    (true, true) => AmdMemoryClass::FlatLdsStore,
                }
            } else if scalar {
                AmdMemoryClass::ScalarLoad
            } else if lds {
                AmdMemoryClass::Ds
            } else if store {
                AmdMemoryClass::VectorStore
            } else {
                AmdMemoryClass::VectorLoad
            }
        } else if flat && store && registers >= 3 {
            AmdMemoryClass::FlatStoreWide
        } else if !scalar && !lds && store && registers >= 3 {
            AmdMemoryClass::VectorStoreWide
        } else if flat {
            AmdMemoryClass::Flat
        } else if scalar {
            AmdMemoryClass::ScalarMemory
        } else if lds {
            AmdMemoryClass::Lds
        } else {
            AmdMemoryClass::VectorMemory
        };
        let counters = if self.architecture == AmdArchitecture::Gfx1250 {
            if scalar {
                vec![(AmdCounter::Km, 1), (AmdCounter::X, 1)]
            } else if lds {
                vec![(AmdCounter::Ds, 1)]
            } else if flat {
                vec![
                    (
                        if store {
                            AmdCounter::Store
                        } else {
                            AmdCounter::Load
                        },
                        1,
                    ),
                    (AmdCounter::Ds, 1),
                    (AmdCounter::X, 1),
                ]
            } else if store {
                vec![(AmdCounter::Store, 1), (AmdCounter::X, 1)]
            } else {
                vec![(AmdCounter::Load, 1), (AmdCounter::X, 1)]
            }
        } else if flat {
            vec![(AmdCounter::Vm, 1), (AmdCounter::Lgkm, 1)]
        } else if scalar {
            vec![(AmdCounter::Lgkm, if registers > 1 { 2 } else { 1 })]
        } else if lds {
            vec![(AmdCounter::Lgkm, 1)]
        } else {
            vec![(AmdCounter::Vm, 1)]
        };
        let (reads, writes) = self.instruction_registers(instruction, wave);
        let held_sources = if self.architecture == AmdArchitecture::Gfx1250 && !lds {
            reads.clone()
        } else if store && !lds && registers >= 3 {
            let data_register = wave.destination(&o[1])?;
            reads
                .iter()
                .copied()
                .filter(|register| {
                    register.index >= data_register.index
                        && register.index < data_register.index + data_register.count
                        && matches!(
                            (register.file, data_register.class),
                            (AmdRegisterFile::Vector, RegisterClass::Vector)
                                | (AmdRegisterFile::Accumulator, RegisterClass::Accumulator)
                        )
                })
                .collect()
        } else {
            Vec::new()
        };
        if self.bounded_profile()
            && buffer_addresses.is_none()
            && let MemoryAddress::Global { vector, scalar } = memory.address()
            && matches!(
                memory.kind,
                MemoryKind::GlobalLoad | MemoryKind::GlobalStore
            )
        {
            let scale = if instruction
                .modifiers()
                .iter()
                .any(|modifier| modifier.name == "scale_offset")
            {
                width * registers as u64
            } else {
                1
            };
            let lanes = wave
                .active()
                .map(|lane| {
                    self.global_memory_address(wave, vector, Some(scalar), lane, offset, scale)
                        .map(|address| (lane, address))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let access = if store {
                AmdGlobalAccess::Store
            } else {
                AmdGlobalAccess::Load
            };
            self.precheck_global_footprint(wave, access, width * u64::from(registers), lanes)?;
        }
        let issue = self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::MemoryIssue {
                class,
                counters,
                destinations: if store { Vec::new() } else { writes },
                held_sources,
            },
        );
        let lanes: Vec<_> = if scalar {
            vec![0]
        } else {
            wave.active().collect()
        };
        for lane in lanes {
            let mut effective_lds = lds;
            let (address, data_operand) = if scalar {
                (
                    scalar_address.expect("scalar address computed before issuance"),
                    &o[0],
                )
            } else if lds {
                if store {
                    (
                        wave.bits(&o[0], lane)?.wrapping_add(offset as u32) as u64,
                        &o[1],
                    )
                } else {
                    (
                        wave.bits(&o[1], lane)?.wrapping_add(offset as u32) as u64,
                        &o[0],
                    )
                }
            } else if let Some(addresses) = &buffer_addresses {
                (addresses[usize::from(lane)], &o[0])
            } else if flat {
                let operand = if store { &o[0] } else { &o[1] };
                let (address, shared) = if self.architecture == AmdArchitecture::Gfx1250 {
                    if flat_scale == 1 {
                        self.flat_memory_address_with_saddr(wave, operand, o.get(2), lane, offset)?
                    } else {
                        self.flat_memory_address_with_scale(
                            wave,
                            operand,
                            o.get(2),
                            lane,
                            offset,
                            flat_scale,
                        )?
                    }
                } else {
                    self.flat_memory_address(wave, operand, lane, offset as u64)?
                };
                effective_lds = shared;
                (address, if store { &o[1] } else { &o[0] })
            } else {
                let (vector, data) = if store {
                    (&o[0], &o[1])
                } else {
                    (&o[1], &o[0])
                };
                let scale = if instruction
                    .modifiers()
                    .iter()
                    .any(|modifier| modifier.name == "scale_offset")
                {
                    width * registers as u64
                } else {
                    1
                };
                (
                    self.global_memory_address(wave, vector, o.get(2), lane, offset, scale)?,
                    data,
                )
            };
            let register = wave.destination(data_operand)?;
            let component_address = |component: u16| -> Result<u64, &str> {
                let displacement = if dual {
                    let part = component / registers_per_part;
                    let unit = width * u64::from(registers_per_part);
                    let stride = if matches!(
                        operation,
                        MemoryOpcode::DsRead2st64B32
                            | MemoryOpcode::DsRead2st64B64
                            | MemoryOpcode::DsWrite2st64B32
                            | MemoryOpcode::DsWrite2st64B64
                    ) {
                        64
                    } else {
                        1
                    };
                    dual_offsets[part as usize] * unit * stride
                        + (component % registers_per_part) as u64 * width
                } else {
                    component as u64 * width
                };
                if lds {
                    Ok((address as u32).wrapping_add(displacement as u32) as u64)
                } else {
                    address
                        .checked_add(displacement)
                        .ok_or("memory address overflow")
                }
            };
            if !effective_lds || !self.config.shared_allocations.is_empty() {
                let mut first = u64::MAX;
                let mut end = 0;
                for component in 0..registers {
                    let addr = component_address(component)
                        .map_err(|error| wave.fail(Some(lane), error))?;
                    first = first.min(addr);
                    end = end.max(addr.checked_add(width).ok_or_else(|| {
                        wave.fail(Some(lane), "shared instruction footprint overflow")
                    })?);
                }
                let kernarg_size = self.program.descriptor.value("kernarg_size", 0);
                let abi_end =
                    if scalar && address >= KERNARG_BASE && address - KERNARG_BASE < kernarg_size {
                        Some(KERNARG_BASE + kernarg_size)
                    } else if scalar && address >= DISPATCH_BASE && address - DISPATCH_BASE < 64 {
                        Some(DISPATCH_BASE + 64)
                    } else {
                        None
                    };
                let fits = if effective_lds {
                    self.config.shared_allocations.iter().any(|allocation| {
                        allocation.offset <= first && end <= allocation.offset + allocation.size
                    })
                } else if let Some(limit) = abi_end {
                    end <= limit
                } else {
                    self.config.arrays.iter().any(|allocation| {
                        allocation.base <= first && end <= allocation.base + allocation.size_bytes()
                    }) || self.module_globals.values().any(|allocation| {
                        allocation.base <= first && end <= allocation.base + allocation.size
                    })
                };
                if !fits {
                    return Err(wave.fail(
                        Some(lane),
                        if effective_lds {
                            "shared instruction footprint crosses configured allocation boundaries"
                        } else if abi_end.is_some() {
                            "load exceeds the declared ABI segment"
                        } else {
                            "out-of-bounds global instruction footprint crosses allocation boundaries"
                        },
                    ));
                }
            }
            for component in 0..registers {
                let addr =
                    component_address(component).map_err(|error| wave.fail(Some(lane), error))?;
                let kernarg_size = self.program.descriptor.value("kernarg_size", 0);
                let kernarg = scalar && addr >= KERNARG_BASE && addr - KERNARG_BASE < kernarg_size;
                let dispatch = scalar && addr >= DISPATCH_BASE && addr - DISPATCH_BASE < 64;
                let abi = kernarg || dispatch;
                if abi
                    && width
                        > if kernarg {
                            kernarg_size - (addr - KERNARG_BASE)
                        } else {
                            64 - (addr - DISPATCH_BASE)
                        }
                {
                    return Err(wave.fail(Some(lane), "load exceeds the declared ABI segment"));
                }
                if !abi {
                    // HSA enables unaligned-access-mode, and gfx1250 has
                    // UnalignedBufferAccess. Limit the established byte-addressed
                    // profile to ordinary GLOBAL B32; SMEM/DS/FLAT, wider
                    // transfers and older targets retain their existing gates.
                    // Whole-instruction allocation bounds were checked above.
                    if self.architecture == AmdArchitecture::Gfx1250
                        && matches!(
                            operation,
                            MemoryOpcode::GlobalLoadB32 | MemoryOpcode::GlobalStoreB32
                        )
                    {
                        self.validate_memory_range(wave, lane, false, addr, width, store)?;
                    } else {
                        self.validate_address(wave, lane, effective_lds, addr, width, store)?;
                    }
                }
                let value = if store {
                    let source_register = if dual && component >= registers_per_part {
                        wave.destination(&o[2])?
                    } else {
                        register
                    };
                    let source_component = if dual {
                        component % registers_per_part
                    } else {
                        component
                    };
                    let value = if operation == MemoryOpcode::FlatStoreD16HiB16 {
                        wave.reg(source_register, lane, source_component)?
                            .high16()
                            .map_err(|error| wave.fail(Some(lane), error))?
                    } else if operation == MemoryOpcode::DsWriteB8D16Hi {
                        match wave.reg(source_register, lane, source_component)? {
                            Value::Bytes(bytes) => Value::Byte(bytes[2]),
                            Value::Byte(_) => Value::Bits(0),
                            value => Value::Bits(
                                (value.bits().map_err(|e| wave.fail(Some(lane), e))? >> 16) & 255,
                            ),
                        }
                    } else if operation == MemoryOpcode::DsWriteB16D16Hi {
                        let mut source = source_register;
                        source.index += source_component;
                        source.count = 1;
                        Value::Half(self.half(wave, &Operand::Register(source), lane, true)?)
                    } else {
                        wave.reg(source_register, lane, source_component)?
                    };
                    // The tag carries an exact zero-filled low half. Expand
                    // only when storing, without inferring bits from a real zero.
                    let value = match value {
                        Value::HighHalf(high) if width == 4 => {
                            Value::Halves(self.arena.int(0), high)
                        }
                        Value::HighHalf(_) if width == 1 || width == 2 => Value::Bits(0),
                        value => value,
                    };
                    let value = if effective_lds {
                        self.lds.write(addr, width, value)
                    } else {
                        self.global_write(addr, width, value)
                    }
                    .map_err(|e| wave.fail(Some(lane), e))?;
                    Some(value)
                } else {
                    let mut value = if dispatch {
                        self.dispatch.read(addr, width)
                    } else if kernarg {
                        self.kernargs
                            .read(addr, width)
                            .or_else(|_| self.kernargs.read_partial_abi(addr, width))
                    } else if effective_lds {
                        self.lds.read(addr, width)
                    } else {
                        self.global_read(addr, width)
                    }
                    .map_err(|e| {
                        wave.fail(
                            Some(lane),
                            format!(
                                "{} {addr:#x}: {e}",
                                if effective_lds {
                                    "LDS"
                                } else if abi {
                                    "kernarg"
                                } else {
                                    "global"
                                }
                            ),
                        )
                    })?;
                    if matches!(
                        operation,
                        MemoryOpcode::BufferLoadSbyte
                            | MemoryOpcode::DsLoadI8
                            | MemoryOpcode::DsReadI8
                            | MemoryOpcode::FlatLoadI8
                            | MemoryOpcode::FlatLoadSbyte
                            | MemoryOpcode::GlobalLoadI8
                            | MemoryOpcode::GlobalLoadSbyte
                    ) {
                        value = match value {
                            Value::Byte(expr) => {
                                self.claim_byte_format(wave, expr, ByteFormat::SignedI8)?;
                                Value::NarrowInt {
                                    expr,
                                    bits: 8,
                                    signed: true,
                                }
                            }
                            Value::NarrowInt {
                                bits: 8,
                                signed: true,
                                ..
                            } => value,
                            _ => Value::SignedInteger(
                                value
                                    .bits()
                                    .map_err(|_| self.unsupported(wave, instruction))?
                                    as i8 as i32,
                            ),
                        };
                    }
                    if matches!(
                        operation,
                        MemoryOpcode::BufferLoadSshort
                            | MemoryOpcode::DsLoadI16
                            | MemoryOpcode::DsReadI16
                            | MemoryOpcode::FlatLoadI16
                            | MemoryOpcode::FlatLoadSshort
                            | MemoryOpcode::GlobalLoadI16
                            | MemoryOpcode::GlobalLoadSshort
                    ) {
                        value = Value::SignedInteger(
                            value
                                .bits()
                                .map_err(|_| self.unsupported(wave, instruction))?
                                as i16 as i32,
                        );
                    }
                    wave.put(register, lane, component, value)?;
                    None
                };
                {
                    let stored = if let Some(value) = value {
                        Some(self.builder.value(match value {
                            Value::Input(expression)
                            | Value::IntExpr(expression)
                            | Value::NarrowInt {
                                expr: expression, ..
                            }
                            | Value::Real(expression)
                            | Value::Half(expression)
                            | Value::Byte(expression) => StoredValue::Scalar(expression),
                            Value::Bits(bits) | Value::Integer(bits) => {
                                StoredValue::Scalar(self.arena.int(bits as i64))
                            }
                            Value::SignedInteger(value) => {
                                StoredValue::Scalar(self.arena.int(value as i64))
                            }
                            Value::Halves(lo, hi) | Value::BytePair(lo, hi) => {
                                StoredValue::Pair(lo, hi)
                            }
                            Value::Bytes(bytes) => StoredValue::Quad(bytes),
                            Value::PartialHalves(..)
                            | Value::HighHalf(_)
                            | Value::WidePart { .. }
                            | Value::PredicateWord { .. }
                            | Value::RelocationWord { .. }
                            | Value::CodeWord { .. }
                            | Value::SharedApertureHi
                            | Value::PrivateApertureHi
                            | Value::PartialBits { .. }
                            | Value::PartialBytePair(..)
                            | Value::Low16Integer(_) => StoredValue::Opaque,
                        }))
                    } else {
                        None
                    };
                    let effect = self.builder.thread_access(
                        ThreadIx(wave.id * wave.width + lane as u32),
                        ProgramPoint(wave.pc as u32),
                        MemAccess {
                            space: if effective_lds {
                                Space::Shared
                            } else {
                                Space::Global
                            },
                            mode: if store {
                                AccessMode::Write
                            } else {
                                AccessMode::Read
                            },
                            ordering: OrderingKind::Ordinary,
                            proxy: Proxy::Generic,
                            footprint: Footprint::Bytes {
                                addr,
                                width: width as u32,
                            },
                            role: OperandRole::Plain,
                            attributed: ThreadIx(wave.id * wave.width + lane as u32),
                            value: stored,
                        },
                    );
                    let scope = if self.architecture == AmdArchitecture::Gfx1250
                        && !effective_lds
                        && (global || flat)
                        && instruction.modifiers().iter().any(|modifier| {
                            modifier.name == "scope"
                                && modifier.value.as_deref() == Some("SCOPE_SYS")
                        }) {
                        AmdMemoryScope::Gfx1250System
                    } else if self.architecture != AmdArchitecture::Gfx1250
                        && global
                        && instruction
                            .modifiers()
                            .iter()
                            .any(|modifier| modifier.name == "sc0")
                        && !instruction
                            .modifiers()
                            .iter()
                            .any(|modifier| modifier.name == "sc1")
                    {
                        AmdMemoryScope::Workgroup
                    } else {
                        AmdMemoryScope::Unspecified
                    };
                    self.builder.amd_memory_effect_scoped(issue, effect, scope);
                }
            }
        }
        Ok(())
    }
    /// A gfx1250 scalar-base vector offset is signed i32 before scaling;
    /// legacy CDNA offsets are unsigned. The `off` vector-pair form retains
    /// all 64 address bits. Address addition wraps at the native 64-bit width.
    pub(super) fn global_memory_address(
        &self,
        wave: &Wave,
        vector: &Operand,
        scalar: Option<&Operand>,
        lane: u8,
        offset: i64,
        scale: u64,
    ) -> Result<u64, EvalError> {
        let index = if self.architecture == AmdArchitecture::Gfx1250
            && matches!(scalar, Some(Operand::Register(_)))
        {
            i64::from(wave.bits(vector, lane)? as i32) as u64
        } else {
            wave.wide(vector, lane)?
        };
        let base = scalar
            .map(|operand| wave.wide(operand, lane))
            .transpose()?
            .unwrap_or(0);
        Ok(base
            .wrapping_add(index.wrapping_mul(scale))
            .wrapping_add_signed(offset))
    }

    pub(super) fn flat_memory_address(
        &self,
        wave: &Wave,
        operand: &Operand,
        lane: u8,
        offset: u64,
    ) -> Result<(u64, bool), EvalError> {
        if let Operand::Register(register) = operand
            && register.count == 2
            && wave.reg(*register, lane, 1)? == Value::SharedApertureHi
        {
            let low = wave
                .reg(*register, lane, 0)?
                .bits()
                .map_err(|error| wave.fail(Some(lane), error))? as u64;
            let address = low
                .checked_add(offset)
                .filter(|value| *value <= u32::MAX as u64)
                .ok_or_else(|| wave.fail(Some(lane), "FLAT offset escapes the shared aperture"))?;
            return Ok((address, true));
        }
        Ok((wave.wide(operand, lane)?.wrapping_add(offset), false))
    }

    /// Resolve the two native FLAT addressing profiles without assigning a
    /// numeric value to the process-dependent shared-memory aperture.
    pub(super) fn flat_memory_address_with_saddr(
        &self,
        wave: &Wave,
        vector: &Operand,
        scalar: Option<&Operand>,
        lane: u8,
        offset: i64,
    ) -> Result<(u64, bool), EvalError> {
        self.flat_memory_address_with_scale(wave, vector, scalar, lane, offset, 1)
    }

    pub(super) fn flat_memory_address_with_scale(
        &self,
        wave: &Wave,
        vector: &Operand,
        scalar: Option<&Operand>,
        lane: u8,
        offset: i64,
        scale: u64,
    ) -> Result<(u64, bool), EvalError> {
        let base = scalar.unwrap_or(vector);
        let vector_offset = if scalar.is_some() {
            i64::from(wave.bits(vector, lane)? as i32)
                .checked_mul(scale as i64)
                .ok_or_else(|| wave.fail(Some(lane), "scaled FLAT index overflow"))?
        } else {
            0
        };
        if let Operand::Register(register) = base
            && register.count == 2
            && wave.reg(*register, lane, 1)? == Value::SharedApertureHi
        {
            let low = wave
                .reg(*register, lane, 0)?
                .bits()
                .map_err(|error| wave.fail(Some(lane), error))?;
            let displacement = vector_offset
                .checked_add(offset)
                .ok_or_else(|| wave.fail(Some(lane), "scaled FLAT offset overflow"))?;
            let address = u64::from(low)
                .checked_add_signed(displacement)
                .filter(|value| *value <= u64::from(u32::MAX))
                .ok_or_else(|| wave.fail(Some(lane), "FLAT offset escapes the shared aperture"))?;
            return Ok((address, true));
        }
        Ok((
            self.global_memory_address(wave, vector, scalar, lane, offset, scale)?,
            false,
        ))
    }
    /// CDNA4 §8.2.1 forbids replay-enabled scalar-memory clauses from overwriting
    /// an address-source SGPR, except one DWORD loaded by an isolated instruction.
    /// Treat each maximal contiguous SMEM run as one clause conservatively.
    pub(super) fn validate_smem_clause(&self, wave: &Wave) -> Result<(), EvalError> {
        let mut start = wave.pc;
        let function = self
            .program
            .functions
            .iter()
            .find(|function| function.start <= wave.pc && wave.pc < function.end)
            .ok_or_else(|| wave.fail(None, "SMEM outside its active source function"))?;
        while start > function.start
            && self.program.instructions[start - 1].opcode() == Opcode::ScalarMemory
        {
            start -= 1;
        }
        let mut end = wave.pc + 1;
        while end < function.end && self.program.instructions[end].opcode() == Opcode::ScalarMemory
        {
            end += 1;
        }
        let clause = &self.program.instructions[start..end];
        if clause.len() == 1
            && matches!(clause[0].operation(), Operation::Memory(memory) if memory.operation == MemoryOpcode::SLoadDword)
        {
            return Ok(());
        }
        let mut sources = std::collections::BTreeSet::new();
        for instruction in clause {
            for operand in instruction.operands().iter().skip(1) {
                if let Operand::Register(register) = operand
                    && register.class == RegisterClass::Scalar
                {
                    sources.extend(register.index..register.index + register.count);
                }
            }
        }
        for instruction in clause {
            if let Some(Operand::Register(destination)) = instruction.operands().first()
                && destination.class == RegisterClass::Scalar
                && let Some(overlap) = (destination.index..destination.index + destination.count)
                    .find(|index| sources.contains(index))
            {
                return Err(EvalError::Unsupported {
                    pc: wave.pc,
                    what: format!(
                        "SMEM clause destination s{overlap} overlaps an address source; only an isolated single-DWORD load permits this alias"
                    ),
                });
            }
        }
        Ok(())
    }
    pub(super) fn permute(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<(), EvalError> {
        if !instruction.modifiers().is_empty() {
            return Err(self.unsupported(wave, instruction));
        }
        if wave.exec == 0 {
            return Ok(());
        }
        let o = instruction.operands();
        let (_, writes) = self.instruction_registers(instruction, wave);
        self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::MemoryIssue {
                class: AmdMemoryClass::CrossLane,
                counters: vec![(
                    if self.architecture == AmdArchitecture::Gfx1250 {
                        AmdCounter::Ds
                    } else {
                        AmdCounter::Lgkm
                    },
                    1,
                )],
                destinations: writes,
                held_sources: Vec::new(),
            },
        );
        let mut data = [Value::Bits(0); 64];
        let mut addresses = [0usize; 64];
        let lanes: Vec<_> = wave.active().collect();
        for &lane in &lanes {
            data[lane as usize] = wave.value(&o[2], lane)?;
            addresses[lane as usize] = ((wave.bits(&o[1], lane)? >> 2) & (wave.width - 1)) as usize;
        }
        if matches!(instruction.operation(), Operation::Memory(memory) if memory.operation == MemoryOpcode::DsBpermuteB32)
        {
            for lane in lanes {
                wave.put_value(&o[0], lane, data[addresses[lane as usize]])?;
            }
        } else {
            let mut outputs = [Value::Bits(0); 64];
            for &lane in &lanes {
                outputs[addresses[lane as usize]] = data[lane as usize];
            }
            for lane in lanes {
                wave.put_value(&o[0], lane, outputs[lane as usize])?;
            }
        }
        self.stats.warp_syncs += 1;
        Ok(())
    }
    pub(super) fn validate_address(
        &self,
        wave: &Wave,
        lane: u8,
        lds: bool,
        addr: u64,
        width: u64,
        write: bool,
    ) -> Result<(), EvalError> {
        self.validate_memory_range(wave, lane, lds, addr, width, write)?;
        if !addr.is_multiple_of(width.min(4)) {
            return Err(wave.fail(
                Some(lane),
                format!("misaligned access at {addr:#x} width {width}"),
            ));
        }
        Ok(())
    }
    pub(super) fn validate_memory_range(
        &self,
        wave: &Wave,
        lane: u8,
        lds: bool,
        addr: u64,
        width: u64,
        write: bool,
    ) -> Result<(), EvalError> {
        let valid = if lds {
            addr <= self.lds_size && width <= self.lds_size - addr
        } else {
            self.config.arrays.iter().any(|array| {
                width <= array.size_bytes()
                    && addr >= array.base
                    && addr - array.base <= array.size_bytes() - width
            }) || self.module_globals.values().any(|object| {
                addr >= object.base
                    && width <= object.size
                    && addr - object.base <= object.size - width
            })
        };
        if !valid {
            return Err(wave.fail(
                Some(lane),
                format!(
                    "out-of-bounds {} access [{addr:#x},+{width})",
                    if lds { "LDS" } else { "global" }
                ),
            ));
        }
        if !lds
            && write
            && self.module_globals.values().any(|object| {
                object.read_only && addr >= object.base && addr - object.base < object.size
            })
        {
            return Err(wave.fail(Some(lane), "store to read-only module-global object"));
        }
        Ok(())
    }
}

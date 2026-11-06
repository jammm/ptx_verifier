//! Physical register roles of a normalized instruction, including implicit mask and pair dependencies.
use super::*;

impl Interpreter<'_> {
    pub(super) fn instruction_registers(
        &self,
        instruction: &Instruction,
        wave: &Wave,
    ) -> (Vec<AmdRegister>, Vec<AmdRegister>) {
        self.instruction_registers_for_operands(instruction, instruction.operands(), wave)
    }
    pub(super) fn instruction_registers_for_operands(
        &self,
        instruction: &Instruction,
        operands: &[Operand],
        wave: &Wave,
    ) -> (Vec<AmdRegister>, Vec<AmdRegister>) {
        fn add(operand: &Operand, mask: u64, result: &mut Vec<AmdRegister>) {
            match operand {
                Operand::Register(register) => {
                    let (file, base, count) = match register.class {
                        RegisterClass::Scalar => {
                            (AmdRegisterFile::Scalar, register.index, register.count)
                        }
                        RegisterClass::Temporary => {
                            (AmdRegisterFile::Temporary, register.index, register.count)
                        }
                        RegisterClass::Vector => {
                            (AmdRegisterFile::Vector, register.index, register.count)
                        }
                        RegisterClass::Accumulator => {
                            (AmdRegisterFile::Accumulator, register.index, register.count)
                        }
                        RegisterClass::Special => match register.special {
                            Some(SpecialRegister::Vcc) => (AmdRegisterFile::Scalar, 106, 2),
                            Some(SpecialRegister::VccLo) => (AmdRegisterFile::Scalar, 106, 1),
                            Some(SpecialRegister::VccHi) => (AmdRegisterFile::Scalar, 107, 1),
                            Some(SpecialRegister::Exec) => (AmdRegisterFile::Scalar, 126, 2),
                            Some(SpecialRegister::ExecLo) => (AmdRegisterFile::Scalar, 126, 1),
                            Some(SpecialRegister::ExecHi) => (AmdRegisterFile::Scalar, 127, 1),
                            Some(SpecialRegister::M0) => (AmdRegisterFile::Scalar, 124, 1),
                            _ => return,
                        },
                    };
                    if mask == 0
                        && !matches!(file, AmdRegisterFile::Scalar | AmdRegisterFile::Temporary)
                    {
                        return;
                    }
                    for component in 0..count {
                        result.push(AmdRegister {
                            file,
                            index: base + component,
                            lanes: if matches!(
                                file,
                                AmdRegisterFile::Scalar | AmdRegisterFile::Temporary
                            ) {
                                u64::MAX
                            } else {
                                mask
                            },
                        });
                    }
                }
                Operand::Modified { value, .. } => add(value, mask, result),
                _ => {}
            }
        }
        if let Operation::Dual { x, y } = instruction.operation() {
            let (mut reads, mut writes) = self.instruction_registers(x, wave);
            let (r, w) = self.instruction_registers(y, wave);
            reads.extend(r);
            writes.extend(w);
            return (reads, writes);
        }
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        if wave.exec == 0
            && matches!(instruction.operation(), Operation::Memory(memory)
            if matches!(memory.kind, MemoryKind::ScratchLoad | MemoryKind::ScratchStore))
        {
            return (reads, writes);
        }
        match instruction.operation() {
            Operation::Call(_) => {
                add(&operands[0], u64::MAX, &mut writes);
                add(&operands[1], u64::MAX, &mut reads);
                return (reads, writes);
            }
            Operation::Return(_) => {
                add(&operands[0], u64::MAX, &mut reads);
                return (reads, writes);
            }
            _ => {}
        }
        let alu = instruction.alu_operation();
        // SDWA uses its separate dependency profile; it did not select the
        // ordinary exact-form operand roles before semantic normalization.
        let exact_alu = (instruction.source().encoding != Encoding::Sdwa)
            .then_some(alu)
            .flatten();
        if matches!(
            alu,
            Some(AluOperation::Compare(Comparison {
                writes_exec: true,
                ..
            }))
        ) {
            writes.extend(
                (126..126 + (wave.width / 32) as u16).map(|index| AmdRegister {
                    file: AmdRegisterFile::Scalar,
                    index,
                    lanes: u64::MAX,
                }),
            );
        }
        if self.architecture == AmdArchitecture::Gfx950
            && matches!(
                exact_alu,
                Some(AluOperation::VReadlaneB32 | AluOperation::VWritelaneB32)
            )
        {
            let lane = match operands[2] {
                Operand::Integer(lane) => lane as u32,
                _ => unreachable!("validated literal lane"),
            };
            let read = exact_alu == Some(AluOperation::VReadlaneB32);
            add(
                &operands[0],
                if read { u64::MAX } else { 1u64 << lane },
                &mut writes,
            );
            add(
                &operands[1],
                if read { 1u64 << lane } else { u64::MAX },
                &mut reads,
            );
            return (reads, writes);
        }
        if exact_alu == Some(AluOperation::VReadlaneB32)
            && self.architecture == AmdArchitecture::Gfx1250
        {
            let lane = wave.bits(&operands[2], 0).unwrap_or(0) & (wave.width - 1);
            add(&operands[0], u64::MAX, &mut writes);
            add(&operands[1], 1u64 << lane, &mut reads);
            add(&operands[2], u64::MAX, &mut reads);
            return (reads, writes);
        }
        if wave.exec == 0
            && matches!(
                instruction.opcode(),
                Opcode::VectorAlu | Opcode::VectorMemory | Opcode::Lds
            )
        {
            // Compare and carry masks overwrite their complete scalar result,
            // including clearing bits for inactive lanes. Ordinary vector data
            // sources and destinations have no obligations under empty EXEC.
            let mask_destination = if matches!(
                alu,
                Some(AluOperation::Compare(Comparison {
                    writes_exec: false,
                    ..
                }))
            ) || matches!(
                alu,
                Some(AluOperation::Compare(Comparison {
                    writes_exec: true,
                    ..
                }))
            ) {
                operands.first()
            } else if matches!(
                exact_alu,
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
            ) {
                operands.get(1)
            } else {
                None
            };
            if let Some(operand) = mask_destination.filter(|_| {
                !(wave.width == 32
                    && matches!(
                        alu,
                        Some(AluOperation::Compare(Comparison {
                            writes_exec: true,
                            ..
                        }))
                    ))
            }) {
                add(operand, u64::MAX, &mut writes);
            }
            return (reads, writes);
        }
        if matches!(
            exact_alu,
            Some(AluOperation::VPkMulF32 | AluOperation::VPkAddF32 | AluOperation::VPkFmaF32)
        ) {
            add(&operands[0], wave.exec, &mut writes);
            let count = operands.len() - 1;
            if let (Some(low), Some(high)) = (
                selectors(instruction, "op_sel", count, 0),
                selectors(instruction, "op_sel_hi", count, 1),
            ) {
                fn selected(
                    operand: &Operand,
                    component: u16,
                    mask: u64,
                    result: &mut Vec<AmdRegister>,
                ) {
                    match operand {
                        Operand::Register(register) if register.count == 2 => {
                            let mut register = *register;
                            register.index += component;
                            register.count = 1;
                            add(&Operand::Register(register), mask, result);
                        }
                        Operand::Modified { value, .. } => selected(value, component, mask, result),
                        _ => add(operand, mask, result),
                    }
                }
                for index in 0..count {
                    selected(&operands[index + 1], low[index], wave.exec, &mut reads);
                    if high[index] != low[index] {
                        selected(&operands[index + 1], high[index], wave.exec, &mut reads);
                    }
                }
                return (reads, writes);
            }
        }
        if exact_alu == Some(AluOperation::VReadfirstlaneB32)
            || exact_alu == Some(AluOperation::VReadlaneB32)
        {
            let lane = if exact_alu == Some(AluOperation::VReadfirstlaneB32) {
                wave.exec.trailing_zeros().min(wave.width - 1)
            } else {
                wave.bits(&operands[2], 0).unwrap_or(0) & (wave.width - 1)
            };
            add(&operands[0], u64::MAX, &mut writes);
            add(&operands[1], 1u64 << lane, &mut reads);
            if operands.len() > 2 {
                add(&operands[2], u64::MAX, &mut reads);
            }
            return (reads, writes);
        }
        if exact_alu == Some(AluOperation::VWritelaneB32) {
            let lane = wave.bits(&operands[2], 0).unwrap_or(0) & (wave.width - 1);
            add(&operands[0], 1u64 << lane, &mut writes);
            add(&operands[1], u64::MAX, &mut reads);
            add(&operands[2], u64::MAX, &mut reads);
            return (reads, writes);
        }
        let store = matches!(instruction.operation(), Operation::Memory(memory) if matches!(memory.kind,
            MemoryKind::GlobalStore | MemoryKind::FlatStore | MemoryKind::LdsStore
            | MemoryKind::LdsWritePair | MemoryKind::TensorStoreFromLds
            | MemoryKind::BufferStore | MemoryKind::ScratchStore))
            || matches!(instruction.operation(), Operation::Memory(memory) if memory.operation == MemoryOpcode::GlobalAtomicAddF32);
        let no_destination = matches!(instruction.operation(), Operation::Memory(memory) if matches!(memory.kind, MemoryKind::TensorLoadToLds | MemoryKind::TensorStoreFromLds | MemoryKind::AsyncLoadToLds | MemoryKind::Prefetch))
            || matches!(instruction.operation(), Operation::Memory(memory) if memory.kind == MemoryKind::BufferLoadToLds)
            || (wave.width == 32
                && matches!(
                    alu,
                    Some(AluOperation::Compare(Comparison {
                        writes_exec: true,
                        ..
                    }))
                ))
            || matches!(
                instruction.opcode(),
                Opcode::Branch | Opcode::Wait | Opcode::Barrier | Opcode::Control
            )
            || matches!(
                alu,
                Some(
                    AluOperation::ScalarCompare { .. }
                        | AluOperation::SBitcmp0B32
                        | AluOperation::SBitcmp1B32
                )
            );
        for (index, operand) in operands.iter().enumerate() {
            if (index == 0
                || (index == 1
                    && (matches!(alu, Some(AluOperation::VAddCoU32))
                        || matches!(alu, Some(AluOperation::VAddcCoU32))
                        || matches!(alu, Some(AluOperation::VSubCoU32))
                        || matches!(alu, Some(AluOperation::VSubbCoU32))
                        || matches!(alu, Some(AluOperation::VSubrevCoU32))
                        || matches!(alu, Some(AluOperation::VSubbrevCoU32))
                        || matches!(
                            exact_alu,
                            Some(AluOperation::VMadU64U32 | AluOperation::VMadI64I32)
                        ))))
                && !store
                && !no_destination
            {
                add(operand, wave.exec, &mut writes)
            } else {
                add(operand, wave.exec, &mut reads)
            }
        }
        if matches!(instruction.operation(), Operation::Memory(memory) if memory.kind == MemoryKind::BufferLoadToLds)
        {
            reads.push(AmdRegister {
                file: AmdRegisterFile::Scalar,
                index: 124,
                lanes: u64::MAX,
            });
        }
        if matches!(
            exact_alu,
            Some(
                AluOperation::VFmacF32
                    | AluOperation::VFmacF64
                    | AluOperation::VFmacF16
                    | AluOperation::VFmaMixloF16
                    | AluOperation::VFmaMixhiF16
                    | AluOperation::SMulkI32
                    | AluOperation::SAddkI32
                    | AluOperation::SBitset1B32
            )
        ) || (self.architecture == AmdArchitecture::Gfx1250
            && instruction.mnemonic().starts_with("v_swmmac_"))
        {
            // SWMMAC reads its C from the destination registers.
            add(&operands[0], wave.exec, &mut reads);
        }
        if self.architecture == AmdArchitecture::Gfx1250
            && matches!(
                exact_alu,
                Some(
                    AluOperation::VAddF16
                        | AluOperation::VSubF16
                        | AluOperation::VMulF16
                        | AluOperation::VMaxF16
                        | AluOperation::VMinF16
                        | AluOperation::VCvtF16F32
                )
            )
        {
            // True16 arithmetic updates low16 and preserves the destination
            // high16, including any asynchronous producer of that half.
            add(&operands[0], wave.exec, &mut reads);
        }
        if self.architecture == AmdArchitecture::Gfx1250
            && instruction.opcode() == Opcode::VectorMemory
            && wave.exec != 0
        {
            reads.push(AmdRegister {
                file: AmdRegisterFile::Scalar,
                index: 126,
                lanes: u64::MAX,
            });
        }
        if matches!(instruction.operation(), Operation::Barrier(barrier) if barrier.operation == BarrierOperation::Init)
            && !matches!(operands, [Operand::Integer(0)])
            && !reads
                .iter()
                .any(|r| r.file == AmdRegisterFile::Scalar && r.index == 124)
        {
            reads.push(AmdRegister {
                file: AmdRegisterFile::Scalar,
                index: 124,
                lanes: u64::MAX,
            });
        }
        (reads, writes)
    }
    pub(super) fn scalar_mask_reads(
        &self,
        instruction: &Instruction,
        wave: &Wave,
    ) -> Vec<AmdRegister> {
        if let Operation::Dual { x, y } = instruction.operation() {
            let rx = self.scalar_mask_reads(x, wave);
            let ry = self.scalar_mask_reads(y, wave);
            let (xr, _) = self.instruction_registers(x, wave);
            let (yr, _) = self.instruction_registers(y, wave);
            let data: Vec<_> = xr
                .into_iter()
                .filter(|r| !rx.contains(r))
                .chain(yr.into_iter().filter(|r| !ry.contains(r)))
                .collect();
            return rx
                .into_iter()
                .chain(ry)
                .filter(|r| !data.iter().any(|d| r.overlaps(*d)))
                .collect();
        }
        if wave.exec == 0 || instruction.opcode() != Opcode::VectorAlu {
            return Vec::new();
        }
        let alu = instruction.alu_operation();
        let role = if alu == Some(AluOperation::VCndmaskB32) {
            Some(3)
        } else if matches!(
            alu,
            Some(AluOperation::VAddcCoU32 | AluOperation::VSubbCoU32 | AluOperation::VSubbrevCoU32)
        ) {
            Some(4)
        } else {
            None
        };
        let Some(role) = role else {
            return Vec::new();
        };
        fn scalar(operand: &Operand, out: &mut Vec<AmdRegister>) {
            match operand {
                Operand::Modified { value, .. } => scalar(value, out),
                Operand::Register(register) => {
                    let (index, count) = match register.class {
                        RegisterClass::Scalar => (register.index, register.count),
                        RegisterClass::Special => match register.special {
                            Some(SpecialRegister::Vcc) => (106, 2),
                            Some(SpecialRegister::VccLo) => (106, 1),
                            Some(SpecialRegister::VccHi) => (107, 1),
                            _ => return,
                        },
                        _ => return,
                    };
                    out.extend((index..index + count).map(|index| AmdRegister {
                        file: AmdRegisterFile::Scalar,
                        index,
                        lanes: u64::MAX,
                    }));
                }
                _ => {}
            }
        }
        let mut masks = Vec::new();
        if let Some(operand) = instruction.operands().get(role) {
            scalar(operand, &mut masks);
        }
        let mut data = Vec::new();
        let first_source = if role == 4 { 2 } else { 1 };
        for operand in &instruction.operands()[first_source..role.min(instruction.operands().len())]
        {
            scalar(operand, &mut data);
        }
        masks.retain(|register| !data.iter().any(|other| register.overlaps(*other)));
        masks
    }
}

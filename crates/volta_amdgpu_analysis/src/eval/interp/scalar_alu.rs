//! Native AMD scalar alu execution.
use super::*;

impl Interpreter<'_> {
    pub(super) fn scalar(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
        operation: AluOperation,
    ) -> Result<(), EvalError> {
        if self.code_address_scalar(wave, instruction)?
            || self.module_address_scalar(wave, instruction)?
        {
            return Ok(());
        }
        if let Operation::ScalarAlu(alu) = instruction.operation()
            && let Some(sources) = alu
                .scalar_sources()
                .map_err(|error| wave.fail(None, error))?
        {
            let destination = &instruction.operands()[0];
            let value = match sources {
                ScalarSources::Move64(source) => return wave.move_scalar64(destination, source),
                ScalarSources::Binary64 { left, right } => {
                    let a = wave.scalar_source64(left)?;
                    let b = wave.scalar_source64(right)?;
                    match alu.operation {
                        AluOperation::Bitwise {
                            domain: AluDomain::Scalar,
                            kind: BitwiseKind::Or,
                            width: BitWidth::B64,
                            save_exec: false,
                        } => a | b,
                        AluOperation::Bitwise {
                            domain: AluDomain::Scalar,
                            kind: BitwiseKind::Xor,
                            width: BitWidth::B64,
                            save_exec: false,
                        } => a ^ b,
                        _ => return Err(self.unsupported(wave, instruction)),
                    }
                }
                ScalarSources::Shift64 { data, count } => {
                    let data = wave.scalar_source64(data)?;
                    let count = wave.scalar_source32(count)? & 63;
                    match alu.operation {
                        AluOperation::Shift {
                            domain: AluDomain::Scalar,
                            kind: ShiftKind::Left,
                            width: BitWidth::B64,
                            reverse: false,
                        } => data.wrapping_shl(count),
                        AluOperation::Shift {
                            domain: AluDomain::Scalar,
                            kind: ShiftKind::LogicalRight,
                            width: BitWidth::B64,
                            reverse: false,
                        } => data.wrapping_shr(count),
                        _ => return Err(self.unsupported(wave, instruction)),
                    }
                }
            };
            wave.set_scc(value != 0);
            if self.architecture == AmdArchitecture::Gfx1250 {
                let destination = wave.destination(destination)?;
                wave.put(destination, 0, 0, Value::Integer(value as u32))?;
                return wave.put(destination, 0, 1, Value::Integer((value >> 32) as u32));
            }
            return wave.put_wide(destination, 0, value);
        }
        if self.architecture == AmdArchitecture::Gfx950
            && operation == AluOperation::SMovB32
            && matches!(instruction.operands()[0], Operand::Register(register) if register.special == Some(SpecialRegister::M0))
        {
            if !instruction.modifiers().is_empty()
                || !matches!(instruction.operands()[1], Operand::Register(register)
                    if register.class == RegisterClass::Scalar && register.count == 1)
                    && !matches!(instruction.operands()[1], Operand::Integer(value) if (i32::MIN as i64..=u32::MAX as i64).contains(&value))
            {
                return Err(self.unsupported(wave, instruction));
            }
            wave.m0 = wave.bits(&instruction.operands()[1], 0)?;
            wave.m0_defined = true;
            return Ok(());
        }
        if let AluOperation::Integer {
            domain: AluDomain::Scalar,
            kind: kind @ (IntegerKind::Add | IntegerKind::Mul),
            ty:
                IntegerType {
                    width: BitWidth::B64,
                    signed: false,
                },
            source_bits: 64,
            form: IntegerForm::Plain,
        } = operation
        {
            let a = wave.wide(&instruction.operands()[1], 0)?;
            let b = wave.wide(&instruction.operands()[2], 0)?;
            let value = if kind == IntegerKind::Mul {
                a.wrapping_mul(b)
            } else {
                a.wrapping_add(b)
            };
            let reg = wave.destination(&instruction.operands()[0])?;
            wave.put(reg, 0, 0, Value::Integer(value as u32))?;
            return wave.put(reg, 0, 1, Value::Integer((value >> 32) as u32));
        }
        if let AluOperation::Float {
            domain: AluDomain::Scalar,
            kind: kind @ (FloatKind::Add | FloatKind::Sub | FloatKind::Mul | FloatKind::MinNum),
            ty: FloatType::F32,
        } = operation
        {
            if !instruction.modifiers().is_empty() {
                return Err(self.unsupported(wave, instruction));
            }
            let a = self.real(wave, &instruction.operands()[1], 0)?;
            let b = self.real(wave, &instruction.operands()[2], 0)?;
            let value = match kind {
                FloatKind::Add => self.arena.add(a, b),
                FloatKind::Sub => self.arena.sub(a, b),
                FloatKind::Mul => self.arena.mul(a, b),
                FloatKind::MinNum => self.arena.min(a, b),
                _ => unreachable!("selected scalar binary float family"),
            };
            return wave.put_value(&instruction.operands()[0], 0, Value::Real(value));
        }
        if let AluOperation::Conversion {
            domain: AluDomain::Scalar,
            source: source @ (ConversionType::I32 | ConversionType::U32),
            destination: ConversionType::F32,
            mode: ConversionMode::Plain,
        } = operation
        {
            if self.architecture != AmdArchitecture::Gfx1250 || !instruction.modifiers().is_empty()
            {
                return Err(self.unsupported(wave, instruction));
            }
            let ty = super::super::scalar::IntType {
                bits: 32,
                signed: source == ConversionType::I32,
            };
            let expr = self.integer_expression(wave, &instruction.operands()[1], 0, ty)?;
            let expr = if let Some(value) = self.arena.as_int_const(expr) {
                self.arena.int(
                    super::super::scalar::canonical_int(value, ty)
                        .map_err(|e| wave.fail(None, e))?,
                )
            } else {
                expr
            };
            return wave.put_value(&instruction.operands()[0], 0, Value::Real(expr));
        }
        if operation == AluOperation::SBfeI64 {
            return self.signed_bitfield64(wave, instruction);
        }
        let name = instruction.mnemonic();
        if self.evidence_instruction(wave, instruction, name)? {
            return Ok(());
        }
        if self.mode_instruction(wave, instruction, name)? {
            return Ok(());
        }
        if self.state_priv_instruction(wave, instruction, name)? {
            return Ok(());
        }
        if operation == AluOperation::SGetregB32 {
            if self.architecture != AmdArchitecture::Gfx1250
                || !matches!(&instruction.operands()[1],Operand::HardwareRegister{name:register_name,offset:6,width:4} if register_name=="HW_REG_IB_STS2")
            {
                return Err(self.unsupported(wave, instruction));
            }
            return wave.put_value(&instruction.operands()[0], 0, Value::Integer(0));
        }
        if !matches!(
            operation,
            AluOperation::SCmpEqI32
                | AluOperation::SCmpEqU32
                | AluOperation::SCmpEqU64
                | AluOperation::SCmpGeI32
                | AluOperation::SCmpGeU32
                | AluOperation::SCmpGtI32
                | AluOperation::SCmpGtU32
                | AluOperation::SCmpLeI32
                | AluOperation::SCmpLeU32
                | AluOperation::SCmpLgI32
                | AluOperation::SCmpLgU32
                | AluOperation::SCmpLgU64
                | AluOperation::SCmpLtI32
                | AluOperation::SCmpLtU32
        ) && !matches!(
            operation,
            AluOperation::SCmpkEqI32
                | AluOperation::SCmpkEqU32
                | AluOperation::SCmpkGtU32
                | AluOperation::SCmpkLgI32
                | AluOperation::SCmpkLgU32
                | AluOperation::SCmpkLtU32
        ) && !matches!(
            operation,
            AluOperation::SBitcmp0B32 | AluOperation::SBitcmp1B32
        ) && !matches!(
            operation,
            AluOperation::SMovB32
                | AluOperation::SMovB64
                | AluOperation::SMovkI32
                | AluOperation::SMulkI32
                | AluOperation::SAddkI32
                | AluOperation::SBitset1B32
                | AluOperation::SSextI32I16
                | AluOperation::SFf1I32B64
                | AluOperation::SFf1I32B32
                | AluOperation::SBrevB32
                | AluOperation::SCselectB32
                | AluOperation::SCselectB64
                | AluOperation::SAndB64
                | AluOperation::SOrB64
                | AluOperation::SXorB64
                | AluOperation::SAndn2B64
                | AluOperation::SOrn2B64
                | AluOperation::SNotB64
                | AluOperation::SAndSaveexecB64
                | AluOperation::SOrSaveexecB64
                | AluOperation::SXorSaveexecB64
                | AluOperation::SAndn2SaveexecB64
                | AluOperation::SLshlB64
                | AluOperation::SLshrB64
                | AluOperation::SAddU32
                | AluOperation::SAddI32
                | AluOperation::SSubI32
                | AluOperation::SAddcU32
                | AluOperation::SSubU32
                | AluOperation::SSubbU32
                | AluOperation::SMulI32
                | AluOperation::SMulHiU32
                | AluOperation::SAndB32
                | AluOperation::SOrB32
                | AluOperation::SNorB32
                | AluOperation::SXorB32
                | AluOperation::SAndn2B32
                | AluOperation::SNotB32
                | AluOperation::SLshlB32
                | AluOperation::SLshrB32
                | AluOperation::SAshrI32
                | AluOperation::SMinU32
                | AluOperation::SMaxU32
                | AluOperation::SMinI32
                | AluOperation::SMaxI32
                | AluOperation::SAndSaveexecB32
                | AluOperation::SAndn2SaveexecB32
                | AluOperation::SOrSaveexecB32
                | AluOperation::SMulU64
                | AluOperation::SAddNcU64
                | AluOperation::SBfeU32
        ) {
            return Err(self.unsupported(wave, instruction));
        }
        let o = instruction.operands();
        if !instruction.modifiers().is_empty() {
            return Err(self.unsupported(wave, instruction));
        }
        if self.architecture == AmdArchitecture::Gfx1250
            && matches!(
                operation,
                AluOperation::SAddI32
                    | AluOperation::SSubI32
                    | AluOperation::SAddU32
                    | AluOperation::SSubU32
                    | AluOperation::SMulI32
                    | AluOperation::SXorB32
                    | AluOperation::SAndB32
                    | AluOperation::SLshrB32
            )
            && [1, 2].into_iter().any(|i| {
                matches!(
                    wave.value(&o[i], 0),
                    Ok(Value::Input(_) | Value::IntExpr(_) | Value::NarrowInt { .. })
                )
            })
        {
            let (ty, op) =
                scalar_symbolic_binary(operation).expect("selected symbolic scalar word operation");
            let a = self.integer_expression(wave, &o[1], 0, ty)?;
            let b = self.integer_expression(wave, &o[2], 0, ty)?;
            let value = super::super::scalar::int_binary(&mut self.arena, ty, op, a, b)
                .map_err(|e| wave.fail(None, e))?;
            if matches!(
                operation,
                AluOperation::SXorB32 | AluOperation::SAndB32 | AluOperation::SLshrB32
            ) {
                let zero = self.arena.int(0);
                let nonzero = self.arena.ne(value, zero);
                if let Some(nonzero) = self.arena.as_bool(nonzero) {
                    wave.set_scc(nonzero);
                } else {
                    wave.clear_scc();
                    wave.scc_condition = Some(nonzero);
                }
            } else if operation != AluOperation::SMulI32 {
                wave.clear_scc();
            }
            return wave.put_value(&o[0], 0, Value::IntExpr(value));
        }
        if let AluOperation::Bitfield {
            domain: AluDomain::Scalar,
            kind: kind @ (BitfieldKind::TestZero | BitfieldKind::TestOne),
            width: BitWidth::B32,
            signed: false,
        } = operation
        {
            let bits = wave.bits(&o[0], 0)?;
            let bit = wave.bits(&o[1], 0)? & 31;
            let set = bits & (1u32 << bit) != 0;
            wave.set_scc(if kind == BitfieldKind::TestOne {
                set
            } else {
                !set
            });
            return Ok(());
        }
        if let AluOperation::ScalarCompare {
            relation,
            ty,
            immediate: true,
        } = operation
        {
            if !matches!(ty, CompareType::I32 | CompareType::U32) {
                return Err(self.unsupported(wave, instruction));
            }
            let signed = ty == CompareType::I32;
            let a = wave.bits(&o[0], 0)?;
            let b = wave.bits(&o[1], 0)?;
            let result = compare(
                relation,
                if signed { a as i32 as i64 } else { a as i64 },
                if signed {
                    b as i16 as i64
                } else {
                    b as u16 as i64
                },
            )
            .ok_or_else(|| self.unsupported(wave, instruction))?;
            wave.set_scc(result);
            return Ok(());
        }
        if self.architecture == AmdArchitecture::Gfx1250
            && matches!(
                operation,
                AluOperation::ScalarCompare {
                    ty: CompareType::I32 | CompareType::U32,
                    immediate: false,
                    ..
                }
            )
            && [0, 1].into_iter().any(|i| {
                matches!(
                    wave.value(&o[i], 0),
                    Ok(Value::Input(_) | Value::IntExpr(_) | Value::NarrowInt { .. })
                )
            })
        {
            use super::super::scalar::{CmpOp, IntType, ScalarType};
            let AluOperation::ScalarCompare {
                relation,
                ty: comparison_type,
                ..
            } = operation
            else {
                unreachable!("scalar comparison family")
            };
            let op = match relation {
                Relation::Eq => CmpOp::Eq,
                Relation::Lg | Relation::Ne => CmpOp::Ne,
                Relation::Lt => CmpOp::Lt,
                Relation::Le => CmpOp::Le,
                Relation::Gt => CmpOp::Gt,
                Relation::Ge => CmpOp::Ge,
                _ => return Err(self.unsupported(wave, instruction)),
            };
            // Equality observes the same32 bits for signed/unsigned encodings.
            // Use signed canonical constants consistently for symbolic words.
            let ty = IntType {
                bits: 32,
                signed: comparison_type == CompareType::I32 || matches!(op, CmpOp::Eq | CmpOp::Ne),
            };
            let a = self.integer_expression(wave, &o[0], 0, ty)?;
            let b = self.integer_expression(wave, &o[1], 0, ty)?;
            let predicate =
                super::super::scalar::compare(&mut self.arena, ScalarType::Int(ty), op, a, b)
                    .map_err(|e| wave.fail(None, e))?;
            if let Some(value) = self.arena.as_bool(predicate) {
                wave.set_scc(value);
            } else {
                wave.clear_scc();
                wave.scc_condition = Some(predicate);
            }
            return Ok(());
        }
        if let AluOperation::ScalarCompare {
            relation: relation @ (Relation::Eq | Relation::Lg),
            ty: CompareType::U64,
            immediate: false,
        } = operation
        {
            // SOPC compares complete 64-bit sources. Keep ambiguous literal
            // widths and special-register aliases outside this selected profile.
            if o.iter().any(|operand| {
                !matches!(operand, Operand::Register(register)
                    if register.class == RegisterClass::Scalar && register.count == 2)
                    && !matches!(operand, Operand::Integer(value) if (-16..=64).contains(value))
            }) {
                return Err(EvalError::Unsupported {
                    pc: wave.pc,
                    what: format!(
                        "selected {} s_cmp_{{eq,lg}}_u64 requires SGPR pairs or integer inline constants -16..64",
                        self.architecture.name()
                    ),
                });
            }
            let a = wave.wide(&o[0], 0)?;
            let b = wave.wide(&o[1], 0)?;
            wave.set_scc(if relation == Relation::Eq {
                a == b
            } else {
                a != b
            });
            return Ok(());
        }
        if let AluOperation::ScalarCompare {
            relation,
            ty,
            immediate: false,
        } = operation
        {
            let a = wave.bits(&o[0], 0)?;
            let b = wave.bits(&o[1], 0)?;
            let signed = ty == CompareType::I32;
            let result = compare(
                relation,
                if signed { a as i32 as i64 } else { a as i64 },
                if signed { b as i32 as i64 } else { b as i64 },
            )
            .ok_or_else(|| self.unsupported(wave, instruction))?;
            wave.set_scc(result);
            return Ok(());
        }
        let dest = &o[0];
        if operation == AluOperation::SSextI32I16 {
            // CDNA4 SOP1 observes only low16 and preserves SCC.
            let value = wave
                .value(&o[1], 0)?
                .selected_bits(0, 16)
                .map_err(|error| wave.fail(None, error))? as i16 as i32;
            return wave.put_value(dest, 0, Value::SignedInteger(value));
        }
        if operation == AluOperation::SBitset1B32 {
            let value = wave.bits(dest, 0)? | (1u32 << (wave.bits(&o[1], 0)? & 31));
            return wave.put_value(dest, 0, Value::Integer(value));
        }
        if operation == AluOperation::SAddkI32 {
            let a = wave.bits(dest, 0)? as i32;
            let b = wave.bits(&o[1], 0)? as i16 as i32;
            let (value, overflow) = a.overflowing_add(b);
            wave.set_scc(overflow);
            return wave.put_value(dest, 0, Value::SignedInteger(value));
        }
        if operation == AluOperation::SBrevB32 {
            let value = wave.bits(&o[1], 0)?.reverse_bits();
            return wave.put_value(dest, 0, Value::Integer(value));
        }
        if operation == AluOperation::SFf1I32B32 {
            let bits = wave.bits(&o[1], 0)?;
            let first = if bits == 0 {
                -1
            } else {
                bits.trailing_zeros() as i32
            };
            return wave.put_value(dest, 0, Value::SignedInteger(first));
        }
        if operation == AluOperation::SFf1I32B64 {
            let source = wave.wide(&o[1], 0)?;
            let first = if source == 0 {
                -1
            } else {
                source.trailing_zeros() as i32
            };
            return wave.put_value(dest, 0, Value::SignedInteger(first));
        }
        if operation == AluOperation::SMovkI32 {
            let value = wave.bits(&o[1], 0)? as i16 as i32 as u32;
            return wave.put_value(dest, 0, Value::Integer(value));
        }
        if operation == AluOperation::SMulkI32 {
            let old = wave.bits(dest, 0)?;
            let multiplier = wave.bits(&o[1], 0)? as i16 as i32 as u32;
            return wave.put_value(dest, 0, Value::Integer(old.wrapping_mul(multiplier)));
        }
        if operation == AluOperation::SMovB32 {
            if self.private.extent.is_some()
                && let (Operand::Register(destination), Operand::Register(source)) = (dest, &o[1])
                && destination.class == RegisterClass::Scalar
                && source.class == RegisterClass::Scalar
                && destination.count == 1
                && source.count == 1
            {
                let value = wave.spill_word(*source, 0)?;
                return wave.put(*destination, 0, 0, value);
            }
            let value = wave.value(&o[1], 0)?;
            return wave.put_value(dest, 0, value);
        }
        if operation == AluOperation::SCselectB32
            && self.architecture == AmdArchitecture::Gfx1250
            && let Some(mut condition) = wave.scc_condition
        {
            let a = wave.value(&o[1], 0)?;
            let b = wave.value(&o[2], 0)?;
            if matches!(
                (a.bits(), b.bits()),
                (Ok(u32::MAX), Ok(0)) | (Ok(0), Ok(u32::MAX))
            ) {
                if a.bits() == Ok(0) {
                    condition = self.arena.not(condition);
                }
                return self.write_predicates(wave, Some(dest), [condition; 64]);
            }
            let ty = super::super::scalar::IntType {
                bits: 32,
                signed: false,
            };
            let a = self.integer_expression(wave, &o[1], 0, ty)?;
            let b = self.integer_expression(wave, &o[2], 0, ty)?;
            let expr = super::super::scalar::select(
                &mut self.arena,
                super::super::scalar::ScalarType::Int(ty),
                condition,
                a,
                b,
            )
            .map_err(|e| wave.fail(None, e))?;
            return wave.put_value(dest, 0, Value::IntExpr(expr));
        }
        if let AluOperation::Move {
            domain: AluDomain::Scalar,
            kind: MoveKind::SelectScc,
            width,
        } = operation
        {
            let source = if wave.read_scc()? { &o[1] } else { &o[2] };
            return if width == BitWidth::B64 {
                wave.move_wide(dest, source)
            } else {
                let value = wave.value(source, 0)?;
                wave.put_value(dest, 0, value)
            };
        }
        if operation == AluOperation::SAndB64 && self.architecture == AmdArchitecture::Gfx1250 {
            if o[1..].iter().any(|operand| {
                !matches!(operand, Operand::Register(register) if register.count == 2)
                    && !matches!(operand, Operand::Integer(value) if (-16..=64).contains(value))
            }) {
                return Err(EvalError::Unsupported {
                    pc: wave.pc,
                    what: "selected gfx1250 s_and_b64 requires register pairs or integer inline constants -16..64".into(),
                });
            }
            // Read both words before writing an aliased destination. A zero
            // mask determines its result without observing discarded bits.
            let mut result = [(0, 0); 2];
            for (word, output) in result.iter_mut().enumerate() {
                let (a, da) = wave.scalar_bitword(&o[1], word as u16)?;
                let (b, db) = wave.scalar_bitword(&o[2], word as u16)?;
                let defined = (da & db) | (da & !a) | (db & !b);
                *output = (a & b & defined, defined);
            }
            if result.iter().any(|&(bits, _)| bits != 0) {
                wave.set_scc(true);
            } else if result.iter().all(|&(_, defined)| defined == u32::MAX) {
                wave.set_scc(false);
            } else {
                wave.clear_scc();
            }
            let destination = wave.destination(dest)?;
            for (word, (bits, defined)) in result.into_iter().enumerate() {
                wave.put(
                    destination,
                    0,
                    word as u16,
                    Value::partial_bits(bits, defined),
                )?;
            }
            return Ok(());
        }
        if let AluOperation::Bitwise {
            domain: AluDomain::Scalar,
            kind,
            width: BitWidth::B64,
            save_exec,
        } = operation
        {
            let a = wave.wide(&o[1], 0)?;
            let b = o
                .get(2)
                .map(|source| wave.wide(source, 0))
                .transpose()?
                .unwrap_or(0);
            let old_exec = wave.exec;
            let value = if save_exec {
                wave.exec = match kind {
                    BitwiseKind::And => old_exec & a,
                    BitwiseKind::Or => old_exec | a,
                    BitwiseKind::Xor => old_exec ^ a,
                    // The explicit S0 is first; implicit EXEC is second.
                    BitwiseKind::AndNot => a & !old_exec,
                    _ => return Err(self.unsupported(wave, instruction)),
                };
                old_exec
            } else {
                match kind {
                    BitwiseKind::And => a & b,
                    BitwiseKind::AndNot => a & !b,
                    BitwiseKind::OrNot => a | !b,
                    BitwiseKind::Not => !a,
                    _ => return Err(self.unsupported(wave, instruction)),
                }
            };
            wave.set_scc(if save_exec {
                wave.exec != 0
            } else {
                value != 0
            });
            if self.architecture == AmdArchitecture::Gfx1250 && !save_exec {
                let dest = wave.destination(dest)?;
                wave.put(dest, 0, 0, Value::Integer(value as u32))?;
                return wave.put(dest, 0, 1, Value::Integer((value >> 32) as u32));
            }
            return wave.put_wide(dest, 0, value);
        }
        if let AluOperation::Bitwise {
            domain: AluDomain::Scalar,
            kind: kind @ (BitwiseKind::And | BitwiseKind::AndNot | BitwiseKind::Or),
            width: BitWidth::B32,
            save_exec: true,
        } = operation
        {
            let source = wave.bits(&o[1], 0)? as u64;
            let old = wave.exec;
            wave.exec = match kind {
                BitwiseKind::And => old & source,
                BitwiseKind::AndNot => source & !old,
                BitwiseKind::Or => old | source,
                _ => unreachable!("selected scalar32 SAVEEXEC family"),
            };
            wave.set_scc(wave.exec != 0);
            return wave.put_value(dest, 0, Value::Bits(old as u32));
        }
        if operation == AluOperation::SAndB32
            && self.architecture != AmdArchitecture::Gfx1250
            && let Value::PartialBits { bits, defined } = wave.value(&o[1], 0)?
        {
            let mask = wave.bits(&o[2], 0)?;
            if mask & !defined != 0 {
                return Err(wave.fail(None, "and observes undefined architected system bits"));
            }
            let result = bits & mask;
            wave.set_scc(result != 0);
            return wave.put_value(dest, 0, Value::Integer(result));
        }
        if operation == AluOperation::SBfeU32 {
            let control = wave.bits(&o[2], 0)?;
            let offset = control & 31;
            let width = (control >> 16) & 127;
            if let Value::PartialBits { bits, defined } = wave.value(&o[1], 0)? {
                if self.architecture == AmdArchitecture::Gfx1250 {
                    let mask = u32::MAX.checked_shr(32 - width.min(32)).unwrap_or(0);
                    let result = (bits >> offset) & mask;
                    let defined = (defined >> offset) | !(u32::MAX >> offset) | !mask;
                    wave.set_partial_scc(result, defined);
                    return wave.put_value(dest, 0, Value::partial_bits(result, defined));
                }
                let value = wave
                    .value(&o[1], 0)?
                    .selected_bits(offset, width)
                    .map_err(|e| wave.fail(None, e))?;
                wave.set_scc(value != 0);
                return wave.put_value(dest, 0, Value::Integer(value));
            }
        }

        if self.architecture == AmdArchitecture::Gfx1250
            && matches!(
                operation,
                AluOperation::SAndB32
                    | AluOperation::SAndn2B32
                    | AluOperation::SOrB32
                    | AluOperation::SNorB32
                    | AluOperation::SXorB32
                    | AluOperation::SNotB32
            )
        {
            let (a, da) = wave.scalar_bitword(&o[1], 0)?;
            let (mut b, db) = if o.len() > 2 {
                wave.scalar_bitword(&o[2], 0)?
            } else {
                (0, u32::MAX)
            };
            if operation == AluOperation::SAndn2B32 {
                b = !b;
            }
            let (bits, defined) = match operation {
                AluOperation::SAndB32 | AluOperation::SAndn2B32 => {
                    (a & b, (da & db) | (da & !a) | (db & !b))
                }
                AluOperation::SOrB32 => (a | b, (da & db) | (da & a) | (db & b)),
                AluOperation::SNorB32 => (!(a | b), (da & db) | (da & a) | (db & b)),
                AluOperation::SXorB32 => (a ^ b, da & db),
                _ => (!a, da),
            };
            if bits & defined != 0 {
                wave.set_scc(true);
            } else if defined == u32::MAX {
                wave.set_scc(false);
            } else {
                wave.clear_scc();
            }
            return wave.put_value(
                dest,
                0,
                if defined == u32::MAX {
                    Value::Integer(bits)
                } else {
                    Value::PartialBits {
                        bits: bits & defined,
                        defined,
                    }
                },
            );
        }
        // LLVM's unclustered prologue calculates a cluster fallback before
        // a concrete CSELECT discards it. Preserve unknown system bits through
        // only these32-bit SALU operations; observing them still fails.
        if self.architecture == AmdArchitecture::Gfx1250
            && matches!(
                operation,
                AluOperation::SAddI32
                    | AluOperation::SAddU32
                    | AluOperation::SSubI32
                    | AluOperation::SSubU32
                    | AluOperation::SMulI32
                    | AluOperation::SLshlB32
                    | AluOperation::SLshrB32
            )
        {
            let a = wave.value(&o[1], 0)?;
            let b = wave.value(&o[2], 0)?;
            if matches!(a, Value::PartialBits { .. }) || matches!(b, Value::PartialBits { .. }) {
                let (bits, defined) =
                    if matches!(operation, AluOperation::SLshlB32 | AluOperation::SLshrB32) {
                        let count = b.bits().map_err(|e| wave.fail(None, e))? & 31;
                        let (bits, defined) = match a {
                            Value::PartialBits { bits, defined } => (bits, defined),
                            value => (value.bits().map_err(|e| wave.fail(None, e))?, u32::MAX),
                        };
                        if operation == AluOperation::SLshlB32 {
                            (bits << count, (defined << count) | !(u32::MAX << count))
                        } else {
                            (bits >> count, (defined >> count) | !(u32::MAX >> count))
                        }
                    } else if operation == AluOperation::SMulI32
                        && (a.bits() == Ok(0) || b.bits() == Ok(0))
                    {
                        (0, u32::MAX)
                    } else {
                        (0, 0)
                    };
                if operation != AluOperation::SMulI32 {
                    wave.set_partial_scc(bits, defined);
                }
                return wave.put_value(dest, 0, Value::partial_bits(bits, defined));
            }
        }
        if self.architecture != AmdArchitecture::Gfx1250 && operation == AluOperation::SAddcU32 {
            let a = wave.value(&o[1], 0)?;
            let b = wave.value(&o[2], 0)?;
            if a == Value::SharedApertureHi || b == Value::SharedApertureHi {
                let plain_word = |operand: &Operand| {
                    matches!(operand, Operand::Register(register)
                        if register.class == RegisterClass::Scalar && register.count == 1)
                };
                let zero = |operand: &Operand, value: Value| {
                    (matches!(operand, Operand::Integer(0)) || plain_word(operand))
                        && matches!(
                            value,
                            Value::Bits(0) | Value::Integer(0) | Value::SignedInteger(0)
                        )
                };
                if !plain_word(&o[0])
                    || !((plain_word(&o[1]) && a == Value::SharedApertureHi && zero(&o[2], b))
                        || (plain_word(&o[2]) && b == Value::SharedApertureHi && zero(&o[1], a)))
                    || wave.read_scc()?
                {
                    return Err(EvalError::Unsupported {
                        pc: wave.pc,
                        what: "selected scalar aperture ADDC requires a plain SGPR destination, one plain SGPR aperture word, concrete zero and carry-in zero".into(),
                    });
                }
                // For every unsigned word H, H+0+0=H with carry-out zero.
                // Snapshot both sources and carry before any aliased write;
                // preserving the tag never exposes a numeric aperture value.
                wave.put_value(&o[0], 0, Value::SharedApertureHi)?;
                wave.set_scc(false);
                return Ok(());
            }
        }
        let a = wave.bits(&o[1], 0)?;
        let b = if o.len() > 2 { wave.bits(&o[2], 0)? } else { 0 };
        let (value, scc) = match operation {
            AluOperation::Integer {
                domain: AluDomain::Scalar,
                kind,
                ty:
                    IntegerType {
                        width: BitWidth::B32,
                        signed,
                    },
                source_bits: 32,
                form,
            } => match (kind, signed, form) {
                (IntegerKind::Add, true, IntegerForm::Plain) => {
                    let (value, overflow) = (a as i32).overflowing_add(b as i32);
                    (value as u32, overflow)
                }
                (IntegerKind::Sub, true, IntegerForm::Plain) => {
                    let (value, overflow) = (a as i32).overflowing_sub(b as i32);
                    (value as u32, overflow)
                }
                (IntegerKind::Add, false, IntegerForm::Plain) => a.overflowing_add(b),
                (IntegerKind::Add, false, IntegerForm::CarryIn) => {
                    let (value, carry) = a.overflowing_add(b);
                    let (value, carry_in) = value.overflowing_add(wave.read_scc()? as u32);
                    (value, carry || carry_in)
                }
                (IntegerKind::Sub, false, IntegerForm::Plain) => a.overflowing_sub(b),
                (IntegerKind::Sub, false, IntegerForm::CarryIn) => {
                    let (value, borrow) = a.overflowing_sub(b);
                    let (value, borrow_in) = value.overflowing_sub(wave.read_scc()? as u32);
                    (value, borrow || borrow_in)
                }
                (IntegerKind::Mul, true, IntegerForm::Plain) => (a.wrapping_mul(b), wave.scc),
                (IntegerKind::MulHigh, false, IntegerForm::Plain) => {
                    (((u64::from(a) * u64::from(b)) >> 32) as u32, wave.scc)
                }
                (IntegerKind::Min, false, IntegerForm::Plain) => (a.min(b), a < b),
                // The m1-based profiles use the strict comparison measured on
                // hardware; every other target and profile keeps `>=`.
                (IntegerKind::Max, false, IntegerForm::Plain) if self.m1_profile() => {
                    (a.max(b), a > b)
                }
                (IntegerKind::Max, true, IntegerForm::Plain) if self.m1_profile() => {
                    ((a as i32).max(b as i32) as u32, (a as i32) > (b as i32))
                }
                (IntegerKind::Max, false, IntegerForm::Plain) => (a.max(b), a >= b),
                (IntegerKind::Min, true, IntegerForm::Plain) => {
                    ((a as i32).min(b as i32) as u32, (a as i32) < (b as i32))
                }
                (IntegerKind::Max, true, IntegerForm::Plain) => {
                    ((a as i32).max(b as i32) as u32, (a as i32) >= (b as i32))
                }
                _ => return Err(self.unsupported(wave, instruction)),
            },
            AluOperation::Bitwise {
                domain: AluDomain::Scalar,
                kind,
                width: BitWidth::B32,
                save_exec: false,
            } => {
                let value = match kind {
                    BitwiseKind::And => a & b,
                    BitwiseKind::Or => a | b,
                    BitwiseKind::Nor => !(a | b),
                    BitwiseKind::Xor => a ^ b,
                    BitwiseKind::AndNot => a & !b,
                    BitwiseKind::Not => !a,
                    _ => return Err(self.unsupported(wave, instruction)),
                };
                (value, value != 0)
            }
            AluOperation::Shift {
                domain: AluDomain::Scalar,
                kind,
                width: BitWidth::B32,
                reverse: false,
            } => {
                let value = match kind {
                    ShiftKind::Left => a.wrapping_shl(b & 31),
                    ShiftKind::LogicalRight => a.wrapping_shr(b & 31),
                    ShiftKind::ArithmeticRight => ((a as i32) >> (b & 31)) as u32,
                };
                (value, value != 0)
            }
            AluOperation::SBfeU32 => {
                let offset = b & 31;
                let width = (b >> 16) & 127;
                let value = if width == 0 {
                    0
                } else {
                    (a >> offset)
                        & if width >= 32 {
                            u32::MAX
                        } else {
                            (1u32 << width) - 1
                        }
                };
                (value, value != 0)
            }
            _ => return Err(self.unsupported(wave, instruction)),
        };
        // Both modeled multiply forms leave SCC unchanged, including symbolic SCC.
        if !matches!(
            operation,
            AluOperation::Integer {
                kind: IntegerKind::Mul | IntegerKind::MulHigh,
                ..
            }
        ) {
            wave.set_scc(scc);
        }
        let signed = matches!(
            operation,
            AluOperation::Integer {
                ty: IntegerType { signed: true, .. },
                ..
            } | AluOperation::Shift {
                kind: ShiftKind::ArithmeticRight,
                ..
            }
        );
        wave.put_value(
            dest,
            0,
            if signed {
                Value::SignedInteger(value as i32)
            } else {
                Value::Integer(value)
            },
        )
    }
    /// CDNA4 SOP2 opcode 40 / CDNA5 SOP2 opcode 41, with a 32-bit control.
    pub(super) fn signed_bitfield64(
        &self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<(), EvalError> {
        let o = instruction.operands();
        let control = wave.bits(&o[2], 0)?;
        let offset = control & 63;
        let width = (control >> 16) & 127;
        if width > 64 {
            return Err(EvalError::Unsupported {
                pc: wave.pc,
                what: "selected s_bfe_i64 profile supports field widths 0 through 64".into(),
            });
        }
        let mut bits = 0u64;
        let mut defined = u64::MAX;
        if width != 0 {
            // Arithmetic right shift repeats source bit 63 beyond the source
            // boundary; sign extension repeats the selected field's top bit.
            // Mapping each output bit also propagates ABI padding exactly.
            let source_bit = |bit: u32| (offset + bit.min(width - 1)).min(63);
            let needed = (0..64).fold(0u64, |mask, bit| mask | (1u64 << source_bit(bit)));
            let (input_bits, input_defined) = match &o[1] {
                Operand::Integer(value) => (*value as u64, u64::MAX),
                Operand::Register(register) => {
                    let mut input_bits = 0u64;
                    let mut input_defined = 0u64;
                    for word in 0..2 {
                        if (needed >> (word * 32)) as u32 == 0 {
                            continue;
                        }
                        let (word_bits, word_defined) =
                            match wave.scalar[register.index as usize + word] {
                                None => (0, 0),
                                Some(Value::PartialBits { bits, defined }) => (bits, defined),
                                Some(value) => (
                                    value.bits().map_err(|error| wave.fail(None, error))?,
                                    u32::MAX,
                                ),
                            };
                        input_bits |= u64::from(word_bits) << (word * 32);
                        input_defined |= u64::from(word_defined) << (word * 32);
                    }
                    (input_bits, input_defined)
                }
                _ => return Err(self.unsupported(wave, instruction)),
            };
            defined = 0;
            for bit in 0..64 {
                let source = 1u64 << source_bit(bit);
                if input_defined & source != 0 {
                    defined |= 1u64 << bit;
                    if input_bits & source != 0 {
                        bits |= 1u64 << bit;
                    }
                }
            }
        }
        if bits != 0 {
            wave.set_scc(true);
        } else if defined == u64::MAX {
            wave.set_scc(false);
        } else {
            wave.clear_scc();
        }
        let destination = wave.destination(&o[0])?;
        for word in 0..2 {
            wave.put(
                destination,
                0,
                word,
                Value::partial_bits(
                    (bits >> (word * 32)) as u32,
                    (defined >> (word * 32)) as u32,
                ),
            )?;
        }
        Ok(())
    }
}

/// The narrow symbolic32 profile uses the same typed arithmetic as the vector
/// evaluator. Carry-dependent and immediate forms keep their dedicated paths.
fn scalar_symbolic_binary(
    operation: AluOperation,
) -> Option<(
    super::super::scalar::IntType,
    super::super::scalar::IntBinOp,
)> {
    use super::super::scalar::{IntBinOp, IntType};
    let (signed, operation) = match operation {
        AluOperation::Integer {
            domain: AluDomain::Scalar,
            kind,
            ty:
                IntegerType {
                    width: BitWidth::B32,
                    signed,
                },
            source_bits: 32,
            form: IntegerForm::Plain,
        } => (
            signed,
            match kind {
                IntegerKind::Add => IntBinOp::Add,
                IntegerKind::Sub => IntBinOp::Sub,
                IntegerKind::Mul if signed => IntBinOp::MulLo,
                _ => return None,
            },
        ),
        AluOperation::Bitwise {
            domain: AluDomain::Scalar,
            kind: kind @ (BitwiseKind::Xor | BitwiseKind::And),
            width: BitWidth::B32,
            save_exec: false,
        } => (
            false,
            match kind {
                BitwiseKind::Xor => IntBinOp::Xor,
                BitwiseKind::And => IntBinOp::And,
                _ => unreachable!("selected scalar symbolic bitwise family"),
            },
        ),
        AluOperation::Shift {
            domain: AluDomain::Scalar,
            kind: ShiftKind::LogicalRight,
            width: BitWidth::B32,
            reverse: false,
        } => (false, IntBinOp::LShr),
        _ => return None,
    };
    Some((IntType { bits: 32, signed }, operation))
}

//! Native AMD matrix ops execution.
use super::*;

impl Interpreter<'_> {
    pub(super) fn matrix_input(
        &mut self,
        wave: &Wave,
        operand: &Operand,
        lane: u8,
        item: usize,
        kind: super::super::matrix::InputKind,
    ) -> Result<ExprId, EvalError> {
        use super::super::matrix::InputKind;
        let per_register = match kind {
            InputKind::F32 => 1,
            InputKind::F16 | InputKind::Bf16 => 2,
            InputKind::Fp8E4m3 | InputKind::Fp8E5m2 => 4,
        };
        let component = item / per_register;
        let selected = if let Operand::Register(register) = operand {
            let mut register = *register;
            register.index += component as u16;
            register.count = 1;
            Operand::Register(register)
        } else {
            operand.clone()
        };
        match kind {
            InputKind::F32 => self.real(wave, &selected, lane),
            InputKind::F16 => self.half(wave, &selected, lane, !item.is_multiple_of(2)),
            InputKind::Fp8E4m3 => self.fp8(wave, &selected, lane, item % 4),
            InputKind::Bf16 => {
                let high = !item.is_multiple_of(2);
                let symbolic = match wave.value(&selected, lane)? {
                    Value::Half(expr) if !high => Some(expr),
                    Value::Half(_) if high => Some(self.arena.int(0)),
                    Value::Halves(lo, hi) => Some(if high { hi } else { lo }),
                    _ => None,
                };
                if let Some(expr) = symbolic {
                    if !matches!(self.arena.node(expr), ExprNode::InputElement { .. })
                        && self.arena.as_f64(expr) != Some(0.0)
                    {
                        return Err(EvalError::Unsupported {
                            pc: wave.pc,
                            what: "produced F16 cannot be reinterpreted as symbolic BF16".into(),
                        });
                    }
                    self.claim_half_format(wave, expr, WordFormat::Bfloat16)?;
                    return Ok(expr);
                }
                let bits = wave.bits(&selected, lane)? >> (16 * (item % 2));
                let value = super::super::matrix::decode_bits(kind, bits)
                    .map_err(|what| EvalError::Unsupported { pc: wave.pc, what })?;
                self.arena
                    .float_from_f64(value)
                    .map_err(|e| wave.fail(Some(lane), e.to_string()))
            }
            InputKind::Fp8E5m2 => {
                let bits =
                    wave.bits(&selected, lane)? >> (8 * (item % per_register) * (4 / per_register));
                let value = super::super::matrix::decode_bits(kind, bits)
                    .map_err(|what| EvalError::Unsupported { pc: wave.pc, what })?;
                self.arena
                    .float_from_f64(value)
                    .map_err(|e| wave.fail(Some(lane), e.to_string()))
            }
        }
    }
    pub(super) fn wmma(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<(), EvalError> {
        use super::super::matrix::InputKind;
        // gfx1250-b0-wmma/3 only: BF8 combinations, F8F6F4, F4 32x16x128 and
        // the block-scaled forms (their modifiers are checked there).
        // gfx1250-b0-wmma/4 only: 16-bit-output WMMA and SWMMAC.
        if self.config.native_profile.b0_wmma_sparse16()
            && self.architecture == AmdArchitecture::Gfx1250
            && wave.valid == 0xffff_ffff
            && wave.exec == 0xffff_ffff
            && let Operation::Matrix(matrix) = instruction.operation()
            && let Some(form) = HalfSparse::of(matrix.operation)
        {
            return self.wmma_half_sparse(wave, instruction, form);
        }
        if self.config.native_profile.b0_wmma_scale()
            && self.architecture == AmdArchitecture::Gfx1250
            && wave.valid == 0xffff_ffff
            && wave.exec == 0xffff_ffff
            && let Operation::Matrix(matrix) = instruction.operation()
            && let Some(form) = SmallWmma::of(matrix.operation)
        {
            return self.wmma_small(wave, instruction, form);
        }
        if self.architecture != AmdArchitecture::Gfx1250
            || wave.valid != 0xffff_ffff
            || wave.exec != 0xffff_ffff
            || !instruction.modifiers().is_empty()
        {
            return Err(EvalError::Unsupported {
                pc: wave.pc,
                what: "WMMA requires one complete active wave32 and supported unmodified operands"
                    .into(),
            });
        }
        let Operation::Matrix(matrix) = instruction.operation() else {
            unreachable!("typed matrix dispatch")
        };
        // gfx1250-b0-wmma/2 only: the FP8 and IU8 forms (uniform tiles, below).
        let types = self.config.native_profile.b0_wmma_types();
        let (kind, input_kind, k) = match matrix.operation {
            MatrixOpcode::VWmmaF3216x16x4F32 => (AmdWmmaKind::F32_16x16x4, InputKind::F32, 4),
            MatrixOpcode::VWmmaF3216x16x32F16 => (AmdWmmaKind::F16_16x16x32, InputKind::F16, 32),
            MatrixOpcode::VWmmaF3216x16x32Bf16 => (AmdWmmaKind::Bf16_16x16x32, InputKind::Bf16, 32),
            MatrixOpcode::VWmmaF3216x16x64Fp8Fp8 if types => {
                (AmdWmmaKind::Fp8Fp8_16x16x64, InputKind::Fp8E4m3, 64)
            }
            MatrixOpcode::VWmmaF3216x16x128Fp8Fp8 if types => {
                (AmdWmmaKind::Fp8Fp8_16x16x128, InputKind::Fp8E4m3, 128)
            }
            // The input kind is unused: IU8 is evaluated as integers.
            MatrixOpcode::VWmmaI3216x16x64Iu8 if types => {
                (AmdWmmaKind::Iu8_16x16x64, InputKind::Fp8E4m3, 64)
            }
            _ => return Err(self.unsupported(wave, instruction)),
        };
        let o = instruction.operands();
        let (_, destinations) = self.instruction_registers(instruction, wave);
        let sources = [o[0].clone(), o[1].clone(), o[2].clone()];
        let (sources_ab, _) = self.instruction_registers_for_operands(instruction, &sources, wave);
        let sources = [o[0].clone(), o[3].clone()];
        let (accumulator, _) = self.instruction_registers_for_operands(instruction, &sources, wave);
        self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::WmmaIssue {
                kind,
                destinations,
                sources_ab,
                accumulator,
                scales: Vec::new(),
                index: Vec::new(),
            },
        );
        if matches!(
            kind,
            AmdWmmaKind::Fp8Fp8_16x16x64
                | AmdWmmaKind::Fp8Fp8_16x16x128
                | AmdWmmaKind::Iu8_16x16x64
        ) {
            return self.wmma_uniform8(wave, instruction, kind, k);
        }
        let mut a = vec![Vec::new(); 32];
        let mut b = vec![Vec::new(); 32];
        let mut c = vec![Vec::new(); 32];
        // gfx1250-b0-wmma/3: a result is certified only from C words that are
        // themselves concrete bits (not reals left by an uncertified WMMA).
        let strict = self.config.native_profile.b0_wmma_scale();
        let mut c_bits = vec![Vec::new(); 32];
        for lane in 0..32u8 {
            for item in 0..k / 2 {
                a[lane as usize].push(self.matrix_input(wave, &o[1], lane, item, input_kind)?);
                b[lane as usize].push(self.matrix_input(wave, &o[2], lane, item, input_kind)?);
            }
            for component in 0..8 {
                let value = if let Operand::Register(register) = o[3] {
                    let value = wave.reg(register, lane, component)?;
                    c_bits[lane as usize].push(value.bits().is_ok());
                    self.real_value(wave, value, lane)?
                } else {
                    c_bits[lane as usize].push(true);
                    self.real(wave, &o[3], lane)?
                };
                c[lane as usize].push(value);
            }
        }
        if a.iter().chain(&b).chain(&c).flatten().any(|&value| {
            self.arena
                .as_f64(value)
                .is_some_and(|value| !value.is_finite())
        }) {
            return Err(wave.fail(None, "WMMA requires finite concrete operands"));
        }
        // Public FlyDSL f37f7a2 MmaAtom.cpp: A/B are column-major
        // (row,K), while C/D row=(lane/16)*8+register,col=lane%16.
        let input = |row: usize, kk: usize| {
            if k == 4 {
                (row + (kk / 2) * 16, kk % 2)
            } else {
                (row + ((kk / 8) % 2) * 16, (kk / 16) * 8 + kk % 8)
            }
        };
        // B0 WMMA profiles: BF16 results in the exact-representable regime
        // are stored as their exact binary32 bits; the rest stay reals.
        // gfx1250-b0-wmma/2 also certifies F16 inputs (normal or zero in F16).
        let certify = self.wmma_profile()
            && (input_kind == InputKind::Bf16
                || (input_kind == InputKind::F16 && self.config.native_profile.b0_wmma_types()));
        let min_normal_exp = if input_kind == InputKind::F16 {
            -14
        } else {
            -126
        };
        let destination = wave.destination(&o[0])?;
        for (lane, accumulator) in c.iter().enumerate() {
            for (component, &initial) in accumulator.iter().enumerate() {
                let row = (lane / 16) * 8 + component;
                let col = lane % 16;
                let mut value = initial;
                let mut factors = Vec::new();
                for kk in 0..k {
                    let (al, av) = input(row, kk);
                    let (bl, bv) = input(col, kk);
                    value = self.arena.fma(a[al][av], b[bl][bv], value);
                    factors.push((a[al][av], b[bl][bv]));
                }
                let bits = if certify && (!strict || c_bits[lane][component]) {
                    self.exact_wmma_bits(initial, &factors, min_normal_exp)
                } else {
                    None
                };
                wave.put(
                    destination,
                    lane as u8,
                    component as u16,
                    bits.map_or(Value::Real(value), Value::Bits),
                )?;
            }
        }
        Ok(())
    }
    fn const_rational(&self, expr: ExprId) -> Option<rug::Rational> {
        match self.arena.node(expr) {
            ExprNode::RealConst(real) => match real.repr() {
                volta_expr::RealRepr::Rational(q) => Some(q.clone()),
                _ => None,
            },
            ExprNode::IntConst(value) => Some(rug::Rational::from(*value)),
            _ => None,
        }
    }
    /// Exact bits of `c + sum(a * b)` when every input is a concrete normal
    /// (or zero) value and the element is in the exact-representable regime.
    /// `min_normal_exp`: the smallest normal exponent of the input format
    /// (-126 for BF16, -14 for F16); smaller nonzero inputs are refused.
    fn exact_wmma_bits(
        &self,
        accumulator: ExprId,
        factors: &[(ExprId, ExprId)],
        min_normal_exp: i64,
    ) -> Option<u32> {
        use super::super::wmma_exact::{exact_binary32, normal_or_zero, normal_or_zero_from};
        let accumulator = self.const_rational(accumulator)?;
        if !normal_or_zero(&accumulator) {
            return None;
        }
        let mut products = Vec::with_capacity(factors.len());
        for &(a, b) in factors {
            let (a, b) = (self.const_rational(a)?, self.const_rational(b)?);
            if !normal_or_zero_from(&a, min_normal_exp) || !normal_or_zero_from(&b, min_normal_exp)
            {
                return None;
            }
            products.push(a * b);
        }
        exact_binary32(&accumulator, &products)
    }
    /// gfx1250-b0-wmma/2 FP8 (E4M3) and IU8 WMMA. Only uniform tiles are
    /// evaluated: every byte of every A register in every lane equal, and the
    /// same for B. Then every D element is `C + K * a * b` whatever the 8-bit
    /// matrix layout, so no layout table is needed. FP8 results carry exact
    /// bits in the exact-representable regime (inputs normal in E4M3), else
    /// they stay reals. IU8 (no NEG modifiers: unsigned A and B) is exact for
    /// bytes below 0x80 and C + K*a*b inside int32; anything else (other
    /// bytes, overflow, symbolic data) is unsupported, never guessed.
    fn wmma_uniform8(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
        kind: AmdWmmaKind,
        k: usize,
    ) -> Result<(), EvalError> {
        use super::super::matrix::{InputKind, decode_bits};
        use super::super::wmma_exact::{exact_binary32, normal_or_zero, normal_or_zero_from};
        let o = instruction.operands();
        let integer = kind == AmdWmmaKind::Iu8_16x16x64;
        let width = kind.input_registers() as u16;
        let refuse = |wave: &Wave, what: &str| EvalError::Unsupported {
            pc: wave.pc,
            what: format!("8-bit WMMA {what}"),
        };
        let uniform = |wave: &Wave, operand: &Operand| -> Result<u8, EvalError> {
            let Operand::Register(register) = operand else {
                return Err(refuse(wave, "requires register A and B tuples"));
            };
            let mut byte = None;
            for lane in 0..32u8 {
                for component in 0..width {
                    let mut selected = *register;
                    selected.index += component;
                    selected.count = 1;
                    let bits = wave
                        .bits(&Operand::Register(selected), lane)
                        .map_err(|_| refuse(wave, "requires concrete A and B words"))?;
                    let low = (bits & 0xff) as u8;
                    if bits != u32::from(low) * 0x0101_0101 || byte.is_some_and(|b| b != low) {
                        return Err(refuse(wave, "is evaluated only for uniform A and B tiles"));
                    }
                    byte = Some(low);
                }
            }
            byte.ok_or_else(|| refuse(wave, "has an empty A or B tuple"))
        };
        let (a, b) = (uniform(wave, &o[1])?, uniform(wave, &o[2])?);
        let Operand::Register(c_register) = o[3] else {
            return Err(refuse(wave, "requires a register C tuple"));
        };
        let destination = wave.destination(&o[0])?;
        if integer {
            if a >= 0x80 || b >= 0x80 {
                return Err(refuse(wave, "IU8 bytes at or above 0x80 are not evaluated"));
            }
            let term = k as i64 * i64::from(a) * i64::from(b);
            for lane in 0..32u8 {
                for component in 0..8u16 {
                    let c = wave
                        .reg(c_register, lane, component)?
                        .bits()
                        .map_err(|_| refuse(wave, "IU8 requires concrete C words"))?;
                    let d = i64::from(c as i32) + term;
                    if i32::try_from(d).is_err() {
                        return Err(refuse(wave, "IU8 result leaves int32"));
                    }
                    wave.put(destination, lane, component, Value::Bits(d as i32 as u32))?;
                }
            }
            return Ok(());
        }
        let decode = |wave: &Wave, byte: u8| -> Result<f64, EvalError> {
            let value = decode_bits(InputKind::Fp8E4m3, u32::from(byte))
                .map_err(|what| EvalError::Unsupported { pc: wave.pc, what })?;
            if !value.is_finite() {
                return Err(refuse(wave, "FP8 input is not finite"));
            }
            Ok(value)
        };
        let (av, bv) = (decode(wave, a)?, decode(wave, b)?);
        let rational = |value: f64| rug::Rational::from_f64(value);
        let (ar, br) = (rational(av), rational(bv));
        // E4M3 normal: |x| >= 2^-6.
        let products = match (&ar, &br) {
            (Some(ar), Some(br)) if normal_or_zero_from(ar, -6) && normal_or_zero_from(br, -6) => {
                Some(vec![rug::Rational::from(ar * br); k])
            }
            _ => None,
        };
        let scaled = self
            .arena
            .float_from_f64(k as f64 * av)
            .map_err(|e| wave.fail(None, e.to_string()))?;
        let factor = self
            .arena
            .float_from_f64(bv)
            .map_err(|e| wave.fail(None, e.to_string()))?;
        for lane in 0..32u8 {
            for component in 0..8u16 {
                let value = wave.reg(c_register, lane, component)?;
                // wmma/3: certified only from concrete C bits (see `wmma`).
                let certified_c =
                    !self.config.native_profile.b0_wmma_scale() || value.bits().is_ok();
                let initial = self.real_value(wave, value, lane)?;
                let bits = products
                    .as_ref()
                    .filter(|_| certified_c)
                    .and_then(|products| {
                        let accumulator = self.const_rational(initial)?;
                        normal_or_zero(&accumulator)
                            .then(|| exact_binary32(&accumulator, products))
                            .flatten()
                    });
                let sum = self.arena.fma(scaled, factor, initial);
                wave.put(
                    destination,
                    lane,
                    component,
                    bits.map_or(Value::Real(sum), Value::Bits),
                )?;
            }
        }
        Ok(())
    }
    /// gfx1250-b0-wmma/3: the BF8 combinations, F8F6F4 (formats from the
    /// format modifiers), F4 32x16x128 and their 4-dword block-scaled forms.
    /// Only uniform tiles are evaluated: every element of every A register in
    /// every lane equal (8-bit: one byte, 4-bit: one nibble, 6-bit: one code
    /// packed contiguously over each lane's six-register group), the same for
    /// B, and every byte of each scale tuple equal. Then every D element is
    /// `C + K * a * b * sa * sb` whatever the matrix and scale layouts. Exact
    /// bits only when the elements are normal or zero in their own format, the
    /// scales are normal powers of two and the exact-representable regime
    /// holds; otherwise the D words are reals. Anything else (non-uniform or
    /// symbolic tiles, non-VGPR scales, reuse hints, NaN or Inf inputs) is
    /// unsupported, never guessed.
    fn wmma_small(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
        form: SmallWmma,
    ) -> Result<(), EvalError> {
        use super::super::wmma_exact::{exact_binary32, normal_or_zero};
        use volta_graph::AmdSmallFloat as F;
        let refuse = |wave: &Wave, what: &str| EvalError::Unsupported {
            pc: wave.pc,
            what: format!("WMMA {what}"),
        };
        let modifier = |name: &str| {
            instruction
                .modifiers()
                .iter()
                .find(|modifier| modifier.name == name)
                .and_then(|modifier| modifier.value.as_deref())
        };
        for modifier in instruction.modifiers() {
            let known = match modifier.name.as_str() {
                "matrix_a_fmt" | "matrix_b_fmt" => form.f8f6f4,
                "matrix_a_scale" | "matrix_b_scale" | "matrix_a_scale_fmt"
                | "matrix_b_scale_fmt" => form.scale.is_some(),
                _ => false,
            };
            if !known {
                return Err(refuse(
                    wave,
                    &format!("modifier `{}` is not evaluated", modifier.name),
                ));
            }
        }
        let format = |wave: &Wave, name: &str, default: F| -> Result<F, EvalError> {
            Ok(match modifier(name) {
                None => default,
                Some("MATRIX_FMT_FP8") => F::Fp8,
                Some("MATRIX_FMT_BF8") => F::Bf8,
                Some("MATRIX_FMT_FP6") => F::Fp6,
                Some("MATRIX_FMT_BF6") => F::Bf6,
                Some("MATRIX_FMT_FP4") => F::Fp4,
                Some(other) => return Err(refuse(wave, &format!("format `{other}`"))),
            })
        };
        let (a_format, b_format) = if form.f8f6f4 {
            (
                format(wave, "matrix_a_fmt", F::Fp8)?,
                format(wave, "matrix_b_fmt", F::Fp8)?,
            )
        } else {
            (form.a, form.b)
        };
        let kind = if form.f8f6f4 {
            AmdWmmaKind::F8f6f4_16x16x128 {
                a: a_format,
                b: b_format,
            }
        } else {
            form.kind
        };
        let scale_format = |wave: &Wave, name: &str| -> Result<SmallScale, EvalError> {
            Ok(match modifier(name) {
                None | Some("MATRIX_SCALE_FMT_E8") => SmallScale::E8m0,
                Some("MATRIX_SCALE_FMT_E5M3") => SmallScale::E5m3,
                Some("MATRIX_SCALE_FMT_E4M3") => SmallScale::E4m3,
                Some(other) => return Err(refuse(wave, &format!("scale format `{other}`"))),
            })
        };
        let o = instruction.operands();
        let (_, destinations) = self.instruction_registers(instruction, wave);
        let sources = [o[0].clone(), o[1].clone(), o[2].clone()];
        let (sources_ab, _) = self.instruction_registers_for_operands(instruction, &sources, wave);
        let sources = [o[0].clone(), o[3].clone()];
        let (accumulator, _) = self.instruction_registers_for_operands(instruction, &sources, wave);
        let scales = if form.scale.is_some() {
            let sources = [o[0].clone(), o[4].clone(), o[5].clone()];
            self.instruction_registers_for_operands(instruction, &sources, wave)
                .0
        } else {
            Vec::new()
        };
        self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::WmmaIssue {
                kind,
                destinations,
                sources_ab,
                accumulator,
                scales,
                index: Vec::new(),
            },
        );
        // Every word of every register of a tuple, in every lane.
        let words =
            |wave: &Wave, operand: &Operand, what: &str| -> Result<Vec<Vec<u32>>, EvalError> {
                let Operand::Register(register) = operand else {
                    return Err(refuse(wave, &format!("requires a VGPR {what} tuple")));
                };
                if register.class != RegisterClass::Vector {
                    return Err(refuse(wave, &format!("requires a VGPR {what} tuple")));
                }
                let mut out = Vec::with_capacity(32);
                for lane in 0..32u8 {
                    let mut lane_words = Vec::with_capacity(register.count as usize);
                    for component in 0..register.count {
                        let mut selected = *register;
                        selected.index += component;
                        selected.count = 1;
                        lane_words.push(wave.bits(&Operand::Register(selected), lane).map_err(
                            |_| refuse(wave, &format!("requires concrete {what} words")),
                        )?);
                    }
                    out.push(lane_words);
                }
                Ok(out)
            };
        let uniform =
            |wave: &Wave, tuple: &[Vec<u32>], format: F, what: &str| -> Result<u32, EvalError> {
                let not_uniform = || {
                    refuse(
                        wave,
                        &format!("is evaluated only for a uniform {what} tile"),
                    )
                };
                let first = tuple
                    .first()
                    .and_then(|lane| lane.first())
                    .copied()
                    .ok_or_else(not_uniform)?;
                let code = match format {
                    F::Fp8 | F::Bf8 => first & 0xff,
                    F::Fp6 | F::Bf6 => first & 0x3f,
                    F::Fp4 => first & 0xf,
                };
                let expected: Vec<u32> = match format {
                    F::Fp8 | F::Bf8 => vec![code * 0x0101_0101],
                    F::Fp4 => vec![code * 0x1111_1111],
                    F::Fp6 => pack6(code).to_vec(),
                    F::Bf6 => pack6(code).to_vec(),
                };
                for lane in tuple {
                    if lane.len() % expected.len() != 0
                        || lane
                            .iter()
                            .enumerate()
                            .any(|(i, &w)| w != expected[i % expected.len()])
                    {
                        return Err(not_uniform());
                    }
                }
                Ok(code)
            };
        let a_words = words(wave, &o[1], "A")?;
        let b_words = words(wave, &o[2], "B")?;
        let a_code = uniform(wave, &a_words, a_format, "A")?;
        let b_code = uniform(wave, &b_words, b_format, "B")?;
        let (av, a_normal) =
            small_float(a_format, a_code).ok_or_else(|| refuse(wave, "input is not finite"))?;
        let (bv, b_normal) =
            small_float(b_format, b_code).ok_or_else(|| refuse(wave, "input is not finite"))?;
        let (mut sa, mut sb, mut scales_exact) = (1.0, 1.0, true);
        if form.scale.is_some() {
            for (index, name, value) in [
                (4, "matrix_a_scale_fmt", &mut sa),
                (5, "matrix_b_scale_fmt", &mut sb),
            ] {
                let scale = scale_format(wave, name)?;
                let tuple = words(wave, &o[index], "scale")?;
                let byte = uniform(wave, &tuple, F::Fp8, "scale")?;
                let (v, power_of_two) = small_scale(scale, byte).ok_or_else(|| {
                    refuse(wave, "scale is NaN or outside the evaluated encodings")
                })?;
                *value = v;
                scales_exact &= power_of_two;
            }
        }
        let Operand::Register(c_register) = o[3] else {
            return Err(refuse(wave, "requires a register C tuple"));
        };
        let k = form.k;
        let product = av * bv * sa * sb;
        let products = (a_normal && b_normal && scales_exact)
            .then(|| rug::Rational::from_f64(product))
            .flatten()
            .map(|p| vec![p; k]);
        // Every value here is a small dyadic, so the f64 factors are exact.
        let scaled = self
            .arena
            .float_from_f64(k as f64 * av * sa * sb)
            .map_err(|e| wave.fail(None, e.to_string()))?;
        let factor = self
            .arena
            .float_from_f64(bv)
            .map_err(|e| wave.fail(None, e.to_string()))?;
        let rows = kind.destination_registers() as u16;
        let destination = wave.destination(&o[0])?;
        for lane in 0..32u8 {
            for component in 0..rows {
                let value = wave.reg(c_register, lane, component)?;
                // wmma/3: certified only from concrete C bits (see `wmma`).
                let certified_c =
                    !self.config.native_profile.b0_wmma_scale() || value.bits().is_ok();
                let initial = self.real_value(wave, value, lane)?;
                let bits = products
                    .as_ref()
                    .filter(|_| certified_c)
                    .and_then(|products| {
                        let accumulator = self.const_rational(initial)?;
                        normal_or_zero(&accumulator)
                            .then(|| exact_binary32(&accumulator, products))
                            .flatten()
                    });
                let sum = self.arena.fma(scaled, factor, initial);
                wave.put(
                    destination,
                    lane,
                    component,
                    bits.map_or(Value::Real(sum), Value::Bits),
                )?;
            }
        }
        Ok(())
    }
    /// gfx1250-b0-wmma/4: 16-bit-output WMMA and SWMMAC, uniform tiles only.
    /// Dense: D = C + K*a*b. SWMMAC (C is D): the packed A is uniform, B is
    /// uniform or half-uniform (registers 0-7 one value b0, 8-15 another b1:
    /// the two K halves of the sparse B layout) and the index words are
    /// uniform with idx0 < idx1 in every pair, so each K half contributes K/4
    /// products: D = C + (K/4)*a*(b0+b1) whatever the index values. F32 and
    /// F16 results carry exact bits in their regimes from concrete C bits;
    /// BF16 and BF16F32 results are never certified (reals). Anything else is
    /// unsupported.
    fn wmma_half_sparse(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
        form: HalfSparse,
    ) -> Result<(), EvalError> {
        use super::super::wmma_exact::{exact_binary16, exact_binary32, f16_value, normal_or_zero};
        let refuse = |wave: &Wave, what: &str| EvalError::Unsupported {
            pc: wave.pc,
            what: format!("WMMA {what}"),
        };
        if !instruction.modifiers().is_empty() {
            return Err(refuse(
                wave,
                "modifiers are not evaluated for 16-bit outputs or SWMMAC",
            ));
        }
        let o = instruction.operands();
        let (_, destinations) = self.instruction_registers(instruction, wave);
        // Operand role tuples straight from the operands (SWMMAC also reads its
        // destination, which must not leak into the A/B or index roles).
        let vgprs = |operand: &Operand| -> Vec<AmdRegister> {
            match operand {
                Operand::Register(register) if register.class == RegisterClass::Vector => (0
                    ..register.count)
                    .map(|component| AmdRegister {
                        file: AmdRegisterFile::Vector,
                        index: register.index + component,
                        lanes: wave.exec,
                    })
                    .collect(),
                _ => Vec::new(),
            }
        };
        let mut sources_ab = vgprs(&o[1]);
        sources_ab.extend(vgprs(&o[2]));
        let (accumulator, index) = if form.sparse {
            (destinations.clone(), vgprs(&o[3]))
        } else {
            (vgprs(&o[3]), Vec::new())
        };
        self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::WmmaIssue {
                kind: form.kind,
                destinations,
                sources_ab,
                accumulator,
                scales: Vec::new(),
                index,
            },
        );
        // Every word of every register of a VGPR tuple, in every lane.
        let words =
            |wave: &Wave, operand: &Operand, what: &str| -> Result<Vec<Vec<u32>>, EvalError> {
                let Operand::Register(register) = operand else {
                    return Err(refuse(wave, &format!("requires a VGPR {what} tuple")));
                };
                if register.class != RegisterClass::Vector {
                    return Err(refuse(wave, &format!("requires a VGPR {what} tuple")));
                }
                let mut out = Vec::with_capacity(32);
                for lane in 0..32u8 {
                    let mut lane_words = Vec::with_capacity(register.count as usize);
                    for component in 0..register.count {
                        let mut selected = *register;
                        selected.index += component;
                        selected.count = 1;
                        lane_words.push(wave.bits(&Operand::Register(selected), lane).map_err(
                            |_| refuse(wave, &format!("requires concrete {what} words")),
                        )?);
                    }
                    out.push(lane_words);
                }
                Ok(out)
            };
        // The one element value of a uniform run of registers (all lanes).
        let element = |wave: &Wave,
                       tuple: &[Vec<u32>],
                       range: std::ops::Range<usize>,
                       format: HalfInput,
                       what: &str|
         -> Result<(f64, bool), EvalError> {
            let not_uniform = || {
                refuse(
                    wave,
                    &format!("is evaluated only for a uniform {what} tile"),
                )
            };
            let first = tuple
                .first()
                .and_then(|lane| lane.get(range.start))
                .copied()
                .ok_or_else(not_uniform)?;
            let repeated = match format {
                HalfInput::F16 | HalfInput::Bf16 => (first & 0xffff) * 0x0001_0001,
                HalfInput::Iu8 | HalfInput::Small(_) => (first & 0xff) * 0x0101_0101,
            };
            if tuple.iter().any(|lane| {
                lane.get(range.clone())
                    .is_none_or(|words| words.iter().any(|&w| w != repeated))
            }) {
                return Err(not_uniform());
            }
            let value = match format {
                HalfInput::F16 => f16_value((first & 0xffff) as u16),
                HalfInput::Bf16 => {
                    let v = f64::from(f32::from_bits((first & 0xffff) << 16));
                    v.is_finite()
                        .then_some((v, v == 0.0 || v.abs() >= f64::from(f32::MIN_POSITIVE)))
                }
                HalfInput::Iu8 => (first & 0xff < 0x80).then_some((f64::from(first & 0xff), true)),
                HalfInput::Small(small) => small_float(small, first & 0xff),
            };
            value.ok_or_else(|| {
                refuse(
                    wave,
                    &format!("{what} element is outside the evaluated encodings"),
                )
            })
        };
        let a_words = words(wave, &o[1], "A")?;
        let b_words = words(wave, &o[2], "B")?;
        let (a, a_normal) = element(wave, &a_words, 0..a_words[0].len(), form.a, "A")?;
        let (b0, b1, b_normal) = if form.sparse {
            let (b0, n0) = element(wave, &b_words, 0..8, form.b, "B (first half)")?;
            let (b1, n1) = element(wave, &b_words, 8..16, form.b, "B (second half)")?;
            (b0, b1, n0 && n1)
        } else {
            let (b, n) = element(wave, &b_words, 0..b_words[0].len(), form.b, "B")?;
            (b, b, n)
        };
        if form.sparse {
            let index = words(wave, &o[3], "index")?;
            let word = index[0][0];
            let valid = (0..8).all(|i| (word >> (4 * i)) & 3 < (word >> (4 * i + 2)) & 3);
            if !valid || index.iter().any(|lane| lane.iter().any(|&w| w != word)) {
                return Err(refuse(
                    wave,
                    "is evaluated only for uniform index words with idx0 < idx1 in every pair",
                ));
            }
        }
        // Per element: `per_half` products a*b0 and a*b1 (dense: K products a*b).
        let per_half = if form.sparse { form.k / 4 } else { form.k / 2 };
        let products = (a_normal && b_normal)
            .then(|| {
                Some((
                    rug::Rational::from_f64(a * b0)?,
                    rug::Rational::from_f64(a * b1)?,
                ))
            })
            .flatten()
            .map(|(p0, p1)| {
                let mut v = vec![p0; per_half];
                v.extend(std::iter::repeat_n(p1, per_half));
                v
            });
        let term = per_half as f64 * a * (b0 + b1);
        let scaled = self
            .arena
            .float_from_f64(per_half as f64 * a)
            .map_err(|e| wave.fail(None, e.to_string()))?;
        let factor = self
            .arena
            .float_from_f64(b0 + b1)
            .map_err(|e| wave.fail(None, e.to_string()))?;
        let c_operand = if form.sparse { &o[0] } else { &o[3] };
        let Operand::Register(c_register) = *c_operand else {
            return Err(refuse(wave, "requires a register C tuple"));
        };
        let destination = wave.destination(&o[0])?;
        let rows = form.kind.destination_registers() as u16;
        for lane in 0..32u8 {
            for component in 0..rows {
                let value = match form.output {
                    AmdWmmaOutput::F32 => {
                        let c = wave.reg(c_register, lane, component)?;
                        let concrete = c.bits().is_ok();
                        let initial = self.real_value(wave, c, lane)?;
                        let bits = products.as_ref().filter(|_| concrete).and_then(|products| {
                            let accumulator = self.const_rational(initial)?;
                            normal_or_zero(&accumulator)
                                .then(|| exact_binary32(&accumulator, products))
                                .flatten()
                        });
                        let sum = self.arena.fma(scaled, factor, initial);
                        bits.map_or(Value::Real(sum), Value::Bits)
                    }
                    AmdWmmaOutput::I32 => {
                        let c = wave
                            .reg(c_register, lane, component)?
                            .bits()
                            .map_err(|_| refuse(wave, "IU8 requires concrete C words"))?;
                        let d = i64::from(c as i32) + term as i64;
                        if i32::try_from(d).is_err() {
                            return Err(refuse(wave, "IU8 result leaves int32"));
                        }
                        Value::Bits(d as i32 as u32)
                    }
                    AmdWmmaOutput::Bf16F32 if form.sparse => {
                        // BF16 result in an F32-shaped tuple: never certified.
                        let c = wave.reg(c_register, lane, component)?;
                        let initial = self.real_value(wave, c, lane)?;
                        Value::Real(self.arena.fma(scaled, factor, initial))
                    }
                    AmdWmmaOutput::F16 | AmdWmmaOutput::Bf16 | AmdWmmaOutput::Bf16F32 => {
                        // Packed halves: register i holds rows 2i (low) and 2i+1
                        // (high); BF16F32 reads them from F32 C registers 2i, 2i+1.
                        let mut halves = [None, None];
                        let mut reals = [None, None];
                        for (h, slot) in halves.iter_mut().enumerate() {
                            let c_value = if form.output == AmdWmmaOutput::Bf16F32 {
                                let c = wave.reg(c_register, lane, 2 * component + h as u16)?;
                                reals[h] = Some(self.real_value(wave, c, lane)?);
                                None
                            } else {
                                match wave.reg(c_register, lane, component)? {
                                    Value::Halves(low, high) => {
                                        reals[h] = Some(if h == 0 { low } else { high });
                                        None
                                    }
                                    c => {
                                        let bits = c.bits().map_err(|_| {
                                            refuse(
                                                wave,
                                                "requires concrete or real 16-bit C halves",
                                            )
                                        })?;
                                        let half = (bits >> (16 * h)) & 0xffff;
                                        let decoded = if form.output == AmdWmmaOutput::F16 {
                                            f16_value(half as u16)
                                        } else {
                                            let v = f64::from(f32::from_bits(half << 16));
                                            v.is_finite().then_some((v, false))
                                        }
                                        .ok_or_else(|| refuse(wave, "C half is not finite"))?;
                                        Some(decoded)
                                    }
                                }
                            };
                            if let Some((c, normal)) = c_value {
                                if form.output == AmdWmmaOutput::F16 && normal {
                                    *slot = products.as_ref().and_then(|products| {
                                        exact_binary16(&rug::Rational::from_f64(c)?, products)
                                    });
                                }
                                if slot.is_none() {
                                    reals[h] = Some(
                                        self.arena
                                            .float_from_f64(c)
                                            .map_err(|e| wave.fail(None, e.to_string()))?,
                                    );
                                }
                            }
                        }
                        if let [Some(low), Some(high)] = halves {
                            Value::Bits(u32::from(low) | u32::from(high) << 16)
                        } else {
                            let mut expressions = [scaled, scaled];
                            for h in 0..2 {
                                let initial = match (halves[h], reals[h]) {
                                    (Some(bits), _) => self
                                        .arena
                                        .float_from_f64(f16_value(bits).map_or(0.0, |v| v.0) - term)
                                        .map_err(|e| wave.fail(None, e.to_string()))?,
                                    (None, Some(real)) => real,
                                    (None, None) => unreachable!("every half has a value"),
                                };
                                expressions[h] = self.arena.fma(scaled, factor, initial);
                            }
                            Value::Halves(expressions[0], expressions[1])
                        }
                    }
                };
                wave.put(destination, lane, component, value)?;
            }
        }
        Ok(())
    }
    pub(super) fn matrix(
        &mut self,
        wave: &mut Wave,
        instruction: &Instruction,
    ) -> Result<(), EvalError> {
        let Operation::Matrix(matrix) = instruction.operation() else {
            unreachable!("typed matrix dispatch")
        };
        if matrix.family == MatrixFamily::Wmma {
            return self.wmma(wave, instruction);
        }

        use super::super::matrix::{Form, InputKind};
        let mut form = Form::from_operation(matrix.operation)
            .ok_or_else(|| self.unsupported(wave, instruction))?;
        let mut cbsz = 0u8;
        let mut blgp = 0u8;
        for modifier in instruction.modifiers() {
            match modifier.name.as_str() {
                "cbsz" => {
                    cbsz = modifier
                        .value
                        .as_deref()
                        .and_then(parse_u64)
                        .and_then(|value| u8::try_from(value).ok())
                        .ok_or_else(|| self.unsupported(wave, instruction))?
                }
                "blgp" => {
                    blgp = modifier
                        .value
                        .as_deref()
                        .and_then(parse_u64)
                        .and_then(|value| u8::try_from(value).ok())
                        .ok_or_else(|| self.unsupported(wave, instruction))?
                }
                "abid" if modifier.value.as_deref() == Some("0") => {}
                "op_sel" | "op_sel_hi" if form.is_scaled() => {}
                _ => return Err(self.unsupported(wave, instruction)),
            }
        }
        if matches!(form.input_kind, InputKind::Fp8E4m3 | InputKind::Fp8E5m2) {
            form = form
                .with_fp8_formats(cbsz, blgp)
                .ok_or_else(|| self.unsupported(wave, instruction))?;
        } else if cbsz != 0 || blgp != 0 {
            return Err(self.unsupported(wave, instruction));
        }
        if wave.exec != u64::MAX || wave.valid != u64::MAX {
            return Err(EvalError::Unsupported {
                pc: wave.pc,
                what: "MFMA requires a complete active wave64".into(),
            });
        }
        let o = instruction.operands();
        let destination = wave.destination(&o[0])?;
        let output_items = form.accumulator_items_per_lane();
        if destination.count as usize != output_items || destination.index % 2 != 0 {
            return Err(wave.fail(None, "MFMA destination register count/alignment mismatch"));
        }
        for source in [&o[1], &o[2]] {
            if let Operand::Register(register) = source {
                if register.count as usize != form.input_registers_per_operand() {
                    return Err(wave.fail(None, "MFMA input register count mismatch"));
                }
            } else if !matches!(source, Operand::Integer(0) | Operand::Float(0.0)) {
                return Err(self.unsupported(wave, instruction));
            }
        }
        if let Operand::Register(register) = o[3]
            && register.count as usize != output_items
        {
            return Err(wave.fail(None, "MFMA accumulator register count mismatch"));
        }
        if form.is_scaled() {
            if o.len() != 6 {
                return Err(self.unsupported(wave, instruction));
            }
            let low = selectors(instruction, "op_sel", 3, 0)
                .ok_or_else(|| self.unsupported(wave, instruction))?;
            let high = selectors(instruction, "op_sel_hi", 3, 0)
                .ok_or_else(|| self.unsupported(wave, instruction))?;
            if low[2] != 0 || high[2] != 0 {
                return Err(self.unsupported(wave, instruction));
            }
            for lane in 0..64 {
                for index in 0..2 {
                    let exponent = if let Operand::Float(value) = o[4 + index] {
                        ((value as f32).to_bits() >> 23) & 255
                    } else {
                        let byte = (2 * high[index] + low[index]) as u32;
                        (wave.bits(&o[4 + index], lane)? >> (8 * byte)) & 255
                    };
                    if exponent != 127 {
                        return Err(EvalError::Unsupported{pc:wave.pc,what:"block-scaled MFMA currently requires every selected E8M0 scale byte to equal127 (unit scale)".into()});
                    }
                }
            }
        } else if o.len() != 4 {
            return Err(self.unsupported(wave, instruction));
        }
        let (_, writes) = self.instruction_registers(instruction, wave);
        let mut sources = vec![o[0].clone(), o[1].clone(), o[2].clone()];
        if form.is_scaled() {
            sources.extend_from_slice(&o[4..6]);
        }
        let (sources_ab, _) = self.instruction_registers_for_operands(instruction, &sources, wave);
        let sources = [o[0].clone(), o[3].clone()];
        let (accumulator, _) = self.instruction_registers_for_operands(instruction, &sources, wave);
        self.builder.amd_event(
            wave.id,
            ProgramPoint(wave.pc as u32),
            AmdEventKind::MatrixIssue {
                kind: form.graph_kind(),
                destinations: writes,
                sources_ab,
                accumulator,
            },
        );
        let mut a = Vec::with_capacity(64);
        let mut b = Vec::with_capacity(64);
        let mut c = Vec::with_capacity(64);
        for lane in 0..64 {
            let mut av = Vec::new();
            let mut bv = Vec::new();
            let mut cv = Vec::new();
            for item in 0..form.input_items_per_lane() {
                av.push(self.matrix_input(wave, &o[1], lane, item, form.input_kind)?);
                bv.push(self.matrix_input(wave, &o[2], lane, item, form.input_kind_b)?);
            }
            for item in 0..output_items {
                cv.push(if let Operand::Register(register) = o[3] {
                    let value = wave.reg(register, lane, item as u16)?;
                    self.real_value(wave, value, lane)?
                } else {
                    self.real(wave, &o[3], lane)?
                });
            }
            a.push(av);
            b.push(bv);
            c.push(cv);
        }
        let outputs = super::super::matrix::compute(&mut self.arena, form, &a, &b, &c)
            .map_err(|what| EvalError::Unsupported { pc: wave.pc, what })?;
        for (lane, values) in outputs.into_iter().enumerate() {
            for (component, expression) in values.into_iter().enumerate() {
                wave.put(
                    destination,
                    lane as u8,
                    component as u16,
                    Value::Real(expression),
                )?;
            }
        }
        self.stats.warp_syncs += 1;
        Ok(())
    }
}

//! Exact selected gfx1250 syntax. Native mnemonics/source spans remain intact.
use super::*;
use crate::target::Architecture;

pub(super) fn validate_encoding(name: &str) -> Result<(), String> {
    let encoded32 = name.ends_with("_e32");
    let encoded64 = name.ends_with("_e64");
    if !encoded32 && !encoded64 {
        return Ok(());
    }
    let stem = name.trim_end_matches("_e32").trim_end_matches("_e64");
    if name.starts_with("s_")
        || name.starts_with("ds_")
        || name.starts_with("global_")
        || name.starts_with("flat_")
        || name.starts_with("tensor_")
    {
        return Err(format!(
            "{stem} has no explicit VOP encoding suffix in the supported native profile"
        ));
    }
    if encoded32
        && (stem.starts_with("v_wmma_")
            || matches!(
                stem,
                "v_lshlrev_b16"
                    | "v_sub_nc_u16"
                    | "v_ashrrev_i16"
                    | "v_s_exp_f32"
                    | "v_bitop3_b16"
                    | "v_max3_num_f32"
                    | "v_max3_num_f16"
                    | "v_pk_max_num_f16"
                    | "v_pk_mul_lo_u16"
                    | "v_pk_sub_i16"
                    | "v_mad_u32"
                    | "v_xad_u32"
                    | "v_bitop2_b32"
            ))
    {
        return Err(format!(
            "{stem} requires VOP3 encoding; _e32 is not supported"
        ));
    }
    Ok(())
}

fn validate_dual_pair(x: &Instruction, y: &Instruction) -> Result<(), String> {
    for operation in [x, y] {
        let stem = match operation.mnemonic.as_str() {
            "v_cndmask_b32_e64" => {
                if operation.operands.len() != 4
                    || !matches!(operation.operands.get(3), Some(Operand::Register(r))
                        if r.class == RegisterClass::Special && r.index == 0 && r.count == 1
                            && r.special == Some(SpecialRegister::VccLo))
                {
                    return Err(
                        "implicit VOPD CNDMASK requires its canonical VCC_LO selector".into(),
                    );
                }
                "v_cndmask_b32"
            }
            name => name,
        };
        if !matches!(
            stem,
            "v_add_f64"
                | "v_mul_f64"
                | "v_add_f32"
                | "v_mul_f32"
                | "v_fma_f32"
                | "v_fmamk_f32"
                | "v_lshlrev_b32"
                | "v_lshrrev_b32"
                | "v_ashrrev_i32"
                | "v_max_i32"
                | "v_mov_b32"
                | "v_bitop3_b32"
                | "v_cndmask_b32"
                | "v_fmac_f32"
                | "v_add_nc_u32"
                | "v_sub_nc_u32"
                | "v_sub_f32"
                | "v_max_num_f32"
        ) {
            return Err("unsupported VOPD constituent operation".into());
        }
        if stem == "v_bitop3_b32" && !matches!(operation.operands.last(), Some(Operand::Integer(0)))
        {
            return Err("VOPD bitop2 requires its canonical zero third source".into());
        }
    }
    if x.operands.is_empty() || y.operands.is_empty() || x.operands.first() == y.operands.first() {
        return Err("VOPD destinations must be distinct".into());
    }
    let implicit = |operation: &Instruction| operation.mnemonic == "v_cndmask_b32_e64";
    if implicit(x) && y.mnemonic == "v_fma_f32" || implicit(y) && x.mnemonic == "v_fma_f32" {
        return Err("implicit CNDMASK selects VOPD and cannot pair with VOPD3 FMA".into());
    }

    if x.mnemonic == "v_max_i32" {
        return Err("gfx1250 signed MAX is supported only in the VOPD Y slot".into());
    }
    if y.mnemonic == "v_max_i32" && (!y.modifiers.is_empty()
        || y.operands.len()!=3
        || y.operands.iter().any(|op| !matches!(op,Operand::Register(r) if r.class==RegisterClass::Vector && r.count==1)))
    {
        return Err("selected VOPD signed MAX requires three plain VGPR operands".into());
    }
    for operation in [x, y] {
        if operation.mnemonic == "v_ashrrev_i32" {
            let count = match operation.operands.get(1) {
                Some(Operand::Integer(value)) => (-16..=64).contains(value),
                Some(Operand::Register(r)) => r.count == 1,
                _ => false,
            };
            if operation.operands.len() != 3
                || !operation.modifiers.is_empty()
                || !count
                || !matches!(operation.operands.first(),Some(Operand::Register(r)) if r.class == RegisterClass::Vector && r.count == 1)
                || !matches!(operation.operands.get(2),Some(Operand::Register(r)) if r.class == RegisterClass::Vector && r.count == 1)
            {
                return Err("selected VOPD ASHR requires a plain VGPR destination/data and an inline or register count".into());
            }
        }
    }
    if matches!(x.mnemonic.as_str(), "v_add_f64" | "v_mul_f64")
        || matches!(y.mnemonic.as_str(), "v_add_f64" | "v_mul_f64")
    {
        // LLVM VOPD3XPseudosExtraGFX1250 adds F64 only in the X slot.
        // This selected profile covers the compiler's paired zero-address
        // initialization, with aligned VGPR pairs and disjoint destinations.
        if !matches!(x.mnemonic.as_str(), "v_add_f64" | "v_mul_f64")
            || y.mnemonic != "v_mov_b32"
            || x.operands.len() != 3
            || y.operands.len() != 2
            || y.operands.get(1) != Some(&Operand::Integer(0))
            || !x.modifiers.is_empty()
            || !y.modifiers.is_empty()
        {
            return Err("selected F64 VOPD3 requires an X-slot add/mul and Y-slot MOV zero".into());
        }
        for operand in &x.operands {
            if !matches!(operand, Operand::Register(r) if r.class == RegisterClass::Vector && r.count == 2 && r.index % 2 == 0)
            {
                return Err("selected F64 VOPD3 requires aligned two-VGPR operands".into());
            }
        }
        let (Some(Operand::Register(xd)), Some(Operand::Register(yd))) =
            (x.operands.first(), y.operands.first())
        else {
            return Err("F64 VOPD3 requires physical VGPR destinations".into());
        };
        if yd.class != RegisterClass::Vector
            || yd.count != 1
            || (u32::from(xd.index)..u32::from(xd.index) + u32::from(xd.count))
                .contains(&u32::from(yd.index))
        {
            return Err("F64 VOPD3 destinations must be disjoint".into());
        }
    }
    if x.mnemonic == "v_fmamk_f32" || y.mnemonic == "v_fmamk_f32" {
        // VOPD MADK carries one shared mandatory literal, unlike VOPD3.
        fn bits(operand: &Operand) -> Option<u32> {
            match operand {
                Operand::Integer(value) if (i32::MIN as i64..=u32::MAX as i64).contains(value) => {
                    Some(*value as u32)
                }
                Operand::Float(value) => Some((*value as f32).to_bits()),
                Operand::Modified { value, .. } => bits(value),
                _ => None,
            }
        }
        let mut shared_literal = None;
        for instruction in [x, y] {
            if instruction.mnemonic == "v_fmamk_f32" {
                let literal = instruction
                    .operands
                    .get(2)
                    .and_then(bits)
                    .ok_or("VOPD FMAMK requires its encoded K literal")?;
                if shared_literal.is_some_and(|old| old != literal) {
                    return Err("VOPD FMAMK halves require the same shared literal".into());
                }
                shared_literal = Some(literal);
            } else if matches!(
                instruction.mnemonic.as_str(),
                "v_fma_f32" | "v_cndmask_b32_e64"
            ) {
                return Err("selected VOPD FMAMK cannot pair with VOPD3 sources".into());
            }
        }
        let literal = shared_literal.expect("at least one FMAMK");
        let inline = |word: u32| {
            word <= 64
                || word >= (-16i32 as u32)
                || [
                    0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
                    0x40800000, 0xc0800000, 0x3e22f983,
                ]
                .contains(&word)
        };
        for instruction in [x, y] {
            for (index, operand) in instruction.operands.iter().enumerate().skip(1) {
                if instruction.mnemonic == "v_fmamk_f32" && index == 2 {
                    continue;
                }
                if bits(operand).is_some_and(|word| !inline(word) && word != literal) {
                    return Err("VOPD source conflicts with the shared FMAMK literal".into());
                }
            }
        }
    }
    Ok(())
}

/// Apply the same selected instruction contract to caller-built typed modules.
pub(crate) fn validate_typed_profile(instruction: &Instruction) -> Result<(), String> {
    if let [Operand::Dual(x, y)] = instruction.operands.as_slice() {
        if instruction.opcode != Opcode::VectorAlu || !instruction.mnemonic.starts_with("v_dual_") {
            return Err("typed dual operands require a native VOPD vector instruction".into());
        }
        validate_dual_pair(x, y)?;
        validate_typed_profile(x)?;
        validate_typed_profile(y)?;
        return Ok(());
    }
    validate_encoding(&instruction.mnemonic)?;
    if instruction.mnemonic == "s_sendmsg"
        && (instruction.operands != [Operand::Integer(3)] || !instruction.modifiers.is_empty())
    {
        return Err("gfx1250 only supports terminal MSG_DEALLOC_VGPRS (message3)".into());
    }
    if instruction.mnemonic == "s_addk_co_i32"
        && !matches!(instruction.operands.get(1), Some(Operand::Integer(value)) if (-32768..=65535).contains(value))
    {
        return Err("s_addk_co_i32 requires one native 16-bit immediate".into());
    }
    if instruction.mnemonic.starts_with("v_cvt_f32_bf8") {
        let source_is_word = match instruction.operands.get(1) {
            Some(Operand::Register(register)) => register.count == 1,
            Some(Operand::Integer(value)) => (i32::MIN as i64..=u32::MAX as i64).contains(value),
            _ => false,
        };
        if !matches!(
            instruction.mnemonic.as_str(),
            "v_cvt_f32_bf8" | "v_cvt_f32_bf8_e32" | "v_cvt_f32_bf8_e64"
        ) || !instruction
            .modifiers
            .iter()
            .all(|modifier| modifier.name == "byte_sel")
            || !source_is_word
        {
            return Err("selected gfx1250 E5M2 decoder requires a plain word source and VOP1 byte0 or VOP3 byte selection".into());
        }
    }
    let byte_selectors = instruction
        .modifiers
        .iter()
        .filter(|modifier| modifier.name == "byte_sel")
        .collect::<Vec<_>>();
    if !byte_selectors.is_empty() {
        let stem = instruction
            .mnemonic
            .trim_end_matches("_e32")
            .trim_end_matches("_e64");
        if !matches!(stem, "v_cvt_f32_fp8" | "v_cvt_f32_bf8")
            || instruction.mnemonic.ends_with("_e32")
            || byte_selectors.len() != 1
            || !byte_selectors[0]
                .value
                .as_deref()
                .and_then(super::instr::integer)
                .is_some_and(|byte| (0..=3).contains(&byte))
        {
            return Err("gfx1250 FP8 byte_sel requires one selector 0..3 and VOP3 encoding".into());
        }
    }
    super::instr::validate_control(instruction).map_err(|e| e.message)?;
    let (opcode, mut min, mut max) = match instruction.mnemonic.as_str() {
        "s_getreg_b32" | "s_setreg_b32" | "s_setreg_imm32_b32" => (Opcode::ScalarAlu, 2, 2),
        "s_round_mode" | "s_denorm_mode" => (Opcode::ScalarAlu, 1, 1),
        "s_delay_alu" => (Opcode::Control, 1, 1),
        name => super::instr::signature(name)
            .or_else(|| signature(name))
            .ok_or_else(|| format!("unsupported gfx1250 typed instruction {name}"))?,
    };
    if instruction.mnemonic.starts_with("v_cmpx_") {
        min = 2;
        max = 2;
    }
    if instruction.opcode != opcode || !(min..=max).contains(&instruction.operands.len()) {
        return Err(format!(
            "typed gfx1250 instruction {} has an invalid opcode/operand count",
            instruction.mnemonic
        ));
    }
    super::instr::validate_registers(
        &instruction.mnemonic,
        instruction.opcode,
        &instruction.operands,
        &instruction.modifiers,
        Architecture::Gfx1250,
    )?;
    super::instr::validate_vop_literal_positions(
        &instruction.mnemonic,
        &instruction.operands,
        &instruction.modifiers,
        Architecture::Gfx1250,
    )?;
    Ok(())
}

pub(crate) fn signature(name: &str) -> Option<(Opcode, usize, usize)> {
    use Opcode::*;
    Some(
        match name.trim_end_matches("_e32").trim_end_matches("_e64") {
            "s_load_b32" | "s_load_b64" | "s_load_b96" | "s_load_b128" | "s_load_b256"
            | "s_load_b512" | "s_load_u8" | "s_load_u16" => (ScalarMemory, 3, 3),
            "s_get_pc_i64" | "s_get_shader_cycles_u64" => (ScalarAlu, 1, 1),
            "s_sendmsg_rtn_b32" => (ScalarAlu, 2, 2),
            "s_cvt_f32_i32" | "s_cvt_f32_u32" => (ScalarAlu, 2, 2),
            "global_load_async_to_lds_b128"
            | "global_load_async_to_lds_b64"
            | "global_load_async_to_lds_b32" => (VectorMemory, 3, 3),
            "buffer_load_b32" | "buffer_load_b64" | "buffer_load_b96" | "buffer_store_b32"
            | "buffer_store_b96" | "buffer_store_b128" => (VectorMemory, 4, 4),
            "global_prefetch_b8" => (VectorMemory, 2, 2),
            "flat_load_u8"
            | "flat_load_i8"
            | "flat_load_u16"
            | "flat_load_i16"
            | "flat_load_b32"
            | "flat_load_b64"
            | "flat_load_b96"
            | "flat_load_b128"
            | "flat_store_b8"
            | "flat_store_b16"
            | "flat_store_b32"
            | "flat_store_b64"
            | "flat_store_b128"
            | "flat_store_d16_hi_b16" => (VectorMemory, 2, 3),
            "global_load_u16" | "global_load_u8" | "global_load_i8" | "global_load_i16"
            | "global_store_b16" | "global_store_b8" | "global_load_b32" | "global_load_b64"
            | "global_load_b96" | "global_load_b128" | "global_store_b32" | "global_store_b64"
            | "global_store_b96" | "global_store_b128" => (VectorMemory, 3, 3),
            "ds_load_2addr_b32" => (Lds, 2, 2),
            "ds_load_u16"
            | "ds_load_i16"
            | "ds_load_u8"
            | "ds_load_i8"
            | "ds_store_b16"
            | "ds_store_b16_d16_hi"
            | "ds_store_b8"
            | "ds_load_b96"
            | "ds_load_b32"
            | "ds_load_b64"
            | "ds_load_b128"
            | "ds_store_b32"
            | "ds_store_b64"
            | "ds_store_b128" => (Lds, 2, 2),
            "s_wait_loadcnt_dscnt"
            | "s_wait_storecnt_dscnt"
            | "s_wait_loadcnt"
            | "s_wait_storecnt"
            | "s_wait_dscnt"
            | "s_wait_kmcnt"
            | "s_wait_xcnt"
            | "s_wait_asynccnt"
            | "s_wait_tensorcnt"
            | "s_wait_alu" => (Wait, 1, 1),
            "s_wait_idle" => (Wait, 0, 0),
            "s_barrier_signal" | "s_barrier_wait" | "s_barrier_init" | "s_barrier_join" => {
                (Barrier, 1, 1)
            }
            "s_barrier_leave" => (Barrier, 0, 0),
            "s_clause" => (Control, 1, 1),
            "v_nop" => (Control, 0, 0),
            "tensor_load_to_lds" | "tensor_store_from_lds" => (VectorMemory, 4, 4),
            "s_add_co_u32" | "s_sub_co_u32" | "s_add_co_ci_u32" | "s_mul_u64" | "s_add_nc_u64"
            | "s_and_not1_b32" | "s_add_co_i32" | "s_sub_co_i32" => (ScalarAlu, 3, 3),
            "s_and_saveexec_b32"
            | "s_andn2_saveexec_b32"
            | "s_and_not1_saveexec_b32"
            | "s_or_saveexec_b32" => (ScalarAlu, 2, 2),
            "s_add_f32" | "s_sub_f32" | "s_mul_f32" | "s_min_num_f32" => (ScalarAlu, 3, 3),
            "s_ctz_i32_b32" | "s_addk_co_i32" => (ScalarAlu, 2, 2),
            "v_fmac_f16" => (VectorAlu, 3, 3),
            "v_s_exp_f32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte3" => (VectorAlu, 2, 2),
            "v_pk_max_num_f16" | "v_max_num_f16" | "v_pk_mul_lo_u16" | "v_pk_sub_i16" => {
                (VectorAlu, 3, 3)
            }
            "v_max3_num_f16" => (VectorAlu, 4, 4),
            "v_cmp_ne_u16" | "v_cmp_u_f32" | "v_min_num_f32" | "v_max_num_f32"
            | "v_lshlrev_b16" | "v_sub_nc_u16" | "v_ashrrev_i16" => (VectorAlu, 3, 3),
            "v_bitop3_b16" | "v_max3_num_f32" | "v_fmaak_f32" => (VectorAlu, 4, 4),
            "v_mad_u32" | "v_xad_u32" => (VectorAlu, 4, 4),
            "v_add_co_ci_u32" => (VectorAlu, 5, 5),
            "v_add_nc_u64" | "v_add_nc_u32" | "v_sub_nc_u32" | "v_subrev_nc_u32"
            | "v_bitop2_b32" => (VectorAlu, 3, 3),
            "v_wmma_f32_16x16x4_f32" | "v_wmma_f32_16x16x32_f16" | "v_wmma_f32_16x16x32_bf16" => {
                (Matrix, 4, 4)
            }
            // Executed only under gfx1250-b0-wmma/2; elsewhere the evaluator
            // refuses them as unsupported.
            "v_wmma_i32_16x16x64_iu8"
            | "v_wmma_f32_16x16x64_fp8_fp8"
            | "v_wmma_f32_16x16x128_fp8_fp8" => (Matrix, 4, 4),
            // Executed only under gfx1250-b0-wmma/3 (BF8 combinations,
            // F8F6F4/F4 and the 4-dword block-scaled forms with their two
            // scale operands).
            "v_wmma_f32_16x16x64_fp8_bf8"
            | "v_wmma_f32_16x16x64_bf8_fp8"
            | "v_wmma_f32_16x16x64_bf8_bf8"
            | "v_wmma_f32_16x16x128_fp8_bf8"
            | "v_wmma_f32_16x16x128_bf8_fp8"
            | "v_wmma_f32_16x16x128_bf8_bf8"
            | "v_wmma_f32_16x16x128_f8f6f4"
            | "v_wmma_f32_32x16x128_f4" => (Matrix, 4, 4),
            "v_wmma_scale_f32_16x16x128_f8f6f4"
            | "v_wmma_scale16_f32_16x16x128_f8f6f4"
            | "v_wmma_scale_f32_32x16x128_f4"
            | "v_wmma_scale16_f32_32x16x128_f4" => (Matrix, 6, 6),
            // Executed only under gfx1250-b0-wmma/4: 16-bit-output WMMA and
            // SWMMAC (D, packed A, B, sparsity index; C is D).
            "v_wmma_f16_16x16x32_f16"
            | "v_wmma_bf16_16x16x32_bf16"
            | "v_wmma_bf16f32_16x16x32_bf16"
            | "v_wmma_f16_16x16x64_fp8_fp8"
            | "v_wmma_f16_16x16x64_fp8_bf8"
            | "v_wmma_f16_16x16x64_bf8_fp8"
            | "v_wmma_f16_16x16x64_bf8_bf8"
            | "v_wmma_f16_16x16x128_fp8_fp8"
            | "v_wmma_f16_16x16x128_fp8_bf8"
            | "v_wmma_f16_16x16x128_bf8_fp8"
            | "v_wmma_f16_16x16x128_bf8_bf8"
            | "v_swmmac_f32_16x16x64_f16"
            | "v_swmmac_f32_16x16x64_bf16"
            | "v_swmmac_f16_16x16x64_f16"
            | "v_swmmac_bf16_16x16x64_bf16"
            | "v_swmmac_bf16f32_16x16x64_bf16"
            | "v_swmmac_f32_16x16x128_fp8_fp8"
            | "v_swmmac_f32_16x16x128_fp8_bf8"
            | "v_swmmac_f32_16x16x128_bf8_fp8"
            | "v_swmmac_f32_16x16x128_bf8_bf8"
            | "v_swmmac_f16_16x16x128_fp8_fp8"
            | "v_swmmac_f16_16x16x128_fp8_bf8"
            | "v_swmmac_f16_16x16x128_bf8_fp8"
            | "v_swmmac_f16_16x16x128_bf8_bf8"
            | "v_swmmac_i32_16x16x128_iu8" => (Matrix, 4, 4),
            _ => return None,
        },
    )
}

/// LLVM's gfx12 DEPCTR operand: a raw unsigned16 immediate or a list of
/// `depctr_<field>(n)` items. Unlisted fields keep the no-wait encoding.
fn parse_wait_alu(text: &str) -> Result<u16, String> {
    if let Some(value) = super::instr::integer(text) {
        return u16::try_from(value).map_err(|_| "s_wait_alu immediate must be unsigned16".into());
    }
    // Matches the event model's no-wait encoding; the frontend stays graph-free.
    let mut encoded: u16 = 0xff9f;
    let mut seen = std::collections::BTreeSet::new();
    let items: Vec<_> = text
        .split(|c: char| c.is_whitespace() || c == '&')
        .filter(|item| !item.is_empty())
        .collect();
    if items.is_empty() {
        return Err("s_wait_alu requires an operand".into());
    }
    for item in items {
        let (field, value) = item
            .strip_prefix("depctr_")
            .and_then(|rest| rest.split_once('('))
            .and_then(|(field, value)| value.strip_suffix(')').map(|value| (field, value)))
            .ok_or_else(|| format!("invalid s_wait_alu field `{item}`"))?;
        let (shift, maximum) = match field {
            "sa_sdst" => (0, 1),
            "va_vcc" => (1, 1),
            "vm_vsrc" => (2, 7),
            "hold_cnt" => (7, 1),
            "va_ssrc" => (8, 1),
            "va_sdst" => (9, 7),
            "va_vdst" => (12, 15),
            _ => return Err(format!("unknown s_wait_alu field `{field}`")),
        };
        if !seen.insert(field) {
            return Err(format!("duplicate s_wait_alu field `{field}`"));
        }
        let value = super::instr::integer(value)
            .filter(|value| (0..=maximum).contains(value))
            .ok_or_else(|| format!("s_wait_alu field `{field}` is outside 0..={maximum}"))?;
        encoded = encoded & !((maximum as u16) << shift) | (value as u16) << shift;
    }
    Ok(encoded)
}

pub(super) fn parse_special(
    text: &str,
    span: SourceSpan,
) -> Result<Option<Instruction>, ParseError> {
    let fail = |message: &str| ParseError {
        message: message.into(),
        span,
    };
    // The returned-message form names its message symbolically; it is the
    // message code (0x87, the SE and XCC id message) as an integer operand.
    if let Some(("s_sendmsg_rtn_b32", rest)) = text.split_once(char::is_whitespace)
        && let Some((destination, message)) = rest.split_once(',')
        && message.trim() == "sendmsg(MSG_RTN_GET_SE_AID_ID)"
    {
        let normalized = format!("s_sendmsg_rtn_b32 {}, 135", destination.trim());
        return super::instr::parse_instruction(
            &normalized,
            span,
            crate::target::Architecture::Gfx1250,
        )
        .map(Some);
    }
    if let Some(("s_sendmsg", rest)) = text.split_once(char::is_whitespace)
        && rest.trim() == "sendmsg(MSG_DEALLOC_VGPRS)"
    {
        return Ok(Some(Instruction {
            mnemonic: "s_sendmsg".into(),
            opcode: Opcode::Control,
            operands: vec![Operand::Integer(3)],
            modifiers: vec![],
            span,
        }));
    }
    if let Some((x, y)) = text.split_once("::") {
        if y.contains("::") {
            return Err(fail("VOPD requires exactly two operations"));
        }
        // An omitted CNDMASK src2 selects VOPD; explicit src2 selects
        // VOPD3. Preserve that encoding choice before normalization.
        let implicit_condition = |part: &str| {
            part.trim()
                .strip_prefix("v_dual_cndmask_b32 ")
                .is_some_and(|rest| rest.split(',').count() == 3)
        };
        let requires_three_source = |part: &str| part.trim().starts_with("v_dual_fma_f32 ");
        if (implicit_condition(x) && requires_three_source(y))
            || (implicit_condition(y) && requires_three_source(x))
        {
            return Err(fail(
                "implicit CNDMASK selects VOPD and cannot pair with a VOPD3 FMA; provide the explicit condition operand",
            ));
        }
        fn half(text: &str, span: SourceSpan) -> Result<Instruction, ParseError> {
            let (name, rest) = text
                .trim()
                .split_once(char::is_whitespace)
                .unwrap_or((text.trim(), ""));
            let name = name.strip_prefix("v_dual_").ok_or_else(|| ParseError {
                message: "VOPD requires v_dual_ operations".into(),
                span,
            })?;
            if ![
                "add_f64",
                "mul_f64",
                "add_f32",
                "mul_f32",
                "fma_f32",
                "fmamk_f32",
                "lshlrev_b32",
                "lshrrev_b32",
                "ashrrev_i32",
                "max_i32",
                "mov_b32",
                "bitop2_b32",
                "cndmask_b32",
                "fmac_f32",
                "add_nc_u32",
                "sub_nc_u32",
                "sub_f32",
                "max_num_f32",
            ]
            .contains(&name)
            {
                return Err(ParseError {
                    message: format!("unsupported VOPD operation {name}"),
                    span,
                });
            }
            let text = if name == "cndmask_b32" && rest.split(',').count() == 3 {
                format!("v_{name}_e64 {rest}, vcc_lo")
            } else {
                format!("v_{name} {rest}")
            };
            let mut op = super::instr::parse_instruction(&text, span, Architecture::Gfx1250)?;
            if name == "bitop2_b32" {
                op.mnemonic = "v_bitop3_b32".into();
                op.operands.push(Operand::Integer(0));
            }
            Ok(op)
        }
        let x = half(x, span)?;
        let y = half(y, span)?;
        validate_dual_pair(&x, &y).map_err(|message| ParseError { message, span })?;
        if x.operands[0] == y.operands[0] {
            return Err(fail("VOPD destinations must be distinct"));
        }
        return Ok(Some(Instruction {
            mnemonic: text.split_whitespace().next().unwrap().into(),
            opcode: Opcode::VectorAlu,
            operands: vec![Operand::Dual(Box::new(x), Box::new(y))],
            modifiers: vec![],
            span,
        }));
    }
    if let Some(rest) = text.strip_prefix("s_wait_alu ") {
        return parse_wait_alu(rest.trim())
            .map(|encoded| {
                Some(Instruction {
                    mnemonic: "s_wait_alu".into(),
                    opcode: Opcode::Wait,
                    operands: vec![Operand::Integer(i64::from(encoded))],
                    modifiers: vec![],
                    span,
                })
            })
            .map_err(|message| fail(&message));
    }
    if let Some(rest) = text.strip_prefix("s_delay_alu ") {
        let encoded = if let Some(value) = super::instr::integer(rest.trim()) {
            value
        } else {
            let mut encoded = 0i64;
            let mut fields = std::collections::BTreeSet::new();
            for item in rest.split('|') {
                let (field, value) = item
                    .trim()
                    .split_once('(')
                    .and_then(|(a, b)| b.strip_suffix(')').map(|b| (a, b)))
                    .ok_or_else(|| fail("invalid s_delay_alu field"))?;
                if !fields.insert(field) {
                    return Err(fail("duplicate s_delay_alu field"));
                }
                let (shift, value) = if field == "instskip" {
                    (
                        4,
                        match value {
                            "SAME" => 0,
                            "NEXT" => 1,
                            "SKIP_1" => 2,
                            "SKIP_2" => 3,
                            "SKIP_3" => 4,
                            "SKIP_4" => 5,
                            _ => return Err(fail("invalid instskip")),
                        },
                    )
                } else {
                    let shift = match field {
                        "instid0" => 0,
                        "instid1" => 7,
                        _ => return Err(fail("invalid dependency field")),
                    };
                    let value = match value {
                        "NO_DEP" => 0,
                        "VALU_DEP_1" => 1,
                        "VALU_DEP_2" => 2,
                        "VALU_DEP_3" => 3,
                        "VALU_DEP_4" => 4,
                        "TRANS32_DEP_1" => 5,
                        "TRANS32_DEP_2" => 6,
                        "TRANS32_DEP_3" => 7,
                        "FMA_ACCUM_CYCLE_1" => 8,
                        "SALU_CYCLE_1" => 9,
                        "SALU_CYCLE_2" => 10,
                        "SALU_CYCLE_3" => 11,
                        _ => return Err(fail("invalid ALU dependency")),
                    };
                    (shift, value)
                };
                encoded |= value << shift;
            }
            encoded
        };
        if !super::instr::valid_alu_delay(encoded) {
            return Err(fail("invalid s_delay_alu encoding"));
        }
        return Ok(Some(Instruction {
            mnemonic: "s_delay_alu".into(),
            opcode: Opcode::Control,
            operands: vec![Operand::Integer(encoded)],
            modifiers: vec![],
            span,
        }));
    }
    // `hwreg(NAME[, offset, width])`; the name is kept as written so that a
    // profile can accept exact spellings only.
    let hwreg = |source: &str| -> Result<Operand, ParseError> {
        let args = source
            .trim()
            .strip_prefix("hwreg(")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| fail("invalid hardware register operand"))?
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>();
        // `hwreg(NAME)` selects the whole register.
        if args.len() == 1 && !args[0].is_empty() {
            return Ok(Operand::HardwareRegister {
                name: args[0].into(),
                offset: 0,
                width: 32,
            });
        }
        if args.len() != 3 {
            return Err(fail("hwreg requires name or name,offset,width"));
        }
        let offset = args[1]
            .parse::<u8>()
            .map_err(|_| fail("invalid hwreg offset"))?;
        let width = args[2]
            .parse::<u8>()
            .map_err(|_| fail("invalid hwreg width"))?;
        if width == 0 || u16::from(offset) + u16::from(width) > 32 {
            return Err(fail("hwreg bit selection outside 32-bit register"));
        }
        Ok(Operand::HardwareRegister {
            name: args[0].into(),
            offset,
            width,
        })
    };
    if let Some(rest) = text.strip_prefix("s_getreg_b32 ") {
        let (destination, source) = rest
            .split_once(',')
            .ok_or_else(|| fail("s_getreg_b32 requires destination and hwreg"))?;
        let mut instruction = super::instr::parse_instruction(
            &format!("s_mov_b32 {}, 0", destination.trim()),
            span,
            Architecture::Gfx1250,
        )?;
        instruction.mnemonic = "s_getreg_b32".into();
        instruction.operands[1] = hwreg(source)?;
        return Ok(Some(instruction));
    }
    // SETREG writes a hardware-register field from an SGPR (SOPK) or from a
    // 32-bit literal (SOPK + literal). The hwreg operand is the destination.
    for (prefix, mnemonic) in [
        ("s_setreg_b32 ", "s_setreg_b32"),
        ("s_setreg_imm32_b32 ", "s_setreg_imm32_b32"),
    ] {
        let Some(rest) = text.strip_prefix(prefix) else {
            continue;
        };
        let (destination, source) = rest
            .rsplit_once(',')
            .ok_or_else(|| fail("s_setreg requires hwreg and source"))?;
        let parsed = super::instr::parse_instruction(
            &format!("s_mov_b32 s0, {}", source.trim()),
            span,
            Architecture::Gfx1250,
        )?;
        let source = parsed.operands[1].clone();
        let shape_ok = if mnemonic == "s_setreg_b32" {
            matches!(&source, Operand::Register(r) if r.class == RegisterClass::Scalar && r.count == 1)
        } else {
            matches!(source, Operand::Integer(value) if (i64::from(i32::MIN)..=i64::from(u32::MAX)).contains(&value))
        };
        if !shape_ok {
            return Err(fail(if mnemonic == "s_setreg_b32" {
                "s_setreg_b32 requires one SGPR source"
            } else {
                "s_setreg_imm32_b32 requires a 32-bit integer source"
            }));
        }
        return Ok(Some(Instruction {
            mnemonic: mnemonic.into(),
            opcode: Opcode::ScalarAlu,
            operands: vec![hwreg(destination)?, source],
            modifiers: vec![],
            span,
        }));
    }
    // SOPP MODE writes with a 16-bit immediate.
    for mnemonic in ["s_round_mode", "s_denorm_mode"] {
        let Some(rest) = text
            .strip_prefix(mnemonic)
            .and_then(|rest| rest.strip_prefix(char::is_whitespace))
        else {
            continue;
        };
        let value = super::instr::integer(rest.trim())
            .filter(|value| (0..=0xffff).contains(value))
            .ok_or_else(|| fail(&format!("{mnemonic} requires an unsigned 16-bit immediate")))?;
        return Ok(Some(Instruction {
            mnemonic: mnemonic.into(),
            opcode: Opcode::ScalarAlu,
            operands: vec![Operand::Integer(value)],
            modifiers: vec![],
            span,
        }));
    }
    Ok(None)
}

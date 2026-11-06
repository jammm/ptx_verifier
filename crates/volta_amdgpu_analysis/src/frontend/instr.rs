use std::collections::BTreeMap;

use super::{
    Instruction, Modifier, Opcode, Operand, ParseError, Register, RegisterClass, Relocation,
    RelocationKind,
};
use super::{SourceSpan, SpecialRegister};

pub(super) fn integer(text: &str) -> Option<i64> {
    let text = text.trim();
    let (negative, magnitude) = text.strip_prefix('-').map_or((false, text), |s| (true, s));
    let magnitude = magnitude.strip_prefix('+').unwrap_or(magnitude);
    let value = if let Some(hex) = magnitude
        .strip_prefix("0x")
        .or_else(|| magnitude.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()?
    } else if let Some(binary) = magnitude.strip_prefix("0b") {
        u64::from_str_radix(binary, 2).ok()?
    } else {
        magnitude.parse::<u64>().ok()?
    };
    if negative {
        if value > (1_u64 << 63) {
            return None;
        }
        Some((value as i64).wrapping_neg())
    } else {
        // Hex constants describe bit patterns, including full-width masks.
        Some(value as i64)
    }
}

/// Small assembly-constant expression grammar. It deliberately excludes
/// relocations and arbitrary assembler functions rather than guessing a value.
pub(super) fn constant_expression(text: &str, constants: &BTreeMap<String, i64>) -> Option<i64> {
    if let Some(value) = integer(text) {
        return Some(value);
    }
    if let Some(value) = constants.get(text.trim()) {
        return Some(*value);
    }
    for (index, ch) in text.char_indices().rev() {
        if index != 0 && (ch == '+' || ch == '-') {
            let left = constant_expression(&text[..index], constants)?;
            let right = constant_expression(&text[index + 1..], constants)?;
            return if ch == '+' {
                left.checked_add(right)
            } else {
                left.checked_sub(right)
            };
        }
    }
    None
}

/// Resolve symbolic register indices such as `v[base:base+3]`. The returned
/// text is only an internal lexical normalization; source locations stay native.
pub(super) fn resolve_register_indices(
    text: &str,
    constants: &BTreeMap<String, i64>,
) -> Result<String, String> {
    let mut normalized = String::new();
    let mut position = 0;
    while let Some(relative) = text[position..].find('[') {
        let begin = position + relative;
        let prefix = &text[..begin];
        let register = prefix.ends_with('s')
            || prefix.ends_with('v')
            || prefix.ends_with('a')
            || prefix.ends_with("acc");
        normalized.push_str(&text[position..=begin]);
        let end = text[begin + 1..]
            .find(']')
            .map(|relative| begin + 1 + relative)
            .ok_or("unterminated operand range")?;
        let range = &text[begin + 1..end];
        if register {
            let components = range.split(':').collect::<Vec<_>>();
            if components.is_empty() || components.len() > 2 {
                return Err("malformed register range".into());
            }
            for (index, component) in components.iter().enumerate() {
                let value = constant_expression(component, constants)
                    .ok_or_else(|| format!("unresolved register index `{component}`"))?;
                if index != 0 {
                    normalized.push(':');
                }
                normalized.push_str(&value.to_string());
            }
        } else {
            normalized.push_str(range);
        }
        normalized.push(']');
        position = end + 1;
    }
    normalized.push_str(&text[position..]);
    Ok(normalized)
}

pub(super) fn split_top_level(text: &str, delimiter: char) -> Result<Vec<&str>, String> {
    let mut start = 0;
    let mut stack = Vec::new();
    let mut result = Vec::new();
    let mut absolute = false;
    for (index, character) in text.char_indices() {
        match character {
            '[' | '(' => stack.push(character),
            ']' | ')' => {
                let expected = if character == ']' { '[' } else { '(' };
                if stack.pop() != Some(expected) {
                    return Err("unbalanced operand delimiters".into());
                }
            }
            '|' => absolute = !absolute,
            _ => {}
        }
        if character == delimiter && stack.is_empty() && !absolute {
            let part = text[start..index].trim();
            if part.is_empty() {
                return Err("empty operand".into());
            }
            result.push(part);
            start = index + character.len_utf8();
        }
    }
    if !stack.is_empty() || absolute {
        return Err("unbalanced operand delimiters".into());
    }
    let part = text[start..].trim();
    if !part.is_empty() {
        result.push(part);
    } else if !result.is_empty() {
        return Err("trailing operand separator".into());
    }
    Ok(result)
}

fn whitespace_tokens(text: &str) -> Result<Vec<&str>, String> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut stack = Vec::new();
    for (index, ch) in text.char_indices() {
        match ch {
            '[' | '(' => stack.push(ch),
            ']' | ')' => {
                stack.pop();
            }
            _ => {}
        }
        if ch.is_whitespace() && stack.is_empty() {
            let token = text[start..index].trim();
            if !token.is_empty() {
                result.push(token);
            }
            start = index + ch.len_utf8();
        }
    }
    if !stack.is_empty() {
        return Err("unbalanced operand delimiters".into());
    }
    if !text[start..].trim().is_empty() {
        result.push(text[start..].trim());
    }
    Ok(result)
}

/// Physical tuple shape shared by the lexer and public AST lowering.
fn validate_register_shape(
    register: Register,
    architecture: crate::target::Architecture,
) -> Result<(), String> {
    use crate::target::Architecture;
    if register.class == RegisterClass::Special {
        let Some(special) = register.special else {
            return Err("special register has no selector".into());
        };
        let count = match special {
            SpecialRegister::Exec
            | SpecialRegister::Vcc
            | SpecialRegister::FlatScratch
            | SpecialRegister::SharedBase
            | SpecialRegister::PrivateBase => 2,
            _ => 1,
        };
        return if register.index == 0 && register.count == count {
            Ok(())
        } else {
            Err("special register selector/width mismatch".into())
        };
    }
    let limit = match register.class {
        RegisterClass::Scalar => {
            if architecture == Architecture::Gfx1250 {
                106
            } else {
                102
            }
        }
        RegisterClass::Vector => {
            if architecture == Architecture::Gfx1250 {
                1024
            } else {
                256
            }
        }
        RegisterClass::Accumulator => 256,
        RegisterClass::Temporary if architecture == Architecture::Gfx1250 => 16,
        RegisterClass::Temporary => return Err("temporary register requires gfx1250".into()),
        RegisterClass::Special => unreachable!(),
    };
    if register.count == 0
        || register.special.is_some()
        || u32::from(register.index) + u32::from(register.count) > limit
    {
        return Err("physical register range is outside its native encoding profile".into());
    }
    if register.count > 1 {
        let alignment = if register.class == RegisterClass::Scalar && register.count >= 3 {
            4
        } else {
            2
        };
        if !register.index.is_multiple_of(alignment) {
            return Err("misaligned register tuple".into());
        }
    }
    Ok(())
}

fn register(
    text: &str,
    architecture: crate::target::Architecture,
) -> Result<Option<Register>, String> {
    let special = match text {
        "exec" => Some((SpecialRegister::Exec, 2)),
        "exec_lo" => Some((SpecialRegister::ExecLo, 1)),
        "exec_hi" => Some((SpecialRegister::ExecHi, 1)),
        "vcc" => Some((SpecialRegister::Vcc, 2)),
        "vcc_lo" => Some((SpecialRegister::VccLo, 1)),
        "vcc_hi" => Some((SpecialRegister::VccHi, 1)),
        "scc" => Some((SpecialRegister::Scc, 1)),
        "m0" => Some((SpecialRegister::M0, 1)),
        "null" => Some((SpecialRegister::Null, 1)),
        "flat_scratch" => Some((SpecialRegister::FlatScratch, 2)),
        "flat_scratch_lo" => Some((SpecialRegister::FlatScratchLo, 1)),
        "flat_scratch_hi" => Some((SpecialRegister::FlatScratchHi, 1)),
        "src_shared_base" => Some((SpecialRegister::SharedBase, 2)),
        "src_private_base" => Some((SpecialRegister::PrivateBase, 2)),
        _ => None,
    };
    if let Some((special, count)) = special {
        return Ok(Some(Register {
            class: RegisterClass::Special,
            index: 0,
            count,
            special: Some(special),
        }));
    }
    let (class, suffix, limit) = if let Some(suffix) = text.strip_prefix("ttmp") {
        if architecture != crate::target::Architecture::Gfx1250 {
            return Err("architected temporary registers require gfx1250".into());
        }
        (RegisterClass::Temporary, suffix, 15)
    } else if let Some(suffix) = text.strip_prefix("acc") {
        (RegisterClass::Accumulator, suffix, 255)
    } else if let Some(suffix) = text.strip_prefix('a') {
        (RegisterClass::Accumulator, suffix, 255)
    } else if let Some(suffix) = text.strip_prefix('v') {
        (
            RegisterClass::Vector,
            suffix,
            if architecture == crate::target::Architecture::Gfx1250 {
                1023
            } else {
                255
            },
        )
    } else if let Some(suffix) = text.strip_prefix('s') {
        (
            RegisterClass::Scalar,
            suffix,
            if architecture == crate::target::Architecture::Gfx1250 {
                105
            } else {
                101
            },
        )
    } else {
        return Ok(None);
    };
    if suffix.is_empty()
        || (!suffix.starts_with('[') && !suffix.starts_with(|c: char| c.is_ascii_digit()))
    {
        return Ok(None);
    }
    let (index, end) =
        if let Some(range) = suffix.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            let (begin, end) = range.split_once(':').unwrap_or((range, range));
            (
                begin
                    .trim()
                    .parse::<u16>()
                    .map_err(|_| "invalid register index")?,
                end.trim()
                    .parse::<u16>()
                    .map_err(|_| "invalid register index")?,
            )
        } else {
            let index = suffix
                .parse::<u16>()
                .map_err(|_| "invalid register index")?;
            (index, index)
        };
    if end < index || end > limit {
        return Err(format!("invalid {class:?} register range `{text}`"));
    }
    let register = Register {
        class,
        index,
        count: end - index + 1,
        special: None,
    };
    validate_register_shape(register, architecture)
        .map_err(|message| format!("{message} `{text}`"))?;
    Ok(Some(register))
}

fn operand(text: &str, architecture: crate::target::Architecture) -> Result<Operand, String> {
    if let Some(value) = integer(text) {
        return Ok(Operand::Integer(value));
    }
    if let Some(register) = register(text, architecture)? {
        return Ok(Operand::Register(register));
    }
    if let Some((symbol, addend)) = text.split_once("@rel64") {
        if architecture != crate::target::Architecture::Gfx1250
            || symbol.is_empty()
            || symbol.starts_with(|c: char| c.is_ascii_digit())
            || !symbol
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "_.$".contains(c))
        {
            return Err(format!(
                "unsupported or malformed REL64 relocation `{text}`"
            ));
        }
        let addend = if addend.is_empty() {
            0
        } else if addend.starts_with(['+', '-']) {
            integer(addend).ok_or_else(|| format!("invalid REL64 addend `{text}`"))?
        } else {
            return Err(format!("invalid REL64 suffix `{text}`"));
        };
        return Ok(Operand::Relocation(Relocation {
            symbol: symbol.into(),
            kind: RelocationKind::Rel64,
            addend,
        }));
    }
    if let Some((symbol, suffix)) = text.split_once("@rel32@") {
        if symbol.is_empty()
            || symbol.starts_with(|c: char| c.is_ascii_digit())
            || !symbol
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || "_.$".contains(character))
        {
            return Err(format!("invalid relocation symbol `{symbol}`"));
        }
        let (kind, addend) = if let Some(rest) = suffix.strip_prefix("lo") {
            (RelocationKind::Rel32Lo, rest)
        } else if let Some(rest) = suffix.strip_prefix("hi") {
            (RelocationKind::Rel32Hi, rest)
        } else {
            return Err(format!("unsupported relocation `{text}`"));
        };
        let addend = if addend.is_empty() {
            0
        } else if addend.starts_with(['+', '-']) {
            integer(addend).ok_or_else(|| format!("invalid relocation addend `{text}`"))?
        } else {
            return Err(format!("invalid relocation suffix `{text}`"));
        };
        return Ok(Operand::Relocation(Relocation {
            symbol: symbol.into(),
            kind,
            addend,
        }));
    }
    if text.starts_with('-') || text.starts_with('|') || text.starts_with("abs(") {
        let (negate, rest) = text
            .strip_prefix('-')
            .map_or((false, text), |rest| (true, rest));
        let (absolute, inner) = if let Some(inner) =
            rest.strip_prefix('|').and_then(|s| s.strip_suffix('|'))
        {
            (true, inner)
        } else if let Some(inner) = rest.strip_prefix("abs(").and_then(|s| s.strip_suffix(')')) {
            (true, inner)
        } else {
            (false, rest)
        };
        if inner == text || inner.is_empty() {
            return Err(format!("malformed operand `{text}`"));
        }
        let value = operand(inner, architecture)?;
        if !matches!(
            value,
            Operand::Register(_) | Operand::Float(_) | Operand::Integer(_)
        ) {
            return Err("source modifiers require a register or literal".into());
        }
        return Ok(Operand::Modified {
            value: Box::new(value),
            negate,
            absolute,
        });
    }
    if (text.contains('.') || text.contains('e') || text.contains('E'))
        && let Ok(value) = text.parse::<f64>()
    {
        if !value.is_finite() {
            return Err("nonfinite float literal is unsupported".into());
        }
        return Ok(Operand::Float(value));
    }
    if text
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "_.$@".contains(c))
        && !text.is_empty()
        && !text.starts_with(|c: char| c.is_ascii_digit())
    {
        return Ok(Operand::Symbol(text.into()));
    }
    Err(format!("unsupported or malformed operand `{text}`"))
}

/// Explicit allowlist, not prefix dispatch. Recognition is not an execution claim:
/// the evaluator must still reject recognized instructions it cannot model.
pub(super) fn signature(mnemonic: &str) -> Option<(Opcode, usize, usize)> {
    use Opcode::*;
    let sdwa = mnemonic.ends_with("_sdwa");
    let (stem, encoding) = mnemonic.strip_suffix("_e32").map_or_else(
        || {
            mnemonic.strip_suffix("_e64").map_or_else(
                || {
                    mnemonic
                        .strip_suffix("_sdwa")
                        .map_or((mnemonic, false), |s| (s, true))
                },
                |s| (s, true),
            )
        },
        |s| (s, true),
    );
    let result = match stem {
        "s_endpgm" => (Control, 0, 1),
        "s_swappc_b64" => (Control, 2, 2),
        "s_setpc_b64" => (Control, 1, 1),
        "s_nop" | "s_sleep" | "s_sendmsg" | "s_setprio" | "s_trap" => (Control, 1, 1),
        "s_barrier" => (Barrier, 0, 0),
        "s_waitcnt" => (Wait, 1, 3),
        "s_waitcnt_vscnt" => (Wait, 2, 2),
        "s_branch" | "s_cbranch_scc0" | "s_cbranch_scc1" | "s_cbranch_vccz" | "s_cbranch_vccnz"
        | "s_cbranch_execz" | "s_cbranch_execnz" => (Branch, 1, 1),
        "s_getpc_b64" => (ScalarAlu, 1, 1),
        "s_cmpk_eq_u32" | "s_cmpk_lg_u32" | "s_cmpk_gt_u32" | "s_cmpk_lt_u32" | "s_addk_i32"
        | "s_cmpk_eq_i32" | "s_cmpk_lg_i32" | "s_mulk_i32" | "s_movk_i32" | "s_mov_b32"
        | "s_mov_b64" | "s_not_b32" | "s_not_b64" | "s_brev_b32" | "s_brev_b64"
        | "s_ff0_i32_b64" | "s_ff1_i32_b64" | "s_ff0_i32_b32" | "s_ff1_i32_b32"
        | "s_flbit_i32_b32" | "s_abs_i32" | "s_bitset1_b32" | "s_sext_i32_i16" => (ScalarAlu, 2, 2),
        "s_add_u32" | "s_add_i32" | "s_addc_u32" | "s_sub_u32" | "s_sub_i32" | "s_subb_u32"
        | "s_mul_i32" | "s_mul_hi_u32" | "s_mul_hi_i32" | "s_and_b32" | "s_and_b64"
        | "s_or_b32" | "s_or_b64" | "s_xor_b32" | "s_xor_b64" | "s_andn2_b32" | "s_andn2_b64"
        | "s_orn2_b32" | "s_orn2_b64" | "s_nand_b32" | "s_nand_b64" | "s_nor_b32" | "s_nor_b64"
        | "s_xnor_b32" | "s_xnor_b64" | "s_lshl_b32" | "s_lshl_b64" | "s_lshr_b32"
        | "s_lshr_b64" | "s_ashr_i32" | "s_ashr_i64" | "s_min_u32" | "s_min_i32" | "s_max_u32"
        | "s_max_i32" | "s_cselect_b32" | "s_cselect_b64" | "s_bfe_u32" | "s_bfe_i32"
        | "s_bfe_i64" | "s_bfm_b32" | "s_bfm_b64" => (ScalarAlu, 3, 3),
        "s_and_saveexec_b64"
        | "s_or_saveexec_b64"
        | "s_xor_saveexec_b64"
        | "s_andn2_saveexec_b64"
        | "s_orn2_saveexec_b64"
        | "s_nand_saveexec_b64"
        | "s_nor_saveexec_b64"
        | "s_xnor_saveexec_b64" => (ScalarAlu, 2, 2),
        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_gt_u32" | "s_cmp_ge_u32" | "s_cmp_lt_u32"
        | "s_cmp_le_u32" | "s_cmp_eq_i32" | "s_cmp_lg_i32" | "s_cmp_gt_i32" | "s_cmp_ge_i32"
        | "s_cmp_lt_i32" | "s_cmp_le_i32" | "s_cmp_eq_u64" | "s_cmp_lg_u64" | "s_bitcmp0_b32"
        | "s_bitcmp1_b32" => (ScalarAlu, 2, 2),
        "s_load_dword"
        | "s_load_dwordx2"
        | "s_load_dwordx4"
        | "s_load_dwordx8"
        | "s_load_dwordx16"
        | "s_buffer_load_dword"
        | "s_buffer_load_dwordx2"
        | "s_buffer_load_dwordx4"
        | "s_buffer_load_dwordx8"
        | "s_buffer_load_dwordx16" => (ScalarMemory, 3, 3),
        "global_atomic_add_f32"
        | "global_atomic_add"
        | "global_atomic_cmpswap"
        | "global_atomic_swap" => (VectorMemory, 3, 4),
        "global_load_ubyte"
        | "global_load_sbyte"
        | "global_load_ushort"
        | "global_load_sshort"
        | "global_load_dword"
        | "global_load_dwordx2"
        | "global_load_dwordx3"
        | "global_load_dwordx4"
        | "global_store_byte"
        | "global_store_short"
        | "global_store_dword"
        | "global_store_dwordx2"
        | "global_store_dwordx3"
        | "global_store_dwordx4" => (VectorMemory, 3, 3),
        "scratch_load_dword"
        | "scratch_store_dword"
        | "scratch_load_dwordx2"
        | "scratch_store_dwordx2"
        | "scratch_load_dwordx3"
        | "scratch_store_dwordx3"
        | "scratch_load_dwordx4"
        | "scratch_store_dwordx4" => (VectorMemory, 3, 3),
        "flat_load_ubyte" | "flat_load_sbyte" | "flat_load_ushort" | "flat_load_sshort"
        | "flat_load_dword" | "flat_load_dwordx2" | "flat_load_dwordx3" | "flat_load_dwordx4"
        | "flat_store_byte" | "flat_store_short" | "flat_store_dword" | "flat_store_dwordx2"
        | "flat_store_dwordx3" | "flat_store_dwordx4" => (VectorMemory, 2, 2),
        "buffer_load_b128" | "buffer_store_b64" => (VectorMemory, 4, 4),
        "buffer_load_dword" | "buffer_load_dwordx4" => (VectorMemory, 3, 4),
        "buffer_load_ubyte"
        | "buffer_load_sbyte"
        | "buffer_load_ushort"
        | "buffer_load_sshort"
        | "buffer_load_dwordx2"
        | "buffer_load_dwordx3"
        | "buffer_store_byte"
        | "buffer_store_short"
        | "buffer_store_dword"
        | "buffer_store_dwordx2"
        | "buffer_store_dwordx3"
        | "buffer_store_dwordx4" => (VectorMemory, 4, 4),
        "ds_write_b16_d16_hi"
        | "ds_write_b8_d16_hi"
        | "ds_read_b8"
        | "ds_read_u8"
        | "ds_read_i8"
        | "ds_read_b16"
        | "ds_read_u16"
        | "ds_read_i16"
        | "ds_read_b32"
        | "ds_read_b64"
        | "ds_read_b96"
        | "ds_read_b128"
        | "ds_write_b8"
        | "ds_write_b16"
        | "ds_write_b32"
        | "ds_write_b64"
        | "ds_write_b96"
        | "ds_write_b128"
        | "ds_swizzle_b32" => (Lds, 2, 2),
        "ds_read2_b32" | "ds_read2_b64" | "ds_read2st64_b32" | "ds_read2st64_b64" => (Lds, 2, 2),
        "ds_write2_b32" | "ds_write2_b64" | "ds_write2st64_b32" | "ds_write2st64_b64"
        | "ds_bpermute_b32" | "ds_permute_b32" => (Lds, 3, 3),
        "v_accvgpr_mov_b32"
        | "v_cvt_f32_fp8"
        | "v_cvt_f32_bf8"
        | "v_cvt_pk_f32_fp8"
        | "v_cvt_pk_f32_bf8"
        | "v_mov_b64"
        | "v_mov_b32"
        | "v_not_b32"
        | "v_bfrev_b32"
        | "v_ffbh_u32"
        | "v_ffbl_b32"
        | "v_floor_f32"
        | "v_ceil_f32"
        | "v_trunc_f32"
        | "v_fract_f32"
        | "v_rcp_f32"
        | "v_rcp_f64"
        | "v_sqrt_f32"
        | "v_sqrt_f64"
        | "v_rsq_f32"
        | "v_exp_f32"
        | "v_log_f32"
        | "v_sin_f32"
        | "v_cos_f32"
        | "v_cvt_f32_u32"
        | "v_cvt_f32_i32"
        | "v_cvt_f32_ubyte2"
        | "v_cvt_u32_f32"
        | "v_cvt_i32_f32"
        | "v_cvt_f16_f32"
        | "v_cvt_f16_i16"
        | "v_cvt_f32_f16"
        | "v_cvt_f32_bf16"
        | "v_cvt_bf16_f32"
        | "v_cvt_f64_f32"
        | "v_cvt_f32_f64"
        | "v_readfirstlane_b32"
        | "v_accvgpr_read_b32"
        | "v_accvgpr_write_b32" => (VectorAlu, 2, 2),
        "v_add_u16"
        | "v_sub_u16"
        | "v_mul_lo_u16"
        | "v_lshlrev_b16"
        | "v_lshrrev_b16"
        | "v_cmp_ne_u16"
        | "v_fmac_f32"
        | "v_fmac_f64"
        | "v_cvt_pk_f16_f32"
        | "v_pack_b32_f16"
        | "v_pk_mul_f32"
        | "v_pk_add_f32"
        | "v_pk_max_f16"
        | "v_pk_min_f16"
        | "v_sub_f16"
        | "v_max_f16"
        | "v_min_f16"
        | "v_mbcnt_lo_u32_b32"
        | "v_mbcnt_hi_u32_b32"
        | "v_add_u32"
        | "v_sub_u32"
        | "v_subrev_u32"
        | "v_add_f32"
        | "v_sub_f32"
        | "v_subrev_f32"
        | "v_mul_f32"
        | "v_mul_lo_u32"
        | "v_mul_u64"
        | "v_mul_lo_i32"
        | "v_mul_hi_u32"
        | "v_mul_hi_i32"
        | "v_mul_u32_u24"
        | "v_mul_i32_i24"
        | "v_and_b32"
        | "v_or_b32"
        | "v_xor_b32"
        | "v_lshlrev_b32"
        | "v_lshrrev_b32"
        | "v_ashrrev_i32"
        | "v_lshlrev_b64"
        | "v_lshrrev_b64"
        | "v_ashrrev_i64"
        | "v_min_f32"
        | "v_max_f32"
        | "v_min_u32"
        | "v_max_u32"
        | "v_min_i32"
        | "v_max_i32"
        | "v_ldexp_f32"
        | "v_readlane_b32"
        | "v_writelane_b32"
        | "v_cvt_pkrtz_f16_f32"
        | "v_cvt_pk_bf16_f32"
        | "v_add_f16"
        | "v_mul_f16"
        | "v_pk_add_f16"
        | "v_pk_mul_f16"
        | "v_add_f64"
        | "v_min_f64"
        | "v_max_f64"
        | "v_mul_f64" => (VectorAlu, 3, 3),
        "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" => (VectorAlu, 4, 4),
        "v_mad_u64_u32" | "v_mad_i64_i32" => (VectorAlu, 5, 5),
        "v_addc_co_u32" | "v_subb_co_u32" | "v_subbrev_co_u32" => (VectorAlu, 5, 5),
        "v_cndmask_b32" => (VectorAlu, if mnemonic.ends_with("_e32") { 3 } else { 4 }, 4),
        "v_fmamk_f32" | "v_or3_b32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" | "v_fma_mix_f32"
        | "v_pk_fma_f32" | "v_alignbit_b32" | "v_max3_f16" | "v_max3_f32" | "v_min3_f16"
        | "v_min3_f32" | "v_fma_f32" | "v_mad_f32" | "v_fma_f16" | "v_pk_fma_f16" | "v_fma_f64"
        | "v_mad_u32_u24" | "v_mad_i32_i24" | "v_add3_u32" | "v_add_lshl_u32"
        | "v_lshl_add_u32" | "v_lshl_add_u64" | "v_lshl_or_b32" | "v_and_or_b32" | "v_bfi_b32"
        | "v_bfe_u32" | "v_bfe_i32" | "v_perm_b32" | "v_bitop3_b32" | "v_bitop3_b16" => {
            (VectorAlu, 4, 4)
        }
        "v_mfma_scale_f32_16x16x128_f8f6f4" | "v_mfma_scale_f32_32x32x64_f8f6f4" => (Matrix, 6, 6),
        "v_mfma_f32_16x16x4_f32"
        | "v_mfma_f32_32x32x2_f32"
        | "v_mfma_f32_16x16x16_f16"
        | "v_mfma_f32_32x32x8_f16"
        | "v_mfma_f32_16x16x16_bf16"
        | "v_mfma_f32_32x32x8_bf16"
        | "v_mfma_f32_16x16x4f32"
        | "v_mfma_f32_32x32x2f32"
        | "v_mfma_f32_16x16x16f16"
        | "v_mfma_f32_32x32x8f16"
        | "v_mfma_f32_16x16x16bf16_1k"
        | "v_mfma_f32_32x32x8bf16_1k"
        | "v_mfma_f32_16x16x32_f16"
        | "v_mfma_f32_32x32x16_f16"
        | "v_mfma_f32_16x16x32_bf16"
        | "v_mfma_f32_32x32x16_bf16"
        | "v_mfma_f32_16x16x128_f8f6f4"
        | "v_mfma_f32_32x32x64_f8f6f4"
        | "v_mfma_i32_16x16x32_i8"
        | "v_mfma_i32_32x32x16_i8"
        | "v_mfma_f64_16x16x4f64" => (Matrix, 4, 4),
        _ => {
            // Compare relations and types are enumerated independently, avoiding
            // accidental acceptance of arbitrary `v_cmp_*` spellings.
            let comparison = stem
                .strip_prefix("v_cmp_")
                .or_else(|| stem.strip_prefix("v_cmpx_"));
            let comparison = comparison?;
            let (relation, ty) = comparison.rsplit_once('_')?;
            if ![
                "eq", "lg", "ne", "lt", "le", "gt", "ge", "o", "u", "nlt", "nle", "ngt", "nge",
                "neq", "nlg", "tru", "f",
            ]
            .contains(&relation)
                || !["u32", "i32", "f32", "u64", "i64", "f64", "f16"].contains(&ty)
            {
                return None;
            }
            (VectorAlu, 3, 3)
        }
    };
    if encoding && !matches!(result.0, VectorAlu) {
        return None;
    }
    if sdwa
        && ![
            "v_add_u32",
            "v_mul_lo_u16",
            "v_mul_u32_u24",
            "v_cvt_f32_fp8",
            "v_cvt_f32_bf8",
            "v_cvt_f32_f16",
            "v_cvt_f16_i16",
            "v_max_f16",
            "v_min_f16",
            "v_add_f16",
            "v_sub_f16",
            "v_mul_f16",
            "v_add_f32",
            "v_sub_f32",
            "v_mul_f32",
            "v_mov_b32",
        ]
        .contains(&stem)
    {
        return None;
    }
    Some(result)
}

fn parse_modifier(token: &str, opcode: Opcode, mnemonic: &str) -> Result<Modifier, String> {
    let (name, value) = token
        .split_once(':')
        .map_or((token, None), |(key, value)| (key, Some(value)));
    let memory = matches!(
        opcode,
        Opcode::ScalarMemory | Opcode::VectorMemory | Opcode::Lds
    );
    let alu = matches!(opcode, Opcode::VectorAlu | Opcode::Matrix);
    let permitted = match name {
        "offset" => memory,
        "offset0" | "offset1" => opcode == Opcode::Lds,
        "offen" | "idxen" | "addr64" | "lds" => mnemonic.starts_with("buffer_"),
        "glc" | "slc" | "dlc" | "sc0" | "sc1" | "nt" => {
            matches!(opcode, Opcode::ScalarMemory | Opcode::VectorMemory)
        }
        "gds" => opcode == Opcode::Lds,
        "clamp" | "mul" | "div" | "op_sel" | "op_sel_hi" | "neg_lo" | "neg_hi" => alu,
        "cbsz" | "abid" | "blgp" => opcode == Opcode::Matrix,
        // gfx1250 F8F6F4 formats, and block-scale lanes, formats and reuse
        // hints of the 4-dword scaled forms (LLVM syntax).
        "matrix_a_fmt" | "matrix_b_fmt" => {
            opcode == Opcode::Matrix
                && mnemonic.starts_with("v_wmma_")
                && mnemonic.ends_with("_f8f6f4")
        }
        "matrix_a_scale" | "matrix_b_scale" | "matrix_a_scale_fmt" | "matrix_b_scale_fmt"
        | "matrix_a_reuse" | "matrix_b_reuse" => {
            opcode == Opcode::Matrix
                && (mnemonic.starts_with("v_wmma_scale_")
                    || mnemonic.starts_with("v_wmma_scale16_"))
        }
        "bitop3" => matches!(mnemonic, "v_bitop3_b32" | "v_bitop2_b32" | "v_bitop3_b16"),
        "scale_offset" => matches!(opcode, Opcode::ScalarMemory | Opcode::VectorMemory),
        "byte_sel" => matches!(
            mnemonic.trim_end_matches("_e64"),
            "v_cvt_f32_fp8" | "v_cvt_f32_bf8"
        ),
        "scope" => {
            mnemonic == "global_atomic_add_f32"
                || mnemonic.starts_with("flat_")
                || mnemonic == "global_prefetch_b8"
        }
        "dst_sel" | "dst_unused" | "src0_sel" | "src1_sel" => mnemonic.ends_with("_sdwa"),
        _ => false,
    };
    if !permitted {
        return Err(format!("unsupported modifier `{token}` for `{mnemonic}`"));
    }
    let numeric = matches!(
        name,
        "offset"
            | "offset0"
            | "offset1"
            | "mul"
            | "div"
            | "cbsz"
            | "abid"
            | "blgp"
            | "bitop3"
            | "byte_sel"
    );
    if numeric && value.and_then(integer).is_none() {
        return Err(format!("modifier `{name}` requires an integer"));
    }
    if name == "scope"
        && mnemonic == "global_prefetch_b8"
        && !matches!(
            value,
            Some("SCOPE_CU" | "SCOPE_SE" | "SCOPE_DEV" | "SCOPE_SYS")
        )
    {
        return Err(
            "GLOBAL prefetch scope must be SCOPE_CU, SCOPE_SE, SCOPE_DEV or SCOPE_SYS".into(),
        );
    }
    if name == "scope"
        && mnemonic != "global_prefetch_b8"
        && value
            != Some(if mnemonic.starts_with("flat_") {
                "SCOPE_SYS"
            } else {
                "SCOPE_DEV"
            })
    {
        return Err("native atomic requires SCOPE_DEV; selected FLAT requires SCOPE_SYS".into());
    }
    if name == "byte_sel"
        && !value
            .and_then(integer)
            .is_some_and(|n| (0..=3).contains(&n))
    {
        return Err("FP8 byte_sel must be in 0..3".into());
    }
    if matches!(name, "offset0" | "offset1" | "bitop3")
        && !value
            .and_then(integer)
            .is_some_and(|n| (0..=255).contains(&n))
    {
        return Err(format!("modifier `{name}` must fit in 8 bits"));
    }
    if matches!(name, "cbsz" | "abid" | "blgp")
        && !value
            .and_then(integer)
            .is_some_and(|n| (0..=7).contains(&n))
    {
        return Err(format!("modifier `{name}` must fit in 3 bits"));
    }
    if matches!(name, "op_sel" | "op_sel_hi" | "neg_lo" | "neg_hi") {
        let content = value
            .and_then(|v| v.strip_prefix('['))
            .and_then(|v| v.strip_suffix(']'))
            .ok_or_else(|| format!("modifier `{name}` requires a bit list"))?;
        let bits = split_top_level(content, ',')?;
        if bits.is_empty() || bits.len() > 4 || bits.iter().any(|bit| !matches!(*bit, "0" | "1")) {
            return Err(format!("invalid bit list for `{name}`"));
        }
    }
    let allowed_values: &[&str] = match name {
        "matrix_a_fmt" | "matrix_b_fmt" => &[
            "MATRIX_FMT_FP8",
            "MATRIX_FMT_BF8",
            "MATRIX_FMT_FP6",
            "MATRIX_FMT_BF6",
            "MATRIX_FMT_FP4",
        ],
        "matrix_a_scale" | "matrix_b_scale" => &["MATRIX_SCALE_ROW0", "MATRIX_SCALE_ROW1"],
        "matrix_a_scale_fmt" | "matrix_b_scale_fmt" => &[
            "MATRIX_SCALE_FMT_E8",
            "MATRIX_SCALE_FMT_E5M3",
            "MATRIX_SCALE_FMT_E4M3",
        ],
        _ => &[],
    };
    if !allowed_values.is_empty() && !value.is_some_and(|value| allowed_values.contains(&value)) {
        return Err(format!("invalid value for `{name}` in `{token}`"));
    }
    if matches!(name, "dst_sel" | "src0_sel" | "src1_sel")
        && !value.is_some_and(|value| {
            [
                "BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD",
            ]
            .contains(&value)
        })
    {
        return Err(format!("invalid SDWA selector `{token}`"));
    }
    if name == "dst_unused"
        && !value
            .is_some_and(|value| ["UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE"].contains(&value))
    {
        return Err(format!("invalid SDWA destination behavior `{token}`"));
    }
    if !numeric
        && !matches!(
            name,
            "scope"
                | "op_sel"
                | "op_sel_hi"
                | "neg_lo"
                | "neg_hi"
                | "dst_sel"
                | "dst_unused"
                | "src0_sel"
                | "src1_sel"
                | "matrix_a_fmt"
                | "matrix_b_fmt"
                | "matrix_a_scale"
                | "matrix_b_scale"
                | "matrix_a_scale_fmt"
                | "matrix_b_scale_fmt"
        )
        && value.is_some()
    {
        return Err(format!("flag modifier `{name}` does not take a value"));
    }
    Ok(Modifier {
        name: name.into(),
        value: value.map(str::to_owned),
    })
}

pub(super) fn parse_instruction(
    text: &str,
    span: SourceSpan,
    architecture: crate::target::Architecture,
) -> Result<Instruction, ParseError> {
    if architecture == crate::target::Architecture::Gfx1250
        && let Some(result) = super::gfx1250::parse_special(text, span)?
    {
        return Ok(result);
    }
    let fail = |message| ParseError { message, span };
    let (mnemonic, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
    if architecture == crate::target::Architecture::Gfx1250 {
        super::gfx1250::validate_encoding(mnemonic).map_err(fail)?;
    }
    let (opcode, mut minimum, mut maximum) = signature(mnemonic)
        .or_else(|| {
            super::gfx1250::signature(mnemonic)
                .filter(|_| architecture == crate::target::Architecture::Gfx1250)
        })
        .ok_or_else(|| fail(format!("unsupported AMD instruction `{mnemonic}`")))?;
    if architecture == crate::target::Architecture::Gfx1250 && mnemonic.starts_with("v_cmpx_") {
        minimum = 2;
        maximum = 2;
    }
    let mut operands = Vec::new();
    let mut modifiers = Vec::new();
    for group in split_top_level(rest, ',').map_err(fail)? {
        for token in whitespace_tokens(group).map_err(fail)? {
            if let Some(inner) = token
                .strip_prefix("sext(")
                .and_then(|s| s.strip_suffix(')'))
            {
                if mnemonic != "v_cvt_f16_i16_sdwa" || operands.len() != 1 || !modifiers.is_empty()
                {
                    return Err(fail("unsupported source sign-extension form".into()));
                }
                let value = operand(inner, architecture).map_err(fail)?;
                if !matches!(&value, Operand::Register(register) if register.class == RegisterClass::Vector && register.count == 1)
                {
                    return Err(fail("SDWA sign extension requires one VGPR".into()));
                }
                operands.push(value);
                // A semantic SDWA source flag, not a separately accepted token.
                modifiers.push(Modifier {
                    name: "src0_sext".into(),
                    value: None,
                });
                continue;
            }
            if opcode == Opcode::Wait
                && mnemonic == "s_waitcnt"
                && let Some((name, value)) = token
                    .split_once('(')
                    .and_then(|(name, tail)| tail.strip_suffix(')').map(|value| (name, value)))
            {
                let limit = match name {
                    "vmcnt" => 63,
                    "lgkmcnt" => 15,
                    "expcnt" => 7,
                    _ => return Err(fail(format!("unsupported wait counter `{name}`"))),
                };
                let value = integer(value)
                    .filter(|n| (0..=limit).contains(n))
                    .ok_or_else(|| fail(format!("invalid `{name}` threshold")))?;
                if operands.iter().any(|operand| matches!(operand, Operand::WaitCounter { name: old, .. } if old == name)) { return Err(fail(format!("duplicate wait counter `{name}`"))); }
                operands.push(Operand::WaitCounter {
                    name: name.into(),
                    value: value as u32,
                });
                continue;
            }
            if token.split_once(':').is_some_and(|(name, _)| {
                !name.is_empty()
                    && name
                        .chars()
                        .all(|character| character.is_ascii_alphanumeric() || character == '_')
            }) || [
                "offen",
                "idxen",
                "addr64",
                "lds",
                "glc",
                "slc",
                "dlc",
                "sc0",
                "sc1",
                "nt",
                "gds",
                "clamp",
                "scale_offset",
            ]
            .contains(&token)
                // gfx1250 block-scaled WMMA reuse hints (refused later by
                // every profile; other mnemonics keep their parse errors).
                || (matches!(token, "matrix_a_reuse" | "matrix_b_reuse")
                    && (mnemonic.starts_with("v_wmma_scale_")
                        || mnemonic.starts_with("v_wmma_scale16_")))
            {
                let modifier = parse_modifier(token, opcode, mnemonic).map_err(fail)?;
                if modifiers
                    .iter()
                    .any(|old: &Modifier| old.name == modifier.name)
                {
                    return Err(fail(format!("duplicate modifier `{}`", modifier.name)));
                }
                modifiers.push(modifier);
            } else {
                if !modifiers.is_empty() {
                    return Err(fail("operand after instruction modifier".into()));
                }
                operands.push(operand(token, architecture).map_err(fail)?);
            }
        }
    }
    if !(minimum..=maximum).contains(&operands.len()) {
        return Err(fail(format!(
            "`{mnemonic}` expects {minimum}{} operands, got {}",
            if minimum == maximum {
                String::new()
            } else {
                format!("..={maximum}")
            },
            operands.len()
        )));
    }
    validate_post_parse(mnemonic, opcode, &operands, &modifiers, architecture, span)?;
    let mut instruction = Instruction {
        mnemonic: mnemonic.into(),
        opcode,
        operands,
        modifiers,
        span,
    };
    normalize_implicit_operands(&mut instruction);
    Ok(instruction)
}

/// Canonical explicit operands for syntax with an implicit VCC selector.
fn normalize_implicit_operands(instruction: &mut Instruction) {
    if instruction.mnemonic == "v_cndmask_b32_e32" && instruction.operands.len() == 3 {
        instruction.operands.push(Operand::Register(Register {
            class: RegisterClass::Special,
            index: 0,
            count: 2,
            special: Some(SpecialRegister::Vcc),
        }));
    }
}

pub(super) fn normalize_for_lowering(instruction: &mut Instruction) -> Result<(), ParseError> {
    if let [Operand::Dual(x, y)] = instruction.operands.as_mut_slice() {
        normalize_for_lowering(x)?;
        normalize_for_lowering(y)?;
    }
    normalize_implicit_operands(instruction);
    normalize_f64_sources(instruction)
}

pub(super) fn valid_alu_delay(encoded: i64) -> bool {
    encoded >= 0
        && encoded >> 11 == 0
        && encoded & 15 <= 11
        && (encoded >> 4) & 7 <= 5
        && (encoded >> 7) & 15 <= 11
}

/// Instruction-form checks shared by text parsing and direct AST lowering.
fn validate_post_parse(
    mnemonic: &str,
    opcode: Opcode,
    operands: &[Operand],
    modifiers: &[Modifier],
    architecture: crate::target::Architecture,
    span: SourceSpan,
) -> Result<(), ParseError> {
    let fail = |message| ParseError { message, span };
    if matches!(mnemonic, "s_swappc_b64" | "s_setpc_b64")
        && architecture != crate::target::Architecture::Gfx950
    {
        return Err(fail(
            "device call/return is supported only in the selected gfx950 profile".into(),
        ));
    }
    if mnemonic == "s_waitcnt" {
        let mut seen = std::collections::BTreeSet::new();
        for operand in operands {
            if let Operand::WaitCounter { name, value } = operand {
                let limit = match name.as_str() {
                    "vmcnt" => 63,
                    "lgkmcnt" => 15,
                    "expcnt" => 7,
                    _ => return Err(fail(format!("unsupported wait counter `{name}`"))),
                };
                if *value > limit {
                    return Err(fail(format!("invalid `{name}` threshold")));
                }
                if !seen.insert(name) {
                    return Err(fail(format!("duplicate wait counter `{name}`")));
                }
            }
        }
    }
    if opcode == Opcode::Branch && !matches!(operands.first(), Some(Operand::Symbol(_))) {
        return Err(fail("branch target must be a label".into()));
    }
    if opcode == Opcode::Wait
        && mnemonic == "s_waitcnt"
        && !operands
            .iter()
            .all(|operand| matches!(operand, Operand::WaitCounter { .. }))
        && !(operands.len() == 1
            && matches!(operands[0], Operand::Integer(value) if (0..=0xffff).contains(&value)))
    {
        return Err(fail(
            "s_waitcnt requires counters or one 16-bit immediate".into(),
        ));
    }
    validate_registers(mnemonic, opcode, operands, modifiers, architecture).map_err(fail)?;
    validate_cdna_integer16(mnemonic, operands, modifiers, architecture).map_err(fail)?;
    if modifiers.iter().any(|m| {
        if architecture == crate::target::Architecture::Gfx1250 {
            matches!(
                m.name.as_str(),
                "glc" | "slc" | "dlc" | "sc0" | "sc1" | "nt"
            )
        } else {
            matches!(m.name.as_str(), "scale_offset" | "scope")
        }
    }) {
        return Err(fail(
            "memory cache/scale modifier is outside this target's native instruction profile"
                .into(),
        ));
    }
    validate_vop_literal_positions(mnemonic, operands, modifiers, architecture).map_err(fail)?;
    if mnemonic == "v_ashrrev_i64_e32" {
        return Err(fail("v_ashrrev_i64 has only a VOP3 encoding".into()));
    }
    if ["v_cvt_f32_fp8_sdwa", "v_cvt_f32_bf8_sdwa"].contains(&mnemonic)
        && modifiers
            .iter()
            .any(|modifier| matches!(modifier.name.as_str(), "dst_sel" | "dst_unused"))
    {
        return Err(fail(
            "FP8/BF8-to-F32 SDWA conversion has no destination selector fields".into(),
        ));
    }
    if matches!(mnemonic, "v_bitop3_b32" | "v_bitop3_b16")
        && (modifiers.len() != 1
            || modifiers[0].name != "bitop3"
            || operands
                .iter()
                .any(|operand| !matches!(operand, Operand::Register(_) | Operand::Integer(_))))
    {
        return Err(fail(
            "v_bitop3_b32 requires one explicit truth table and unmodified bit operands".into(),
        ));
    }
    // MFMA F8F6F4 (cbsz/blgp formats); the gfx1250 WMMA F8F6F4 forms carry
    // their formats in `matrix_{a,b}_fmt` (checked with their registers).
    if opcode == Opcode::Matrix && mnemonic.ends_with("f8f6f4") && !mnemonic.starts_with("v_wmma_")
    {
        for (operand_index, modifier_name) in [(1, "cbsz"), (2, "blgp")] {
            let format = modifiers
                .iter()
                .find(|modifier| modifier.name == modifier_name)
                .and_then(|modifier| modifier.value.as_deref())
                .and_then(integer)
                .unwrap_or(0);
            let width = match format {
                0 | 1 => 8,
                2 | 3 => 6,
                4 => 4,
                _ => return Err(fail(format!("reserved f8f6f4 input format `{format}`"))),
            };
            if !matches!(operands.get(operand_index), Some(Operand::Register(register)) if register.count == width)
            {
                return Err(fail(format!(
                    "input format {format} requires {width} source registers"
                )));
            }
        }
        if operands.len() == 6 && operands[4..].iter().any(|operand| !matches!(operand, Operand::Register(register) if register.class == RegisterClass::Vector && register.count == 1)) { return Err(fail("scaled MFMA requires one VGPR per scale source".into())); }
    }
    Ok(())
}

/// Selected CDNA integer16 encodings, shared by parsed and caller-built ASTs.
/// No source modifiers, OPSEL, saturation or literal-bearing VOP3 variants are
/// modeled here. The gfx1250 true16 profile has its own encoding checks.
fn validate_cdna_integer16(
    mnemonic: &str,
    operands: &[Operand],
    modifiers: &[Modifier],
    architecture: crate::target::Architecture,
) -> Result<(), String> {
    use RegisterClass::{Scalar, Special, Vector};
    if architecture == crate::target::Architecture::Gfx1250 {
        return Ok(());
    }
    let stem = mnemonic.trim_end_matches("_e32").trim_end_matches("_e64");
    if stem == "v_cmp_u_f32" {
        // The shared finite-real form is used with encoded register sources.
        // Keep the native VOPC fixed destination and SRC1 restriction too.
        if !modifiers.is_empty() || operands.iter().skip(1).any(|source| {
            !matches!(source, Operand::Register(r) if matches!(r.class, Scalar | Vector) && r.count == 1)
        }) || (mnemonic.ends_with("_e32")
            && (!matches!(operands.get(2), Some(Operand::Register(r)) if r.class == Vector)
                || !matches!(operands.first(), Some(Operand::Register(r))
                    if r.class == Special && r.special == Some(SpecialRegister::Vcc))))
        {
            return Err("selected CDNA unordered comparison requires native register operands".into());
        }
        let mut scalar = None;
        for source in operands.iter().skip(1) {
            if let Operand::Register(r) = source
                && r.class == Scalar
            {
                if scalar.is_some_and(|previous| previous != *r) {
                    return Err("CDNA comparison reads more than one scalar source".into());
                }
                scalar = Some(*r);
            }
        }
        return Ok(());
    }
    if !matches!(stem, "v_lshlrev_b16" | "v_cmp_ne_u16" | "v_bitop3_b16") {
        return Ok(());
    }
    let bitop = stem == "v_bitop3_b16";
    if bitop && mnemonic.ends_with("_e32") {
        return Err("V_BITOP3_B16 requires VOP3 encoding".into());
    }
    if (!bitop && !modifiers.is_empty())
        || (bitop && (modifiers.len() != 1 || modifiers[0].name != "bitop3"))
    {
        return Err("selected CDNA integer16 form does not support these modifiers".into());
    }
    let src1_vgpr =
        matches!(operands.get(2), Some(Operand::Register(r)) if r.class == Vector && r.count == 1);
    if mnemonic.ends_with("_e32")
        && (!src1_vgpr
            || (stem == "v_cmp_ne_u16"
                && !matches!(operands.first(), Some(Operand::Register(r))
                    if r.class == Special && r.special == Some(SpecialRegister::Vcc))))
    {
        return Err("CDNA integer16 e32 requires VGPR SRC1 and the native destination".into());
    }
    let vop3 = bitop || mnemonic.ends_with("_e64") || !src1_vgpr;
    let mut scalar_source = None;
    for (index, source) in operands.iter().enumerate().skip(1) {
        match source {
            Operand::Register(r)
                if matches!(r.class, Vector | Scalar | Special) && r.count == 1 =>
            {
                if r.class != Vector {
                    if scalar_source.is_some_and(|previous| previous != *r) {
                        return Err("CDNA integer16 form reads more than one scalar source".into());
                    }
                    scalar_source = Some(*r);
                }
            }
            Operand::Integer(n) if (i16::MIN as i64..=u16::MAX as i64).contains(n) => {
                let inline = (-16..=64).contains(n)
                    || (65520..=65535).contains(n)
                    || [
                        0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x4400, 0xc400, 0x3118,
                    ]
                    .contains(n);
                if !inline && (vop3 || index != 1 || scalar_source.is_some()) {
                    return Err("CDNA integer16 literal requires VOP2/VOPC SRC0".into());
                }
            }
            _ => return Err(
                "CDNA integer16 source requires one unmodified register or encoded16bit integer"
                    .into(),
            ),
        }
    }
    Ok(())
}

/// Validate controls after assembly constants have been resolved. SOPP source
/// fields are immediates, never register operands. For controls which use only
/// part of SIMM16, accept its canonical meaningful bits: LLVM also accepts some
/// larger/negative values and truncates them, but those aliases are outside our
/// supported syntax. CDNA4 ISA section 12.5 specifies the 4/7/2-bit fields below.
pub(super) fn validate_control(instruction: &Instruction) -> Result<(), ParseError> {
    if instruction.opcode != Opcode::Control {
        return Ok(());
    }
    if matches!(
        instruction.mnemonic.as_str(),
        "s_swappc_b64" | "s_setpc_b64"
    ) {
        let arity = if instruction.mnemonic == "s_swappc_b64" {
            2
        } else {
            1
        };
        if instruction.operands.len() != arity
            || !instruction.modifiers.is_empty()
            || instruction.operands.iter().any(|operand| {
                !matches!(operand,
                Operand::Register(register) if register.class == RegisterClass::Scalar
                    && register.count == 2 && register.special.is_none())
            })
        {
            return Err(ParseError {
                message: "device call/return requires plain SGPR pairs and no modifiers".into(),
                span: instruction.span,
            });
        }
        return Ok(());
    }
    if instruction.mnemonic == "s_delay_alu"
        && instruction
            .operands
            .iter()
            .any(|value| !matches!(value, Operand::Integer(encoded) if valid_alu_delay(*encoded)))
    {
        return Err(ParseError {
            message: "invalid s_delay_alu encoding".into(),
            span: instruction.span,
        });
    }
    let maximum = match instruction.mnemonic.as_str() {
        "s_nop" => 15,
        "s_clause" => 63,
        "s_delay_alu" => 0x7ff,
        "v_nop" => 0,
        "s_sleep" => 127,
        "s_setprio" => 3,
        // CDNA4 SOPP S_TRAP uses SIMM16[7:0]. LLVM also accepts truncating
        // aliases; our recognized syntax preserves only the actual8bit ID.
        // Entering a trap handler remains explicitly unsupported by execution.
        "s_trap" => 255,
        // The gfx950 assembler accepts an optional unsigned SIMM16 on ENDPGM;
        // omitting it is the ordinary compiler-generated spelling.
        "s_endpgm" | "s_sendmsg" => 0xffff,
        _ => {
            return Err(ParseError {
                message: format!("unsupported control instruction `{}`", instruction.mnemonic),
                span: instruction.span,
            });
        }
    };
    if instruction
        .operands
        .iter()
        .any(|operand| !matches!(operand, Operand::Integer(value) if (0..=maximum).contains(value)))
    {
        return Err(ParseError {
            message: format!(
                "`{}` requires an integer immediate in 0..={maximum}",
                instruction.mnemonic
            ),
            span: instruction.span,
        });
    }
    Ok(())
}

/// Canonicalize only encodable FP64 source constants. On gfx950 a literal word
/// occupies the high 32 bits of a double; it is not a raw 64-bit immediate.
/// LLVM accepts decimal constants with truncation, which is intentionally
/// unsupported here rather than inventing precision absent from the encoding.
pub(super) fn normalize_f64_sources(instruction: &mut Instruction) -> Result<(), ParseError> {
    let name = instruction
        .mnemonic
        .trim_end_matches("_e32")
        .trim_end_matches("_e64");
    if instruction.opcode != Opcode::VectorAlu || !name.ends_with("_f64") {
        return Ok(());
    }
    let inline = |value: f64| {
        value == 0.0 && !value.is_sign_negative() || [0.5_f64, 1.0, 2.0, 4.0].contains(&value.abs())
    };
    fn normalize(
        operand: &mut Operand,
        literal: bool,
        inline: &impl Fn(f64) -> bool,
    ) -> Result<(), String> {
        match operand {
            Operand::Register(_) => Ok(()),
            Operand::Modified { value, .. } => normalize(value, false, inline),
            Operand::Integer(value) => {
                if (-16..=64).contains(value) {
                    return Ok(());
                }
                let bits = *value as u64;
                // LLVM recognizes both the high-word and full-width spelling
                // of the floating inline constants. Preserve its canonical kind.
                let float = f64::from_bits(if bits <= u32::MAX as u64 {
                    bits << 32
                } else {
                    bits
                });
                if inline(float) {
                    *operand = Operand::Float(float);
                    Ok(())
                } else if literal && (i32::MIN as i64..=u32::MAX as i64).contains(value) {
                    Ok(())
                } else {
                    Err("FP64 source is not an encodable inline constant or literal word".into())
                }
            }
            Operand::Float(value) => {
                if inline(*value) {
                    Ok(())
                } else if literal && value.to_bits() & 0xffff_ffff == 0 {
                    *operand = Operand::Integer((value.to_bits() >> 32) as i64);
                    Ok(())
                } else {
                    Err(
                        "FP64 decimal source requires unsupported truncation or literal encoding"
                            .into(),
                    )
                }
            }
            _ => Err("FP64 source must be a register or encodable immediate".into()),
        }
    }
    let literal_src0 = ["v_rcp_f64", "v_sqrt_f64", "v_cvt_f32_f64", "v_fmac_f64"].contains(&name)
        && !instruction.mnemonic.ends_with("_e64")
        || (name.starts_with("v_cmp_") || name.starts_with("v_cmpx_"))
            && !instruction.mnemonic.ends_with("_e64");
    for (index, operand) in instruction.operands.iter_mut().enumerate().skip(1) {
        // The lexer represents a leading '-' uniformly as a source modifier.
        // A negative decimal inline constant is a native e32 immediate too.
        let negative_literal = match operand {
            Operand::Modified {
                value,
                negate: true,
                absolute: false,
            } if matches!(value.as_ref(), Operand::Float(_)) => {
                let Operand::Float(value) = value.as_ref() else {
                    unreachable!()
                };
                Some(-*value)
            }
            _ => None,
        };
        if let Some(value) = negative_literal {
            *operand = Operand::Float(value);
        }
        if instruction.mnemonic.ends_with("_e32")
            && (matches!(operand, Operand::Modified { .. })
                || index == 2
                    && (name == "v_fmac_f64" || name.starts_with("v_cmp"))
                    && !matches!(operand, Operand::Register(register) if register.class == RegisterClass::Vector))
        {
            return Err(ParseError {
                message: "FP64 e32 source requires its native register/inline operand form".into(),
                span: instruction.span,
            });
        }
        normalize(operand, literal_src0 && index == 1, &inline).map_err(|message| ParseError {
            message,
            span: instruction.span,
        })?;
    }
    Ok(())
}

pub(super) fn validate_registers(
    mnemonic: &str,
    opcode: Opcode,
    operands: &[Operand],
    modifiers: &[Modifier],
    architecture: crate::target::Architecture,
) -> Result<(), String> {
    use RegisterClass::*;
    let stem = mnemonic
        .strip_suffix("_e32")
        .or_else(|| mnemonic.strip_suffix("_e64"))
        .or_else(|| mnemonic.strip_suffix("_sdwa"))
        .unwrap_or(mnemonic);
    fn register(operand: &Operand) -> Option<Register> {
        match operand {
            Operand::Register(register) => Some(*register),
            Operand::Modified { value, .. } => register(value),
            _ => None,
        }
    }
    let register_at = |index: usize| operands.get(index).and_then(register);
    let require = |index: usize, classes: &[RegisterClass], count: u16| -> Result<(), String> {
        let register = register_at(index)
            .ok_or_else(|| format!("operand {} of `{mnemonic}` must be a register", index + 1))?;
        if !classes.contains(&register.class) || register.count != count {
            return Err(format!(
                "invalid register class/width at operand {} of `{mnemonic}`: expected {count} register(s)",
                index + 1
            ));
        }
        Ok(())
    };
    let source_width = |index: usize, count: u16| -> Result<(), String> {
        if register_at(index).is_some_and(|register| register.count != count) {
            return Err(format!(
                "invalid source register width at operand {} of `{mnemonic}`: expected {count} register(s)",
                index + 1
            ));
        }
        Ok(())
    };
    // SCC is an implicit condition bit, never a general-purpose encoded source
    // or destination. Treating it as an SGPR would accept nonexistent programs.
    if operands
        .iter()
        .filter_map(register)
        .any(|register| register.special == Some(SpecialRegister::Scc))
    {
        return Err("SCC cannot be used as an explicit general-purpose register operand".into());
    }
    for (index, operand) in operands.iter().enumerate() {
        if architecture == crate::target::Architecture::Gfx1250
            && register(operand).is_some_and(|r| r.special == Some(SpecialRegister::M0))
            && index == 0
            && !matches!(
                stem,
                "s_mov_b32" | "s_barrier_init" | "s_barrier_join" | "s_barrier_signal"
            )
        {
            return Err("gfx1250 M0 writes are modeled only for explicit s_mov_b32".into());
        }
        if register(operand)
            .is_some_and(|register| register.special == Some(SpecialRegister::SharedBase))
        {
            // CDNA4 selector235 and LLVM Generic address-space ABI: the base
            // is2^32 aligned. Numeric values remain unknown to this frontend.
            if mnemonic != "s_mov_b64" || index != 1 || !matches!(operand, Operand::Register(_)) {
                return Err("SRC_SHARED_BASE is modeled only as the source of s_mov_b64".into());
            }
            require(0, &[Scalar], 2)?;
        }
    }
    if operands.iter().any(|operand| {
        register(operand).is_some_and(|r| r.special == Some(SpecialRegister::PrivateBase))
    }) {
        if architecture != crate::target::Architecture::Gfx950
            || mnemonic != "s_mov_b64"
            || !matches!(operands.get(1), Some(Operand::Register(r)) if r.special == Some(SpecialRegister::PrivateBase))
        {
            return Err(
                "SRC_PRIVATE_BASE is modeled only as the gfx950 source of s_mov_b64".into(),
            );
        }
        require(0, &[Scalar], 2)?;
    }
    if stem.starts_with("tensor_") {
        for (index, count) in [4, 8, 4, 4].into_iter().enumerate() {
            require(index, &[Scalar], count)?;
        }
        return Ok(());
    }
    match opcode {
        Opcode::Barrier if architecture == crate::target::Architecture::Gfx1250 => {
            match (stem, operands) {
                ("s_barrier_leave", []) => {},
                ("s_barrier_signal" | "s_barrier_wait", [Operand::Integer(id)]) if *id == -1 || (0..=16).contains(id) => {},
                ("s_barrier_init" | "s_barrier_join", [Operand::Integer(id)]) if (0..=16).contains(id) => {},
                ("s_barrier_init" | "s_barrier_join" | "s_barrier_signal", [Operand::Register(r)]) if r.special == Some(SpecialRegister::M0) => {},
                _ => return Err("unsupported gfx1250 barrier operand; shader IDs are -1 or 0..16, with M0 only on init/join/signal".into()),
            }
        }
        // gfx1250 MODE writes have no SGPR destination: SETREG writes the
        // hwreg field named by its first operand, ROUND/DENORM_MODE take a
        // 16-bit immediate. Their shapes are fixed by the gfx1250 parser.
        Opcode::ScalarAlu
            if architecture == crate::target::Architecture::Gfx1250
                && matches!(
                    stem,
                    "s_setreg_b32" | "s_setreg_imm32_b32" | "s_round_mode" | "s_denorm_mode"
                ) =>
        {
            let shape = match (stem, operands) {
                ("s_setreg_b32", [Operand::HardwareRegister { .. }, source]) => {
                    register(source).is_some_and(|r| r.class == Scalar && r.count == 1)
                }
                ("s_setreg_imm32_b32", [Operand::HardwareRegister { .. }, Operand::Integer(_)]) => {
                    true
                }
                ("s_round_mode" | "s_denorm_mode", [Operand::Integer(value)]) => {
                    (0..=0xffff).contains(value)
                }
                _ => false,
            };
            if !shape || !modifiers.is_empty() {
                return Err(format!("invalid operands for `{mnemonic}`"));
            }
        }
        Opcode::ScalarAlu => {
            // SOPK has a dedicated immediate field, not a general scalar
            // source. LLVM's s16imm accepts signed values or their u16 bits;
            // unsigned CMPK instead requires an actual nonnegative u16.
            if stem.starts_with("s_cmpk_")
                || matches!(
                    stem,
                    "s_movk_i32" | "s_addk_i32" | "s_addk_co_i32" | "s_mulk_i32"
                )
            {
                if architecture == crate::target::Architecture::Gfx1250
                    && stem.starts_with("s_cmpk_")
                {
                    return Err("SOPK compare instructions have no gfx1250 encoding".into());
                }
                let minimum = if stem.starts_with("s_cmpk_") && stem.ends_with("_u32") {
                    0
                } else {
                    -32768
                };
                if !matches!(operands.get(1), Some(Operand::Integer(value)) if (minimum..=65535).contains(value))
                {
                    return Err(format!(
                        "`{mnemonic}` requires a 16-bit integer immediate in {minimum}..=65535"
                    ));
                }
            }
            if operands
                .iter()
                .any(|operand| matches!(operand, Operand::Modified { .. }))
            {
                return Err("scalar ALU operands do not support vector source modifiers".into());
            }
            for operand in operands {
                if register(operand)
                    .is_some_and(|register| matches!(register.class, Vector | Accumulator))
                {
                    return Err("scalar instruction cannot access vector registers".into());
                }
            }
            let compare = stem.starts_with("s_cmp_")
                || stem.starts_with("s_cmpk_")
                || stem.starts_with("s_bitcmp");
            let wide = stem.ends_with("b64") || stem.ends_with("i64") || stem.ends_with("u64");
            let data_width = if wide { 2 } else { 1 };
            if let Some(layout) = crate::source_encoding::ScalarLayout::for_mnemonic(stem) {
                require(0, &[Scalar, Special], 2)?;
                for index in 1..operands.len() {
                    let ty = layout
                        .source_type(index)
                        .ok_or("invalid scalar source position")?;
                    source_width(index, ty.words())?;
                }
                crate::source_encoding::resolve(mnemonic, operands, architecture, true)?;
            } else if compare {
                for index in 0..operands.len() {
                    source_width(index, data_width)?;
                }
            } else {
                let destination_width = if stem.starts_with("s_ff") {
                    1
                } else {
                    data_width
                };
                require(0, &[Scalar, Special], destination_width)?;
                for index in 1..operands.len() {
                    let shift =
                        index == 2 && (stem.starts_with("s_lsh") || stem.starts_with("s_ashr"));
                    let bfm = stem.starts_with("s_bfm_");
                    let bfe_control = stem == "s_bfe_i64" && index == 2;
                    source_width(
                        index,
                        if shift || bfm || bfe_control {
                            1
                        } else {
                            data_width
                        },
                    )?;
                }
            }
            if stem == "s_bfe_i64"
                && (!matches!(operands[0], Operand::Register(r) if r.class == Scalar && r.count == 2)
                    || !matches!(operands[1], Operand::Register(r) if r.class == Scalar && r.count == 2)
                        && !matches!(operands[1], Operand::Integer(value) if (-16..=64).contains(&value))
                    || !matches!(operands[2], Operand::Register(r) if r.class == Scalar && r.count == 1)
                        && !matches!(operands[2], Operand::Integer(value) if (i32::MIN as i64..=u32::MAX as i64).contains(&value))
                    || !modifiers.is_empty())
            {
                return Err("selected s_bfe_i64 requires a plain SGPR destination pair, SGPR data pair/integer inline, and one 32-bit SGPR/integer control".into());
            }
        }
        Opcode::ScalarMemory => {
            if stem.starts_with("s_load_") {
                let limit = crate::target::scalar_memory_offset_max(architecture);
                if !matches!(operands.get(2), Some(Operand::Integer(value)) if (0..=limit).contains(value))
                    && !matches!(operands.get(2), Some(Operand::Register(r))
                        if r.count == 1 && (r.class == Scalar || r.special == Some(SpecialRegister::M0)))
                {
                    return Err(format!(
                        "scalar load offset requires a nonnegative encoded immediate at most {limit:#x}, one SGPR or M0"
                    ));
                }
                if modifiers.iter().any(|modifier| modifier.name == "offset")
                    && matches!(operands.get(2), Some(Operand::Integer(_)))
                {
                    return Err("scalar load has one immediate offset; an additional offset requires a register source".into());
                }
            }
            if architecture == crate::target::Architecture::Gfx1250
                && modifiers
                    .iter()
                    .any(|m| matches!(m.name.as_str(), "offset" | "scale_offset"))
                && !matches!(operands.get(2),Some(Operand::Register(r)) if r.class==Scalar&&r.count==1)
            {
                return Err("gfx1250 optional/scaled SMEM offset requires a numbered SGPR source; immediate-only scaling is outside the supported profile".into());
            }
            require(0, &[Scalar], memory_width(stem))?;
            require(
                1,
                &[Scalar],
                if stem.starts_with("s_buffer_") { 4 } else { 2 },
            )?;
            source_width(2, 1)?;
            if register_at(2).is_some_and(|register| !matches!(register.class, Scalar | Special)) {
                return Err("scalar memory offset must use a scalar register".into());
            }
        }
        Opcode::VectorMemory => {
            if stem.starts_with("scratch_") {
                let load = stem.starts_with("scratch_load_");
                let words = memory_width(stem);
                let data = usize::from(!load);
                let off = usize::from(load);
                if architecture != crate::target::Architecture::Gfx950
                    || mnemonic != stem
                    || !matches!(
                        stem,
                        "scratch_load_dword"
                            | "scratch_store_dword"
                            | "scratch_load_dwordx2"
                            | "scratch_store_dwordx2"
                            | "scratch_load_dwordx3"
                            | "scratch_store_dwordx3"
                            | "scratch_load_dwordx4"
                            | "scratch_store_dwordx4"
                    )
                    || !matches!(&operands[off], Operand::Symbol(value) if value == "off")
                    || !matches!(operands[data], Operand::Register(r) if r.class == Vector && r.count == words)
                    || !matches!(operands[2], Operand::Register(r) if r.class == Scalar && r.count == 1)
                    || modifiers
                        .iter()
                        .any(|modifier| match modifier.name.as_str() {
                            "offset" => modifier
                                .value
                                .as_deref()
                                .and_then(integer)
                                .is_none_or(|offset| !(-4096..=4095).contains(&offset)),
                            "sc0" | "sc1" => modifier.value.is_some(),
                            _ => true,
                        })
                    || modifiers.iter().any(|m| m.name == "sc0")
                        != modifiers.iter().any(|m| m.name == "sc1")
                {
                    return Err("selected gfx950 scratch DWORD requires bare scalar-offset addressing, one VGPR, `off`, one numbered SGPR, signed13 offset and default or `sc0 sc1` cache policy".into());
                }
                return Ok(());
            }
            if stem.starts_with("buffer_")
                && (operands.len() == 3 || modifiers.iter().any(|m| m.name == "lds"))
            {
                if architecture != crate::target::Architecture::Gfx950
                    || !matches!(stem, "buffer_load_dword" | "buffer_load_dwordx4")
                    || mnemonic != stem
                    || operands.len() != 3
                    || !matches!(operands[2], Operand::Integer(0))
                    || modifiers.len() != 2
                    || !["offen", "lds"].iter().all(|name| {
                        modifiers
                            .iter()
                            .any(|m| m.name == *name && m.value.is_none())
                    })
                {
                    return Err("selected gfx950 buffer-to-LDS load requires three operands, scalar offset0 and exactly `offen lds`".into());
                }
                require(0, &[Vector], 1)?;
                require(1, &[Scalar], 4)?;
                return Ok(());
            }
            if stem == "global_prefetch_b8" {
                // Scalar-base GVS form only: a 32-bit VGPR offset and an SGPR
                // pair. There is no data operand and no register destination.
                require(0, &[Vector], 1)?;
                require(1, &[Scalar], 2)?;
                if architecture != crate::target::Architecture::Gfx1250
                    || modifiers.iter().any(|m| match m.name.as_str() {
                        "scope" => false,
                        "offset" => m
                            .value
                            .as_deref()
                            .and_then(integer)
                            .is_none_or(|offset| !(-8_388_608..=8_388_607).contains(&offset)),
                        _ => true,
                    })
                {
                    return Err("selected gfx1250 GLOBAL prefetch requires VGPR offset, SGPR pair, signed24 offset and a cache scope".into());
                }
                return Ok(());
            }
            if stem.starts_with("global_load_async_to_lds_") {
                // VDST supplies an LDS address, not a register result. B128
                // keeps the compiler's 64-bit GV form (`off`); B32/B64/B128 also
                // parse in the scalar-base GVS form, which only the B0 litmus
                // profile executes.
                require(0, &[Vector], 1)?;
                let gv = matches!(operands.get(2), Some(Operand::Symbol(off)) if off == "off");
                if gv {
                    require(1, &[Vector], 2)?;
                } else {
                    require(1, &[Vector], 1)?;
                    require(2, &[Scalar], 2)?;
                }
                if architecture != crate::target::Architecture::Gfx1250
                    || mnemonic != stem
                    || (gv && stem != "global_load_async_to_lds_b128")
                    || modifiers.len() > 1
                    || modifiers.iter().any(|modifier| modifier.name != "offset")
                {
                    return Err("selected async global-to-LDS load requires GV addressing (`off`, B128) or scalar-base GVS addressing and an optional byte offset".into());
                }
                if modifiers.iter().any(|modifier| {
                    modifier
                        .value
                        .as_deref()
                        .and_then(integer)
                        .is_none_or(|offset| !(-8_388_608..=8_388_607).contains(&offset))
                }) {
                    return Err("async global-to-LDS byte offset must fit signed24".into());
                }
                return Ok(());
            }
            let atomic = stem.contains("_atomic_");
            let store = stem.contains("_store_") || atomic;
            let buffer = stem.starts_with("buffer_");
            let data = if atomic && operands.len() == 4 {
                2
            } else {
                usize::from(store && !buffer)
            };
            require(data, &[Vector, Accumulator], memory_width(stem))?;
            let address = if atomic && operands.len() == 4 {
                1
            } else {
                usize::from(!store || buffer)
            };
            if atomic && operands.len() == 4 {
                require(0, &[Vector], 1)?;
            }
            if stem.starts_with("global_") {
                let scalar_address = if atomic && operands.len() == 4 { 3 } else { 2 };
                match operands.get(scalar_address) {
                    Some(Operand::Symbol(off)) if off == "off" => {
                        if architecture == crate::target::Architecture::Gfx1250
                            && modifiers
                                .iter()
                                .any(|modifier| modifier.name == "scale_offset")
                        {
                            return Err(
                                "GLOBAL scale_offset requires the scalar-base GVS addressing mode"
                                    .into(),
                            );
                        }
                        require(address, &[Vector], 2)?;
                    }
                    Some(Operand::Register(register))
                        if register.class == Scalar && register.count == 2 =>
                    {
                        require(address, &[Vector], 1)?
                    }
                    _ => {
                        return Err(
                            "global instruction needs `off` or a scalar address pair".into()
                        );
                    }
                }
            } else if stem.starts_with("flat_") {
                if architecture == crate::target::Architecture::Gfx1250 && operands.len() == 3 {
                    require(address, &[Vector], 1)?;
                    require(2, &[Scalar], 2)?;
                } else {
                    require(address, &[Vector], 2)?;
                }
                if architecture == crate::target::Architecture::Gfx1250
                    && modifiers.iter().any(|m| {
                        !matches!(m.name.as_str(), "offset" | "scope" | "scale_offset")
                            || (m.name == "scope" && m.value.as_deref() != Some("SCOPE_SYS"))
                            || (m.name == "scale_offset"
                                && (operands.len() != 3
                                    || !matches!(stem, "flat_load_b32" | "flat_store_b32")))
                    })
                {
                    return Err(
                        "selected gfx1250 FLAT supports byte offset, SCOPE_SYS, and B32 scalar-base index scaling".into(),
                    );
                }
            } else if buffer {
                require(2, &[Scalar], 4)?;
                source_width(3, 1)?;
                if architecture == crate::target::Architecture::Gfx950
                    && matches!(
                        stem,
                        "buffer_load_dword"
                            | "buffer_store_dword"
                            | "buffer_load_dwordx2"
                            | "buffer_store_dwordx2"
                    )
                    && (mnemonic != stem
                        || !matches!(operands[0], Operand::Register(r) if r.class == Vector && r.count == memory_width(stem))
                        || !matches!(operands[1], Operand::Register(r) if r.class == Vector && r.count == 1)
                        || !matches!(operands[2], Operand::Register(r) if r.class == Scalar && r.count == 4)
                        || !matches!(operands[3], Operand::Integer(0))
                        || !modifiers
                            .iter()
                            .any(|m| m.name == "offen" && m.value.is_none())
                        || modifiers.iter().any(|m| match m.name.as_str() {
                            "offen" => m.value.is_some(),
                            "offset" => m
                                .value
                                .as_deref()
                                .and_then(integer)
                                .is_none_or(|offset| !(0..=4095).contains(&offset)),
                            _ => true,
                        }))
                {
                    return Err("selected gfx950 buffer DWORD/DWORDX2 requires plain four-operand VGPR-data/SGPR-quad/zero OFFEN, default policy and unsigned12-bit offset".into());
                }
                if matches!(
                    stem,
                    "buffer_load_b32"
                        | "buffer_load_b64"
                        | "buffer_load_b96"
                        | "buffer_load_b128"
                        | "buffer_store_b32"
                        | "buffer_store_b64"
                        | "buffer_store_b96"
                        | "buffer_store_b128"
                ) {
                    // CDNA5 VBUFFER raw subset, distinct from CDNA4's
                    // three-operand buffer->LDS instructions and M0 address:
                    // a VGPR offset with OFFEN, or no VGPR address (`off`).
                    require(0, &[Vector], memory_width(stem))?;
                    let no_vaddr = matches!(&operands[1], Operand::Symbol(off) if off == "off");
                    if !no_vaddr {
                        require(1, &[Vector], 1)?;
                    }
                    let offen = modifiers.iter().any(|m| m.name == "offen");
                    if architecture != crate::target::Architecture::Gfx1250
                        || mnemonic != stem
                        || !matches!(operands[0], Operand::Register(r)
                            if r.class == Vector && u32::from(r.index) + u32::from(r.count) <= 256)
                        || !(no_vaddr
                            || matches!(operands[1], Operand::Register(r)
                            if r.class == Vector && r.count == 1 && r.index < 256))
                        || !matches!(operands[2], Operand::Register(r) if r.class == Scalar && r.count == 4)
                        || !matches!(operands[3], Operand::Register(r)
                            if r.special == Some(SpecialRegister::Null))
                        || offen == no_vaddr
                        || modifiers.iter().any(|m| match m.name.as_str() {
                            "offen" => m.value.is_some(),
                            "offset" => m
                                .value
                                .as_deref()
                                .and_then(integer)
                                .is_none_or(|offset| !(0..=0x7f_ffff).contains(&offset)),
                            _ => true,
                        })
                    {
                        return Err("selected gfx1250 raw buffer load/store requires plain four-operand VGPR data, VGPR OFFEN or `off`, SGPR quad, null soffset, default policy and nonnegative23-bit offset".into());
                    }
                }
            }
        }
        Opcode::Lds => {
            let element_width = if stem.ends_with("b128") {
                4
            } else if stem.ends_with("b96") {
                3
            } else if stem.ends_with("b64") {
                2
            } else {
                1
            };
            let pair = stem.starts_with("ds_read2")
                || stem.starts_with("ds_write2")
                || (architecture == crate::target::Architecture::Gfx1250
                    && stem == "ds_load_2addr_b32");
            if stem.starts_with("ds_read") || stem.starts_with("ds_load") {
                require(0, &[Vector], element_width * if pair { 2 } else { 1 })?;
                require(1, &[Vector], 1)?;
            } else if stem.starts_with("ds_write") || stem.starts_with("ds_store") {
                require(0, &[Vector], 1)?;
                require(1, &[Vector], element_width)?;
                if pair {
                    require(2, &[Vector], element_width)?;
                }
            } else {
                for index in 0..operands.len() {
                    require(index, &[Vector], 1)?;
                }
            }
        }
        Opcode::Matrix => {
            if architecture == crate::target::Architecture::Gfx1250
                && (stem.ends_with("_f8f6f4") || stem.ends_with("_32x16x128_f4"))
                && stem.starts_with("v_wmma_")
            {
                // F8F6F4: A/B widths from the format modifiers (FP8/BF8 16,
                // FP6/BF6 12, FP4 8); F4 32x16x128: A 16, B 8, C/D 16. The
                // scaled forms add two scale operands (one VGPR, or an even
                // pair for SCALE16; SGPRs and inline constants parse).
                let format = |name: &str| -> Result<u16, String> {
                    match modifiers
                        .iter()
                        .find(|modifier| modifier.name == name)
                        .and_then(|modifier| modifier.value.as_deref())
                    {
                        None | Some("MATRIX_FMT_FP8" | "MATRIX_FMT_BF8") => Ok(16),
                        Some("MATRIX_FMT_FP6" | "MATRIX_FMT_BF6") => Ok(12),
                        Some("MATRIX_FMT_FP4") => Ok(8),
                        Some(other) => Err(format!("unsupported matrix format `{other}`")),
                    }
                };
                let (d, a, b) = if stem.ends_with("_f4") {
                    (16, 16, 8)
                } else {
                    (8, format("matrix_a_fmt")?, format("matrix_b_fmt")?)
                };
                require(0, &[Vector], d)?;
                require(1, &[Vector], a)?;
                require(2, &[Vector], b)?;
                if !matches!(operands.get(3), Some(Operand::Integer(0))) {
                    require(3, &[Vector], d)?;
                }
                let scale = if stem.starts_with("v_wmma_scale16_") {
                    2
                } else if stem.starts_with("v_wmma_scale_") {
                    1
                } else {
                    return Ok(());
                };
                for index in 4..6 {
                    match operands.get(index) {
                        Some(Operand::Integer(_)) => {}
                        Some(_) if register_at(index).is_some_and(|r| r.class == Scalar) => {
                            require(index, &[Scalar], 1)?;
                        }
                        _ => require(index, &[Vector], scale)?,
                    }
                }
                return Ok(());
            }
            if architecture == crate::target::Architecture::Gfx1250 && stem.starts_with("v_swmmac_")
            {
                // D (also C), packed A 8, B 16, index 1 (16-bit inputs) or 2.
                let d = if stem.starts_with("v_swmmac_f16_") || stem.starts_with("v_swmmac_bf16_") {
                    4
                } else {
                    8
                };
                require(0, &[Vector], d)?;
                require(1, &[Vector], 8)?;
                require(2, &[Vector], 16)?;
                require(3, &[Vector], if stem.contains("16x16x128") { 2 } else { 1 })?;
                return Ok(());
            }
            if architecture == crate::target::Architecture::Gfx1250
                && (stem.starts_with("v_wmma_f16_") || stem.starts_with("v_wmma_bf16"))
            {
                // 16-bit output: D 4 registers; C 4 (BF16F32: 8 F32).
                let width = if stem.contains("16x16x128") { 16 } else { 8 };
                require(0, &[Vector], 4)?;
                require(1, &[Vector], width)?;
                require(2, &[Vector], width)?;
                if !matches!(operands.get(3), Some(Operand::Integer(0))) {
                    require(
                        3,
                        &[Vector],
                        if stem.starts_with("v_wmma_bf16f32_") {
                            8
                        } else {
                            4
                        },
                    )?;
                }
                return Ok(());
            }
            if architecture == crate::target::Architecture::Gfx1250 && stem.starts_with("v_wmma_") {
                require(0, &[Vector], 8)?;
                let width = if stem.ends_with("_f32") {
                    2
                } else if stem.contains("16x16x128") {
                    16
                } else {
                    8
                };
                require(1, &[Vector], width)?;
                require(2, &[Vector], width)?;
                if !matches!(operands.get(3), Some(Operand::Integer(0))) {
                    require(3, &[Vector], 8)?;
                }
                return Ok(());
            }
            if operands
                .iter()
                .any(|operand| matches!(operand, Operand::Modified { .. }))
            {
                return Err(
                    "MFMA operands require unmodified native registers or C literals".into(),
                );
            }
            let width = if stem.contains("32x32") {
                16
            } else if stem.contains("f64") {
                8
            } else {
                4
            };
            require(0, &[Vector, Accumulator], width)?;
            for index in [1, 2] {
                let input_width = if stem.contains("f8f6f4") {
                    let selector = if index == 1 { "cbsz" } else { "blgp" };
                    let format = modifiers
                        .iter()
                        .find(|modifier| modifier.name == selector)
                        .and_then(|modifier| modifier.value.as_deref())
                        .and_then(integer)
                        .unwrap_or(0);
                    match format {
                        0 | 1 => 8,
                        2 | 3 => 6,
                        4 => 4,
                        _ => return Err("invalid native MFMA F8F6F4 operand format".into()),
                    }
                } else if stem.ends_with("_f32")
                    || stem.ends_with("x4f32")
                    || stem.ends_with("x2f32")
                {
                    1
                } else if (stem.contains("f16") || stem.contains("bf16"))
                    && (stem.contains("16x16x32") || stem.contains("32x32x16"))
                {
                    4
                } else {
                    2
                };
                // CDNA4 MFMA A/B selectors address vector register files;
                // scalar and inline sources are not encodable, even for zero.
                require(index, &[Vector, Accumulator], input_width)?;
            }
            match operands.get(3) {
                Some(Operand::Register(accumulator)) => {
                    let destination = register_at(0).expect("validated destination");
                    require(3, &[destination.class], width)?;
                    debug_assert_eq!(accumulator.class, destination.class);
                }
                Some(Operand::Integer(value))
                    if if stem.contains("f64") {
                        (-16..=64).contains(value)
                    } else {
                        (i32::MIN as i64..=u32::MAX as i64).contains(value)
                    } => {}
                Some(Operand::Float(value))
                    if !stem.contains("f64")
                        || [0.0, 0.5, 1.0, 2.0, 4.0].contains(&value.abs()) => {}
                _ => return Err("invalid native MFMA C source".into()),
            }
        }
        Opcode::VectorAlu => {
            if architecture == crate::target::Architecture::Gfx1250
                && matches!(stem, "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte3")
                && (!modifiers.is_empty()
                    || !matches!(operands.get(1),Some(Operand::Register(r)) if matches!(r.class,Scalar|Vector) && r.count==1)
                        && !matches!(operands.get(1),Some(Operand::Integer(v)) if (i32::MIN as i64..=u32::MAX as i64).contains(v)))
            {
                return Err(
                    "selected UBYTE conversion needs one unmodified encoded32bit source".into(),
                );
            }
            if stem == "v_s_exp_f32" {
                // LLVM VOP_Pseudo_Scalar_F32 uses SSrc_f32, not VSrc_f32:
                // reading an arbitrary VGPR lane would change its semantics.
                require(0, &[Scalar], 1)?;
                source_width(1, 1)?;
                if register_at(1).is_some_and(|r| !matches!(r.class, Scalar | Temporary | Special))
                {
                    return Err(
                        "v_s_exp_f32 source must be scalar or an encodable immediate".into(),
                    );
                }
                return Ok(());
            }
            let comparison = stem.starts_with("v_cmp");
            if architecture == crate::target::Architecture::Gfx1250 && mnemonic.ends_with("_e32") {
                if comparison && stem.ends_with("_f16") {
                    let cmpx = stem.starts_with("v_cmpx_");
                    let start = usize::from(!cmpx);
                    if !cmpx
                        && !matches!(operands.first(), Some(Operand::Register(r))
                        if r.special == Some(SpecialRegister::VccLo) && r.count == 1 && r.class == Special)
                    {
                        return Err(
                            "gfx1250 F16 e32 comparison writes the fixed VCC_LO destination".into(),
                        );
                    }
                    if !matches!(operands.get(start + 1), Some(Operand::Register(r))
                        if r.class == Vector && r.count == 1)
                    {
                        return Err("gfx1250 F16 e32 comparison SRC1 requires a VGPR".into());
                    }
                    let src0 = match operands.get(start) {
                        Some(Operand::Register(r)) => r.count == 1,
                        Some(Operand::Integer(n)) => {
                            (i16::MIN as i64..=u16::MAX as i64).contains(n)
                        }
                        Some(Operand::Float(_)) => true,
                        _ => false,
                    };
                    if !src0 {
                        return Err("gfx1250 F16 e32 comparison SRC0 requires an unmodified source with a16-bit literal".into());
                    }
                }
                let all_half = matches!(
                    stem,
                    "v_add_f16"
                        | "v_sub_f16"
                        | "v_mul_f16"
                        | "v_min_f16"
                        | "v_max_f16"
                        | "v_max_num_f16"
                        | "v_fmac_f16"
                        | "v_cvt_f16_i16"
                ) || comparison && stem.ends_with("_f16");
                for (index, operand) in operands.iter().enumerate() {
                    let half_operand = all_half
                        || stem == "v_cvt_f16_f32" && index == 0
                        || matches!(stem, "v_cvt_f32_f16" | "v_cvt_f32_bf16") && index == 1;
                    if half_operand
                        && register(operand).is_some_and(|r| r.class == Vector && r.index >= 128)
                    {
                        return Err("gfx1250 true16 e32 operands require VGPRs below v128".into());
                    }
                }
            }
            let carry = stem.starts_with("v_add_co_")
                || stem.starts_with("v_addc_co_")
                || stem.starts_with("v_sub_co_")
                || stem.starts_with("v_subb_co_")
                || stem.starts_with("v_subrev_co_")
                || stem.starts_with("v_subbrev_co_");
            let scalar_destination = stem == "v_s_exp_f32"
                || stem.starts_with("v_readlane_")
                || stem.starts_with("v_readfirstlane_");
            if comparison {
                let implicit = architecture == crate::target::Architecture::Gfx1250
                    && stem.starts_with("v_cmpx_");
                if !implicit {
                    require(
                        0,
                        &[Scalar, Special],
                        if architecture.wave_size() == 32 { 1 } else { 2 },
                    )?;
                }
                let start = usize::from(!implicit);
                let width = if stem.ends_with("64") { 2 } else { 1 };
                source_width(start, width)?;
                source_width(start + 1, width)?;
            } else if carry {
                require(0, &[Vector], 1)?;
                require(
                    1,
                    &[Scalar, Special],
                    if architecture.wave_size() == 32 { 1 } else { 2 },
                )?;
                source_width(2, 1)?;
                source_width(3, 1)?;
                if operands.len() == 5 {
                    source_width(
                        4,
                        if architecture.wave_size() == 32 && carry {
                            1
                        } else {
                            2
                        },
                    )?;
                }
            } else if matches!(stem, "v_mad_u64_u32" | "v_mad_i64_i32") {
                require(0, &[Vector], 2)?;
                require(
                    1,
                    &[Scalar, Special],
                    if architecture.wave_size() == 32 { 1 } else { 2 },
                )?;
                source_width(2, 1)?;
                source_width(3, 1)?;
                source_width(
                    4,
                    if architecture.wave_size() == 32 && carry {
                        1
                    } else {
                        2
                    },
                )?;
                if stem == "v_mad_i64_i32" {
                    // CDNA4 VOP3B: two 32-bit factors, one 64-bit addend and
                    // a wave64 scalar result mask. Keep a finite unmodified
                    // register/inline profile rather than guessing literal,
                    // clamp, floating-inline or special-source semantics.
                    if architecture != crate::target::Architecture::Gfx950
                        || mnemonic.ends_with("_e32")
                        || !modifiers.is_empty()
                        || !matches!(operands[0], Operand::Register(_))
                        || !matches!(operands[1], Operand::Register(r)
                            if r.class == Scalar || r.special == Some(SpecialRegister::Vcc))
                    {
                        return Err("selected signed64 MAD requires unmodified gfx950 VOP3B and an SGPR-pair/VCC mask".into());
                    }
                    let mut scalar_sources = std::collections::BTreeSet::new();
                    for (index, operand) in operands.iter().enumerate().skip(2) {
                        match operand {
                            Operand::Register(r)
                                if matches!(r.class, Scalar | Vector)
                                    && r.count == if index == 4 { 2 } else { 1 } =>
                            {
                                if r.class == Scalar {
                                    scalar_sources.insert((r.index, r.count));
                                }
                            }
                            Operand::Integer(value) if (-16..=64).contains(value) => {}
                            _ => return Err("signed64 MAD sources require plain data registers or integer inline constants".into()),
                        }
                    }
                    if scalar_sources.len() > 1 {
                        return Err("selected signed64 MAD supports at most one distinct scalar data source".into());
                    }
                }
            } else if ["v_cvt_pk_f32_fp8", "v_cvt_pk_f32_bf8", "v_cvt_f64_f32"].contains(&stem) {
                require(0, &[Vector], 2)?;
                source_width(1, 1)?;
            } else if stem == "v_cvt_f32_f64" {
                require(0, &[Vector], 1)?;
                source_width(1, 2)?;
            } else if stem == "v_cndmask_b32" {
                require(0, &[Vector], 1)?;
                source_width(1, 1)?;
                source_width(2, 1)?;
                if operands.len() == 4 {
                    require(
                        3,
                        &[Scalar, Special],
                        if architecture.wave_size() == 32 { 1 } else { 2 },
                    )?;
                }
            } else if stem == "v_mul_u64" {
                // CDNA5 VOP2 opcode42: keep the compiler's plain pair E32
                // subset. E64, special selectors, inline/literal sources and
                // modifiers have separate encodings outside this profile.
                require(0, &[Vector], 2)?;
                require(1, &[Scalar, Vector], 2)?;
                require(2, &[Vector], 2)?;
                if architecture != crate::target::Architecture::Gfx1250
                    || mnemonic.ends_with("_e64")
                    || !modifiers.is_empty()
                    || operands.iter().any(|operand| {
                        !matches!(operand, Operand::Register(r)
                            if r.count == 2 && (r.class == Scalar
                                || r.class == Vector && r.index <= 254))
                    })
                {
                    return Err("selected v_mul_u64 requires unmodified gfx1250 E32 register pairs (src0 SGPR/VGPR, src1 VGPR)".into());
                }
            } else if stem == "v_add_nc_u64" {
                require(0, &[Vector], 2)?;
                source_width(1, 2)?;
                source_width(2, 2)?;
            } else if stem == "v_lshl_add_u64" {
                require(0, &[Vector], 2)?;
                source_width(1, 2)?;
                source_width(2, 1)?;
                source_width(3, 2)?;
            } else if ["v_lshlrev_b64", "v_lshrrev_b64", "v_ashrrev_i64"].contains(&stem) {
                require(0, &[Vector], 2)?;
                source_width(1, 1)?;
                source_width(2, 2)?;
                if stem == "v_ashrrev_i64" {
                    // The supported signed64 profile transports an actual
                    // ordered data register pair. VOP3 has no literal word;
                    // reject float/source modifiers and immediate data aliases
                    // rather than inventing their64bit source interpretation.
                    if !modifiers.is_empty()
                        || !matches!(operands[0], Operand::Register(_))
                        || !matches!(operands[1], Operand::Register(register) if matches!(register.class, Scalar | Vector) && register.count == 1)
                            && !matches!(operands[1], Operand::Integer(value) if (-16..=64).contains(&value))
                        || !matches!(operands[2], Operand::Register(register) if matches!(register.class, Scalar | Vector) && register.count == 2)
                    {
                        return Err("v_ashrrev_i64 requires an unmodified32bit register/inline shift and64bit data register pair".into());
                    }
                }
            } else {
                let width = if stem == "v_mov_b64"
                    || stem.ends_with("f64")
                    || stem.starts_with("v_pk_") && stem.ends_with("f32")
                {
                    2
                } else {
                    1
                };
                let destination_class: &[RegisterClass] = if scalar_destination {
                    &[Scalar, Special]
                } else if ["v_accvgpr_write_b32", "v_accvgpr_mov_b32"].contains(&stem) {
                    &[Accumulator]
                } else {
                    &[Vector]
                };
                require(0, destination_class, width)?;
                for index in 1..operands.len() {
                    source_width(index, width)?;
                }
                if ["v_accvgpr_read_b32", "v_accvgpr_mov_b32"].contains(&stem) {
                    require(1, &[Accumulator], 1)?;
                }
                if stem.starts_with("v_readlane_") {
                    require(1, &[Vector], 1)?;
                    if architecture == crate::target::Architecture::Gfx1250 {
                        let inline = |bits: u32| {
                            bits <= 64
                                || bits >= (-16_i32) as u32
                                || [
                                    0x3f00_0000,
                                    0xbf00_0000,
                                    0x3f80_0000,
                                    0xbf80_0000,
                                    0x4000_0000,
                                    0xc000_0000,
                                    0x4080_0000,
                                    0xc080_0000,
                                    0x3e22_f983,
                                ]
                                .contains(&bits)
                        };
                        let selector = match operands.get(2) {
                            Some(Operand::Register(r)) => {
                                r.count == 1 && matches!(r.class, Scalar | Temporary | Special)
                            }
                            Some(Operand::Integer(n)) => {
                                (i32::MIN as i64..=u32::MAX as i64).contains(n) && inline(*n as u32)
                            }
                            Some(Operand::Float(n)) => inline((*n as f32).to_bits()),
                            _ => false,
                        };
                        if !selector {
                            return Err("gfx1250 readlane selector requires an unmodified scalar source or inline constant".into());
                        }
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Recheck selected scalar constants after assembly Symbol substitution.
pub(super) fn validate_scalar_sources(
    instruction: &Instruction,
    architecture: crate::target::Architecture,
) -> Result<(), ParseError> {
    if instruction.opcode != Opcode::ScalarAlu {
        return Ok(());
    }
    crate::source_encoding::resolve(
        &instruction.mnemonic,
        &instruction.operands,
        architecture,
        true,
    )
    .map(|_| ())
    .map_err(|message| ParseError {
        message,
        span: instruction.span,
    })
}

/// Pinned gfx950 arithmetic forms: VOP2 can encode one literal in SRC0; VOP3
/// has inline constants but no literal word. Bare aliases select VOP3 when
/// their second source or modifiers require it. Other instruction families
/// retain their separate encoding profiles rather than inheriting this rule.
pub(super) fn validate_vop_literal_positions(
    mnemonic: &str,
    operands: &[Operand],
    modifiers: &[Modifier],
    architecture: crate::target::Architecture,
) -> Result<(), String> {
    let explicit32 = mnemonic.ends_with("_e32");
    let explicit64 = mnemonic.ends_with("_e64");
    let stem = mnemonic
        .strip_suffix("_e32")
        .or_else(|| mnemonic.strip_suffix("_e64"))
        .unwrap_or(mnemonic);
    if ![
        "s_sub_co_i32",
        "s_cvt_f32_i32",
        "s_cvt_f32_u32",
        "s_add_f32",
        "s_sub_f32",
        "s_mul_f32",
        "s_min_num_f32",
        "v_add_u32",
        "v_sub_u32",
        "v_subrev_u32",
        "v_add_f32",
        "v_sub_f32",
        "v_subrev_f32",
        "v_mul_f32",
        "v_min_f32",
        "v_max_f32",
        "v_min_num_f32",
        "v_max_num_f32",
        "v_fmac_f32",
        "v_fmac_f16",
        "v_min_f16",
        "v_max_f16",
        "v_max_num_f16",
        "v_add_f64",
        "v_mul_f64",
        "v_min_f64",
        "v_max_f64",
        "v_fmac_f64",
        "v_fma_f32",
        "v_fma_f16",
        "v_fma_f64",
        "v_and_b32",
        "v_or_b32",
        "v_xor_b32",
        "v_lshlrev_b32",
        "v_lshrrev_b32",
        "v_ashrrev_i32",
        "v_mul_lo_u32",
    ]
    .contains(&stem)
    {
        return Ok(());
    }
    fn inline(bits: u32) -> bool {
        bits <= 64
            || bits >= (-16_i32) as u32
            || [
                0x3f00_0000,
                0xbf00_0000,
                0x3f80_0000,
                0xbf80_0000,
                0x4000_0000,
                0xc000_0000,
                0x4080_0000,
                0xc080_0000,
                0x3e22_f983,
            ]
            .contains(&bits)
    }
    fn inline_f64(value: f64) -> bool {
        value == 0.0 && !value.is_sign_negative() || [0.5_f64, 1.0, 2.0, 4.0].contains(&value.abs())
    }
    fn literal(operand: &Operand, f64_source: bool, f16_source: bool) -> Result<bool, String> {
        match operand {
            Operand::Integer(value) => {
                if f16_source && !(i16::MIN as i64..=u16::MAX as i64).contains(value) {
                    return Err("native F16 literal is outside its encoded16bit range".into());
                }
                if f64_source {
                    let bits = *value as u64;
                    let decoded = f64::from_bits(if bits <= u32::MAX as u64 {
                        bits << 32
                    } else {
                        bits
                    });
                    if (-16..=64).contains(value) || inline_f64(decoded) {
                        return Ok(false);
                    }
                }
                if !(i32::MIN as i64..=u32::MAX as i64).contains(value) {
                    return Err("native 32-bit literal is outside its encoded range".into());
                }
                if f16_source {
                    return Ok(!((-16..=64).contains(value)
                        || (65520..=65535).contains(value)
                        || [
                            0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x4400, 0xc400, 0x3118,
                        ]
                        .contains(value)));
                }
                Ok(f64_source || !inline(*value as u32))
            }
            Operand::Float(value) => Ok(if f64_source {
                !inline_f64(*value)
            } else {
                !inline((*value as f32).to_bits())
            }),
            Operand::Modified { value, .. } => literal(value, f64_source, f16_source),
            _ => Ok(false),
        }
    }
    if architecture == crate::target::Architecture::Gfx1250
        && !explicit32
        && !stem.ends_with("_f64")
    {
        let mut values = std::collections::BTreeSet::new();
        fn bits(op: &Operand) -> Option<u32> {
            match op {
                Operand::Integer(n) => Some(*n as u32),
                Operand::Float(n) => Some((*n as f32).to_bits()),
                Operand::Modified { value, .. } => bits(value),
                _ => None,
            }
        }
        for op in operands.iter().skip(1) {
            if literal(op, false, stem.ends_with("_f16"))? {
                values.insert(bits(op).unwrap());
            }
        }
        if values.len() > 1 {
            return Err(
                "native gfx1250 instruction has more than one distinct literal word".into(),
            );
        }
        return Ok(());
    }
    let f64_source = stem.ends_with("_f64");
    let f16_source = stem.ends_with("_f16");
    if architecture == crate::target::Architecture::Gfx1250
        && explicit32
        && matches!(stem, "v_add_f64" | "v_mul_f64")
        && operands.iter().any(|operand| {
            matches!(operand, Operand::Register(r) if matches!(r.class, RegisterClass::Vector | RegisterClass::Scalar) && r.index % 2 != 0)
        })
    {
        return Err("gfx1250 F64 VOP2 requires aligned register pairs".into());
    }
    let vop3_only = matches!(
        stem,
        "v_mul_lo_u32" | "v_fma_f32" | "v_fma_f16" | "v_fma_f64" | "v_min_f64" | "v_max_f64"
    ) || (architecture != crate::target::Architecture::Gfx1250
        && matches!(stem, "v_add_f64" | "v_mul_f64"));
    let src1_vgpr = matches!(
        operands.get(2),
        Some(Operand::Register(Register {
            class: RegisterClass::Vector,
            count,
            ..
        })) if *count == if f64_source { 2 } else { 1 }
    );
    if explicit32 && (vop3_only || !src1_vgpr) {
        return Err("native e32 form requires an encoded VOP2 and VGPR SRC1".into());
    }
    let vop3 = explicit64
        || vop3_only
        || !src1_vgpr
        || !modifiers.is_empty()
        || operands
            .iter()
            .any(|operand| matches!(operand, Operand::Modified { .. }));
    for (index, operand) in operands.iter().enumerate().skip(1) {
        if literal(operand, f64_source, f16_source)? && (vop3 || index != 1) {
            return Err(
                "native AMD VOP3 has no literal word; materialize this source in a VGPR".into(),
            );
        }
    }
    Ok(())
}

fn memory_width(mnemonic: &str) -> u16 {
    if mnemonic.ends_with("dwordx16") || mnemonic.ends_with("b512") {
        16
    } else if mnemonic.ends_with("dwordx8") || mnemonic.ends_with("b256") {
        8
    } else if mnemonic.ends_with("dwordx4") || mnemonic.ends_with("b128") {
        4
    } else if mnemonic.ends_with("dwordx3") || mnemonic.ends_with("b96") {
        3
    } else if mnemonic.ends_with("dwordx2") || mnemonic.ends_with("b64") {
        2
    } else {
        1
    }
}

/// Validate syntax supplied through the public typed AST, rather than text.
/// In particular, executable role accessors must never receive a short tuple.
pub(super) fn validate_for_lowering(
    instruction: &Instruction,
    architecture: crate::target::Architecture,
) -> Result<(), String> {
    use crate::target::Architecture;
    if let [Operand::Dual(x, y)] = instruction.operands.as_slice() {
        if architecture != Architecture::Gfx1250
            || instruction.opcode != Opcode::VectorAlu
            || !instruction.modifiers.is_empty()
            || x.opcode != Opcode::VectorAlu
            || y.opcode != Opcode::VectorAlu
            || x.operands
                .iter()
                .chain(&y.operands)
                .any(|value| matches!(value, Operand::Dual(..)))
        {
            return Err("invalid typed VOPD instruction or nested issue".into());
        }
        validate_for_lowering(x, architecture)?;
        validate_for_lowering(y, architecture)?;
        return super::gfx1250::validate_typed_profile(instruction);
    }
    let (opcode, mut minimum, mut maximum) = match instruction.mnemonic.as_str() {
        "s_getreg_b32" | "s_setreg_b32" | "s_setreg_imm32_b32"
            if architecture == Architecture::Gfx1250 =>
        {
            (Opcode::ScalarAlu, 2, 2)
        }
        "s_round_mode" | "s_denorm_mode" if architecture == Architecture::Gfx1250 => {
            (Opcode::ScalarAlu, 1, 1)
        }
        "s_delay_alu" if architecture == Architecture::Gfx1250 => (Opcode::Control, 1, 1),
        name => signature(name)
            .or_else(|| {
                super::gfx1250::signature(name).filter(|_| architecture == Architecture::Gfx1250)
            })
            .ok_or_else(|| format!("unsupported typed instruction {name}"))?,
    };
    if architecture == Architecture::Gfx1250 && instruction.mnemonic.starts_with("v_cmpx_") {
        minimum = 2;
        maximum = 2;
    }
    if instruction.opcode != opcode || !(minimum..=maximum).contains(&instruction.operands.len()) {
        return Err(format!(
            "typed instruction {} has an invalid opcode/operand count",
            instruction.mnemonic
        ));
    }
    fn value(operand: &Operand, architecture: Architecture) -> Result<(), String> {
        match operand {
            Operand::Register(register) => validate_register_shape(*register, architecture)?,
            Operand::Modified {
                value: inner,
                negate,
                absolute,
            } => {
                if !(*negate || *absolute)
                    || !matches!(
                        inner.as_ref(),
                        Operand::Register(_) | Operand::Integer(_) | Operand::Float(_)
                    )
                {
                    return Err("source modifiers require one register or literal".into());
                }
                value(inner, architecture)?;
            }
            Operand::HardwareRegister { offset, width, .. }
                if *width == 0 || u16::from(*offset) + u16::from(*width) > 32 =>
            {
                return Err("hwreg bit selection outside 32-bit register".into());
            }
            Operand::Float(v) if !v.is_finite() => {
                return Err("nonfinite float literal is unsupported".into());
            }
            Operand::Dual(..) => return Err("dual issue is not a value operand".into()),
            _ => {}
        }
        Ok(())
    }
    for operand in &instruction.operands {
        value(operand, architecture)?;
    }
    let mut seen = std::collections::BTreeSet::new();
    for modifier in &instruction.modifiers {
        if !seen.insert(&modifier.name) {
            return Err(format!("duplicate modifier `{}`", modifier.name));
        }
        if modifier.name == "src0_sext" {
            if instruction.mnemonic != "v_cvt_f16_i16_sdwa"
                || modifier.value.is_some()
                || !matches!(instruction.operands.get(1), Some(Operand::Register(r)) if r.class == RegisterClass::Vector && r.count == 1)
            {
                return Err(
                    "SDWA sext source requires v_cvt_f16_i16_sdwa and one plain VGPR".into(),
                );
            }
            continue;
        }
        let token = modifier.value.as_ref().map_or_else(
            || modifier.name.clone(),
            |value| format!("{}:{value}", modifier.name),
        );
        parse_modifier(&token, instruction.opcode, &instruction.mnemonic)?;
    }
    validate_control(instruction).map_err(|e| e.message)?;
    validate_post_parse(
        &instruction.mnemonic,
        instruction.opcode,
        &instruction.operands,
        &instruction.modifiers,
        architecture,
        instruction.span,
    )
    .map_err(|error| error.message)?;
    if architecture == Architecture::Gfx1250 {
        super::gfx1250::validate_typed_profile(instruction)?;
    }
    Ok(())
}

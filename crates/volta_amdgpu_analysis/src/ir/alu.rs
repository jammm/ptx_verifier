//! Semantic ALU families. Instruction spelling is resolved once during lowering.
//! Widths, signedness, source order, carry behavior and packing remain explicit;
//! evaluators consume these fields without reparsing native opcode names.
use super::instruction::{CompareType, Comparison, Relation};
use super::ops::canonical_alu;
use std::borrow::Cow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluDomain {
    Scalar,
    Vector,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitWidth {
    B16,
    B32,
    B64,
}
impl BitWidth {
    pub const fn bits(self) -> u8 {
        match self {
            Self::B16 => 16,
            Self::B32 => 32,
            Self::B64 => 64,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegerType {
    pub width: BitWidth,
    pub signed: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatType {
    F16,
    F32,
    F64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveKind {
    Plain,
    Immediate16,
    SelectScc,
    SelectVcc,
    AccumulatorRead,
    AccumulatorWrite,
    AccumulatorMove,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerKind {
    Abs,
    Add,
    Sub,
    Mul,
    MulHigh,
    Min,
    Max,
    MultiplyAdd,
    Add3,
    AddShiftLeft,
    ShiftLeftAdd,
    XorAdd,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerForm {
    Plain,
    Immediate16,
    Reverse,
    CarryIn,
    CarryOut,
    CarryInOut,
    ReverseCarryOut,
    ReverseCarryInOut,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitwiseKind {
    And,
    Or,
    Xor,
    Not,
    AndNot,
    OrNot,
    Nand,
    Nor,
    Xnor,
    AndOr,
    Or3,
    ShiftLeftOr,
    TruthTable2,
    TruthTable3,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftKind {
    Left,
    LogicalRight,
    ArithmeticRight,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitfieldKind {
    Extract,
    Insert,
    Mask,
    TestZero,
    TestOne,
    SetOne,
    Align,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitScanKind {
    Reverse,
    FirstZeroLow,
    FirstOneLow,
    FirstOneHigh,
    LeadingSign,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatKind {
    Add,
    Sub,
    SubReverse,
    Mul,
    Min,
    MinNum,
    Max,
    Min3,
    Max3,
    Max3Num,
    Ceil,
    Cos,
    Exp2,
    Floor,
    Fract,
    Ldexp,
    Log2,
    Reciprocal,
    ReciprocalSqrt,
    Sin,
    Sqrt,
    Trunc,
    ScalarExp2,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedKind {
    Fma,
    Accumulate,
    MultiplyAdd,
    LiteralAdd,
    LiteralMultiply,
    Mixed,
    MixedLow,
    MixedHigh,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionType {
    I16,
    I32,
    U32,
    F16,
    Bf16,
    F32,
    F64,
    Fp8,
    Bf8,
    Bits32,
    UnsignedByte(u8),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionMode {
    Plain,
    Pack,
    PackRoundToZero,
    UnpackPair,
    SignExtend,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedKind {
    Add,
    Sub,
    Mul,
    Fma,
    Min,
    Max,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedType {
    F16,
    F32,
    I16,
    U16,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneKind {
    Read,
    ReadFirst,
    Write,
    MaskCountLow,
    MaskCountHigh,
    Permute,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarSpecialKind {
    GetPcLegacy,
    GetPc,
    GetHardwareRegister,
    SetHardwareRegister,
    SetHardwareRegisterImmediate,
    RoundMode,
    DenormMode,
    ShaderCycles,
    SendMessageReturn,
}

/// The executable operation is a semantic family, independent of source spelling.
/// Target validation and operand encoding remain responsibilities of lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOperation {
    Move {
        domain: AluDomain,
        kind: MoveKind,
        width: BitWidth,
    },
    Integer {
        domain: AluDomain,
        kind: IntegerKind,
        ty: IntegerType,
        source_bits: u8,
        form: IntegerForm,
    },
    Bitwise {
        domain: AluDomain,
        kind: BitwiseKind,
        width: BitWidth,
        save_exec: bool,
    },
    Shift {
        domain: AluDomain,
        kind: ShiftKind,
        width: BitWidth,
        reverse: bool,
    },
    Bitfield {
        domain: AluDomain,
        kind: BitfieldKind,
        width: BitWidth,
        signed: bool,
    },
    BitScan {
        domain: AluDomain,
        kind: BitScanKind,
        width: BitWidth,
    },
    ScalarCompare {
        relation: Relation,
        ty: CompareType,
        immediate: bool,
    },
    Compare(Comparison),
    Float {
        domain: AluDomain,
        kind: FloatKind,
        ty: FloatType,
    },
    Fused {
        kind: FusedKind,
        ty: FloatType,
    },
    Conversion {
        domain: AluDomain,
        source: ConversionType,
        destination: ConversionType,
        mode: ConversionMode,
    },
    Packed {
        kind: PackedKind,
        ty: PackedType,
    },
    Lane(LaneKind),
    ScalarSpecial(ScalarSpecialKind),
}

// These constructors describe fields only; they generate no executable dispatch.
macro_rules! move_form {
    ($domain:ident, $kind:ident, $width:ident) => {
        AluOperation::Move {
            domain: AluDomain::$domain,
            kind: MoveKind::$kind,
            width: BitWidth::$width,
        }
    };
}
macro_rules! integer_form {
    ($domain:ident, $kind:ident, $width:ident, $signed:literal, $bits:literal, $form:ident) => {
        AluOperation::Integer {
            domain: AluDomain::$domain,
            kind: IntegerKind::$kind,
            ty: IntegerType {
                width: BitWidth::$width,
                signed: $signed,
            },
            source_bits: $bits,
            form: IntegerForm::$form,
        }
    };
}
macro_rules! bitwise_form {
    ($domain:ident, $kind:ident, $width:ident, $save:literal) => {
        AluOperation::Bitwise {
            domain: AluDomain::$domain,
            kind: BitwiseKind::$kind,
            width: BitWidth::$width,
            save_exec: $save,
        }
    };
}
macro_rules! shift_form {
    ($domain:ident, $kind:ident, $width:ident, $reverse:literal) => {
        AluOperation::Shift {
            domain: AluDomain::$domain,
            kind: ShiftKind::$kind,
            width: BitWidth::$width,
            reverse: $reverse,
        }
    };
}
macro_rules! bitfield_form {
    ($domain:ident, $kind:ident, $width:ident, $signed:literal) => {
        AluOperation::Bitfield {
            domain: AluDomain::$domain,
            kind: BitfieldKind::$kind,
            width: BitWidth::$width,
            signed: $signed,
        }
    };
}
macro_rules! scan_form {
    ($domain:ident, $kind:ident, $width:ident) => {
        AluOperation::BitScan {
            domain: AluDomain::$domain,
            kind: BitScanKind::$kind,
            width: BitWidth::$width,
        }
    };
}
macro_rules! scalar_compare_form {
    ($relation:ident, $ty:ident, $immediate:literal) => {
        AluOperation::ScalarCompare {
            relation: Relation::$relation,
            ty: CompareType::$ty,
            immediate: $immediate,
        }
    };
}
macro_rules! float_form {
    ($domain:ident, $kind:ident, $ty:ident) => {
        AluOperation::Float {
            domain: AluDomain::$domain,
            kind: FloatKind::$kind,
            ty: FloatType::$ty,
        }
    };
}
macro_rules! fused_form {
    ($kind:ident, $ty:ident) => {
        AluOperation::Fused {
            kind: FusedKind::$kind,
            ty: FloatType::$ty,
        }
    };
}
macro_rules! conversion_form {
    ($domain:ident, $source:expr, $destination:expr, $mode:ident) => {
        AluOperation::Conversion {
            domain: AluDomain::$domain,
            source: $source,
            destination: $destination,
            mode: ConversionMode::$mode,
        }
    };
}
macro_rules! packed_form {
    ($kind:ident, $ty:ident) => {
        AluOperation::Packed {
            kind: PackedKind::$kind,
            ty: PackedType::$ty,
        }
    };
}

// Named forms let target-specific guards select an exact legal encoding while
// arithmetic evaluators share the semantic fields of the enclosing family.
macro_rules! semantic_forms {
    ($($variant:ident => $name:literal => $operation:expr,)*) => {
        #[allow(non_upper_case_globals)]
        impl AluOperation {
            $(pub const $variant: Self = $operation;)*
            pub(crate) fn parse(name: &str) -> Option<Self> {
                let name = canonical_alu(name);
                if let Some(comparison) = Comparison::parse(name) {
                    Some(Self::Compare(comparison))
                } else {
                    match name {
                        $($name => Some(Self::$variant),)*
                        _ => None,
                    }
                }
            }
            /// Canonical diagnostic spelling. Execution never uses this lookup.
            pub fn name(self) -> Cow<'static, str> {
                if let Self::Compare(comparison) = self {
                    return Cow::Owned(comparison.name());
                }
                const NAMES: &[(AluOperation, &str)] = &[$((AluOperation::$variant, $name),)*];
                NAMES.iter().find_map(|(operation, name)| (*operation == self).then_some(Cow::Borrowed(*name)))
                    .unwrap_or_else(|| Cow::Owned(format!("{self:?}")))
            }
        }
    };
}

semantic_forms! {
    SAbsI32 => "s_abs_i32" => integer_form!(Scalar, Abs, B32, true, 32, Plain),
    SAddF32 => "s_add_f32" => float_form!(Scalar, Add, F32),
    SAddI32 => "s_add_i32" => integer_form!(Scalar, Add, B32, true, 32, Plain),
    SAddNcU64 => "s_add_nc_u64" => integer_form!(Scalar, Add, B64, false, 64, Plain),
    SAddU32 => "s_add_u32" => integer_form!(Scalar, Add, B32, false, 32, Plain),
    SAddcU32 => "s_addc_u32" => integer_form!(Scalar, Add, B32, false, 32, CarryIn),
    SAddkI32 => "s_addk_i32" => integer_form!(Scalar, Add, B32, true, 32, Immediate16),
    SAndB32 => "s_and_b32" => bitwise_form!(Scalar, And, B32, false),
    SAndB64 => "s_and_b64" => bitwise_form!(Scalar, And, B64, false),
    SAndSaveexecB32 => "s_and_saveexec_b32" => bitwise_form!(Scalar, And, B32, true),
    SAndSaveexecB64 => "s_and_saveexec_b64" => bitwise_form!(Scalar, And, B64, true),
    SAndn2B32 => "s_andn2_b32" => bitwise_form!(Scalar, AndNot, B32, false),
    SAndn2B64 => "s_andn2_b64" => bitwise_form!(Scalar, AndNot, B64, false),
    SAndn2SaveexecB32 => "s_andn2_saveexec_b32" => bitwise_form!(Scalar, AndNot, B32, true),
    SAndn2SaveexecB64 => "s_andn2_saveexec_b64" => bitwise_form!(Scalar, AndNot, B64, true),
    SAshrI32 => "s_ashr_i32" => shift_form!(Scalar, ArithmeticRight, B32, false),
    SAshrI64 => "s_ashr_i64" => shift_form!(Scalar, ArithmeticRight, B64, false),
    SBfeI32 => "s_bfe_i32" => bitfield_form!(Scalar, Extract, B32, true),
    SBfeI64 => "s_bfe_i64" => bitfield_form!(Scalar, Extract, B64, true),
    SBfeU32 => "s_bfe_u32" => bitfield_form!(Scalar, Extract, B32, false),
    SBfmB32 => "s_bfm_b32" => bitfield_form!(Scalar, Mask, B32, false),
    SBfmB64 => "s_bfm_b64" => bitfield_form!(Scalar, Mask, B64, false),
    SBitcmp0B32 => "s_bitcmp0_b32" => bitfield_form!(Scalar, TestZero, B32, false),
    SBitcmp1B32 => "s_bitcmp1_b32" => bitfield_form!(Scalar, TestOne, B32, false),
    SBitset1B32 => "s_bitset1_b32" => bitfield_form!(Scalar, SetOne, B32, false),
    SBrevB32 => "s_brev_b32" => scan_form!(Scalar, Reverse, B32),
    SBrevB64 => "s_brev_b64" => scan_form!(Scalar, Reverse, B64),
    SCmpEqI32 => "s_cmp_eq_i32" => scalar_compare_form!(Eq, I32, false),
    SCmpEqU32 => "s_cmp_eq_u32" => scalar_compare_form!(Eq, U32, false),
    SCmpEqU64 => "s_cmp_eq_u64" => scalar_compare_form!(Eq, U64, false),
    SCmpGeI32 => "s_cmp_ge_i32" => scalar_compare_form!(Ge, I32, false),
    SCmpGeU32 => "s_cmp_ge_u32" => scalar_compare_form!(Ge, U32, false),
    SCmpGtI32 => "s_cmp_gt_i32" => scalar_compare_form!(Gt, I32, false),
    SCmpGtU32 => "s_cmp_gt_u32" => scalar_compare_form!(Gt, U32, false),
    SCmpLeI32 => "s_cmp_le_i32" => scalar_compare_form!(Le, I32, false),
    SCmpLeU32 => "s_cmp_le_u32" => scalar_compare_form!(Le, U32, false),
    SCmpLgI32 => "s_cmp_lg_i32" => scalar_compare_form!(Lg, I32, false),
    SCmpLgU32 => "s_cmp_lg_u32" => scalar_compare_form!(Lg, U32, false),
    SCmpLgU64 => "s_cmp_lg_u64" => scalar_compare_form!(Lg, U64, false),
    SCmpLtI32 => "s_cmp_lt_i32" => scalar_compare_form!(Lt, I32, false),
    SCmpLtU32 => "s_cmp_lt_u32" => scalar_compare_form!(Lt, U32, false),
    SCmpkEqI32 => "s_cmpk_eq_i32" => scalar_compare_form!(Eq, I32, true),
    SCmpkEqU32 => "s_cmpk_eq_u32" => scalar_compare_form!(Eq, U32, true),
    SCmpkGtU32 => "s_cmpk_gt_u32" => scalar_compare_form!(Gt, U32, true),
    SCmpkLgI32 => "s_cmpk_lg_i32" => scalar_compare_form!(Lg, I32, true),
    SCmpkLgU32 => "s_cmpk_lg_u32" => scalar_compare_form!(Lg, U32, true),
    SCmpkLtU32 => "s_cmpk_lt_u32" => scalar_compare_form!(Lt, U32, true),
    SCselectB32 => "s_cselect_b32" => move_form!(Scalar, SelectScc, B32),
    SCselectB64 => "s_cselect_b64" => move_form!(Scalar, SelectScc, B64),
    SCvtF32I32 => "s_cvt_f32_i32" => conversion_form!(Scalar, ConversionType::I32, ConversionType::F32, Plain),
    SCvtF32U32 => "s_cvt_f32_u32" => conversion_form!(Scalar, ConversionType::U32, ConversionType::F32, Plain),
    SDenormMode => "s_denorm_mode" => AluOperation::ScalarSpecial(ScalarSpecialKind::DenormMode),
    SFf0I32B32 => "s_ff0_i32_b32" => scan_form!(Scalar, FirstZeroLow, B32),
    SFf0I32B64 => "s_ff0_i32_b64" => scan_form!(Scalar, FirstZeroLow, B64),
    SFf1I32B32 => "s_ff1_i32_b32" => scan_form!(Scalar, FirstOneLow, B32),
    SFf1I32B64 => "s_ff1_i32_b64" => scan_form!(Scalar, FirstOneLow, B64),
    SFlbitI32B32 => "s_flbit_i32_b32" => scan_form!(Scalar, LeadingSign, B32),
    SGetPcI64 => "s_get_pc_i64" => AluOperation::ScalarSpecial(ScalarSpecialKind::GetPc),
    SGetShaderCyclesU64 => "s_get_shader_cycles_u64" => AluOperation::ScalarSpecial(ScalarSpecialKind::ShaderCycles),
    SGetpcB64 => "s_getpc_b64" => AluOperation::ScalarSpecial(ScalarSpecialKind::GetPcLegacy),
    SGetregB32 => "s_getreg_b32" => AluOperation::ScalarSpecial(ScalarSpecialKind::GetHardwareRegister),
    SLshlB32 => "s_lshl_b32" => shift_form!(Scalar, Left, B32, false),
    SLshlB64 => "s_lshl_b64" => shift_form!(Scalar, Left, B64, false),
    SLshrB32 => "s_lshr_b32" => shift_form!(Scalar, LogicalRight, B32, false),
    SLshrB64 => "s_lshr_b64" => shift_form!(Scalar, LogicalRight, B64, false),
    SMaxI32 => "s_max_i32" => integer_form!(Scalar, Max, B32, true, 32, Plain),
    SMaxU32 => "s_max_u32" => integer_form!(Scalar, Max, B32, false, 32, Plain),
    SMinI32 => "s_min_i32" => integer_form!(Scalar, Min, B32, true, 32, Plain),
    SMinNumF32 => "s_min_num_f32" => float_form!(Scalar, MinNum, F32),
    SMinU32 => "s_min_u32" => integer_form!(Scalar, Min, B32, false, 32, Plain),
    SMovB32 => "s_mov_b32" => move_form!(Scalar, Plain, B32),
    SMovB64 => "s_mov_b64" => move_form!(Scalar, Plain, B64),
    SMovkI32 => "s_movk_i32" => move_form!(Scalar, Immediate16, B32),
    SMulF32 => "s_mul_f32" => float_form!(Scalar, Mul, F32),
    SMulHiI32 => "s_mul_hi_i32" => integer_form!(Scalar, MulHigh, B32, true, 32, Plain),
    SMulHiU32 => "s_mul_hi_u32" => integer_form!(Scalar, MulHigh, B32, false, 32, Plain),
    SMulI32 => "s_mul_i32" => integer_form!(Scalar, Mul, B32, true, 32, Plain),
    SMulU64 => "s_mul_u64" => integer_form!(Scalar, Mul, B64, false, 64, Plain),
    SMulkI32 => "s_mulk_i32" => integer_form!(Scalar, Mul, B32, true, 32, Immediate16),
    SNandB32 => "s_nand_b32" => bitwise_form!(Scalar, Nand, B32, false),
    SNandB64 => "s_nand_b64" => bitwise_form!(Scalar, Nand, B64, false),
    SNandSaveexecB64 => "s_nand_saveexec_b64" => bitwise_form!(Scalar, Nand, B64, true),
    SNorB32 => "s_nor_b32" => bitwise_form!(Scalar, Nor, B32, false),
    SNorB64 => "s_nor_b64" => bitwise_form!(Scalar, Nor, B64, false),
    SNorSaveexecB64 => "s_nor_saveexec_b64" => bitwise_form!(Scalar, Nor, B64, true),
    SNotB32 => "s_not_b32" => bitwise_form!(Scalar, Not, B32, false),
    SNotB64 => "s_not_b64" => bitwise_form!(Scalar, Not, B64, false),
    SOrB32 => "s_or_b32" => bitwise_form!(Scalar, Or, B32, false),
    SOrB64 => "s_or_b64" => bitwise_form!(Scalar, Or, B64, false),
    SOrSaveexecB32 => "s_or_saveexec_b32" => bitwise_form!(Scalar, Or, B32, true),
    SOrSaveexecB64 => "s_or_saveexec_b64" => bitwise_form!(Scalar, Or, B64, true),
    SOrn2B32 => "s_orn2_b32" => bitwise_form!(Scalar, OrNot, B32, false),
    SOrn2B64 => "s_orn2_b64" => bitwise_form!(Scalar, OrNot, B64, false),
    SOrn2SaveexecB64 => "s_orn2_saveexec_b64" => bitwise_form!(Scalar, OrNot, B64, true),
    SRoundMode => "s_round_mode" => AluOperation::ScalarSpecial(ScalarSpecialKind::RoundMode),
    SSendmsgRtnB32 => "s_sendmsg_rtn_b32" => AluOperation::ScalarSpecial(ScalarSpecialKind::SendMessageReturn),
    SSetregB32 => "s_setreg_b32" => AluOperation::ScalarSpecial(ScalarSpecialKind::SetHardwareRegister),
    SSetregImm32B32 => "s_setreg_imm32_b32" => AluOperation::ScalarSpecial(ScalarSpecialKind::SetHardwareRegisterImmediate),
    SSextI32I16 => "s_sext_i32_i16" => conversion_form!(Scalar, ConversionType::I16, ConversionType::I32, SignExtend),
    SSubF32 => "s_sub_f32" => float_form!(Scalar, Sub, F32),
    SSubI32 => "s_sub_i32" => integer_form!(Scalar, Sub, B32, true, 32, Plain),
    SSubU32 => "s_sub_u32" => integer_form!(Scalar, Sub, B32, false, 32, Plain),
    SSubbU32 => "s_subb_u32" => integer_form!(Scalar, Sub, B32, false, 32, CarryIn),
    SXnorB32 => "s_xnor_b32" => bitwise_form!(Scalar, Xnor, B32, false),
    SXnorB64 => "s_xnor_b64" => bitwise_form!(Scalar, Xnor, B64, false),
    SXnorSaveexecB64 => "s_xnor_saveexec_b64" => bitwise_form!(Scalar, Xnor, B64, true),
    SXorB32 => "s_xor_b32" => bitwise_form!(Scalar, Xor, B32, false),
    SXorB64 => "s_xor_b64" => bitwise_form!(Scalar, Xor, B64, false),
    SXorSaveexecB64 => "s_xor_saveexec_b64" => bitwise_form!(Scalar, Xor, B64, true),
    VAccvgprMovB32 => "v_accvgpr_mov_b32" => move_form!(Vector, AccumulatorMove, B32),
    VAccvgprReadB32 => "v_accvgpr_read_b32" => move_form!(Vector, AccumulatorRead, B32),
    VAccvgprWriteB32 => "v_accvgpr_write_b32" => move_form!(Vector, AccumulatorWrite, B32),
    VAdd3U32 => "v_add3_u32" => integer_form!(Vector, Add3, B32, false, 32, Plain),
    VAddCoU32 => "v_add_co_u32" => integer_form!(Vector, Add, B32, false, 32, CarryOut),
    VAddF16 => "v_add_f16" => float_form!(Vector, Add, F16),
    VAddF32 => "v_add_f32" => float_form!(Vector, Add, F32),
    VAddF64 => "v_add_f64" => float_form!(Vector, Add, F64),
    VAddLshlU32 => "v_add_lshl_u32" => integer_form!(Vector, AddShiftLeft, B32, false, 32, Plain),
    VAddNcU64 => "v_add_nc_u64" => integer_form!(Vector, Add, B64, false, 64, Plain),
    VAddU16 => "v_add_u16" => integer_form!(Vector, Add, B16, false, 16, Plain),
    VAddU32 => "v_add_u32" => integer_form!(Vector, Add, B32, false, 32, Plain),
    VAddcCoU32 => "v_addc_co_u32" => integer_form!(Vector, Add, B32, false, 32, CarryInOut),
    VAlignbitB32 => "v_alignbit_b32" => bitfield_form!(Vector, Align, B32, false),
    VAndB32 => "v_and_b32" => bitwise_form!(Vector, And, B32, false),
    VAndOrB32 => "v_and_or_b32" => bitwise_form!(Vector, AndOr, B32, false),
    VAshrrevI16 => "v_ashrrev_i16" => shift_form!(Vector, ArithmeticRight, B16, true),
    VAshrrevI32 => "v_ashrrev_i32" => shift_form!(Vector, ArithmeticRight, B32, true),
    VAshrrevI64 => "v_ashrrev_i64" => shift_form!(Vector, ArithmeticRight, B64, true),
    VBfeI32 => "v_bfe_i32" => bitfield_form!(Vector, Extract, B32, true),
    VBfeU32 => "v_bfe_u32" => bitfield_form!(Vector, Extract, B32, false),
    VBfiB32 => "v_bfi_b32" => bitfield_form!(Vector, Insert, B32, false),
    VBfrevB32 => "v_bfrev_b32" => scan_form!(Vector, Reverse, B32),
    VBitop2B32 => "v_bitop2_b32" => bitwise_form!(Vector, TruthTable2, B32, false),
    VBitop3B16 => "v_bitop3_b16" => bitwise_form!(Vector, TruthTable3, B16, false),
    VBitop3B32 => "v_bitop3_b32" => bitwise_form!(Vector, TruthTable3, B32, false),
    VCeilF32 => "v_ceil_f32" => float_form!(Vector, Ceil, F32),
    VCmpNeU16 => "v_cmp_ne_u16" => AluOperation::Compare(Comparison { relation: Relation::Ne, ty: CompareType::U16, writes_exec: false }),
    VCmpUF32 => "v_cmp_u_f32" => AluOperation::Compare(Comparison { relation: Relation::Unordered, ty: CompareType::F32, writes_exec: false }),
    VCndmaskB32 => "v_cndmask_b32" => move_form!(Vector, SelectVcc, B32),
    VCosF32 => "v_cos_f32" => float_form!(Vector, Cos, F32),
    VCvtBf16F32 => "v_cvt_bf16_f32" => conversion_form!(Vector, ConversionType::F32, ConversionType::Bf16, Plain),
    VCvtF16F32 => "v_cvt_f16_f32" => conversion_form!(Vector, ConversionType::F32, ConversionType::F16, Plain),
    VCvtF16I16 => "v_cvt_f16_i16" => conversion_form!(Vector, ConversionType::I16, ConversionType::F16, Plain),
    VCvtF32Bf16 => "v_cvt_f32_bf16" => conversion_form!(Vector, ConversionType::Bf16, ConversionType::F32, Plain),
    VCvtF32Bf8 => "v_cvt_f32_bf8" => conversion_form!(Vector, ConversionType::Bf8, ConversionType::F32, Plain),
    VCvtF32F16 => "v_cvt_f32_f16" => conversion_form!(Vector, ConversionType::F16, ConversionType::F32, Plain),
    VCvtF32F64 => "v_cvt_f32_f64" => conversion_form!(Vector, ConversionType::F64, ConversionType::F32, Plain),
    VCvtF32Fp8 => "v_cvt_f32_fp8" => conversion_form!(Vector, ConversionType::Fp8, ConversionType::F32, Plain),
    VCvtF32I32 => "v_cvt_f32_i32" => conversion_form!(Vector, ConversionType::I32, ConversionType::F32, Plain),
    VCvtF32U32 => "v_cvt_f32_u32" => conversion_form!(Vector, ConversionType::U32, ConversionType::F32, Plain),
    VCvtF32Ubyte0 => "v_cvt_f32_ubyte0" => conversion_form!(Vector, ConversionType::UnsignedByte(0), ConversionType::F32, Plain),
    VCvtF32Ubyte2 => "v_cvt_f32_ubyte2" => conversion_form!(Vector, ConversionType::UnsignedByte(2), ConversionType::F32, Plain),
    VCvtF32Ubyte3 => "v_cvt_f32_ubyte3" => conversion_form!(Vector, ConversionType::UnsignedByte(3), ConversionType::F32, Plain),
    VCvtF64F32 => "v_cvt_f64_f32" => conversion_form!(Vector, ConversionType::F32, ConversionType::F64, Plain),
    VCvtI32F32 => "v_cvt_i32_f32" => conversion_form!(Vector, ConversionType::F32, ConversionType::I32, Plain),
    VCvtPkBf16F32 => "v_cvt_pk_bf16_f32" => conversion_form!(Vector, ConversionType::F32, ConversionType::Bf16, Pack),
    VCvtPkF16F32 => "v_cvt_pk_f16_f32" => conversion_form!(Vector, ConversionType::F32, ConversionType::F16, Pack),
    VCvtPkF32Bf8 => "v_cvt_pk_f32_bf8" => conversion_form!(Vector, ConversionType::Bf8, ConversionType::F32, UnpackPair),
    VCvtPkF32Fp8 => "v_cvt_pk_f32_fp8" => conversion_form!(Vector, ConversionType::Fp8, ConversionType::F32, UnpackPair),
    VCvtPkrtzF16F32 => "v_cvt_pkrtz_f16_f32" => conversion_form!(Vector, ConversionType::F32, ConversionType::F16, PackRoundToZero),
    VCvtU32F32 => "v_cvt_u32_f32" => conversion_form!(Vector, ConversionType::F32, ConversionType::U32, Plain),
    VExpF32 => "v_exp_f32" => float_form!(Vector, Exp2, F32),
    VFfbhU32 => "v_ffbh_u32" => scan_form!(Vector, FirstOneHigh, B32),
    VFfblB32 => "v_ffbl_b32" => scan_form!(Vector, FirstOneLow, B32),
    VFloorF32 => "v_floor_f32" => float_form!(Vector, Floor, F32),
    VFmaF16 => "v_fma_f16" => fused_form!(Fma, F16),
    VFmaF32 => "v_fma_f32" => fused_form!(Fma, F32),
    VFmaF64 => "v_fma_f64" => fused_form!(Fma, F64),
    VFmaMixF32 => "v_fma_mix_f32" => fused_form!(Mixed, F32),
    VFmaMixhiF16 => "v_fma_mixhi_f16" => fused_form!(MixedHigh, F16),
    VFmaMixloF16 => "v_fma_mixlo_f16" => fused_form!(MixedLow, F16),
    VFmaakF32 => "v_fmaak_f32" => fused_form!(LiteralAdd, F32),
    VFmacF16 => "v_fmac_f16" => fused_form!(Accumulate, F16),
    VFmacF32 => "v_fmac_f32" => fused_form!(Accumulate, F32),
    VFmacF64 => "v_fmac_f64" => fused_form!(Accumulate, F64),
    VFmamkF32 => "v_fmamk_f32" => fused_form!(LiteralMultiply, F32),
    VFractF32 => "v_fract_f32" => float_form!(Vector, Fract, F32),
    VLdexpF32 => "v_ldexp_f32" => float_form!(Vector, Ldexp, F32),
    VLogF32 => "v_log_f32" => float_form!(Vector, Log2, F32),
    VLshlAddU32 => "v_lshl_add_u32" => integer_form!(Vector, ShiftLeftAdd, B32, false, 32, Plain),
    VLshlAddU64 => "v_lshl_add_u64" => integer_form!(Vector, ShiftLeftAdd, B64, false, 64, Plain),
    VLshlOrB32 => "v_lshl_or_b32" => bitwise_form!(Vector, ShiftLeftOr, B32, false),
    VLshlrevB16 => "v_lshlrev_b16" => shift_form!(Vector, Left, B16, true),
    VLshlrevB32 => "v_lshlrev_b32" => shift_form!(Vector, Left, B32, true),
    VLshlrevB64 => "v_lshlrev_b64" => shift_form!(Vector, Left, B64, true),
    VLshrrevB16 => "v_lshrrev_b16" => shift_form!(Vector, LogicalRight, B16, true),
    VLshrrevB32 => "v_lshrrev_b32" => shift_form!(Vector, LogicalRight, B32, true),
    VLshrrevB64 => "v_lshrrev_b64" => shift_form!(Vector, LogicalRight, B64, true),
    VMadF32 => "v_mad_f32" => fused_form!(MultiplyAdd, F32),
    VMadI32I24 => "v_mad_i32_i24" => integer_form!(Vector, MultiplyAdd, B32, true, 24, Plain),
    VMadU32 => "v_mad_u32" => integer_form!(Vector, MultiplyAdd, B32, false, 32, Plain),
    VMadU32U24 => "v_mad_u32_u24" => integer_form!(Vector, MultiplyAdd, B32, false, 24, Plain),
    VMadI64I32 => "v_mad_i64_i32" => integer_form!(Vector, MultiplyAdd, B64, true, 32, Plain),
    VMadU64U32 => "v_mad_u64_u32" => integer_form!(Vector, MultiplyAdd, B64, false, 32, Plain),
    VMax3F16 => "v_max3_f16" => float_form!(Vector, Max3, F16),
    VMax3F32 => "v_max3_f32" => float_form!(Vector, Max3, F32),
    VMax3NumF32 => "v_max3_num_f32" => float_form!(Vector, Max3Num, F32),
    VMaxF16 => "v_max_f16" => float_form!(Vector, Max, F16),
    VMaxF32 => "v_max_f32" => float_form!(Vector, Max, F32),
    VMaxF64 => "v_max_f64" => float_form!(Vector, Max, F64),
    VMaxI32 => "v_max_i32" => integer_form!(Vector, Max, B32, true, 32, Plain),
    VMaxU32 => "v_max_u32" => integer_form!(Vector, Max, B32, false, 32, Plain),
    VMbcntHiU32B32 => "v_mbcnt_hi_u32_b32" => AluOperation::Lane(LaneKind::MaskCountHigh),
    VMbcntLoU32B32 => "v_mbcnt_lo_u32_b32" => AluOperation::Lane(LaneKind::MaskCountLow),
    VMin3F16 => "v_min3_f16" => float_form!(Vector, Min3, F16),
    VMin3F32 => "v_min3_f32" => float_form!(Vector, Min3, F32),
    VMinF16 => "v_min_f16" => float_form!(Vector, Min, F16),
    VMinF32 => "v_min_f32" => float_form!(Vector, Min, F32),
    VMinF64 => "v_min_f64" => float_form!(Vector, Min, F64),
    VMinI32 => "v_min_i32" => integer_form!(Vector, Min, B32, true, 32, Plain),
    VMinU32 => "v_min_u32" => integer_form!(Vector, Min, B32, false, 32, Plain),
    VMovB32 => "v_mov_b32" => move_form!(Vector, Plain, B32),
    VMovB64 => "v_mov_b64" => move_form!(Vector, Plain, B64),
    VMulF16 => "v_mul_f16" => float_form!(Vector, Mul, F16),
    VMulF32 => "v_mul_f32" => float_form!(Vector, Mul, F32),
    VMulF64 => "v_mul_f64" => float_form!(Vector, Mul, F64),
    VMulHiI32 => "v_mul_hi_i32" => integer_form!(Vector, MulHigh, B32, true, 32, Plain),
    VMulHiU32 => "v_mul_hi_u32" => integer_form!(Vector, MulHigh, B32, false, 32, Plain),
    VMulI32I24 => "v_mul_i32_i24" => integer_form!(Vector, Mul, B32, true, 24, Plain),
    VMulLoI32 => "v_mul_lo_i32" => integer_form!(Vector, Mul, B32, true, 32, Plain),
    VMulLoU16 => "v_mul_lo_u16" => integer_form!(Vector, Mul, B16, false, 16, Plain),
    VMulLoU32 => "v_mul_lo_u32" => integer_form!(Vector, Mul, B32, false, 32, Plain),
    VMulU64 => "v_mul_u64" => integer_form!(Vector, Mul, B64, false, 64, Plain),
    VMulU32U24 => "v_mul_u32_u24" => integer_form!(Vector, Mul, B32, false, 24, Plain),
    VNotB32 => "v_not_b32" => bitwise_form!(Vector, Not, B32, false),
    VOr3B32 => "v_or3_b32" => bitwise_form!(Vector, Or3, B32, false),
    VOrB32 => "v_or_b32" => bitwise_form!(Vector, Or, B32, false),
    VPackB32F16 => "v_pack_b32_f16" => conversion_form!(Vector, ConversionType::F16, ConversionType::Bits32, Pack),
    VPermB32 => "v_perm_b32" => AluOperation::Lane(LaneKind::Permute),
    VPkAddF16 => "v_pk_add_f16" => packed_form!(Add, F16),
    VPkAddF32 => "v_pk_add_f32" => packed_form!(Add, F32),
    VPkFmaF16 => "v_pk_fma_f16" => packed_form!(Fma, F16),
    VPkFmaF32 => "v_pk_fma_f32" => packed_form!(Fma, F32),
    VPkMaxF16 => "v_pk_max_f16" => packed_form!(Max, F16),
    VPkMinF16 => "v_pk_min_f16" => packed_form!(Min, F16),
    VPkMulF16 => "v_pk_mul_f16" => packed_form!(Mul, F16),
    VPkMulF32 => "v_pk_mul_f32" => packed_form!(Mul, F32),
    VPkMulLoU16 => "v_pk_mul_lo_u16" => packed_form!(Mul, U16),
    VPkSubI16 => "v_pk_sub_i16" => packed_form!(Sub, I16),
    VRcpF32 => "v_rcp_f32" => float_form!(Vector, Reciprocal, F32),
    VRcpF64 => "v_rcp_f64" => float_form!(Vector, Reciprocal, F64),
    VReadfirstlaneB32 => "v_readfirstlane_b32" => AluOperation::Lane(LaneKind::ReadFirst),
    VReadlaneB32 => "v_readlane_b32" => AluOperation::Lane(LaneKind::Read),
    VRsqF32 => "v_rsq_f32" => float_form!(Vector, ReciprocalSqrt, F32),
    VSExpF32 => "v_s_exp_f32" => float_form!(Vector, ScalarExp2, F32),
    VSinF32 => "v_sin_f32" => float_form!(Vector, Sin, F32),
    VSqrtF32 => "v_sqrt_f32" => float_form!(Vector, Sqrt, F32),
    VSqrtF64 => "v_sqrt_f64" => float_form!(Vector, Sqrt, F64),
    VSubCoU32 => "v_sub_co_u32" => integer_form!(Vector, Sub, B32, false, 32, CarryOut),
    VSubF16 => "v_sub_f16" => float_form!(Vector, Sub, F16),
    VSubF32 => "v_sub_f32" => float_form!(Vector, Sub, F32),
    VSubU16 => "v_sub_u16" => integer_form!(Vector, Sub, B16, false, 16, Plain),
    VSubU32 => "v_sub_u32" => integer_form!(Vector, Sub, B32, false, 32, Plain),
    VSubbCoU32 => "v_subb_co_u32" => integer_form!(Vector, Sub, B32, false, 32, CarryInOut),
    VSubbrevCoU32 => "v_subbrev_co_u32" => integer_form!(Vector, Sub, B32, false, 32, ReverseCarryInOut),
    VSubrevCoU32 => "v_subrev_co_u32" => integer_form!(Vector, Sub, B32, false, 32, ReverseCarryOut),
    VSubrevF32 => "v_subrev_f32" => float_form!(Vector, SubReverse, F32),
    VSubrevU32 => "v_subrev_u32" => integer_form!(Vector, Sub, B32, false, 32, Reverse),
    VTruncF32 => "v_trunc_f32" => float_form!(Vector, Trunc, F32),
    VWritelaneB32 => "v_writelane_b32" => AluOperation::Lane(LaneKind::Write),
    VXadU32 => "v_xad_u32" => integer_form!(Vector, XorAdd, B32, false, 32, Plain),
    VXorB32 => "v_xor_b32" => bitwise_form!(Vector, Xor, B32, false),
}

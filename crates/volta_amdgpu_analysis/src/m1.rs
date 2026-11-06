//! The gfx1250 B0 m1 integer profile `gfx1250-b0-m1-int/1` and its
//! m2 superset `gfx1250-b0-m2-mode/1` (runtime MODE FP_ROUND and
//! FP_DENORM): instruction allowlists, operand forms and source layout. A
//! refused instruction makes a program unsupported; it is never skipped or
//! deleted.
//!
//! The lists equal `tests/gfx1250/m1/schema/m1_int_subset.json` and
//! `tests/gfx1250/m2/schema/m2_mode_subset.json`. Lexical
//! classification covers every line of the selected entry, including forms the
//! instruction parser refuses and unreachable code; operand forms are checked
//! on the parsed instructions.
use crate::frontend::{
    Instruction, Kernel, Module, Operand, Register, RegisterClass, SourceSpan, SpecialRegister,
};

pub const PROFILE_ID: &str = "gfx1250-b0-m1-int/1";
pub const DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-m1/1";
pub const M2_PROFILE_ID: &str = "gfx1250-b0-m2-mode/1";
pub const M2_DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-m2/1";
pub const LITMUS_PROFILE_ID: &str = "gfx1250-b0-litmus/1";
pub const LITMUS_DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-litmus/1";
pub const LITMUS2_PROFILE_ID: &str = "gfx1250-b0-litmus/2";
pub const LITMUS2_DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-litmus/2";
pub const WMMA_PROFILE_ID: &str = "gfx1250-b0-wmma/1";
pub const WMMA_DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-wmma/1";
pub const MW_PROFILE_ID: &str = "gfx1250-b0-mw/1";
pub const WMMA2_PROFILE_ID: &str = "gfx1250-b0-wmma/2";
pub const WMMA2_DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-wmma2/1";
pub const WMMA3_PROFILE_ID: &str = "gfx1250-b0-wmma/3";
pub const WMMA3_DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-wmma3/1";
pub const WMMA4_PROFILE_ID: &str = "gfx1250-b0-wmma/4";
pub const WMMA4_DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-wmma4/1";
pub const MW_DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-mw/1";
pub const MW2_PROFILE_ID: &str = "gfx1250-b0-mw/2";
pub const MW2_DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-mw2/1";
pub const MWPRED_PROFILE_ID: &str = "gfx1250-b0-mwpred/1";
pub const MWPRED_DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-mwpred/1";
pub const COMMS_PROFILE_ID: &str = "gfx1250-b0-comms/1";
pub const COMMS_DEPENDENCY_PROFILE: &str = "gfx1250-b0-sched0-comms/1";
pub const ENTRY: &str = "kernel";

/// The selected B0 profile. m2 is a strict superset of m1: the m1 lists plus
/// the MODE operations, which are pending until their hardware controls pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    M1Int,
    M2Mode,
    /// fuzzer litmus templates (raw buffers, async loads to LDS); its
    /// own allowlist, not a superset of m1.
    Litmus,
    /// The litmus profile plus the two-wave named-barrier JOIN template
    /// (`barjoin`): named barriers, S_WAIT_IDLE and the STATE_PRIV read.
    Litmus2,
    /// m2 plus BF16 WMMA (`v_wmma_f32_16x16x32_bf16`) with exact-regime
    /// result bits and the profile's WMMA spacing check.
    Wmma,
    /// The WMMA profile for 2..8 waves of one workgroup: the fuzzer's
    /// multiwave wrapper (wave index, start barrier, role branch, opaque
    /// evidence reads and the evidence store).
    Multiwave,
    /// The WMMA profile plus F16, IU8 and FP8 (16x16x64, 16x16x128) WMMA and
    /// the two fixed EXEC writes of an EXEC=0 spacer
    /// (`s_mov_b32 exec_lo, 0` and `s_mov_b32 exec_lo, -1`).
    Wmma2,
    /// The `Wmma2` profile plus the BF8 combinations, F8F6F4, F4 32x16x128
    /// and their 4-dword block-scaled forms.
    Wmma3,
    /// The `Wmma3` profile plus the 16-bit-output WMMA forms and SWMMAC.
    Wmma4,
    /// The multiwave profile with the `Wmma4` WMMA forms in the aggressor
    /// bodies (every rule applied per wave).
    Multiwave2,
    /// Fixed multi-workgroup predecessor kernels: every workgroup of a
    /// non-communicating grid runs the multiwave wrapper's evidence reads, the
    /// SE/XCC id message return and a `Wmma` body.
    MwPred,
    /// Communicating waves: the multiwave wrapper, the m1 integer body and LDS
    /// and GLOBAL producer/consumer transfers ordered by completion waits and
    /// workgroup or named barriers, for one workgroup or every workgroup of a
    /// non-communicating grid.
    Comms,
}

impl Profile {
    pub const fn id(self) -> &'static str {
        match self {
            Self::M1Int => PROFILE_ID,
            Self::M2Mode => M2_PROFILE_ID,
            Self::Litmus => LITMUS_PROFILE_ID,
            Self::Litmus2 => LITMUS2_PROFILE_ID,
            Self::Wmma => WMMA_PROFILE_ID,
            Self::Multiwave => MW_PROFILE_ID,
            Self::Wmma2 => WMMA2_PROFILE_ID,
            Self::Wmma3 => WMMA3_PROFILE_ID,
            Self::Wmma4 => WMMA4_PROFILE_ID,
            Self::Multiwave2 => MW2_PROFILE_ID,
            Self::MwPred => MWPRED_PROFILE_ID,
            Self::Comms => COMMS_PROFILE_ID,
        }
    }
    /// The multiwave profiles (`mw/1`, `mw/2`): the multiwave wrapper.
    pub const fn multiwave(self) -> bool {
        matches!(
            self,
            Self::Multiwave | Self::Multiwave2 | Self::MwPred | Self::Comms
        )
    }
    /// Profiles with BF16 WMMA and the `wmma_hazards` check.
    pub const fn wmma(self) -> bool {
        matches!(
            self,
            Self::Wmma
                | Self::Multiwave
                | Self::Wmma2
                | Self::Wmma3
                | Self::Wmma4
                | Self::Multiwave2
                | Self::MwPred
        )
    }
    /// The wmma/2 WMMA types (F16, IU8, FP8) and the EXEC=0 spacer forms.
    pub const fn wmma_types(self) -> bool {
        matches!(
            self,
            Self::Wmma2 | Self::Wmma3 | Self::Wmma4 | Self::Multiwave2
        )
    }
    /// The BF8 combinations, F8F6F4/F4 and block-scaled WMMA.
    pub const fn wmma_scale(self) -> bool {
        matches!(self, Self::Wmma3 | Self::Wmma4 | Self::Multiwave2)
    }
    /// The 16-bit-output WMMA forms and SWMMAC.
    pub const fn wmma_sparse16(self) -> bool {
        matches!(self, Self::Wmma4 | Self::Multiwave2)
    }
    /// The litmus profiles (`gfx1250-b0-litmus/1` and `/2`).
    pub const fn litmus(self) -> bool {
        matches!(self, Self::Litmus | Self::Litmus2)
    }
    /// m2 and its supersets: runtime MODE operations.
    pub const fn mode(self) -> bool {
        matches!(
            self,
            Self::M2Mode
                | Self::Wmma
                | Self::Multiwave
                | Self::Wmma2
                | Self::Wmma3
                | Self::Wmma4
                | Self::Multiwave2
                | Self::MwPred
        )
    }
    pub const fn dependency_profile(self) -> &'static str {
        match self {
            Self::M1Int => DEPENDENCY_PROFILE,
            Self::M2Mode => M2_DEPENDENCY_PROFILE,
            Self::Litmus => LITMUS_DEPENDENCY_PROFILE,
            Self::Litmus2 => LITMUS2_DEPENDENCY_PROFILE,
            Self::Wmma => WMMA_DEPENDENCY_PROFILE,
            Self::Multiwave => MW_DEPENDENCY_PROFILE,
            Self::Wmma2 => WMMA2_DEPENDENCY_PROFILE,
            Self::Wmma3 => WMMA3_DEPENDENCY_PROFILE,
            Self::Wmma4 => WMMA4_DEPENDENCY_PROFILE,
            Self::Multiwave2 => MW2_DEPENDENCY_PROFILE,
            Self::MwPred => MWPRED_DEPENDENCY_PROFILE,
            Self::Comms => COMMS_DEPENDENCY_PROFILE,
        }
    }
    pub fn pending_ops(self) -> &'static [&'static str] {
        match self {
            Self::M1Int => &PENDING_OPS,
            Self::M2Mode
            | Self::Wmma
            | Self::Multiwave
            | Self::Wmma2
            | Self::Wmma3
            | Self::Wmma4
            | Self::Multiwave2
            | Self::MwPred => &M2_PENDING_OPS,
            Self::Litmus | Self::Litmus2 | Self::Comms => &[],
        }
    }
    pub fn wait_alu_encodings(self) -> &'static [u16] {
        match self {
            Self::M1Int => &WAIT_ALU_ENCODINGS,
            Self::M2Mode
            | Self::Wmma
            | Self::Multiwave
            | Self::Wmma2
            | Self::Wmma3
            | Self::Wmma4
            | Self::Multiwave2
            | Self::MwPred => &M2_WAIT_ALU_ENCODINGS,
            Self::Litmus => &[],
            Self::Litmus2 => &LITMUS2_WAIT_ALU_ENCODINGS,
            Self::Comms => &WAIT_ALU_ENCODINGS,
        }
    }
    pub fn native_profile(self) -> crate::config::NativeProfile {
        match self {
            Self::M1Int => crate::config::NativeProfile::Gfx1250B0M1Int,
            Self::M2Mode => crate::config::NativeProfile::Gfx1250B0M2Mode,
            Self::Litmus => crate::config::NativeProfile::Gfx1250B0Litmus,
            Self::Litmus2 => crate::config::NativeProfile::Gfx1250B0Litmus2,
            Self::Wmma => crate::config::NativeProfile::Gfx1250B0Wmma,
            Self::Multiwave => crate::config::NativeProfile::Gfx1250B0Multiwave,
            Self::Wmma2 => crate::config::NativeProfile::Gfx1250B0Wmma2,
            Self::Wmma3 => crate::config::NativeProfile::Gfx1250B0Wmma3,
            Self::Wmma4 => crate::config::NativeProfile::Gfx1250B0Wmma4,
            Self::Multiwave2 => crate::config::NativeProfile::Gfx1250B0Multiwave2,
            Self::MwPred => crate::config::NativeProfile::Gfx1250B0MwPred,
            Self::Comms => crate::config::NativeProfile::Gfx1250B0Comms,
        }
    }
    pub fn body_ops(self) -> &'static [&'static str] {
        match self {
            Self::M1Int | Self::M2Mode => &BODY_OPS,
            Self::Litmus => &LITMUS_BODY_OPS,
            Self::Litmus2 => &LITMUS2_BODY_OPS,
            Self::Wmma | Self::Multiwave | Self::MwPred => &WMMA_BODY_OPS,
            Self::Wmma2 => &WMMA2_BODY_OPS,
            Self::Wmma3 => &WMMA3_BODY_OPS,
            Self::Wmma4 | Self::Multiwave2 => &WMMA4_BODY_OPS,
            Self::Comms => &COMMS_BODY_OPS,
        }
    }
    pub fn wrapper_mnemonics(self) -> &'static [&'static str] {
        match self {
            Self::M1Int | Self::M2Mode | Self::Wmma | Self::Wmma2 | Self::Wmma3 | Self::Wmma4 => {
                &WRAPPER_MNEMONICS
            }
            Self::Litmus | Self::Litmus2 => &LITMUS_WRAPPER_MNEMONICS,
            Self::Multiwave | Self::Multiwave2 => &MW_WRAPPER_MNEMONICS,
            Self::MwPred => &MWPRED_WRAPPER_MNEMONICS,
            Self::Comms => &COMMS_WRAPPER_MNEMONICS,
        }
    }
    pub fn control_only_ops(self) -> &'static [&'static str] {
        match self {
            Self::M1Int | Self::M2Mode | Self::Wmma | Self::Wmma2 | Self::Wmma3 | Self::Wmma4 => {
                &CONTROL_ONLY_OPS
            }
            Self::Litmus
            | Self::Litmus2
            | Self::Multiwave
            | Self::Multiwave2
            | Self::MwPred
            | Self::Comms => &[],
        }
    }
}

/// Template operations of the qualified litmus templates (bufload, bufstore,
/// asynclds); equals `tests/gfx1250/litmus/schema/litmus_b0_subset.json`.
pub const LITMUS_BODY_OPS: [&str; 29] = [
    "buffer_load_b32",
    "buffer_load_b64",
    "buffer_load_b96",
    "buffer_load_b128",
    "buffer_store_b32",
    "buffer_store_b64",
    "buffer_store_b96",
    "buffer_store_b128",
    "global_load_async_to_lds_b32",
    "global_load_async_to_lds_b64",
    "global_load_async_to_lds_b128",
    "ds_store_b32",
    "ds_load_b32",
    "global_store_b32",
    "global_store_b128",
    "s_add_nc_u64",
    "s_mov_b32",
    "s_or_b32",
    "v_mov_b32",
    "v_lshlrev_b32",
    "v_add_nc_u32",
    "v_lshl_add_u32",
    "v_cmp_eq_u32",
    "v_cndmask_b32",
    "v_or_b32",
    "s_wait_loadcnt",
    "s_wait_storecnt",
    "s_wait_dscnt",
    "s_wait_asynccnt",
];
pub const LITMUS_WRAPPER_MNEMONICS: [&str; 6] = [
    "global_prefetch_b8",
    "s_endpgm",
    "s_load_b64",
    "s_mov_b64",
    "s_wait_kmcnt",
    "v_nop",
];
/// `gfx1250-b0-litmus/2`: the litmus/1 operations plus the `barjoin`
/// template's; equals `tests/gfx1250/litmus2/schema/litmus_b0_t2_subset.json`.
pub const LITMUS2_BODY_OPS: [&str; 41] = [
    "buffer_load_b32",
    "buffer_load_b64",
    "buffer_load_b96",
    "buffer_load_b128",
    "buffer_store_b32",
    "buffer_store_b64",
    "buffer_store_b96",
    "buffer_store_b128",
    "global_load_async_to_lds_b32",
    "global_load_async_to_lds_b64",
    "global_load_async_to_lds_b128",
    "ds_store_b32",
    "ds_load_b32",
    "global_store_b32",
    "global_store_b128",
    "s_add_nc_u64",
    "s_mov_b32",
    "s_or_b32",
    "v_mov_b32",
    "v_lshlrev_b32",
    "v_add_nc_u32",
    "v_lshl_add_u32",
    "v_cmp_eq_u32",
    "v_cndmask_b32",
    "v_or_b32",
    "s_wait_loadcnt",
    "s_wait_storecnt",
    "s_wait_dscnt",
    "s_wait_asynccnt",
    "s_and_b32",
    "s_barrier_init",
    "s_barrier_join",
    "s_barrier_signal",
    "s_barrier_wait",
    "s_cbranch_scc1",
    "s_cmp_lg_u32",
    "s_getreg_b32",
    "s_nop",
    "s_wait_alu",
    "s_wait_idle",
    "v_readfirstlane_b32",
];
/// litmus/2 `s_wait_alu`: only the full ALU drain after `v_readfirstlane_b32`.
const LITMUS2_WAIT_ALU_ENCODINGS: [u16; 1] = [0];
/// The STATE_PRIV read of the `barjoin` template (whole register).
pub const STATE_PRIV_HWREG: &str = "HW_REG_WAVE_STATE_PRIV";
/// Operations of the litmus templates that are not qualified (asyncstore,
/// barjoin, bufswz): refused by name.
const LITMUS_UNQUALIFIED_OPS: [&str; 12] = [
    "global_store_async_from_lds_b32",
    "global_store_async_from_lds_b64",
    "global_store_async_from_lds_b128",
    "s_wait_idle",
    "s_barrier_init",
    "s_barrier_join",
    "s_barrier_signal",
    "s_barrier_wait",
    "s_cbranch_scc1",
    "s_getreg_b32",
    "global_load_b32",
    "s_sub_co_u32",
];

pub const BODY_OPS: [&str; 40] = [
    "s_add_co_u32",
    "s_sub_co_u32",
    "s_add_co_ci_u32",
    "s_mul_i32",
    "s_and_b32",
    "s_or_b32",
    "s_xor_b32",
    "s_lshl_b32",
    "s_lshr_b32",
    "s_min_u32",
    "s_max_u32",
    "s_min_i32",
    "s_cselect_b32",
    "s_mov_b32",
    "s_not_b32",
    "s_brev_b32",
    "v_add_nc_u32",
    "v_sub_nc_u32",
    "v_subrev_nc_u32",
    "v_and_b32",
    "v_or_b32",
    "v_xor_b32",
    "v_lshlrev_b32",
    "v_lshrrev_b32",
    "v_min_u32",
    "v_max_u32",
    "v_mov_b32",
    "v_not_b32",
    "v_bfrev_b32",
    "v_mul_lo_u32",
    "v_readfirstlane_b32",
    "v_cmp_eq_u32",
    "v_cmp_ne_u32",
    "v_cmp_gt_u32",
    "v_cmp_lt_u32",
    "v_cndmask_b32",
    "s_wait_alu",
    "s_nop",
    "v_nop",
    "s_wait_kmcnt",
];
/// The WMMA and multiwave profiles: the m1 body operations plus BF16 WMMA.
pub const WMMA_BODY_OPS: [&str; 41] = [
    "s_add_co_u32",
    "s_sub_co_u32",
    "s_add_co_ci_u32",
    "s_mul_i32",
    "s_and_b32",
    "s_or_b32",
    "s_xor_b32",
    "s_lshl_b32",
    "s_lshr_b32",
    "s_min_u32",
    "s_max_u32",
    "s_min_i32",
    "s_cselect_b32",
    "s_mov_b32",
    "s_not_b32",
    "s_brev_b32",
    "v_add_nc_u32",
    "v_sub_nc_u32",
    "v_subrev_nc_u32",
    "v_and_b32",
    "v_or_b32",
    "v_xor_b32",
    "v_lshlrev_b32",
    "v_lshrrev_b32",
    "v_min_u32",
    "v_max_u32",
    "v_mov_b32",
    "v_not_b32",
    "v_bfrev_b32",
    "v_mul_lo_u32",
    "v_readfirstlane_b32",
    "v_cmp_eq_u32",
    "v_cmp_ne_u32",
    "v_cmp_gt_u32",
    "v_cmp_lt_u32",
    "v_cndmask_b32",
    "s_wait_alu",
    "s_nop",
    "v_nop",
    "s_wait_kmcnt",
    "v_wmma_f32_16x16x32_bf16",
];
/// The `gfx1250-b0-wmma/2` profile: the WMMA body operations plus F16, IU8
/// and FP8 WMMA. The EXEC=0 spacer uses `s_mov_b32` (exec_lo forms only, see
/// the operand forms), `v_nop` and `v_mov_b32`.
pub const WMMA2_BODY_OPS: [&str; 45] = [
    "s_add_co_u32",
    "s_sub_co_u32",
    "s_add_co_ci_u32",
    "s_mul_i32",
    "s_and_b32",
    "s_or_b32",
    "s_xor_b32",
    "s_lshl_b32",
    "s_lshr_b32",
    "s_min_u32",
    "s_max_u32",
    "s_min_i32",
    "s_cselect_b32",
    "s_mov_b32",
    "s_not_b32",
    "s_brev_b32",
    "v_add_nc_u32",
    "v_sub_nc_u32",
    "v_subrev_nc_u32",
    "v_and_b32",
    "v_or_b32",
    "v_xor_b32",
    "v_lshlrev_b32",
    "v_lshrrev_b32",
    "v_min_u32",
    "v_max_u32",
    "v_mov_b32",
    "v_not_b32",
    "v_bfrev_b32",
    "v_mul_lo_u32",
    "v_readfirstlane_b32",
    "v_cmp_eq_u32",
    "v_cmp_ne_u32",
    "v_cmp_gt_u32",
    "v_cmp_lt_u32",
    "v_cndmask_b32",
    "s_wait_alu",
    "s_nop",
    "v_nop",
    "s_wait_kmcnt",
    "v_wmma_f32_16x16x32_bf16",
    "v_wmma_f32_16x16x32_f16",
    "v_wmma_i32_16x16x64_iu8",
    "v_wmma_f32_16x16x64_fp8_fp8",
    "v_wmma_f32_16x16x128_fp8_fp8",
];
/// The `gfx1250-b0-wmma/3` profile: the `wmma/2` body operations plus the BF8
/// combinations, F8F6F4, F4 32x16x128 and their block-scaled forms.
pub const WMMA3_BODY_OPS: [&str; 57] = [
    "s_add_co_u32",
    "s_sub_co_u32",
    "s_add_co_ci_u32",
    "s_mul_i32",
    "s_and_b32",
    "s_or_b32",
    "s_xor_b32",
    "s_lshl_b32",
    "s_lshr_b32",
    "s_min_u32",
    "s_max_u32",
    "s_min_i32",
    "s_cselect_b32",
    "s_mov_b32",
    "s_not_b32",
    "s_brev_b32",
    "v_add_nc_u32",
    "v_sub_nc_u32",
    "v_subrev_nc_u32",
    "v_and_b32",
    "v_or_b32",
    "v_xor_b32",
    "v_lshlrev_b32",
    "v_lshrrev_b32",
    "v_min_u32",
    "v_max_u32",
    "v_mov_b32",
    "v_not_b32",
    "v_bfrev_b32",
    "v_mul_lo_u32",
    "v_readfirstlane_b32",
    "v_cmp_eq_u32",
    "v_cmp_ne_u32",
    "v_cmp_gt_u32",
    "v_cmp_lt_u32",
    "v_cndmask_b32",
    "s_wait_alu",
    "s_nop",
    "v_nop",
    "s_wait_kmcnt",
    "v_wmma_f32_16x16x32_bf16",
    "v_wmma_f32_16x16x32_f16",
    "v_wmma_i32_16x16x64_iu8",
    "v_wmma_f32_16x16x64_fp8_fp8",
    "v_wmma_f32_16x16x128_fp8_fp8",
    "v_wmma_f32_16x16x64_fp8_bf8",
    "v_wmma_f32_16x16x64_bf8_fp8",
    "v_wmma_f32_16x16x64_bf8_bf8",
    "v_wmma_f32_16x16x128_fp8_bf8",
    "v_wmma_f32_16x16x128_bf8_fp8",
    "v_wmma_f32_16x16x128_bf8_bf8",
    "v_wmma_f32_16x16x128_f8f6f4",
    "v_wmma_scale_f32_16x16x128_f8f6f4",
    "v_wmma_scale16_f32_16x16x128_f8f6f4",
    "v_wmma_f32_32x16x128_f4",
    "v_wmma_scale_f32_32x16x128_f4",
    "v_wmma_scale16_f32_32x16x128_f4",
];
/// The `gfx1250-b0-wmma/4` profile: the `wmma/3` body operations plus the
/// 16-bit-output WMMA forms and SWMMAC.
pub const WMMA4_BODY_OPS: [&str; 82] = [
    "s_add_co_u32",
    "s_sub_co_u32",
    "s_add_co_ci_u32",
    "s_mul_i32",
    "s_and_b32",
    "s_or_b32",
    "s_xor_b32",
    "s_lshl_b32",
    "s_lshr_b32",
    "s_min_u32",
    "s_max_u32",
    "s_min_i32",
    "s_cselect_b32",
    "s_mov_b32",
    "s_not_b32",
    "s_brev_b32",
    "v_add_nc_u32",
    "v_sub_nc_u32",
    "v_subrev_nc_u32",
    "v_and_b32",
    "v_or_b32",
    "v_xor_b32",
    "v_lshlrev_b32",
    "v_lshrrev_b32",
    "v_min_u32",
    "v_max_u32",
    "v_mov_b32",
    "v_not_b32",
    "v_bfrev_b32",
    "v_mul_lo_u32",
    "v_readfirstlane_b32",
    "v_cmp_eq_u32",
    "v_cmp_ne_u32",
    "v_cmp_gt_u32",
    "v_cmp_lt_u32",
    "v_cndmask_b32",
    "s_wait_alu",
    "s_nop",
    "v_nop",
    "s_wait_kmcnt",
    "v_wmma_f32_16x16x32_bf16",
    "v_wmma_f32_16x16x32_f16",
    "v_wmma_i32_16x16x64_iu8",
    "v_wmma_f32_16x16x64_fp8_fp8",
    "v_wmma_f32_16x16x128_fp8_fp8",
    "v_wmma_f32_16x16x64_fp8_bf8",
    "v_wmma_f32_16x16x64_bf8_fp8",
    "v_wmma_f32_16x16x64_bf8_bf8",
    "v_wmma_f32_16x16x128_fp8_bf8",
    "v_wmma_f32_16x16x128_bf8_fp8",
    "v_wmma_f32_16x16x128_bf8_bf8",
    "v_wmma_f32_16x16x128_f8f6f4",
    "v_wmma_scale_f32_16x16x128_f8f6f4",
    "v_wmma_scale16_f32_16x16x128_f8f6f4",
    "v_wmma_f32_32x16x128_f4",
    "v_wmma_scale_f32_32x16x128_f4",
    "v_wmma_scale16_f32_32x16x128_f4",
    "v_wmma_f16_16x16x32_f16",
    "v_wmma_bf16_16x16x32_bf16",
    "v_wmma_bf16f32_16x16x32_bf16",
    "v_wmma_f16_16x16x64_fp8_fp8",
    "v_wmma_f16_16x16x64_fp8_bf8",
    "v_wmma_f16_16x16x64_bf8_fp8",
    "v_wmma_f16_16x16x64_bf8_bf8",
    "v_wmma_f16_16x16x128_fp8_fp8",
    "v_wmma_f16_16x16x128_fp8_bf8",
    "v_wmma_f16_16x16x128_bf8_fp8",
    "v_wmma_f16_16x16x128_bf8_bf8",
    "v_swmmac_f32_16x16x64_f16",
    "v_swmmac_f32_16x16x64_bf16",
    "v_swmmac_f16_16x16x64_f16",
    "v_swmmac_bf16_16x16x64_bf16",
    "v_swmmac_bf16f32_16x16x64_bf16",
    "v_swmmac_f32_16x16x128_fp8_fp8",
    "v_swmmac_f32_16x16x128_fp8_bf8",
    "v_swmmac_f32_16x16x128_bf8_fp8",
    "v_swmmac_f32_16x16x128_bf8_bf8",
    "v_swmmac_f16_16x16x128_fp8_fp8",
    "v_swmmac_f16_16x16x128_fp8_bf8",
    "v_swmmac_f16_16x16x128_bf8_fp8",
    "v_swmmac_f16_16x16x128_bf8_bf8",
    "v_swmmac_i32_16x16x128_iu8",
];
/// The multiwave wrapper: the m1 wrapper plus the wave-index branch, the
/// start barrier, the opaque evidence reads (`s_getreg_b32` only of the
/// whole HW_ID1 register) and the evidence store.
pub const MW_WRAPPER_MNEMONICS: [&str; 23] = [
    "global_prefetch_b8",
    "global_store_b32",
    "global_store_b128",
    "s_and_b32",
    "s_barrier_signal",
    "s_barrier_wait",
    "s_branch",
    "s_cbranch_scc1",
    "s_cmp_eq_u32",
    "s_endpgm",
    "s_get_shader_cycles_u64",
    "s_load_b64",
    "s_lshl_b32",
    "s_lshr_b32",
    "s_mov_b32",
    "s_mov_b64",
    "s_wait_alu",
    "s_wait_kmcnt",
    "v_and_b32",
    "v_lshlrev_b32",
    "v_mov_b32",
    "v_nop",
    "v_readfirstlane_b32",
];
/// The multi-workgroup predecessor wrapper: the multiwave wrapper plus the
/// SE/XCC id message return (`s_sendmsg_rtn_b32`, message 0x87).
pub const MWPRED_WRAPPER_MNEMONICS: [&str; 24] = [
    "global_prefetch_b8",
    "global_store_b32",
    "global_store_b128",
    "s_and_b32",
    "s_barrier_signal",
    "s_barrier_wait",
    "s_branch",
    "s_cbranch_scc1",
    "s_cmp_eq_u32",
    "s_endpgm",
    "s_get_shader_cycles_u64",
    "s_load_b64",
    "s_lshl_b32",
    "s_lshr_b32",
    "s_mov_b32",
    "s_mov_b64",
    "s_sendmsg_rtn_b32",
    "s_wait_alu",
    "s_wait_kmcnt",
    "v_and_b32",
    "v_lshlrev_b32",
    "v_mov_b32",
    "v_nop",
    "v_readfirstlane_b32",
];
/// `gfx1250-b0-comms/1` body: the m1 integer operations (filler) plus the
/// producer/consumer transfers, their completion waits and the barriers.
pub const COMMS_BODY_OPS: [&str; 57] = [
    "s_add_co_u32",
    "s_sub_co_u32",
    "s_add_co_ci_u32",
    "s_mul_i32",
    "s_and_b32",
    "s_or_b32",
    "s_xor_b32",
    "s_lshl_b32",
    "s_lshr_b32",
    "s_min_u32",
    "s_max_u32",
    "s_min_i32",
    "s_cselect_b32",
    "s_mov_b32",
    "s_not_b32",
    "s_brev_b32",
    "v_add_nc_u32",
    "v_sub_nc_u32",
    "v_subrev_nc_u32",
    "v_and_b32",
    "v_or_b32",
    "v_xor_b32",
    "v_lshlrev_b32",
    "v_lshrrev_b32",
    "v_min_u32",
    "v_max_u32",
    "v_mov_b32",
    "v_not_b32",
    "v_bfrev_b32",
    "v_mul_lo_u32",
    "v_readfirstlane_b32",
    "v_cmp_eq_u32",
    "v_cmp_ne_u32",
    "v_cmp_gt_u32",
    "v_cmp_lt_u32",
    "v_cndmask_b32",
    "s_wait_alu",
    "s_nop",
    "v_nop",
    "s_wait_kmcnt",
    "ds_store_b32",
    "ds_store_b64",
    "ds_store_b128",
    "ds_load_b32",
    "ds_load_b64",
    "ds_load_b128",
    "global_store_b64",
    "global_load_b32",
    "global_load_b64",
    "global_load_b128",
    "s_wait_dscnt",
    "s_wait_loadcnt",
    "s_wait_storecnt",
    "s_barrier_init",
    "s_barrier_join",
    "s_barrier_signal",
    "s_barrier_wait",
];
/// The communicating-waves wrapper: the multiwave wrapper plus TTMP9 (the
/// workgroup id of a grid) through `s_mov_b32`.
pub const COMMS_WRAPPER_MNEMONICS: [&str; 23] = MW_WRAPPER_MNEMONICS;
/// The only hwreg the multiwave wrapper reads besides MODE: whole HW_ID1.
pub const EVIDENCE_HWREG: &str = "HW_REG_WAVE_HW_ID1";
/// Enabled only by an explicit request after its hardware controls pass.
pub const PENDING_OPS: [&str; 1] = ["s_max_i32"];
pub const WRAPPER_MNEMONICS: [&str; 12] = [
    "global_prefetch_b8",
    "global_store_b32",
    "s_endpgm",
    "s_load_b64",
    "s_mov_b32",
    "s_mov_b64",
    "s_wait_alu",
    "s_wait_kmcnt",
    "v_and_b32",
    "v_lshlrev_b32",
    "v_mov_b32",
    "v_nop",
];
/// Accepted only in calibration-control kernels.
pub const CONTROL_ONLY_OPS: [&str; 1] = ["s_cmp_eq_u32"];
/// `s_wait_alu` encodings of `0x0` and the six single `depctr_<field>(0)`.
pub const WAIT_ALU_ENCODINGS: [u16; 7] = [0x0000, 0xff9e, 0x0f9f, 0xf19f, 0xff9d, 0xfe9f, 0xff83];
/// m2 also accepts `0xfffe`, a literal whose only zero field is SA_SDST. It
/// differs from `depctr_sa_sdst(0)` (`0xff9e`) only in bits 6:5, which name no
/// field.
pub const M2_WAIT_ALU_ENCODINGS: [u16; 8] = [
    0x0000, 0xff9e, 0x0f9f, 0xf19f, 0xff9d, 0xfe9f, 0xff83, 0xfffe,
];
/// m2 pending operations: `s_max_i32` (m1 gate) and the MODE operations,
/// each enabled only by an explicit request after its hardware controls pass.
pub const M2_PENDING_OPS: [&str; 6] = [
    "s_max_i32",
    "s_getreg_b32",
    "s_setreg_b32",
    "s_setreg_imm32_b32",
    "s_round_mode",
    "s_denorm_mode",
];
/// The only hwreg spelling the m2 profile accepts for MODE. Aliases that
/// assemble to the same encoding are refused rather than assumed equal.
pub const MODE_HWREG: &str = "HW_REG_WAVE_MODE";
pub const MODE_OPS: [&str; 5] = [
    "s_getreg_b32",
    "s_setreg_b32",
    "s_setreg_imm32_b32",
    "s_round_mode",
    "s_denorm_mode",
];

/// Canonical m1 allowlist text whose sha256 the report carries: sorted lists in
/// a key-sorted compact JSON object, as Python `json.dumps(..., sort_keys=True,
/// separators=(",", ":"))` writes it.
pub fn canonical_allowlist() -> String {
    canonical_allowlist_for(Profile::M1Int)
}

/// Canonical allowlist text of either profile (same layout as m1).
pub fn canonical_allowlist_for(profile: Profile) -> String {
    fn list(items: &[&str]) -> String {
        let mut items = items.to_vec();
        items.sort_unstable();
        format!(
            "[{}]",
            items
                .iter()
                .map(|item| format!("\"{item}\""))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
    format!(
        "{{\"body\":{},\"control_only\":{},\"pending\":{},\"wrapper\":{}}}",
        list(profile.body_ops()),
        list(profile.control_only_ops()),
        list(profile.pending_ops()),
        list(profile.wrapper_mnemonics())
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Candidate,
    Twin,
    Control,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RefusalReason {
    NotInProfile,
    PendingHardwareControl,
    ExcludedOp,
    ModeAccess,
    FloatingPoint,
    OperandFormNotInProfile,
    /// m2: a MODE operation outside the selected FP_ROUND/FP_DENORM forms.
    ModeFormNotInProfile,
    /// litmus: an operation of a litmus template that is not qualified.
    LitmusTemplateNotQualified,
}

impl RefusalReason {
    pub const fn name(self) -> &'static str {
        match self {
            Self::NotInProfile => "NOT_IN_PROFILE",
            Self::PendingHardwareControl => "PENDING_HARDWARE_CONTROL",
            Self::ExcludedOp => "EXCLUDED_OP",
            Self::ModeAccess => "MODE_ACCESS",
            Self::FloatingPoint => "FLOATING_POINT",
            Self::OperandFormNotInProfile => "OPERAND_FORM_NOT_IN_PROFILE",
            Self::ModeFormNotInProfile => "MODE_FORM_NOT_IN_PROFILE",
            Self::LitmusTemplateNotQualified => "LITMUS_TEMPLATE_NOT_QUALIFIED",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefusedOp {
    pub line: usize,
    pub mnemonic: String,
    pub reason: RefusalReason,
}

fn stem(mnemonic: &str) -> &str {
    mnemonic.trim_end_matches("_e32").trim_end_matches("_e64")
}

fn floating_point(stem: &str) -> bool {
    stem.starts_with("v_wmma")
        || stem.starts_with("v_swmmac")
        || stem.split('_').any(|part| {
            matches!(
                part,
                "f16" | "f32" | "f64" | "bf16" | "fp8" | "bf8" | "f8" | "f6" | "f4" | "num"
            )
        })
}

fn mentions_scratch_base(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("scratch")
}

/// litmus/2: the litmus/1 unqualified operations that stay unqualified
/// (asyncstore, bufswz).
const LITMUS2_UNQUALIFIED_OPS: [&str; 5] = [
    "global_store_async_from_lds_b32",
    "global_store_async_from_lds_b64",
    "global_store_async_from_lds_b128",
    "global_load_b32",
    "s_sub_co_u32",
];

/// Classify one mnemonic of the selected entry under the m1 profile.
pub fn refusal(mnemonic: &str, role: Role, enabled_pending: &[String]) -> Option<RefusalReason> {
    refusal_for(Profile::M1Int, mnemonic, role, enabled_pending)
}

/// Classify one mnemonic of the selected entry under `profile`.
pub fn refusal_for(
    profile: Profile,
    mnemonic: &str,
    role: Role,
    enabled_pending: &[String],
) -> Option<RefusalReason> {
    let stem = stem(mnemonic);
    if profile.body_ops().contains(&stem) || profile.wrapper_mnemonics().contains(&stem) {
        return None;
    }
    if profile.pending_ops().contains(&stem) {
        return (!enabled_pending.iter().any(|op| op == stem))
            .then_some(RefusalReason::PendingHardwareControl);
    }
    if profile.control_only_ops().contains(&stem) {
        return (role != Role::Control).then_some(RefusalReason::NotInProfile);
    }
    Some(if MODE_OPS.contains(&stem) {
        RefusalReason::ModeAccess
    } else if stem.starts_with("v_perm_pk16") || (profile.mode() && stem == "s_set_vgpr_msb") {
        RefusalReason::ExcludedOp
    } else if floating_point(stem) {
        RefusalReason::FloatingPoint
    } else if profile == Profile::Litmus && LITMUS_UNQUALIFIED_OPS.contains(&stem)
        || profile == Profile::Litmus2 && LITMUS2_UNQUALIFIED_OPS.contains(&stem)
    {
        RefusalReason::LitmusTemplateNotQualified
    } else {
        RefusalReason::NotInProfile
    })
}

/// Lexical pass over the raw lines of the selected entry, before instruction
/// parsing. Scratch-base operands are refused for every mnemonic.
pub fn classify_lines(
    lines: &[(String, SourceSpan)],
    role: Role,
    enabled_pending: &[String],
) -> Vec<RefusedOp> {
    classify_lines_for(Profile::M1Int, lines, role, enabled_pending)
}

/// [`classify_lines`] under `profile`.
pub fn classify_lines_for(
    profile: Profile,
    lines: &[(String, SourceSpan)],
    role: Role,
    enabled_pending: &[String],
) -> Vec<RefusedOp> {
    lines
        .iter()
        .filter_map(|(text, span)| {
            let mnemonic = text.split_whitespace().next().unwrap_or_default();
            let reason = if mentions_scratch_base(text) {
                Some(RefusalReason::ExcludedOp)
            } else if profile.multiwave()
                && mnemonic == "s_getreg_b32"
                && text.contains(&format!("hwreg({EVIDENCE_HWREG})"))
            {
                // The evidence read; its exact form is checked after parsing.
                None
            } else {
                refusal_for(profile, mnemonic, role, enabled_pending)
            };
            reason.map(|reason| RefusedOp {
                line: span.line,
                mnemonic: mnemonic.into(),
                reason,
            })
        })
        .collect()
}

fn scalar(operand: &Operand, count: u16) -> bool {
    matches!(operand, Operand::Register(Register { class: RegisterClass::Scalar, count: c, .. }) if *c == count)
}
fn vector(operand: &Operand) -> bool {
    matches!(
        operand,
        Operand::Register(Register {
            class: RegisterClass::Vector,
            count: 1,
            ..
        })
    )
}
fn exec_lo(operand: &Operand) -> bool {
    matches!(
        operand,
        Operand::Register(Register {
            special: Some(SpecialRegister::ExecLo),
            ..
        })
    )
}
fn vcc(operand: &Operand) -> bool {
    matches!(
        operand,
        Operand::Register(Register {
            special: Some(SpecialRegister::VccLo | SpecialRegister::Vcc),
            ..
        })
    )
}
fn scalar_source(operand: &Operand) -> bool {
    scalar(operand, 1)
        || matches!(operand, Operand::Integer(_))
        || matches!(
            operand,
            Operand::Register(Register {
                special: Some(SpecialRegister::VccLo),
                ..
            })
        )
}
fn vector_source(operand: &Operand) -> bool {
    vector(operand) || scalar_source(operand)
}

/// The m2 MODE forms: FP_ROUND (offset 0) or FP_DENORM (offset 4), width 4,
/// `HW_REG_WAVE_MODE` spelling; immediates are 4-bit field values only.
fn mode_form(instruction: &Instruction) -> bool {
    let o = instruction.operands.as_slice();
    let field = |operand: &Operand| matches!(operand, Operand::HardwareRegister { name, offset: 0 | 4, width: 4 } if name == MODE_HWREG);
    let nibble = |operand: &Operand| matches!(operand, Operand::Integer(0..=15));
    instruction.modifiers.is_empty()
        && match stem(&instruction.mnemonic) {
            "s_getreg_b32" => matches!(o, [d, h] if scalar(d, 1) && field(h)),
            "s_setreg_b32" => matches!(o, [h, s] if field(h) && scalar(s, 1)),
            "s_setreg_imm32_b32" => matches!(o, [h, v] if field(h) && nibble(v)),
            "s_round_mode" | "s_denorm_mode" => matches!(o, [v] if nibble(v)),
            _ => false,
        }
}

/// The multiwave evidence read: `s_getreg_b32 sD, hwreg(HW_REG_WAVE_HW_ID1)`
/// of the whole register, no modifiers.
fn evidence_getreg(instruction: &Instruction) -> bool {
    stem(&instruction.mnemonic) == "s_getreg_b32"
        && instruction.modifiers.is_empty()
        && matches!(instruction.operands.as_slice(), [d, Operand::HardwareRegister { name, offset: 0, width: 32 }]
            if scalar(d, 1) && name == EVIDENCE_HWREG)
}

fn null(operand: &Operand) -> bool {
    matches!(
        operand,
        Operand::Register(Register {
            special: Some(SpecialRegister::Null),
            ..
        })
    )
}
fn vectors(operand: &Operand, count: u16) -> bool {
    matches!(operand, Operand::Register(Register { class: RegisterClass::Vector, count: c, .. }) if *c == count)
}
fn zero(operand: &Operand) -> bool {
    matches!(operand, Operand::Integer(0))
}
fn memory_dwords(stem: &str) -> u16 {
    match stem.rsplit('_').next() {
        Some("b128") => 4,
        Some("b96") => 3,
        Some("b64") => 2,
        _ => 1,
    }
}

/// Exact operand forms of the litmus profile on a parsed instruction.
fn litmus_operand_form(instruction: &Instruction) -> bool {
    let o = instruction.operands.as_slice();
    let modifiers_ok = |allowed: &[&str]| {
        instruction
            .modifiers
            .iter()
            .all(|m| allowed.contains(&m.name.as_str()))
    };
    let explicit64 = instruction.mnemonic.ends_with("_e64");
    let stem = stem(&instruction.mnemonic);
    let plain = modifiers_ok(&[]);
    match stem {
        "s_add_nc_u64" => {
            plain && matches!(o, [d, a, Operand::Integer(_)] if scalar(d, 2) && scalar(a, 2))
        }
        "s_mov_b32" => plain && matches!(o, [d, a] if scalar(d, 1) && scalar_source(a)),
        "s_or_b32" => {
            plain && matches!(o, [d, a, b] if scalar(d, 1) && scalar_source(a) && scalar_source(b))
        }
        "s_mov_b64" => plain && matches!(o, [d, Operand::Integer(_)] if scalar(d, 2)),
        "v_mov_b32" => plain && matches!(o, [d, a] if vector(d) && vector_source(a)),
        "v_lshlrev_b32" | "v_add_nc_u32" | "v_or_b32" => {
            plain && matches!(o, [d, a, b] if vector(d) && vector_source(a) && vector(b))
        }
        "v_lshl_add_u32" => {
            plain
                && matches!(o, [d, a, b, c] if vector(d) && vector(a) && vector_source(b) && vector_source(c))
        }
        "v_cmp_eq_u32" => {
            plain && matches!(o, [d, a, b] if vcc(d) && vector_source(a) && vector(b))
        }
        "v_cndmask_b32" => {
            plain
                && explicit64
                && matches!(o, [d, a, b, c] if vector(d) && vector_source(a) && vector_source(b) && vcc(c))
        }
        "v_nop" | "s_endpgm" => plain && o.is_empty(),
        "s_wait_kmcnt" | "s_wait_loadcnt" | "s_wait_storecnt" | "s_wait_dscnt"
        | "s_wait_asynccnt" => plain && matches!(o, [z] if zero(z)),
        "s_load_b64" => {
            plain && matches!(o, [d, b, Operand::Integer(_)] if scalar(d, 2) && scalar(b, 2))
        }
        "global_prefetch_b8" => {
            modifiers_ok(&["scope", "offset"]) && matches!(o, [a, b] if vector(a) && scalar(b, 2))
        }
        "global_store_b32" | "global_store_b128" => {
            modifiers_ok(&["offset"])
                && matches!(o, [a, d, b] if vector(a) && vectors(d, memory_dwords(stem)) && scalar(b, 2))
        }
        "ds_store_b32" => {
            modifiers_ok(&["offset"]) && matches!(o, [a, d] if vector(a) && vector(d))
        }
        "ds_load_b32" => modifiers_ok(&["offset"]) && matches!(o, [d, a] if vector(d) && vector(a)),
        "global_load_async_to_lds_b32"
        | "global_load_async_to_lds_b64"
        | "global_load_async_to_lds_b128" => {
            modifiers_ok(&["offset"])
                && matches!(o, [l, a, b] if vector(l) && vector(a) && scalar(b, 2))
        }
        _ if stem.starts_with("buffer_load_b") || stem.starts_with("buffer_store_b") => {
            let off = |operand: &Operand| matches!(operand, Operand::Symbol(s) if s == "off");
            let offen = instruction.modifiers.iter().any(|m| m.name == "offen");
            modifiers_ok(&["offen", "offset"])
                && matches!(o, [d, a, r, n] if vectors(d, memory_dwords(stem))
                    && ((vector(a) && offen) || (off(a) && !offen))
                    && scalar(r, 4) && null(n))
        }
        _ => false,
    }
}

/// Exact operand forms of the litmus/2 `barjoin` operations.
fn barjoin_operand_form(instruction: &Instruction) -> bool {
    let o = instruction.operands.as_slice();
    let plain = instruction.modifiers.is_empty();
    let label = |operand: &Operand| matches!(operand, Operand::Symbol(_));
    let m0 = |operand: &Operand| {
        matches!(
            operand,
            Operand::Register(Register {
                special: Some(SpecialRegister::M0),
                ..
            })
        )
    };
    match stem(&instruction.mnemonic) {
        "s_mov_b32" => plain && matches!(o, [d, Operand::Integer(_)] if m0(d)),
        "s_and_b32" => {
            plain && matches!(o, [d, a, Operand::Integer(_)] if scalar(d, 1) && scalar(a, 1))
        }
        "s_cmp_lg_u32" => plain && matches!(o, [a, b] if scalar(a, 1) && scalar_source(b)),
        "s_cbranch_scc1" => plain && matches!(o, [t] if label(t)),
        "v_readfirstlane_b32" => plain && matches!(o, [d, a] if scalar(d, 1) && vector(a)),
        "s_wait_alu" => plain && matches!(o, [Operand::Integer(0)]),
        "s_wait_idle" => plain && o.is_empty(),
        "s_nop" => plain && matches!(o, [Operand::Integer(0..=7)]),
        "s_barrier_init" => plain && matches!(o, [r] if m0(r)),
        "s_barrier_join" => plain && matches!(o, [Operand::Integer(0..=16)]),
        "s_barrier_signal" | "s_barrier_wait" => plain && matches!(o, [Operand::Integer(1..=16)]),
        "s_getreg_b32" => {
            plain
                && matches!(o, [d, Operand::HardwareRegister { name, offset: 0, width: 32 }]
                    if scalar(d, 1) && name == STATE_PRIV_HWREG)
        }
        _ => false,
    }
}

/// Exact operand forms of the profile on a parsed instruction.
/// `gfx1250-b0-comms/1` transfer, wait and barrier forms.
fn comms_operand_form(instruction: &Instruction) -> bool {
    let o = instruction.operands.as_slice();
    let offset_only = instruction.modifiers.iter().all(|m| m.name == "offset");
    let plain = instruction.modifiers.is_empty();
    let stem = stem(&instruction.mnemonic);
    let m0 = |operand: &Operand| {
        matches!(
            operand,
            Operand::Register(Register {
                special: Some(SpecialRegister::M0),
                ..
            })
        )
    };
    match stem {
        "ds_store_b32" | "ds_store_b64" | "ds_store_b128" => {
            offset_only && matches!(o, [a, d] if vector(a) && vectors(d, memory_dwords(stem)))
        }
        "ds_load_b32" | "ds_load_b64" | "ds_load_b128" => {
            offset_only && matches!(o, [d, a] if vectors(d, memory_dwords(stem)) && vector(a))
        }
        "global_store_b32" | "global_store_b64" | "global_store_b128" => {
            offset_only
                && matches!(o, [a, d, b] if vector(a) && vectors(d, memory_dwords(stem)) && scalar(b, 2))
        }
        "global_load_b32" | "global_load_b64" | "global_load_b128" => {
            offset_only
                && matches!(o, [d, a, b] if vectors(d, memory_dwords(stem)) && vector(a) && scalar(b, 2))
        }
        "s_wait_dscnt" | "s_wait_loadcnt" | "s_wait_storecnt" => {
            plain && matches!(o, [z] if zero(z))
        }
        "s_mov_b32" => {
            plain
                && (matches!(o, [d, Operand::Integer(_)] if m0(d))
                    || matches!(o, [d, t] if scalar(d, 1) && workgroup_id_x(t)))
        }
        "s_barrier_init" => plain && matches!(o, [r] if m0(r)),
        "s_barrier_join" => plain && matches!(o, [Operand::Integer(1..=2)]),
        "s_barrier_signal" | "s_barrier_wait" => {
            plain && matches!(o, [Operand::Integer(-1 | 1 | 2)])
        }
        _ => false,
    }
}

fn operand_form(instruction: &Instruction, profile: Profile) -> bool {
    if profile == Profile::Comms && comms_operand_form(instruction) {
        return true;
    }
    if profile == Profile::Comms && stem(&instruction.mnemonic) == "s_load_b64" {
        // Communicating waves: the only scalar memory read is the kernarg pointer
        // load. A scalar load of the output region could observe data another
        // wave stored through the vector memory path, and scalar loads are not
        // kept coherent with those stores.
        return instruction.modifiers.is_empty() && kernarg_pointer_load(&instruction.operands);
    }
    if profile == Profile::Litmus {
        return litmus_operand_form(instruction);
    }
    if profile == Profile::Litmus2 {
        return litmus_operand_form(instruction) || barjoin_operand_form(instruction);
    }
    let o = instruction.operands.as_slice();
    let modifiers_ok = |allowed: &[&str]| {
        instruction
            .modifiers
            .iter()
            .all(|m| allowed.contains(&m.name.as_str()))
    };
    let explicit64 = instruction.mnemonic.ends_with("_e64");
    let stem = stem(&instruction.mnemonic);
    let plain = modifiers_ok(&[]);
    match stem {
        "s_add_co_u32" | "s_sub_co_u32" | "s_add_co_ci_u32" | "s_mul_i32" | "s_and_b32"
        | "s_or_b32" | "s_xor_b32" | "s_lshl_b32" | "s_lshr_b32" | "s_min_u32" | "s_max_u32"
        | "s_min_i32" | "s_max_i32" | "s_cselect_b32" => {
            plain && matches!(o, [d, a, b] if scalar(d, 1) && scalar_source(a) && scalar_source(b))
        }
        "s_mov_b32" | "s_not_b32" | "s_brev_b32" => {
            plain
                && (matches!(o, [d, a] if scalar(d, 1) && scalar_source(a))
                    || (profile.wmma_types()
                        && stem == "s_mov_b32"
                        && matches!(o, [d, Operand::Integer(0 | -1)] if exec_lo(d)))
                    || (profile == Profile::MwPred
                        && stem == "s_mov_b32"
                        && matches!(o, [d, t] if scalar(d, 1) && workgroup_id_x(t))))
        }
        "s_cmp_eq_u32" => plain && matches!(o, [a, b] if scalar_source(a) && scalar_source(b)),
        "s_mov_b64" => plain && matches!(o, [d, Operand::Integer(_)] if scalar(d, 2)),
        "v_add_nc_u32" | "v_sub_nc_u32" | "v_subrev_nc_u32" | "v_and_b32" | "v_or_b32"
        | "v_xor_b32" | "v_lshlrev_b32" | "v_lshrrev_b32" | "v_min_u32" | "v_max_u32"
        | "v_mul_lo_u32" => {
            plain && matches!(o, [d, a, b] if vector(d) && vector_source(a) && vector_source(b))
        }
        "v_mov_b32" | "v_not_b32" | "v_bfrev_b32" => {
            plain && matches!(o, [d, a] if vector(d) && vector_source(a))
        }
        "v_cmp_eq_u32" | "v_cmp_ne_u32" | "v_cmp_gt_u32" | "v_cmp_lt_u32" => {
            plain && matches!(o, [d, a, b] if vcc(d) && vector_source(a) && vector(b))
        }
        "v_cndmask_b32" => {
            plain
                && matches!(o, [d, a, b, c] if vector(d) && vector_source(a) && vector(b) && vcc(c))
        }
        "v_readfirstlane_b32" => plain && matches!(o, [d, a] if scalar(d, 1) && vector(a)),
        "s_nop" => plain && matches!(o, [Operand::Integer(0..=7)]),
        "v_nop" | "s_endpgm" => plain && o.is_empty(),
        "s_wait_alu" => {
            plain
                && matches!(o, [Operand::Integer(v)] if u16::try_from(*v).is_ok_and(|v| profile.wait_alu_encodings().contains(&v)))
        }
        "s_wait_kmcnt" => plain && matches!(o, [Operand::Integer(0)]),
        "s_load_b64" => {
            plain && matches!(o, [d, b, Operand::Integer(_)] if scalar(d, 2) && scalar(b, 2))
        }
        "global_prefetch_b8" => {
            modifiers_ok(&["scope", "offset"]) && matches!(o, [a, b] if vector(a) && scalar(b, 2))
        }
        "global_store_b32" => {
            modifiers_ok(&["offset"])
                && matches!(o, [a, d, b] if vector(a) && vector(d) && scalar(b, 2))
        }
        "v_wmma_f32_16x16x32_bf16" if profile.wmma() => {
            plain && !explicit64 && o.len() == 4 && o.iter().all(|operand| vectors(operand, 8))
        }
        "v_wmma_f32_16x16x32_f16" | "v_wmma_i32_16x16x64_iu8" | "v_wmma_f32_16x16x64_fp8_fp8"
            if profile.wmma_types() =>
        {
            plain && !explicit64 && o.len() == 4 && o.iter().all(|operand| vectors(operand, 8))
        }
        "v_wmma_f32_16x16x64_fp8_bf8"
        | "v_wmma_f32_16x16x64_bf8_fp8"
        | "v_wmma_f32_16x16x64_bf8_bf8"
            if profile.wmma_scale() =>
        {
            plain && !explicit64 && o.len() == 4 && o.iter().all(|operand| vectors(operand, 8))
        }
        "v_wmma_f32_16x16x128_fp8_bf8"
        | "v_wmma_f32_16x16x128_bf8_fp8"
        | "v_wmma_f32_16x16x128_bf8_bf8"
            if profile.wmma_scale() =>
        {
            plain
                && !explicit64
                && matches!(o, [d, a, b, c] if vectors(d, 8) && vectors(a, 16) && vectors(b, 16) && vectors(c, 8))
        }
        "v_wmma_f32_16x16x128_f8f6f4"
        | "v_wmma_scale_f32_16x16x128_f8f6f4"
        | "v_wmma_scale16_f32_16x16x128_f8f6f4"
        | "v_wmma_f32_32x16x128_f4"
        | "v_wmma_scale_f32_32x16x128_f4"
        | "v_wmma_scale16_f32_32x16x128_f4"
            if profile.wmma_scale() =>
        {
            !explicit64 && small_wmma_form(stem, instruction)
        }
        "v_wmma_f16_16x16x32_f16"
        | "v_wmma_bf16_16x16x32_bf16"
        | "v_wmma_f16_16x16x64_fp8_fp8"
        | "v_wmma_f16_16x16x64_fp8_bf8"
        | "v_wmma_f16_16x16x64_bf8_fp8"
        | "v_wmma_f16_16x16x64_bf8_bf8"
            if profile.wmma_sparse16() =>
        {
            plain
                && !explicit64
                && matches!(o, [d, a, b, c] if vectors(d, 4) && vectors(a, 8) && vectors(b, 8) && vectors(c, 4))
        }
        "v_wmma_bf16f32_16x16x32_bf16" if profile.wmma_sparse16() => {
            plain
                && !explicit64
                && matches!(o, [d, a, b, c] if vectors(d, 4) && vectors(a, 8) && vectors(b, 8) && vectors(c, 8))
        }
        "v_wmma_f16_16x16x128_fp8_fp8"
        | "v_wmma_f16_16x16x128_fp8_bf8"
        | "v_wmma_f16_16x16x128_bf8_fp8"
        | "v_wmma_f16_16x16x128_bf8_bf8"
            if profile.wmma_sparse16() =>
        {
            plain
                && !explicit64
                && matches!(o, [d, a, b, c] if vectors(d, 4) && vectors(a, 16) && vectors(b, 16) && vectors(c, 4))
        }
        _ if profile.wmma_sparse16() && stem.starts_with("v_swmmac_") => {
            // D (also C), packed A, B, sparsity index (one VGPR for 16-bit
            // inputs, an even pair for 8-bit inputs); no modifiers.
            let d_width = if stem.starts_with("v_swmmac_f16_") || stem.starts_with("v_swmmac_bf16_")
            {
                4
            } else {
                8
            };
            let index = if stem.contains("16x16x128") { 2 } else { 1 };
            plain
                && !explicit64
                && matches!(o, [d, a, b, i] if vectors(d, d_width) && vectors(a, 8) && vectors(b, 16)
                    && vectors(i, index)
                    && (index == 1 || matches!(i, Operand::Register(Register { index, .. }) if index % 2 == 0)))
        }
        "v_wmma_f32_16x16x128_fp8_fp8" if profile.wmma_types() => {
            plain
                && !explicit64
                && matches!(o, [d, a, b, c] if vectors(d, 8) && vectors(a, 16) && vectors(b, 16) && vectors(c, 8))
        }
        "s_sendmsg_rtn_b32" if profile == Profile::MwPred => {
            plain && matches!(o, [d, Operand::Integer(0x87)] if scalar(d, 1))
        }
        _ if profile.multiwave() => multiwave_wrapper_form(instruction),
        _ => false,
    }
}

/// `gfx1250-b0-wmma/3` operand forms of the F8F6F4 and F4 32x16x128 WMMA
/// and their block-scaled forms: documented tuple widths per format, VGPR
/// scale operands (one register, or an even-aligned pair for SCALE16), a
/// legal pair of data and scale formats (the documented table of A/B data and
/// scale formats),
/// no modifiers but the format and scale-lane/format ones, and a 32x16 C
/// equal to D or disjoint from it.
fn small_wmma_form(stem: &str, instruction: &Instruction) -> bool {
    let o = instruction.operands.as_slice();
    let scale16 = stem.starts_with("v_wmma_scale16_");
    let scaled = scale16 || stem.starts_with("v_wmma_scale_");
    let f4 = stem.ends_with("_32x16x128_f4");
    let mut allowed = Vec::new();
    if !f4 {
        allowed.extend(["matrix_a_fmt", "matrix_b_fmt"]);
    }
    if scaled {
        allowed.extend([
            "matrix_a_scale",
            "matrix_b_scale",
            "matrix_a_scale_fmt",
            "matrix_b_scale_fmt",
        ]);
    }
    let value = |name: &str| {
        instruction
            .modifiers
            .iter()
            .find(|m| m.name == name)
            .and_then(|m| m.value.as_deref())
    };
    if !instruction
        .modifiers
        .iter()
        .all(|m| allowed.contains(&m.name.as_str()))
    {
        return false;
    }
    // (registers, class): 0 = 8-bit, 1 = 6-bit, 2 = 4-bit.
    let format = |name: &str| match value(name) {
        None | Some("MATRIX_FMT_FP8" | "MATRIX_FMT_BF8") => Some((16, 0)),
        Some("MATRIX_FMT_FP6" | "MATRIX_FMT_BF6") => Some((12, 1)),
        Some("MATRIX_FMT_FP4") => Some((8, 2)),
        Some(_) => None,
    };
    let (d_width, (a_width, a_class), (b_width, b_class)) = if f4 {
        (16, (16, 2), (8, 2))
    } else {
        match (format("matrix_a_fmt"), format("matrix_b_fmt")) {
            (Some(a), Some(b)) => (8, a, b),
            _ => return false,
        }
    };
    let span = |operand: &Operand| match operand {
        Operand::Register(Register {
            class: RegisterClass::Vector,
            index,
            count,
            ..
        }) => Some((*index, *index + *count)),
        _ => None,
    };
    let [d, a, b, c, rest @ ..] = o else {
        return false;
    };
    if !(vectors(d, d_width) && vectors(a, a_width) && vectors(b, b_width) && vectors(c, d_width)) {
        return false;
    }
    if f4 {
        let (Some((d0, d1)), Some((c0, c1))) = (span(d), span(c)) else {
            return false;
        };
        if (d0, d1) != (c0, c1) && d0 < c1 && c0 < d1 {
            return false;
        }
    }
    if !scaled {
        return rest.is_empty();
    }
    let [sa, sb] = rest else {
        return false;
    };
    let width = if scale16 { 2 } else { 1 };
    for operand in [sa, sb] {
        if !vectors(operand, width) || (scale16 && span(operand).is_none_or(|(i, _)| i % 2 != 0)) {
            return false;
        }
    }
    // 0 = E8M0, 1 = E5M3, 2 = E4M3.
    let scale = |name: &str| match value(name) {
        None | Some("MATRIX_SCALE_FMT_E8") => Some(0),
        Some("MATRIX_SCALE_FMT_E5M3") => Some(1),
        Some("MATRIX_SCALE_FMT_E4M3") => Some(2),
        Some(_) => None,
    };
    let lanes_ok = |name: &str| {
        matches!(
            value(name),
            None | Some("MATRIX_SCALE_ROW0" | "MATRIX_SCALE_ROW1")
        )
    };
    let (Some(sa), Some(sb)) = (scale("matrix_a_scale_fmt"), scale("matrix_b_scale_fmt")) else {
        return false;
    };
    lanes_ok("matrix_a_scale")
        && lanes_ok("matrix_b_scale")
        && matches!(
            (a_class, sa, b_class, sb),
            (_, 0, _, 0)
                | (0 | 1, 0, 2, 1 | 2)
                | (2, 1 | 2, 0 | 1, 0)
                | (2, 1, 2, 1)
                | (2, 2, 2, 2)
        )
}

/// Exact operand forms of the multiwave wrapper's own instructions.
/// TTMP9: the architected workgroup id X on gfx1250.
/// `s_load_b64 s[2:3], s[0:1], 0x0`: the kernarg pointer load of the wrappers.
fn kernarg_pointer_load(operands: &[Operand]) -> bool {
    let pair = |operand: &Operand, first: u16| {
        matches!(operand, Operand::Register(Register { class: RegisterClass::Scalar, index, count: 2, special: None })
            if *index == first)
    };
    matches!(operands, [d, b, Operand::Integer(0)] if pair(d, 2) && pair(b, 0))
}

fn workgroup_id_x(operand: &Operand) -> bool {
    matches!(
        operand,
        Operand::Register(Register {
            class: RegisterClass::Temporary,
            index: 9,
            count: 1,
            ..
        })
    )
}

fn multiwave_wrapper_form(instruction: &Instruction) -> bool {
    let o = instruction.operands.as_slice();
    let plain = instruction.modifiers.is_empty();
    let label = |operand: &Operand| matches!(operand, Operand::Symbol(_));
    match stem(&instruction.mnemonic) {
        "s_barrier_signal" | "s_barrier_wait" => plain && matches!(o, [Operand::Integer(-1)]),
        "s_get_shader_cycles_u64" => plain && matches!(o, [d] if scalar(d, 2)),
        "s_branch" | "s_cbranch_scc1" => plain && matches!(o, [t] if label(t)),
        "global_store_b128" => {
            instruction.modifiers.iter().all(|m| m.name == "offset")
                && matches!(o, [a, d, b] if vector(a) && vectors(d, 4) && scalar(b, 2))
        }
        _ => false,
    }
}

/// Operand-form refusals of every parsed instruction of the entry whose
/// mnemonic the lexical pass accepted (m1 profile).
pub fn check_operand_forms(
    kernel: &Kernel,
    role: Role,
    enabled_pending: &[String],
) -> Vec<RefusedOp> {
    check_operand_forms_for(Profile::M1Int, kernel, role, enabled_pending)
}

/// [`check_operand_forms`] under `profile`. Every MODE mnemonic is checked,
/// enabled or not, so an unsupported MODE form is always named as such.
pub fn check_operand_forms_for(
    profile: Profile,
    kernel: &Kernel,
    role: Role,
    enabled_pending: &[String],
) -> Vec<RefusedOp> {
    let instructions = &kernel.instructions;
    instructions
        .iter()
        .enumerate()
        .filter_map(|(position, i)| {
            // mwpred: a message return must be drained by the very next
            // instruction, so its destination is never read while pending.
            let drained = stem(&i.mnemonic) != "s_sendmsg_rtn_b32"
                || instructions.get(position + 1).is_some_and(|next| {
                    stem(&next.mnemonic) == "s_wait_kmcnt"
                        && next.operands == [Operand::Integer(0)]
                        && next.modifiers.is_empty()
                });
            let evidence = profile.multiwave() && evidence_getreg(i);
            let mode = profile.mode() && !evidence && MODE_OPS.contains(&stem(&i.mnemonic));
            let reason = if mode && !mode_form(i) {
                RefusalReason::ModeFormNotInProfile
            } else if !mode
                && !evidence
                && refusal_for(profile, &i.mnemonic, role, enabled_pending).is_none()
                && (!operand_form(i, profile) || !drained)
            {
                RefusalReason::OperandFormNotInProfile
            } else {
                return None;
            };
            Some(RefusedOp {
                line: i.span.line,
                mnemonic: i.mnemonic.clone(),
                reason,
            })
        })
        .collect()
}

fn literal(operand: &Operand) -> bool {
    matches!(operand, Operand::Integer(v) if !(-16..=64).contains(v))
}

/// Encoded byte size of a profile instruction (gfx1250 encodings: SOP 4,
/// SMEM 8, VOP1/VOP2/VOPC 4, VOP3 8, VGLOBAL 12, plus a 4-byte literal).
/// A VOP2 operation whose second source is not a VGPR is encoded as VOP3.
pub fn encoded_size(instruction: &Instruction) -> Option<u64> {
    encoded_size_for(Profile::M1Int, instruction)
}

/// [`encoded_size`] under `profile`; m2 adds the MODE forms (SOPK/SOPP 4,
/// S_SETREG_IMM32_B32 8 with its literal).
pub fn encoded_size_for(profile: Profile, instruction: &Instruction) -> Option<u64> {
    let o = instruction.operands.as_slice();
    if profile.mode() {
        match stem(&instruction.mnemonic) {
            "s_getreg_b32" | "s_setreg_b32" | "s_round_mode" | "s_denorm_mode" => return Some(4),
            "s_setreg_imm32_b32" => return Some(8),
            _ => {}
        }
    }
    if profile.wmma() {
        // VOP3P 8; VOP3PX2 (block-scaled WMMA) 16; SOPP/SOP1 4; VGLOBAL 12.
        match stem(&instruction.mnemonic) {
            "v_wmma_f32_16x16x32_bf16" => return Some(8),
            "v_wmma_f32_16x16x32_f16"
            | "v_wmma_i32_16x16x64_iu8"
            | "v_wmma_f32_16x16x64_fp8_fp8"
            | "v_wmma_f32_16x16x128_fp8_fp8"
                if profile.wmma_types() =>
            {
                return Some(8);
            }
            "v_wmma_f32_16x16x64_fp8_bf8"
            | "v_wmma_f32_16x16x64_bf8_fp8"
            | "v_wmma_f32_16x16x64_bf8_bf8"
            | "v_wmma_f32_16x16x128_fp8_bf8"
            | "v_wmma_f32_16x16x128_bf8_fp8"
            | "v_wmma_f32_16x16x128_bf8_bf8"
            | "v_wmma_f32_16x16x128_f8f6f4"
            | "v_wmma_f32_32x16x128_f4"
                if profile.wmma_scale() =>
            {
                return Some(8);
            }
            "v_wmma_scale_f32_16x16x128_f8f6f4"
            | "v_wmma_scale16_f32_16x16x128_f8f6f4"
            | "v_wmma_scale_f32_32x16x128_f4"
            | "v_wmma_scale16_f32_32x16x128_f4"
                if profile.wmma_scale() =>
            {
                return Some(16);
            }
            stem if profile.wmma_sparse16() && WMMA4_BODY_OPS[57..].contains(&stem) => {
                return Some(8);
            }
            "s_barrier_signal"
            | "s_barrier_wait"
            | "s_branch"
            | "s_cbranch_scc1"
            | "s_get_shader_cycles_u64"
                if profile.multiwave() =>
            {
                return Some(4);
            }
            "global_store_b128" if profile.multiwave() => return Some(12),
            "s_sendmsg_rtn_b32" if profile == Profile::MwPred => return Some(4),
            _ => {}
        }
    }
    let literal_bytes = if o.iter().any(literal) { 4 } else { 0 };
    let stem = stem(&instruction.mnemonic);
    let explicit64 = instruction.mnemonic.ends_with("_e64");
    if profile == Profile::Litmus2 {
        // SOPP/SOPK/SOPC 4; SOP2/SOP1 with a literal 8 (generic rule below).
        match stem {
            "s_barrier_init" | "s_barrier_join" | "s_barrier_signal" | "s_barrier_wait"
            | "s_cbranch_scc1" | "s_getreg_b32" | "s_wait_idle" => return Some(4),
            _ => {}
        }
    }
    if profile == Profile::Comms {
        // VGLOBAL 12, DS 8; waits and barrier operations 4 (SOPP).
        match stem {
            _ if stem.starts_with("global_") => return Some(12),
            _ if stem.starts_with("ds_") => return Some(8),
            "s_wait_loadcnt" | "s_wait_storecnt" | "s_wait_dscnt" | "s_barrier_init"
            | "s_barrier_join" => return Some(4),
            _ => {}
        }
    }
    if profile.litmus() {
        // VBUFFER and VGLOBAL 12, DS 8, VOP3 8 (+ literal); the rest as m1.
        match stem {
            _ if stem.starts_with("buffer_") || stem.starts_with("global_") => return Some(12),
            _ if stem.starts_with("ds_") => return Some(8),
            "s_wait_loadcnt" | "s_wait_storecnt" | "s_wait_dscnt" | "s_wait_asynccnt" => {
                return Some(4);
            }
            "v_lshl_add_u32" => return Some(8 + literal_bytes),
            _ => {}
        }
    }
    Some(match stem {
        "s_nop" | "s_endpgm" | "s_wait_alu" | "s_wait_kmcnt" => 4,
        "s_load_b64" => 8,
        "global_prefetch_b8" | "global_store_b32" => 12,
        "v_nop" => 4,
        _ if stem.starts_with("s_") => 4 + literal_bytes,
        "v_mul_lo_u32" => 8 + literal_bytes,
        "v_mov_b32" | "v_not_b32" | "v_bfrev_b32" | "v_readfirstlane_b32" => {
            (if explicit64 { 8 } else { 4 }) + literal_bytes
        }
        _ if stem.starts_with("v_") => {
            if explicit64 || !o.get(2).is_some_and(vector) {
                8 + literal_bytes
            } else {
                4 + literal_bytes
            }
        }
        _ => return None,
    })
}

/// Source layout of the sized m1 entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelLayout {
    pub size_bytes: u64,
    pub instructions: usize,
    pub padding_after_size_bytes: u64,
}

const TARGET_SPELLINGS: [&str; 2] = [
    "amdgcn-amd-amdhsa-unknown-gfx1250",
    "amdgcn-amd-amdhsa--gfx1250",
];

/// Target, code-object and `.size` requirements; each failure is one
/// diagnostic `(line, message)`.
pub fn check_layout(module: &Module) -> Result<KernelLayout, Vec<(usize, String)>> {
    check_layout_for(Profile::M1Int, module)
}

/// [`check_layout`] under `profile`.
pub fn check_layout_for(
    profile: Profile,
    module: &Module,
) -> Result<KernelLayout, Vec<(usize, String)>> {
    let mut diagnostics = Vec::new();
    for (what, spelling) in [
        ("assembler target directive", &module.target_directive),
        ("amdhsa.target metadata", &module.metadata_target),
    ] {
        match spelling {
            Some(spelling) if TARGET_SPELLINGS.contains(&spelling.as_str()) => {}
            Some(spelling) => diagnostics.push((
                0,
                format!("{what} `{spelling}` is not a profile spelling of bare gfx1250"),
            )),
            None => diagnostics.push((0, format!("missing {what}"))),
        }
    }
    if module.target.as_deref() != Some("gfx1250") {
        diagnostics.push((
            0,
            "profile requires bare gfx1250 with no feature fields".into(),
        ));
    }
    if module.code_object_version != Some(6) {
        diagnostics.push((0, "profile requires AMDHSA code-object version 6".into()));
    }
    if module.metadata_version.as_deref() != Some(&[1, 2]) {
        diagnostics.push((0, "profile requires AMDGPU metadata version 1.2".into()));
    }
    let Some(kernel) = module.kernels.iter().find(|k| k.name == ENTRY) else {
        diagnostics.push((0, format!("no entry kernel named `{ENTRY}`")));
        return Err(diagnostics);
    };
    match &kernel.size_directive {
        Some(size) if size.expression.split_whitespace().collect::<String>() == ".-kernel" => {}
        Some(size) => diagnostics.push((
            size.span.line,
            format!("entry .size must be `.-{ENTRY}`, not `{}`", size.expression),
        )),
        None => diagnostics.push((
            0,
            format!("entry requires an explicit `.size {ENTRY}, .-{ENTRY}`"),
        )),
    }
    if kernel.instructions.last().map(|i| i.mnemonic.as_str()) != Some("s_endpgm") {
        diagnostics.push((
            kernel.instructions.last().map_or(0, |i| i.span.line),
            "the sized extent must end exactly after s_endpgm".into(),
        ));
    }
    let mut size_bytes = 0u64;
    for instruction in &kernel.instructions {
        match encoded_size_for(profile, instruction) {
            Some(bytes) => size_bytes += bytes,
            None => diagnostics.push((
                instruction.span.line,
                format!("no profile encoding size for `{}`", instruction.mnemonic),
            )),
        }
    }
    // Alignment is relative to the entry start, which the profile's
    // `.p2align 8` places on a 256-byte boundary.
    let mut position = size_bytes;
    for directive in &kernel.trailing_text_directives {
        let numbers: Option<Vec<i64>> = directive
            .arguments
            .split(',')
            .map(|part| crate::frontend::integer_literal(part.trim()))
            .collect();
        let added = match (directive.name.as_str(), numbers.as_deref()) {
            (".p2alignl" | ".p2align", Some([shift, ..])) if (0..=8).contains(shift) => {
                let alignment = 1u64 << shift;
                Some(position.next_multiple_of(alignment) - position)
            }
            (".fill", Some([count, rest @ ..]))
                if *count >= 0 && rest.first().is_none_or(|size| (1..=8).contains(size)) =>
            {
                Some(*count as u64 * rest.first().map_or(1, |size| *size as u64))
            }
            _ => None,
        };
        match added {
            Some(bytes) => position += bytes,
            None => diagnostics.push((
                directive.span.line,
                format!(
                    "unsupported padding directive `{} {}` after .size",
                    directive.name, directive.arguments
                ),
            )),
        }
    }
    if diagnostics.is_empty() {
        Ok(KernelLayout {
            size_bytes,
            instructions: kernel.instructions.len(),
            padding_after_size_bytes: position - size_bytes,
        })
    } else {
        Err(diagnostics)
    }
}

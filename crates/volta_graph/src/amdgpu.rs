//! Additive native AMD trace metadata. Checks interpret these records without borrowing
//! PTX synchronization certificates or treating trace order as memory completion.

use crate::{AccessMode, EventId, ThreadIx};
mod tensor;
pub use tensor::{AmdTensorDescriptor, AmdTensorFootprint, TENSOR_BOUNDS_ABORT_REASON};

/// Architectures with an explicitly implemented native graph contract.
/// Target features remain in `Provenance::targets`; recognizing a target does
/// not imply that every instruction or execution mode is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdArchitecture {
    Gfx942,
    Gfx950,
    Gfx1250,
}

impl AmdArchitecture {
    pub fn from_target(target: &str) -> Option<Self> {
        let mut fields = target.split(':');
        let architecture = match fields.next()? {
            "gfx942" => Self::Gfx942,
            "gfx950" => Self::Gfx950,
            "gfx1250" => Self::Gfx1250,
            _ => return None,
        };
        let (mut xnack, mut sramecc) = (false, false);
        for feature in fields {
            // gfx1250 XNACK is hardwired on, rather than a selectable mode.
            // Only the bare target is in the implemented feature envelope.
            if architecture == Self::Gfx1250 {
                return None;
            }
            let seen = match feature {
                "xnack+" | "xnack-" => &mut xnack,
                "sramecc+" | "sramecc-" => &mut sramecc,
                _ => return None,
            };
            if *seen {
                return None;
            }
            *seen = true;
        }
        Some(architecture)
    }

    pub fn from_targets(targets: &[String]) -> Option<Self> {
        if targets.len() != 1 {
            return None;
        }
        Self::from_target(&targets[0])
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Gfx942 => "gfx942",
            Self::Gfx950 => "gfx950",
            Self::Gfx1250 => "gfx1250",
        }
    }

    pub const fn wave_size(self) -> u32 {
        match self {
            Self::Gfx942 | Self::Gfx950 => 64,
            Self::Gfx1250 => 32,
        }
    }

    pub const fn full_exec_mask(self) -> u64 {
        u64::MAX >> (64 - self.wave_size())
    }

    pub const fn lds_capacity(self) -> u64 {
        match self {
            Self::Gfx942 => 65_536,
            Self::Gfx950 => 163_840,
            // LLVM AMDGPU.td FeatureISAVersion12_50 at the public reference
            // 5fd50ec8e07abb94e203933017e8d6e40fb51824 selects 320 KiB LDS.
            Self::Gfx1250 => 327_680,
        }
    }

    /// This model's selected MFMA forms, not an exhaustive ISA inventory.
    /// CDNA4's doubled-K and f8f6f4/scale forms must not become CDNA3
    /// instructions merely by changing a graph's target string.
    pub fn supports_matrix(self, kind: AmdMatrixKind) -> bool {
        self == Self::Gfx950
            || self == Self::Gfx942
                && matches!(
                    kind,
                    AmdMatrixKind::F32_16x16x4
                        | AmdMatrixKind::F32_32x32x2
                        | AmdMatrixKind::F16_16x16x16
                        | AmdMatrixKind::F16_32x32x8
                        | AmdMatrixKind::Bf16_16x16x16
                        | AmdMatrixKind::Bf16_32x32x8
                )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AmdCounter {
    Vm,
    Lgkm,
    Exp,
    Load,
    Store,
    Ds,
    Km,
    /// Address translation/replay sources, independent of data completion.
    X,
    Async,
    Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdMemoryClass {
    VectorMemory,
    VectorStoreWide,
    ScalarMemory,
    Lds,
    Flat,
    FlatStoreWide,
    Message,
    CrossLane,
    VectorLoad,
    VectorStore,
    ScalarLoad,
    Ds,
    AsyncLoadToLds,
    AsyncStoreFromLds,
    TensorLoadToLds,
    TensorStoreFromLds,
    /// Selected gfx1250 F32 add RMW: device scope, one active lane, no return.
    GlobalAtomicAddF32NoReturn,
    /// Genuine gfx1250 FLAT accesses with a resolved segment. Separate domains
    /// prevent completion/order frontiers from promoting an unrelated aperture.
    FlatGlobalLoad,
    FlatGlobalStore,
    FlatLdsLoad,
    FlatLdsStore,
    /// Selected gfx950 MUBUF raw DWORD -> LDS DMA, with no register destination.
    BufferLoadToLds32,
    /// Selected gfx950 MUBUF raw DWORDX4 -> LDS DMA, with no register destination.
    BufferLoadToLds128,
    /// Selected gfx950 scalar-offset DWORD scratch load; private VM0 profile.
    ScratchLoad32,
    /// Selected gfx950 scalar-offset DWORD scratch store; private VM0 profile.
    ScratchStore32,
    /// Selected gfx950 contiguous private DWORDX2/3/4 tuple.
    ScratchLoadWide,
    ScratchStoreWide,
    /// Opaque-private-aperture FLAT; both VM and LGKM completion are required.
    FlatPrivateLoad,
    FlatPrivateStore,
    /// gfx1250 GLOBAL prefetch: X-count address-source lease, no data,
    /// no destination and no load completion.
    Prefetch,
}

/// Encoded S_DELAY_ALU dependency selectors. These refer to prior producer
/// instructions, not a count of independent NOP cycles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdAluDependency {
    None,
    Valu(u8),
    Trans(u8),
    SaluCycles(u8),
}

impl AmdAluDependency {
    pub fn from_encoding(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::None),
            1..=4 => Some(Self::Valu(value)),
            5..=7 => Some(Self::Trans(value - 4)),
            9..=11 => Some(Self::SaluCycles(value - 8)),
            _ => None,
        }
    }
    pub fn encoding(self) -> Option<u8> {
        match self {
            Self::None => Some(0),
            Self::Valu(value @ 1..=4) => Some(value),
            Self::Trans(value @ 1..=3) => Some(value + 4),
            Self::SaluCycles(value @ 1..=3) => Some(value + 8),
            _ => None,
        }
    }
}

/// gfx1250 runtime MODE instruction recorded by the B0 m2 profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdModeOp {
    /// S_GETREG_B32 of MODE: reads the selected field into an SGPR.
    Getreg,
    /// S_SETREG_B32 to MODE from an SGPR.
    Setreg,
    /// S_SETREG_IMM32_B32 to MODE from a literal.
    SetregImm32,
    /// S_ROUND_MODE: writes MODE[3:0].
    RoundMode,
    /// S_DENORM_MODE: writes MODE[7:4].
    DenormMode,
}

impl AmdModeOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Getreg => "s_getreg_b32",
            Self::Setreg => "s_setreg_b32",
            Self::SetregImm32 => "s_setreg_imm32_b32",
            Self::RoundMode => "s_round_mode",
            Self::DenormMode => "s_denorm_mode",
        }
    }
    pub const fn writes(self) -> bool {
        !matches!(self, Self::Getreg)
    }
}

/// Encoded gfx12 S_WAIT_ALU (DEPCTR) operand. Every field names an ALU or
/// VMEM-source dependency counter; none names a memory completion counter.
/// HOLD_CNT is retained raw because its polarity is not established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AmdAluWait {
    pub encoded: u16,
}

impl AmdAluWait {
    /// The assembler's encoding when a depctr list leaves a field unspecified.
    pub const NO_WAIT: u16 = 0xff9f;
    pub const fn sa_sdst(self) -> u8 {
        (self.encoded & 1) as u8
    }
    pub const fn va_vcc(self) -> u8 {
        ((self.encoded >> 1) & 1) as u8
    }
    pub const fn vm_vsrc(self) -> u8 {
        ((self.encoded >> 2) & 7) as u8
    }
    /// Bits 6:5, which name no field.
    pub const fn reserved(self) -> u8 {
        ((self.encoded >> 5) & 3) as u8
    }
    pub const fn hold_cnt(self) -> u8 {
        ((self.encoded >> 7) & 1) as u8
    }
    pub const fn va_ssrc(self) -> u8 {
        ((self.encoded >> 8) & 1) as u8
    }
    pub const fn va_sdst(self) -> u8 {
        ((self.encoded >> 9) & 7) as u8
    }
    pub const fn va_vdst(self) -> u8 {
        ((self.encoded >> 12) & 15) as u8
    }
}

/// A configured global byte range. Objects may be accessed by their own
/// instructions; guards are forbidden even inside the backing allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdGlobalRegion {
    pub name: String,
    pub base: u64,
    pub size: u64,
    pub guard: bool,
}

/// gfx12 cache scope of a GLOBAL prefetch. Scopes wider than the work-group
/// (SE, device, system) with default temporal hint are speculative: a failed
/// translation is dropped rather than faulting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdCacheScope {
    Cu,
    Se,
    Device,
    System,
}

impl AmdCacheScope {
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "SCOPE_CU" => Self::Cu,
            "SCOPE_SE" => Self::Se,
            "SCOPE_DEV" => Self::Device,
            "SCOPE_SYS" => Self::System,
            _ => return None,
        })
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cu => "SCOPE_CU",
            Self::Se => "SCOPE_SE",
            Self::Device => "SCOPE_DEV",
            Self::System => "SCOPE_SYS",
        }
    }
    pub const fn speculative_prefetch(self) -> bool {
        !matches!(self, Self::Cu)
    }
}

/// The access kind of an attempted global footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdGlobalAccess {
    Load,
    Store,
    Prefetch(AmdCacheScope),
}

/// Abort reason of a global access whose footprint leaves its configured
/// objects or touches a guard; only an independently validated certificate
/// turns this abort into a rejection.
pub const GLOBAL_BOUNDS_ABORT_REASON: &str =
    "global access footprint leaves configured objects or touches a guard region";

/// Selected gfx1250 wave32 WMMA forms. This is distinct from legacy MFMA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum AmdWmmaKind {
    F32_16x16x4,
    F16_16x16x32,
    Bf16_16x16x32,
    Fp8Fp8_16x16x64,
    Fp8Bf8_16x16x64,
    Bf8Fp8_16x16x64,
    Bf8Bf8_16x16x64,
    /// FP8 (E4M3) A and B, K = 128: 16-register A and B tuples.
    Fp8Fp8_16x16x128,
    /// Integer 8-bit A and B (signedness from NEG), I32 C and D.
    Iu8_16x16x64,
    /// FP8/BF8 mixes, K = 128: 16-register A and B tuples.
    Fp8Bf8_16x16x128,
    Bf8Fp8_16x16x128,
    Bf8Bf8_16x16x128,
    /// `v_wmma_[scale[16]_]f32_16x16x128_f8f6f4`: A and B formats from the
    /// format modifiers (scale operands, if any, are recorded separately).
    F8f6f4_16x16x128 {
        a: AmdSmallFloat,
        b: AmdSmallFloat,
    },
    /// `v_wmma_[scale[16]_]f32_32x16x128_f4`: FP4 32x128 A (16 registers),
    /// FP4 128x16 B (8 registers), 32x16 F32 C and D (16 registers).
    F4_32x16x128,
    /// 16-bit-output dense WMMA, K = 32: F16 or BF16 inputs, 4-register
    /// packed D; `Bf16F32` reads an 8-register F32 C.
    Half16x16x32 {
        output: AmdWmmaOutput,
    },
    /// F16-output FP8/BF8 dense WMMA, K = 64 (8-register A/B) or 128 (16).
    F16Out8 {
        a: AmdSmallFloat,
        b: AmdSmallFloat,
        k128: bool,
    },
    /// SWMMAC: 4:2-sparse packed A (8 registers), B (16 registers), an
    /// index operand (recorded separately), C read from D.
    Swmmac {
        input: AmdSwmmacInput,
        output: AmdWmmaOutput,
    },
}

/// Result formats of the gfx1250 16-bit-output WMMA and SWMMAC forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdWmmaOutput {
    F32,
    F16,
    Bf16,
    /// F32 C, BF16 result.
    Bf16F32,
    I32,
}

impl AmdWmmaOutput {
    /// Registers of a 16x16 C/D tile of this format in wave32.
    pub const fn registers(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 4,
            _ => 8,
        }
    }
}

/// Input formats of the gfx1250 SWMMAC forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdSwmmacInput {
    /// K = 64.
    F16,
    /// K = 64.
    Bf16,
    /// K = 128.
    Small { a: AmdSmallFloat, b: AmdSmallFloat },
    /// K = 128.
    Iu8,
}

/// Element formats of the gfx1250 F8F6F4 WMMA forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdSmallFloat {
    /// E4M3
    Fp8,
    /// E5M2
    Bf8,
    /// E2M3
    Fp6,
    /// E3M2
    Bf6,
    /// E2M1
    Fp4,
}

impl AmdSmallFloat {
    /// Registers of a 16x128 (or 128x16) matrix of this format in wave32.
    pub const fn registers(self) -> usize {
        match self {
            Self::Fp8 | Self::Bf8 => 16,
            Self::Fp6 | Self::Bf6 => 12,
            Self::Fp4 => 8,
        }
    }
}

impl AmdWmmaKind {
    pub const fn destination_registers(self) -> usize {
        match self {
            Self::F4_32x16x128 => 16,
            Self::Half16x16x32 { .. } | Self::F16Out8 { .. } => 4,
            // The SWMMAC D (also its C) has the C format: BF16F32 keeps 8.
            Self::Swmmac { output, .. } => output.registers(),
            _ => 8,
        }
    }
    /// Registers of the C tuple (BF16F32 dense reads an 8-register F32 C).
    pub const fn accumulator_registers(self) -> usize {
        match self {
            Self::Half16x16x32 {
                output: AmdWmmaOutput::Bf16F32,
            } => 8,
            _ => self.destination_registers(),
        }
    }
    /// Registers of each of A and B for the forms whose A and B have the same
    /// width (every form except F8F6F4 with mixed widths and F4 32x16x128).
    pub const fn input_registers(self) -> usize {
        match self {
            Self::F32_16x16x4 => 2,
            Self::Fp8Fp8_16x16x128
            | Self::Fp8Bf8_16x16x128
            | Self::Bf8Fp8_16x16x128
            | Self::Bf8Bf8_16x16x128 => 16,
            Self::F8f6f4_16x16x128 { a, .. } => a.registers(),
            Self::F4_32x16x128 => 16,
            Self::F16Out8 { k128: true, .. } => 16,
            _ => 8,
        }
    }
    /// Registers of the A tuple.
    pub const fn a_registers(self) -> usize {
        self.input_registers()
    }
    /// Registers of the B tuple.
    pub const fn b_registers(self) -> usize {
        match self {
            Self::F8f6f4_16x16x128 { b, .. } => b.registers(),
            Self::F4_32x16x128 => 8,
            Self::Swmmac { .. } => 16,
            _ => self.input_registers(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdRegisterFile {
    Scalar,
    Vector,
    Accumulator,
    /// Architected trap temporaries used by the gfx1250 workgroup ABI.
    Temporary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)] // Match the ISA's matrix dimensions and data-type names.
pub enum AmdMatrixKind {
    F32_16x16x4,
    F32_32x32x2,
    F16_16x16x16,
    F16_32x32x8,
    F16_16x16x32,
    F16_32x32x16,
    Bf16_16x16x16,
    Bf16_32x32x8,
    Bf16_16x16x32,
    Bf16_32x32x16,
    Fp8_16x16x128,
    Fp8_32x32x64,
    Fp8Scale_16x16x128,
    Fp8Scale_32x32x64,
}

impl AmdMatrixKind {
    pub fn destination_registers(self) -> usize {
        match self {
            Self::F32_32x32x2
            | Self::F16_32x32x8
            | Self::F16_32x32x16
            | Self::Bf16_32x32x8
            | Self::Bf16_32x32x16
            | Self::Fp8_32x32x64
            | Self::Fp8Scale_32x32x64 => 16,
            _ => 4,
        }
    }
    pub fn input_registers(self) -> usize {
        match self {
            Self::F32_16x16x4 | Self::F32_32x32x2 => 1,
            Self::F16_16x16x16 | Self::F16_32x32x8 | Self::Bf16_16x16x16 | Self::Bf16_32x32x8 => 2,
            Self::F16_16x16x32 | Self::F16_32x32x16 | Self::Bf16_16x16x32 | Self::Bf16_32x32x16 => {
                4
            }
            _ => 8,
        }
    }
    pub fn scale_registers(self) -> usize {
        if matches!(self, Self::Fp8Scale_16x16x128 | Self::Fp8Scale_32x32x64) {
            2
        } else {
            0
        }
    }
    pub fn passes(self) -> u64 {
        match self {
            Self::F32_32x32x2 | Self::Fp8_32x32x64 | Self::Fp8Scale_32x32x64 => 16,
            Self::F16_16x16x16 | Self::F16_16x16x32 | Self::Bf16_16x16x16 | Self::Bf16_16x16x32 => {
                4
            }
            _ => 8,
        }
    }
    pub fn is_xdl(self) -> bool {
        !matches!(self, Self::F32_16x16x4 | Self::F32_32x32x2)
    }
}

/// One 32-bit register, restricted to a lane mask for vector/accumulator registers.
/// A register pair/range is represented by one entry per register. Scalar masks are ignored.
/// VCC low/high use scalar106/107; EXEC low/high use scalar126/127.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AmdRegister {
    pub file: AmdRegisterFile,
    pub index: u16,
    pub lanes: u64,
}

impl AmdRegister {
    pub fn is_uniform(self) -> bool {
        matches!(
            self.file,
            AmdRegisterFile::Scalar | AmdRegisterFile::Temporary
        )
    }
    pub fn overlaps(self, other: Self) -> bool {
        self.file == other.file
            && self.index == other.index
            && (self.is_uniform() || self.lanes & other.lanes != 0)
    }
}

/// A physical named object and its initialization lifetime. Reinitialization
/// creates a new incarnation even when the hardware ID is reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AmdNamedBarrier {
    pub id: u8,
    pub incarnation: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AmdEventKind {
    WmmaIssue {
        kind: AmdWmmaKind,
        destinations: Vec<AmdRegister>,
        sources_ab: Vec<AmdRegister>,
        accumulator: Vec<AmdRegister>,
        /// Block-scale operand registers of the 4-dword scaled forms (A
        /// scale, then B scale); empty for every other form.
        scales: Vec<AmdRegister>,
        /// SWMMAC sparsity-index operand registers; empty for dense forms.
        index: Vec<AmdRegister>,
    },
    /// A supported MFMA form, with operand roles retained for forwarding checks.
    MatrixIssue {
        kind: AmdMatrixKind,
        destinations: Vec<AmdRegister>,
        sources_ab: Vec<AmdRegister>,
        accumulator: Vec<AmdRegister>,
    },
    /// S_NOP's encoded immediate plus one (1..16 independent wait states).
    Delay {
        cycles: u16,
    },
    /// `instid0` applies to the next instruction; `instid1`, if present,
    /// applies after `instskip` intervening machine instructions (0..5).
    DelayAlu {
        instid0: AmdAluDependency,
        instskip: u8,
        instid1: Option<AmdAluDependency>,
    },
    /// Every executed instruction, including scalar instructions with EXEC=0.
    Instruction {
        exec_mask: u64,
        reads: Vec<AmdRegister>,
        writes: Vec<AmdRegister>,
        /// Scalar reads serving only as carry-in/condition masks, rather than
        /// ordinary scalar data. This is a validated subset of `reads`.
        scalar_mask_reads: Vec<AmdRegister>,
    },
    /// One issued instruction, not one lane. SMEM charges 1 for dword and 2 for wider
    /// transfers. FLAT charges both VM and LGKM. Destinations remain unavailable until
    /// completion. Wide-store held sources use Table 11's independent-state rules;
    /// other explicitly held sources remain held until operation completion.
    MemoryIssue {
        class: AmdMemoryClass,
        counters: Vec<(AmdCounter, u16)>,
        destinations: Vec<AmdRegister>,
        held_sources: Vec<AmdRegister>,
    },
    Wait {
        vmcnt: Option<u16>,
        lgkmcnt: Option<u16>,
        expcnt: Option<u16>,
    },
    /// GFX12.5 completion and replay thresholds. Counter identities are never
    /// aliases for the legacy VM/LGKM fields. Empty/duplicate waits are invalid.
    WaitCounters {
        thresholds: Vec<(AmdCounter, u16)>,
    },
    /// Nonblocking signal of a specific GFX12 barrier object and phase.
    SplitBarrierSignal {
        barrier: i16,
        generation: u32,
        exec_mask: u64,
    },
    /// The wave is blocked until a matching SplitBarrierRelease is recorded.
    SplitBarrierWait {
        barrier: i16,
        generation: u32,
    },
    SplitBarrierRelease {
        barrier: i16,
        generation: u32,
    },
    NamedBarrierInit {
        object: AmdNamedBarrier,
        expected: u16,
    },
    NamedBarrierJoin {
        object: AmdNamedBarrier,
        phase: u32,
    },
    NamedBarrierSignal {
        object: AmdNamedBarrier,
        phase: u32,
        exec_mask: u64,
    },
    /// The encoded ID is preserved. For named waits the hardware actually uses
    /// the last joined object, which may differ from that encoded operand.
    NamedBarrierWait {
        encoded: u8,
        object: AmdNamedBarrier,
        phase: u32,
    },
    NamedBarrierRelease {
        object: AmdNamedBarrier,
        phase: u32,
    },
    NamedBarrierLeave {
        object: AmdNamedBarrier,
        phase: u32,
    },
    /// One arrival per live wave. The mask is recorded for diagnostics, not a requirement
    /// that EXEC be all ones: S_BARRIER is a scalar instruction.
    BarrierArrive {
        generation: u32,
        exec_mask: u64,
    },
    BarrierRelease {
        generation: u32,
    },
    /// A named signal with an explicit nonzero M0[22:16] expected-count update.
    /// Appended so existing event variants and their layouts remain unchanged.
    NamedBarrierSignalWithCount {
        object: AmdNamedBarrier,
        phase: u32,
        exec_mask: u64,
        expected: u16,
    },
    /// Attempted supported tensor transfer, aborted before issuing any effects.
    /// Appended to preserve the layouts/tags of existing records.
    TensorAccessFault {
        load: bool,
        descriptor: AmdTensorDescriptor,
        /// Starting scalar register of each D0/D1/D2/D3 tuple.
        sources: [u16; 4],
    },
    /// One successful contiguous word-tuple access to this work item's private allocation.
    /// Selected widths are 4/8/12/16 bytes; a tuple emits one record per active lane.
    /// This is not a cross-thread Global/Shared effect. Address decoding and exact
    /// word transport remain producer premises; the checker validates membership,
    /// bounds, operand roles and the explicit private completion profile.
    PrivateMemoryAccess {
        issue: EventId,
        thread: ThreadIx,
        offset: u64,
        width: u32,
        kind: AccessMode,
    },
    /// A selected gfx950 literal-lane read, independent of EXEC. Paired with the
    /// preceding same-wave instruction whose provenance kind is `readlane`.
    LaneRead {
        lane: u8,
    },
    /// A selected gfx950 literal-lane write, independent of EXEC. Paired with the
    /// preceding same-wave instruction whose provenance kind is `writelane`.
    LaneWrite {
        lane: u8,
    },
    /// gfx12 S_WAIT_ALU. It orders only ALU dependency counters and never
    /// completes a memory, kernarg or replay counter.
    AluWait {
        wait: AmdAluWait,
    },
    /// Per-lane address footprint of an issued prefetch (one byte per active
    /// lane). It follows its `MemoryIssue`, carries no data and has no effects.
    PrefetchFootprint {
        issue: EventId,
        scope: AmdCacheScope,
        lanes: Vec<(u8, u64)>,
    },
    /// Attempted global access recorded before issue, with every active lane's
    /// start address. Execution aborts; nothing is issued or written.
    GlobalAccessFault {
        access: AmdGlobalAccess,
        bytes_per_lane: u8,
        lanes: Vec<(u8, u64)>,
    },
    /// One executed MODE read or write of bits `[offset, offset + width)`,
    /// with the exact field value read or written. It follows the
    /// instruction's `Instruction` record and has no register effects of its
    /// own.
    ModeAccess {
        op: AmdModeOp,
        offset: u8,
        width: u8,
        value: u32,
    },
    /// B0 litmus profile: the DWORD range check of one issued untyped raw
    /// buffer access, after the issue's effects. In-range DWORDs have ordinary
    /// effects; out-of-range DWORDs have none (loads read zero).
    BufferRangeCheck {
        issue: EventId,
        check: AmdBufferRangeCheck,
    },
}

/// Descriptor fields and per-lane results of one buffer range check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdBufferRangeCheck {
    pub base: u64,
    pub num_records: u64,
    /// Stride in bytes after scaling (the profile accepts scale 1 only).
    pub stride: u64,
    pub oob_select: bool,
    pub dwords: u8,
    pub lanes: Vec<AmdBufferLane>,
}

/// One active lane: byte `offset` (SGPR + immediate + VGPR, no index) and the
/// in-range DWORD mask, bit `c` for bytes `[offset + 4c, offset + 4c + 4)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmdBufferLane {
    pub lane: u8,
    pub offset: u64,
    pub in_range: u8,
}

impl AmdBufferRangeCheck {
    /// The documented per-DWORD rule: DWORD `c` is out of range when
    /// `offset + 4c + 4 > num_records`, or, with OOB_SELECT, `> stride`.
    pub fn expected_in_range(&self, offset: u64) -> u8 {
        (0..self.dwords).fold(0, |mask, c| {
            let end = offset.saturating_add(4 * u64::from(c) + 4);
            let inside = end <= self.num_records && (!self.oob_select || end <= self.stride);
            mask | (u8::from(inside) << c)
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdEvent {
    pub event: EventId,
    pub wave: u32,
    pub kind: AmdEventKind,
}

/// Native visibility policy of an ordinary AMD memory effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmdMemoryScope {
    #[default]
    Unspecified,
    /// Native GLOBAL SC0=1, SC1=0. Visibility additionally requires a recorded
    /// single-CU workgroup premise; scope does not create memory ordering.
    Workgroup,
    /// Explicit gfx1250 SCOPE_SYS on ordinary GLOBAL or resolved-global FLAT
    /// accesses. Both endpoints must carry it; completion and ordering remain
    /// separate requirements within the analyzed workgroup.
    Gfx1250System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Associates an ordinary memory event with its issuing AMD instruction. Its
/// lane attribution and this association retain asynchronous completion.
pub struct AmdMemoryEffect {
    pub issue: EventId,
    pub effect: EventId,
    pub scope: AmdMemoryScope,
}

pub fn amd_wave_thread(wave: u32) -> ThreadIx {
    ThreadIx(wave * 64)
}

/// First physical thread of a wave for an explicitly selected native target.
/// The original `amd_wave_thread` retains its wave64 meaning for old callers.
pub fn amd_wave_thread_for(architecture: AmdArchitecture, wave: u32) -> ThreadIx {
    ThreadIx(wave.saturating_mul(architecture.wave_size()))
}

#[cfg(test)]
mod architecture_tests {
    use super::*;

    #[test]
    fn delay_alu_dependency_encoding_is_typed_and_rejects_reserved_selectors() {
        for encoding in 0..=15 {
            let dependency = AmdAluDependency::from_encoding(encoding);
            assert_eq!(dependency.is_some(), matches!(encoding, 0..=7 | 9..=11));
            if let Some(dependency) = dependency {
                assert_eq!(dependency.encoding(), Some(encoding));
            }
        }
        for dependency in [
            AmdAluDependency::Valu(0),
            AmdAluDependency::Valu(5),
            AmdAluDependency::Trans(4),
            AmdAluDependency::SaluCycles(4),
        ] {
            assert_eq!(dependency.encoding(), None);
        }
        let temporary = AmdRegister {
            file: AmdRegisterFile::Temporary,
            index: 9,
            lanes: 0,
        };
        assert!(temporary.overlaps(AmdRegister {
            lanes: u64::MAX,
            ..temporary
        }));
        assert!(!temporary.overlaps(AmdRegister {
            file: AmdRegisterFile::Scalar,
            ..temporary
        }));
    }

    #[test]
    fn targets_retain_architecture_and_reject_conflicting_or_unknown_features() {
        for (target, expected) in [
            ("gfx942", Some(AmdArchitecture::Gfx942)),
            ("gfx942:sramecc+:xnack-", Some(AmdArchitecture::Gfx942)),
            ("gfx950:xnack+:sramecc-", Some(AmdArchitecture::Gfx950)),
            ("gfx1250", Some(AmdArchitecture::Gfx1250)),
            ("gfx1250:xnack-", None),
            ("gfx1250:xnack+", None),
            ("gfx1250-strict", None),
            ("gfx1251", None),
            ("gfx942:xnack+:xnack-", None),
            ("gfx942:sramecc+:sramecc+", None),
            ("gfx942:wave32", None),
            ("gfx942:", None),
            ("gfx941", None),
            ("sm_100a", None),
        ] {
            assert_eq!(AmdArchitecture::from_target(target), expected, "{target}");
        }
        assert_eq!(AmdArchitecture::from_targets(&[]), None);
        assert_eq!(
            AmdArchitecture::from_targets(&["gfx942".into(), "gfx950".into()]),
            None
        );
        assert_eq!(AmdArchitecture::Gfx942.lds_capacity(), 65_536);
        assert_eq!(AmdArchitecture::Gfx950.lds_capacity(), 163_840);
        assert_eq!(AmdArchitecture::Gfx1250.lds_capacity(), 327_680);
        for architecture in [
            AmdArchitecture::Gfx942,
            AmdArchitecture::Gfx950,
            AmdArchitecture::Gfx1250,
        ] {
            let size = architecture.wave_size();
            assert_eq!(
                size,
                if architecture == AmdArchitecture::Gfx1250 {
                    32
                } else {
                    64
                }
            );
            assert_eq!(amd_wave_thread_for(architecture, 3), ThreadIx(3 * size));
            assert_eq!(architecture.full_exec_mask().count_ones(), size);
            assert!(!AmdArchitecture::Gfx1250.supports_matrix(AmdMatrixKind::F32_16x16x4));
        }
        assert!(AmdArchitecture::Gfx942.supports_matrix(AmdMatrixKind::F16_16x16x16));
        for kind in [
            AmdMatrixKind::F16_16x16x32,
            AmdMatrixKind::Bf16_32x32x16,
            AmdMatrixKind::Fp8_16x16x128,
            AmdMatrixKind::Fp8Scale_32x32x64,
        ] {
            assert!(!AmdArchitecture::Gfx942.supports_matrix(kind));
            assert!(AmdArchitecture::Gfx950.supports_matrix(kind));
        }
    }
}

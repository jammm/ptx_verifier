//! Analysis configuration: launch dimensions, parameter values, and the
//! input/output arrays that define the kernel's memory interface.

use crate::tensor_map::{
    Im2colBox, TensorElemType, TensorFillMode, TensorInterleaveLayout, TensorSwizzleAtomicity,
    TensorSwizzleMode,
};

/// Value bound to a kernel parameter, positional by `ParamId` order.
#[derive(Debug, Clone)]
pub enum ParamValue {
    /// Concrete integer (also used for raw pointer values)
    Int(i64),
    /// Concrete float
    Float(f64),
    /// Symbolic float input with the given name (e.g. "alpha")
    SymFloat(String),
    /// Pointer to a named array from `AnalysisConfig::arrays`
    ArrayPtr(String),
    /// Exact native by-value ABI bytes, including caller-supplied padding.
    /// PTX does not accept this parameter form.
    Bytes(Vec<u8>),
    /// PTX by-value aggregate fields; uncovered bytes remain undefined.
    Struct(Vec<StructField>),
}

/// One field of a struct parameter, at a byte offset inside it.
#[derive(Debug, Clone)]
pub struct StructField {
    pub offset: u64,
    pub value: FieldValue,
}

#[derive(Debug, Clone)]
pub enum FieldValue {
    /// A little-endian integer; `value` holds its two's-complement bits, so a
    /// signed and an unsigned field of one width store alike
    Int { value: i64, width: IntWidth },
    /// The 8-byte base address of a named array from `AnalysisConfig::arrays`
    ArrayPtr(String),
    /// A 128-byte `CUtensorMap` (a TMA descriptor the host encoded), given
    /// by its fields since its bytes are opaque (PTX ISA 5.5.8)
    TensorMap(Box<TensorMapValue>),
}

/// The fields of a tensor map, as `tensormap.replace` names them (PTX ISA 9.7.9.27).
#[derive(Debug, Clone)]
pub struct TensorMapValue {
    /// The array whose base is the map's global address
    pub array: String,
    pub elemtype: TensorElemType,
    /// Each dimension's size in elements, innermost first
    pub dims: Vec<u64>,
    /// The byte stride of each dimension but the innermost
    pub strides: Vec<u64>,
    pub mode: TensorMapMode,
    pub element_strides: Vec<u32>,
    pub interleave: TensorInterleaveLayout,
    pub swizzle: TensorSwizzleMode,
    pub swizzle_atomicity: TensorSwizzleAtomicity,
    pub fill: TensorFillMode,
}

/// What one copy through a tensor map moves, fixed by the host function that encoded it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorMapMode {
    /// `cuTensorMapEncodeTiled`: a box of `box_dims` elements, innermost first
    Tiled { box_dims: Vec<u32> },
    /// `cuTensorMapEncodeIm2col`
    Im2col(Im2colBox),
}

/// The size of a tensor-map object (`.b1024`).
pub const TENSOR_MAP_BYTES: u64 = 128;

impl FieldValue {
    pub fn width_bytes(&self) -> u64 {
        match self {
            FieldValue::Int { width, .. } => width.bytes(),
            FieldValue::ArrayPtr(_) => 8,
            FieldValue::TensorMap(_) => TENSOR_MAP_BYTES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntWidth {
    W1,
    W2,
    W4,
    W8,
}

impl IntWidth {
    pub fn bytes(self) -> u64 {
        match self {
            IntWidth::W1 => 1,
            IntWidth::W2 => 2,
            IntWidth::W4 => 4,
            IntWidth::W8 => 8,
        }
    }

    /// Whether `value` is the bits of a signed or an unsigned integer of this width.
    pub fn holds(self, value: i64) -> bool {
        let bits = 8 * self.bytes();
        bits == 64 || (-(1i64 << (bits - 1))..(1i64 << bits)).contains(&value)
    }
}

/// Whether an array is a kernel input (pre-initialized with fresh symbols),
/// an output (extracted after execution), or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayKind {
    Input,
    Output,
    InputOutput,
    /// An input of concrete integers where element i holds i, for index arrays
    /// feeding addresses (OpenMM's `posq[particles[i]]`): symbolic elements
    /// would make the addresses symbolic and violate structured-CTA.
    IndexInput,
}

impl ArrayKind {
    pub fn is_input(self) -> bool {
        matches!(self, Self::Input | Self::InputOutput | Self::IndexInput)
    }

    pub fn is_output(self) -> bool {
        matches!(self, Self::Output | Self::InputOutput)
    }
}

/// A global-memory array visible to the kernel. Inputs read as `name[i]`
/// symbols at `elem_width`. Must not overlap the module-global window
/// (`ir::symbols::MODULE_GLOBAL_BASE`); `Interpreter::new` rejects that.
#[derive(Debug, Clone)]
pub struct ArrayDef {
    pub name: String,
    pub base: u64,
    /// Element width in bytes (2 for f16, 4 for f32/int, 8 for f64)
    pub elem_width: u64,
    /// Number of elements
    pub len: u64,
    pub kind: ArrayKind,
}

impl ArrayDef {
    pub fn size_bytes(&self) -> u64 {
        self.elem_width * self.len
    }
}

/// Optional source-allocation boundaries supplied for native assembly whose
/// descriptor exposes only aggregate workgroup LDS bytes. These are premises
/// of the analysis, not names or boundaries inferred from instruction offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedAllocation {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

pub fn validate_shared_allocations(layout: &[SharedAllocation]) -> Result<(), String> {
    for (index, region) in layout.iter().enumerate() {
        if region.name.trim().is_empty() {
            return Err("shared allocation name is empty".into());
        }
        if region.size == 0 {
            return Err(format!("shared allocation '{}' has zero size", region.name));
        }
        let end = region
            .offset
            .checked_add(region.size)
            .ok_or_else(|| format!("shared allocation '{}' range overflows", region.name))?;
        for previous in &layout[..index] {
            if previous.name == region.name {
                return Err(format!(
                    "duplicate shared allocation name '{}'",
                    region.name
                ));
            }
            // Earlier regions have already passed the checked-end validation.
            let previous_end = previous.offset + previous.size;
            if region.offset < previous_end && previous.offset < end {
                return Err(format!(
                    "shared allocations '{}' and '{}' overlap",
                    previous.name, region.name
                ));
            }
        }
    }
    Ok(())
}

/// Native execution profile. The default keeps each target's existing contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NativeProfile {
    #[default]
    TargetDefault,
    /// gfx1250 B0 m1 integer profile `gfx1250-b0-m1-int/1` with
    /// dependency profile `gfx1250-b0-sched0-m1/1`: strict-greater S_MAX SCC,
    /// pre-issue global bounds certificates against declared objects and
    /// guards. It never changes gfx942/gfx950 or the default gfx1250 contract.
    Gfx1250B0M1Int,
    /// gfx1250 B0 m2 profile `gfx1250-b0-m2-mode/1`: everything of
    /// `Gfx1250B0M1Int` plus runtime MODE state for the FP_ROUND and
    /// FP_DENORM fields (S_GETREG/S_SETREG/S_SETREG_IMM32 of those fields,
    /// S_ROUND_MODE, S_DENORM_MODE) with recorded MODE access events.
    Gfx1250B0M2Mode,
    /// gfx1250 B0 litmus profile `gfx1250-b0-litmus/1`: the m1 bounds,
    /// guard and raw-word machinery without the integer-subset semantics, plus
    /// untyped raw-buffer DWORD range checking, scalar-base async loads to LDS
    /// below a 64 KiB displacement and launcher-defined initial object words.
    Gfx1250B0Litmus,
    /// gfx1250 B0 litmus profile `gfx1250-b0-litmus/2`: everything of
    /// `Gfx1250B0Litmus` plus two-wave named-barrier JOIN read-back through
    /// STATE_PRIV, S_WAIT_IDLE and lane-ordered same-address global stores.
    Gfx1250B0Litmus2,
    /// gfx1250 B0 WMMA profile `gfx1250-b0-wmma/1`: everything of
    /// `Gfx1250B0M2Mode` plus BF16 WMMA results reported as exact binary32
    /// bits only inside the exact-representable regime.
    Gfx1250B0Wmma,
    /// gfx1250 B0 multiwave profile `gfx1250-b0-mw/1`: the WMMA profile for
    /// several waves of the selected workgroup, opaque per-wave evidence reads
    /// (shader cycles, HW_ID1), single-group XNACK replay draining and
    /// lane-ordered same-address global stores.
    Gfx1250B0Multiwave,
    /// gfx1250 B0 WMMA profile `gfx1250-b0-wmma/2`: everything of
    /// `Gfx1250B0Wmma` plus F16, IU8 and FP8 (16x16x64, 16x16x128) WMMA with
    /// exact results in their regimes, per-type spacing, and the two fixed
    /// EXEC writes of an EXEC=0 spacer (EXEC=0 instructions are never counted
    /// as WMMA spacing slots).
    Gfx1250B0Wmma2,
    /// gfx1250 B0 WMMA profile `gfx1250-b0-wmma/3`: everything of
    /// `Gfx1250B0Wmma2` plus the BF8 combinations, F8F6F4 and F4 32x16x128
    /// WMMA and their block-scaled 4-dword forms, evaluated for uniform tiles
    /// and uniform power-of-two scales only.
    Gfx1250B0Wmma3,
    /// gfx1250 B0 WMMA profile `gfx1250-b0-wmma/4`: everything of
    /// `Gfx1250B0Wmma3` plus the 16-bit-output WMMA forms and SWMMAC (F16
    /// results certified in an 11-bit regime, BF16 results never).
    Gfx1250B0Wmma4,
    /// gfx1250 B0 multiwave profile `gfx1250-b0-mw/2`: `Gfx1250B0Multiwave`
    /// (2..8 waves of the selected workgroup, opaque evidence reads,
    /// single-group replay draining, lane-ordered stores) with the WMMA forms
    /// and rules of `Gfx1250B0Wmma4`, applied per wave.
    Gfx1250B0Multiwave2,
    /// gfx1250 B0 multi-workgroup predecessor profile `gfx1250-b0-mwpred/1`:
    /// the multiwave behaviours with BF16 WMMA for every workgroup of a
    /// non-communicating grid, plus the opaque SE/XCC id message return.
    Gfx1250B0MwPred,
    /// gfx1250 B0 communicating-waves profile `gfx1250-b0-comms/1`: the
    /// multiwave wrapper behaviours, LDS and GLOBAL producer/consumer
    /// communication between the waves of each analysed workgroup ordered by
    /// completion waits and workgroup or named barriers.
    Gfx1250B0Comms,
}

impl NativeProfile {
    /// The B0 integer profiles: m1 and its m2 superset.
    pub const fn b0_integer(self) -> bool {
        matches!(
            self,
            Self::Gfx1250B0M1Int
                | Self::Gfx1250B0M2Mode
                | Self::Gfx1250B0Wmma
                | Self::Gfx1250B0Multiwave
                | Self::Gfx1250B0Wmma2
                | Self::Gfx1250B0Wmma3
                | Self::Gfx1250B0Wmma4
                | Self::Gfx1250B0Multiwave2
                | Self::Gfx1250B0MwPred
                | Self::Gfx1250B0Comms
        )
    }
    /// Profiles with runtime MODE FP_ROUND/FP_DENORM state (m2 and its
    /// supersets).
    pub const fn b0_mode_state(self) -> bool {
        matches!(
            self,
            Self::Gfx1250B0M2Mode
                | Self::Gfx1250B0Wmma
                | Self::Gfx1250B0Multiwave
                | Self::Gfx1250B0Wmma2
                | Self::Gfx1250B0Wmma3
                | Self::Gfx1250B0Wmma4
                | Self::Gfx1250B0Multiwave2
                | Self::Gfx1250B0MwPred
        )
    }
    /// Profiles that report BF16 WMMA results as exact bits in the
    /// exact-representable regime.
    pub const fn b0_wmma(self) -> bool {
        matches!(
            self,
            Self::Gfx1250B0Wmma
                | Self::Gfx1250B0Multiwave
                | Self::Gfx1250B0Wmma2
                | Self::Gfx1250B0Wmma3
                | Self::Gfx1250B0Wmma4
                | Self::Gfx1250B0Multiwave2
                | Self::Gfx1250B0MwPred
        )
    }
    /// The WMMA types beyond BF16 (F16, IU8, FP8) and the EXEC=0 spacer.
    pub const fn b0_wmma_types(self) -> bool {
        matches!(
            self,
            Self::Gfx1250B0Wmma2
                | Self::Gfx1250B0Wmma3
                | Self::Gfx1250B0Wmma4
                | Self::Gfx1250B0Multiwave2
        )
    }
    /// The BF8 combinations, F8F6F4/F4 and block-scaled WMMA (`wmma/3`).
    pub const fn b0_wmma_scale(self) -> bool {
        matches!(
            self,
            Self::Gfx1250B0Wmma3 | Self::Gfx1250B0Wmma4 | Self::Gfx1250B0Multiwave2
        )
    }
    /// The 16-bit-output WMMA forms and SWMMAC (`wmma/4`).
    pub const fn b0_wmma_sparse16(self) -> bool {
        matches!(self, Self::Gfx1250B0Wmma4 | Self::Gfx1250B0Multiwave2)
    }
    /// The multiwave profiles (`mw/1` and `mw/2`): several waves of the
    /// selected workgroup with the multiwave wrapper semantics.
    pub const fn b0_multiwave(self) -> bool {
        matches!(
            self,
            Self::Gfx1250B0Multiwave
                | Self::Gfx1250B0Multiwave2
                | Self::Gfx1250B0MwPred
                | Self::Gfx1250B0Comms
        )
    }
    /// The communicating-waves profile (`comms/1`).
    pub const fn b0_comms(self) -> bool {
        matches!(self, Self::Gfx1250B0Comms)
    }
    /// The multi-workgroup predecessor profile (`mwpred/1`): the SE/XCC id
    /// message return is an opaque evidence read.
    pub const fn b0_mwpred(self) -> bool {
        matches!(self, Self::Gfx1250B0MwPred)
    }
    /// The B0 litmus profiles: raw-buffer range checks, async loads to LDS
    /// and launcher-defined initial object words.
    pub const fn b0_litmus(self) -> bool {
        matches!(self, Self::Gfx1250B0Litmus | Self::Gfx1250B0Litmus2)
    }
    /// Every B0 profile with declared objects, guards and pre-issue global
    /// bounds certificates.
    pub const fn b0_bounded(self) -> bool {
        matches!(
            self,
            Self::Gfx1250B0M1Int
                | Self::Gfx1250B0M2Mode
                | Self::Gfx1250B0Litmus
                | Self::Gfx1250B0Litmus2
                | Self::Gfx1250B0Wmma
                | Self::Gfx1250B0Multiwave
                | Self::Gfx1250B0Wmma2
                | Self::Gfx1250B0Wmma3
                | Self::Gfx1250B0Wmma4
                | Self::Gfx1250B0Multiwave2
                | Self::Gfx1250B0MwPred
                | Self::Gfx1250B0Comms
        )
    }
}

/// Exact initial words of one configured array: the caller's launcher writes
/// them before the kernel runs. Word `i` is `value` (`Fill`) or
/// `value + i` modulo 2^32 (`Index`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialPattern {
    Fill(u32),
    Index(u32),
}

impl InitialPattern {
    pub const fn word(self, index: u64) -> u32 {
        match self {
            Self::Fill(value) => value,
            Self::Index(base) => base.wrapping_add(index as u32),
        }
    }
}

/// The initial words of a named four-byte array (B0 litmus profile only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitialWords {
    pub array: String,
    pub pattern: InitialPattern,
}

/// How output arrays are observed after a completed execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputContract {
    /// Mathematical expressions; raw untyped bits are refused.
    #[default]
    Mathematical,
    /// Every four-byte output word as its exact stored bits or a typed refusal.
    RawU32,
}

/// A forbidden global byte range: any access to it is a bounds violation even
/// though it may lie inside the caller's backing allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardRegion {
    pub name: String,
    pub base: u64,
    pub bytes: u64,
}

/// Full configuration for analyzing one kernel.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Threads per block or workgroup.
    pub block_dim: (u32, u32, u32),
    /// Dispatch grid dimensions, measured in blocks or workgroups.
    pub grid_dim: (u32, u32, u32),
    /// The single native AMD workgroup to analyze. PTX currently requires zero.
    pub workgroup_id: (u32, u32, u32),
    /// Opt in to the pipeline's first certified gfx1250 scalar-readiness finding.
    /// Default false preserves complete candidate traces and complete-trace diagnostics.
    pub early_readiness: bool,
    /// Opt in to the pipeline's first independently certified same-instruction
    /// native gfx950/gfx1250 write/write conflict. Never enables early acceptance.
    /// Default false preserves complete traces and complete-trace diagnostics.
    pub early_memory_conflicts: bool,
    /// Parameter values in `ParamId` (declaration) order
    pub params: Vec<ParamValue>,
    /// Global-memory arrays
    pub arrays: Vec<ArrayDef>,
    /// Concrete values for module-scope `.global` variables, by PTX name
    pub global_values: Vec<(String, i64)>,
    /// Size of dynamic (extern) shared memory in bytes
    pub dynamic_shared_bytes: u64,
    /// Optional native LDS object boundaries; empty means aggregate bounds only.
    /// The PTX analyzer rejects this field because PTX declares its own objects.
    pub shared_allocations: Vec<SharedAllocation>,
    /// Abort analysis after this many executed instructions
    pub max_instructions: u64,
    /// Native profile; anything other than the default requires gfx1250.
    pub native_profile: NativeProfile,
    /// Forbidden global regions; only the gfx1250 B0 m1 profile accepts them.
    pub guard_regions: Vec<GuardRegion>,
    pub output_contract: OutputContract,
    /// Exact initial array words; only the gfx1250 B0 litmus profile accepts them.
    pub initial_words: Vec<InitialWords>,
}

impl AnalysisConfig {
    pub fn new(block_dim: (u32, u32, u32)) -> Self {
        Self {
            block_dim,
            grid_dim: (1, 1, 1),
            workgroup_id: (0, 0, 0),
            early_readiness: false,
            early_memory_conflicts: false,
            params: Vec::new(),
            arrays: Vec::new(),
            global_values: Vec::new(),
            dynamic_shared_bytes: 0,
            shared_allocations: Vec::new(),
            max_instructions: 2_000_000_000,
            native_profile: NativeProfile::TargetDefault,
            guard_regions: Vec::new(),
            output_contract: OutputContract::Mathematical,
            initial_words: Vec::new(),
        }
    }

    /// Total threads in a block. Call `validate()` first for untrusted dimensions.
    pub fn num_threads(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }

    pub fn array(&self, name: &str) -> Option<&ArrayDef> {
        self.arrays.iter().find(|a| a.name == name)
    }

    /// Reject invalid launch geometry and ambiguous identities: duplicate array/`sym:`
    /// names or overlapping ranges would silently conflate distinct values.
    /// Overlap with the module-global window is checked in `Interpreter::new`.
    pub fn validate(&self) -> Result<(), String> {
        let threads = self
            .block_dim
            .0
            .checked_mul(self.block_dim.1)
            .and_then(|n| n.checked_mul(self.block_dim.2))
            .ok_or("block dimension overflow")?;
        if threads == 0 {
            return Err("block has zero threads".into());
        }
        if self.grid_dim.0 == 0 || self.grid_dim.1 == 0 || self.grid_dim.2 == 0 {
            return Err("grid dimensions must be nonzero".into());
        }
        for (axis, coordinate, groups) in [
            ("x", self.workgroup_id.0, self.grid_dim.0),
            ("y", self.workgroup_id.1, self.grid_dim.1),
            ("z", self.workgroup_id.2, self.grid_dim.2),
        ] {
            if coordinate >= groups {
                return Err(format!(
                    "workgroup_id.{axis} {coordinate} is outside grid dimension {groups}"
                ));
            }
        }
        validate_shared_allocations(&self.shared_allocations)?;
        // Per-array checks first, so the pairwise pass below may safely
        // form `base + size_bytes()` for every array.
        for a in &self.arrays {
            if a.elem_width == 0 || a.len == 0 {
                return Err(format!(
                    "array '{}' has zero element width or length",
                    a.name
                ));
            }
            // The byte range must fit in u64 (checked, release-active): every
            // later use of `[base, base + size_bytes())` relies on the end
            // being representable.
            let end = a
                .elem_width
                .checked_mul(a.len)
                .and_then(|size| a.base.checked_add(size));
            if end.is_none() {
                return Err(format!(
                    "array '{}' (base {:#x}, {} elements of {} bytes) overflows the \
                     address space",
                    a.name, a.base, a.len, a.elem_width
                ));
            }
            // PTX requires naturally aligned accesses (ISA 6.4.1); a base that
            // is not a multiple of the element width misaligns every element.
            if a.base % a.elem_width != 0 {
                return Err(format!(
                    "array '{}' base address {:#x} is not a multiple of its element \
                     width {}; every element access would be misaligned (PTX requires \
                     naturally aligned accesses)",
                    a.name, a.base, a.elem_width
                ));
            }
        }
        for (i, a) in self.arrays.iter().enumerate() {
            for b in &self.arrays[i + 1..] {
                if a.name == b.name {
                    return Err(format!(
                        "two arrays share the name '{}'; array names are identities \
                         and must be unique",
                        a.name
                    ));
                }
                let (a_end, b_end) = (a.base + a.size_bytes(), b.base + b.size_bytes());
                if a.base < b_end && b.base < a_end {
                    return Err(format!(
                        "arrays '{}' and '{}' overlap in memory \
                         ([{:#x}, {:#x}) vs [{:#x}, {:#x}))",
                        a.name, b.name, a.base, a_end, b.base, b_end
                    ));
                }
            }
        }
        self.validate_native_profile()?;
        let mut sym_names = std::collections::BTreeSet::new();
        for (i, value) in self.params.iter().enumerate() {
            if matches!(value, ParamValue::Bytes(bytes) if bytes.is_empty()) {
                return Err(format!("parameter {i} has an empty native byte payload"));
            }
            // NaN denotes no real number: it cannot enter the analysis
            // model (the infinities can - they are running-max/min seeds).
            if let ParamValue::Float(v) = value
                && v.is_nan()
            {
                return Err(format!(
                    "parameter {} is NaN; NaN is outside the analysis model (reals)",
                    i
                ));
            }
            if let ParamValue::Struct(fields) = value {
                validate_struct(i, fields)?;
            }
            if let ParamValue::SymFloat(name) = value {
                if name.is_empty() {
                    return Err("sym: parameter name is empty".to_string());
                }
                if name.contains('[') || name.contains(']') {
                    return Err(format!(
                        "sym: parameter name '{}' contains brackets; bracketed names \
                         are reserved for array elements",
                        name
                    ));
                }
                if !sym_names.insert(name) {
                    return Err(format!(
                        "two sym: parameters share the name '{}'; parameter names are \
                         identities and must be unique (a duplicate would silently \
                         constrain the check to inputs where both parameters are equal)",
                        name
                    ));
                }
            }
        }
        Ok(())
    }
}

impl AnalysisConfig {
    /// Guards and the raw-word contract belong to the explicit m1 profile.
    /// Guards are disjoint from each other and from every configured array.
    fn validate_native_profile(&self) -> Result<(), String> {
        if self.native_profile == NativeProfile::TargetDefault {
            if !self.guard_regions.is_empty() {
                return Err("guard regions require the gfx1250 B0 m1 profile".into());
            }
            if self.output_contract != OutputContract::Mathematical {
                return Err(
                    "the raw-u32 output contract requires the gfx1250 B0 m1 profile".into(),
                );
            }
        }
        if !self.initial_words.is_empty() {
            if !self.native_profile.b0_litmus() {
                return Err("initial array words require the gfx1250 B0 litmus profile".into());
            }
            for (i, init) in self.initial_words.iter().enumerate() {
                let array = self
                    .array(&init.array)
                    .ok_or_else(|| format!("initial words name unknown array '{}'", init.array))?;
                if array.elem_width != 4 || array.kind.is_input() {
                    return Err(format!(
                        "initial words require a four-byte non-input array, not '{}'",
                        init.array
                    ));
                }
                if self.initial_words[..i]
                    .iter()
                    .any(|old| old.array == init.array)
                {
                    return Err(format!("duplicate initial words for '{}'", init.array));
                }
            }
        }
        if self.native_profile == NativeProfile::TargetDefault {
            return Ok(());
        }
        if self.output_contract == OutputContract::RawU32
            && self
                .arrays
                .iter()
                .any(|array| array.kind.is_output() && array.elem_width != 4)
        {
            return Err("the raw-u32 output contract requires four-byte output elements".into());
        }
        let mut ranges = Vec::new();
        for guard in &self.guard_regions {
            if guard.name.is_empty() {
                return Err("guard region name is empty".into());
            }
            let end = guard
                .base
                .checked_add(guard.bytes)
                .ok_or_else(|| format!("guard region '{}' overflows", guard.name))?;
            ranges.push((guard.base, end, guard.name.as_str()));
        }
        for array in &self.arrays {
            ranges.push((
                array.base,
                array.base + array.size_bytes(),
                array.name.as_str(),
            ));
        }
        for (i, a) in ranges.iter().enumerate() {
            for b in &ranges[i + 1..] {
                if a.2 == b.2 {
                    return Err(format!("guard/array name '{}' is not unique", a.2));
                }
                if a.0 < b.1 && b.0 < a.1 {
                    return Err(format!("regions '{}' and '{}' overlap", a.2, b.2));
                }
            }
        }
        Ok(())
    }
}

/// A struct parameter's fields fit their widths and their byte ranges are disjoint.
fn validate_struct(param: usize, fields: &[StructField]) -> Result<(), String> {
    let mut ranges = Vec::new();
    for field in fields {
        if let FieldValue::Int { value, width } = field.value
            && !width.holds(value)
        {
            return Err(format!(
                "parameter {param}: the field at byte {} holds {value}, which does not fit in {} bytes",
                field.offset,
                width.bytes()
            ));
        }
        if let FieldValue::TensorMap(map) = &field.value {
            validate_tensor_map_field(param, field.offset, map)?;
        }
        let end = field
            .offset
            .checked_add(field.value.width_bytes())
            .ok_or_else(|| {
                format!(
                    "parameter {param}: the field at byte {} overflows",
                    field.offset
                )
            })?;
        ranges.push((field.offset, end));
    }
    ranges.sort();
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(format!(
                "parameter {param}: the fields at bytes {} and {} overlap",
                pair[0].0, pair[1].0
            ));
        }
    }
    Ok(())
}

fn validate_tensor_map_field(
    param: usize,
    offset: u64,
    map: &TensorMapValue,
) -> Result<(), String> {
    let rank = map.dims.len();
    let consistent = (1..=5).contains(&rank)
        && map.strides.len() + 1 == rank
        && map.element_strides.len() == rank;
    if !consistent {
        return Err(format!(
            "parameter {param}: the tensor map at byte {offset} needs 1 to 5 dims, one stride \
             fewer, and as many element strides"
        ));
    }

    let mode = match &map.mode {
        TensorMapMode::Tiled { box_dims } if box_dims.len() == rank => Ok(()),
        TensorMapMode::Tiled { .. } => Err("as many box dims as dims".to_string()),
        TensorMapMode::Im2col(im2col) => validate_im2col(rank, im2col),
    };

    mode.map_err(|e| format!("parameter {param}: the tensor map at byte {offset} needs {e}"))
}

/// The ranges `cuTensorMapEncodeIm2col` accepts for an im2col map of `rank` dims.
fn validate_im2col(rank: usize, im2col: &Im2colBox) -> Result<(), String> {
    let corner_range = match rank {
        3 => -32768..=32767,
        4 => -128..=127,
        5 => -16..=15,
        _ => return Err(format!("3 to 5 dims to be an im2col map, not {rank}")),
    };

    let spatial = rank - 2;
    if im2col.lower_corner.len() != spatial || im2col.upper_corner.len() != spatial {
        return Err(format!(
            "{spatial} lower and {spatial} upper corner offsets, one per spatial dim"
        ));
    }

    for corner in im2col.lower_corner.iter().chain(&im2col.upper_corner) {
        if !corner_range.contains(corner) {
            return Err(format!(
                "corner offsets in {corner_range:?} for {rank} dims, not {corner}"
            ));
        }
    }

    if !(1..=256).contains(&im2col.channels_per_pixel) {
        return Err(format!(
            "1 to 256 channels per pixel, not {}",
            im2col.channels_per_pixel
        ));
    }

    if !(1..=1024).contains(&im2col.pixels_per_column) {
        return Err(format!(
            "1 to 1024 pixels per column, not {}",
            im2col.pixels_per_column
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_workgroup_defaults_to_zero_and_must_be_inside_each_grid_axis() {
        let mut config = AnalysisConfig::new((4, 3, 2));
        assert_eq!(config.workgroup_id, (0, 0, 0));
        config.validate().unwrap();
        config.grid_dim = (2, 3, 4);
        config.workgroup_id = (1, 2, 3);
        config.validate().unwrap();
        for selection in [(2, 2, 3), (1, 3, 3), (1, 2, 4), (u32::MAX, 0, 0)] {
            config.workgroup_id = selection;
            assert!(config.validate().unwrap_err().contains("workgroup_id"));
        }
    }

    #[test]
    fn launch_geometry_requires_positive_dimensions_and_a_representable_thread_count() {
        for block in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (65_536, 65_536, 1),
            (65_536, 1, 65_536),
            (u32::MAX, 2, 1),
        ] {
            assert!(AnalysisConfig::new(block).validate().is_err(), "{block:?}");
        }
        let mut config = AnalysisConfig::new((8, 4, 2));
        for grid in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            config.grid_dim = grid;
            assert!(config.validate().is_err(), "{grid:?}");
        }
        config.grid_dim = (2, 3, 4);
        assert_eq!(config.validate(), Ok(()));
        assert_eq!(config.num_threads(), 64);
    }

    #[test]
    fn shared_layout_preserves_separate_adjacent_objects_and_rejects_ambiguous_ranges() {
        let a = SharedAllocation {
            name: "A".into(),
            offset: 0,
            size: 40,
        };
        let b = SharedAllocation {
            name: "B".into(),
            offset: 40,
            size: 64,
        };
        assert_eq!(validate_shared_allocations(&[]), Ok(()));
        assert_eq!(validate_shared_allocations(&[a.clone(), b.clone()]), Ok(()));
        for invalid in [
            SharedAllocation {
                name: "".into(),
                ..a.clone()
            },
            SharedAllocation {
                name: "A".into(),
                ..b.clone()
            },
            SharedAllocation {
                offset: 39,
                ..b.clone()
            },
            SharedAllocation {
                size: 0,
                ..b.clone()
            },
            SharedAllocation {
                offset: u64::MAX,
                size: 1,
                ..b.clone()
            },
        ] {
            assert!(validate_shared_allocations(&[a.clone(), invalid]).is_err());
        }
    }

    fn array(name: &str, base: u64, len: u64) -> ArrayDef {
        ArrayDef {
            name: name.to_string(),
            base,
            elem_width: 4,
            len,
            kind: ArrayKind::Input,
        }
    }

    #[test]
    fn validate_accepts_disjoint_arrays_and_plain_sym_names() {
        let mut config = AnalysisConfig::new((1, 1, 1));
        config.arrays.push(array("in", 0x1000, 16));
        config.arrays.push(array("out", 0x2000, 16));
        config
            .params
            .push(ParamValue::SymFloat("alpha".to_string()));
        assert_eq!(config.validate(), Ok(()));
    }

    /// Identity ambiguities are rejected: shared names would give unrelated
    /// storage the same `InputElement` symbols, and overlapping ranges would
    /// give one address two identities.
    #[test]
    fn validate_rejects_ambiguous_identities() {
        let mut dup = AnalysisConfig::new((1, 1, 1));
        dup.arrays.push(array("x", 0x1000, 16));
        dup.arrays.push(array("x", 0x2000, 16));
        assert!(dup.validate().is_err());

        let mut overlap = AnalysisConfig::new((1, 1, 1));
        overlap.arrays.push(array("a", 0x1000, 16));
        overlap.arrays.push(array("b", 0x1020, 16));
        assert!(overlap.validate().is_err());

        let mut bracketed = AnalysisConfig::new((1, 1, 1));
        bracketed
            .params
            .push(ParamValue::SymFloat("x[0]".to_string()));
        assert!(bracketed.validate().is_err());

        let mut dup_sym = AnalysisConfig::new((1, 1, 1));
        dup_sym
            .params
            .push(ParamValue::SymFloat("alpha".to_string()));
        dup_sym
            .params
            .push(ParamValue::SymFloat("alpha".to_string()));
        assert!(dup_sym.validate().is_err());
    }

    /// NaN float parameters are rejected at validation: NaN denotes no
    /// real number, so it has no place in the reals model. The infinities
    /// are fine (running-max/min seeds).
    #[test]
    fn validate_rejects_nan_float_param() {
        let mut nan = AnalysisConfig::new((1, 1, 1));
        nan.params.push(ParamValue::Float(f64::NAN));
        let err = nan.validate().unwrap_err();
        assert!(err.contains("NaN"), "unexpected message: {}", err);

        let mut inf = AnalysisConfig::new((1, 1, 1));
        inf.params.push(ParamValue::Float(f64::NEG_INFINITY));
        assert_eq!(inf.validate(), Ok(()));
    }

    #[test]
    fn validate_rejects_overlapping_or_overflowing_struct_fields() {
        let int = |offset, value, width| StructField {
            offset,
            value: FieldValue::Int { value, width },
        };
        let config = |fields| {
            let mut config = AnalysisConfig::new((1, 1, 1));
            config.params.push(ParamValue::Struct(fields));
            config.validate()
        };
        assert_eq!(
            config(vec![int(0, -1, IntWidth::W4), int(4, 255, IntWidth::W1)]),
            Ok(())
        );
        assert!(config(vec![int(0, 1, IntWidth::W8), int(4, 1, IntWidth::W4)]).is_err());
        assert!(config(vec![int(0, 256, IntWidth::W1)]).is_err());
    }

    /// A base that is not a multiple of the element width would make every
    /// element access misaligned (PTX ISA 6.4.1), so it is rejected once at
    /// config time rather than per access.
    #[test]
    fn validate_rejects_misaligned_array_base() {
        let mut misaligned = AnalysisConfig::new((1, 1, 1));
        misaligned.arrays.push(array("in", 2, 4));
        assert!(misaligned.validate().is_err());

        // The same base is fine for a 2-byte-element array: natural
        // alignment is relative to the element width, not any fixed size.
        let mut halves = AnalysisConfig::new((1, 1, 1));
        halves.arrays.push(ArrayDef {
            name: "in".to_string(),
            base: 2,
            elem_width: 2,
            len: 4,
            kind: ArrayKind::Input,
        });
        assert_eq!(halves.validate(), Ok(()));
    }
}

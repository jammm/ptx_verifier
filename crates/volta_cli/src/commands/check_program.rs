//! `volta check-program`: one kernel under the gfx1250 B0 profile its request
//! names: the m1 integer profile `gfx1250-b0-m1-int/1` (`ptxv-m1-request/1` ->
//! `ptxv-m1-report/1`) or its m2 MODE superset `gfx1250-b0-m2-mode/1`
//! (`ptxv-m2-request/1` -> `ptxv-m2-report/1`, which adds the `mode_order`
//! check and the MODE access record). The litmus profile `gfx1250-b0-litmus/1`
//! (`ptxv-litmus-request/1` -> `ptxv-litmus-report/1`) checks fuzzer
//! litmus template kernels over several initialized objects and reports
//! their final images; `gfx1250-b0-litmus/2` (`ptxv-litmus-request/2` ->
//! `ptxv-litmus-report/2`) adds the two-wave named-barrier JOIN template.
//!
//! The report is authoritative and the exit status equals its class
//! (0 PASS, 1 REJECT, 2 UNKNOWN_UNSUPPORTED, 3 ERROR, 4 TIMEOUT). Parser,
//! profile, execution, each graph check and the raw output words are kept as
//! separate results; nothing falls back to PASS.
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use serde_json::{Map, Value, json};
use volta_amdgpu_analysis::m1::{self, Profile, RefusedOp, Role};
use volta_amdgpu_analysis::{
    AnalysisConfig, ArrayDef, ArrayKind, GuardRegion, InitialPattern, InitialWords, OutputContract,
    ParamValue, RawWord, frontend,
};
use volta_graph::{
    AmdEventKind, AmdGlobalAccess, AmdMemoryClass, AmdRegister, AmdRegisterFile, EventGraph,
    EventId, EventKind, Space, Terminal,
};
use volta_graph_check::Verdict;
use volta_graph_check::amdgpu::global_bounds::GlobalBoundsRegion;
use volta_graph_check::amdgpu::mode_order::{ModeOrderKind, check_mode_order};
use volta_graph_check::amdgpu::{AmdRegisterHazard, AmdRegisterHazardKind};
use volta_pipeline::{Backend, KernelModule, SourceKernelVerification, verify_source_kernel};

use crate::sha256;

pub const REPORT_SCHEMA: &str = "ptxv-m1-report/1";
pub const REQUEST_SCHEMA: &str = "ptxv-m1-request/1";
pub const M2_REPORT_SCHEMA: &str = "ptxv-m2-report/1";
pub const M2_REQUEST_SCHEMA: &str = "ptxv-m2-request/1";
pub const LITMUS_REPORT_SCHEMA: &str = "ptxv-litmus-report/1";
pub const LITMUS_REQUEST_SCHEMA: &str = "ptxv-litmus-request/1";
pub const LITMUS2_REPORT_SCHEMA: &str = "ptxv-litmus-report/2";
pub const LITMUS2_REQUEST_SCHEMA: &str = "ptxv-litmus-request/2";
pub const WMMA_REPORT_SCHEMA: &str = "ptxv-wmma-report/1";
pub const WMMA_REQUEST_SCHEMA: &str = "ptxv-wmma-request/1";
pub const MW_REPORT_SCHEMA: &str = "ptxv-mw-report/1";
pub const MW_REQUEST_SCHEMA: &str = "ptxv-mw-request/1";
pub const WMMA2_REPORT_SCHEMA: &str = "ptxv-wmma-report/2";
pub const WMMA2_REQUEST_SCHEMA: &str = "ptxv-wmma-request/2";
pub const WMMA3_REPORT_SCHEMA: &str = "ptxv-wmma-report/3";
pub const WMMA3_REQUEST_SCHEMA: &str = "ptxv-wmma-request/3";
pub const WMMA4_REPORT_SCHEMA: &str = "ptxv-wmma-report/4";
pub const WMMA4_REQUEST_SCHEMA: &str = "ptxv-wmma-request/4";
pub const MW2_REPORT_SCHEMA: &str = "ptxv-mw-report/2";
pub const MW2_REQUEST_SCHEMA: &str = "ptxv-mw-request/2";
pub const MWPRED_REPORT_SCHEMA: &str = "ptxv-mwpred-report/1";
pub const MWPRED_REQUEST_SCHEMA: &str = "ptxv-mwpred-request/1";
pub const COMMS_REPORT_SCHEMA: &str = "ptxv-comms-report/1";
pub const COMMS_REQUEST_SCHEMA: &str = "ptxv-comms-request/1";
const CHECK_IDS: [&str; 9] = [
    "trace_completeness",
    "wait_readiness",
    "barrier_generations",
    "memory_conflicts",
    "object_bounds",
    "prefetch_policy",
    "store_completion",
    "mode_order",
    "wmma_hazards",
];
/// `gfx1250-b0-mwpred/1`: the nine checks of every workgroup, then the
/// composition check over the whole grid.
const MWPRED_CHECK_IDS: [&str; 10] = [
    "trace_completeness",
    "wait_readiness",
    "barrier_generations",
    "memory_conflicts",
    "object_bounds",
    "prefetch_policy",
    "store_completion",
    "mode_order",
    "wmma_hazards",
    "workgroup_isolation",
];
/// `gfx1250-b0-comms/1`: the seven integer-profile checks of every workgroup,
/// then the composition check over the grid.
const COMMS_CHECK_IDS: [&str; 8] = [
    "trace_completeness",
    "wait_readiness",
    "barrier_generations",
    "memory_conflicts",
    "object_bounds",
    "prefetch_policy",
    "store_completion",
    "workgroup_isolation",
];

fn check_ids(profile: Profile) -> &'static [&'static str] {
    match profile {
        Profile::M1Int | Profile::Litmus | Profile::Litmus2 => &CHECK_IDS[..7],
        Profile::M2Mode => &CHECK_IDS[..8],
        Profile::Wmma
        | Profile::Multiwave
        | Profile::Wmma2
        | Profile::Wmma3
        | Profile::Wmma4
        | Profile::Multiwave2 => &CHECK_IDS,
        Profile::MwPred => &MWPRED_CHECK_IDS,
        Profile::Comms => &COMMS_CHECK_IDS,
    }
}

fn report_schema(profile: Profile) -> &'static str {
    match profile {
        Profile::M1Int => REPORT_SCHEMA,
        Profile::M2Mode => M2_REPORT_SCHEMA,
        Profile::Litmus => LITMUS_REPORT_SCHEMA,
        Profile::Litmus2 => LITMUS2_REPORT_SCHEMA,
        Profile::Wmma => WMMA_REPORT_SCHEMA,
        Profile::Multiwave => MW_REPORT_SCHEMA,
        Profile::Wmma2 => WMMA2_REPORT_SCHEMA,
        Profile::Wmma3 => WMMA3_REPORT_SCHEMA,
        Profile::Wmma4 => WMMA4_REPORT_SCHEMA,
        Profile::Multiwave2 => MW2_REPORT_SCHEMA,
        Profile::MwPred => MWPRED_REPORT_SCHEMA,
        Profile::Comms => COMMS_REPORT_SCHEMA,
    }
}

/// The profile a request selects by its `schema` (m1 when absent or unknown,
/// so malformed requests keep the m1 report shape).
fn requested_profile(value: &Value) -> Profile {
    if value.get("schema") == Some(&json!(M2_REQUEST_SCHEMA)) {
        Profile::M2Mode
    } else if value.get("schema") == Some(&json!(LITMUS_REQUEST_SCHEMA)) {
        Profile::Litmus
    } else if value.get("schema") == Some(&json!(LITMUS2_REQUEST_SCHEMA)) {
        Profile::Litmus2
    } else if value.get("schema") == Some(&json!(WMMA_REQUEST_SCHEMA)) {
        Profile::Wmma
    } else if value.get("schema") == Some(&json!(MW_REQUEST_SCHEMA)) {
        Profile::Multiwave
    } else if value.get("schema") == Some(&json!(WMMA2_REQUEST_SCHEMA)) {
        Profile::Wmma2
    } else if value.get("schema") == Some(&json!(WMMA3_REQUEST_SCHEMA)) {
        Profile::Wmma3
    } else if value.get("schema") == Some(&json!(WMMA4_REQUEST_SCHEMA)) {
        Profile::Wmma4
    } else if value.get("schema") == Some(&json!(MW2_REQUEST_SCHEMA)) {
        Profile::Multiwave2
    } else if value.get("schema") == Some(&json!(MWPRED_REQUEST_SCHEMA)) {
        Profile::MwPred
    } else if value.get("schema") == Some(&json!(COMMS_REQUEST_SCHEMA)) {
        Profile::Comms
    } else {
        Profile::M1Int
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResultClass {
    Pass,
    UnknownUnsupported,
    Timeout,
    Error,
    Reject,
}

impl ResultClass {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Reject => "REJECT",
            Self::UnknownUnsupported => "UNKNOWN_UNSUPPORTED",
            Self::Error => "ERROR",
            Self::Timeout => "TIMEOUT",
        }
    }
    pub const fn exit_code(self) -> u8 {
        match self {
            Self::Pass => 0,
            Self::Reject => 1,
            Self::UnknownUnsupported => 2,
            Self::Error => 3,
            Self::Timeout => 4,
        }
    }
}

/// The validated request: the frozen `ptxv-m1-request/1` launch premises.
#[derive(Debug, Clone)]
struct Request {
    profile: Profile,
    role: Role,
    role_name: String,
    enabled_pending: Vec<String>,
    source_path: PathBuf,
    source_path_text: String,
    source_sha256: String,
    object_sha256: String,
    max_instructions: u64,
    out_base: u64,
    out_words: u64,
    guards: Vec<GuardRegion>,
    /// Litmus profile: the declared objects, contiguous from `out_base`, in
    /// output order; empty for m1/m2 (whose single object is `out`).
    objects: Vec<LitmusObject>,
    block: u32,
    /// mwpred: workgroups in the grid (every one is analysed); 1 otherwise.
    grid: u32,
    /// mwpred per-workgroup evaluation: the declared-output check runs once
    /// on the composed image, not per workgroup.
    defer_outputs: bool,
    lds_bytes: u64,
    /// Multiwave profile: `out` word ranges `(first, count)` that must hold an
    /// opaque evidence value, and ranges that must stay unwritten.
    opaque_words: Vec<(u64, u64)>,
    unwritten_words: Vec<(u64, u64)>,
}

#[derive(Debug, Clone)]
struct LitmusObject {
    name: String,
    base: u64,
    words: u64,
    initial: InitialPattern,
}

fn hex_u64(value: &Value, what: &str) -> Result<u64, String> {
    let text = value
        .as_str()
        .filter(|t| {
            t.len() > 2
                && t.starts_with("0x")
                && t[2..].chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
        })
        .ok_or_else(|| format!("{what} must be a lowercase 0x-prefixed hex string"))?;
    u64::from_str_radix(&text[2..], 16).map_err(|_| format!("{what} exceeds 64 bits"))
}

fn integer(value: &Value, what: &str, range: std::ops::RangeInclusive<u64>) -> Result<u64, String> {
    value
        .as_u64()
        .filter(|v| range.contains(v))
        .ok_or_else(|| format!("{what} must be an integer in {range:?}"))
}

fn sha_field(value: &Value, what: &str) -> Result<String, String> {
    value
        .as_str()
        .filter(|t| t.len() == 64 && t.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')))
        .map(str::to_owned)
        .ok_or_else(|| format!("{what} must be 64 lowercase hex digits"))
}

fn object<'a>(
    value: &'a Value,
    what: &str,
    required: &[&str],
    optional: &[&str],
) -> Result<&'a Map<String, Value>, String> {
    let map = value
        .as_object()
        .ok_or_else(|| format!("{what} must be an object"))?;
    if let Some(key) = map
        .keys()
        .find(|key| !required.contains(&key.as_str()) && !optional.contains(&key.as_str()))
    {
        return Err(format!("{what} has unknown field `{key}`"));
    }
    if let Some(key) = required.iter().find(|key| !map.contains_key(**key)) {
        return Err(format!("{what} is missing `{key}`"));
    }
    Ok(map)
}

fn constant(value: &Value, expected: Value, what: &str) -> Result<(), String> {
    (value == &expected)
        .then_some(())
        .ok_or_else(|| format!("{what} must be {expected}"))
}

/// Exact validation of the frozen request schema plus the launch relations
/// the schema cannot express (word count, guard and object placement).
fn parse_request(value: &Value, request_dir: &Path) -> Result<Request, String> {
    let litmus = requested_profile(value);
    if litmus.litmus() {
        return parse_litmus_request(value, request_dir, litmus);
    }
    let top = object(
        value,
        "request",
        &[
            "schema",
            "profile",
            "enable_pending_ops",
            "role",
            "kernel",
            "source",
            "object",
            "launch_file",
            "backing_allocation",
            "guards",
            "outputs",
        ],
        &["lds_bytes"],
    )?;
    let profile = requested_profile(value);
    let lds_bytes = match (profile, top.get("lds_bytes")) {
        (Profile::Comms, Some(bytes)) => integer(bytes, "lds_bytes", 0..=327_680)?,
        (Profile::Comms, None) => return Err("request is missing `lds_bytes`".into()),
        (_, None) => 0,
        (_, Some(_)) => return Err("lds_bytes is defined for gfx1250-b0-comms/1 only".into()),
    };
    let schema = match profile {
        Profile::M1Int => REQUEST_SCHEMA,
        Profile::M2Mode => M2_REQUEST_SCHEMA,
        Profile::Wmma => WMMA_REQUEST_SCHEMA,
        Profile::Multiwave => MW_REQUEST_SCHEMA,
        Profile::Wmma2 => WMMA2_REQUEST_SCHEMA,
        Profile::Wmma3 => WMMA3_REQUEST_SCHEMA,
        Profile::Wmma4 => WMMA4_REQUEST_SCHEMA,
        Profile::Multiwave2 => MW2_REQUEST_SCHEMA,
        Profile::MwPred => MWPRED_REQUEST_SCHEMA,
        Profile::Comms => COMMS_REQUEST_SCHEMA,
        Profile::Litmus | Profile::Litmus2 => {
            unreachable!("litmus requests have their own parser")
        }
    };
    constant(&top["schema"], json!(schema), "schema")?;
    constant(&top["profile"], json!(profile.id()), "profile")?;
    constant(&top["kernel"], json!(m1::ENTRY), "kernel")?;
    let pending = top["enable_pending_ops"]
        .as_array()
        .ok_or("enable_pending_ops must be an array")?;
    let mut enabled_pending = Vec::new();
    for op in pending {
        let op = op
            .as_str()
            .filter(|op| profile.pending_ops().contains(op))
            .ok_or("enable_pending_ops items must be pending profile ops")?;
        if enabled_pending.iter().any(|old| old == op) {
            return Err("enable_pending_ops items must be unique".into());
        }
        enabled_pending.push(op.to_owned());
    }
    let role_name = top["role"].as_str().unwrap_or_default().to_owned();
    let role = match role_name.as_str() {
        "cand" => Role::Candidate,
        "twin" => Role::Twin,
        "control" => Role::Control,
        _ => return Err("role must be cand, twin or control".into()),
    };
    let source = object(&top["source"], "source", &["path", "sha256"], &[])?;
    let source_path_text = source["path"]
        .as_str()
        .ok_or("source.path must be a string")?
        .to_owned();
    let source_sha256 = sha_field(&source["sha256"], "source.sha256")?;
    let object_value = object(&top["object"], "object", &["sha256"], &["path"])?;
    if object_value
        .get("path")
        .is_some_and(|path| !path.is_string())
    {
        return Err("object.path must be a string".into());
    }
    let object_sha256 = sha_field(&object_value["sha256"], "object.sha256")?;
    let launch = object(
        &top["launch_file"],
        "launch_file",
        &[
            "block",
            "grid",
            "kernel",
            "arrays",
            "params",
            "max_instructions",
        ],
        &["workgroup_id", "workgroups"],
    )?;
    // mwpred: a grid of 1..1024 workgroups of 1..32 wave32 waves, every
    // workgroup analysed; comms: a grid of 1..64 workgroups of 2..8 waves.
    let mwpred = profile == Profile::MwPred;
    let grid_profile = mwpred || profile == Profile::Comms;
    let block = if mwpred {
        let threads = launch["block"]
            .as_array()
            .filter(|b| b.len() == 3 && b[1] == json!(1) && b[2] == json!(1))
            .and_then(|b| b[0].as_u64())
            .filter(|t| t % 32 == 0 && (32..=1024).contains(t))
            .ok_or("launch_file.block must be [32*W, 1, 1] with W in 1..32")?;
        threads as u32
    } else if profile.multiwave() {
        let threads = launch["block"]
            .as_array()
            .filter(|b| b.len() == 3 && b[1] == json!(1) && b[2] == json!(1))
            .and_then(|b| b[0].as_u64())
            .filter(|t| t % 32 == 0 && (64..=256).contains(t))
            .ok_or("launch_file.block must be [32*W, 1, 1] with W in 2..8")?;
        threads as u32
    } else {
        constant(&launch["block"], json!([32, 1, 1]), "launch_file.block")?;
        32
    };
    let grid = if grid_profile {
        constant(
            launch.get("workgroups").unwrap_or(&Value::Null),
            json!("all"),
            "launch_file.workgroups",
        )?;
        if launch.get("workgroup_id").is_some() {
            return Err(
                "launch_file.workgroup_id is not allowed: every workgroup is analysed".into(),
            );
        }
        launch["grid"]
            .as_array()
            .filter(|g| g.len() == 3 && g[1] == json!(1) && g[2] == json!(1))
            .and_then(|g| g[0].as_u64())
            .filter(|g| (1..=if mwpred { 1024 } else { 64 }).contains(g))
            .ok_or("launch_file.grid must be [G, 1, 1] with G in 1..1024 (comms: 1..64)")?
            as u32
    } else {
        constant(&launch["grid"], json!([1, 1, 1]), "launch_file.grid")?;
        if let Some(workgroup) = launch.get("workgroup_id") {
            constant(workgroup, json!([0, 0, 0]), "launch_file.workgroup_id")?;
        }
        if launch.get("workgroups").is_some() {
            return Err(
                "launch_file.workgroups is defined for gfx1250-b0-mwpred/1 and gfx1250-b0-comms/1 only".into(),
            );
        }
        1
    };
    constant(&launch["kernel"], json!(m1::ENTRY), "launch_file.kernel")?;
    constant(
        &launch["params"],
        json!([{"ptr": "out"}]),
        "launch_file.params",
    )?;
    let max_words = if grid_profile { 65536 } else { 8192 };
    let max_instructions = integer(
        &launch["max_instructions"],
        "launch_file.max_instructions",
        1..=200_000,
    )?;
    let arrays = launch["arrays"]
        .as_array()
        .filter(|arrays| arrays.len() == 1)
        .ok_or("launch_file.arrays must contain exactly one array")?;
    let array = object(
        &arrays[0],
        "launch_file.arrays[0]",
        &["name", "base", "elem_width", "len", "kind"],
        &[],
    )?;
    constant(&array["name"], json!("out"), "array name")?;
    constant(&array["elem_width"], json!(4), "array elem_width")?;
    constant(&array["kind"], json!("out"), "array kind")?;
    let out_base = hex_u64(&array["base"], "array base")?;
    let out_len = integer(&array["len"], "array len", 32..=max_words)?;
    let backing = object(
        &top["backing_allocation"],
        "backing_allocation",
        &["base", "bytes"],
        &[],
    )?;
    let backing_base = hex_u64(&backing["base"], "backing_allocation.base")?;
    let backing_bytes = integer(&backing["bytes"], "backing_allocation.bytes", 1..=u64::MAX)?;
    let guard_values = top["guards"]
        .as_array()
        .filter(|guards| guards.len() == 2)
        .ok_or("guards must contain exactly two regions")?;
    let mut guards = Vec::new();
    for guard in guard_values {
        let guard = object(guard, "guard", &["name", "base", "bytes", "fill_u32"], &[])?;
        let name = guard["name"]
            .as_str()
            .filter(|name| matches!(*name, "pre" | "post"))
            .ok_or("guard name must be pre or post")?;
        constant(&guard["fill_u32"], json!("0xffffffff"), "guard fill_u32")?;
        if guards.iter().any(|g: &GuardRegion| g.name == name) {
            return Err(format!("duplicate {name} guard"));
        }
        guards.push(GuardRegion {
            name: name.into(),
            base: hex_u64(&guard["base"], "guard base")?,
            bytes: integer(&guard["bytes"], "guard bytes", 0..=u64::MAX)?,
        });
    }
    let multiwave = profile.multiwave();
    let outputs = if multiwave {
        object(
            &top["outputs"],
            "outputs",
            &["contract", "object", "words", "opaque", "not_written"],
            &[],
        )?
    } else {
        object(
            &top["outputs"],
            "outputs",
            &["contract", "object", "words"],
            &[],
        )?
    };
    let contract = if multiwave { "raw-u32-mw" } else { "raw-u32" };
    constant(&outputs["contract"], json!(contract), "outputs.contract")?;
    constant(&outputs["object"], json!("out"), "outputs.object")?;
    let out_words = integer(&outputs["words"], "outputs.words", 32..=max_words)?;
    if out_words != out_len {
        return Err("outputs.words must equal the out array length".into());
    }
    // Disjoint, in-bounds `[first, count]` word ranges.
    let ranges = |key: &str| -> Result<Vec<(u64, u64)>, String> {
        let Some(items) = outputs.get(key) else {
            return Ok(Vec::new());
        };
        let items = items
            .as_array()
            .ok_or_else(|| format!("outputs.{key} must be an array"))?;
        items
            .iter()
            .map(|item| {
                let pair = item
                    .as_array()
                    .filter(|p| p.len() == 2)
                    .ok_or_else(|| format!("outputs.{key} items must be [first, count]"))?;
                let first = integer(&pair[0], "range first", 0..=out_words)?;
                let count = integer(&pair[1], "range count", 1..=out_words)?;
                (first + count <= out_words)
                    .then_some((first, count))
                    .ok_or_else(|| format!("outputs.{key} range exceeds the out array"))
            })
            .collect()
    };
    let opaque_words = ranges("opaque")?;
    let unwritten_words = ranges("not_written")?;
    let mut all: Vec<_> = opaque_words
        .iter()
        .chain(&unwritten_words)
        .copied()
        .collect();
    all.sort_unstable();
    if all.windows(2).any(|w| w[0].0 + w[0].1 > w[1].0) {
        return Err("outputs opaque/not_written ranges overlap".into());
    }
    let backing_end = backing_base
        .checked_add(backing_bytes)
        .ok_or("backing allocation overflows")?;
    let mut regions = vec![("out", out_base, out_base + 4 * out_len)];
    for guard in &guards {
        let end = guard
            .base
            .checked_add(guard.bytes)
            .ok_or("guard overflows")?;
        regions.push((guard.name.as_str(), guard.base, end));
    }
    for (name, start, end) in &regions {
        if *start < backing_base || *end > backing_end {
            return Err(format!("region {name} lies outside the backing allocation"));
        }
    }
    for (i, a) in regions.iter().enumerate() {
        for b in &regions[i + 1..] {
            if a.1 < b.2 && b.1 < a.2 {
                return Err(format!("regions {} and {} overlap", a.0, b.0));
            }
        }
    }
    let path = PathBuf::from(&source_path_text);
    Ok(Request {
        profile,
        role,
        role_name,
        enabled_pending,
        source_path: if path.is_absolute() {
            path
        } else {
            request_dir.join(path)
        },
        source_path_text,
        source_sha256,
        object_sha256,
        max_instructions,
        out_base,
        out_words,
        guards,
        objects: Vec::new(),
        block,
        grid,
        defer_outputs: false,
        lds_bytes,
        opaque_words,
        unwritten_words,
    })
}

/// The u32 of a `0x`-prefixed lowercase hex string.
fn hex_u32(value: &Value, what: &str) -> Result<u32, String> {
    u32::try_from(hex_u64(value, what)?).map_err(|_| format!("{what} exceeds 32 bits"))
}

/// `ptxv-litmus-request/1`: the m1 envelope with contiguous initialized
/// objects instead of the single `out` array. `/2` also allows a 64-workitem
/// block (two waves of one workgroup).
fn parse_litmus_request(
    value: &Value,
    request_dir: &Path,
    profile: Profile,
) -> Result<Request, String> {
    let top = object(
        value,
        "request",
        &[
            "schema",
            "profile",
            "role",
            "kernel",
            "source",
            "object",
            "launch_file",
            "backing_allocation",
            "guards",
            "objects",
            "lds_bytes",
            "outputs",
        ],
        &["template"],
    )?;
    let schema = if profile == Profile::Litmus2 {
        LITMUS2_REQUEST_SCHEMA
    } else {
        LITMUS_REQUEST_SCHEMA
    };
    constant(&top["schema"], json!(schema), "schema")?;
    constant(&top["profile"], json!(profile.id()), "profile")?;
    constant(&top["kernel"], json!(m1::ENTRY), "kernel")?;
    if top.get("template").is_some_and(|t| !t.is_string()) {
        return Err("template must be a string".into());
    }
    let role_name = top["role"].as_str().unwrap_or_default().to_owned();
    let role = match role_name.as_str() {
        "cand" => Role::Candidate,
        "twin" => Role::Twin,
        "control" => Role::Control,
        _ => return Err("role must be cand, twin or control".into()),
    };
    let source = object(&top["source"], "source", &["path", "sha256"], &[])?;
    let source_path_text = source["path"]
        .as_str()
        .ok_or("source.path must be a string")?
        .to_owned();
    let source_sha256 = sha_field(&source["sha256"], "source.sha256")?;
    let object_value = object(&top["object"], "object", &["sha256"], &["path"])?;
    if object_value
        .get("path")
        .is_some_and(|path| !path.is_string())
    {
        return Err("object.path must be a string".into());
    }
    let object_sha256 = sha_field(&object_value["sha256"], "object.sha256")?;
    let launch = object(
        &top["launch_file"],
        "launch_file",
        &["block", "grid", "kernel", "params", "max_instructions"],
        &["workgroup_id"],
    )?;
    let block = if profile == Profile::Litmus2 && launch["block"] == json!([64, 1, 1]) {
        64
    } else {
        constant(&launch["block"], json!([32, 1, 1]), "launch_file.block")?;
        32
    };
    constant(&launch["grid"], json!([1, 1, 1]), "launch_file.grid")?;
    if let Some(workgroup) = launch.get("workgroup_id") {
        constant(workgroup, json!([0, 0, 0]), "launch_file.workgroup_id")?;
    }
    constant(&launch["kernel"], json!(m1::ENTRY), "launch_file.kernel")?;
    let max_instructions = integer(
        &launch["max_instructions"],
        "launch_file.max_instructions",
        1..=200_000,
    )?;
    let object_values = top["objects"]
        .as_array()
        .filter(|objects| (1..=4).contains(&objects.len()))
        .ok_or("objects must contain one to four objects")?;
    let mut objects: Vec<LitmusObject> = Vec::new();
    for item in object_values {
        let item = object(item, "object", &["name", "base", "words", "initial"], &[])?;
        let name = item["name"]
            .as_str()
            .filter(|name| matches!(*name, "out" | "data"))
            .ok_or("object name must be out or data")?
            .to_owned();
        if objects.iter().any(|o| o.name == name) {
            return Err(format!("duplicate object {name}"));
        }
        let base = hex_u64(&item["base"], "object base")?;
        let words = integer(&item["words"], "object words", 1..=65_536)?;
        let initial = object(
            &item["initial"],
            "object initial",
            &[],
            &["fill_u32", "index_base_u32"],
        )?;
        let initial = match (initial.get("fill_u32"), initial.get("index_base_u32")) {
            (Some(fill), None) => InitialPattern::Fill(hex_u32(fill, "initial.fill_u32")?),
            (None, Some(index)) => InitialPattern::Index(hex_u32(index, "initial.index_base_u32")?),
            _ => return Err("object initial needs exactly one of fill_u32, index_base_u32".into()),
        };
        if let Some(previous) = objects.last()
            && previous.base + 4 * previous.words != base
        {
            return Err("objects must be contiguous in request order".into());
        }
        objects.push(LitmusObject {
            name,
            base,
            words,
            initial,
        });
    }
    constant(
        &launch["params"],
        json!([{"ptr": objects[0].name}]),
        "launch_file.params",
    )?;
    let lds_bytes = integer(&top["lds_bytes"], "lds_bytes", 0..=327_680)?;
    let backing = object(
        &top["backing_allocation"],
        "backing_allocation",
        &["base", "bytes"],
        &[],
    )?;
    let backing_base = hex_u64(&backing["base"], "backing_allocation.base")?;
    let backing_bytes = integer(&backing["bytes"], "backing_allocation.bytes", 1..=u64::MAX)?;
    let guard_values = top["guards"]
        .as_array()
        .filter(|guards| guards.len() == 2)
        .ok_or("guards must contain exactly two regions")?;
    let mut guards = Vec::new();
    for guard in guard_values {
        let guard = object(guard, "guard", &["name", "base", "bytes", "fill_u32"], &[])?;
        let name = guard["name"]
            .as_str()
            .filter(|name| matches!(*name, "pre" | "post"))
            .ok_or("guard name must be pre or post")?;
        constant(&guard["fill_u32"], json!("0xffffffff"), "guard fill_u32")?;
        if guards.iter().any(|g: &GuardRegion| g.name == name) {
            return Err(format!("duplicate {name} guard"));
        }
        guards.push(GuardRegion {
            name: name.into(),
            base: hex_u64(&guard["base"], "guard base")?,
            bytes: integer(&guard["bytes"], "guard bytes", 0..=u64::MAX)?,
        });
    }
    let out_base = objects[0].base;
    let out_words: u64 = objects.iter().map(|o| o.words).sum();
    let outputs = object(
        &top["outputs"],
        "outputs",
        &["contract", "objects", "words"],
        &[],
    )?;
    constant(
        &outputs["contract"],
        json!("raw-u32-image"),
        "outputs.contract",
    )?;
    constant(
        &outputs["objects"],
        json!(objects.iter().map(|o| o.name.clone()).collect::<Vec<_>>()),
        "outputs.objects",
    )?;
    constant(&outputs["words"], json!(out_words), "outputs.words")?;
    let backing_end = backing_base
        .checked_add(backing_bytes)
        .ok_or("backing allocation overflows")?;
    let mut regions = vec![("region", out_base, out_base + 4 * out_words)];
    for guard in &guards {
        let end = guard
            .base
            .checked_add(guard.bytes)
            .ok_or("guard overflows")?;
        regions.push((guard.name.as_str(), guard.base, end));
    }
    for (name, start, end) in &regions {
        if *start < backing_base || *end > backing_end {
            return Err(format!("region {name} lies outside the backing allocation"));
        }
    }
    for (i, a) in regions.iter().enumerate() {
        for b in &regions[i + 1..] {
            if a.1 < b.2 && b.1 < a.2 {
                return Err(format!("regions {} and {} overlap", a.0, b.0));
            }
        }
    }
    let path = PathBuf::from(&source_path_text);
    Ok(Request {
        profile,
        role,
        role_name,
        enabled_pending: Vec::new(),
        source_path: if path.is_absolute() {
            path
        } else {
            request_dir.join(path)
        },
        source_path_text,
        source_sha256,
        object_sha256,
        max_instructions,
        out_base,
        out_words,
        guards,
        objects,
        block,
        grid: 1,
        defer_outputs: false,
        lds_bytes,
        opaque_words: Vec::new(),
        unwritten_words: Vec::new(),
    })
}

fn verifier_identity() -> Value {
    let exe = std::fs::read("/proc/self/exe")
        .map(|bytes| sha256::hex(&bytes))
        .unwrap_or_else(|_| "0".repeat(64));
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "git_rev": option_env!("VOLTA_GIT_REV").unwrap_or("unknown"),
        "git_dirty": option_env!("VOLTA_GIT_DIRTY") != Some("false"),
        "exe_sha256": exe,
    })
}

fn profile_block(profile: Profile, enabled: &[String]) -> Value {
    let mut rules = vec![
        ("D1-sched-mode0-hardware-interlocks", "documented"),
        ("D2-wait-alu-depctr-fields-alu-only", "documented"),
        (
            "D3-wait-alu-hold-cnt-raw-timing-only",
            "conservative_policy",
        ),
        ("D4-kmcnt-kernarg-readiness", "documented"),
        ("D5-global-prefetch-scope-se-xcnt-no-data", "documented"),
        ("D6-s-max-scc-strict-greater", "hardware_observation"),
    ];
    let source = match profile {
        Profile::M1Int => {
            rules.push(("D7-no-m1-required-waits", "conservative_policy"));
            "tests/gfx1250/CONTRACT.md: m1 integer profile"
        }
        Profile::M2Mode
        | Profile::Wmma
        | Profile::Multiwave
        | Profile::Wmma2
        | Profile::Wmma3
        | Profile::Wmma4
        | Profile::Multiwave2
        | Profile::MwPred => {
            rules.extend([
                (
                    "M2-getreg-mode-then-mode-write-without-sa-sdst-wait-unresolved",
                    "conservative_policy",
                ),
                ("M3-setreg-mode-then-getreg-mode-ordered", "documented"),
                (
                    "M4-mode-write-then-read-or-write-program-order",
                    "hardware_observation",
                ),
                (
                    "M5-only-getreg-releases-x-count-leases",
                    "conservative_policy",
                ),
                ("X1-fp-valu-outside-mode-profile", "conservative_policy"),
                (
                    "X2-setreg-imm32-subset-fp-fields-4-bit-immediate",
                    "conservative_policy",
                ),
                ("X3-s-set-vgpr-msb-refused", "conservative_policy"),
            ]);
            if profile == Profile::M2Mode {
                "tests/gfx1250/CONTRACT.md: m2 MODE profile"
            } else if profile == Profile::Wmma2 {
                rules.extend([
                    (
                        "W1-wmma-d-to-next-wmma-ab-slots-per-type-f16-bf16-5-fp8x64-2-fp8x128-4-iu8-9",
                        "documented",
                    ),
                    (
                        "W2-wmma-to-valu-raw-waw-war-slots-per-type-4-1-3-8",
                        "documented",
                    ),
                    (
                        "W3-unlisted-wmma-chain-below-the-type-wmma-count-unresolved",
                        "conservative_policy",
                    ),
                    ("W4-same-argument-wmma-back-to-back-refused", "documented"),
                    (
                        "W5-slot-is-v-nop-or-independent-vector-alu-issued-with-nonzero-exec",
                        "documented_interpretation",
                    ),
                    ("W6-wmma-requires-full-exec", "documented"),
                    (
                        "W7-exact-representable-regime-result-bits",
                        "conservative_policy",
                    ),
                    (
                        "W8-exec-zero-instructions-are-not-slots-unresolved",
                        "conservative_policy",
                    ),
                    (
                        "W9-fp8-iu8-uniform-tiles-only-iu8-bytes-below-0x80-int32",
                        "conservative_policy",
                    ),
                    (
                        "W10-exec-writes-only-s-mov-b32-exec-lo-0-or-minus-1",
                        "conservative_policy",
                    ),
                ]);
                "tests/gfx1250/CONTRACT.md: WMMA types profile"
            } else if profile.wmma_scale() {
                rules.extend([
                    (
                        "W1-wmma-d-to-next-wmma-ab-slots-per-type-f16-bf16-5-fp8-bf8x64-2-fp8-bf8x128-4-iu8-9-f8f6f4-both-fp4-2-other-f8f6f4-4-f4-32x16-4",
                        "documented",
                    ),
                    (
                        "W2-wmma-to-valu-raw-waw-war-slots-per-type-4-1-3-8-1-3-3",
                        "documented",
                    ),
                    (
                        "W3-unlisted-wmma-chain-below-the-type-wmma-count-unresolved",
                        "conservative_policy",
                    ),
                    ("W4-same-argument-wmma-back-to-back-refused", "documented"),
                    (
                        "W5-slot-is-v-nop-or-independent-vector-alu-issued-with-nonzero-exec",
                        "documented_interpretation",
                    ),
                    ("W6-wmma-requires-full-exec", "documented"),
                    (
                        "W7-exact-representable-regime-result-bits",
                        "conservative_policy",
                    ),
                    (
                        "W8-exec-zero-instructions-are-not-slots-unresolved",
                        "conservative_policy",
                    ),
                    (
                        "W9-fp8-iu8-uniform-tiles-only-iu8-bytes-below-0x80-int32",
                        "conservative_policy",
                    ),
                    (
                        "W10-exec-writes-only-s-mov-b32-exec-lo-0-or-minus-1",
                        "conservative_policy",
                    ),
                    (
                        "W11-scale-operand-dependency-below-the-type-count-unresolved",
                        "conservative_policy",
                    ),
                    (
                        "W12-block-scaled-wmma-is-one-instruction-load-scale-is-not-a-slot",
                        "documented",
                    ),
                    (
                        "W13-small-float-uniform-tiles-power-of-two-scales-only",
                        "conservative_policy",
                    ),
                    (
                        "W14-f8f6f4-operand-widths-vgpr-scales-legal-format-pairs-32x16-in-place-or-disjoint",
                        "conservative_policy",
                    ),
                ]);
                if profile.wmma_sparse16() {
                    rules.extend([
                        (
                            "W15-16-bit-output-and-swmmac-slots-f16-bf16-5-4-fp8x64-2-1-fp8x128-4-3-swmmac-3-2-swmmac-iu8-5-4",
                            "documented",
                        ),
                        (
                            "W16-swmmac-index-is-an-a-b-index-source-documented",
                            "documented",
                        ),
                        (
                            "W17-f16-results-11-bit-regime-bf16-results-never-certified",
                            "conservative_policy",
                        ),
                        (
                            "W18-swmmac-uniform-packed-a-half-uniform-b-valid-uniform-index-only",
                            "conservative_policy",
                        ),
                    ]);
                    if profile == Profile::Multiwave2 {
                        rules.extend([
                            ("MW1-one-workgroup-2-to-8-waves", "documented"),
                            ("MW2-cycle-and-hw-id1-reads-opaque", "conservative_policy"),
                            (
                                "MW3-single-group-xnack-replay-drain",
                                "hardware_observation",
                            ),
                            (
                                "MW4-same-address-global-store-lanes-ordered",
                                "hardware_observation",
                            ),
                            (
                                "MW5-wmma-spacing-per-wave-cross-wave-effects-not-modeled",
                                "conservative_policy",
                            ),
                        ]);
                        "tests/gfx1250/CONTRACT.md: multiwave WMMA-types profile"
                    } else {
                        "tests/gfx1250/CONTRACT.md: WMMA 16-bit output and SWMMAC profile"
                    }
                } else {
                    "tests/gfx1250/CONTRACT.md: WMMA scale profile"
                }
            } else {
                rules.extend([
                    ("W1-wmma-d-to-next-wmma-ab-5-slots", "documented"),
                    ("W2-wmma-to-valu-raw-waw-war-4-slots", "documented"),
                    (
                        "W3-unlisted-wmma-chain-below-5-slots-unresolved",
                        "conservative_policy",
                    ),
                    ("W4-same-argument-wmma-back-to-back-refused", "documented"),
                    (
                        "W5-slot-is-v-nop-or-independent-vector-alu",
                        "documented_interpretation",
                    ),
                    ("W6-wmma-requires-full-exec", "documented"),
                    (
                        "W7-exact-representable-regime-result-bits",
                        "conservative_policy",
                    ),
                ]);
                if profile == Profile::Multiwave {
                    rules.extend([
                        ("MW1-one-workgroup-2-to-8-waves", "documented"),
                        ("MW2-cycle-and-hw-id1-reads-opaque", "conservative_policy"),
                        (
                            "MW3-single-group-xnack-replay-drain",
                            "hardware_observation",
                        ),
                        (
                            "MW4-same-address-global-store-lanes-ordered",
                            "hardware_observation",
                        ),
                    ]);
                }
                if profile == Profile::MwPred {
                    rules.extend([
                        (
                            "P1-every-workgroup-of-a-grid-of-1-to-1024-analysed",
                            "conservative_policy",
                        ),
                        (
                            "P2-workgroups-isolated-disjoint-writes-no-shared-reads",
                            "conservative_policy",
                        ),
                        ("P3-output-image-one-writer-per-word", "conservative_policy"),
                        (
                            "P4-se-xcc-id-message-return-opaque-drained-by-the-next-instruction",
                            "documented",
                        ),
                        ("P5-ttmp9-is-the-workgroup-id-x", "documented"),
                        ("MW2-cycle-and-hw-id1-reads-opaque", "conservative_policy"),
                        (
                            "MW3-single-group-xnack-replay-drain",
                            "hardware_observation",
                        ),
                        (
                            "MW4-same-address-global-store-lanes-ordered",
                            "hardware_observation",
                        ),
                    ]);
                    "ptx_verifier mwpred/1 profile (multi-workgroup predecessor grids)"
                } else {
                    "tests/gfx1250/CONTRACT.md: WMMA and multiwave profiles"
                }
            }
        }
        Profile::Comms => {
            rules.extend([
                ("MW1-one-workgroup-2-to-8-waves-per-analysed-workgroup", "documented"),
                ("MW2-cycle-and-hw-id1-reads-opaque", "conservative_policy"),
                ("MW3-single-group-xnack-replay-drain", "hardware_observation"),
                ("MW4-same-address-global-store-lanes-ordered", "hardware_observation"),
                (
                    "C1-lds-store-completion-then-barrier-then-load-sees-the-stored-bytes",
                    "documented",
                ),
                (
                    "C2-global-store-completion-then-barrier-then-load-visible-inside-the-workgroup-at-any-scope",
                    "documented",
                ),
                ("C3-unordered-cross-wave-conflict-refused", "documented"),
                (
                    "C4-workgroup-and-named-barriers-1-2-join-once-init-before-use",
                    "conservative_policy",
                ),
                (
                    "C5-every-workgroup-of-the-grid-analysed-composed-only-when-isolated",
                    "conservative_policy",
                ),
            ]);
            "ptx_verifier comms/1 profile (communicating waves of one workgroup, isolated grids)"
        }
        Profile::Litmus | Profile::Litmus2 => {
            rules = vec![
                ("D1-sched-mode0-hardware-interlocks", "documented"),
                ("D4-kmcnt-kernarg-readiness", "documented"),
                ("D5-global-prefetch-scope-se-xcnt-no-data", "documented"),
                ("B1-raw-buffer-descriptor-fields", "documented"),
                (
                    "B1-swizzle-stride-scale-unbound-refused",
                    "conservative_policy",
                ),
                ("B2-linear-raw-offset-no-index-dword-aligned", "documented"),
                ("B3-per-dword-range-check-oob-select", "documented"),
                (
                    "B3-oob-select-stride-applies-to-raw-buffers",
                    "documented_interpretation",
                ),
                ("B4-oob-load-dword-zero-store-dropped", "documented"),
                (
                    "B5-unclamped-footprint-inside-one-object",
                    "conservative_policy",
                ),
                (
                    "B6-range-check-graph-record-recomputed",
                    "conservative_policy",
                ),
                ("A1-async-gvs-global-and-lds-full-offset", "documented"),
                (
                    "A2-async-lds-offset-at-or-above-64k-refused",
                    "conservative_policy",
                ),
                (
                    "A3-async-aligned-in-allocation-footprints",
                    "conservative_policy",
                ),
                ("A4-async-asynccnt-completion", "documented"),
            ];
            if profile == Profile::Litmus2 {
                rules.extend([
                    (
                        "J1-one-workgroup-1-or-2-waves-allocated-named-barriers",
                        "documented",
                    ),
                    ("J2-named-barrier-init-join-signal-wait", "documented"),
                    (
                        "J3-join-with-unconsumed-completion-refused",
                        "conservative_policy",
                    ),
                    ("J4-wait-idle-all-counters-zero", "documented"),
                    (
                        "J5-state-priv-after-wait-idle-named-id-and-clear-complete-only",
                        "conservative_policy",
                    ),
                    ("J6-and-with-defined-mask-exact", "documented"),
                    (
                        "J7-same-address-global-store-lanes-ordered",
                        "hardware_observation",
                    ),
                ]);
                "tests/gfx1250/CONTRACT.md: litmus/2 profile"
            } else {
                "tests/gfx1250/CONTRACT.md: litmus profile"
            }
        }
    };
    json!({
        "id": profile.id(),
        "dependency_profile": profile.dependency_profile(),
        "allowlist_sha256": sha256::hex(m1::canonical_allowlist_for(profile).as_bytes()),
        "pending_ops": profile.pending_ops(),
        "enabled_pending_ops": enabled,
        "rules": rules.iter().map(|(id, basis)| json!({
            "id": id, "basis": basis, "source": source,
        })).collect::<Vec<_>>(),
    })
}

/// Source line and text of an event's instruction.
struct SourceIndex<'a> {
    source: &'a str,
    line_starts: Vec<usize>,
}

impl<'a> SourceIndex<'a> {
    fn new(source: &'a str) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(source.match_indices('\n').map(|(i, _)| i + 1));
        Self {
            source,
            line_starts,
        }
    }
    fn line(&self, offset: usize) -> usize {
        self.line_starts.partition_point(|&start| start <= offset)
    }
    fn event(&self, graph: &EventGraph, id: EventId) -> Value {
        let pc = graph.events().get(id).and_then(|event| event.pc);
        let span = pc.and_then(|pc| graph.provenance().instructions.get(pc.0 as usize)?.span);
        let mut event = json!({
            "id": id.0,
            "line": span.map_or(0, |(start, _)| self.line(start)),
            "text": span.and_then(|(s, e)| self.source.get(s..e)).unwrap_or_default().trim(),
        });
        if let Some(pc) = pc {
            event["pc"] = json!(pc.0);
        }
        event
    }
}

fn register_name(register: AmdRegister) -> String {
    match (register.file, register.index) {
        (AmdRegisterFile::Scalar, 106) => "vcc_lo".into(),
        (AmdRegisterFile::Scalar, 107) => "vcc_hi".into(),
        (AmdRegisterFile::Scalar, 124) => "m0".into(),
        (AmdRegisterFile::Scalar, 126) => "exec_lo".into(),
        (AmdRegisterFile::Scalar, 127) => "exec_hi".into(),
        (AmdRegisterFile::Scalar, index) => format!("s{index}"),
        (AmdRegisterFile::Vector, index) => format!("v{index}"),
        (AmdRegisterFile::Accumulator, index) => format!("a{index}"),
        (AmdRegisterFile::Temporary, index) => format!("ttmp{index}"),
    }
}

fn hazard_witness(index: &SourceIndex, graph: &EventGraph, hazard: &AmdRegisterHazard) -> Value {
    use AmdRegisterHazardKind::*;
    let kind = match hazard.kind {
        ReadBeforeReady
        | ValuReadBeforeReady
        | ValuScalarReadBeforeReady
        | MatrixReadBeforeReady
        | MatrixOperandBeforeReady
        | SaluM0ReadBeforeReady => "READ_BEFORE_READY",
        SourceReuse | MatrixSourceReuse => "WRITE_BEFORE_SOURCE_RELEASED",
        _ => "OTHER",
    };
    let mut witness = json!({
        "kind": kind,
        "events": [index.event(graph, hazard.issue), index.event(graph, hazard.use_event)],
        "register": register_name(hazard.register),
        "detail": format!("{:?}", hazard.kind),
    });
    if hazard.register.file == AmdRegisterFile::Vector && hazard.register.lanes != 0 {
        witness["lane"] = json!(hazard.register.lanes.trailing_zeros());
    }
    witness
}

fn check(
    id: &str,
    status: &str,
    certified: bool,
    witnesses: Vec<Value>,
    unresolved: Vec<String>,
) -> Value {
    let mut check = json!({
        "id": id,
        "status": status,
        "witnesses": witnesses,
        "unresolved": unresolved,
    });
    if status == "FAIL" {
        check["certified"] = json!(certified);
    }
    check
}

fn not_evaluated(profile: Profile, reason: &str) -> Vec<Value> {
    check_ids(profile)
        .iter()
        .map(|id| check(id, "UNKNOWN", false, vec![], vec![reason.to_owned()]))
        .collect()
}

/// Everything derived from one completed or aborted execution.
struct Evaluation {
    litmus: Option<Value>,
    execution: Value,
    checks: Vec<Value>,
    memory_events: Value,
    outputs: Value,
    assumptions: Vec<Value>,
    mode: Option<Value>,
    class: ResultClass,
    reasons: Vec<String>,
}

fn empty_memory_events() -> Value {
    json!({
        "kernarg_loads": [],
        "loads": [],
        "stores": {"instructions": 0, "bytes_in_out": 0, "bytes_outside_out": 0},
        "prefetches": [],
    })
}

fn unavailable_outputs(words: u64) -> Value {
    json!({
        "contract": "raw-u32",
        "object": "out",
        "words": words,
        "status": "NOT_AVAILABLE",
        "values": vec![Value::Null; words as usize],
        "undefined": [],
    })
}

fn evaluate(
    request: &Request,
    source: &str,
    entry_mode: [Option<u8>; 2],
    verification: &SourceKernelVerification,
) -> Evaluation {
    let graph = &verification.analysis.graph;
    let index = SourceIndex::new(source);
    let mut reasons = Vec::new();
    let mut class = ResultClass::Pass;
    fn raise(class: &mut ResultClass, c: ResultClass, reason: String, reasons: &mut Vec<String>) {
        *class = (*class).max(c);
        reasons.push(reason);
    }

    // Execution.
    let budget = verification.analysis.stats.instructions >= request.max_instructions;
    let (exec_status, abort) = match graph.terminal() {
        Terminal::Completed => ("COMPLETE", Value::Null),
        Terminal::Stuck { .. } => ("ABORTED", json!({"reason": "execution is stuck"})),
        Terminal::Aborted {
            pc, reason, thread, ..
        } => {
            let mut abort = json!({"reason": reason});
            if let Some(pc) = pc {
                abort["pc"] = json!(pc.0);
                if let Some(span) = graph
                    .provenance()
                    .instructions
                    .get(pc.0 as usize)
                    .and_then(|i| i.span)
                {
                    abort["line"] = json!(index.line(span.0));
                }
            }
            abort["lane"] = match thread {
                Some(thread) if !budget => json!(thread.0 % 32),
                _ => Value::Null,
            };
            (
                if budget {
                    "BUDGET_EXHAUSTED"
                } else {
                    "ABORTED"
                },
                abort,
            )
        }
    };
    let ended = |wave: u32| {
        graph.amd_events().iter().any(|record| {
            record.wave == wave
                && matches!(record.kind, AmdEventKind::Instruction { .. })
                && graph.event(record.event).pc.is_some_and(|pc| {
                    graph.provenance().instructions[pc.0 as usize].kind == "s_endpgm"
                })
        })
    };
    let waves: Vec<_> = (0..request.block.div_ceil(32))
        .map(|wave| json!({"wave": wave, "terminated_by": if ended(wave) { "s_endpgm" } else { "none" }}))
        .collect();
    let execution = json!({
        "status": exec_status,
        "instructions_executed": verification.analysis.stats.instructions,
        "waves": waves,
        "abort": abort,
    });
    let complete = exec_status == "COMPLETE";
    if budget {
        raise(
            &mut class,
            ResultClass::Timeout,
            "BUDGET_EXHAUSTED".into(),
            &mut reasons,
        );
    } else if !complete {
        raise(
            &mut class,
            ResultClass::UnknownUnsupported,
            format!(
                "EXECUTION_ABORTED: {}",
                abort["reason"].as_str().unwrap_or_default()
            ),
            &mut reasons,
        );
    }

    // Graph checks.
    let amd = verification.graph.amd.as_ref();
    let unresolved = |reasons: &[volta_graph_check::Unresolved]| {
        reasons.iter().map(|u| u.reason.clone()).collect::<Vec<_>>()
    };
    let mut checks = Vec::new();
    let trace_pass = complete && amd.is_some_and(|a| a.trace.verdict() == Verdict::Pass);
    let trace_unresolved = amd.map_or_else(
        || vec!["no native gfx1250 graph report".to_owned()],
        |a| unresolved(a.trace.unresolved()),
    );
    let incomplete = |mut list: Vec<String>| {
        if !complete {
            list.push(format!("execution {exec_status}"));
        }
        list
    };
    checks.push(if trace_pass {
        check("trace_completeness", "PASS", false, vec![], vec![])
    } else {
        check(
            "trace_completeness",
            "UNKNOWN",
            false,
            vec![],
            incomplete(trace_unresolved.clone()),
        )
    });
    checks.push(match amd.map(|a| &a.wait_readiness) {
        Some(volta_graph_check::PropertyResult::Fail(hazards)) => check(
            "wait_readiness",
            "FAIL",
            true,
            hazards
                .iter()
                .map(|h| hazard_witness(&index, graph, h))
                .collect(),
            vec![],
        ),
        Some(result) if complete && result.verdict() == Verdict::Pass => {
            check("wait_readiness", "PASS", false, vec![], vec![])
        }
        Some(result) => check(
            "wait_readiness",
            "UNKNOWN",
            false,
            vec![],
            incomplete(unresolved(result.unresolved())),
        ),
        None => check(
            "wait_readiness",
            "UNKNOWN",
            false,
            vec![],
            trace_unresolved.clone(),
        ),
    });
    let has_barrier = graph.amd_events().iter().any(|record| {
        matches!(
            record.kind,
            AmdEventKind::SplitBarrierSignal { .. }
                | AmdEventKind::SplitBarrierWait { .. }
                | AmdEventKind::NamedBarrierInit { .. }
                | AmdEventKind::NamedBarrierJoin { .. }
                | AmdEventKind::NamedBarrierSignal { .. }
                | AmdEventKind::NamedBarrierWait { .. }
                | AmdEventKind::NamedBarrierLeave { .. }
                | AmdEventKind::BarrierArrive { .. }
        )
    });
    checks.push(match amd.map(|a| &a.barrier) {
        Some(volta_graph_check::PropertyResult::Fail(violations)) => check(
            "barrier_generations",
            "FAIL",
            true,
            violations
                .iter()
                .map(|v| json!({"kind": "OTHER", "events": [index.event(graph, v.event)], "detail": v.reason}))
                .collect(),
            vec![],
        ),
        Some(result) if complete && result.verdict() == Verdict::Pass && !has_barrier => {
            check("barrier_generations", "NOT_APPLICABLE", false, vec![], vec![])
        }
        Some(result) if complete && result.verdict() == Verdict::Pass => {
            check("barrier_generations", "PASS", false, vec![], vec![])
        }
        Some(result) => check("barrier_generations", "UNKNOWN", false, vec![], incomplete(unresolved(result.unresolved()))),
        None => check("barrier_generations", "UNKNOWN", false, vec![], trace_unresolved.clone()),
    });
    checks.push(match amd.map(|a| &a.memory_conflict) {
        Some(volta_graph_check::PropertyResult::Fail(conflicts)) => check(
            "memory_conflicts",
            "FAIL",
            true,
            conflicts
                .iter()
                .map(|c| {
                    json!({
                        "kind": "MEMORY_CONFLICT",
                        "events": [index.event(graph, c.earlier), index.event(graph, c.later)],
                        "address": format!("{:#x}", c.location.addr),
                        "detail": format!("{:?}", c.kind),
                    })
                })
                .collect(),
            vec![],
        ),
        Some(result) if complete && result.verdict() == Verdict::Pass => {
            check("memory_conflicts", "PASS", false, vec![], vec![])
        }
        Some(result) => check(
            "memory_conflicts",
            "UNKNOWN",
            false,
            vec![],
            incomplete(unresolved(result.unresolved())),
        ),
        None => check(
            "memory_conflicts",
            "UNKNOWN",
            false,
            vec![],
            trace_unresolved.clone(),
        ),
    });
    // Bounds: a certified pre-issue fault, or a complete trace whose every
    // global effect the checker validated against the recorded layout.
    let bounds = verification.global_bounds.as_ref().map(|c| c.violation());
    let bounds_witness = |v: &volta_graph_check::amdgpu::global_bounds::GlobalBoundsViolation| {
        let prefetch = matches!(v.access, AmdGlobalAccess::Prefetch(_));
        let mut witness = json!({
            "events": [index.event(graph, v.event)],
            "lane": v.lane,
            "address": format!("{:#x}", v.address),
            "bytes": v.bytes,
            "detail": format!("{:?} by lanes {:?}", v.access, v.lanes),
        });
        match &v.region {
            GlobalBoundsRegion::Guard { name, offset } => {
                witness["kind"] = json!("GUARD_ACCESS");
                witness["guard"] = json!(name);
                witness["word_index"] = json!(offset / 4);
            }
            GlobalBoundsRegion::OutsideObjects => {
                witness["kind"] = json!(if prefetch {
                    "NON_SPECULATIVE_PREFETCH_OUTSIDE_OBJECT"
                } else {
                    "OUT_OF_OBJECT_ACCESS"
                });
            }
        }
        witness
    };
    let prefetch_records: Vec<_> = graph
        .amd_events()
        .iter()
        .filter_map(|record| match &record.kind {
            AmdEventKind::PrefetchFootprint {
                issue,
                scope,
                lanes,
            } => Some((record, *issue, *scope, lanes)),
            _ => None,
        })
        .collect();
    for (id, applies) in [("object_bounds", false), ("prefetch_policy", true)] {
        checks.push(match bounds {
            Some(v) if matches!(v.access, AmdGlobalAccess::Prefetch(_)) == applies => {
                check(id, "FAIL", true, vec![bounds_witness(v)], vec![])
            }
            _ if applies && trace_pass && prefetch_records.is_empty() => {
                check(id, "NOT_APPLICABLE", false, vec![], vec![])
            }
            _ if trace_pass => check(id, "PASS", false, vec![], vec![]),
            _ => check(
                id,
                "UNKNOWN",
                false,
                vec![],
                incomplete(trace_unresolved.clone()),
            ),
        });
    }
    checks.push(if trace_pass {
        check("store_completion", "PASS", false, vec![], vec![])
    } else {
        check(
            "store_completion",
            "UNKNOWN",
            false,
            vec![],
            incomplete(trace_unresolved.clone()),
        )
    });
    let mode = request.profile.mode().then(|| {
        let (mode_check, mode) = mode_order(&index, graph, entry_mode, complete, trace_pass);
        checks.push(mode_check);
        mode
    });
    if matches!(
        request.profile,
        Profile::Wmma | Profile::Multiwave | Profile::MwPred
    ) {
        checks.push(wmma_hazards(&index, graph, complete, trace_pass, false));
    } else if request.profile.wmma_types() {
        checks.push(wmma_hazards(&index, graph, complete, trace_pass, true));
    }
    for check in &checks {
        let id = check["id"].as_str().unwrap();
        match check["status"].as_str().unwrap() {
            "FAIL" if check["certified"] == json!(true) => raise(
                &mut class,
                ResultClass::Reject,
                format!("CHECK_FAILED:{id}"),
                &mut reasons,
            ),
            "FAIL" | "UNKNOWN" => raise(
                &mut class,
                ResultClass::UnknownUnsupported,
                format!("CHECK_UNKNOWN:{id}"),
                &mut reasons,
            ),
            _ => {}
        }
    }

    // Memory events.
    let mut kernarg_loads = Vec::new();
    let mut loads = Vec::new();
    let mut store_instructions = 0u64;
    for record in graph.amd_events() {
        if let AmdEventKind::MemoryIssue {
            class,
            destinations,
            ..
        } = &record.kind
        {
            let event = index.event(graph, record.event);
            match class {
                AmdMemoryClass::ScalarLoad => kernarg_loads.push(json!({
                    "line": event["line"],
                    "text": event["text"],
                    "destinations": destinations.iter().map(|r| register_name(*r)).collect::<Vec<_>>(),
                })),
                AmdMemoryClass::VectorStore => store_instructions += 1,
                AmdMemoryClass::Prefetch => {}
                _ => loads.push(json!({"line": event["line"], "text": event["text"], "class": format!("{class:?}")})),
            }
        }
    }
    let out_end = request.out_base + 4 * request.out_words;
    let (mut in_out, mut outside_out) = (0u64, 0u64);
    for (_, event) in graph.events().iter() {
        if let EventKind::Access(access) = event.kind {
            let access = graph.access(access);
            if access.space != Space::Global || access.mode != volta_graph::AccessMode::Write {
                continue;
            }
            if let volta_graph::Footprint::Bytes { addr, width } = access.footprint {
                for byte in addr..addr + width as u64 {
                    if (request.out_base..out_end).contains(&byte) {
                        in_out += 1;
                    } else {
                        outside_out += 1;
                    }
                }
            }
        }
    }
    let layout = graph.global_region_layout();
    let prefetches: Vec<_> = prefetch_records
        .iter()
        .map(|(record, issue, scope, lanes)| {
            let lo = lanes.iter().map(|l| l.1).min().unwrap_or(0);
            let hi = lanes.iter().map(|l| l.1).max().unwrap_or(0);
            let objects_hit: Vec<_> = layout
                .iter()
                .filter(|r| {
                    lanes
                        .iter()
                        .any(|&(_, a)| r.base <= a && a < r.base + r.size)
                })
                .map(|r| r.name.clone())
                .collect();
            let xcnt = graph.amd_events().iter().any(|r| {
                r.event == *issue
                    && matches!(&r.kind, AmdEventKind::MemoryIssue { counters, .. }
                        if counters.iter().any(|(c, _)| *c == volta_graph::AmdCounter::X))
            });
            json!({
                "line": index.event(graph, record.event)["line"],
                "scope": scope.name(),
                "lo": format!("{lo:#x}"),
                "hi": format!("{hi:#x}"),
                "objects_hit": objects_hit,
                "xcnt_charged": xcnt,
            })
        })
        .collect();
    let memory_events = json!({
        "kernarg_loads": kernarg_loads,
        "loads": loads,
        "stores": {"instructions": store_instructions, "bytes_in_out": in_out, "bytes_outside_out": outside_out},
        "prefetches": prefetches,
    });

    // Raw output words: `out`, or the litmus objects' images in request order.
    let object_names: Vec<String> = if request.profile.litmus() {
        request.objects.iter().map(|o| o.name.clone()).collect()
    } else {
        vec!["out".into()]
    };
    let raw: Option<Vec<RawWord>> = verification.analysis.raw_outputs.as_ref().and_then(|raw| {
        object_names
            .iter()
            .map(|name| {
                raw.iter()
                    .find(|(object, _)| object == name)
                    .map(|(_, words)| words.clone())
            })
            .collect::<Option<Vec<_>>>()
            .map(|parts| parts.concat())
    });
    let raw = raw.as_ref();
    let outputs = match raw {
        Some(words) if complete => {
            let values: Vec<_> = words
                .iter()
                .map(|w| match w {
                    RawWord::Exact(bits) => json!(bits),
                    RawWord::Undefined(_) => Value::Null,
                })
                .collect();
            let undefined: Vec<_> = words
                .iter()
                .enumerate()
                .filter_map(|(i, w)| match w {
                    RawWord::Undefined(reason) => {
                        Some(json!({"index": i, "reason": reason.name()}))
                    }
                    RawWord::Exact(_) => None,
                })
                .collect();
            // Multiwave: declared opaque/unwritten words are expected, every
            // other word must be exact; a declared word that is exact or has
            // another refusal makes the outputs unexpected.
            let declared = |ranges: &[(u64, u64)], i: usize| {
                ranges
                    .iter()
                    .any(|&(first, count)| (first..first + count).contains(&(i as u64)))
            };
            let unexpected: Vec<_> = words
                .iter()
                .enumerate()
                .filter(|(i, w)| {
                    let expected = if declared(&request.opaque_words, *i) {
                        Some(RawWord::Undefined(
                            volta_amdgpu_analysis::RawWordRefusal::Opaque,
                        ))
                    } else if declared(&request.unwritten_words, *i) {
                        Some(RawWord::Undefined(
                            volta_amdgpu_analysis::RawWordRefusal::NotWritten,
                        ))
                    } else {
                        None
                    };
                    match expected {
                        Some(expected) => **w != expected,
                        None => matches!(w, RawWord::Undefined(_)),
                    }
                })
                .map(|(i, _)| i)
                .collect();
            let status = if undefined.is_empty() {
                "COMPLETE"
            } else if unexpected.is_empty() && request.profile.multiwave() {
                "COMPLETE_AS_DECLARED"
            } else {
                "PARTIAL"
            };
            if !unexpected.is_empty() && !request.defer_outputs {
                raise(
                    &mut class,
                    ResultClass::UnknownUnsupported,
                    format!(
                        "OUTPUTS_PARTIAL: {} undefined or undeclared words",
                        unexpected.len()
                    ),
                    &mut reasons,
                );
            }
            if request.profile.litmus() {
                json!({
                    "contract": "raw-u32-image", "objects": object_names, "words": request.out_words,
                    "status": status, "values": values, "undefined": undefined,
                })
            } else if request.profile.multiwave() {
                json!({
                    "contract": "raw-u32-mw", "object": "out", "words": request.out_words,
                    "status": status, "values": values, "undefined": undefined,
                })
            } else {
                json!({
                    "contract": "raw-u32", "object": "out", "words": request.out_words,
                    "status": status, "values": values, "undefined": undefined,
                })
            }
        }
        _ => {
            if class == ResultClass::Pass {
                raise(
                    &mut class,
                    ResultClass::UnknownUnsupported,
                    "OUTPUTS_NOT_AVAILABLE".into(),
                    &mut reasons,
                );
            }
            unavailable_outputs_for(request)
        }
    };

    let mut assumptions: Vec<_> = graph
        .provenance()
        .assumptions
        .iter()
        .map(|a| json!({"id": format!("{a:?}"), "text": a.describe()}))
        .collect();
    if request.profile.litmus() {
        assumptions.extend(litmus_premises(request.profile));
    } else if request.profile == Profile::Comms {
        assumptions.extend(comms_premises());
    } else if request.profile.multiwave() {
        assumptions.extend(mw_premises());
    } else {
        assumptions.extend(launch_premises());
    }
    if request.profile.mode() {
        assumptions.extend(m2_premises());
    }
    if matches!(
        request.profile,
        Profile::Wmma | Profile::Multiwave | Profile::MwPred
    ) {
        assumptions.extend(wmma_premises());
    }
    if request.profile == Profile::MwPred {
        assumptions.extend(mwpred_premises());
    } else if request.profile == Profile::Wmma2 {
        assumptions.extend(wmma2_premises());
    } else if request.profile == Profile::Wmma3 {
        assumptions.extend(wmma3_premises());
    } else if request.profile.wmma_sparse16() {
        assumptions.extend(wmma4_premises());
    }
    let litmus = (request.profile.litmus()).then(|| litmus_record(&index, graph));
    Evaluation {
        execution,
        checks,
        memory_events,
        outputs,
        assumptions,
        mode,
        litmus,
        class,
        reasons,
    }
}

/// Litmus report record: every buffer issue's range check (lanes summarised
/// by their distinct in-range masks) and every async load to LDS.
fn litmus_record(index: &SourceIndex, graph: &EventGraph) -> Value {
    let mut buffer = Vec::new();
    for record in graph.amd_events() {
        if let AmdEventKind::BufferRangeCheck { issue, check } = &record.kind {
            let event = index.event(graph, *issue);
            let full = (1u8 << check.dwords) - 1;
            let mut masks: Vec<u8> = check.lanes.iter().map(|l| l.in_range).collect();
            masks.sort_unstable();
            masks.dedup();
            buffer.push(json!({
                "line": event["line"],
                "text": event["text"],
                "lanes": check.lanes.len(),
                "dwords": check.dwords,
                "num_records": check.num_records,
                "stride": check.stride,
                "oob_select": check.oob_select,
                "in_range_masks": masks,
                "dwords_in_range": check.lanes.iter().map(|l| u64::from(l.in_range.count_ones())).sum::<u64>(),
                "dwords_out_of_range": check.lanes.iter().map(|l| u64::from((full & !l.in_range).count_ones())).sum::<u64>(),
            }));
        }
    }
    let async_lds: Vec<_> = graph
        .amd_events()
        .iter()
        .filter(|r| {
            matches!(
                r.kind,
                AmdEventKind::MemoryIssue {
                    class: AmdMemoryClass::AsyncLoadToLds,
                    ..
                }
            )
        })
        .map(|r| {
            let event = index.event(graph, r.event);
            json!({"line": event["line"], "text": event["text"]})
        })
        .collect();
    json!({"buffer": buffer, "async_lds": async_lds})
}

fn unavailable_outputs_for(request: &Request) -> Value {
    let mut outputs = unavailable_outputs(request.out_words);
    if request.profile.litmus() {
        outputs = litmus_unavailable(request);
    } else if request.profile.multiwave() {
        outputs["contract"] = json!("raw-u32-mw");
    }
    outputs
}

fn litmus_unavailable(request: &Request) -> Value {
    json!({
        "contract": "raw-u32-image",
        "objects": request.objects.iter().map(|o| o.name.clone()).collect::<Vec<_>>(),
        "words": request.out_words,
        "status": "NOT_AVAILABLE",
        "values": vec![Value::Null; request.out_words as usize],
        "undefined": [],
    })
}

fn litmus_premises(profile: Profile) -> Vec<Value> {
    let launch = if profile == Profile::Litmus2 {
        (
            "A-LAUNCH",
            "one workgroup (0,0,0) of 32 or 64 workitems (one or two wave32 waves), unclustered, no dynamic LDS; named barriers as the descriptor allocates",
        )
    } else {
        (
            "A-LAUNCH",
            "one workgroup (0,0,0) of 32 workitems, one wave32, unclustered, no dynamic LDS",
        )
    };
    let mut premises = vec![
        launch,
        (
            "A-LANE-ID",
            "v0 holds the packed work-item id; X = lane in bits 9:0, Y = Z = 0",
        ),
        ("A-EXEC-FULL", "EXEC = 0xffffffff at entry"),
        (
            "A-ENTRY-UNDEFINED",
            "SCC, VCC, LDS and every register other than s[0:1] and v0 are undefined at entry; reading them is unsupported",
        ),
        (
            "A-KERNARG",
            "kernarg offset 0 holds the 64-bit address of the first object",
        ),
        (
            "A-INIT-LAUNCHER",
            "every object starts with the exact words the launcher writes (fill, then the template pattern); guards are forbidden regions",
        ),
        (
            "A-SCHED-MODE0",
            "SCHED_MODE 0 is never written; hardware interlocks ALU dependencies",
        ),
        (
            "B3-OOB-SELECT-RAW",
            "OOB_SELECT=1 applies the stride bound to raw (index-free) buffers; the raw-buffer sentence that the stride field is ignored is read as concerning addressing only",
        ),
        (
            "L-PER-DWORD",
            "each DWORD of a B64/B96/B128 access is range checked as a 4-byte payload at its own offset",
        ),
        (
            "P-ASYNC-LDS-64K",
            "async LDS displacements at or above 64 KiB are outside the profile and refused",
        ),
    ];
    if profile == Profile::Litmus2 {
        premises.extend([
            ("P-BJ-CONSUMED", "a named barrier JOIN or a STATE_PRIV read is analysed only after the wave has waited on every completion of its joined barrier; earlier ones are refused, since the order of a completion report against a later JOIN is not documented"),
        ]);
    }
    premises
        .iter()
        .map(|(id, text)| json!({"id": id, "text": text}))
        .collect()
}

/// Entry MODE fields from the descriptor's 2-bit controls, as the interpreter
/// derives them; `None` when a control is missing or out of range.
fn descriptor_mode(kernel: Option<&frontend::Kernel>) -> [Option<u8>; 2] {
    let get = |key: &str| {
        kernel
            .and_then(|k| k.descriptor.get(key))
            .filter(|value| *value <= 3)
    };
    let field = |low: &str, high: &str| Some((get(low)? | get(high)? << 2) as u8);
    [
        field("float_round_mode_32", "float_round_mode_16_64"),
        field("float_denorm_mode_32", "float_denorm_mode_16_64"),
    ]
}

fn mode_fields(fields: [Option<u8>; 2]) -> Value {
    json!({"fp_round": fields[0], "fp_denorm": fields[1]})
}

/// The m2 `mode_order` check and the MODE access record. Every executed MODE
/// instruction must have exactly one access record; the independent order
/// check then decides UNKNOWN (unresolved) or PASS.
fn mode_order(
    index: &SourceIndex,
    graph: &EventGraph,
    entry: [Option<u8>; 2],
    complete: bool,
    trace_pass: bool,
) -> (Value, Value) {
    let mut accesses = Vec::new();
    let mut fields = entry;
    let mut executed = 0usize;
    for record in graph.amd_events() {
        match &record.kind {
            AmdEventKind::Instruction { .. } => {
                let event = index.event(graph, record.event);
                let text = event["text"].as_str().unwrap_or_default();
                // The multiwave HW_ID1 evidence read is not a MODE access.
                if m1::MODE_OPS.contains(&text.split_whitespace().next().unwrap_or_default())
                    && !text.contains(m1::EVIDENCE_HWREG)
                {
                    executed += 1;
                }
            }
            AmdEventKind::ModeAccess {
                op,
                offset,
                width,
                value,
            } => {
                let event = index.event(graph, record.event);
                if op.writes() {
                    fields[usize::from(offset / 4)] = Some(*value as u8);
                }
                accesses.push(json!({
                    "line": event["line"],
                    "text": event["text"],
                    "op": op.name(),
                    "offset": offset,
                    "width": width,
                    "access": if op.writes() { "write" } else { "read" },
                    "value": value,
                }));
            }
            _ => {}
        }
    }
    let report = check_mode_order(graph);
    let mut unresolved: Vec<String> = report.malformed.clone();
    if executed != report.accesses {
        unresolved.push(format!(
            "{executed} MODE instructions executed but {} MODE access records",
            report.accesses
        ));
    }
    unresolved.extend(report.unresolved.iter().map(|f| {
        let ModeOrderKind::UnresolvedWriteAfterRead = f.kind;
        format!(
            "s_getreg_b32 of MODE then {} with no SA_SDST==0 wait: ordering outside the profile",
            f.write_op.name()
        ) + &format!(
            " (lines {} and {})",
            index.event(graph, f.read)["line"],
            index.event(graph, f.write)["line"]
        )
    }));
    let check_value = if !complete || !trace_pass || !unresolved.is_empty() {
        if !complete {
            unresolved.push("execution incomplete".into());
        } else if !trace_pass {
            unresolved.push("trace incomplete".into());
        }
        check("mode_order", "UNKNOWN", false, vec![], unresolved)
    } else if report.accesses == 0 {
        check("mode_order", "NOT_APPLICABLE", false, vec![], vec![])
    } else {
        check("mode_order", "PASS", false, vec![], vec![])
    };
    let mode = json!({
        "entry": dict_with_source(mode_fields(entry)),
        "accesses": accesses,
        "final": mode_fields(fields),
    });
    (check_value, mode)
}

fn dict_with_source(mut fields: Value) -> Value {
    fields["source"] = json!("kernel descriptor float_round_mode_* / float_denorm_mode_*");
    fields
}

fn launch_premises() -> Vec<Value> {
    [
        ("A-LAUNCH", "one workgroup (0,0,0) of 32 workitems, one wave32, unclustered, no dynamic LDS"),
        ("A-LANE-ID", "v0 holds the packed work-item id; X = lane in bits 9:0, Y = Z = 0"),
        ("A-EXEC-FULL", "EXEC = 0xffffffff at entry"),
        ("A-ENTRY-UNDEFINED", "SCC, VCC and every register other than s[0:1] and v0 are undefined at entry; reading them is unsupported"),
        ("A-KERNARG", "kernarg offset 0 holds the 64-bit address of out"),
        ("A-GUARDS", "pre and post guards are forbidden regions of the backing allocation; out is an uninitialized output object"),
        ("A-SCHED-MODE0", "SCHED_MODE 0 is never written; hardware interlocks ALU dependencies"),
    ]
    .iter()
    .map(|(id, text)| json!({"id": id, "text": text}))
    .collect()
}

fn m2_premises() -> Vec<Value> {
    [
        ("A-MODE-ENTRY", "MODE FP_ROUND/FP_DENORM at entry come from the kernel descriptor; MODE[31:8] is never read or written"),
        ("A-MODE-NO-FP", "no instruction of the profile computes a MODE-dependent result; MODE support does not establish FP rounding or denormal behaviour"),
        ("A-MODE-ORDER", "MODE writes are ordered before later MODE reads and writes; the profile certifies a MODE read followed by a later MODE write only when s_wait_alu with SA_SDST 0 separates them"),
        ("A-SETREG-IMM32-4BIT", "the profile admits s_setreg_imm32_b32 only on FP_ROUND or FP_DENORM with an immediate of at most 4 bits; other MODE bits, including VGPR_MSB, are never written and VGPR_MSB stays 0"),
    ]
    .iter()
    .map(|(id, text)| json!({"id": id, "text": text}))
    .collect()
}

fn mw_premises() -> Vec<Value> {
    [
        ("A-LAUNCH", "one workgroup (0,0,0) of 32*W workitems, W = 2..8 wave32 waves, unclustered, no dynamic LDS"),
        ("A-LANE-ID", "v0 holds the packed work-item id; X = 32*wave + lane in bits 9:0, Y = Z = 0"),
        ("A-EXEC-FULL", "EXEC = 0xffffffff at entry in every wave"),
        ("A-ENTRY-UNDEFINED", "SCC, VCC and every register other than s[0:1] and v0 are undefined at entry; reading them is unsupported"),
        ("A-KERNARG", "kernarg offset 0 holds the 64-bit address of out"),
        ("A-GUARDS", "pre and post guards are forbidden regions of the backing allocation; out is an uninitialized output object"),
        ("A-SCHED-MODE0", "SCHED_MODE 0 is never written; hardware interlocks ALU dependencies"),
        ("A-REPLAY-MODE0", "REPLAY_MODE stays 0 (single-group XNACK replay, its wave-start value): MODE writes outside FP_ROUND/FP_DENORM are refused"),
        ("A-EVIDENCE-OPAQUE", "shader-cycle and HW_ID1 values are opaque; the declared evidence words carry them and no value is claimed"),
        ("A-NO-COMMUNICATION", "waves do not communicate; any cross-wave memory conflict is a failure"),
    ]
    .iter()
    .map(|(id, text)| json!({"id": id, "text": text}))
    .collect()
}

fn comms_premises() -> Vec<Value> {
    let mut premises: Vec<Value> = mw_premises()
        .into_iter()
        .filter(|p| p["id"] != "A-NO-COMMUNICATION" && p["id"] != "A-LAUNCH")
        .collect();
    premises.extend(
        [
            ("A-LAUNCH", "a grid of 1..64 unclustered workgroups of 32*W workitems (W = 2..8 wave32 waves), the declared static LDS allocation, no dynamic LDS; every workgroup (k, 0, 0) is analysed with its id in TTMP9"),
            ("A-ORDERED-COMMUNICATION", "waves of a workgroup communicate through LDS and GLOBAL memory only through transfers ordered by completion waits and barriers; an unordered cross-wave conflict is a failure"),
            ("A-WORKGROUP-ISOLATION", "workgroups do not communicate: composition is claimed only when their written bytes are pairwise disjoint and no workgroup reads bytes that another workgroup writes; otherwise the result is not composed"),
        ]
        .iter()
        .map(|(id, text)| json!({"id": id, "text": text})),
    );
    premises
}

fn mwpred_premises() -> Vec<Value> {
    [
        ("A-GRID-EVERY-WORKGROUP", "every workgroup (k, 0, 0) of the grid is analysed with its workgroup id in TTMP9; the report composes their results"),
        ("A-WORKGROUP-ISOLATION", "workgroups do not communicate: composition is claimed only when their written bytes are pairwise disjoint and no workgroup reads bytes that another workgroup writes; otherwise the result is not composed"),
        ("A-SE-ID-OPAQUE", "the SE/XCC id message return is an opaque value, drained by the next instruction; no value is claimed"),
    ]
    .iter()
    .map(|(id, text)| json!({"id": id, "text": text}))
    .collect()
}

fn wmma_premises() -> Vec<Value> {
    [
        ("A-WMMA-EXACT-REGIME", "a BF16 WMMA result element has exact bits only when its inputs are normal or zero and every partial sum of its terms is an exactly representable normal binary32 value; the device accumulator is assumed to keep at least binary32 precision for such sums"),
        ("A-WMMA-COEXEC", "VALU co-execution is enabled (default); the co-execution-disabled slot counts are not used"),
        ("A-WMMA-UNLISTED-POLICY", "WMMA to WMMA dependencies not listed in the documented table (D as C, WAW, WAR) need 5 slots by conservative policy; fewer is unresolved, not a documented failure"),
        ("A-NO-PREDECESSOR", "results do not depend on kernels that ran before; earlier matrix work on the device is outside this result"),
    ]
    .iter()
    .map(|(id, text)| json!({"id": id, "text": text}))
    .collect()
}

fn wmma2_premises() -> Vec<Value> {
    [
        ("A-WMMA-EXACT-REGIME", "an F32 WMMA result element (BF16, F16 or FP8 inputs) has exact bits only when its inputs are normal (in their own format) or zero and every partial sum of its terms is an exactly representable normal binary32 value; the device accumulator is assumed to keep at least binary32 precision for such sums"),
        ("A-WMMA-UNIFORM-8BIT", "FP8 and IU8 WMMA are evaluated only for uniform A and B tiles (every byte equal), so no 8-bit matrix layout is assumed; IU8 without NEG modifiers is unsigned, evaluated for bytes below 0x80 and results inside int32"),
        ("A-WMMA-COEXEC", "VALU co-execution is enabled (default); the co-execution-disabled slot counts are not used"),
        ("A-WMMA-UNLISTED-POLICY", "WMMA to WMMA dependencies not listed in the documented table (D as C, WAW, WAR) need the type's WMMA-to-WMMA count by conservative policy; fewer is unresolved, not a documented failure"),
        ("A-WMMA-EXEC0-SLOTS", "instructions issued with EXEC = 0 are never counted as WMMA spacing slots; a dependency that meets its count only by counting them is unresolved"),
        ("A-NO-PREDECESSOR", "results do not depend on kernels that ran before; earlier matrix work on the device is outside this result"),
    ]
    .iter()
    .map(|(id, text)| json!({"id": id, "text": text}))
    .collect()
}

fn wmma3_premises() -> Vec<Value> {
    let mut premises = wmma2_premises();
    let tail = premises.split_off(premises.len() - 1);
    premises.extend(
        [
            ("A-WMMA-UNIFORM-SMALL", "BF8 combinations, F8F6F4 and F4 32x16x128 WMMA (scaled or not) are evaluated only for uniform A and B tiles (one byte, nibble or 6-bit code everywhere) and uniform block-scale tuples, so no matrix or scale layout is assumed; elements normal or zero in their own format and scales normal powers of two give exact bits, other uniform values give reals"),
            ("A-WMMA-F6-PACKING", "a uniform FP6/BF6 tile is one 6-bit code packed contiguously over each lane's group of six registers (K = 0 in bits [5:0])"),
            ("A-WMMA-BLOCK-SCALE", "block scales multiply the per-block dot products before accumulation; with power-of-two scales and integer scaled products, per-block and per-product application give the same exact partial sums"),
            ("A-WMMA-VOP3PX2", "a block-scaled WMMA is one 16-byte instruction; its load-scale half is never a spacing slot"),
            ("A-WMMA-32X16-LAYOUT", "register i of a 32x16 WMMA D is register i of C plus the product term (same layout for C and D)"),
        ]
        .iter()
        .map(|(id, text)| json!({"id": id, "text": text})),
    );
    premises.extend(tail);
    premises
}

fn wmma4_premises() -> Vec<Value> {
    let mut premises = wmma3_premises();
    let tail = premises.split_off(premises.len() - 1);
    premises.extend(
        [
            ("A-WMMA-F16-OUTPUT-REGIME", "an F16 WMMA or SWMMAC result half has exact bits only when its inputs are normal or zero in their own format and every partial sum of its terms is an integer below 2^11 in magnitude (exact in F16 and in any wider accumulator); BF16 and BF16F32 results are never certified"),
            ("A-WMMA-16BIT-LAYOUT", "register i, half h of a 16-bit WMMA or SWMMAC D is register i, half h of C plus the product term (same packed layout for C and D)"),
            ("A-WMMA-SPARSE-B-HALVES", "a SWMMAC B operand's registers 0-7 hold the first half of K and registers 8-15 the second half; with uniform packed A and a uniform index whose pairs satisfy idx0 < idx1, each K half contributes K/4 products, whatever the index values"),
        ]
        .iter()
        .map(|(id, text)| json!({"id": id, "text": text})),
    );
    premises.extend(tail);
    premises
}

/// The WMMA spacing check: certified failures for the documented rules,
/// UNKNOWN for unlisted dependencies below the conservative policy (and,
/// under the types profile, for spacing met only by EXEC = 0 instructions).
fn wmma_hazards(
    index: &SourceIndex,
    graph: &EventGraph,
    complete: bool,
    trace_pass: bool,
    types: bool,
) -> Value {
    use volta_graph_check::amdgpu::wmma_hazards::{
        WmmaHazardFinding, WmmaHazardKind, check_wmma_hazards, check_wmma_hazards_types,
    };
    let report = if types {
        check_wmma_hazards_types(graph)
    } else {
        check_wmma_hazards(graph)
    };
    let witness = |f: &WmmaHazardFinding| {
        let detail = if f.kind == WmmaHazardKind::Exec0SlotsUnresolved {
            format!(
                "{:?}: {} slots, {} required, {} more issued with EXEC = 0",
                f.kind, f.slots, f.required, f.exec0
            )
        } else {
            format!("{:?}: {} slots, {} required", f.kind, f.slots, f.required)
        };
        let mut witness = json!({
            "kind": f.kind.name(),
            "events": [index.event(graph, f.first), index.event(graph, f.second)],
            "detail": detail,
        });
        if let Some(register) = f.register {
            witness["register"] = json!(register_name(register));
        }
        witness
    };
    let mut unresolved = report.malformed.clone();
    unresolved.extend(report.policy.iter().map(|f| {
        if f.kind == WmmaHazardKind::Exec0SlotsUnresolved {
            format!(
                "{} (lines {} and {}): {} slots, {} required; {} more issued with EXEC = 0 are not counted",
                f.kind.name(),
                index.event(graph, f.first)["line"],
                index.event(graph, f.second)["line"],
                f.slots,
                f.required,
                f.exec0
            )
        } else {
            format!(
                "{} (lines {} and {}): {} slots, policy requires {}",
                f.kind.name(),
                index.event(graph, f.first)["line"],
                index.event(graph, f.second)["line"],
                f.slots,
                f.required
            )
        }
    }));
    if !report.violations.is_empty() {
        check(
            "wmma_hazards",
            "FAIL",
            true,
            report.violations.iter().map(witness).collect(),
            unresolved,
        )
    } else if !complete || !trace_pass || !unresolved.is_empty() {
        if !complete {
            unresolved.push("execution incomplete".into());
        } else if !trace_pass {
            unresolved.push("trace incomplete".into());
        }
        check(
            "wmma_hazards",
            "UNKNOWN",
            false,
            report.policy.iter().map(witness).collect(),
            unresolved,
        )
    } else if report.wmmas == 0 {
        check("wmma_hazards", "NOT_APPLICABLE", false, vec![], vec![])
    } else {
        check("wmma_hazards", "PASS", false, vec![], vec![])
    }
}

pub struct Outcome {
    pub class: ResultClass,
    pub report: Value,
}

/// Evaluate one request. Every outcome, including invalid requests, yields a
/// schema-complete report.
pub fn run_request(request_path: &Path, backend: Result<Backend, String>) -> Outcome {
    let started = Instant::now();
    let request_bytes = std::fs::read(request_path);
    let request_sha256 = request_bytes
        .as_ref()
        .map(|bytes| sha256::hex(bytes))
        .unwrap_or_else(|_| "0".repeat(64));
    let mut report = json!({
        "schema": REPORT_SCHEMA,
        "request_sha256": request_sha256,
        "role": "cand",
        "profile": profile_block(Profile::M1Int, &[]),
        "verifier": verifier_identity(),
        "source": {"path": "", "sha256": "0".repeat(64), "matches_request": false},
        "object": {"sha256": "0".repeat(64)},
        "target": {"directive": "", "metadata_target": "", "normalized": "gfx1250", "features": [], "code_object_version": 6},
        "kernel": {"name": m1::ENTRY, "size_bytes": 0, "instructions": 0, "padding_after_size_bytes": 0, "descriptor": {}},
        "parser": {"status": "UNSUPPORTED", "diagnostics": [], "refused_ops": []},
        "execution": {"status": "NOT_RUN", "instructions_executed": 0, "waves": [], "abort": null},
        "checks": not_evaluated(Profile::M1Int, "not evaluated"),
        "memory_events": empty_memory_events(),
        "outputs": unavailable_outputs(32),
        "assumptions": launch_premises(),
        "result_class": "ERROR",
        "reasons": [],
        "elapsed_seconds": 0.0,
    });
    let class = evaluate_request(request_path, request_bytes, backend, &mut report);
    report["result_class"] = json!(class.name());
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    Outcome { class, report }
}

/// The profile whose report shape `report` already has.
fn report_profile(report: &Value) -> Profile {
    if report["schema"] == json!(M2_REPORT_SCHEMA) {
        Profile::M2Mode
    } else if report["schema"] == json!(LITMUS_REPORT_SCHEMA) {
        Profile::Litmus
    } else if report["schema"] == json!(LITMUS2_REPORT_SCHEMA) {
        Profile::Litmus2
    } else if report["schema"] == json!(WMMA_REPORT_SCHEMA) {
        Profile::Wmma
    } else if report["schema"] == json!(MW_REPORT_SCHEMA) {
        Profile::Multiwave
    } else if report["schema"] == json!(WMMA2_REPORT_SCHEMA) {
        Profile::Wmma2
    } else if report["schema"] == json!(WMMA3_REPORT_SCHEMA) {
        Profile::Wmma3
    } else if report["schema"] == json!(WMMA4_REPORT_SCHEMA) {
        Profile::Wmma4
    } else if report["schema"] == json!(MW2_REPORT_SCHEMA) {
        Profile::Multiwave2
    } else if report["schema"] == json!(MWPRED_REPORT_SCHEMA) {
        Profile::MwPred
    } else if report["schema"] == json!(COMMS_REPORT_SCHEMA) {
        Profile::Comms
    } else {
        Profile::M1Int
    }
}

fn error(report: &mut Value, reason: impl Into<String>) -> ResultClass {
    let reason = reason.into();
    report["reasons"]
        .as_array_mut()
        .unwrap()
        .push(json!(reason));
    report["checks"] = json!(not_evaluated(
        report_profile(report),
        &format!("not evaluated: {reason}")
    ));
    ResultClass::Error
}

/// An m2 report before evaluation: no MODE access is known yet.
fn empty_mode() -> Value {
    json!({
        "entry": dict_with_source(mode_fields([None, None])),
        "accesses": [],
        "final": mode_fields([None, None]),
    })
}

fn evaluate_request(
    request_path: &Path,
    request_bytes: std::io::Result<Vec<u8>>,
    backend: Result<Backend, String>,
    report: &mut Value,
) -> ResultClass {
    let bytes = match request_bytes {
        Ok(bytes) => bytes,
        Err(e) => return error(report, format!("INVALID_REQUEST: cannot read request: {e}")),
    };
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(e) => return error(report, format!("INVALID_REQUEST: {e}")),
    };
    if let Some(role) = value
        .get("role")
        .and_then(Value::as_str)
        .filter(|role| matches!(*role, "cand" | "twin" | "control"))
    {
        report["role"] = json!(role);
    }
    let profile = requested_profile(&value);
    if profile.mode() {
        report["schema"] = json!(report_schema(profile));
        report["profile"] = profile_block(profile, &[]);
        report["checks"] = json!(not_evaluated(profile, "not evaluated"));
        report["mode"] = empty_mode();
        if profile.multiwave() {
            report["assumptions"] = json!(mw_premises());
            report["outputs"]["contract"] = json!("raw-u32-mw");
        }
        report["assumptions"]
            .as_array_mut()
            .unwrap()
            .extend(m2_premises());
        if profile == Profile::Wmma2 {
            report["assumptions"]
                .as_array_mut()
                .unwrap()
                .extend(wmma2_premises());
        } else if profile == Profile::Wmma3 {
            report["assumptions"]
                .as_array_mut()
                .unwrap()
                .extend(wmma3_premises());
        } else if profile.wmma_sparse16() {
            report["assumptions"]
                .as_array_mut()
                .unwrap()
                .extend(wmma4_premises());
        } else if profile != Profile::M2Mode {
            report["assumptions"]
                .as_array_mut()
                .unwrap()
                .extend(wmma_premises());
        }
    }
    if profile == Profile::Comms {
        report["schema"] = json!(report_schema(profile));
        report["profile"] = profile_block(profile, &[]);
        report["checks"] = json!(not_evaluated(profile, "not evaluated"));
        report["outputs"]["contract"] = json!("raw-u32-mw");
    }
    if profile.litmus() {
        report["schema"] = json!(report_schema(profile));
        report["profile"] = profile_block(profile, &[]);
        report["checks"] = json!(not_evaluated(profile, "not evaluated"));
        report["assumptions"] = json!(litmus_premises(profile));
        report["litmus"] = json!({"buffer": [], "async_lds": []});
        report["outputs"] = json!({
            "contract": "raw-u32-image", "objects": [], "words": 0,
            "status": "NOT_AVAILABLE", "values": [], "undefined": [],
        });
    }
    let request_dir = request_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let request = match parse_request(&value, &request_dir) {
        Ok(request) => request,
        Err(e) => return error(report, format!("INVALID_REQUEST: {e}")),
    };
    report["role"] = json!(request.role_name);
    report["profile"] = profile_block(profile, &request.enabled_pending);
    report["object"] = json!({"sha256": request.object_sha256});
    report["outputs"] = unavailable_outputs_for(&request);
    report["source"] = json!({"path": request.source_path_text, "sha256": request.source_sha256, "matches_request": false});
    match backend {
        Ok(Backend::AmdgpuGfx1250) => {}
        Ok(_) => {
            return error(
                report,
                "INVALID_INVOCATION: check-program requires --backend amdgpu --target gfx1250",
            );
        }
        Err(e) => return error(report, format!("INVALID_INVOCATION: {e}")),
    }
    let source_bytes = match std::fs::read(&request.source_path) {
        Ok(bytes) => bytes,
        Err(e) => return error(report, format!("SOURCE_UNREADABLE: {e}")),
    };
    let actual = sha256::hex(&source_bytes);
    report["source"]["sha256"] = json!(actual);
    if actual != request.source_sha256 {
        return error(report, "SOURCE_HASH_MISMATCH");
    }
    report["source"]["matches_request"] = json!(true);
    let Ok(source) = String::from_utf8(source_bytes) else {
        report["parser"]["diagnostics"] = json!([{"line": 0, "message": "source is not UTF-8"}]);
        return error(report, "SOURCE_NOT_UTF8");
    };
    let mut class = ResultClass::Pass;
    let mut reasons = Vec::new();
    let mut diagnostics = Vec::new();
    let mut refused: Vec<RefusedOp> = Vec::new();
    let diagnostic = |line: usize, message: String| json!({"line": line, "message": message});

    // Parser and profile.
    match frontend::function_source_lines(&source, m1::ENTRY) {
        Ok(Some(lines)) => refused.extend(m1::classify_lines_for(
            profile,
            &lines,
            request.role,
            &request.enabled_pending,
        )),
        Ok(None) => diagnostics.push(diagnostic(0, format!("no function `{}`", m1::ENTRY))),
        Err(e) => diagnostics.push(diagnostic(e.span.line, e.message)),
    }
    let module = match frontend::parse(&source) {
        Ok(module) => Some(module),
        Err(e) => {
            if !diagnostics
                .iter()
                .any(|d| d["line"] == json!(e.span.line) && d["message"] == json!(e.message))
            {
                diagnostics.push(diagnostic(e.span.line, e.message));
            }
            None
        }
    };
    if let Some(module) = &module {
        report["target"]["directive"] = json!(module.target_directive.clone().unwrap_or_default());
        report["target"]["metadata_target"] =
            json!(module.metadata_target.clone().unwrap_or_default());
        report["target"]["features"] = json!(
            module
                .target
                .as_deref()
                .map(|t| t.split(':').skip(1).collect::<Vec<_>>())
                .unwrap_or_default()
        );
        if let Err(e) = backend_check(module) {
            diagnostics.push(diagnostic(0, e));
        }
        match m1::check_layout_for(profile, module) {
            Ok(layout) => {
                report["kernel"]["size_bytes"] = json!(layout.size_bytes);
                report["kernel"]["instructions"] = json!(layout.instructions);
                report["kernel"]["padding_after_size_bytes"] =
                    json!(layout.padding_after_size_bytes);
            }
            Err(errors) => diagnostics.extend(errors.into_iter().map(|(l, m)| diagnostic(l, m))),
        }
        if let Some(kernel) = module.kernels.iter().find(|k| k.name == m1::ENTRY) {
            let lds = kernel.descriptor.get("group_segment_fixed_size");
            if (profile.litmus() || profile == Profile::Comms) && lds != Some(request.lds_bytes) {
                diagnostics.push(diagnostic(
                    0,
                    format!(
                        "descriptor group_segment_fixed_size {lds:?} differs from the request's lds_bytes {}",
                        request.lds_bytes
                    ),
                ));
            }
            report["kernel"]["descriptor"] = json!(kernel.descriptor.fields);
            report["kernel"]["instructions"] = json!(kernel.instructions.len());
            for op in
                m1::check_operand_forms_for(profile, kernel, request.role, &request.enabled_pending)
            {
                if !refused.iter().any(|r| r.line == op.line) {
                    refused.push(op);
                }
            }
        }
    }
    refused.sort_by_key(|op| op.line);
    report["parser"]["status"] = json!(if diagnostics.is_empty() {
        "SUPPORTED"
    } else {
        "UNSUPPORTED"
    });
    report["parser"]["diagnostics"] = json!(diagnostics);
    report["parser"]["refused_ops"] = json!(
        refused
            .iter()
            .map(|op| json!({"line": op.line, "mnemonic": op.mnemonic, "reason": op.reason.name()}))
            .collect::<Vec<_>>()
    );
    if !diagnostics.is_empty() {
        class = class.max(ResultClass::UnknownUnsupported);
        reasons.push("PARSER_UNSUPPORTED".to_owned());
    }
    for op in &refused {
        class = class.max(ResultClass::UnknownUnsupported);
        reasons.push(format!(
            "REFUSED_OP:{}:{}:{}",
            op.line,
            op.mnemonic,
            op.reason.name()
        ));
    }
    if class != ResultClass::Pass {
        report["checks"] = json!(not_evaluated(
            profile,
            "not evaluated: program is outside the profile"
        ));
        report["reasons"] = json!(reasons);
        return class;
    }
    let module = module.expect("supported parser status has a module");
    let entry_mode = descriptor_mode(module.kernels.iter().find(|k| k.name == m1::ENTRY));

    // Execution and checks.
    let config = if profile.litmus() {
        AnalysisConfig {
            grid_dim: (1, 1, 1),
            workgroup_id: (0, 0, 0),
            arrays: request
                .objects
                .iter()
                .map(|o| ArrayDef {
                    name: o.name.clone(),
                    base: o.base,
                    elem_width: 4,
                    len: o.words,
                    kind: ArrayKind::Output,
                })
                .collect(),
            params: vec![ParamValue::ArrayPtr(request.objects[0].name.clone())],
            max_instructions: request.max_instructions,
            native_profile: profile.native_profile(),
            guard_regions: request.guards.clone(),
            output_contract: OutputContract::RawU32,
            initial_words: request
                .objects
                .iter()
                .map(|o| InitialWords {
                    array: o.name.clone(),
                    pattern: o.initial,
                })
                .collect(),
            ..AnalysisConfig::new((request.block, 1, 1))
        }
    } else {
        AnalysisConfig {
            grid_dim: (1, 1, 1),
            workgroup_id: (0, 0, 0),
            arrays: vec![ArrayDef {
                name: "out".into(),
                base: request.out_base,
                elem_width: 4,
                len: request.out_words,
                kind: ArrayKind::Output,
            }],
            params: vec![ParamValue::ArrayPtr("out".into())],
            max_instructions: request.max_instructions,
            native_profile: profile.native_profile(),
            guard_regions: request.guards.clone(),
            output_contract: OutputContract::RawU32,
            ..AnalysisConfig::new((request.block, 1, 1))
        }
    };
    let evaluation = if matches!(profile, Profile::MwPred | Profile::Comms) {
        match evaluate_grid(&request, &source, entry_mode, &module, config) {
            Ok((evaluation, workgroups)) => {
                report["workgroups"] = json!(workgroups);
                evaluation
            }
            Err(e) => {
                report["parser"]["status"] = json!("UNSUPPORTED");
                report["parser"]["diagnostics"] = json!([diagnostic(0, e.clone())]);
                report["reasons"] = json!([format!("ANALYSIS_REFUSED: {e}")]);
                return ResultClass::UnknownUnsupported;
            }
        }
    } else {
        let verification =
            match verify_source_kernel(&KernelModule::Amdgpu(module), Some(m1::ENTRY), config) {
                Ok(verification) => verification,
                Err(e) => {
                    report["parser"]["status"] = json!("UNSUPPORTED");
                    report["parser"]["diagnostics"] = json!([diagnostic(0, e.clone())]);
                    report["reasons"] = json!([format!("ANALYSIS_REFUSED: {e}")]);
                    return ResultClass::UnknownUnsupported;
                }
            };
        evaluate(&request, &source, entry_mode, &verification)
    };
    report["execution"] = evaluation.execution;
    report["checks"] = json!(evaluation.checks);
    report["memory_events"] = evaluation.memory_events;
    report["outputs"] = evaluation.outputs;
    report["assumptions"] = json!(evaluation.assumptions);
    if let Some(mode) = evaluation.mode {
        report["mode"] = mode;
    }
    if let Some(litmus) = evaluation.litmus {
        report["litmus"] = litmus;
    }
    reasons.extend(evaluation.reasons);
    report["reasons"] = json!(reasons);
    class.max(evaluation.class)
}

/// `gfx1250-b0-mwpred/1` and `gfx1250-b0-comms/1`: analyse every workgroup
/// of the grid and compose. Composition is claimed only for isolated
/// workgroups (pairwise disjoint written bytes, no workgroup reading a byte
/// another workgroup writes); the composed image takes each word from its
/// single writer. Returns the composed evaluation and a per-workgroup summary.
fn evaluate_grid(
    request: &Request,
    source: &str,
    entry_mode: [Option<u8>; 2],
    module: &frontend::Module,
    base: AnalysisConfig,
) -> Result<(Evaluation, Vec<Value>), String> {
    use std::collections::BTreeMap;
    let index = SourceIndex::new(source);
    let mut sub = request.clone();
    sub.defer_outputs = true;
    let mut reads: Vec<(u64, u32, Value)> = Vec::new();
    let mut evaluations = Vec::new();
    let mut summaries = Vec::new();
    let mut owner: BTreeMap<u64, u32> = BTreeMap::new();
    let mut overlap: Option<Value> = None;
    let mut shared_reads: Vec<Value> = Vec::new();
    for k in 0..request.grid {
        let config = AnalysisConfig {
            grid_dim: (request.grid, 1, 1),
            workgroup_id: (k, 0, 0),
            ..base.clone()
        };
        let verification = verify_source_kernel(
            &KernelModule::Amdgpu(module.clone()),
            Some(m1::ENTRY),
            config,
        )
        .map_err(|e| format!("workgroup {k}: {e}"))?;
        let evaluation = evaluate(&sub, source, entry_mode, &verification);
        let graph = &verification.analysis.graph;
        for (id, event) in graph.events().iter() {
            let EventKind::Access(access) = event.kind else {
                continue;
            };
            let access = graph.access(access);
            let volta_graph::Footprint::Bytes { addr, width } = access.footprint else {
                continue;
            };
            if access.space != Space::Global {
                continue;
            }
            for byte in addr..addr + width as u64 {
                match access.mode {
                    volta_graph::AccessMode::Write => match owner.get(&byte) {
                        Some(&other) if other != k => {
                            if overlap.is_none() {
                                overlap = Some(json!({
                                    "kind": "CROSS_WORKGROUP_WRITE_OVERLAP",
                                    "events": [index.event(graph, id)],
                                    "address": format!("{byte:#x}"),
                                    "workgroups": [other, k],
                                    "detail": format!("workgroups {other} and {k} write the same byte"),
                                }));
                            }
                        }
                        _ => {
                            owner.insert(byte, k);
                        }
                    },
                    volta_graph::AccessMode::Read => {
                        reads.push((byte, k, index.event(graph, id)["line"].clone()));
                    }
                }
            }
        }
        summaries.push(json!({
            "workgroup": k,
            "class": evaluation.class.name(),
            "execution": evaluation.execution["status"],
            "instructions": evaluation.execution["instructions_executed"],
            "reasons": evaluation.reasons.iter().take(8).collect::<Vec<_>>(),
        }));
        evaluations.push(evaluation);
    }
    // A read of a byte that another workgroup writes would make this
    // workgroup's result depend on the others: not composed.
    for (byte, k, line) in &reads {
        if shared_reads.len() >= 16 {
            break;
        }
        if let Some(&writer) = owner.get(byte)
            && writer != *k
        {
            shared_reads.push(json!({
                "workgroup": k,
                "writer": writer,
                "address": format!("{byte:#x}"),
                "line": line,
            }));
        }
    }
    let mut class = ResultClass::Pass;
    let mut reasons: Vec<String> = Vec::new();
    let mut raise = |class: &mut ResultClass, c: ResultClass, reason: String| {
        *class = (*class).max(c);
        if !reasons.contains(&reason) {
            reasons.push(reason);
        }
    };
    // Execution: every workgroup must complete.
    let first = &evaluations[0];
    let mut execution = first.execution.clone();
    let instructions: u64 = evaluations
        .iter()
        .map(|e| e.execution["instructions_executed"].as_u64().unwrap_or(0))
        .sum();
    execution["instructions_executed"] = json!(instructions);
    if let Some((k, e)) = evaluations
        .iter()
        .enumerate()
        .find(|(_, e)| e.execution["status"] != "COMPLETE")
    {
        execution["status"] = e.execution["status"].clone();
        execution["abort"] = e.execution["abort"].clone();
        if execution["abort"].is_object() {
            execution["abort"]["workgroup"] = json!(k);
        }
    }
    for e in &evaluations {
        for reason in &e.reasons {
            if reason.starts_with("BUDGET_EXHAUSTED") {
                raise(&mut class, ResultClass::Timeout, reason.clone());
            } else if reason.starts_with("EXECUTION_ABORTED") {
                raise(&mut class, ResultClass::UnknownUnsupported, reason.clone());
            }
        }
    }
    let complete = execution["status"] == "COMPLETE";
    // Checks: the worst workgroup result of each per-workgroup check.
    let mut checks = Vec::new();
    for (position, id) in check_ids(request.profile).iter().enumerate() {
        if *id == "workgroup_isolation" {
            continue;
        }
        let entries: Vec<_> = evaluations
            .iter()
            .enumerate()
            .filter_map(|(k, e)| e.checks.get(position).map(|c| (k, c)))
            .collect();
        let annotate = |k: usize, w: &Value| {
            let mut w = w.clone();
            w["workgroup"] = json!(k);
            w
        };
        let fails: Vec<_> = entries
            .iter()
            .filter(|(_, c)| c["status"] == "FAIL")
            .collect();
        let unresolved: Vec<String> = entries
            .iter()
            .flat_map(|(k, c)| {
                c["unresolved"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(move |u| format!("workgroup {k}: {u}"))
            })
            .take(16)
            .collect();
        let merged = if fails.iter().any(|(_, c)| c["certified"] == json!(true)) {
            let witnesses = fails
                .iter()
                .filter(|(_, c)| c["certified"] == json!(true))
                .flat_map(|(k, c)| {
                    c["witnesses"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(|w| annotate(*k, w))
                })
                .take(16)
                .collect();
            check(id, "FAIL", true, witnesses, unresolved)
        } else if !fails.is_empty() || entries.iter().any(|(_, c)| c["status"] == "UNKNOWN") {
            let witnesses = entries
                .iter()
                .flat_map(|(k, c)| {
                    c["witnesses"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(|w| annotate(*k, w))
                })
                .take(16)
                .collect();
            check(id, "UNKNOWN", false, witnesses, unresolved)
        } else if entries.iter().all(|(_, c)| c["status"] == "NOT_APPLICABLE") {
            check(id, "NOT_APPLICABLE", false, vec![], vec![])
        } else {
            check(id, "PASS", false, vec![], vec![])
        };
        checks.push(merged);
    }
    checks.push(if let Some(witness) = overlap {
        check("workgroup_isolation", "FAIL", true, vec![witness], vec![])
    } else if !shared_reads.is_empty() {
        let mut check = check(
            "workgroup_isolation",
            "UNKNOWN",
            false,
            vec![],
            vec!["CROSS_WORKGROUP_READ_NOT_MODELLED: a workgroup reads bytes another workgroup writes".into()],
        );
        check["witnesses"] = json!(shared_reads.iter().map(|r| json!({
            "kind": "CROSS_WORKGROUP_READ_NOT_MODELLED",
            "events": [],
            "address": r["address"],
            "workgroup": r["workgroup"],
            "detail": format!("read at line {} of bytes written by workgroup {}", r["line"], r["writer"]),
        })).collect::<Vec<_>>());
        check
    } else if !complete {
        check(
            "workgroup_isolation",
            "UNKNOWN",
            false,
            vec![],
            vec![format!("execution {}", execution["status"].as_str().unwrap_or("?"))],
        )
    } else {
        check("workgroup_isolation", "PASS", false, vec![], vec![])
    });
    for c in &checks {
        let id = c["id"].as_str().unwrap();
        match c["status"].as_str().unwrap() {
            "FAIL" if c["certified"] == json!(true) => raise(
                &mut class,
                ResultClass::Reject,
                format!("CHECK_FAILED:{id}"),
            ),
            "FAIL" | "UNKNOWN" => raise(
                &mut class,
                ResultClass::UnknownUnsupported,
                format!("CHECK_UNKNOWN:{id}"),
            ),
            _ => {}
        }
    }
    // Outputs: one writer per word; then the declared opaque/unwritten check.
    let available = evaluations
        .iter()
        .all(|e| e.outputs["status"] != "NOT_AVAILABLE" && e.outputs["values"].is_array());
    let outputs = if available && complete {
        let words = request.out_words as usize;
        let mut values = vec![Value::Null; words];
        let mut undefined: Vec<Option<String>> = vec![Some("NOT_WRITTEN".into()); words];
        let mut writers = vec![0u32; words];
        for e in &evaluations {
            let reasons: BTreeMap<u64, &str> = e.outputs["undefined"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|u| Some((u["index"].as_u64()?, u["reason"].as_str()?)))
                .collect();
            for (i, value) in e.outputs["values"].as_array().unwrap().iter().enumerate() {
                let reason = reasons.get(&(i as u64)).copied();
                if value.is_null() && reason == Some("NOT_WRITTEN") {
                    continue;
                }
                writers[i] += 1;
                values[i] = value.clone();
                undefined[i] = if value.is_null() {
                    Some(reason.unwrap_or("UNDEFINED_BITS").into())
                } else {
                    None
                };
            }
        }
        for i in 0..words {
            if writers[i] > 1 {
                values[i] = Value::Null;
                undefined[i] = Some("MULTIPLE_WRITES_UNORDERED".into());
            }
        }
        let in_ranges = |ranges: &[(u64, u64)], i: usize| {
            ranges
                .iter()
                .any(|&(first, count)| (first..first + count).contains(&(i as u64)))
        };
        let unexpected = (0..words)
            .filter(|&i| {
                let expected = if in_ranges(&request.opaque_words, i) {
                    Some("OPAQUE")
                } else if in_ranges(&request.unwritten_words, i) {
                    Some("NOT_WRITTEN")
                } else {
                    None
                };
                match expected {
                    Some(reason) => undefined[i].as_deref() != Some(reason),
                    None => undefined[i].is_some(),
                }
            })
            .count();
        let any_undefined = undefined.iter().any(Option::is_some);
        let status = if !any_undefined {
            "COMPLETE"
        } else if unexpected == 0 {
            "COMPLETE_AS_DECLARED"
        } else {
            "PARTIAL"
        };
        if unexpected > 0 {
            raise(
                &mut class,
                ResultClass::UnknownUnsupported,
                format!("OUTPUTS_PARTIAL: {unexpected} undefined or undeclared words"),
            );
        }
        json!({
            "contract": "raw-u32-mw", "object": "out", "words": request.out_words,
            "status": status, "values": values,
            "undefined": undefined.iter().enumerate().filter_map(|(i, u)| u.as_ref().map(|r| json!({"index": i, "reason": r}))).collect::<Vec<_>>(),
        })
    } else {
        if class == ResultClass::Pass {
            raise(
                &mut class,
                ResultClass::UnknownUnsupported,
                "OUTPUTS_NOT_AVAILABLE".into(),
            );
        }
        unavailable_outputs_for(request)
    };
    let first = evaluations.into_iter().next().unwrap();
    Ok((
        Evaluation {
            litmus: None,
            execution,
            checks,
            memory_events: first.memory_events,
            outputs,
            assumptions: first.assumptions,
            mode: first.mode,
            class,
            reasons,
        },
        summaries,
    ))
}

fn backend_check(module: &frontend::Module) -> Result<(), String> {
    Backend::AmdgpuGfx1250.validate_amd_target(module.target.as_deref())
}

/// Atomically replace `path` with the report (temporary file + rename).
pub fn write_report(path: &Path, report: &Value) -> std::io::Result<()> {
    let mut text = serde_json::to_string_pretty(report).expect("report serializes");
    text.push('\n');
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "report.json".into());
    let temporary = path.with_file_name(format!(".{file_name}.tmp.{}", std::process::id()));
    std::fs::write(&temporary, text)?;
    std::fs::rename(&temporary, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temporary);
    })
}

/// `check-program` arguments.
#[derive(clap::Args)]
pub(crate) struct CheckProgramArgs {
    /// Request JSON (one of the frozen `ptxv-*-request/*` schemas)
    #[arg(long)]
    request: PathBuf,
    /// Report JSON to write atomically
    #[arg(long)]
    report: PathBuf,
}

/// Run one request and write its report; the exit status is the report class.
pub(crate) fn cmd_check_program(
    args: &CheckProgramArgs,
    backend: Result<Backend, String>,
) -> ExitCode {
    let report_path = &args.report;
    let outcome = run_request(&args.request, backend);
    if let Err(e) = write_report(report_path, &outcome.report) {
        eprintln!("error: cannot write report {}: {e}", report_path.display());
        return ExitCode::from(ResultClass::Error.exit_code());
    }
    ExitCode::from(outcome.class.exit_code())
}

#[cfg(test)]
mod tests {
    #[test]
    fn allowlist_hash_matches_the_frozen_subset() {
        assert_eq!(
            super::sha256::hex(super::m1::canonical_allowlist().as_bytes()),
            "d627ed3f3ac8c1852e4a63cce6109a2faa70d0ef9a2ba1db000d61ffec11a919"
        );
    }

    #[test]
    fn litmus_allowlist_hash_matches_the_frozen_subset() {
        let subset: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/gfx1250/litmus/schema/litmus_b0_subset.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let hash = super::sha256::hex(
            super::m1::canonical_allowlist_for(super::Profile::Litmus).as_bytes(),
        );
        assert_eq!(hash, subset["allowlist_sha256"].as_str().unwrap());
        assert_eq!(
            hash,
            "4aacd89bd5fff95332289898b983f6da9ec1163654f51dfdd61c58ffa9efd4e2"
        );
    }

    #[test]
    fn wmma2_allowlist_hash_matches_the_frozen_subset() {
        let subset: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/gfx1250/wmma2/schema/wmma_types_b0.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let hash = super::sha256::hex(
            super::m1::canonical_allowlist_for(super::Profile::Wmma2).as_bytes(),
        );
        assert_eq!(hash, subset["allowlist_sha256"].as_str().unwrap());
        let mut body: Vec<&str> = subset["body_allowed_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|op| op.as_str().unwrap())
            .collect();
        body.sort_unstable();
        let mut ours = super::m1::WMMA2_BODY_OPS.to_vec();
        ours.sort_unstable();
        assert_eq!(body, ours);
    }

    #[test]
    fn wmma3_allowlist_hash_matches_the_frozen_subset() {
        let subset: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/gfx1250/wmma3/schema/wmma_scale_b0.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let hash = super::sha256::hex(
            super::m1::canonical_allowlist_for(super::Profile::Wmma3).as_bytes(),
        );
        assert_eq!(hash, subset["allowlist_sha256"].as_str().unwrap());
        let mut body: Vec<&str> = subset["body_allowed_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|op| op.as_str().unwrap())
            .collect();
        body.sort_unstable();
        let mut ours = super::m1::WMMA3_BODY_OPS.to_vec();
        ours.sort_unstable();
        assert_eq!(body, ours);
    }

    #[test]
    fn wmma4_allowlist_hash_matches_the_frozen_subset() {
        let subset: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/gfx1250/wmma4/schema/wmma_sparse16_b0.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let hash = super::sha256::hex(
            super::m1::canonical_allowlist_for(super::Profile::Wmma4).as_bytes(),
        );
        assert_eq!(hash, subset["allowlist_sha256"].as_str().unwrap());
        let mut body: Vec<&str> = subset["body_allowed_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|op| op.as_str().unwrap())
            .collect();
        body.sort_unstable();
        let mut ours = super::m1::WMMA4_BODY_OPS.to_vec();
        ours.sort_unstable();
        assert_eq!(body, ours);
    }

    #[test]
    fn mw2_allowlist_hash_matches_the_frozen_subset() {
        let subset: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/gfx1250/mw2/schema/mw_types_b0.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let hash = super::sha256::hex(
            super::m1::canonical_allowlist_for(super::Profile::Multiwave2).as_bytes(),
        );
        assert_eq!(hash, subset["allowlist_sha256"].as_str().unwrap());
        let mut body: Vec<&str> = subset["body_allowed_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|op| op.as_str().unwrap())
            .collect();
        body.sort_unstable();
        let mut ours = super::m1::WMMA4_BODY_OPS.to_vec();
        ours.sort_unstable();
        assert_eq!(body, ours);
        let mut wrapper: Vec<&str> = subset["wrapper_mnemonics"]
            .as_array()
            .unwrap()
            .iter()
            .map(|op| op.as_str().unwrap())
            .collect();
        wrapper.sort_unstable();
        let mut ours = super::m1::MW_WRAPPER_MNEMONICS.to_vec();
        ours.sort_unstable();
        assert_eq!(wrapper, ours);
    }

    #[test]
    fn comms_allowlist_hash_matches_the_frozen_subset() {
        let subset: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/gfx1250/comms/schema/comms_b0.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let hash = super::sha256::hex(
            super::m1::canonical_allowlist_for(super::Profile::Comms).as_bytes(),
        );
        assert_eq!(hash, subset["allowlist_sha256"].as_str().unwrap());
        let mut body: Vec<&str> = subset["body_allowed_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|op| op.as_str().unwrap())
            .collect();
        body.sort_unstable();
        let mut ours = super::m1::COMMS_BODY_OPS.to_vec();
        ours.sort_unstable();
        assert_eq!(body, ours);
        let mut wrapper: Vec<&str> = subset["wrapper_mnemonics"]
            .as_array()
            .unwrap()
            .iter()
            .map(|op| op.as_str().unwrap())
            .collect();
        wrapper.sort_unstable();
        let mut ours = super::m1::COMMS_WRAPPER_MNEMONICS.to_vec();
        ours.sort_unstable();
        assert_eq!(wrapper, ours);
        assert!(subset["pending_ops"].as_object().unwrap().is_empty());
    }

    #[test]
    fn mwpred_allowlist_hash_matches_the_frozen_subset() {
        let subset: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/gfx1250/mwpred/schema/mw_pred_b0.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let hash = super::sha256::hex(
            super::m1::canonical_allowlist_for(super::Profile::MwPred).as_bytes(),
        );
        assert_eq!(hash, subset["allowlist_sha256"].as_str().unwrap());
        let mut body: Vec<&str> = subset["body_allowed_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|op| op.as_str().unwrap())
            .collect();
        body.sort_unstable();
        let mut ours = super::m1::WMMA_BODY_OPS.to_vec();
        ours.sort_unstable();
        assert_eq!(body, ours);
        let mut wrapper: Vec<&str> = subset["wrapper_mnemonics"]
            .as_array()
            .unwrap()
            .iter()
            .map(|op| op.as_str().unwrap())
            .collect();
        wrapper.sort_unstable();
        let mut ours = super::m1::MWPRED_WRAPPER_MNEMONICS.to_vec();
        ours.sort_unstable();
        assert_eq!(wrapper, ours);
    }

    #[test]
    fn litmus2_allowlist_hash_matches_the_frozen_subset() {
        let subset: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/gfx1250/litmus2/schema/litmus_b0_t2_subset.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let hash = super::sha256::hex(
            super::m1::canonical_allowlist_for(super::Profile::Litmus2).as_bytes(),
        );
        assert_eq!(hash, subset["allowlist_sha256"].as_str().unwrap());
        let body: Vec<&str> = subset["body_allowed_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|op| op.as_str().unwrap())
            .collect();
        let mut ours = super::m1::LITMUS2_BODY_OPS.to_vec();
        ours.sort_unstable();
        assert_eq!(body, ours);
    }
}

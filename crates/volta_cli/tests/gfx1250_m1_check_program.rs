//! `volta check-program` on the real m1 corpus (`tests/gfx1250/m1`):
//! fuzzer-rendered kernels assembled by the fuzzer's clang, an independent
//! reference for every output word, harmful negatives with typed witnesses,
//! refusal, undefined-state and parser controls, and the report contract.
#[path = "support/gfx1250_cases.rs"]
mod cases;

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/gfx1250/m1")
}

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "volta-m1-check-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

const CLASSES: [(&str, i32); 5] = [
    ("PASS", 0),
    ("REJECT", 1),
    ("UNKNOWN_UNSUPPORTED", 2),
    ("ERROR", 3),
    ("TIMEOUT", 4),
];

/// Run one request; the exit status must equal the report's class and the
/// report must validate against the frozen schema.
fn check(request: &Path) -> Value {
    let report_path = scratch().join("report.json");
    let output = Command::new(env!("CARGO_BIN_EXE_volta"))
        .args([
            "--backend",
            "amdgpu",
            "--target",
            "gfx1250",
            "--no-log-file",
        ])
        .arg("check-program")
        .arg("--request")
        .arg(request)
        .arg("--report")
        .arg(&report_path)
        .output()
        .unwrap();
    let report: Value = serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    let class = report["result_class"].as_str().unwrap();
    let code = CLASSES.iter().find(|(name, _)| *name == class).unwrap().1;
    assert_eq!(
        output.status.code(),
        Some(code),
        "{}: {class}",
        request.display()
    );
    validate_report(&report);
    let ids: Vec<_> = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        [
            "trace_completeness",
            "wait_readiness",
            "barrier_generations",
            "memory_conflicts",
            "object_bounds",
            "prefetch_policy",
            "store_completion"
        ]
    );
    for check in report["checks"].as_array().unwrap() {
        if check["certified"] == json!(true) {
            assert_eq!(check["status"], "FAIL");
            assert!(!check["witnesses"].as_array().unwrap().is_empty());
        }
    }
    report
}

// ---------------------------------------------------------------------------
// A JSON Schema subset interpreter for the frozen report schema (no new
// dependency): type, const, enum, required, properties, additionalProperties,
// items, min/maxItems, minimum, maximum, oneOf, $ref and its two patterns.
// ---------------------------------------------------------------------------

fn pattern(pattern: &str, text: &str) -> bool {
    match pattern {
        "^[0-9a-f]{64}$" => {
            text.len() == 64 && text.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
        }
        "^0x[0-9a-f]+$" => {
            text.len() > 2
                && text.starts_with("0x")
                && text[2..]
                    .chars()
                    .all(|c| matches!(c, '0'..='9' | 'a'..='f'))
        }
        _ => panic!("schema pattern {pattern} is not supported by the test validator"),
    }
}

fn type_matches(name: &str, value: &Value) -> bool {
    match name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => panic!("unknown schema type {name}"),
    }
}

fn validate(schema: &Value, root: &Value, value: &Value, path: &str) -> Result<(), String> {
    let fail = |what: String| Err(format!("{path}: {what}"));
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        let name = reference.strip_prefix("#/$defs/").unwrap();
        return validate(&root["$defs"][name], root, value, path);
    }
    if let Some(types) = schema.get("type") {
        let ok = match types {
            Value::String(name) => type_matches(name, value),
            Value::Array(names) => names
                .iter()
                .any(|n| type_matches(n.as_str().unwrap(), value)),
            _ => unreachable!(),
        };
        if !ok {
            return fail(format!("type {types} does not match {value}"));
        }
    }
    if let Some(expected) = schema.get("const")
        && expected != value
    {
        return fail(format!("{value} is not {expected}"));
    }
    if let Some(options) = schema.get("enum").and_then(Value::as_array)
        && !options.contains(value)
    {
        return fail(format!("{value} not in {options:?}"));
    }
    if let Some(options) = schema.get("oneOf").and_then(Value::as_array) {
        let matched = options
            .iter()
            .filter(|option| validate(option, root, value, path).is_ok())
            .count();
        if matched != 1 {
            return fail(format!("{matched} oneOf alternatives match"));
        }
    }
    if let (Some(p), Some(text)) = (
        schema.get("pattern").and_then(Value::as_str),
        value.as_str(),
    ) && !pattern(p, text)
    {
        return fail(format!("{text} does not match {p}"));
    }
    if let Some(number) = value.as_f64() {
        if schema
            .get("minimum")
            .and_then(Value::as_f64)
            .is_some_and(|m| number < m)
        {
            return fail("below minimum".into());
        }
        if schema
            .get("maximum")
            .and_then(Value::as_f64)
            .is_some_and(|m| number > m)
        {
            return fail("above maximum".into());
        }
    }
    if let Some(object) = value.as_object() {
        for key in schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if !object.contains_key(key.as_str().unwrap()) {
                return fail(format!("missing {key}"));
            }
        }
        let properties = schema.get("properties").and_then(Value::as_object);
        for (key, item) in object {
            match properties.and_then(|p| p.get(key)) {
                Some(property) => validate(property, root, item, &format!("{path}.{key}"))?,
                None if schema.get("additionalProperties") == Some(&json!(false)) => {
                    return fail(format!("unknown property {key}"));
                }
                None => {}
            }
        }
    }
    if let Some(items) = value.as_array() {
        if schema
            .get("minItems")
            .and_then(Value::as_u64)
            .is_some_and(|m| (items.len() as u64) < m)
        {
            return fail("too few items".into());
        }
        if schema
            .get("maxItems")
            .and_then(Value::as_u64)
            .is_some_and(|m| items.len() as u64 > m)
        {
            return fail("too many items".into());
        }
        if let Some(item_schema) = schema.get("items") {
            for (index, item) in items.iter().enumerate() {
                validate(item_schema, root, item, &format!("{path}[{index}]"))?;
            }
        }
    }
    Ok(())
}

fn validate_report(report: &Value) {
    let schema: Value = serde_json::from_slice(
        &fs::read(corpus().join("schema/ptxv-m1-report.schema.json")).unwrap(),
    )
    .unwrap();
    validate(&schema, &schema, report, "report").unwrap();
}

#[test]
fn schema_validator_rejects_malformed_reports() {
    let report = check(&cases::case_dir(&corpus(), "seed-scc_chain-cand").join("request.json"));
    let schema: Value = serde_json::from_slice(
        &fs::read(corpus().join("schema/ptxv-m1-report.schema.json")).unwrap(),
    )
    .unwrap();
    for (pointer, value) in [
        ("/result_class", json!("ACCEPT")),
        ("/verifier/exe_sha256", json!("abc")),
        ("/outputs/values/0", json!(-1)),
        ("/checks/0/status", json!("OK")),
    ] {
        let mut broken = report.clone();
        *broken.pointer_mut(pointer).unwrap() = value;
        assert!(
            validate(&schema, &schema, &broken, "report").is_err(),
            "{pointer}"
        );
    }
    let mut extra = report.clone();
    extra["unexpected"] = json!(1);
    assert!(validate(&schema, &schema, &extra, "report").is_err());
}

// ---------------------------------------------------------------------------
// Positive controls: every fuzzer program, candidate and twin.
// ---------------------------------------------------------------------------

/// Per-program provenance (assembly facts of each role), keyed by program.
fn programs() -> Value {
    serde_json::from_slice::<Value>(&fs::read(corpus().join("provenance.json")).unwrap()).unwrap()
        ["programs"]
        .clone()
}

fn expected_words() -> Value {
    serde_json::from_slice(&fs::read(corpus().join("expected-words.json")).unwrap()).unwrap()
}

fn assert_exact_words(report: &Value, expected: &Value, what: &str) {
    assert_eq!(
        report["result_class"], "PASS",
        "{what}: {}",
        report["reasons"]
    );
    assert_eq!(report["outputs"]["status"], "COMPLETE", "{what}");
    assert_eq!(report["outputs"]["values"], *expected, "{what}");
    assert_eq!(report["parser"]["status"], "SUPPORTED");
    assert_eq!(report["execution"]["status"], "COMPLETE");
    assert_eq!(report["source"]["matches_request"], true);
}

/// llvm-objdump of the clang-24 object: entry start and end of s_endpgm.
fn objdump_extent(objdump: &str) -> (u64, u64, u64) {
    let address = |line: &str| {
        u64::from_str_radix(
            line.rsplit("// ")
                .next()
                .unwrap()
                .split(':')
                .next()
                .unwrap()
                .trim(),
            16,
        )
        .unwrap()
    };
    let start = objdump
        .lines()
        .find(|l| l.ends_with("<kernel>:"))
        .map(|l| u64::from_str_radix(l.split_whitespace().next().unwrap(), 16).unwrap())
        .unwrap();
    let end = objdump
        .lines()
        .find(|l| l.trim_start().starts_with("s_endpgm"))
        .map(address)
        .unwrap()
        + 4;
    // Repeated s_code_end padding is recorded as one `s_code_end xN` line.
    let padding = objdump
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("s_code_end"))
        .map(|rest| {
            rest.trim()
                .strip_prefix('x')
                .map_or(1, |n| n.parse::<u64>().unwrap())
        })
        .sum::<u64>()
        * 4;
    (start, end, padding)
}

/// Every `s_wait_alu` operand decodes to the encoding in the object.
fn assert_wait_alu_encodings(source: &str, objdump: &str) {
    let module = volta_amdgpu_analysis::frontend::parse(source).unwrap();
    let parsed: Vec<i64> = module.kernels[0]
        .instructions
        .iter()
        .filter(|i| i.mnemonic == "s_wait_alu")
        .map(|i| match i.operands[0] {
            volta_amdgpu_analysis::frontend::Operand::Integer(value) => value,
            _ => panic!("s_wait_alu operand"),
        })
        .collect();
    let encoded: Vec<i64> = objdump
        .lines()
        .filter(|l| l.trim_start().starts_with("s_wait_alu"))
        .map(|l| {
            let word = l.rsplit(": ").next().unwrap().trim();
            assert!(word.starts_with("BF88"), "{l}");
            i64::from_str_radix(&word[4..8], 16).unwrap()
        })
        .collect();
    assert_eq!(parsed, encoded);
}

#[test]
fn every_fuzzer_program_passes_with_exact_reference_words() {
    let expected = expected_words();
    let mut checked = 0;
    for (name, provenance) in programs().as_object().unwrap() {
        if name.starts_with("control-") {
            continue;
        }
        for role in ["cand", "twin"] {
            let pending = name == "probe-s_max_i32";
            let directory = cases::case_dir(
                &corpus(),
                &format!("{name}-{role}{}", if pending { "-pending" } else { "" }),
            );
            let request = directory.join("request.json");
            let report = check(&request);
            assert_exact_words(&report, &expected[name], &format!("{name}/{role}"));
            assert_eq!(report["role"], role);
            // The fuzzer's clang-24 object: target flags, size, padding and
            // every s_wait_alu encoding agree with the verifier's reading.
            let facts = &provenance[role];
            assert!(
                facts["readobj_file_headers"]
                    .as_str()
                    .unwrap()
                    .contains("EF_AMDGPU_MACH_AMDGCN_GFX1250 (0x49)")
            );
            let objdump = facts["objdump"].as_str().unwrap();
            let (start, end, padding) = objdump_extent(objdump);
            assert_eq!(report["kernel"]["size_bytes"], end - start, "{name}/{role}");
            assert_eq!(
                report["kernel"]["padding_after_size_bytes"], padding,
                "{name}/{role}"
            );
            assert_eq!(report["object"]["sha256"], facts["object_sha256"]);
            let source = fs::read_to_string(directory.join("kernel.s")).unwrap();
            assert_wait_alu_encodings(&source, objdump);
            let prefetch = &report["memory_events"]["prefetches"][0];
            assert_eq!(
                (
                    &prefetch["scope"],
                    &prefetch["lo"],
                    &prefetch["hi"],
                    &prefetch["xcnt_charged"]
                ),
                (
                    &json!("SCOPE_SE"),
                    &json!("0x0"),
                    &json!("0x1f"),
                    &json!(true)
                )
            );
            assert_eq!(prefetch["objects_hit"], json!([]));
            assert_eq!(report["memory_events"]["stores"]["bytes_outside_out"], 0);
            assert_eq!(
                report["memory_events"]["stores"]["bytes_in_out"],
                4 * report["outputs"]["words"].as_u64().unwrap()
            );
            checked += 1;
        }
    }
    // Two seeds, 41 op probes (s_max_i32 with its pending control) and two mutants.
    assert_eq!(checked, 2 * (2 + 41 + 2));
}

#[test]
fn pending_s_max_i32_requires_the_explicit_request_gate() {
    for role in ["cand", "twin"] {
        let report = check(
            &cases::case_dir(&corpus(), &format!("probe-s_max_i32-{role}")).join("request.json"),
        );
        assert_eq!(report["result_class"], "UNKNOWN_UNSUPPORTED");
        assert_eq!(report["execution"]["status"], "NOT_RUN");
        let refused = report["parser"]["refused_ops"].as_array().unwrap();
        assert!(!refused.is_empty());
        for op in refused {
            assert_eq!(op["mnemonic"], "s_max_i32");
            assert_eq!(op["reason"], "PENDING_HARDWARE_CONTROL");
        }
    }
}

#[test]
fn signed_s_max_scc_controls_use_the_strict_b0_comparison() {
    let expected = expected_words();
    for (name, s11, s12) in [("smax_i32_eq_scc1", 0u64, 1u64), ("smax_i32_gt_scc0", 1, 0)] {
        for role in ["cand", "twin"] {
            let report = check(
                &cases::case_dir(&corpus(), &format!("control-{name}-{role}")).join("request.json"),
            );
            assert_exact_words(&report, &expected[format!("control-{name}")], name);
            // Output slot k occupies words 32k..32k+32.
            let provenance = &programs()[format!("control-{name}")];
            let slot = |register: &str| {
                let k = provenance["slots"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .position(|s| s == register)
                    .unwrap();
                32 * k..32 * k + 32
            };
            let values = report["outputs"]["values"].as_array().unwrap();
            assert!(values[slot("s11")].iter().all(|v| *v == s11), "{name} s11");
            assert!(values[slot("s12")].iter().all(|v| *v == s12), "{name} s12");
            assert_eq!(report["role"], "control");
            assert_eq!(
                report["profile"]["enabled_pending_ops"],
                json!(["s_max_i32"])
            );
        }
    }
    // The control-only s_cmp_eq_u32 preset is refused outside role control.
    let directory = scratch();
    let source = cases::case_dir(&corpus(), "control-smax_i32_eq_scc1-cand").join("kernel.s");
    fs::copy(&source, directory.join("kernel.s")).unwrap();
    let mut request: Value = serde_json::from_slice(
        &fs::read(cases::case_dir(&corpus(), "control-smax_i32_eq_scc1-cand").join("request.json"))
            .unwrap(),
    )
    .unwrap();
    request["role"] = json!("cand");
    fs::write(directory.join("request.json"), request.to_string()).unwrap();
    let report = check(&directory.join("request.json"));
    assert_eq!(report["result_class"], "UNKNOWN_UNSUPPORTED");
    assert!(
        report["parser"]["refused_ops"]
            .as_array()
            .unwrap()
            .iter()
            .any(|op| op["mnemonic"] == "s_cmp_eq_u32" && op["reason"] == "NOT_IN_PROFILE")
    );
}

// ---------------------------------------------------------------------------
// Harmful negatives derived from the real seed: certified typed witnesses.
// ---------------------------------------------------------------------------

fn derived(case: &str) -> Value {
    check(&cases::case_dir(&corpus(), case).join("request.json"))
}

fn failed<'a>(report: &'a Value, id: &str) -> &'a Value {
    assert_eq!(report["result_class"], "REJECT", "{}", report["reasons"]);
    let check = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == id)
        .unwrap();
    assert_eq!(check["status"], "FAIL");
    assert_eq!(check["certified"], true);
    &check["witnesses"][0]
}

#[test]
fn missing_or_misplaced_kernarg_waits_reject_with_read_before_ready() {
    for case in [
        "neg-no-kmcnt",
        "neg-wait-alu-for-kmcnt",
        "neg-kmcnt-after-store",
    ] {
        let report = derived(case);
        let witness = failed(&report, "wait_readiness");
        assert_eq!(witness["kind"], "READ_BEFORE_READY", "{case}");
        assert_eq!(witness["register"], "s2", "{case}");
        let events = witness["events"].as_array().unwrap();
        assert_eq!(events[0]["text"], "s_load_b64 s[2:3], s[0:1], 0x0");
        assert_eq!(events[0]["line"], 12);
        assert_eq!(
            events[1]["text"], "global_store_b32 v1, v10, s[2:3]",
            "{case}"
        );
    }
    // s_wait_alu 0x0 is an executed ALU-only wait at the kmcnt position.
    let source =
        fs::read_to_string(cases::case_dir(&corpus(), "neg-wait-alu-for-kmcnt").join("kernel.s"))
            .unwrap();
    assert_eq!(source.lines().nth(16).unwrap(), "  s_wait_alu 0x0");
}

#[test]
fn guard_and_out_of_object_stores_reject_before_issue() {
    // (case, guard, first offending lane, its address, word index in guard, line)
    for (case, guard, lane, address, word, line) in [
        (
            "neg-store-post-guard-all",
            Some("post"),
            0u64,
            "0x10600",
            Some(0u64),
            44u64,
        ),
        (
            "neg-store-post-guard-upper",
            Some("post"),
            16,
            "0x10600",
            Some(0),
            44,
        ),
        (
            "neg-store-pre-guard",
            Some("pre"),
            0,
            "0x100c0",
            Some(48),
            28,
        ),
        ("neg-store-far", None, 0, "0x110100", None, 44),
    ] {
        let report = derived(case);
        let witness = failed(&report, "object_bounds");
        assert_eq!(
            witness["kind"],
            if guard.is_some() {
                "GUARD_ACCESS"
            } else {
                "OUT_OF_OBJECT_ACCESS"
            },
            "{case}"
        );
        assert_eq!(witness["lane"], lane, "{case}");
        assert_eq!(witness["address"], address, "{case}");
        assert_eq!(witness["bytes"], 4);
        assert_eq!(witness["events"][0]["line"], line, "{case}");
        if let Some(guard) = guard {
            assert_eq!(witness["guard"], guard);
            assert_eq!(witness["word_index"], word.unwrap());
        }
        assert_eq!(report["execution"]["status"], "ABORTED");
        assert_eq!(report["outputs"]["status"], "NOT_AVAILABLE");
    }
    // Partial footprints: exactly the offending lanes are reported.
    let upper = derived("neg-store-post-guard-upper");
    let detail = failed(&upper, "object_bounds")["detail"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(
        detail.ends_with(&format!("{:?}", (16..32).collect::<Vec<u8>>())),
        "{detail}"
    );
    let pre = derived("neg-store-pre-guard");
    let detail = failed(&pre, "object_bounds")["detail"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(
        detail.ends_with(&format!("{:?}", (0..16).collect::<Vec<u8>>())),
        "{detail}"
    );
}

#[test]
fn prefetch_scope_and_source_lifetime_are_checked() {
    let report = derived("neg-prefetch-scope-cu");
    let witness = failed(&report, "prefetch_policy");
    assert_eq!(witness["kind"], "NON_SPECULATIVE_PREFETCH_OUTSIDE_OBJECT");
    assert_eq!(
        (&witness["lane"], &witness["address"], &witness["bytes"]),
        (&json!(0), &json!("0x0"), &json!(1))
    );
    assert_eq!(
        witness["events"][0]["text"],
        "global_prefetch_b8 v0, s[30:31] scope:SCOPE_CU"
    );
    // Overwriting the prefetch address VGPR before its X-count lease retires.
    let report = derived("neg-prefetch-source-overwrite");
    let witness = failed(&report, "wait_readiness");
    assert_eq!(witness["kind"], "WRITE_BEFORE_SOURCE_RELEASED");
    assert_eq!(witness["register"], "v0");
    assert_eq!(
        witness["events"][0]["text"],
        "global_prefetch_b8 v0, s[30:31] scope:SCOPE_SE"
    );
    assert_eq!(witness["events"][1]["text"], "v_mov_b32 v0, v0");
}

// ---------------------------------------------------------------------------
// Refusal, undefined-state and parser controls: never PASS.
// ---------------------------------------------------------------------------

#[test]
fn excluded_operations_are_refused_with_typed_reasons() {
    for (case, mnemonic, reason, line) in [
        ("ref-getreg-mode", "s_getreg_b32", "MODE_ACCESS", 18),
        ("ref-setreg", "s_setreg_b32", "MODE_ACCESS", 18),
        ("ref-setreg-imm32", "s_setreg_imm32_b32", "MODE_ACCESS", 18),
        ("ref-round-mode", "s_round_mode", "MODE_ACCESS", 18),
        ("ref-denorm-mode", "s_denorm_mode", "MODE_ACCESS", 18),
        ("ref-perm-pk16", "v_perm_pk16_b4_u4", "EXCLUDED_OP", 18),
        ("ref-scratch-base-read", "s_mov_b32", "EXCLUDED_OP", 18),
        ("ref-fma-f32", "v_fma_f32", "FLOATING_POINT", 18),
        ("ref-exp-f32", "v_exp_f32", "FLOATING_POINT", 18),
        ("ref-add-f16", "v_add_f16", "FLOATING_POINT", 18),
        ("ref-ds-load", "ds_load_b32", "NOT_IN_PROFILE", 18),
        ("ref-s-max-i32", "s_max_i32", "PENDING_HARDWARE_CONTROL", 18),
        ("ref-s-cmp-in-cand", "s_cmp_eq_u32", "NOT_IN_PROFILE", 18),
        ("ref-unreachable-mode", "s_getreg_b32", "MODE_ACCESS", 46),
    ] {
        let report = derived(case);
        assert_eq!(report["result_class"], "UNKNOWN_UNSUPPORTED", "{case}");
        assert_eq!(report["execution"]["status"], "NOT_RUN", "{case}");
        assert_eq!(
            report["parser"]["refused_ops"],
            json!([{"line": line, "mnemonic": mnemonic, "reason": reason}]),
            "{case}"
        );
    }
}

#[test]
fn undefined_entry_state_aborts_as_unknown() {
    for (case, reason) in [
        ("undef-scc", "uninitialized SCC flag"),
        ("undef-vcc", "uninitialized VCC_LO flag"),
        ("undef-sgpr", "uninitialized Scalar register 13"),
        ("undef-vgpr", "uninitialized Vector register 21"),
    ] {
        let report = derived(case);
        assert_eq!(report["result_class"], "UNKNOWN_UNSUPPORTED", "{case}");
        assert_eq!(report["execution"]["status"], "ABORTED");
        assert!(
            report["execution"]["abort"]["reason"]
                .as_str()
                .unwrap()
                .contains(reason),
            "{case}"
        );
        assert_eq!(report["execution"]["abort"]["line"], 18);
        assert_eq!(report["outputs"]["status"], "NOT_AVAILABLE");
        assert!(
            report["checks"]
                .as_array()
                .unwrap()
                .iter()
                .all(|c| c["status"] != "PASS" || c["id"] == "barrier_generations")
        );
    }
}

#[test]
fn target_and_size_layout_violations_are_parser_unsupported() {
    for case in [
        "parse-target-gfx950",
        "parse-feature-suffix",
        "parse-metadata-mismatch",
        "parse-missing-size",
        "parse-missing-size-no-padding",
        "parse-truncated-size",
        "parse-padding-inside-size",
    ] {
        let report = derived(case);
        assert_eq!(report["result_class"], "UNKNOWN_UNSUPPORTED", "{case}");
        assert_eq!(report["parser"]["status"], "UNSUPPORTED", "{case}");
        assert!(
            !report["parser"]["diagnostics"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(report["execution"]["status"], "NOT_RUN");
    }
}

// ---------------------------------------------------------------------------
// Report contract.
// ---------------------------------------------------------------------------

fn modified_request(edit: impl FnOnce(&mut Value)) -> PathBuf {
    let directory = scratch();
    fs::copy(
        cases::case_dir(&corpus(), "seed-scc_chain-cand").join("kernel.s"),
        directory.join("kernel.s"),
    )
    .unwrap();
    let mut request: Value = serde_json::from_slice(
        &fs::read(cases::case_dir(&corpus(), "seed-scc_chain-cand").join("request.json")).unwrap(),
    )
    .unwrap();
    edit(&mut request);
    let path = directory.join("request.json");
    fs::write(&path, request.to_string()).unwrap();
    path
}

#[test]
fn hash_mismatch_invalid_requests_and_budget_have_their_own_classes() {
    let report = check(&modified_request(|r| {
        r["source"]["sha256"] = json!("0".repeat(64))
    }));
    assert_eq!(report["result_class"], "ERROR");
    assert_eq!(report["reasons"], json!(["SOURCE_HASH_MISMATCH"]));
    assert_eq!(report["source"]["matches_request"], false);
    assert_eq!(report["execution"]["status"], "NOT_RUN");

    let report = check(&modified_request(|r| {
        r["launch_file"]["max_instructions"] = json!(10)
    }));
    assert_eq!(report["result_class"], "TIMEOUT");
    assert_eq!(report["execution"]["status"], "BUDGET_EXHAUSTED");
    assert_eq!(report["execution"]["instructions_executed"], 10);

    for edit in [
        (|r: &mut Value| r["unexpected"] = json!(1)) as fn(&mut Value),
        |r| r["role"] = json!("candidate"),
        |r| r["outputs"]["words"] = json!(352),
        |r| r["guards"][0]["base"] = json!("0x10100"),
        |r| r["launch_file"]["block"] = json!([64, 1, 1]),
        |r| r["enable_pending_ops"] = json!(["s_max_i32", "s_max_i32"]),
        |r| r["object"]["sha256"] = json!("XYZ"),
    ] {
        let report = check(&modified_request(edit));
        assert_eq!(report["result_class"], "ERROR");
        assert!(
            report["reasons"][0]
                .as_str()
                .unwrap()
                .starts_with("INVALID_REQUEST")
        );
    }
    // The object hash is echoed unchanged; the verifier reads the source.
    let report = check(&modified_request(|r| {
        r["object"]["sha256"] = json!("a".repeat(64))
    }));
    assert_eq!(report["result_class"], "PASS");
    assert_eq!(report["object"]["sha256"], "a".repeat(64));
}

#[test]
fn report_identifies_profile_and_verifier() {
    let report = check(&cases::case_dir(&corpus(), "seed-xunit_chain-twin").join("request.json"));
    assert_eq!(
        report["profile"]["allowlist_sha256"],
        "d627ed3f3ac8c1852e4a63cce6109a2faa70d0ef9a2ba1db000d61ffec11a919"
    );
    assert_eq!(report["profile"]["pending_ops"], json!(["s_max_i32"]));
    assert_eq!(
        report["target"]["directive"],
        "amdgcn-amd-amdhsa-unknown-gfx1250"
    );
    assert_eq!(report["target"]["features"], json!([]));
    let digest = Command::new("sha256sum")
        .arg(env!("CARGO_BIN_EXE_volta"))
        .output()
        .unwrap();
    let digest = String::from_utf8(digest.stdout).unwrap();
    assert_eq!(
        report["verifier"]["exe_sha256"].as_str().unwrap(),
        digest.split_whitespace().next().unwrap()
    );
    assert_eq!(report["verifier"]["version"], env!("CARGO_PKG_VERSION"));
    assert_ne!(report["verifier"]["git_rev"], "");
}

#[test]
fn embedded_allowlist_equals_the_frozen_subset_file() {
    let subset: Value =
        serde_json::from_slice(&fs::read(corpus().join("schema/m1_int_subset.json")).unwrap())
            .unwrap();
    let sorted = |value: &Value| {
        let mut items: Vec<String> = match value {
            Value::Array(items) => items
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect(),
            Value::Object(map) => map.keys().cloned().collect(),
            _ => unreachable!(),
        };
        items.sort();
        items
    };
    let canonical = json!({
        "body": sorted(&subset["body_allowed_ops"]),
        "control_only": sorted(&subset["control_only_ops"]),
        "pending": sorted(&subset["pending_ops"]),
        "wrapper": sorted(&subset["wrapper_mnemonics"]),
    });
    assert_eq!(
        canonical.to_string(),
        volta_amdgpu_analysis::m1::canonical_allowlist()
    );
    assert_eq!(
        subset["allowlist_sha256"],
        "d627ed3f3ac8c1852e4a63cce6109a2faa70d0ef9a2ba1db000d61ffec11a919"
    );
}

//! `volta check-program` under the gfx1250 B0 m2 MODE profile
//! (`tests/gfx1250/m2`): fuzzer-rendered MODE seeds, op probes and calibration
//! controls with independent expected words, MODE read -> write orderings
//! without an SA_SDST wait (unresolved), unsupported MODE forms, the pending
//! gate and the unchanged
//! m1 refusal. Expectations are written by the fixture generator,
//! never copied from a verifier report.
#[path = "support/gfx1250_cases.rs"]
mod cases;

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/gfx1250/m2")
}

fn schema_dir(profile: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../tests/gfx1250/{profile}/schema"))
}

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "volta-m2-check-{}-{}",
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

const M1_CHECKS: [&str; 7] = [
    "trace_completeness",
    "wait_readiness",
    "barrier_generations",
    "memory_conflicts",
    "object_bounds",
    "prefetch_policy",
    "store_completion",
];

/// Run one request; its exit status must equal the report's class and the
/// report must validate against the frozen schema of its profile.
fn check(request: &Path, profile: &str) -> Value {
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
    let schema_name = format!("ptxv-{profile}-report.schema.json");
    let schema: Value =
        serde_json::from_slice(&fs::read(schema_dir(profile).join(schema_name)).unwrap()).unwrap();
    validate(&schema, &schema, &report, "report").unwrap();
    let ids: Vec<_> = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_str().unwrap())
        .collect();
    let mut expected: Vec<&str> = M1_CHECKS.to_vec();
    if profile == "m2" {
        expected.push("mode_order");
    }
    assert_eq!(ids, expected);
    for check in report["checks"].as_array().unwrap() {
        if check["certified"] == json!(true) {
            assert_eq!(check["status"], "FAIL");
            assert!(!check["witnesses"].as_array().unwrap().is_empty());
        }
    }
    report
}

// ---------------------------------------------------------------------------
// The JSON Schema subset interpreter of the m1 tests (no new dependency).
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

// ---------------------------------------------------------------------------
// Fixture helpers shared in shape with the m1 tests.
// ---------------------------------------------------------------------------

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

fn json_file(path: PathBuf) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn mode_order(report: &Value) -> &Value {
    report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == "mode_order")
        .unwrap()
}

#[test]
fn every_m2_case_has_its_independently_written_result() {
    let cases = json_file(corpus().join("cases.json"));
    let words = json_file(corpus().join("expected-words.json"));
    let provenance = json_file(corpus().join("provenance.json"));
    let mut classes = std::collections::BTreeMap::<String, usize>::new();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let profile = case["profile"].as_str().unwrap();
        let expect = &case["expect"];
        let report = check(
            &cases::case_dir(&corpus(), &name).join("request.json"),
            profile,
        );
        *classes
            .entry(report["result_class"].as_str().unwrap().to_owned())
            .or_default() += 1;
        assert_eq!(
            report["result_class"], expect["class"],
            "{name}: {}",
            report["reasons"]
        );
        assert_eq!(report["role"], case["role"], "{name}");
        if let Some(status) = expect.get("mode_order") {
            assert_eq!(mode_order(&report)["status"], *status, "{name}");
        }
        if let Some(kind) = expect.get("witness") {
            let check = mode_order(&report);
            assert_eq!(check["certified"], true, "{name}");
            let witness = check["witnesses"]
                .as_array()
                .unwrap()
                .iter()
                .find(|w| w["kind"] == *kind)
                .unwrap_or_else(|| panic!("{name}: no {kind} witness"));
            // The witness names the MODE read and the later MODE write.
            let events = witness["events"].as_array().unwrap();
            assert_eq!(events.len(), 2, "{name}");
            assert!(
                events[0]["text"]
                    .as_str()
                    .unwrap()
                    .starts_with("s_getreg_b32 ")
            );
            assert!(
                ["s_round_mode ", "s_denorm_mode "]
                    .iter()
                    .any(|op| events[1]["text"].as_str().unwrap().starts_with(op)),
                "{name}"
            );
            assert!(events[0]["line"].as_u64() < events[1]["line"].as_u64());
        }
        if expect.get("unresolved") == Some(&json!(true)) {
            assert!(
                !mode_order(&report)["unresolved"]
                    .as_array()
                    .unwrap()
                    .is_empty(),
                "{name}"
            );
        }
        if let Some(reasons) = expect.get("refused") {
            let mut found: Vec<_> = report["parser"]["refused_ops"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r["reason"].clone())
                .collect();
            found.sort_by_key(|r| r.to_string());
            found.dedup();
            assert_eq!(Value::Array(found), *reasons, "{name}");
            assert_eq!(report["execution"]["status"], "NOT_RUN", "{name}");
        }
        if expect.get("aborted") == Some(&json!(true)) {
            assert_eq!(report["execution"]["status"], "ABORTED", "{name}");
        }
        if expect.get("words") == Some(&json!(true)) {
            assert_eq!(report["outputs"]["status"], "COMPLETE", "{name}");
            assert_eq!(report["outputs"]["values"], words[name], "{name}");
            // Encoded size and every s_wait_alu encoding agree with clang 24.
            let facts = &provenance["assembly"][name];
            assert_eq!(report["object"]["sha256"], facts["object_sha256"]);
            let objdump = facts["objdump"].as_str().unwrap();
            let (start, end, padding) = objdump_extent(objdump);
            assert_eq!(report["kernel"]["size_bytes"], end - start, "{name}");
            assert_eq!(
                report["kernel"]["padding_after_size_bytes"], padding,
                "{name}"
            );
            let source =
                fs::read_to_string(cases::case_dir(&corpus(), &name).join("kernel.s")).unwrap();
            assert_wait_alu_encodings(&source, objdump);
        }
        if profile == "m2" {
            assert_eq!(report["profile"]["id"], "gfx1250-b0-m2-mode/1");
        } else {
            assert_eq!(report["profile"]["id"], "gfx1250-b0-m1-int/1");
            assert!(report.get("mode").is_none(), "{name}");
        }
    }
    assert_eq!(
        classes,
        [
            ("PASS".to_owned(), 36),
            ("UNKNOWN_UNSUPPORTED".to_owned(), 27),
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn mode_record_follows_the_descriptor_and_every_access() {
    let report = check(
        &cases::case_dir(&corpus(), "seed-mode_mix-cand").join("request.json"),
        "m2",
    );
    assert_eq!(report["mode"]["entry"]["fp_round"], 0);
    assert_eq!(report["mode"]["entry"]["fp_denorm"], 15);
    let accesses: Vec<_> = report["mode"]["accesses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            (
                a["op"].as_str().unwrap().to_owned(),
                a["access"].as_str().unwrap().to_owned(),
                a["offset"].as_u64().unwrap(),
                a["value"].as_u64().unwrap(),
            )
        })
        .collect();
    let expected: Vec<(String, String, u64, u64)> = [
        ("s_getreg_b32", "read", 4, 15),
        ("s_setreg_b32", "write", 0, 6),
        ("s_getreg_b32", "read", 0, 6),
        ("s_denorm_mode", "write", 4, 3),
        ("s_getreg_b32", "read", 4, 3),
        ("s_setreg_imm32_b32", "write", 0, 2),
        ("s_getreg_b32", "read", 0, 2),
    ]
    .iter()
    .map(|(o, a, f, v)| (o.to_string(), a.to_string(), *f, *v))
    .collect();
    assert_eq!(accesses, expected);
    assert_eq!(
        report["mode"]["final"],
        json!({"fp_round": 2, "fp_denorm": 3})
    );
    assert!(
        report["assumptions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["id"] == "A-SETREG-IMM32-4BIT")
    );
}

#[test]
fn m2_profile_block_lists_the_pending_mode_operations_and_rules() {
    let report = check(
        &cases::case_dir(&corpus(), "seed-mode_getreg_wait-cand").join("request.json"),
        "m2",
    );
    let profile = &report["profile"];
    assert_eq!(profile["dependency_profile"], "gfx1250-b0-sched0-m2/1");
    assert_eq!(
        profile["pending_ops"],
        json!([
            "s_max_i32",
            "s_getreg_b32",
            "s_setreg_b32",
            "s_setreg_imm32_b32",
            "s_round_mode",
            "s_denorm_mode"
        ])
    );
    let rule = |id: &str| {
        profile["rules"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == id)
            .unwrap_or_else(|| panic!("rule {id}"))["basis"]
            .clone()
    };
    assert_eq!(
        rule("M2-getreg-mode-then-mode-write-without-sa-sdst-wait-unresolved"),
        "conservative_policy"
    );
    assert!(
        !profile["rules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["id"] == "D7-no-m1-required-waits")
    );
}

#[test]
fn embedded_m2_allowlist_equals_the_frozen_subset_file() {
    let subset = json_file(corpus().join("schema/m2_mode_subset.json"));
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
    let profile = volta_amdgpu_analysis::m1::Profile::M2Mode;
    assert_eq!(
        canonical.to_string(),
        volta_amdgpu_analysis::m1::canonical_allowlist_for(profile)
    );
    let report = check(
        &cases::case_dir(&corpus(), "seed-mode_getreg_wait-cand").join("request.json"),
        "m2",
    );
    assert_eq!(
        report["profile"]["allowlist_sha256"],
        subset["allowlist_sha256"]
    );
    // m2 is a superset of m1: every m1 body, wrapper and control-only op.
    let m1 = json_file(schema_dir("m1").join("m1_int_subset.json"));
    for key in [
        "body_allowed_ops",
        "wrapper_mnemonics",
        "control_only_ops",
        "pending_ops",
    ] {
        for op in sorted(&m1[key]) {
            assert!(sorted(&subset[key]).contains(&op), "{key}: {op}");
        }
    }
}

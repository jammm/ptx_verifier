//! `volta check-program` under the gfx1250 B0 litmus/2 profile
//! (`tests/gfx1250/litmus2`): the two-wave `barjoin` template in its waited
//! form with documented-contract images (STATE_PRIV named barrier id, clear
//! completion bit), the litmus/1 seeds and controls under /2, harmful
//! negatives with certified witnesses, refusals (unwaited JOIN forms, a
//! STATE_PRIV read without s_wait_idle, of another register or of an
//! unconsumed completion, a barrier id beyond the allocation, a wait that
//! never completes, unqualified templates), the litmus/1 profile on a barjoin
//! kernel and exact request validation. Expectations are written by the
//! fixture generator, never copied from a verifier report.
#[path = "support/gfx1250_cases.rs"]
mod cases;

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/gfx1250/litmus2")
}

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "volta-litmus2-check-{}-{}",
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

const CHECKS: [&str; 7] = [
    "trace_completeness",
    "wait_readiness",
    "barrier_generations",
    "memory_conflicts",
    "object_bounds",
    "prefetch_policy",
    "store_completion",
];

/// Run one request; the exit status must equal the report's class and the
/// report must validate against the frozen schema of its profile.
fn check(request: &Path, schema_file: &Path) -> Value {
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
    let schema = json_file(schema_file.to_path_buf());
    validate(&schema, &schema, &report, "report").unwrap();
    let ids: Vec<_> = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, CHECKS);
    for check in report["checks"].as_array().unwrap() {
        if check["certified"] == json!(true) {
            assert_eq!(check["status"], "FAIL");
            assert!(!check["witnesses"].as_array().unwrap().is_empty());
        }
    }
    report
}

fn litmus2_schema() -> PathBuf {
    corpus().join("schema/ptxv-litmus2-report.schema.json")
}

fn json_file(path: PathBuf) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[test]
fn every_litmus2_case_has_its_independently_written_result() {
    let cases = json_file(corpus().join("cases.json"));
    let words = json_file(corpus().join("expected-words.json"));
    let provenance = json_file(corpus().join("provenance.json"));
    let mut classes = std::collections::BTreeMap::<String, usize>::new();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let expect = &case["expect"];
        assert_eq!(
            provenance["assembly"][name]["assembles"],
            json!(true),
            "{name}"
        );
        let folder = cases::case_dir(&corpus(), &name);
        let request = json_file(folder.join("request.json"));
        assert_eq!(
            request["object"]["sha256"], provenance["assembly"][name]["object_sha256"],
            "{name}"
        );
        let report = check(&folder.join("request.json"), &litmus2_schema());
        let class = report["result_class"].as_str().unwrap();
        assert_eq!(class, expect["class"], "{name}: {}", report["reasons"]);
        *classes.entry(class.to_owned()).or_default() += 1;
        if expect["words"] == json!(true) {
            assert_eq!(report["outputs"]["status"], "COMPLETE", "{name}");
            assert_eq!(report["outputs"]["values"], words[name], "{name}");
        } else {
            assert_ne!(class, "PASS", "{name}");
        }
        if let Some(id) = expect["check"].as_str() {
            let check = report["checks"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["id"] == id)
                .unwrap();
            assert_eq!(check["status"], "FAIL", "{name}");
            assert_eq!(check["certified"], json!(true), "{name}");
            assert!(
                check["witnesses"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|w| w["kind"] == expect["witness"]),
                "{name}: {}",
                check["witnesses"]
            );
        }
        if let Some(text) = expect["abort"].as_str() {
            let abort = report["execution"]["abort"]["reason"]
                .as_str()
                .unwrap_or_default();
            assert!(abort.contains(text), "{name}: {abort}");
            assert_eq!(report["execution"]["status"], "ABORTED", "{name}");
        }
        if let Some(reason) = expect["refused"].as_str() {
            assert!(
                report["parser"]["refused_ops"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|op| op["reason"] == reason),
                "{name}: {}",
                report["parser"]["refused_ops"]
            );
        }
    }
    assert_eq!(classes.get("PASS"), Some(&31));
    assert_eq!(classes.get("REJECT"), Some(&2));
    assert_eq!(classes.get("UNKNOWN_UNSUPPORTED"), Some(&14));
}

#[test]
fn barjoin_reads_the_joined_barrier_id_with_a_clear_completion_bit() {
    let report = check(
        &cases::case_dir(&corpus(), "control-barjoin_wait_cured-cand").join("request.json"),
        &litmus2_schema(),
    );
    assert_eq!(report["result_class"], "PASS");
    // b = 4: NAMED_BARRIER_ID 4 in bits 8:4, NAMED_BARRIER_COMPLETE clear.
    assert_eq!(report["outputs"]["values"], json!([0x40]));
    assert_eq!(report["profile"]["id"], "gfx1250-b0-litmus/2");
    assert_eq!(
        report["profile"]["dependency_profile"],
        "gfx1250-b0-sched0-litmus/2"
    );
    let waves = report["execution"]["waves"].as_array().unwrap();
    assert_eq!(waves.len(), 2);
    assert!(waves.iter().all(|w| w["terminated_by"] == "s_endpgm"));
    assert_eq!(report["checks"][2]["status"], "PASS");
    let ids: Vec<_> = report["assumptions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["id"].as_str().unwrap().to_owned())
        .collect();
    for id in [
        "P-BJ-CONSUMED",
        "Gfx1250B0LitmusBarrierProfile",
        "AllocatedNamedBarriers(4)",
    ] {
        assert!(ids.iter().any(|i| i == id), "{id}: {ids:?}");
    }
    let rules: Vec<_> = report["profile"]["rules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_owned())
        .collect();
    for prefix in [
        "J1-", "J2-", "J3-", "J4-", "J5-", "J6-", "J7-", "B3-", "A4-",
    ] {
        assert!(rules.iter().any(|r| r.starts_with(prefix)), "{prefix}");
    }
}

#[test]
fn the_litmus_one_profile_still_refuses_barjoin() {
    let folder = cases::case_dir(&corpus(), "seed-barjoin_wait-cand");
    let mut request = json_file(folder.join("request.json"));
    request["schema"] = json!("ptxv-litmus-request/1");
    request["profile"] = json!("gfx1250-b0-litmus/1");
    request["source"]["path"] = json!(folder.join("kernel.s").to_str().unwrap());
    let litmus1 = corpus().join("../litmus/schema/ptxv-litmus-report.schema.json");
    // litmus/1 has one wave only: a 64-workitem block is an invalid request.
    let path = scratch().join("request.json");
    fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
    let report = check(&path, &litmus1);
    assert_eq!(report["result_class"], "ERROR");
    request["launch_file"]["block"] = json!([32, 1, 1]);
    fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
    let report = check(&path, &litmus1);
    assert_eq!(report["result_class"], "UNKNOWN_UNSUPPORTED");
    let refused = report["parser"]["refused_ops"].as_array().unwrap();
    // (GETREG is named a MODE access by the earlier refusal rule.)
    assert!(refused.iter().any(|op| op["mnemonic"] == "s_getreg_b32"));
    for mnemonic in ["s_barrier_join", "s_wait_idle"] {
        assert!(
            refused
                .iter()
                .any(|op| op["mnemonic"] == mnemonic
                    && op["reason"] == "LITMUS_TEMPLATE_NOT_QUALIFIED"),
            "{mnemonic}: {refused:?}"
        );
    }
}

#[test]
fn litmus2_requests_are_validated_exactly() {
    let folder = cases::case_dir(&corpus(), "seed-barjoin_wait-cand");
    let good = json_file(folder.join("request.json"));
    let mut variants = Vec::new();
    let mut block = good.clone();
    block["launch_file"]["block"] = json!([96, 1, 1]);
    variants.push(("three waves", block));
    let mut profile = good.clone();
    profile["profile"] = json!("gfx1250-b0-litmus/1");
    variants.push(("profile of another schema", profile));
    let mut grid = good.clone();
    grid["launch_file"]["grid"] = json!([2, 1, 1]);
    variants.push(("two workgroups", grid));
    for (what, request) in variants {
        let mut request = request;
        request["source"]["path"] = json!(folder.join("kernel.s").to_str().unwrap());
        let path = scratch().join("request.json");
        fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
        let report = check(&path, &litmus2_schema());
        assert_eq!(report["result_class"], "ERROR", "{what}");
        assert!(
            report["reasons"][0]
                .as_str()
                .unwrap()
                .starts_with("INVALID_REQUEST"),
            "{what}"
        );
    }
}

#[test]
fn litmus2_allowlist_hash_matches_the_frozen_subset() {
    let subset = json_file(corpus().join("schema/litmus_b0_t2_subset.json"));
    assert_eq!(
        subset["allowlist_sha256"],
        "d1d877a297c3fc20af348270bf3d13446d093fcccd65e2e515e28d57909cae7b"
    );
    let report = check(
        &cases::case_dir(&corpus(), "seed-bufload-cand").join("request.json"),
        &litmus2_schema(),
    );
    assert_eq!(
        report["profile"]["allowlist_sha256"],
        subset["allowlist_sha256"]
    );
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

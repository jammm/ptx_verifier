//! `volta check-program` under the gfx1250 B0 litmus profile
//! (`tests/gfx1250/litmus`): fuzzer-rendered bufload, bufstore and asynclds
//! kernels with their documented-contract images, harmful negatives with
//! certified witnesses (missing waits, guard accesses, the unclamped-footprint
//! envelope, async/DS races), refusals (unbound, swizzled, stride-scaled,
//! bounds-disabled and unaligned resources, indexed addressing, out-of-profile and
//! unaligned async offsets, unqualified templates, LDS premises) and the
//! m1 profile on a litmus kernel. Expectations are written by the
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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/gfx1250/litmus")
}

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "volta-litmus-check-{}-{}",
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

fn litmus_schema() -> PathBuf {
    corpus().join("schema/ptxv-litmus-report.schema.json")
}

fn json_file(path: PathBuf) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
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

#[test]
fn every_litmus_case_has_its_independently_written_result() {
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
        let report = check(&folder.join("request.json"), &litmus_schema());
        let class = report["result_class"].as_str().unwrap();
        assert_eq!(class, expect["class"], "{name}: {}", report["reasons"]);
        *classes.entry(class.to_owned()).or_default() += 1;
        if expect["words"] == json!(true) {
            assert_eq!(report["outputs"]["status"], "COMPLETE", "{name}");
            assert_eq!(report["outputs"]["values"], words[name], "{name}");
            for record in report["litmus"]["buffer"].as_array().unwrap() {
                let dwords = record["dwords"].as_u64().unwrap();
                let lanes = record["lanes"].as_u64().unwrap();
                assert_eq!(
                    record["dwords_in_range"].as_u64().unwrap()
                        + record["dwords_out_of_range"].as_u64().unwrap(),
                    dwords * lanes,
                    "{name}"
                );
            }
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
        if let Some(status) = expect["parser"].as_str() {
            assert_eq!(report["parser"]["status"], status, "{name}");
        }
    }
    assert_eq!(classes.get("PASS"), Some(&36));
    assert_eq!(classes.get("REJECT"), Some(&7));
    assert_eq!(classes.get("UNKNOWN_UNSUPPORTED"), Some(&18));
}

#[test]
fn range_checks_follow_the_documented_rule() {
    // num_records 28 and a 16-byte load at byte 16.
    // DWORDs ending at 20, 24 and 28 are in range; the fourth (ending at 32) is
    // not and reads zero.
    let report = check(
        &cases::case_dir(&corpus(), "seed-bufload-cand").join("request.json"),
        &litmus_schema(),
    );
    assert_eq!(report["result_class"], "PASS");
    let buffer = &report["litmus"]["buffer"][0];
    assert_eq!(buffer["in_range_masks"], json!([7]));
    assert_eq!(buffer["num_records"], 28);
    let values = report["outputs"]["values"].as_array().unwrap();
    for lane in 0..32 {
        let words: Vec<_> = values[4 * lane..4 * lane + 4].to_vec();
        assert_eq!(
            words,
            vec![
                json!(0xDA7A0004u32),
                json!(0xDA7A0005u32),
                json!(0xDA7A0006u32),
                json!(0)
            ]
        );
    }
    let ids: Vec<_> = report["assumptions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["id"].as_str().unwrap().to_owned())
        .collect();
    for id in ["P-ASYNC-LDS-64K", "B3-OOB-SELECT-RAW", "A-INIT-LAUNCHER"] {
        assert!(ids.iter().any(|i| i == id), "{id}");
    }
    // OOB_SELECT=1 with STRIDE 8 below num_records 64: only the two DWORDs
    // within the stride are written.
    let report = check(
        &cases::case_dir(&corpus(), "control-bufstore_oob1_stride-cand").join("request.json"),
        &litmus_schema(),
    );
    assert_eq!(report["litmus"]["buffer"][0]["in_range_masks"], json!([3]));
    assert_eq!(report["litmus"]["buffer"][0]["oob_select"], json!(true));
}

#[test]
fn the_m1_profile_still_refuses_litmus_operations() {
    let folder = cases::case_dir(&corpus(), "seed-bufload-cand");
    let litmus = json_file(folder.join("request.json"));
    let words = litmus["outputs"]["words"].as_u64().unwrap();
    let base = litmus["objects"][0]["base"].as_str().unwrap();
    let request = json!({
        "schema": "ptxv-m1-request/1",
        "profile": "gfx1250-b0-m1-int/1",
        "enable_pending_ops": [],
        "role": "cand",
        "kernel": "kernel",
        "source": {"path": folder.join("kernel.s").to_str().unwrap(), "sha256": litmus["source"]["sha256"]},
        "object": litmus["object"],
        "launch_file": {
            "block": [32, 1, 1], "grid": [1, 1, 1], "kernel": "kernel",
            "arrays": [{"name": "out", "base": base, "elem_width": 4, "len": words, "kind": "out"}],
            "params": [{"ptr": "out"}], "max_instructions": 100000,
        },
        "backing_allocation": litmus["backing_allocation"],
        "guards": litmus["guards"],
        "outputs": {"contract": "raw-u32", "object": "out", "words": words},
    });
    let path = scratch().join("m1.request.json");
    fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
    let report = check(
        &path,
        &corpus().join("../m1/schema/ptxv-m1-report.schema.json"),
    );
    assert_eq!(report["result_class"], "UNKNOWN_UNSUPPORTED");
    assert!(
        report["parser"]["refused_ops"]
            .as_array()
            .unwrap()
            .iter()
            .any(|op| op["mnemonic"] == "buffer_load_b128" && op["reason"] == "NOT_IN_PROFILE")
    );
}

#[test]
fn litmus_requests_are_validated_exactly() {
    let folder = cases::case_dir(&corpus(), "seed-bufstore-cand");
    let good = json_file(folder.join("request.json"));
    let mut variants = Vec::new();
    let mut gap = good.clone();
    gap["objects"][0]["base"] = json!("0x10200");
    variants.push(("non-contiguous or misplaced object", gap));
    let mut init = good.clone();
    init["objects"][0]["initial"] = json!({"fill_u32": "0xeeeeeeee", "index_base_u32": "0x0"});
    variants.push(("two initial patterns", init));
    let mut name = good.clone();
    name["objects"][0]["name"] = json!("region");
    variants.push(("unknown object name", name));
    let mut words = good.clone();
    words["outputs"]["words"] = json!(65);
    variants.push(("output words", words));
    let mut pending = good.clone();
    pending["enable_pending_ops"] = json!([]);
    variants.push(("m1 field", pending));
    for (what, request) in variants {
        let mut request = request;
        request["source"]["path"] = json!(folder.join("kernel.s").to_str().unwrap());
        let path = scratch().join("request.json");
        fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
        let report = check(&path, &litmus_schema());
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

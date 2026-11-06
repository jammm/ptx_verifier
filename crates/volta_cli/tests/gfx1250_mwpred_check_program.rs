//! `volta check-program` under the gfx1250 B0 multi-workgroup predecessor
//! profile `gfx1250-b0-mwpred/1` (`tests/gfx1250/mwpred`): fixed fuzzer
//! predecessor grids whose every workgroup is analysed and composed, with the
//! fuzzer's own reference words (placement evidence declared opaque); harmful
//! negatives with certified witnesses (workgroups writing the same bytes, a
//! cross-wave store conflict, a VALU reading a WMMA result too early, a guard
//! store); refusals (a message return that is not drained by the next
//! instruction, an opaque value used as an address, the `gfx1250-b0-mw/1`
//! profile on these kernels) and malformed grids. Expectations are written by
//! the fixture generator, never copied from a verifier report.
#[path = "support/gfx1250_cases.rs"]
mod cases;

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/gfx1250/mwpred")
}

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "volta-mwpred-check-{}-{}",
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

const CHECKS: [&str; 10] = [
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

const PASS_CASES: usize = 4;
const REJECT_CASES: usize = 4;
const UNKNOWN_CASES: usize = 3;
const ERROR_CASES: usize = 3;

fn json_file(path: PathBuf) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

/// The frozen report schema of a request's profile.
fn schema_for(request: &Value) -> PathBuf {
    let name = match request["schema"].as_str().unwrap() {
        "ptxv-mwpred-request/1" => "schema/ptxv-mwpred-report.schema.json",
        "ptxv-mw-request/1" => "../mw_wmma/schema/ptxv-mw-report.schema.json",
        other => panic!("no schema for {other}"),
    };
    corpus().join(name)
}

/// Run one request; the exit status must equal the report's class and the
/// report must validate against the frozen schema of its profile.
fn check(request_path: &Path) -> Value {
    let request = json_file(request_path.to_path_buf());
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
        .arg(request_path)
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
        request_path.display()
    );
    let schema = json_file(schema_for(&request));
    validate(&schema, &schema, &report, "report").unwrap();
    {
        let ids: Vec<_> = report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["id"].as_str().unwrap())
            .collect();
        if request["schema"] == "ptxv-mwpred-request/1" {
            assert_eq!(ids, CHECKS);
        } else {
            assert_eq!(ids, CHECKS[..9]);
        }
    }
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

fn declared(request: &Value, key: &str) -> Vec<usize> {
    request["outputs"][key]
        .as_array()
        .map(|ranges| {
            ranges
                .iter()
                .flat_map(|r| {
                    let first = r[0].as_u64().unwrap() as usize;
                    first..first + r[1].as_u64().unwrap() as usize
                })
                .collect()
        })
        .unwrap_or_default()
}

fn find_check<'a>(report: &'a Value, id: &str) -> &'a Value {
    report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == id)
        .unwrap()
}

#[test]
fn every_case_has_its_independently_written_result() {
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
        let report = check(&folder.join("request.json"));
        let class = report["result_class"].as_str().unwrap();
        assert_eq!(class, expect["class"], "{name}: {}", report["reasons"]);
        *classes.entry(class.to_owned()).or_default() += 1;
        if expect["words"] == json!(true) {
            // Exact words everywhere the reference has one; null exactly at
            // the declared opaque and unwritten words.
            let values = report["outputs"]["values"].as_array().unwrap();
            assert_eq!(values, words[name].as_array().unwrap(), "{name}");
            let mut nulls = declared(&request, "opaque");
            nulls.extend(declared(&request, "not_written"));
            nulls.sort_unstable();
            let found: Vec<_> = (0..values.len()).filter(|&i| values[i].is_null()).collect();
            assert_eq!(found, nulls, "{name}");
            for &i in &declared(&request, "opaque") {
                assert!(
                    report["outputs"]["undefined"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|u| u["index"] == json!(i) && u["reason"] == "OPAQUE"),
                    "{name}: word {i}"
                );
            }
        } else {
            assert_ne!(class, "PASS", "{name}");
        }
        if let Some(id) = expect["check"].as_str() {
            let check = find_check(&report, id);
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
        if let Some(id) = expect["check_unknown"].as_str() {
            let check = find_check(&report, id);
            assert_eq!(check["status"], "UNKNOWN", "{name}");
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
        if let Some(reason) = expect["undefined"].as_str() {
            assert!(
                report["outputs"]["undefined"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|u| u["reason"] == reason),
                "{name}: {}",
                report["outputs"]["undefined"]
            );
        }
        if expect.get("abort").is_some() {
            assert_eq!(report["execution"]["status"], "ABORTED", "{name}");
        }
    }

    let count = |class: &str| classes.get(class).copied().unwrap_or(0);
    assert_eq!(count("PASS"), PASS_CASES);
    assert_eq!(count("REJECT"), REJECT_CASES);
    assert_eq!(count("UNKNOWN_UNSUPPORTED"), UNKNOWN_CASES);
    assert_eq!(count("ERROR"), ERROR_CASES);
}

#[test]
fn every_workgroup_is_analysed_and_composed() {
    let report = check(&cases::case_dir(&corpus(), "pred-wmma-g4").join("request.json"));
    let subset = json_file(corpus().join("schema/mw_pred_b0.json"));
    assert_eq!(report["profile"]["id"], "gfx1250-b0-mwpred/1");
    assert_eq!(
        report["profile"]["dependency_profile"],
        "gfx1250-b0-sched0-mwpred/1"
    );
    assert_eq!(
        report["profile"]["allowlist_sha256"],
        subset["allowlist_sha256"]
    );
    let workgroups = report["workgroups"].as_array().unwrap();
    assert_eq!(workgroups.len(), 4);
    for (k, workgroup) in workgroups.iter().enumerate() {
        assert_eq!(workgroup["workgroup"], json!(k));
        assert_eq!(workgroup["class"], "PASS");
    }
    assert_eq!(find_check(&report, "workgroup_isolation")["status"], "PASS");
    assert_eq!(report["outputs"]["status"], "COMPLETE_AS_DECLARED");
    let ids: Vec<_> = report["assumptions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["id"].as_str().unwrap().to_owned())
        .collect();
    for id in [
        "A-GRID-EVERY-WORKGROUP",
        "A-WORKGROUP-ISOLATION",
        "A-SE-ID-OPAQUE",
        "A-REPLAY-MODE0",
        "A-WMMA-EXACT-REGIME",
    ] {
        assert!(ids.iter().any(|i| i == id), "{id}");
    }
}

#[test]
fn overlapping_workgroups_are_a_certified_failure() {
    let report =
        check(&cases::case_dir(&corpus(), "neg-pred-workgroups-overlap").join("request.json"));
    let isolation = find_check(&report, "workgroup_isolation");
    assert_eq!(isolation["status"], "FAIL");
    assert_eq!(isolation["certified"], json!(true));
    let witness = &isolation["witnesses"][0];
    assert_eq!(witness["kind"], "CROSS_WORKGROUP_WRITE_OVERLAP");
    assert_eq!(witness["workgroups"].as_array().unwrap().len(), 2);
}

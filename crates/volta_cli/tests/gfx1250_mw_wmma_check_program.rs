//! `volta check-program` under the gfx1250 B0 WMMA and multiwave profiles
//! (`tests/gfx1250/mw_wmma`): fuzzer-rendered single-wave kernels with a BF16
//! WMMA chain and multiwave aggressor/victim experiments of 2..8 waves, with
//! the fuzzer's own reference words; harmful negatives with certified
//! witnesses (WMMA spacing below the documented counts, same-argument
//! back-to-back WMMA, missing kernarg waits, a guard store, a cross-wave
//! conflict); refusals (unlisted WMMA chains below the policy, data outside
//! the exact-representable regime, floating-point bodies, opaque evidence used
//! as an address, pending MODE operations, undeclared evidence words); the
//! m2 profile on these kernels and malformed requests. Expectations
//! are written by the fixture generator, never copied from a verifier report.
#[path = "support/gfx1250_cases.rs"]
mod cases;

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/gfx1250/mw_wmma")
}

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "volta-mw-wmma-check-{}-{}",
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

const CHECKS: [&str; 9] = [
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

fn json_file(path: PathBuf) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

/// The frozen report schema of a request's profile.
fn schema_for(request: &Value) -> PathBuf {
    let name = match request["schema"].as_str().unwrap() {
        "ptxv-wmma-request/1" => "schema/ptxv-wmma-report.schema.json",
        "ptxv-mw-request/1" => "schema/ptxv-mw-report.schema.json",
        "ptxv-m2-request/1" => "../m2/schema/ptxv-m2-report.schema.json",
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
    if request["schema"] != json!("ptxv-m2-request/1") {
        let ids: Vec<_> = report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, CHECKS);
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
    assert_eq!(classes.get("PASS"), Some(&19));
    assert_eq!(classes.get("REJECT"), Some(&9));
    assert_eq!(classes.get("UNKNOWN_UNSUPPORTED"), Some(&13));
}

#[test]
fn reports_carry_the_profiles_and_their_frozen_allowlists() {
    for (case, subset, profile) in [
        (
            "wmma-int_chain_wmma-cand",
            "schema/wmma_b0.json",
            "gfx1250-b0-wmma/1",
        ),
        (
            "mw-mw_wmma_int8-cand",
            "schema/mw_b0.json",
            "gfx1250-b0-mw/1",
        ),
    ] {
        let report = check(&cases::case_dir(&corpus(), &case).join("request.json"));
        let subset = json_file(corpus().join(subset));
        assert_eq!(report["profile"]["id"], profile);
        assert_eq!(
            report["profile"]["allowlist_sha256"],
            subset["allowlist_sha256"]
        );
        let ids: Vec<_> = report["assumptions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["id"].as_str().unwrap().to_owned())
            .collect();
        for id in [
            "A-WMMA-EXACT-REGIME",
            "A-WMMA-UNLISTED-POLICY",
            "A-NO-PREDECESSOR",
        ] {
            assert!(ids.iter().any(|i| i == id), "{case}: {id}");
        }
        assert_eq!(find_check(&report, "wmma_hazards")["status"], "PASS");
    }
    // Eight waves, each ending at s_endpgm; the evidence words are opaque.
    let report = check(&cases::case_dir(&corpus(), "mw-mw_split8-cand").join("request.json"));
    let waves = report["execution"]["waves"].as_array().unwrap();
    assert_eq!(waves.len(), 8);
    assert!(waves.iter().all(|w| w["terminated_by"] == "s_endpgm"));
    assert_eq!(report["outputs"]["status"], "COMPLETE_AS_DECLARED");
    assert_eq!(find_check(&report, "barrier_generations")["status"], "PASS");
    assert_eq!(
        find_check(&report, "wmma_hazards")["status"],
        "NOT_APPLICABLE"
    );
}

#[test]
fn wmma_results_are_exact_only_in_the_representable_regime() {
    // C = 2^23 + 8*lane (register 0 of tile v56); the in-place WMMA adds
    // 32 * 1.0 * 2.0. The second WMMA reads that D as C with A = B = 2.0.
    let report =
        check(&cases::case_dir(&corpus(), "wmma-int_chain_wmma-cand").join("request.json"));
    let request =
        json_file(cases::case_dir(&corpus(), "wmma-int_chain_wmma-cand").join("request.json"));
    let words = request["outputs"]["words"].as_u64().unwrap() as usize;
    let values = report["outputs"]["values"].as_array().unwrap();
    assert_eq!(values.len(), words);
    let expected_d2 = |lane: u32| (8_388_608.0f32 + (8 * lane) as f32 + 64.0 + 128.0).to_bits();
    // v64 (second D, register 0) is the slot after s5, s6 and v2..v6.
    for lane in 0..32u32 {
        assert_eq!(values[7 * 32 + lane as usize], json!(expected_d2(lane)));
    }
    for case in ["ref-wmma-magnitude-2-24", "ref-wmma-denormal-input"] {
        let report = check(&cases::case_dir(&corpus(), &case).join("request.json"));
        assert_eq!(report["result_class"], "UNKNOWN_UNSUPPORTED", "{case}");
        assert_eq!(
            find_check(&report, "wmma_hazards")["status"],
            "PASS",
            "{case}"
        );
    }
}

#[test]
fn the_m2_profile_still_refuses_these_kernels() {
    for (case, mnemonic, reason) in [
        (
            "wmma-int_chain_wmma-cand",
            "v_wmma_f32_16x16x32_bf16",
            "FLOATING_POINT",
        ),
        ("mw-mw_split8-cand", "s_barrier_signal", "NOT_IN_PROFILE"),
    ] {
        let folder = cases::case_dir(&corpus(), &case);
        let mut request = json_file(folder.join("request.json"));
        request["schema"] = json!("ptxv-m2-request/1");
        request["profile"] = json!("gfx1250-b0-m2-mode/1");
        request["launch_file"]["block"] = json!([32, 1, 1]);
        request["outputs"] =
            json!({"contract": "raw-u32", "object": "out", "words": request["outputs"]["words"]});
        request["source"]["path"] = json!(folder.join("kernel.s").to_str().unwrap());
        let path = scratch().join("m2.request.json");
        fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
        let report = check(&path);
        assert_eq!(report["result_class"], "UNKNOWN_UNSUPPORTED", "{case}");
        assert!(
            report["parser"]["refused_ops"]
                .as_array()
                .unwrap()
                .iter()
                .any(|op| op["mnemonic"] == mnemonic && op["reason"] == reason),
            "{case}: {}",
            report["parser"]["refused_ops"]
        );
    }
}

#[test]
fn multiwave_and_wmma_requests_are_validated_exactly() {
    let folder = cases::case_dir(&corpus(), "mw-mw_split8-cand");
    let good = json_file(folder.join("request.json"));
    let mut variants = Vec::new();
    let mut nine = good.clone();
    nine["launch_file"]["block"] = json!([288, 1, 1]);
    variants.push(("nine waves", nine));
    let mut one = good.clone();
    one["launch_file"]["block"] = json!([32, 1, 1]);
    variants.push(("one wave", one));
    let mut overlap = good.clone();
    overlap["outputs"]["not_written"] = overlap["outputs"]["opaque"].clone();
    variants.push(("overlapping declared ranges", overlap));
    let mut outside = good.clone();
    outside["outputs"]["opaque"] = json!([[2079, 4]]);
    variants.push(("declared range outside out", outside));
    let mut missing = good.clone();
    missing["outputs"].as_object_mut().unwrap().remove("opaque");
    variants.push(("missing opaque", missing));
    let mut contract = good.clone();
    contract["outputs"]["contract"] = json!("raw-u32");
    variants.push(("contract", contract));
    let mut wmma =
        json_file(cases::case_dir(&corpus(), "wmma-int_chain_wmma-cand").join("request.json"));
    wmma["outputs"]["opaque"] = json!([]);
    variants.push(("wmma request with multiwave fields", wmma));
    for (what, request) in variants {
        let mut request = request;
        let source = if what.starts_with("wmma") {
            cases::case_dir(&corpus(), "wmma-int_chain_wmma-cand").join("kernel.s")
        } else {
            folder.join("kernel.s")
        };
        request["source"]["path"] = json!(source.to_str().unwrap());
        let path = scratch().join("request.json");
        fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
        let report = check(&path);
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

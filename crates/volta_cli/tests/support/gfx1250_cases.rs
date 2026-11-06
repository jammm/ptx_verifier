//! Materialize one packed gfx1250 check-program case into a scratch folder.
//!
//! A profile directory (`tests/gfx1250/<profile>`) holds genuine kernels in
//! `kernels/`, the profile's common request in `request-template.json`, and
//! one `cases.json` entry per case: its base kernel, optional line edits (the
//! negative and refusal variants), an RFC 7386 merge patch over the template
//! and the object hash of the assembled kernel. The folder this returns holds
//! `kernel.s` and `request.json`, exactly what a client would pass to
//! `volta check-program`.
#![allow(dead_code)]

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use serde_json::{Map, Value, json};

#[path = "../../src/sha256.rs"]
mod sha256;

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn merge(target: &mut Value, patch: &Value) {
    match (target, patch) {
        (Value::Object(target), Value::Object(patch)) => {
            for (key, value) in patch {
                if value.is_null() {
                    target.remove(key);
                } else {
                    merge(target.entry(key.clone()).or_insert(Value::Null), value);
                }
            }
        }
        (target, patch) => *target = patch.clone(),
    }
}

/// The `cases.json` entry of `name`.
pub fn case_entry(profile: &Path, name: &str) -> Value {
    read_json(&profile.join("cases.json"))
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("{}: no case {name}", profile.display()))
        .clone()
}

/// Source text of a case: its base kernel with the case's edits applied.
pub fn kernel_source(profile: &Path, case: &Value) -> String {
    let base = fs::read_to_string(profile.join(case["kernel"].as_str().unwrap())).unwrap();
    let mut lines: Vec<String> = base.split_inclusive('\n').map(str::to_owned).collect();
    let mut edits: Vec<&Value> = case["edits"]
        .as_array()
        .map(|e| e.iter().collect())
        .unwrap_or_default();
    edits.sort_by_key(|edit| std::cmp::Reverse(edit["at"].as_u64().unwrap()));
    for edit in edits {
        let at = edit["at"].as_u64().unwrap() as usize;
        let delete = edit["delete"].as_u64().unwrap() as usize;
        let insert = edit["insert"].as_str().unwrap();
        lines.splice(
            at..at + delete,
            insert.split_inclusive('\n').map(str::to_owned),
        );
    }
    lines.concat()
}

/// Request of a case: template, merge patch, then the source and object
/// identities of this exact kernel.
pub fn request(profile: &Path, case: &Value, source: &str) -> Value {
    let mut request = read_json(&profile.join("request-template.json"));
    if let Some(patch) = case.get("request") {
        merge(&mut request, patch);
    }
    let object = request.as_object_mut().unwrap();
    let mut ordered = Map::new();
    for (key, value) in object.iter() {
        ordered.insert(key.clone(), value.clone());
        if key == "kernel" {
            ordered.insert(
                "source".into(),
                json!({"path": "kernel.s", "sha256": sha256::hex(source.as_bytes())}),
            );
            ordered.insert("object".into(), json!({"sha256": case["object_sha256"]}));
        }
    }
    Value::Object(ordered)
}

static MATERIALIZED: Mutex<BTreeMap<PathBuf, PathBuf>> = Mutex::new(BTreeMap::new());

/// Folder with `kernel.s` and `request.json` of case `name` of `profile`
/// (created once per test process).
pub fn case_dir(profile: &Path, name: &str) -> PathBuf {
    let key = profile.join(name);
    let mut done = MATERIALIZED.lock().unwrap();
    if let Some(dir) = done.get(&key) {
        return dir.clone();
    }
    let case = case_entry(profile, name);
    let source = kernel_source(profile, &case);
    // The rebuilt kernel is byte-identical to the generated one.
    assert_eq!(
        sha256::hex(source.as_bytes()),
        case["source_sha256"].as_str().unwrap(),
        "{}: {name} kernel differs from its recorded source",
        profile.display()
    );
    let dir = std::env::temp_dir().join(format!(
        "volta-gfx1250-case-{}-{}-{}",
        std::process::id(),
        profile.file_name().unwrap().to_string_lossy(),
        name
    ));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("kernel.s"), &source).unwrap();
    let request = request(profile, &case, &source);
    fs::write(
        dir.join("request.json"),
        serde_json::to_vec_pretty(&request).unwrap(),
    )
    .unwrap();
    done.insert(key, dir.clone());
    dir
}

//! Embed the source revision for `check-program` reports. A build outside a
//! git checkout, or without git, reports `unknown` and a dirty tree.
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    for path in [
        "crates",
        "tests",
        "docs",
        "Cargo.toml",
        "Cargo.lock",
        "justfile",
    ] {
        println!("cargo:rerun-if-changed={root}/{path}");
    }
    if let Some(git_dir) = git(&["-C", root, "rev-parse", "--absolute-git-dir"]) {
        for path in ["HEAD", "index"] {
            println!("cargo:rerun-if-changed={git_dir}/{path}");
        }
    }
    if let Some(common) = git(&[
        "-C",
        root,
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
    ]) {
        println!("cargo:rerun-if-changed={common}/refs");
        println!("cargo:rerun-if-changed={common}/packed-refs");
    }
    let revision = git(&["-C", root, "rev-parse", "HEAD"]);
    let dirty = git(&["-C", root, "status", "--porcelain"]).map(|status| !status.is_empty());
    match (revision, dirty) {
        (Some(revision), Some(dirty)) => {
            println!("cargo:rustc-env=VOLTA_GIT_REV={revision}");
            println!("cargo:rustc-env=VOLTA_GIT_DIRTY={dirty}");
        }
        _ => println!("cargo:rustc-env=VOLTA_GIT_REV=unknown"),
    }
}

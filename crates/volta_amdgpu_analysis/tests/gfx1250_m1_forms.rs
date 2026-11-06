//! gfx1250 forms emitted by the m1 integer fuzzer wrapper.
use volta_amdgpu_analysis::{AnalysisConfig, ArrayDef, ArrayKind, analyze, frontend};
use volta_graph::{AmdAluWait, AmdEventKind};

fn native(body: &str) -> String {
    format!(
        r#"
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
.text
.globl k
.type k,@function
k:
s_load_b64 s[2:3], s[0:1], 0x0
{body}
s_endpgm
.size k, .-k
.amdhsa_kernel k
.amdhsa_wavefront_size32 1
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_kernarg_size 8
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.target: amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.kernels:
  - .name: k
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
...
.end_amdgpu_metadata
"#
    )
}

fn config() -> AnalysisConfig {
    let mut config = AnalysisConfig::new((32, 1, 1));
    config.arrays.push(ArrayDef {
        name: "out".into(),
        base: 0x10100,
        elem_width: 4,
        len: 32,
        kind: ArrayKind::Output,
    });
    config
        .params
        .push(volta_amdgpu_analysis::ParamValue::ArrayPtr("out".into()));
    config
}

fn alu_waits(body: &str) -> Vec<u16> {
    let module = frontend::parse(&native(body)).unwrap();
    let analysis = analyze(&module, Some("k"), config()).unwrap();
    analysis.graph().validate().unwrap();
    analysis
        .graph()
        .amd_events()
        .iter()
        .filter_map(|record| match record.kind {
            AmdEventKind::AluWait { wait } => Some(wait.encoded),
            _ => None,
        })
        .collect()
}

/// Encodings printed by llvm-objdump (AMD clang 24 `9fa7f0cb`) for every
/// m1 spelling; recorded in `tests/gfx1250/m1/provenance.json`.
#[test]
fn wait_alu_forms_decode_to_the_assembler_encodings() {
    for (text, encoded) in [
        ("0x0", 0x0000),
        ("depctr_sa_sdst(0)", 0xff9e),
        ("depctr_va_vcc(0)", 0xff9d),
        ("depctr_vm_vsrc(0)", 0xff83),
        ("depctr_va_ssrc(0)", 0xfe9f),
        ("depctr_va_sdst(0)", 0xf19f),
        ("depctr_va_vdst(0)", 0x0f9f),
        ("depctr_va_vdst(0) depctr_sa_sdst(0)", 0x0f9e),
        ("0xffff", 0xffff),
    ] {
        assert_eq!(
            alu_waits(&format!("s_wait_kmcnt 0x0\ns_wait_alu {text}")),
            [encoded],
            "{text}"
        );
    }
    let all = AmdAluWait { encoded: 0 };
    assert_eq!(
        (
            all.sa_sdst(),
            all.va_vcc(),
            all.vm_vsrc(),
            all.hold_cnt(),
            all.va_ssrc(),
            all.va_sdst(),
            all.va_vdst()
        ),
        (0, 0, 0, 0, 0, 0, 0)
    );
    let none = AmdAluWait {
        encoded: AmdAluWait::NO_WAIT,
    };
    assert_eq!(
        (
            none.sa_sdst(),
            none.va_vcc(),
            none.vm_vsrc(),
            none.hold_cnt(),
            none.va_ssrc(),
            none.va_sdst(),
            none.va_vdst(),
            none.reserved()
        ),
        (1, 1, 7, 1, 1, 7, 15, 0)
    );
    for malformed in [
        "",
        "0x10000",
        "-1",
        "depctr_va_vdst(16)",
        "depctr_sa_sdst(2)",
        "depctr_sa_sdst(0) depctr_sa_sdst(0)",
        "depctr_unknown(0)",
        "va_vdst(0)",
    ] {
        assert!(
            frontend::parse(&native(&format!(
                "s_wait_kmcnt 0x0\ns_wait_alu {malformed}"
            )))
            .is_err(),
            "{malformed}"
        );
    }
}

fn m1_config(mut config: AnalysisConfig) -> AnalysisConfig {
    use volta_amdgpu_analysis::{GuardRegion, NativeProfile, OutputContract};
    config.native_profile = NativeProfile::Gfx1250B0M1Int;
    config.output_contract = OutputContract::RawU32;
    config.guard_regions = vec![
        GuardRegion {
            name: "pre".into(),
            base: 0x10000,
            bytes: 256,
        },
        GuardRegion {
            name: "post".into(),
            base: 0x10180,
            bytes: 1024,
        },
    ];
    config
}

const STORE_S11: &str = "v_and_b32 v1, 0x3ff, v0\nv_lshlrev_b32 v1, 2, v1\nv_mov_b32 v10, s11\nglobal_store_b32 v1, v10, s[2:3]";

/// Equal S_MAX operands: the default gfx1250 contract keeps `>=` (SCC 1),
/// the B0 m1 profile uses strict `>` (SCC 0). Only SCC differs.
#[test]
fn s_max_scc_is_strict_only_under_the_b0_m1_profile() {
    for (op, a) in [("s_max_u32", "7"), ("s_max_i32", "0x80000001")] {
        let body = format!(
            "s_wait_kmcnt 0x0\ns_mov_b32 s4, {a}\ns_add_co_u32 s5, s4, 1\ns_add_co_u32 s6, s4, 2\n{op} s10, s4, s4\ns_cselect_b32 s11, s5, s6\n{STORE_S11}"
        );
        let module = frontend::parse(&native(&body)).unwrap();
        let legacy = analyze(&module, Some("k"), config()).unwrap();
        let outputs = legacy.symbolic_outputs().unwrap();
        let first = outputs.outputs[0].1[0].1;
        let base: i64 = if a == "7" { 7 } else { 0x8000_0001 };
        assert_eq!(
            outputs.arena.as_int_const(first),
            Some(base + 1),
            "{op} legacy >="
        );
        let m1 = analyze(&module, Some("k"), m1_config(config())).unwrap();
        let raw = m1.raw_outputs().unwrap();
        assert_eq!(
            raw[0].1[0],
            volta_amdgpu_analysis::RawWord::Exact((base + 2) as u32),
            "{op} B0 strict >"
        );
    }
}

#[test]
fn the_m1_profile_requires_gfx1250() {
    use volta_amdgpu_analysis::{AnalysisError, EvalError, NativeProfile};
    let source = ".amdgcn_target \"amdgcn-amd-amdhsa--gfx950\"\n.text\n.type k,@function\nk:\ns_endpgm\n.size k, .-k\n";
    let module = frontend::parse(source).unwrap();
    let mut config = AnalysisConfig::new((64, 1, 1));
    config.native_profile = NativeProfile::Gfx1250B0M1Int;
    assert!(matches!(
        analyze(&module, None, config),
        Err(AnalysisError::Eval(EvalError::InvalidConfig(_)))
    ));
}

/// The prefetch is executed: an X-count issue holding its address sources and
/// a footprint record, with no destination, no data and no load counter.
#[test]
fn prefetch_records_its_x_count_lease_sources_and_footprint() {
    use volta_graph::{AmdCacheScope, AmdCounter, AmdMemoryClass, AmdRegisterFile};
    let body = format!(
        "s_mov_b64 s[30:31], 0\nglobal_prefetch_b8 v0, s[30:31] scope:SCOPE_SE\ns_wait_kmcnt 0x0\ns_mov_b32 s11, 5\n{STORE_S11}"
    );
    let module = frontend::parse(&native(&body)).unwrap();
    for config in [config(), m1_config(config())] {
        let analysis = analyze(&module, Some("k"), config).unwrap();
        let graph = analysis.graph();
        let records = graph.amd_events();
        let (index, issue) = records
            .iter()
            .enumerate()
            .find_map(|(i, r)| match &r.kind {
                AmdEventKind::MemoryIssue {
                    class: AmdMemoryClass::Prefetch,
                    counters,
                    destinations,
                    held_sources,
                } => Some((
                    i,
                    (
                        r.event,
                        counters.clone(),
                        destinations.clone(),
                        held_sources.clone(),
                    ),
                )),
                _ => None,
            })
            .unwrap();
        assert_eq!(issue.1, [(AmdCounter::X, 1)]);
        assert!(issue.2.is_empty());
        let held: Vec<_> = issue.3.iter().map(|r| (r.file, r.index)).collect();
        assert_eq!(
            held,
            [
                (AmdRegisterFile::Vector, 0),
                (AmdRegisterFile::Scalar, 30),
                (AmdRegisterFile::Scalar, 31),
                (AmdRegisterFile::Scalar, 126)
            ]
        );
        assert!(issue.3[0].lanes == u32::MAX as u64);
        let AmdEventKind::PrefetchFootprint {
            issue: owner,
            scope,
            lanes,
        } = &records[index + 1].kind
        else {
            panic!("prefetch footprint must follow its issue");
        };
        assert_eq!((*owner, *scope), (issue.0, AmdCacheScope::Se));
        assert_eq!(
            *lanes,
            (0..32).map(|l| (l, u64::from(l))).collect::<Vec<_>>()
        );
        let checked = volta_graph_check::check_graph(graph);
        let amd = checked.amd.unwrap();
        assert_eq!(amd.trace.verdict(), volta_graph_check::Verdict::Pass);
        assert_eq!(
            amd.wait_readiness.verdict(),
            volta_graph_check::Verdict::Pass
        );
    }
    // A CU-scope prefetch outside configured objects is unsupported by default.
    let module = frontend::parse(&native(&body.replace("SCOPE_SE", "SCOPE_CU"))).unwrap();
    let analysis = analyze(&module, Some("k"), config()).unwrap();
    assert!(analysis.symbolic_outputs().is_err());
}

fn m2_config(config: AnalysisConfig) -> AnalysisConfig {
    let mut config = m1_config(config);
    config.native_profile = volta_amdgpu_analysis::NativeProfile::Gfx1250B0M2Mode;
    config
}

/// The MODE write forms parse with typed hwreg operands; malformed operands
/// are syntax errors, not refusals.
#[test]
fn mode_write_forms_parse_with_typed_hwreg_operands() {
    use frontend::Operand;
    let body = "s_wait_kmcnt 0x0\ns_setreg_b32 hwreg(HW_REG_WAVE_MODE, 4, 4), s4\ns_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 0, 4), 0x105\ns_round_mode 0x10\ns_denorm_mode 3";
    let module = frontend::parse(&native(body)).unwrap();
    let operands: Vec<_> = module.kernels[0].instructions[2..6]
        .iter()
        .map(|i| (i.mnemonic.as_str(), i.operands.clone()))
        .collect();
    let mode = |offset| Operand::HardwareRegister {
        name: "HW_REG_WAVE_MODE".into(),
        offset,
        width: 4,
    };
    assert!(
        matches!(&operands[0], ("s_setreg_b32", o) if o[0] == mode(4) && matches!(o[1], Operand::Register(_)))
    );
    assert_eq!(
        operands[1],
        ("s_setreg_imm32_b32", vec![mode(0), Operand::Integer(0x105)])
    );
    assert_eq!(operands[2], ("s_round_mode", vec![Operand::Integer(0x10)]));
    assert_eq!(operands[3], ("s_denorm_mode", vec![Operand::Integer(3)]));
    for malformed in [
        "s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0, 4), v4",
        "s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0, 4), 5",
        "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 0, 4), s4",
        "s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 30, 4), s4",
        "s_round_mode -1",
        "s_denorm_mode 0x10000",
    ] {
        assert!(
            frontend::parse(&native(&format!("s_wait_kmcnt 0x0\n{malformed}"))).is_err(),
            "{malformed}"
        );
    }
}

/// MODE state executes only under the m2 profile: the default contract and
/// the m1 profile keep refusing the write forms.
#[test]
fn mode_state_executes_only_under_the_m2_profile() {
    use volta_graph::{AmdModeOp, Terminal};
    use volta_graph_check::amdgpu::mode_order::{ModeOrderKind, check_mode_order};
    let body = format!(
        "s_wait_kmcnt 0x0\ns_round_mode 0x5\ns_getreg_b32 s11, hwreg(HW_REG_WAVE_MODE, 0, 4)\n{STORE_S11}"
    );
    let module = frontend::parse(&native(&body)).unwrap();
    for refused in [config(), m1_config(config())] {
        let analysis = analyze(&module, Some("k"), refused).unwrap();
        assert!(matches!(
            analysis.graph().terminal(),
            Terminal::Aborted { .. }
        ));
    }
    let analysis = analyze(&module, Some("k"), m2_config(config())).unwrap();
    assert_eq!(
        analysis.raw_outputs().unwrap()[0].1[0],
        volta_amdgpu_analysis::RawWord::Exact(5)
    );
    let accesses: Vec<_> = analysis
        .graph()
        .amd_events()
        .iter()
        .filter_map(|r| match r.kind {
            AmdEventKind::ModeAccess {
                op, offset, value, ..
            } => Some((op, offset, value)),
            _ => None,
        })
        .collect();
    assert_eq!(
        accesses,
        [(AmdModeOp::RoundMode, 0, 5), (AmdModeOp::Getreg, 0, 5)]
    );
    assert!(check_mode_order(analysis.graph()).unresolved.is_empty());

    // Entry MODE is undefined without descriptor controls: no guessed value.
    let body =
        format!("s_wait_kmcnt 0x0\ns_getreg_b32 s11, hwreg(HW_REG_WAVE_MODE, 0, 4)\n{STORE_S11}");
    let module = frontend::parse(&native(&body)).unwrap();
    let analysis = analyze(&module, Some("k"), m2_config(config())).unwrap();
    assert!(matches!(
        analysis.graph().terminal(),
        Terminal::Aborted { .. }
    ));

    // Read then a later MODE write: unresolved without the SA_SDST wait,
    // resolved with it.
    for (wait, unresolved) in [
        ("", 1),
        ("s_wait_alu depctr_sa_sdst(0)\n", 0),
        ("s_wait_alu depctr_va_vdst(0)\n", 1),
    ] {
        let body = format!(
            "s_wait_kmcnt 0x0\ns_round_mode 0x5\ns_getreg_b32 s11, hwreg(HW_REG_WAVE_MODE, 0, 4)\n{wait}s_denorm_mode 0x2\n{STORE_S11}"
        );
        let module = frontend::parse(&native(&body)).unwrap();
        let analysis = analyze(&module, Some("k"), m2_config(config())).unwrap();
        let report = check_mode_order(analysis.graph());
        assert_eq!(report.unresolved.len(), unresolved, "{wait}");
        assert!(
            report
                .unresolved
                .iter()
                .all(|v| v.kind == ModeOrderKind::UnresolvedWriteAfterRead)
        );
    }
}

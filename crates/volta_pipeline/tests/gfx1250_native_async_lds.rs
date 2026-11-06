//! Native per-lane GV copies preserve payloads and require real LDS completion.
#[path = "gfx950_common/mod.rs"]
mod native_common;

use std::collections::BTreeSet;
use volta_amdgpu_analysis::{AnalysisConfig, ArrayDef, ArrayKind, frontend, ir, lower};
use volta_expr::ExprNode;
use volta_graph::{AccessMode, AmdEventKind, AmdMemoryClass, Footprint, Space};
use volta_graph_check::{GraphConclusion, Property, Verdict};

fn native(body: &str) -> String {
    format!(
        r#".amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.type k,@function
k:
{body}
s_endpgm
.size k,.-k
.amdhsa_kernel k
.amdhsa_wavefront_size32 1
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_group_segment_fixed_size 2048
.end_amdhsa_kernel
"#
    )
}

fn copy(wait: &str, offset: i64) -> String {
    format!(
        "v_lshlrev_b32 v1,4,v0\nv_mov_b32 v2,v1\nv_mov_b32 v3,0\n\
         s_mov_b64 s[4:5],0x10000\nv_add_nc_u64 v[2:3],s[4:5],v[2:3]\n\
         global_load_async_to_lds_b128 v1,v[2:3],off offset:{offset}\n{wait}\n\
         ds_load_b128 v[8:11],v1 offset:{offset}\ns_wait_dscnt 0\n\
         s_mov_b64 s[6:7],0x20000\nglobal_store_b128 v1,v[8:11],s[6:7]\n\
         s_wait_storecnt 0\ns_wait_xcnt 0"
    )
}

fn config(threads: u32, input_len: u64) -> AnalysisConfig {
    let mut config = AnalysisConfig::new((threads, 1, 1));
    config.arrays = vec![
        ArrayDef {
            name: "input".into(),
            base: 0x10000,
            elem_width: 4,
            len: input_len,
            kind: ArrayKind::Input,
        },
        ArrayDef {
            name: "out".into(),
            base: 0x20000,
            elem_width: 4,
            len: u64::from(threads) * 4,
            kind: ArrayKind::Output,
        },
    ];
    config
}

fn check(body: &str, config: AnalysisConfig) -> native_common::Checked {
    let source = native(body);
    let module = frontend::parse(&source).unwrap();
    native_common::check(&source, &module, Some("k"), config).unwrap()
}

fn run(body: &str, threads: u32, input_len: u64) -> native_common::Checked {
    check(body, config(threads, input_len))
}

#[test]
fn typed_copy_has_two_address_inputs_and_no_register_payload() {
    let module = frontend::parse(&native("global_load_async_to_lds_b128 v1,v[2:3],off")).unwrap();
    let program = lower::lower_kernel_in_module(&module, &module.kernels[0]).unwrap();
    let ir::Operation::Memory(memory) = program.instructions[0].operation() else {
        panic!("memory")
    };
    assert_eq!(memory.kind, ir::MemoryKind::AsyncLoadToLds);
    assert_eq!(memory.shape, ir::TransferShape::PerLaneBytes(16));
    assert!(memory.data_destinations().is_empty());
    assert!(memory.data_sources().is_empty());
    assert!(
        matches!(memory.address(), ir::MemoryAddress::GlobalToLds { global: ir::Operand::Register(g), lds: ir::Operand::Register(l) } if g.index == 2 && g.count == 2 && l.index == 1 && l.count == 1)
    );
}

#[test]
fn async_zero_wait_preserves_every_lane_word_and_both_byte_displacements() {
    for threads in [1, 3, 32, 64] {
        for offset in [0, 16] {
            let checked = run(
                &copy("s_wait_asynccnt 0", offset),
                threads,
                4 * u64::from(threads) + 4,
            );
            assert_eq!(
                checked.checks.conclusion(),
                GraphConclusion::Verified,
                "{:?}",
                checked.checks
            );
            let outputs = checked.analysis.symbolic_outputs().unwrap();
            assert_eq!(outputs.outputs[0].1.len(), threads as usize * 4);
            for index in 0..u64::from(threads) * 4 {
                let ExprNode::InputElement {
                    array,
                    index: actual,
                } = outputs.arena.node(outputs.element("out", index).unwrap())
                else {
                    panic!("input transport")
                };
                assert_eq!(outputs.arena.string(*array), "input");
                assert_eq!(*actual, index + offset as u64 / 4);
            }
            let graph = checked.analysis.graph();
            assert!(
                graph
                    .provenance()
                    .assumes(volta_graph::Assumption::AmdDefaultScheduling)
            );
            for issue in graph.amd_events().iter().filter(|event| {
                matches!(
                    event.kind,
                    AmdEventKind::MemoryIssue {
                        class: AmdMemoryClass::AsyncLoadToLds,
                        ..
                    }
                )
            }) {
                let effects: Vec<_> = graph
                    .amd_memory_effects()
                    .iter()
                    .filter(|link| link.issue == issue.event)
                    .collect();
                let lanes = threads.min(32);
                assert_eq!(effects.len(), lanes as usize * 8);
                let AmdEventKind::MemoryIssue {
                    counters,
                    destinations,
                    held_sources,
                    ..
                } = &issue.kind
                else {
                    unreachable!()
                };
                assert!(destinations.is_empty());
                assert_eq!(counters, &[(volta_graph::AmdCounter::Async, 1)]);
                assert!(held_sources.is_empty());
                let mut reads = BTreeSet::new();
                let mut writes = BTreeSet::new();
                for link in effects {
                    let volta_graph::EventKind::Access(id) = graph.event(link.effect).kind else {
                        panic!("access")
                    };
                    let access = graph.access(id);
                    let Footprint::Bytes { addr, width } = access.footprint else {
                        panic!("bytes")
                    };
                    assert_eq!(width, 4);
                    assert!(matches!(
                        (access.space, access.mode),
                        (Space::Global, AccessMode::Read) | (Space::Shared, AccessMode::Write)
                    ));
                    if access.space == Space::Global {
                        reads.extend(addr..addr + 4);
                    } else {
                        writes.extend(addr..addr + 4);
                    }
                }
                let first = u64::from(issue.wave) * 32 * 16 + offset as u64;
                let end = first + u64::from(lanes) * 16;
                assert_eq!(writes, (first..end).collect());
                assert_eq!(reads, (0x10000 + first..0x10000 + end).collect());
            }
        }
    }
}

#[test]
fn unrelated_or_insufficient_waits_do_not_publish_async_lds_writes() {
    for wait in [
        "",
        "s_wait_loadcnt 0\ns_wait_xcnt 0",
        "s_wait_dscnt 0",
        "s_wait_tensorcnt 0",
        "s_wait_asynccnt 1",
    ] {
        let checked = run(&copy(wait, 0), 32, 128);
        assert_eq!(
            checked.checks.verdict(Property::AmdMemoryConflict),
            Verdict::Fail,
            "{wait}: {:?}",
            checked.checks
        );
    }
}

#[test]
fn captured_async_addresses_allow_immediate_valu_overwrites() {
    let body = copy("s_wait_asynccnt 0", 0)
        .replace(
            "global_load_async_to_lds_b128 v1,v[2:3],off offset:0",
            "global_load_async_to_lds_b128 v1,v[2:3],off offset:0\n\
             v_mov_b32 v12,v1\nv_mov_b64 v[2:3],0\nv_mov_b32 v1,0",
        )
        .replace("ds_load_b128 v[8:11],v1", "ds_load_b128 v[8:11],v12")
        .replace("global_store_b128 v1,", "global_store_b128 v12,");
    let checked = run(&body, 32, 128);
    assert_eq!(
        checked.checks.conclusion(),
        GraphConclusion::Verified,
        "{:?}",
        checked.checks
    );
    let outputs = checked.analysis.symbolic_outputs().unwrap();
    for index in 0..128 {
        let ExprNode::InputElement {
            array,
            index: actual,
        } = outputs.arena.node(outputs.element("out", index).unwrap())
        else {
            panic!("input transport")
        };
        assert_eq!(outputs.arena.string(*array), "input");
        assert_eq!(*actual, index);
    }
}

#[test]
fn captured_addresses_still_require_source_readiness_at_issue() {
    for wait in ["", "s_wait_dscnt 0"] {
        let body = copy("s_wait_asynccnt 0", 0).replace(
            "global_load_async_to_lds_b128",
            &format!(
                "v_mov_b32 v13,0\nv_mov_b32 v14,0\n\
                 ds_store_b32 v13,v14\ns_wait_dscnt 0\n\
                 ds_load_b32 v1,v13\n{wait}\nglobal_load_async_to_lds_b128"
            ),
        );
        let checked = run(&body, 1, 4);
        assert_eq!(
            checked.checks.verdict(Property::AmdWaitReadiness),
            if wait.is_empty() {
                Verdict::Fail
            } else {
                Verdict::Pass
            },
            "wait={wait}: {:?}",
            checked.checks
        );
    }
}

#[test]
fn partial_async_wait_completes_only_the_first_of_two_native_loads() {
    for read_second in [false, true] {
        let mut body = copy("s_wait_asynccnt 1", 0).replace(
            "s_wait_asynccnt 1",
            "v_add_nc_u32 v12,512,v1\n\
             global_load_async_to_lds_b128 v12,v[2:3],off\ns_wait_asynccnt 1",
        );
        if read_second {
            body = body.replace("ds_load_b128 v[8:11],v1", "ds_load_b128 v[8:11],v12");
        }
        let checked = run(&body, 32, 128);
        assert_eq!(
            checked.checks.verdict(Property::AmdMemoryConflict),
            if read_second {
                Verdict::Fail
            } else {
                Verdict::Pass
            },
            "{:?}",
            checked.checks
        );
    }
}

#[test]
fn sparse_exec_preserves_inactive_lds_without_observing_undefined_addresses() {
    let body = copy("s_wait_asynccnt 0\ns_mov_b32 exec_lo,7", 0).replacen(
        "v_mov_b32 v2,v1",
        "v_mov_b32 v8,0\nv_mov_b32 v9,0\nv_mov_b32 v10,0\nv_mov_b32 v11,0\n\
         ds_store_b128 v1,v[8:11]\ns_wait_dscnt 0\ns_mov_b32 exec_lo,5\nv_mov_b32 v2,v1",
        1,
    );
    let checked = run(&body, 3, 12);
    assert_eq!(
        checked.checks.conclusion(),
        GraphConclusion::Verified,
        "{:?}",
        checked.checks
    );
    let output = checked.analysis.symbolic_outputs().unwrap();
    for index in 0..12 {
        let value = output.element("out", index).unwrap();
        if (4..8).contains(&index) {
            assert_eq!(output.arena.as_int_const(value), Some(0));
        } else {
            assert!(
                matches!(output.arena.node(value), ExprNode::InputElement { index: actual, .. } if *actual == index)
            );
        }
    }
}

#[test]
fn async_wait_completes_only_its_wave_and_cross_wave_reads_need_publication() {
    for (completion, publication) in [(true, false), (true, true), (false, true)] {
        let wait = match (completion, publication) {
            (true, true) => "s_wait_asynccnt 0\ns_barrier_signal -1\ns_barrier_wait -1",
            (true, false) => "s_wait_asynccnt 0",
            (false, true) => "s_barrier_signal -1\ns_barrier_wait -1",
            _ => unreachable!(),
        };
        let body = copy(wait, 0).replace(
            "ds_load_b128 v[8:11],v1",
            "v_xor_b32 v12,512,v1\nds_load_b128 v[8:11],v12",
        );
        let checked = run(&body, 64, 256);
        assert_eq!(
            checked.checks.verdict(Property::AmdMemoryConflict),
            if completion && publication {
                Verdict::Pass
            } else {
                Verdict::Fail
            },
            "{:?}",
            checked.checks
        );
        if completion && publication {
            assert_eq!(
                checked.checks.conclusion(),
                GraphConclusion::Verified,
                "{:?}",
                checked.checks
            );
            let output = checked.analysis.symbolic_outputs().unwrap();
            for index in 0..256 {
                assert!(
                    matches!(output.arena.node(output.element("out", index).unwrap()), ExprNode::InputElement { index: actual, .. } if *actual == index ^ 128)
                );
            }
        }
    }
}

#[test]
fn negative_displacement_applies_to_both_inputs_without_wrapping() {
    let body = copy("s_wait_asynccnt 0", 0)
        .replace("global_load_async_to_lds_b128 v1,v[2:3],off offset:0", "v_add_nc_u32 v12,16,v1\nv_add_nc_u64 v[2:3],16,v[2:3]\nglobal_load_async_to_lds_b128 v12,v[2:3],off offset:-16");
    let checked = run(&body, 3, 12);
    assert_eq!(
        checked.checks.conclusion(),
        GraphConclusion::Verified,
        "{:?}",
        checked.checks
    );
    let output = checked.analysis.symbolic_outputs().unwrap();
    for index in 0..12 {
        assert!(
            matches!(output.arena.node(output.element("out", index).unwrap()), ExprNode::InputElement { index: actual, .. } if *actual == index)
        );
    }
    let wrapped = run(
        "v_mov_b32 v1,0\nv_mov_b32 v2,-16\nv_mov_b32 v3,-1\nglobal_load_async_to_lds_b128 v1,v[2:3],off offset:65552",
        1,
        4,
    );
    assert!(
        wrapped
            .analysis
            .symbolic_outputs()
            .unwrap_err()
            .to_string()
            .contains("unsupported")
    );
}

#[test]
fn whole_copy_bounds_are_checked_before_issuing_any_payload_effect() {
    for (threads, input_len) in [(1, 3), (2, 7)] {
        let checked = run(&copy("s_wait_asynccnt 0", 0), threads, input_len);
        assert!(
            checked
                .analysis
                .symbolic_outputs()
                .unwrap_err()
                .to_string()
                .contains("out-of-bounds")
        );
        assert!(
            !checked
                .analysis
                .graph()
                .amd_events()
                .iter()
                .any(|event| matches!(
                    event.kind,
                    AmdEventKind::MemoryIssue {
                        class: AmdMemoryClass::AsyncLoadToLds,
                        ..
                    }
                ))
        );
    }
}

#[test]
fn complete_lds_footprint_fits_one_object_and_the_last_allocation_byte() {
    let mut bounded = config(1, 4);
    bounded.shared_allocations = vec![
        volta_amdgpu_analysis::SharedAllocation {
            name: "first".into(),
            offset: 0,
            size: 8,
        },
        volta_amdgpu_analysis::SharedAllocation {
            name: "second".into(),
            offset: 8,
            size: 8,
        },
    ];
    let crossed = check(&copy("s_wait_asynccnt 0", 0), bounded);
    assert!(
        crossed
            .analysis
            .symbolic_outputs()
            .unwrap_err()
            .to_string()
            .contains("shared allocation boundaries")
    );
    assert!(
        !crossed
            .analysis
            .graph()
            .amd_events()
            .iter()
            .any(|event| matches!(
                event.kind,
                AmdEventKind::MemoryIssue {
                    class: AmdMemoryClass::AsyncLoadToLds,
                    ..
                }
            ))
    );
    for start in [2032, 2048] {
        let body = copy("s_wait_asynccnt 0", 0)
            .replace(
                "global_load_async_to_lds_b128 v1",
                &format!("v_mov_b32 v12,{start}\nglobal_load_async_to_lds_b128 v12"),
            )
            .replace("ds_load_b128 v[8:11],v1", "ds_load_b128 v[8:11],v12");
        let checked = run(&body, 1, 4);
        if start == 2032 {
            assert_eq!(
                checked.checks.conclusion(),
                GraphConclusion::Verified,
                "{:?}",
                checked.checks
            );
        } else {
            assert!(
                checked
                    .analysis
                    .symbolic_outputs()
                    .unwrap_err()
                    .to_string()
                    .contains("out-of-bounds LDS")
            );
        }
    }
}

#[test]
fn inactive_copy_observes_no_address_or_memory_and_unsupported_modes_are_refused() {
    let checked = run(
        "s_mov_b32 exec_lo,0\nglobal_load_async_to_lds_b128 v1,v[2:3],off",
        1,
        4,
    );
    assert!(
        !checked
            .analysis
            .graph()
            .amd_events()
            .iter()
            .any(|event| matches!(
                event.kind,
                AmdEventKind::MemoryIssue {
                    class: AmdMemoryClass::AsyncLoadToLds,
                    ..
                }
            ))
    );
    for bad in [
        "global_load_async_to_lds_b128 v[1:4],v[2:3],off",
        "global_load_async_to_lds_b128 v1,v2,off",
        "global_load_async_to_lds_b32 v1,v[2:3],off",
        "global_load_async_to_lds_b64 v1,v[2:3],s[4:5]",
        "global_load_async_to_lds_b128 v1,v2,s[4:5] scale_offset",
        "global_load_async_to_lds_b128 v1,v[2:3],off scale_offset",
        "global_load_async_to_lds_b128 v1,v[2:3],off offset:8388608",
        "global_load_async_to_lds_b128 v1,v[2:3],off offset:-8388609",
        "global_load_async_to_lds_b128 v1,v[2:3],off offset:thing",
    ] {
        assert!(frontend::parse(&native(bad)).is_err(), "{bad}");
    }
    // The scalar-base GVS forms parse, but only the B0 litmus profile
    // executes them; the default contract keeps refusing them.
    for gvs in [
        "global_load_async_to_lds_b128 v1,v2,s[4:5]",
        "global_load_async_to_lds_b32 v1,v2,s[4:5]",
    ] {
        let gvs =
            format!("s_mov_b32 s4,0x10000\ns_mov_b32 s5,0\nv_mov_b32 v1,0\nv_mov_b32 v2,0\n{gvs}");
        let checked = run(&gvs, 1, 4);
        assert!(
            !checked
                .analysis
                .graph()
                .amd_events()
                .iter()
                .any(|event| matches!(
                    event.kind,
                    AmdEventKind::MemoryIssue {
                        class: AmdMemoryClass::AsyncLoadToLds,
                        ..
                    }
                )),
            "{gvs}"
        );
        assert!(checked.analysis.symbolic_outputs().is_err(), "{gvs}");
    }
    for target in ["gfx942", "gfx950"] {
        assert!(
            frontend::parse(
                &native("global_load_async_to_lds_b128 v1,v[2:3],off").replace("gfx1250", target)
            )
            .is_err()
        );
    }
    let mut module = frontend::parse(&native(
        "s_mov_b32 exec_lo,0\nglobal_load_async_to_lds_b128 v1,v[2:3],off offset:16",
    ))
    .unwrap();
    let modifier = module.kernels[0].instructions[1].modifiers[0].clone();
    module.kernels[0].instructions[1].modifiers.push(modifier);
    assert!(lower::lower_kernel_in_module(&module, &module.kernels[0]).is_err());
}

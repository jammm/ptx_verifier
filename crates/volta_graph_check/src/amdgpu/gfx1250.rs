//! GFX12.5 completion and execution synchronization. The u64 mask storage is
//! shared with old graphs; only its low 32 bits describe this target's waves.
//!
//! Semantic references: LLVM 5fd50ec8e07abb94e203933017e8d6e40fb51824,
//! AMDGPUHWEvents.cpp, SIInsertWaitcnts.cpp and
//! AMDGPUExecutionSynchronization.md. X_CNT protects address-translation replay
//! operands, independently of memory completion. Signals publish only already
//! completed effects. Neither trace order nor a signal fabricates a data wait.

use super::*;
use volta_graph::{
    AccessMode, AmdBufferRangeCheck, AmdGlobalAccess, AmdSmallFloat, AmdSwmmacInput, AmdWmmaKind,
    Assumption, ThreadIx,
};
mod named;

const WAVE: usize = 32;

fn flat_profile(class: AmdMemoryClass) -> Option<(Space, AccessMode)> {
    Some(match class {
        AmdMemoryClass::FlatGlobalLoad => (Space::Global, AccessMode::Read),
        AmdMemoryClass::FlatGlobalStore => (Space::Global, AccessMode::Write),
        AmdMemoryClass::FlatLdsLoad => (Space::Shared, AccessMode::Read),
        AmdMemoryClass::FlatLdsStore => (Space::Shared, AccessMode::Write),
        _ => return None,
    })
}

fn maximum(counter: AmdCounter) -> Option<u16> {
    match counter {
        AmdCounter::Load
        | AmdCounter::Store
        | AmdCounter::Ds
        | AmdCounter::X
        | AmdCounter::Async => Some(63),
        AmdCounter::Km => Some(31),
        // The pinned LLVM hardware-limits helper does not specify TENSOR_CNT's
        // hardware limit. Zero is supported; nonzero waits remain inconclusive.
        AmdCounter::Tensor => Some(u16::MAX),
        _ => None,
    }
}

fn charges(class: AmdMemoryClass) -> Option<Vec<(AmdCounter, u16)>> {
    use AmdCounter::*;
    Some(match class {
        AmdMemoryClass::VectorLoad => vec![(Load, 1), (X, 1)],
        AmdMemoryClass::Prefetch => vec![(X, 1)],
        AmdMemoryClass::VectorStore => vec![(Store, 1), (X, 1)],
        AmdMemoryClass::GlobalAtomicAddF32NoReturn => vec![(Store, 1), (X, 1)],
        AmdMemoryClass::FlatGlobalLoad | AmdMemoryClass::FlatLdsLoad => {
            vec![(Load, 1), (Ds, 1), (X, 1)]
        }
        AmdMemoryClass::FlatGlobalStore | AmdMemoryClass::FlatLdsStore => {
            vec![(Store, 1), (Ds, 1), (X, 1)]
        }
        AmdMemoryClass::ScalarLoad => vec![(Km, 1), (X, 1)],
        AmdMemoryClass::Ds | AmdMemoryClass::CrossLane => vec![(Ds, 1)],
        AmdMemoryClass::AsyncLoadToLds | AmdMemoryClass::AsyncStoreFromLds => vec![(Async, 1)],
        AmdMemoryClass::TensorLoadToLds | AmdMemoryClass::TensorStoreFromLds => vec![(Tensor, 1)],
        _ => return None,
    })
}

/// Provenance kinds of memory instructions (they form replay groups).
fn memory_kind(kind: &str) -> bool {
    matches!(
        kind,
        "scalar memory"
            | "vector memory"
            | "LDS"
            | "async memory"
            | "tensor memory"
            | "atomic f32 add no return"
            | "flat memory"
            | "prefetch"
    )
}

fn replay_group(class: AmdMemoryClass) -> Option<bool> {
    // This conservative source-group classification does not infer the
    // compiler's LOCAL MMO from a resolved LDS effect. Generic FLAT encoding
    // remains a possible receiver of older VMEM translation dependencies below.
    // Older LDS source tokens still require their own explicit retirement.
    match class {
        AmdMemoryClass::ScalarLoad => Some(false),
        AmdMemoryClass::VectorLoad
        | AmdMemoryClass::Prefetch
        | AmdMemoryClass::VectorStore
        | AmdMemoryClass::GlobalAtomicAddF32NoReturn
        | AmdMemoryClass::FlatGlobalLoad
        | AmdMemoryClass::FlatGlobalStore => Some(true),
        _ => None,
    }
}

fn expected(kind: &str, mask: u64) -> Option<Option<&'static str>> {
    Some(match kind {
        "scalar memory" => Some("memory"),
        "vector memory"
        | "LDS"
        | "async memory"
        | "tensor memory"
        | "atomic f32 add no return"
        | "flat memory"
            if mask != 0 =>
        {
            Some("memory")
        }
        "wait" => Some("wait"),
        "alu wait" => Some("alu wait"),
        "prefetch" if mask != 0 => Some("memory"),
        "prefetch" => None,
        "s_nop" => Some("delay"),
        "s_delay_alu" => Some("delay alu"),
        "barrier signal" => Some("signal"),
        "barrier wait" => Some("barrier wait"),
        "named barrier init" => Some("named init"),
        "named barrier join" => Some("named join"),
        "named barrier signal" => Some("named signal"),
        "named barrier wait" => Some("named wait"),
        "named barrier leave" => Some("named leave"),
        "WMMA" => Some("wmma"),
        "scalar ALU"
        | "vector ALU"
        | "trans ALU"
        | "shifted ALU"
        | "branch"
        | "control"
        | "xcnt drain"
        | "s_endpgm"
        | "vector memory"
        | "LDS"
        | "async memory"
        | "tensor memory"
        | "flat memory"
        | "atomic f32 add no return" => None,
        _ => return None,
    })
}

/// Lanes are strictly increasing and exactly the set bits of `mask`.
fn exact_active_lanes(lanes: &[(u8, u64)], mask: u64) -> bool {
    lanes.windows(2).all(|pair| pair[0].0 < pair[1].0)
        && lanes.iter().try_fold(0u64, |seen, &(lane, _)| {
            (lane < 32).then(|| seen | (1u64 << lane))
        }) == Some(mask)
}

/// Independent check of one footprint against a configured global layout.
/// Without a layout only non-speculative prefetches are unvalidated; ordinary
/// accesses keep the target's existing allocation contract.
pub(super) fn global_bounds_defect(
    graph: &EventGraph,
    access: AmdGlobalAccess,
    address: u64,
    bytes: u64,
) -> Option<&'static str> {
    let layout = graph.global_region_layout();
    let speculative = matches!(access, AmdGlobalAccess::Prefetch(s) if s.speculative_prefetch());
    if layout.is_empty() {
        return (matches!(access, AmdGlobalAccess::Prefetch(_)) && !speculative)
            .then_some("non-speculative prefetch requires a configured global object layout");
    }
    let Some(end) = address.checked_add(bytes) else {
        return Some("global footprint overflows the address space");
    };
    if layout
        .iter()
        .any(|r| r.guard && address < r.base.saturating_add(r.size) && r.base < end)
    {
        return Some("global footprint touches a configured guard region");
    }
    if !speculative
        && !layout
            .iter()
            .any(|r| !r.guard && r.base <= address && end <= r.base.saturating_add(r.size))
    {
        return Some("global footprint leaves every configured object");
    }
    None
}

fn registers(regs: &[AmdRegister], mask: u64, event: EventId, unresolved: &mut Vec<Unresolved>) {
    for reg in regs {
        let valid = match reg.file {
            AmdRegisterFile::Scalar => reg.index <= 106 || matches!(reg.index, 124 | 126),
            AmdRegisterFile::Temporary => reg.index < 16,
            AmdRegisterFile::Vector => reg.index < 1024 && reg.lanes != 0 && reg.lanes & !mask == 0,
            AmdRegisterFile::Accumulator => false,
        };
        if !valid {
            unresolved.push(Unresolved::at(
                event,
                "invalid gfx1250 register namespace, index or physical lane mask",
            ));
        }
    }
}

fn claim<'a>(
    graph: &EventGraph,
    record: &volta_graph::AmdEvent,
    instructions: &'a mut HashMap<u32, InstructionRecord>,
    kind: &str,
    unresolved: &mut Vec<Unresolved>,
) -> Option<&'a InstructionRecord> {
    // A semantic record cannot turn an ordinary ALU or masked-out memory
    // instruction into a different operation merely by borrowing its PC.
    if instructions
        .get(&record.wave)
        .is_some_and(|i| i.expected != Some(kind))
    {
        unresolved.push(Unresolved::at(
            record.event,
            "gfx1250 semantic event disagrees with its instruction provenance",
        ));
        return None;
    }
    claim_instruction(graph, record, instructions, kind, unresolved)
}

fn vector_tuple(registers: &[AmdRegister], count: usize) -> bool {
    registers.len() == count
        && registers.first().is_some_and(|first| {
            registers.iter().enumerate().all(|(offset, register)| {
                register.file == AmdRegisterFile::Vector
                    && register.index as usize == first.index as usize + offset
                    && register.lanes == u32::MAX as u64
            })
        })
}

fn contract(graph: &EventGraph, prefix: bool) -> AmdTraceResult {
    if super::private_memory::present(graph) {
        return unknown("fixed private DWORD and literal-lane contracts require gfx950");
    }
    if let Err(defects) = graph.validate() {
        return PropertyResult::Unknown(
            defects
                .into_iter()
                .map(|d| Unresolved {
                    events: d
                        .event
                        .filter(|e| graph.events().get(*e).is_some())
                        .into_iter()
                        .collect(),
                    reason: d.message,
                })
                .collect(),
        );
    }
    let mut unresolved = Vec::new();
    let default_scheduling_count = graph
        .provenance()
        .assumptions
        .iter()
        .filter(|&&assumption| assumption == Assumption::AmdDefaultScheduling)
        .count();
    let named_allocations: Vec<_> = graph
        .provenance()
        .assumptions
        .iter()
        .filter_map(|a| {
            if let Assumption::AllocatedNamedBarriers(count) = a {
                Some(*count)
            } else {
                None
            }
        })
        .collect();
    if named_allocations.len() > 1
        || named_allocations
            .iter()
            .any(|count| !matches!(count, 4 | 8 | 12 | 16))
    {
        unresolved.push(Unresolved::global(
            "invalid or duplicate gfx1250 named barrier allocation premise",
        ));
    }
    let named_capacity = named_allocations.first().copied().unwrap_or(0);
    let mut atomic_pairs =
        HashMap::<(EventId, ThreadIx), (Option<(EventId, u64)>, Option<(EventId, u64)>)>::new();
    if graph.provenance().assumes(Assumption::XnackReplayDisabled) {
        unresolved.push(Unresolved::global(
            "gfx1250 XNACK is hardwired on; a replay-disabled premise is invalid",
        ));
    }
    if !graph.provenance().assumes(Assumption::UnclusteredWorkgroup) {
        unresolved.push(Unresolved::global(
            "gfx1250 requires the explicit unclustered-workgroup execution envelope",
        ));
    }
    if graph
        .provenance()
        .assumes(Assumption::ConfiguredSharedAllocationLayout)
        != !graph.shared_allocation_layout().is_empty()
    {
        unresolved.push(Unresolved::global(
            "configured LDS layout and its provenance premise disagree",
        ));
    }
    if graph.provenance().grid_dim != (1, 1, 1)
        && !graph
            .provenance()
            .assumes(Assumption::SingleWorkgroupAnalyzed)
    {
        unresolved.push(Unresolved::global(
            "a multi-workgroup logical grid requires the explicit single analyzed workgroup premise",
        ));
    }
    let waves = (graph.threads() as usize).div_ceil(WAVE);
    let mut seen = vec![false; waves];
    let mut m0_defined = vec![false; waves];
    let mut instructions = HashMap::<u32, InstructionRecord>::new();
    let mut issues = HashMap::new();
    let mut open_prefetch = HashMap::<u32, EventId>::new();
    let mut prefetch_footprints = HashSet::new();
    let mut range_checks = HashMap::<EventId, &AmdBufferRangeCheck>::new();
    let mut instruction_count = 0;
    let mut ended = HashSet::new();
    for record in graph.amd_events() {
        let valid = (graph.threads() as usize - record.wave as usize * WAVE).min(WAVE);
        let physical_mask = (1u64 << valid) - 1;
        if ended.contains(&record.wave) {
            unresolved.push(Unresolved::at(
                record.event,
                "gfx1250 wave recorded an instruction after S_ENDPGM",
            ));
        }
        match &record.kind {
            AmdEventKind::Instruction {
                exec_mask,
                reads,
                writes,
                scalar_mask_reads,
            } => {
                instruction_count += 1;
                seen[record.wave as usize] = true;
                if exec_mask & !physical_mask != 0 {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "gfx1250 EXEC contains a nonexistent lane or a wave64 high bit",
                    ));
                }
                registers(reads, physical_mask, record.event, &mut unresolved);
                registers(writes, physical_mask, record.event, &mut unresolved);
                if scalar_mask_reads.iter().any(|r| {
                    r.file != AmdRegisterFile::Scalar || r.index == 126 || !reads.contains(r)
                }) {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "invalid carry/condition role in gfx1250 register signature",
                    ));
                }
                if let Some(previous) = instructions.get(&record.wave) {
                    check_missing_semantic(previous, &mut unresolved);
                }
                let pc = graph.event(record.event).pc;
                let kind = pc.map(|p| graph.provenance().instructions[p.0 as usize].kind);
                let uses_m0 = reads
                    .iter()
                    .any(|r| r.file == AmdRegisterFile::Scalar && r.index == 124);
                let writes_m0 = writes
                    .iter()
                    .any(|r| r.file == AmdRegisterFile::Scalar && r.index == 124);
                if uses_m0
                    && (!m0_defined[record.wave as usize]
                        || !matches!(
                            kind,
                            Some(
                                "named barrier init"
                                    | "named barrier join"
                                    | "named barrier signal"
                            )
                        ))
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "M0 read is uninitialized or outside the selected named barrier role",
                    ));
                }
                if writes_m0 {
                    if kind != Some("scalar ALU") {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "M0 write is not a scalar move",
                        ));
                    }
                    m0_defined[record.wave as usize] = true;
                }
                let required = kind.and_then(|k| expected(k, *exec_mask));
                if kind == Some("s_endpgm") {
                    ended.insert(record.wave);
                }
                if required.is_none() {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "unrecognized gfx1250 instruction provenance kind",
                    ));
                }
                instructions.insert(
                    record.wave,
                    InstructionRecord {
                        event: record.event,
                        pc,
                        exec_mask: *exec_mask,
                        reads: reads.clone(),
                        writes: writes.clone(),
                        expected: required.flatten(),
                        semantic: false,
                    },
                );
            }
            AmdEventKind::MemoryIssue {
                class,
                counters,
                destinations,
                held_sources,
            } => {
                let instruction =
                    claim(graph, record, &mut instructions, "memory", &mut unresolved);
                if matches!(
                    class,
                    AmdMemoryClass::AsyncLoadToLds | AmdMemoryClass::AsyncStoreFromLds
                ) && default_scheduling_count != 1
                {
                    unresolved.push(Unresolved::at(record.event,
                        "per-lane async copies require exactly one explicit default-scheduling premise"));
                }
                registers(destinations, physical_mask, record.event, &mut unresolved);
                registers(held_sources, physical_mask, record.event, &mut unresolved);
                if matches!(
                    class,
                    AmdMemoryClass::TensorLoadToLds
                        | AmdMemoryClass::TensorStoreFromLds
                        | AmdMemoryClass::AsyncLoadToLds
                        | AmdMemoryClass::AsyncStoreFromLds
                ) && !held_sources.is_empty()
                {
                    // TDM and default-scheduled per-lane async copies capture
                    // their inputs at issue. Neither has VMEM_GROUP/X leases.
                    // Read readiness still comes from the Instruction record.
                    unresolved.push(Unresolved::at(
                        record.event,
                        "async and tensor inputs are captured at issue and have no completion-held register sources",
                    ));
                }
                let correct_destinations = match class {
                    AmdMemoryClass::ScalarLoad => {
                        !destinations.is_empty() && destinations.iter().all(|r| r.is_uniform())
                    }
                    AmdMemoryClass::VectorLoad
                    | AmdMemoryClass::FlatGlobalLoad
                    | AmdMemoryClass::FlatLdsLoad
                    | AmdMemoryClass::CrossLane => {
                        !destinations.is_empty()
                            && destinations
                                .iter()
                                .all(|r| r.file == AmdRegisterFile::Vector)
                    }
                    AmdMemoryClass::Ds => destinations
                        .iter()
                        .all(|r| r.file == AmdRegisterFile::Vector),
                    AmdMemoryClass::VectorStore
                    | AmdMemoryClass::Prefetch
                    | AmdMemoryClass::FlatGlobalStore
                    | AmdMemoryClass::FlatLdsStore
                    | AmdMemoryClass::GlobalAtomicAddF32NoReturn
                    | AmdMemoryClass::AsyncLoadToLds
                    | AmdMemoryClass::AsyncStoreFromLds
                    | AmdMemoryClass::TensorLoadToLds
                    | AmdMemoryClass::TensorStoreFromLds => destinations.is_empty(),
                    _ => false,
                };
                if !correct_destinations {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "gfx1250 memory class has invalid destination register roles",
                    ));
                }
                if let Some(instruction) = instruction {
                    let source_kind =
                        graph.provenance().instructions[instruction.pc.unwrap().0 as usize].kind;
                    let correct_kind = match class {
                        AmdMemoryClass::Prefetch => source_kind == "prefetch",
                        AmdMemoryClass::ScalarLoad => source_kind == "scalar memory",
                        AmdMemoryClass::VectorLoad | AmdMemoryClass::VectorStore => {
                            source_kind == "vector memory"
                        }
                        AmdMemoryClass::FlatGlobalLoad
                        | AmdMemoryClass::FlatGlobalStore
                        | AmdMemoryClass::FlatLdsLoad
                        | AmdMemoryClass::FlatLdsStore => source_kind == "flat memory",
                        AmdMemoryClass::GlobalAtomicAddF32NoReturn => {
                            source_kind == "atomic f32 add no return"
                        }
                        AmdMemoryClass::Ds | AmdMemoryClass::CrossLane => source_kind == "LDS",
                        AmdMemoryClass::AsyncLoadToLds | AmdMemoryClass::AsyncStoreFromLds => {
                            source_kind == "async memory"
                        }
                        AmdMemoryClass::TensorLoadToLds | AmdMemoryClass::TensorStoreFromLds => {
                            source_kind == "tensor memory"
                        }
                        _ => false,
                    };
                    if !correct_kind {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "gfx1250 operation class disagrees with its instruction provenance",
                        ));
                    }
                    if destinations.len() != instruction.writes.len()
                        || !destinations.iter().all(|r| instruction.writes.contains(r))
                    {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "memory destinations differ from the instruction's writes",
                        ));
                    }
                    if (replay_group(*class).is_some() || flat_profile(*class).is_some())
                        && (held_sources.len() != instruction.reads.len()
                            || !instruction.reads.iter().all(|r| held_sources.contains(r)))
                    {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "replay-sensitive memory issue must retain every instruction source",
                        ));
                    }
                    if !held_sources.iter().all(|r| instruction.reads.contains(r)) {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "memory held source is missing from its instruction",
                        ));
                    }
                    if flat_profile(*class).is_some()
                        && !instruction
                            .reads
                            .iter()
                            .any(|r| r.file == AmdRegisterFile::Scalar && r.index == 126)
                    {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "gfx1250 FLAT requires its implicit EXEC replay source",
                        ));
                    }
                    if *class == AmdMemoryClass::GlobalAtomicAddF32NoReturn
                        && (instruction.exec_mask.count_ones() != 1
                            || !held_sources
                                .iter()
                                .any(|r| r.file == AmdRegisterFile::Scalar && r.index == 126))
                    {
                        unresolved.push(Unresolved::at(record.event,
                            "selected atomic RMW requires one active lane and its implicit EXEC source"));
                    }
                }
                if !charges(*class).is_some_and(|expected| {
                    expected.len() == counters.len()
                        && expected.iter().all(|c| counters.contains(c))
                }) {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "gfx1250 counter charges do not match the native operation class",
                    ));
                }
                issues.insert(
                    record.event,
                    (record, instruction.map_or(0, |i| i.exec_mask)),
                );
                if *class == AmdMemoryClass::Prefetch {
                    open_prefetch.insert(record.wave, record.event);
                }
            }
            AmdEventKind::TensorAccessFault {
                descriptor,
                sources,
                ..
            } => {
                let instruction =
                    claim(graph, record, &mut instructions, "memory", &mut unresolved);
                let mut required = HashSet::new();
                for (&start, count) in sources.iter().zip([4u16, 8, 4, 4]) {
                    for offset in 0..count {
                        if let Some(index) = start.checked_add(offset) {
                            required.insert(index);
                        } else {
                            unresolved.push(Unresolved::at(
                                record.event,
                                "tensor descriptor source register overflow",
                            ));
                        }
                    }
                }
                if !prefix
                    || instruction.is_none_or(|i| {
                        i.exec_mask == 0
                            || !i.writes.is_empty()
                            || graph.provenance().instructions[i.pc.unwrap().0 as usize].kind
                                != "tensor memory"
                            || required.iter().any(|index| {
                                !i.reads
                                    .iter()
                                    .any(|r| r.file == AmdRegisterFile::Scalar && r.index == *index)
                            })
                            || i.reads.iter().any(|r| {
                                r.file != AmdRegisterFile::Scalar
                                    || !required.contains(&r.index) && r.index != 126
                            })
                    })
                    || descriptor.decode().is_err()
                    || record.event.0 as usize + 1 != graph.events().len()
                {
                    unresolved.push(Unresolved::at(record.event, "tensor access fault requires an exact active terminal tensor instruction and independent prefix validation"));
                }
            }
            AmdEventKind::WaitCounters { thresholds } => {
                claim(graph, record, &mut instructions, "wait", &mut unresolved);
                let mut distinct = HashSet::new();
                if thresholds.is_empty()
                    || thresholds.iter().any(|(counter, value)| {
                        !distinct.insert(*counter)
                            || maximum(*counter).is_none_or(|max| *value > max)
                    })
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "empty, duplicate, legacy or out-of-range gfx1250 wait threshold",
                    ));
                }
            }
            AmdEventKind::SplitBarrierSignal {
                barrier, exec_mask, ..
            } => {
                let instruction =
                    claim(graph, record, &mut instructions, "signal", &mut unresolved);
                if instruction.is_some_and(|i| i.exec_mask != *exec_mask) {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "split-barrier EXEC differs from its instruction record",
                    ));
                }
                if *barrier != -1 || exec_mask & !physical_mask != 0 {
                    unresolved.push(Unresolved::at(record.event, "only the shader workgroup barrier -1 is implemented; invalid EXEC or privileged/named/cluster barrier"));
                }
            }
            AmdEventKind::SplitBarrierWait { barrier, .. } => {
                claim(
                    graph,
                    record,
                    &mut instructions,
                    "barrier wait",
                    &mut unresolved,
                );
                if *barrier != -1 {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "unsupported gfx1250 barrier object",
                    ));
                }
            }
            AmdEventKind::SplitBarrierRelease { barrier, .. } => {
                if *barrier != -1 {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "unsupported gfx1250 barrier release object",
                    ));
                }
            }
            AmdEventKind::NamedBarrierInit { object, .. }
            | AmdEventKind::NamedBarrierJoin { object, .. }
            | AmdEventKind::NamedBarrierSignal { object, .. }
            | AmdEventKind::NamedBarrierSignalWithCount { object, .. }
            | AmdEventKind::NamedBarrierWait { object, .. }
            | AmdEventKind::NamedBarrierRelease { object, .. }
            | AmdEventKind::NamedBarrierLeave { object, .. } => {
                if object.id > named_capacity || object.id > 16 {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "named object exceeds the explicit native descriptor allocation",
                    ));
                }
                let required = match record.kind {
                    AmdEventKind::NamedBarrierInit { .. } => Some("named init"),
                    AmdEventKind::NamedBarrierJoin { .. } => Some("named join"),
                    AmdEventKind::NamedBarrierSignal { .. }
                    | AmdEventKind::NamedBarrierSignalWithCount { .. } => Some("named signal"),
                    AmdEventKind::NamedBarrierWait { encoded, .. } => {
                        if encoded > 16 {
                            unresolved.push(Unresolved::at(
                                record.event,
                                "invalid encoded named WAIT selector",
                            ));
                        }
                        Some("named wait")
                    }
                    AmdEventKind::NamedBarrierLeave { .. } => Some("named leave"),
                    _ => None,
                };
                if let Some(required) = required
                    && let Some(instruction) =
                        claim(graph, record, &mut instructions, required, &mut unresolved)
                {
                    if matches!(record.kind, AmdEventKind::NamedBarrierInit { object, .. } if object.id != 0)
                        && !instruction
                            .reads
                            .iter()
                            .any(|r| r.file == AmdRegisterFile::Scalar && r.index == 124)
                    {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "named INIT requires its implicit M0 expected-count source",
                        ));
                    }
                    if let AmdEventKind::NamedBarrierSignalWithCount {
                        object, expected, ..
                    } = record.kind
                        && (object.id == 0
                            || !(1..=127).contains(&expected)
                            || !instruction
                                .reads
                                .iter()
                                .any(|r| r.file == AmdRegisterFile::Scalar && r.index == 124))
                    {
                        unresolved.push(Unresolved::at(record.event, "named count-updating signal requires non-NULL object, 7-bit nonzero expected count and explicit M0 source"));
                    }
                    if let AmdEventKind::NamedBarrierSignal { exec_mask, .. }
                    | AmdEventKind::NamedBarrierSignalWithCount { exec_mask, .. } = record.kind
                        && (exec_mask != instruction.exec_mask || exec_mask & !physical_mask != 0)
                    {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "named signal EXEC differs from the instruction record",
                        ));
                    }
                }
            }
            AmdEventKind::PrefetchFootprint {
                issue,
                scope,
                lanes,
            } => {
                let mask = instructions
                    .get(&record.wave)
                    .filter(|i| i.event < *issue)
                    .map_or(0, |i| i.exec_mask);
                if open_prefetch.remove(&record.wave) != Some(*issue)
                    || !prefetch_footprints.insert(*issue)
                    || !exact_active_lanes(lanes, mask)
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "prefetch footprint must follow its own issue and list every active lane once",
                    ));
                }
                let access = AmdGlobalAccess::Prefetch(*scope);
                if let Some(reason) = lanes
                    .iter()
                    .find_map(|&(_, address)| global_bounds_defect(graph, access, address, 1))
                {
                    unresolved.push(Unresolved::at(record.event, reason));
                }
            }
            AmdEventKind::GlobalAccessFault {
                access,
                bytes_per_lane,
                lanes,
            } => {
                let instruction =
                    claim(graph, record, &mut instructions, "memory", &mut unresolved);
                let kind = match access {
                    AmdGlobalAccess::Prefetch(_) => "prefetch",
                    AmdGlobalAccess::Load | AmdGlobalAccess::Store => "vector memory",
                };
                if !prefix
                    || graph.global_region_layout().is_empty()
                    || *bytes_per_lane == 0
                    || record.event.0 as usize + 1 != graph.events().len()
                    || instruction.is_none_or(|i| {
                        i.exec_mask == 0
                            || graph.provenance().instructions[i.pc.unwrap().0 as usize].kind
                                != kind
                            || !exact_active_lanes(lanes, i.exec_mask)
                    })
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "global access fault requires an exact active terminal instruction, a configured layout and independent prefix validation",
                    ));
                }
            }
            AmdEventKind::ModeAccess { offset, width, .. } => {
                // A MODE access is recorded only by the B0 m2 profile, right
                // after its own ordinary scalar instruction record (GETREG is
                // provenance kind "xcnt drain"); it has no register effects.
                let profile = graph
                    .provenance()
                    .assumptions
                    .contains(&Assumption::Gfx1250B0M2ModeProfile);
                let own = instructions.get(&record.wave).is_some_and(|i| {
                    i.pc == graph.event(record.event).pc
                        && i.expected.is_none()
                        && i.pc.is_some_and(|pc| {
                            matches!(
                                graph.provenance().instructions[pc.0 as usize].kind,
                                "scalar ALU" | "xcnt drain"
                            )
                        })
                });
                if !profile || !own || !matches!((offset, width), (0 | 4, 4)) {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "MODE access requires the B0 m2 profile, its own scalar instruction record and an FP_ROUND/FP_DENORM field",
                    ));
                }
            }
            AmdEventKind::BufferRangeCheck { issue, check } => {
                // Recorded only by the B0 litmus profile, after the effects of
                // its own untyped buffer issue, once per issue, for exactly the
                // issue's active lanes; each in-range mask is recomputed here.
                let profile = graph
                    .provenance()
                    .assumptions
                    .contains(&Assumption::Gfx1250B0LitmusProfile);
                let mask = issues.get(issue).and_then(|(issued, mask)| {
                    (issued.wave == record.wave
                        && graph.event(*issue).pc == graph.event(record.event).pc
                        && matches!(
                            issued.kind,
                            AmdEventKind::MemoryIssue {
                                class: AmdMemoryClass::VectorLoad | AmdMemoryClass::VectorStore,
                                ..
                            }
                        ))
                    .then_some(*mask)
                });
                let lanes: Vec<_> = check.lanes.iter().map(|l| (l.lane, l.offset)).collect();
                if !profile
                    || !(1..=4).contains(&check.dwords)
                    || mask.is_none_or(|mask| !exact_active_lanes(&lanes, mask))
                    || check
                        .lanes
                        .iter()
                        .any(|l| l.in_range != check.expected_in_range(l.offset))
                    || range_checks.insert(*issue, check).is_some()
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "buffer range check requires the B0 litmus profile, its own buffer issue, every active lane once and the documented per-DWORD rule",
                    ));
                }
            }
            AmdEventKind::AluWait { .. } => {
                // Every encoding is a valid ALU-only wait; none retires memory.
                claim(
                    graph,
                    record,
                    &mut instructions,
                    "alu wait",
                    &mut unresolved,
                );
            }
            AmdEventKind::Delay { cycles } => {
                claim(graph, record, &mut instructions, "delay", &mut unresolved);
                if !(1..=16).contains(cycles) {
                    unresolved.push(Unresolved::at(record.event, "invalid S_NOP duration"));
                }
            }
            AmdEventKind::DelayAlu {
                instid0,
                instskip,
                instid1,
            } => {
                claim(
                    graph,
                    record,
                    &mut instructions,
                    "delay alu",
                    &mut unresolved,
                );
                if instid0.encoding().is_none()
                    || instid1.is_some_and(|d| d.encoding().is_none())
                    || *instskip > 5
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "unsupported S_DELAY_ALU selector or skip encoding",
                    ));
                }
            }
            AmdEventKind::WmmaIssue {
                destinations,
                sources_ab,
                accumulator,
                kind,
                scales,
                index,
            } => {
                let instruction = claim(graph, record, &mut instructions, "wmma", &mut unresolved);
                for regs in [destinations, sources_ab, accumulator, scales, index] {
                    registers(regs, physical_mask, record.event, &mut unresolved);
                }
                let (a_inputs, b_inputs) = (kind.a_registers(), kind.b_registers());
                if physical_mask != u32::MAX as u64
                    || instruction.is_none_or(|i| i.exec_mask != u32::MAX as u64)
                    || !vector_tuple(destinations, kind.destination_registers())
                    || sources_ab.len() != a_inputs + b_inputs
                    || !vector_tuple(&sources_ab[..sources_ab.len().min(a_inputs)], a_inputs)
                    || !vector_tuple(&sources_ab[sources_ab.len().min(a_inputs)..], b_inputs)
                    || (!accumulator.is_empty()
                        && !vector_tuple(accumulator, kind.accumulator_registers()))
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "WMMA requires a full physical wave32 and exact contiguous operand tuples",
                    ));
                }
                if instruction.is_some_and(|i| {
                    i.writes != *destinations
                        || !sources_ab
                            .iter()
                            .chain(accumulator)
                            .chain(scales)
                            .chain(index)
                            .all(|r| i.reads.contains(r))
                        || !i
                            .reads
                            .iter()
                            .any(|r| r.file == AmdRegisterFile::Scalar && r.index == 126)
                        || i.reads.iter().any(|r| {
                            !sources_ab.contains(r)
                                && !accumulator.contains(r)
                                && !scales.contains(r)
                                && !index.contains(r)
                                && !(r.file == AmdRegisterFile::Scalar && r.index == 126)
                        })
                }) {
                    unresolved.push(Unresolved::at(record.event, "WMMA operand roles or implicit EXEC do not match instruction register signatures"));
                }
            }
            _ => unresolved.push(Unresolved::at(
                record.event,
                "legacy CDNA semantic event cannot describe gfx1250",
            )),
        }
    }
    for instruction in instructions.values() {
        check_missing_semantic(instruction, &mut unresolved);
    }
    if matches!(graph.terminal(), Terminal::Completed) && ended.len() != waves {
        unresolved.push(Unresolved::global(
            "completed gfx1250 trace requires an S_ENDPGM instruction for every wave",
        ));
    }
    if waves == 0 || !prefix && seen.iter().any(|seen| !seen) {
        unresolved.push(Unresolved::global(
            "every launched gfx1250 wave requires instruction/register records",
        ));
    }
    for (&wave, &issue) in &open_prefetch {
        if prefix
            && graph
                .amd_events()
                .last()
                .is_some_and(|r| r.event == issue && r.wave == wave)
        {
            continue;
        }
        unresolved.push(Unresolved::at(
            issue,
            "prefetch issue has no footprint record",
        ));
    }
    let mut associations = HashMap::new();
    let mut represented = HashMap::<EventId, HashSet<u32>>::new();
    let mut transferred = HashMap::<EventId, HashMap<ThreadIx, (u64, u64)>>::new();
    let mut footprints = HashMap::<(EventId, ThreadIx), (u64, u64)>::new();
    let mut buffer_effects = HashMap::<(EventId, u32), Vec<(u64, u32)>>::new();
    let mut last_record = HashMap::new();
    for link in graph.amd_memory_effects() {
        associations.insert(link.effect, link);
    }
    for (event_id, event) in graph.events().iter() {
        match event.kind {
            EventKind::Amd(index) => {
                last_record.insert(graph.amd_event(index).unwrap().wave, event_id);
            }
            EventKind::Access(access_id) => {
                let access = graph.access(access_id);
                let Some(link) = associations.get(&event_id) else {
                    unresolved.push(Unresolved::at(
                        event_id,
                        "memory effect is missing its gfx1250 issue association",
                    ));
                    continue;
                };
                let Some(&(issue, _)) = issues.get(&link.issue) else {
                    unresolved.push(Unresolved::at(
                        event_id,
                        "effect names an unsupported or absent memory issue",
                    ));
                    continue;
                };
                let AmdEventKind::MemoryIssue {
                    class,
                    destinations,
                    ..
                } = &issue.kind
                else {
                    unreachable!()
                };
                let wave = access.attributed.0 / WAVE as u32;
                if wave != issue.wave
                    || last_record.get(&wave) != Some(&link.issue)
                    || graph.event(link.issue).pc != event.pc
                {
                    unresolved.push(Unresolved::at(
                        event_id,
                        "effect has wrong wave, source point, or issuing instruction order",
                    ));
                }
                represented
                    .entry(link.issue)
                    .or_default()
                    .insert(access.attributed.0 % WAVE as u32);
                let mask = issues[&link.issue].1;
                if matches!(
                    class,
                    AmdMemoryClass::VectorLoad
                        | AmdMemoryClass::VectorStore
                        | AmdMemoryClass::FlatGlobalLoad
                        | AmdMemoryClass::FlatGlobalStore
                        | AmdMemoryClass::FlatLdsLoad
                        | AmdMemoryClass::FlatLdsStore
                        | AmdMemoryClass::GlobalAtomicAddF32NoReturn
                        | AmdMemoryClass::Ds
                        | AmdMemoryClass::AsyncLoadToLds
                        | AmdMemoryClass::AsyncStoreFromLds
                        | AmdMemoryClass::TensorLoadToLds
                        | AmdMemoryClass::TensorStoreFromLds
                ) && mask & (1u64 << (access.attributed.0 % 32)) == 0
                {
                    unresolved.push(Unresolved::at(
                        event_id,
                        "memory effect belongs to an inactive EXEC lane",
                    ));
                }
                let read = access.mode == AccessMode::Read;
                let valid = match class {
                    AmdMemoryClass::VectorLoad | AmdMemoryClass::ScalarLoad => {
                        access.space == Space::Global && read
                    }
                    AmdMemoryClass::VectorStore => access.space == Space::Global && !read,
                    AmdMemoryClass::GlobalAtomicAddF32NoReturn => access.space == Space::Global,
                    AmdMemoryClass::Ds => {
                        access.space == Space::Shared && (read != destinations.is_empty())
                    }
                    AmdMemoryClass::AsyncLoadToLds | AmdMemoryClass::TensorLoadToLds => matches!(
                        (access.space, read),
                        (Space::Global, true) | (Space::Shared, false)
                    ),
                    AmdMemoryClass::AsyncStoreFromLds | AmdMemoryClass::TensorStoreFromLds => {
                        matches!(
                            (access.space, read),
                            (Space::Shared, true) | (Space::Global, false)
                        )
                    }
                    _ => flat_profile(*class)
                        .is_some_and(|(space, mode)| access.space == space && access.mode == mode),
                };
                let atomic = *class == AmdMemoryClass::GlobalAtomicAddF32NoReturn;
                if !valid
                    || access.ordering
                        != if atomic {
                            OrderingKind::Atomic
                        } else {
                            OrderingKind::Ordinary
                        }
                    || access.proxy != Proxy::Generic
                {
                    unresolved.push(Unresolved::at(
                        event_id,
                        "gfx1250 memory class/effect space, direction or ordering disagree",
                    ));
                }
                if link.scope == AmdMemoryScope::Workgroup {
                    unresolved.push(Unresolved::at(event_id,
                        "legacy CDNA workgroup visibility does not certify gfx1250 SCOPE/cache publication"));
                }
                if link.scope == AmdMemoryScope::Gfx1250System
                    && !(access.space == Space::Global
                        && matches!(
                            class,
                            AmdMemoryClass::VectorLoad
                                | AmdMemoryClass::VectorStore
                                | AmdMemoryClass::FlatGlobalLoad
                                | AmdMemoryClass::FlatGlobalStore
                        ))
                {
                    unresolved.push(Unresolved::at(
                        event_id,
                        "gfx1250 system scope requires an ordinary GLOBAL or resolved-global FLAT effect",
                    ));
                }
                if let Footprint::Bytes { addr, width } = access.footprint {
                    if range_checks.contains_key(&link.issue) {
                        buffer_effects
                            .entry((link.issue, access.attributed.0 % WAVE as u32))
                            .or_default()
                            .push((addr, width));
                    }
                    let global_access = match (access.space, *class, read) {
                        (Space::Global, AmdMemoryClass::ScalarLoad, _) | (Space::Shared, ..) => {
                            None
                        }
                        (Space::Global, _, true) => Some(AmdGlobalAccess::Load),
                        (Space::Global, _, false) => Some(AmdGlobalAccess::Store),
                        _ => None,
                    };
                    if let Some(reason) = global_access
                        .and_then(|global| global_bounds_defect(graph, global, addr, width as u64))
                    {
                        unresolved.push(Unresolved::at(event_id, reason));
                    }
                    if atomic {
                        let pair = atomic_pairs
                            .entry((link.issue, access.attributed))
                            .or_default();
                        let part = if read { &mut pair.0 } else { &mut pair.1 };
                        if width != 4 || addr % 4 != 0 || part.replace((event_id, addr)).is_some() {
                            unresolved.push(Unresolved::at(event_id,
                                "atomic F32 RMW requires exactly one aligned 4-byte read and write per active lane"));
                        }
                    }
                    let count = transferred
                        .entry(link.issue)
                        .or_default()
                        .entry(access.attributed)
                        .or_default();
                    if read {
                        count.0 += width as u64;
                    } else {
                        count.1 += width as u64;
                    }
                    if let Some(end) = addr.checked_add(width as u64) {
                        if access.space == Space::Shared {
                            if end > AmdArchitecture::Gfx1250.lds_capacity() {
                                unresolved.push(Unresolved::at(
                                    event_id,
                                    "LDS effect exceeds gfx1250 capacity",
                                ));
                            }
                            footprints
                                .entry((link.issue, access.attributed))
                                .and_modify(|span| {
                                    span.0 = span.0.min(addr);
                                    span.1 = span.1.max(end);
                                })
                                .or_insert((addr, end));
                        }
                    } else {
                        unresolved.push(Unresolved::at(
                            event_id,
                            "memory footprint overflows concrete address space",
                        ));
                    }
                } else {
                    unresolved.push(Unresolved::at(
                        event_id,
                        "PTX tensor-memory footprints are not gfx1250 LDS",
                    ));
                }
            }
            EventKind::Exit => {}
            _ => unresolved.push(Unresolved::at(
                event_id,
                "PTX synchronization/async certificate is not a gfx1250 event",
            )),
        }
    }
    for (&id, &(record, mask)) in &issues {
        let AmdEventKind::MemoryIssue {
            class,
            destinations,
            ..
        } = &record.kind
        else {
            unreachable!()
        };
        let lanes = represented.get(&id);
        if let Some(check) = range_checks.get(&id) {
            // A litmus buffer issue: each lane's effects are exactly its
            // in-range DWORDs at base + offset + 4c; nothing else.
            let exact = check.lanes.iter().all(|lane| {
                let mut want: Vec<_> = (0..check.dwords)
                    .filter(|c| lane.in_range & (1 << c) != 0)
                    .map(|c| (check.base + lane.offset + 4 * u64::from(c), 4))
                    .collect();
                let mut got = buffer_effects
                    .get(&(id, u32::from(lane.lane)))
                    .cloned()
                    .unwrap_or_default();
                want.sort_unstable();
                got.sort_unstable();
                got == want
            }) && lanes.is_none_or(|lanes| {
                lanes
                    .iter()
                    .all(|lane| check.lanes.iter().any(|l| u32::from(l.lane) == *lane))
            });
            if !exact {
                unresolved.push(Unresolved::at(
                    id,
                    "buffer effects differ from the in-range DWORDs of the range check",
                ));
            }
        } else if *class == AmdMemoryClass::Prefetch {
            if lanes.is_some() || !destinations.is_empty() || !prefetch_footprints.contains(&id) {
                unresolved.push(Unresolved::at(
                    id,
                    "prefetch must have one footprint record and no data effects or results",
                ));
            }
        } else if *class == AmdMemoryClass::CrossLane {
            if lanes.is_some() || destinations.is_empty() {
                unresolved.push(Unresolved::at(
                    id,
                    "cross-lane operation must have destinations and no fabricated LDS effects",
                ));
            }
        } else if lanes.is_none() {
            unresolved.push(Unresolved::at(
                id,
                "memory instruction has no memory effects",
            ));
        } else if matches!(
            class,
            AmdMemoryClass::VectorLoad
                | AmdMemoryClass::VectorStore
                | AmdMemoryClass::FlatGlobalLoad
                | AmdMemoryClass::FlatGlobalStore
                | AmdMemoryClass::FlatLdsLoad
                | AmdMemoryClass::FlatLdsStore
                | AmdMemoryClass::GlobalAtomicAddF32NoReturn
                | AmdMemoryClass::Ds
                | AmdMemoryClass::AsyncLoadToLds
                | AmdMemoryClass::AsyncStoreFromLds
        ) && (0..32)
            .any(|lane| mask & (1u64 << lane) != 0 && !lanes.unwrap().contains(&lane))
        {
            unresolved.push(Unresolved::at(
                id,
                "memory instruction omits an active lane's effects",
            ));
        }
        if *class == AmdMemoryClass::GlobalAtomicAddF32NoReturn
            && atomic_pairs.iter().filter(|((issue, _), _)| *issue == id).any(|(_, (read, write))| {
                !matches!((read, write), (Some((r, a)), Some((w, b))) if r < w && a == b)
            })
        {
            unresolved.push(Unresolved::at(id,
                "atomic RMW effects must read then write the same 4-byte location"));
        }
        if matches!(
            class,
            AmdMemoryClass::AsyncLoadToLds
                | AmdMemoryClass::AsyncStoreFromLds
                | AmdMemoryClass::TensorLoadToLds
                | AmdMemoryClass::TensorStoreFromLds
        ) && transferred.get(&id).is_none_or(|lanes| {
            lanes
                .values()
                .any(|&(read, write)| read == 0 || read != write)
        }) {
            unresolved.push(Unresolved::at(id, "DMA instruction requires paired source/destination footprints of equal byte size for each attributed lane"));
        }
    }
    if !graph.shared_allocation_layout().is_empty() {
        for ((issue, _), (start, end)) in footprints {
            if !graph.shared_allocation_layout().iter().any(|allocation| {
                start >= allocation.offset
                    && allocation
                        .offset
                        .checked_add(allocation.size)
                        .is_some_and(|limit| end <= limit)
            }) {
                unresolved.push(Unresolved::at(
                    issue,
                    "full LDS instruction footprint crosses configured allocation boundaries",
                ));
            }
        }
    }
    PropertyResult::from_findings(
        AmdTraceEvidence {
            waves,
            instructions: instruction_count,
            issues: issues.len(),
            effects: associations.len(),
        },
        Vec::new(),
        unresolved,
    )
}

fn complete(op: &Operation) -> bool {
    op.remaining.is_empty()
}
fn has(op: &Operation, counter: AmdCounter) -> bool {
    op.remaining.iter().any(|&(c, _)| c == counter)
}

fn wait(counter: AmdCounter, threshold: u16, pending: &[usize], operations: &mut [Operation]) {
    if Some(threshold) == maximum(counter) || counter == AmdCounter::Tensor && threshold != 0 {
        return;
    }
    let selected: Vec<_> = pending
        .iter()
        .copied()
        .filter(|&i| has(&operations[i], counter))
        .collect();
    // SMEM completion/translation may return out of order. A partial count does
    // not establish which scalar destination or replay source has retired.
    if counter == AmdCounter::Km && !super::early_readiness::km_wait_completes(threshold)
        || threshold != 0
            && counter == AmdCounter::X
            && selected
                .iter()
                .any(|&i| operations[i].class == AmdMemoryClass::ScalarLoad)
    {
        return;
    }
    // CDNA5 section10.8: async loads complete FIFO with loads, and stores
    // with stores; the two streams may complete out of order. A partial
    // mixed-stream count cannot identify the globally oldest operation.
    if counter == AmdCounter::Async
        && threshold != 0
        && selected.first().is_some_and(|&first| {
            selected
                .iter()
                .any(|&i| operations[i].class != operations[first].class)
        })
    {
        return;
    }
    let count = selected.len().saturating_sub(threshold as usize);
    for index in selected.into_iter().take(count) {
        let op = &mut operations[index];
        op.remaining.retain(|&(c, _)| c != counter);
        // Data cannot complete before its own address translation. This does
        // not drain translations of unrelated operations or other counters.
        if matches!(
            counter,
            AmdCounter::Load | AmdCounter::Store | AmdCounter::Km
        ) {
            op.remaining.retain(|&(c, _)| c != AmdCounter::X);
        }
    }
}

#[derive(Clone)]
struct WmmaOperation {
    event: EventId,
    ordinal: u64,
    kind: AmdWmmaKind,
    destinations: Vec<AmdRegister>,
    sources: Vec<AmdRegister>,
}
impl WmmaOperation {
    fn is_xdl(&self) -> bool {
        self.kind != AmdWmmaKind::F32_16x16x4
    }
    fn valu_wait(&self) -> u64 {
        match self.kind {
            AmdWmmaKind::F16_16x16x32 | AmdWmmaKind::Bf16_16x16x32 => 4,
            AmdWmmaKind::F32_16x16x4 => 0,
            AmdWmmaKind::Fp8Fp8_16x16x128
            | AmdWmmaKind::Fp8Bf8_16x16x128
            | AmdWmmaKind::Bf8Fp8_16x16x128
            | AmdWmmaKind::Bf8Bf8_16x16x128
            | AmdWmmaKind::F4_32x16x128 => 3,
            AmdWmmaKind::F8f6f4_16x16x128 { a, b }
                if !matches!((a, b), (AmdSmallFloat::Fp4, AmdSmallFloat::Fp4)) =>
            {
                3
            }
            AmdWmmaKind::Iu8_16x16x64 => 8,
            AmdWmmaKind::Half16x16x32 { .. } => 4,
            AmdWmmaKind::F16Out8 { k128: true, .. } => 3,
            AmdWmmaKind::Swmmac {
                input: AmdSwmmacInput::Iu8,
                ..
            } => 4,
            AmdWmmaKind::Swmmac { .. } => 2,
            _ => 1,
        }
    }
}

pub(super) fn check(graph: &EventGraph) -> AmdReport {
    check_with_scope(graph, false)
}

pub(super) fn check_prefix(graph: &EventGraph) -> AmdReport {
    check_with_scope(graph, true)
}

fn check_with_scope(graph: &EventGraph, prefix: bool) -> AmdReport {
    let trace = contract(graph, prefix);
    if trace.verdict() != Verdict::Pass {
        return AmdReport {
            trace,
            wait_readiness: unknown("gfx1250 trace contract did not pass"),
            barrier: unknown("gfx1250 trace contract did not pass"),
            memory_conflict: unknown("gfx1250 trace contract did not pass"),
        };
    }
    let waves = (graph.threads() as usize).div_ceil(WAVE);
    let mut clocks = vec![vec![0; waves * CLASSES]; waves];
    let mut execution_clocks = vec![vec![0; waves]; waves];
    let mut execution_signals = HashMap::new();
    let mut named = named::State::new(waves);
    let mut arrival_clocks: Option<Vec<Vec<u32>>> = None;
    let mut arrival_signals = HashMap::new();
    let mut sequences = vec![0; waves * CLASSES];
    let mut pending = vec![Vec::<usize>::new(); waves];
    let mut operations = Vec::<Operation>::new();
    let mut issue_index = HashMap::new();
    let mut hazards = Vec::new();
    let mut scalar_readiness = super::early_readiness::ScalarReadinessState::default();
    let mut unresolved = Vec::new();
    let mut barriers = HashMap::<(i16, u32), HashMap<usize, Vec<u32>>>::new();
    let mut signals = vec![None; waves];
    let mut blocked = vec![None; waves];
    let mut released = HashSet::new();
    let mut barrier_violations = Vec::new();
    let mut barrier_unresolved = Vec::new();
    let waited_phases: HashSet<_> = graph
        .amd_events()
        .iter()
        .filter_map(|record| match record.kind {
            AmdEventKind::SplitBarrierWait {
                barrier,
                generation,
            } => Some((barrier, generation)),
            _ => None,
        })
        .collect();
    let mut completed_waits = HashMap::<(i16, u32), Vec<(usize, u32)>>::new();
    let mut arrivals = 0;
    let mut waits = 0;
    let mut next_generation = vec![0; waves];
    let mut groups = HashMap::new();
    let mut translation_receivers = HashSet::new();
    let mut instruction_at = HashMap::new();
    let mut wmma_at = HashMap::new();
    // B0 multiwave profile: the wave-start single-group replay mode issues
    // a group of memory instructions, then waits for XCNT to drain before
    // the first non-memory instruction.
    let single_group_replay = graph
        .provenance()
        .assumptions
        .contains(&Assumption::Gfx1250B0MultiwaveProfile);
    let mut valu_ordinal = vec![0u64; waves];
    let mut matrices = vec![Vec::<WmmaOperation>::new(); waves];
    for record in graph.amd_events() {
        match record.kind {
            AmdEventKind::Instruction { .. } => {
                instruction_at.insert(record.wave, record.event);
            }
            AmdEventKind::MemoryIssue { class, .. } => {
                if let Some(group) = replay_group(class) {
                    groups.insert(instruction_at[&record.wave], group);
                }
                if replay_group(class) == Some(true) || class == AmdMemoryClass::FlatLdsLoad {
                    // The generic FLAT encoding can return into a VGPR after
                    // older VMEM translations, even when its concrete address
                    // resolves to LDS. This does not retire older LDS leases
                    // or establish any scalar/group-switch shortcut.
                    translation_receivers.insert(instruction_at[&record.wave]);
                }
            }
            AmdEventKind::WmmaIssue {
                ref sources_ab,
                ref index,
                kind,
                ..
            } => {
                // The sparsity index is a documented A/B/Index source.
                let mut sources: Vec<AmdRegister> = sources_ab.clone();
                sources.extend(index.iter().copied());
                wmma_at.insert(instruction_at[&record.wave], (kind, sources));
            }
            _ => {}
        }
    }
    for record in graph.amd_events() {
        let wave = record.wave as usize;
        hazards.extend(scalar_readiness.observe(record));
        execution_clocks[wave][wave] += 1;
        if let Some((object, phase, _)) = blocked[wave]
            && !matches!(record.kind, AmdEventKind::SplitBarrierRelease { barrier, generation } if (barrier, generation) == (object, phase))
        {
            barrier_violations.push(AmdBarrierViolation {
                generation: phase,
                event: record.event,
                wave: record.wave,
                reason: "wave progressed while blocked at its split-barrier wait".into(),
            });
        }
        match &record.kind {
            AmdEventKind::Instruction { reads, writes, .. } => {
                named.instruction(record);
                let kind = graph
                    .event(record.event)
                    .pc
                    .map(|pc| graph.provenance().instructions[pc.0 as usize].kind)
                    .unwrap();
                // END drops the unnamed object. An arrival participating in a
                // wait needs that wait execution-before its same-wave drop
                // (public execution model, "Barrier drop races").
                if kind == "s_endpgm"
                    && let Some(phase) = signals[wave]
                    && waited_phases.contains(&phase)
                    && !completed_waits.get(&phase).is_some_and(|waits| {
                        waits.iter().any(|&(waiting_wave, tick)| {
                            execution_clocks[wave][waiting_wave] >= tick
                        })
                    })
                {
                    barrier_unresolved.push(Unresolved::at(
                        record.event,
                        "unnamed barrier arrival participates in a wait without a completing wait execution-before END's implicit drop",
                    ));
                }
                let is_valu = matches!(kind, "vector ALU" | "trans ALU" | "shifted ALU" | "WMMA");
                let next_wmma = wmma_at.get(&record.event);
                if is_valu {
                    let ordinal = valu_ordinal[wave];
                    matrices[wave].retain(|m| ordinal.saturating_sub(m.ordinal + 1) < 5);
                    for operation in &matrices[wave] {
                        let independent = ordinal.saturating_sub(operation.ordinal + 1);
                        if let Some((next_kind, ab)) = next_wmma {
                            // GFX12 SrcC forwarding is interlocked, including
                            // overlapping tuples. A/B consumes need explicit
                            // intervening VALU instructions. F32 is non-XDL.
                            let required =
                                if operation.is_xdl() && *next_kind != AmdWmmaKind::F32_16x16x4 {
                                    operation.valu_wait() + 1
                                } else {
                                    1
                                };
                            if independent < required {
                                for &register in ab.iter().filter(|r| {
                                    operation.destinations.iter().any(|d| d.overlaps(**r))
                                }) {
                                    hazards.push(AmdRegisterHazard {
                                        issue: operation.event,
                                        use_event: record.event,
                                        wave: record.wave,
                                        register,
                                        kind: AmdRegisterHazardKind::MatrixReadBeforeReady,
                                    });
                                }
                            }
                        } else if operation.is_xdl() && independent < operation.valu_wait() {
                            for &register in reads
                                .iter()
                                .filter(|r| operation.destinations.iter().any(|d| d.overlaps(**r)))
                            {
                                hazards.push(AmdRegisterHazard {
                                    issue: operation.event,
                                    use_event: record.event,
                                    wave: record.wave,
                                    register,
                                    kind: AmdRegisterHazardKind::MatrixReadBeforeReady,
                                });
                            }
                            for &register in writes {
                                let hazard = if operation
                                    .destinations
                                    .iter()
                                    .any(|d| d.overlaps(register))
                                {
                                    Some(AmdRegisterHazardKind::MatrixDestinationReuse)
                                } else if operation.sources.iter().any(|r| r.overlaps(register)) {
                                    Some(AmdRegisterHazardKind::MatrixSourceReuse)
                                } else {
                                    None
                                };
                                if let Some(kind) = hazard {
                                    hazards.push(AmdRegisterHazard {
                                        issue: operation.event,
                                        use_event: record.event,
                                        wave: record.wave,
                                        register,
                                        kind,
                                    });
                                }
                            }
                        }
                    }
                    valu_ordinal[wave] += 1;
                }
                if matches!(
                    kind,
                    "branch"
                        | "s_endpgm"
                        | "xcnt drain"
                        | "barrier signal"
                        | "barrier wait"
                        | "named barrier signal"
                        | "named barrier wait"
                ) || (single_group_replay && !memory_kind(kind))
                {
                    wait(AmdCounter::X, 0, &pending[wave], &mut operations);
                }
                if let Some(group) = groups.get(&record.event) {
                    for &index in &pending[wave] {
                        let op = &mut operations[index];
                        if replay_group(op.class).is_some_and(|old| old != *group) {
                            op.remaining.retain(|&(c, _)| c != AmdCounter::X);
                        }
                    }
                }
                if translation_receivers.contains(&record.event) {
                    // VMEM translations issue in order. LLVM applies the X
                    // dependency when a later VMEM destination aliases an
                    // older source; neither operation's data completes here.
                    let through = pending[wave]
                        .iter()
                        .copied()
                        .filter(|&index| {
                            let op = &operations[index];
                            replay_group(op.class) == Some(true)
                                && has(op, AmdCounter::X)
                                && writes
                                    .iter()
                                    .any(|r| op.held_sources.iter().any(|s| s.overlaps(*r)))
                        })
                        .max();
                    if let Some(through) = through {
                        for &index in pending[wave].iter().take_while(|&&i| i <= through) {
                            if replay_group(operations[index].class) == Some(true) {
                                operations[index]
                                    .remaining
                                    .retain(|&(c, _)| c != AmdCounter::X);
                            }
                        }
                    }
                }
                for &index in &pending[wave] {
                    let op = &operations[index];
                    if complete(op) {
                        continue;
                    }
                    for reg in reads
                        .iter()
                        .filter(|_| op.class != AmdMemoryClass::ScalarLoad)
                    {
                        if op.destinations.iter().any(|d| d.overlaps(*reg)) {
                            hazards.push(AmdRegisterHazard {
                                issue: op.event,
                                use_event: record.event,
                                wave: record.wave,
                                register: *reg,
                                kind: AmdRegisterHazardKind::ReadBeforeReady,
                            });
                        }
                    }
                    for reg in writes {
                        if op.destinations.iter().any(|d| d.overlaps(*reg)) {
                            hazards.push(AmdRegisterHazard {
                                issue: op.event,
                                use_event: record.event,
                                wave: record.wave,
                                register: *reg,
                                kind: AmdRegisterHazardKind::DestinationReuse,
                            });
                        }
                        let held = if replay_group(op.class).is_some()
                            || flat_profile(op.class).is_some()
                        {
                            has(op, AmdCounter::X)
                        } else {
                            true
                        };
                        if held && op.held_sources.iter().any(|s| s.overlaps(*reg)) {
                            hazards.push(AmdRegisterHazard {
                                issue: op.event,
                                use_event: record.event,
                                wave: record.wave,
                                register: *reg,
                                kind: AmdRegisterHazardKind::SourceReuse,
                            });
                        }
                    }
                }
            }
            AmdEventKind::MemoryIssue {
                class,
                counters,
                destinations,
                held_sources,
            } => {
                // Interleaved scalar/vector groups implicitly drain the older
                // group's X_CNT, not its independent data completion counter.
                if let Some(group) = replay_group(*class) {
                    for &index in &pending[wave] {
                        let op = &mut operations[index];
                        if replay_group(op.class).is_some_and(|old| old != group) {
                            op.remaining.retain(|&(c, _)| c != AmdCounter::X);
                        }
                    }
                }
                if *class == AmdMemoryClass::GlobalAtomicAddF32NoReturn {
                    for &index in &pending[wave] {
                        let old = &operations[index];
                        if has(old, AmdCounter::X)
                            && old.class != AmdMemoryClass::GlobalAtomicAddF32NoReturn
                        {
                            hazards.push(AmdRegisterHazard {
                                issue: old.event,
                                use_event: record.event,
                                wave: record.wave,
                                register: AmdRegister {
                                    file: AmdRegisterFile::Scalar,
                                    index: 126,
                                    lanes: 0,
                                },
                                kind: AmdRegisterHazardKind::AtomicReplayBeforeReady,
                            });
                        }
                    }
                }
                let slot = wave * CLASSES + class_index(*class);
                sequences[slot] += 1;
                let mut clock = clocks[wave].clone();
                let mut arrival_clock = arrival_clocks.as_ref().map(|clocks| clocks[wave].clone());
                // Effect order is within the resolved segment. A GLOBAL prefix
                // must never publish an incomplete LDS operation, or vice versa.
                let streams: &[AmdMemoryClass] = match class {
                    AmdMemoryClass::VectorLoad
                    | AmdMemoryClass::VectorStore
                    | AmdMemoryClass::FlatGlobalLoad
                    | AmdMemoryClass::FlatGlobalStore => &[
                        AmdMemoryClass::VectorLoad,
                        AmdMemoryClass::VectorStore,
                        AmdMemoryClass::FlatGlobalLoad,
                        AmdMemoryClass::FlatGlobalStore,
                    ],
                    AmdMemoryClass::Ds
                    | AmdMemoryClass::FlatLdsLoad
                    | AmdMemoryClass::FlatLdsStore => &[
                        AmdMemoryClass::Ds,
                        AmdMemoryClass::FlatLdsLoad,
                        AmdMemoryClass::FlatLdsStore,
                    ],
                    _ => &[],
                };
                for &stream in streams {
                    let source = wave * CLASSES + class_index(stream);
                    clock[source] = sequences[source] - u32::from(source == slot);
                    if let Some(clock) = &mut arrival_clock {
                        clock[source] = sequences[source] - u32::from(source == slot);
                    }
                }
                issue_index.insert(record.event, operations.len());
                pending[wave].push(operations.len());
                operations.push(Operation {
                    event: record.event,
                    wave: record.wave,
                    class: *class,
                    sequence: sequences[slot],
                    issued_tick: 0,
                    remaining: counters.clone(),
                    destinations: destinations.clone(),
                    held_sources: held_sources.clone(),
                    clock,
                    arrival_clock,
                });
            }
            AmdEventKind::WaitCounters { thresholds } => {
                waits += 1;
                for &(counter, threshold) in thresholds {
                    if counter == AmdCounter::X
                        && threshold != 0
                        && Some(threshold) != maximum(counter)
                        && pending[wave].iter().any(|&index| {
                            matches!(
                                operations[index].class,
                                AmdMemoryClass::FlatLdsLoad | AmdMemoryClass::FlatLdsStore
                            ) && has(&operations[index], counter)
                        })
                    {
                        unresolved.push(Unresolved::at(record.event,
                            "nonzero X wait with pending LDS-aperture FLAT tokens requires an unpinned translation prefix"));
                        continue;
                    }
                    if counter == AmdCounter::Tensor && threshold != 0 {
                        unresolved.push(Unresolved::at(record.event, "nonzero TENSOR_CNT wait requires a pinned hardware counter-width contract"));
                    }
                    wait(counter, threshold, &pending[wave], &mut operations);
                }
            }
            AmdEventKind::SplitBarrierSignal {
                barrier,
                generation,
                ..
            } => {
                arrivals += 1;
                if *generation != next_generation[wave] || signals[wave].is_some() {
                    barrier_violations.push(AmdBarrierViolation {
                        generation: *generation,
                        event: record.event,
                        wave: record.wave,
                        reason: "duplicate or unexpected split-barrier phase signal".into(),
                    });
                }
                barriers
                    .entry((*barrier, *generation))
                    .or_default()
                    .insert(wave, clocks[wave].clone());
                execution_signals.insert(
                    (*barrier, *generation, wave),
                    execution_clocks[wave].clone(),
                );
                if let Some(clocks) = &arrival_clocks {
                    arrival_signals.insert((*barrier, *generation, wave), clocks[wave].clone());
                }
                signals[wave] = Some((*barrier, *generation));
            }
            AmdEventKind::SplitBarrierWait {
                barrier,
                generation,
            } => {
                if signals[wave] != Some((*barrier, *generation)) || blocked[wave].is_some() {
                    barrier_violations.push(AmdBarrierViolation {
                        generation: *generation,
                        event: record.event,
                        wave: record.wave,
                        reason: "wait lacks its own signal or repeats a blocked phase".into(),
                    });
                }
                blocked[wave] = Some((*barrier, *generation, graph.event(record.event).pc));
            }
            AmdEventKind::SplitBarrierRelease {
                barrier,
                generation,
            } => {
                let phase = barriers.get(&(*barrier, *generation));
                let exited = |w: usize| {
                    (w * WAVE..((w + 1) * WAVE).min(graph.threads() as usize)).all(|t| {
                        graph
                            .program_order(ThreadIx(t as u32))
                            .last()
                            .is_some_and(|&event| {
                                event < record.event && graph.event(event).kind == EventKind::Exit
                            })
                    })
                };
                let missing =
                    (0..waves).any(|w| !exited(w) && phase.is_none_or(|p| !p.contains_key(&w)));
                if missing
                    || blocked[wave] != Some((*barrier, *generation, graph.event(record.event).pc))
                    || !released.insert((*barrier, *generation, wave))
                {
                    barrier_violations.push(AmdBarrierViolation { generation: *generation, event: record.event, wave: record.wave, reason: "split-barrier release lacks matching wait, source point or live-wave arrivals".into() });
                } else {
                    let phase = phase.unwrap();
                    let adds_drop_memory = (0..waves).any(|w| {
                        exited(w)
                            && clocks[w].iter().enumerate().any(|(slot, value)| {
                                *value > phase.get(&w).map_or(0, |clock| clock[slot])
                            })
                    });
                    if arrival_clocks.is_none() && adds_drop_memory {
                        arrival_clocks = Some(clocks.clone());
                    }
                    for (&participant, clock) in phase {
                        join(&mut clocks[wave], clock);
                        join(
                            &mut execution_clocks[wave],
                            &execution_signals[&(*barrier, *generation, participant)],
                        );
                        if let Some(clocks) = &mut arrival_clocks {
                            join(
                                &mut clocks[wave],
                                arrival_signals
                                    .get(&(*barrier, *generation, participant))
                                    .unwrap_or(clock),
                            );
                        }
                    }
                    for w in 0..waves {
                        if exited(w) {
                            let frontier = clocks[w].clone();
                            join(&mut clocks[wave], &frontier);
                            let frontier = execution_clocks[w].clone();
                            join(&mut execution_clocks[wave], &frontier);
                        }
                    }
                    completed_waits
                        .entry((*barrier, *generation))
                        .or_default()
                        .push((wave, execution_clocks[wave][wave]));
                    blocked[wave] = None;
                    signals[wave] = None;
                    next_generation[wave] += 1;
                }
            }
            AmdEventKind::NamedBarrierInit { .. }
            | AmdEventKind::NamedBarrierJoin { .. }
            | AmdEventKind::NamedBarrierSignal { .. }
            | AmdEventKind::NamedBarrierSignalWithCount { .. }
            | AmdEventKind::NamedBarrierWait { .. }
            | AmdEventKind::NamedBarrierRelease { .. }
            | AmdEventKind::NamedBarrierLeave { .. } => {
                let arrivals = arrival_clocks.as_ref().map_or(&clocks[wave], |a| &a[wave]);
                if let Some(release) = named.event(
                    record,
                    graph.event(record.event).pc,
                    &mut execution_clocks,
                    &clocks[wave],
                    arrivals,
                ) {
                    if release.drops
                        && release.memory != release.arrivals
                        && arrival_clocks.is_none()
                    {
                        arrival_clocks = Some(clocks.clone());
                    }
                    clocks[wave] = release.memory;
                    if let Some(clocks) = &mut arrival_clocks {
                        clocks[wave] = release.arrivals;
                    }
                }
            }
            AmdEventKind::WmmaIssue {
                kind,
                destinations,
                sources_ab,
                accumulator,
                index,
                ..
            } => {
                matrices[wave].push(WmmaOperation {
                    event: record.event,
                    ordinal: valu_ordinal[wave] - 1,
                    kind: *kind,
                    destinations: destinations.clone(),
                    sources: sources_ab
                        .iter()
                        .chain(accumulator)
                        .chain(index)
                        .copied()
                        .collect(),
                });
            }
            // Default scheduling interlocks ordinary ALU dependencies. LLVM's
            // AMDGPUInsertDelayAlu pass inserts S_DELAY_ALU to avoid stalls, not
            // to turn outstanding memory into completed data. Expert mode is
            // outside this execution envelope.
            AmdEventKind::DelayAlu { .. }
            | AmdEventKind::Delay { .. }
            | AmdEventKind::AluWait { .. }
            | AmdEventKind::PrefetchFootprint { .. }
            | AmdEventKind::GlobalAccessFault { .. }
            | AmdEventKind::ModeAccess { .. }
            | AmdEventKind::BufferRangeCheck { .. }
            | AmdEventKind::TensorAccessFault { .. } => {}
            _ => unreachable!("trace contract rejected legacy target events"),
        }
        pending[wave].retain(|&index| {
            if complete(&operations[index]) {
                let op = &operations[index];
                // Scalar completions only occur at KM0, so no out-of-order
                // completion is promoted into a false scalar FIFO prefix.
                let slot = wave * CLASSES + class_index(op.class);
                clocks[wave][slot] = clocks[wave][slot].max(op.sequence);
                if let Some(clocks) = &mut arrival_clocks {
                    clocks[wave][slot] = clocks[wave][slot].max(op.sequence);
                }
                false
            } else {
                true
            }
        });
    }
    if matches!(graph.terminal(), Terminal::Completed) {
        for (wave, phase) in blocked.iter().enumerate() {
            if let Some((_, generation, _)) = phase {
                barrier_violations.push(AmdBarrierViolation {
                    generation: *generation,
                    event: graph.amd_events().last().unwrap().event,
                    wave: wave as u32,
                    reason: "completed trace leaves a split-barrier wait blocked".into(),
                });
            }
        }
    }
    let wait_readiness = PropertyResult::from_findings(
        AmdWaitEvidence {
            issues: operations.len(),
            waits,
        },
        hazards,
        unresolved,
    );
    named.finish(graph);
    barrier_violations.extend(named.violations);
    barrier_unresolved.extend(named.unresolved);
    let barrier = PropertyResult::from_findings(
        AmdBarrierEvidence {
            generations: barriers.len() + named.phases,
            arrivals: arrivals + named.signals,
            releases: released.len() + named.releases,
        },
        barrier_violations,
        barrier_unresolved,
    );
    let memory_conflict = if barrier.verdict() == Verdict::Pass
        && wait_readiness.verdict() != Verdict::Unknown
    {
        let candidate = memory_conflicts(graph, &operations, &issue_index);
        if arrival_clocks.is_some() && candidate.verdict() == Verdict::Pass {
            match memory_conflicts_with_clock(graph, &operations, &issue_index, true) {
                PropertyResult::Pass(_) => candidate,
                PropertyResult::Fail(conflicts) => {
                    let reason = "gfx1250 memory ordering depends on an implicit barrier drop or named-barrier drop; drop publication is not pinned";
                    PropertyResult::Unknown(
                        conflicts
                            .iter()
                            .map(|conflict| {
                                Unresolved::between(conflict.earlier, conflict.later, reason)
                            })
                            .collect(),
                    )
                }
                PropertyResult::Unknown(reasons) => PropertyResult::Unknown(reasons),
            }
        } else {
            candidate
        }
    } else {
        unknown("gfx1250 barrier or completion contract is inconclusive")
    };
    AmdReport {
        trace,
        wait_readiness,
        barrier,
        memory_conflict,
    }
}

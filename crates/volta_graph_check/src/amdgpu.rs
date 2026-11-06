//! Target-selected gfx942/gfx950 checks over the existing event graph, independent of PTX certificates.
//!
//! Common semantics: AMD MI300/CDNA3 and MI350/CDNA4 ISA §4.3–4.4
//! (workgroups and counters), §8.3 (SMEM),
//! §10.2 (FLAT). Ordinary VM loads/stores and homogeneous LDS instructions complete in order;
//! mixed LGKM classes, SMEM and FLAT do not. A nonzero LGKM wait can still certify
//! a prefix of the ordered LDS stream from that stream's own outstanding count.
//! Unordered SMEM and FLAT need zero; FLAT requires both counters to retire.
//! S_BARRIER joins *completed* memory effects of participating waves; it does not
//! manufacture a wait for outstanding instructions. EXEC masks are full 64-bit masks.

use std::collections::{HashMap, HashSet};

use volta_graph::{
    AmdArchitecture, AmdCounter, AmdEventKind, AmdMatrixKind, AmdMemoryClass, AmdMemoryScope,
    AmdRegister, AmdRegisterFile, EventGraph, EventId, EventKind, Footprint, OrderingKind, Proxy,
    Space, Terminal,
};

use crate::domain::{ByteLocation, ConflictKind, bytes, written_bytes};
use crate::memory_conflict::{
    CONFLICT_CAP, MemoryConflict, MemoryConflictEvidence, MemoryConflictResult,
};
use crate::{PropertyResult, Unresolved, Verdict};
mod buffer_lds;
pub mod early_memory_conflicts;
pub mod early_readiness;
mod gfx1250;
pub mod global_bounds;
pub mod mode_order;
mod private_memory;
pub mod tensor_bounds;
pub mod wmma_hazards;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdTraceEvidence {
    pub waves: usize,
    pub instructions: usize,
    pub issues: usize,
    pub effects: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdWaitEvidence {
    pub issues: usize,
    pub waits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmdRegisterHazardKind {
    ReadBeforeReady,
    DestinationReuse,
    SourceReuse,
    MatrixReadBeforeReady,
    MatrixDestinationReuse,
    MatrixOperandBeforeReady,
    MatrixSourceReuse,
    ValuReadBeforeReady,
    ValuDestinationBeforeReady,
    ValuScalarReadBeforeReady,
    /// The atomic's implicit EXEC is unsafe while an older non-RMW translation is pending.
    AtomicReplayBeforeReady,
    /// A gfx950 LDS DMA reads M0 before one independent state after a SALU write.
    SaluM0ReadBeforeReady,
    /// A selected gfx950 literal READLANE consumes a preceding VALU VGPR too soon.
    ValuLaneReadBeforeReady,
    /// A selected gfx950 lane move follows a VALU EXEC write too soon.
    ValuLaneExecBeforeReady,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdRegisterHazard {
    pub issue: EventId,
    pub use_event: EventId,
    pub wave: u32,
    pub register: AmdRegister,
    pub kind: AmdRegisterHazardKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdBarrierEvidence {
    pub generations: usize,
    pub arrivals: usize,
    pub releases: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdBarrierViolation {
    pub generation: u32,
    pub event: EventId,
    pub wave: u32,
    pub reason: String,
}

pub type AmdTraceResult = PropertyResult<AmdTraceEvidence, Unresolved>;
pub type AmdWaitResult = PropertyResult<AmdWaitEvidence, AmdRegisterHazard>;
pub type AmdBarrierResult = PropertyResult<AmdBarrierEvidence, AmdBarrierViolation>;

#[derive(Debug, Clone)]
pub struct AmdReport {
    pub trace: AmdTraceResult,
    pub wait_readiness: AmdWaitResult,
    pub barrier: AmdBarrierResult,
    pub memory_conflict: MemoryConflictResult,
}

const CLASSES: usize = 25;
fn class_index(class: AmdMemoryClass) -> usize {
    // CDNA4 §4.4 orders same-type LDS instructions. §12.12's DS_(B)PERMUTE
    // uses LDS hardware, with no LDS memory footprint. LLVM AMDGPUHWEvents.cpp
    // getEventsForImpl at846473237377990d00b9c353f6a2c86116b52ea5 assigns both
    // ordinary non-GDS DS operations and permutations to LDS_ACCESS. They share
    // the completion FIFO, while memory effects remain explicitly separate.
    match class {
        AmdMemoryClass::VectorMemory | AmdMemoryClass::VectorStoreWide => 0,
        AmdMemoryClass::ScalarMemory => 1,
        AmdMemoryClass::Lds | AmdMemoryClass::CrossLane => 2,
        AmdMemoryClass::Flat | AmdMemoryClass::FlatStoreWide => 3,
        AmdMemoryClass::Message => 4,
        AmdMemoryClass::VectorLoad => 5,
        AmdMemoryClass::VectorStore => 6,
        AmdMemoryClass::ScalarLoad => 7,
        AmdMemoryClass::Ds => 2,
        AmdMemoryClass::AsyncLoadToLds => 8,
        AmdMemoryClass::AsyncStoreFromLds => 9,
        AmdMemoryClass::TensorLoadToLds => 10,
        AmdMemoryClass::TensorStoreFromLds => 11,
        AmdMemoryClass::GlobalAtomicAddF32NoReturn => 12,
        AmdMemoryClass::FlatGlobalLoad => 13,
        AmdMemoryClass::FlatGlobalStore => 14,
        AmdMemoryClass::FlatLdsLoad => 15,
        AmdMemoryClass::FlatLdsStore => 16,
        // DMA shares VM accounting with ordinary VMEM, but has no proven FIFO
        // relation with it or with other DMA issues. It must not use slot0.
        AmdMemoryClass::BufferLoadToLds32 | AmdMemoryClass::BufferLoadToLds128 => 17,
        // Distinct unordered domains deliberately admit private retirement only
        // at VM0, without changing ordinary VM or DMA FIFO premises.
        AmdMemoryClass::ScratchLoad32 => 18,
        AmdMemoryClass::ScratchStore32 => 19,
        AmdMemoryClass::ScratchLoadWide => 20,
        AmdMemoryClass::ScratchStoreWide => 21,
        AmdMemoryClass::FlatPrivateLoad => 22,
        AmdMemoryClass::FlatPrivateStore => 23,
        // Prefetches have no data completion or effects to order.
        AmdMemoryClass::Prefetch => 24,
    }
}
fn ordered(class: AmdMemoryClass) -> bool {
    matches!(
        class,
        AmdMemoryClass::VectorMemory
            | AmdMemoryClass::VectorStoreWide
            | AmdMemoryClass::Lds
            | AmdMemoryClass::CrossLane
    )
}
fn join(to: &mut [u32], from: &[u32]) {
    for (a, b) in to.iter_mut().zip(from) {
        *a = (*a).max(*b);
    }
}

#[derive(Clone)]
struct Operation {
    event: EventId,
    wave: u32,
    class: AmdMemoryClass,
    sequence: u32,
    issued_tick: u64,
    remaining: Vec<(AmdCounter, u16)>,
    destinations: Vec<AmdRegister>,
    held_sources: Vec<AmdRegister>,
    /// Completed-effect frontiers at issue, not the candidate trace's earlier effects.
    clock: Vec<u32>,
    /// Present only after a gfx1250 implicit drop adds an unpinned memory edge.
    /// This frontier retains explicit arrivals and excludes those drop edges.
    arrival_clock: Option<Vec<u32>>,
}

fn unknown<E, V>(reason: &str) -> PropertyResult<E, V> {
    PropertyResult::Unknown(vec![Unresolved::global(reason)])
}

fn trace_contract(
    graph: &EventGraph,
    architecture: AmdArchitecture,
    prefix: bool,
) -> AmdTraceResult {
    let mut unresolved = Vec::new();
    if graph
        .provenance()
        .assumptions
        .iter()
        .any(|a| matches!(a, volta_graph::Assumption::AllocatedNamedBarriers(_)))
    {
        unresolved.push(Unresolved::global(
            "shader named barrier allocation requires gfx1250",
        ));
    }
    if let Err(defects) = graph.validate() {
        unresolved.extend(defects.into_iter().map(|d| {
            Unresolved {
                events: d
                    .event
                    .filter(|event| graph.events().get(*event).is_some())
                    .into_iter()
                    .collect(),
                reason: d.message,
            }
        }));
        return PropertyResult::Unknown(unresolved);
    }
    let layout = graph.shared_allocation_layout();
    unresolved.extend(private_memory::validate(graph, architecture));
    if graph
        .provenance()
        .assumes(volta_graph::Assumption::ConfiguredSharedAllocationLayout)
        != !layout.is_empty()
    {
        unresolved.push(Unresolved::global(
            "configured LDS layout and its provenance premise disagree",
        ));
    }
    if let Err(reason) = volta_graph::validate_shared_allocations(layout) {
        return unknown(&reason);
    }
    let waves = (graph.threads() as usize).div_ceil(64);
    let mut seen = vec![false; waves];
    let mut last_instruction = HashMap::<u32, InstructionRecord>::new();
    let mut issues = 0;
    let mut instructions = 0;
    let mut issue_events = HashMap::new();
    let mut dma = buffer_lds::Trace::new(architecture);
    for record in graph.amd_events() {
        if record.wave as usize >= waves {
            continue;
        }
        let valid_lanes = (graph.threads() - record.wave * 64).min(64);
        let mask = if valid_lanes == 64 {
            u64::MAX
        } else {
            (1u64 << valid_lanes) - 1
        };
        let check_regs = |regs: &[AmdRegister], unresolved: &mut Vec<Unresolved>| {
            for r in regs {
                if r.file == AmdRegisterFile::Temporary
                    || r.index >= 256
                    || (r.file == AmdRegisterFile::Scalar
                        && r.index > 101
                        && !matches!(r.index, 106 | 107 | 126 | 127)
                        && !(architecture == AmdArchitecture::Gfx950 && r.index == 124))
                    || (r.file != AmdRegisterFile::Scalar && (r.lanes == 0 || r.lanes & !mask != 0))
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "invalid AMD register index or physical lane mask",
                    ));
                }
            }
        };
        match &record.kind {
            AmdEventKind::Instruction {
                exec_mask,
                reads,
                writes,
                scalar_mask_reads,
            } => {
                instructions += 1;
                seen[record.wave as usize] = true;
                if exec_mask & !mask != 0 {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "EXEC includes a nonexistent lane",
                    ));
                }
                check_regs(reads, &mut unresolved);
                check_regs(writes, &mut unresolved);
                if scalar_mask_reads.iter().any(|r| {
                    r.file != AmdRegisterFile::Scalar
                        || matches!(r.index, 126 | 127)
                        || !reads.contains(r)
                }) {
                    unresolved.push(Unresolved::at(record.event,
                        "scalar carry/condition roles must be scalar-register reads of this instruction"));
                }
                if let Some(previous) = last_instruction.get(&record.wave) {
                    check_missing_semantic(previous, &mut unresolved);
                }
                let pc = graph.event(record.event).pc;
                let kind = pc.map(|p| graph.provenance().instructions[p.0 as usize].kind);
                dma.instruction(record, kind, *exec_mask, reads, writes, &mut unresolved);
                if !scalar_mask_reads.is_empty() && kind != Some("vector ALU") {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "carry/condition roles require a supported vector ALU instruction",
                    ));
                }
                let expected = match kind {
                    Some("scalar memory") => Some("memory"),
                    Some("vector memory" | "LDS" | "scratch load" | "scratch store")
                        if *exec_mask != 0 =>
                    {
                        Some("memory")
                    }
                    Some("readlane" | "writelane") => Some("lane"),
                    Some("wait" | "s_endpgm") => Some("wait"),
                    Some("barrier") => Some("barrier"),
                    Some("MFMA") => Some("matrix"),
                    Some("s_nop") => Some("delay"),
                    _ => None,
                };
                last_instruction.insert(
                    record.wave,
                    InstructionRecord {
                        event: record.event,
                        pc,
                        exec_mask: *exec_mask,
                        reads: reads.clone(),
                        writes: writes.clone(),
                        expected,
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
                issues += 1;
                check_regs(destinations, &mut unresolved);
                check_regs(held_sources, &mut unresolved);
                if let Some(instruction) = claim_instruction(
                    graph,
                    record,
                    &mut last_instruction,
                    "memory",
                    &mut unresolved,
                ) {
                    dma.issue(
                        graph,
                        record,
                        instruction,
                        *class,
                        destinations,
                        held_sources,
                        &mut unresolved,
                    );
                    if instruction.writes.len() != destinations.len()
                        || !instruction.writes.iter().all(|r| destinations.contains(r))
                    {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "memory destinations do not match the instruction's register writes",
                        ));
                    }
                    if !held_sources.iter().all(|r| instruction.reads.contains(r)) {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "held memory sources are absent from the instruction's register reads",
                        ));
                    }
                }
                let valid = match class {
                    AmdMemoryClass::VectorMemory | AmdMemoryClass::VectorStoreWide => {
                        counters == &[(AmdCounter::Vm, 1)]
                    }
                    AmdMemoryClass::BufferLoadToLds32 | AmdMemoryClass::BufferLoadToLds128 => {
                        architecture == AmdArchitecture::Gfx950
                            && counters == &[(AmdCounter::Vm, 1)]
                    }
                    AmdMemoryClass::ScratchLoad32
                    | AmdMemoryClass::ScratchStore32
                    | AmdMemoryClass::ScratchLoadWide
                    | AmdMemoryClass::ScratchStoreWide => {
                        architecture == AmdArchitecture::Gfx950
                            && counters == &[(AmdCounter::Vm, 1)]
                    }
                    AmdMemoryClass::ScalarMemory => {
                        counters == &[(AmdCounter::Lgkm, 1)] || counters == &[(AmdCounter::Lgkm, 2)]
                    }
                    AmdMemoryClass::Lds | AmdMemoryClass::Message | AmdMemoryClass::CrossLane => {
                        counters == &[(AmdCounter::Lgkm, 1)]
                    }
                    AmdMemoryClass::Flat
                    | AmdMemoryClass::FlatStoreWide
                    | AmdMemoryClass::FlatPrivateLoad
                    | AmdMemoryClass::FlatPrivateStore => {
                        counters.len() == 2
                            && counters.contains(&(AmdCounter::Vm, 1))
                            && counters.contains(&(AmdCounter::Lgkm, 1))
                    }
                    _ => false,
                };
                if !valid {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "counter charges do not match the native AMD instruction class",
                    ));
                }
                if matches!(
                    class,
                    AmdMemoryClass::VectorStoreWide | AmdMemoryClass::FlatStoreWide
                ) && (!destinations.is_empty()
                    || !matches!(held_sources.len(), 3 | 4)
                    || held_sources.iter().any(|r| {
                        !matches!(
                            r.file,
                            AmdRegisterFile::Vector | AmdRegisterFile::Accumulator
                        )
                    })
                    || !held_sources.windows(2).all(|r| {
                        r[0].file == r[1].file
                            && r[0].lanes == r[1].lanes
                            && r[0].index + 1 == r[1].index
                    }))
                {
                    unresolved.push(Unresolved::at(record.event,"wide store needs its contiguous three/four data registers as source-timing operands"));
                }
                issue_events.insert(record.event, (record.wave, *class));
            }
            AmdEventKind::Wait {
                vmcnt,
                lgkmcnt,
                expcnt,
            } => {
                let instruction = claim_instruction(
                    graph,
                    record,
                    &mut last_instruction,
                    "wait",
                    &mut unresolved,
                );
                if instruction.is_some_and(|i| {
                    i.pc.is_some_and(|pc| {
                        graph.provenance().instructions[pc.0 as usize].kind == "s_endpgm"
                    })
                }) && (*vmcnt != Some(0) || *lgkmcnt != Some(0) || *expcnt != Some(0))
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "S_ENDPGM must record its implicit all-counter zero wait",
                    ));
                }
                if vmcnt.is_some_and(|n| n > 63)
                    || lgkmcnt.is_some_and(|n| n > 15)
                    || expcnt.is_some_and(|n| n > 7)
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "wait threshold exceeds the native AMD counter width",
                    ));
                }
            }
            AmdEventKind::LaneRead { .. } | AmdEventKind::LaneWrite { .. } => {
                claim_instruction(
                    graph,
                    record,
                    &mut last_instruction,
                    "lane",
                    &mut unresolved,
                );
            }
            AmdEventKind::BarrierArrive { exec_mask, .. } => {
                if exec_mask & !mask != 0 {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "barrier EXEC includes a nonexistent lane",
                    ));
                }
                claim_instruction(
                    graph,
                    record,
                    &mut last_instruction,
                    "barrier",
                    &mut unresolved,
                );
            }
            AmdEventKind::MatrixIssue {
                kind,
                destinations,
                sources_ab,
                accumulator,
            } => {
                if !architecture.supports_matrix(*kind) {
                    unresolved.push(Unresolved::at(
                        record.event,
                        format!(
                            "MFMA form {kind:?} is outside the {} graph contract",
                            architecture.name()
                        ),
                    ));
                }
                for regs in [destinations, sources_ab, accumulator] {
                    check_regs(regs, &mut unresolved);
                }
                if let Some(instruction) = claim_instruction(
                    graph,
                    record,
                    &mut last_instruction,
                    "matrix",
                    &mut unresolved,
                ) {
                    if mask != u64::MAX || instruction.exec_mask != u64::MAX {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "MFMA requires a complete wave64 in this model",
                        ));
                    }
                    if ![126, 127].iter().all(|&index| {
                        instruction
                            .reads
                            .iter()
                            .any(|r| r.file == AmdRegisterFile::Scalar && r.index == index)
                    }) {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "MFMA instruction must record its implicit EXEC low/high reads",
                        ));
                    }
                    if instruction.writes.len() != destinations.len()
                        || !instruction.writes.iter().all(|r| destinations.contains(r))
                        || !sources_ab
                            .iter()
                            .chain(accumulator.iter())
                            .all(|r| instruction.reads.contains(r))
                    {
                        unresolved.push(Unresolved::at(
                            record.event,
                            "MFMA operand roles do not match instruction register records",
                        ));
                    }
                }
                let contiguous = |registers: &[AmdRegister]| {
                    registers
                        .windows(2)
                        .all(|r| r[0].file == r[1].file && r[0].index + 1 == r[1].index)
                };
                let inputs = kind.input_registers();
                if destinations.len() != kind.destination_registers()
                    || sources_ab.len() != 2 * inputs + kind.scale_registers()
                    || (!accumulator.is_empty()
                        && accumulator.len() != kind.destination_registers())
                    || sources_ab.get(..inputs).is_some_and(|r| !contiguous(r))
                    || sources_ab
                        .get(inputs..2 * inputs)
                        .is_some_and(|r| !contiguous(r))
                    || destinations
                        .iter()
                        .chain(sources_ab)
                        .chain(accumulator)
                        .any(|r| {
                            !matches!(
                                r.file,
                                AmdRegisterFile::Vector | AmdRegisterFile::Accumulator
                            ) || r.lanes != u64::MAX
                        })
                    || destinations.windows(2).any(|r| r[0].file != r[1].file)
                    || accumulator
                        .iter()
                        .any(|r| destinations.first().is_some_and(|d| d.file != r.file))
                    || destinations.first().is_some_and(|r| r.index % 2 != 0)
                    || accumulator.first().is_some_and(|r| r.index % 2 != 0)
                    || !destinations
                        .windows(2)
                        .all(|r| r[1].index == r[0].index + 1)
                    || !accumulator.windows(2).all(|r| r[1].index == r[0].index + 1)
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "unsupported register layout for the selected MFMA form",
                    ));
                }
            }
            AmdEventKind::Delay { cycles } => {
                let instruction = claim_instruction(
                    graph,
                    record,
                    &mut last_instruction,
                    "delay",
                    &mut unresolved,
                );
                if !(1..=16).contains(cycles)
                    || !instruction.is_some_and(|i| {
                        i.pc.is_some_and(|pc| {
                            graph.provenance().instructions[pc.0 as usize].kind == "s_nop"
                        })
                    })
                {
                    unresolved.push(Unresolved::at(
                        record.event,
                        "delay must be a pinned S_NOP instruction with 1..16 wait states",
                    ));
                }
            }
            AmdEventKind::WmmaIssue { .. }
            | AmdEventKind::DelayAlu { .. }
            | AmdEventKind::WaitCounters { .. }
            | AmdEventKind::SplitBarrierSignal { .. }
            | AmdEventKind::SplitBarrierWait { .. }
            | AmdEventKind::SplitBarrierRelease { .. }
            | AmdEventKind::NamedBarrierInit { .. }
            | AmdEventKind::NamedBarrierJoin { .. }
            | AmdEventKind::NamedBarrierSignal { .. }
            | AmdEventKind::NamedBarrierWait { .. }
            | AmdEventKind::NamedBarrierRelease { .. }
            | AmdEventKind::NamedBarrierLeave { .. }
            | AmdEventKind::NamedBarrierSignalWithCount { .. }
            | AmdEventKind::AluWait { .. }
            | AmdEventKind::ModeAccess { .. }
            | AmdEventKind::BufferRangeCheck { .. } => unresolved.push(Unresolved::at(
                record.event,
                "GFX12 event cannot use a legacy CDNA target contract",
            )),
            _ => {}
        }
    }
    for instruction in last_instruction.values() {
        check_missing_semantic(instruction, &mut unresolved);
    }
    if waves == 0 || (!prefix && seen.iter().any(|v| !v)) {
        unresolved.push(Unresolved::global(
            "each launched wave requires AMD instruction/register records",
        ));
    }
    let mut linked = HashSet::new();
    let mut linked_issues = HashSet::new();
    let mut effect_issues = HashMap::new();
    let mut shared_footprints = HashMap::<(EventId, volta_graph::ThreadIx), (u64, u64)>::new();
    for link in graph.amd_memory_effects() {
        dma.effect(link);
        linked.insert(link.effect);
        linked_issues.insert(link.issue);
        effect_issues.insert(link.effect, link.issue);
        if let Some((wave, class)) = issue_events.get(&link.issue)
            && let Some(event) = graph.events().get(link.effect)
            && let EventKind::Access(a) = event.kind
        {
            let access = graph.access(a);
            if !layout.is_empty()
                && access.space == Space::Shared
                && let Footprint::Bytes { addr, width } = access.footprint
                && let Some(end) = addr.checked_add(u64::from(width))
            {
                shared_footprints
                    .entry((link.issue, access.attributed))
                    .and_modify(|(lo, hi)| {
                        *lo = (*lo).min(addr);
                        *hi = (*hi).max(end);
                    })
                    .or_insert((addr, end));
            }
            if access.attributed.0 / 64 != *wave {
                unresolved.push(Unresolved::at(
                    link.effect,
                    "AMD effect lane belongs to another wave",
                ));
            }
            if graph.event(link.issue).pc != event.pc {
                unresolved.push(Unresolved::at(
                    link.effect,
                    "AMD effect source point differs from its issue",
                ));
            }
            let correct_space = match class {
                AmdMemoryClass::Lds => access.space == Space::Shared,
                AmdMemoryClass::BufferLoadToLds32 | AmdMemoryClass::BufferLoadToLds128 => {
                    matches!(
                        (access.space, access.mode),
                        (Space::Global, volta_graph::AccessMode::Read)
                            | (Space::Shared, volta_graph::AccessMode::Write)
                    )
                }
                AmdMemoryClass::VectorMemory
                | AmdMemoryClass::VectorStoreWide
                | AmdMemoryClass::ScalarMemory => access.space == Space::Global,
                AmdMemoryClass::Flat | AmdMemoryClass::FlatStoreWide => {
                    matches!(access.space, Space::Global | Space::Shared)
                }
                AmdMemoryClass::Message | AmdMemoryClass::CrossLane => false,
                _ => false,
            };
            if !correct_space {
                unresolved.push(Unresolved::at(
                    link.effect,
                    "AMD instruction class does not match its memory space",
                ));
            }
            if link.scope == AmdMemoryScope::Workgroup
                && !(access.space == Space::Global
                    && matches!(
                        class,
                        AmdMemoryClass::VectorMemory | AmdMemoryClass::VectorStoreWide
                    ))
            {
                unresolved.push(Unresolved::at(
                    link.effect,
                    "workgroup scope is supported only for native GLOBAL vector effects",
                ));
            }
            if link.scope == AmdMemoryScope::Gfx1250System {
                unresolved.push(Unresolved::at(
                    link.effect,
                    "gfx1250 system scope cannot certify a legacy AMD target",
                ));
            }
        }
    }
    dma.finish(graph, &mut unresolved);
    for ((issue, _), (start, end)) in shared_footprints {
        if !layout
            .iter()
            .any(|region| start >= region.offset && end <= region.offset + region.size)
        {
            unresolved.push(Unresolved::at(
                issue,
                "whole LDS transfer straddles or escapes the configured shared allocation layout",
            ));
        }
    }
    for (&event, (_, class)) in &issue_events {
        if !matches!(
            class,
            AmdMemoryClass::Message
                | AmdMemoryClass::CrossLane
                | AmdMemoryClass::ScratchLoad32
                | AmdMemoryClass::ScratchStore32
                | AmdMemoryClass::ScratchLoadWide
                | AmdMemoryClass::ScratchStoreWide
                | AmdMemoryClass::FlatPrivateLoad
                | AmdMemoryClass::FlatPrivateStore
        ) && !linked_issues.contains(&event)
        {
            unresolved.push(Unresolved::at(
                event,
                "AMD memory instruction has no recorded effects",
            ));
        }
    }
    let mut last_wave_record = HashMap::new();
    for (id, event) in graph.events().iter() {
        match event.kind {
            EventKind::Amd(index) => {
                let record = &graph.amd_events()[index as usize];
                last_wave_record.insert(record.wave, id);
            }
            EventKind::Access(a) => {
                let a = graph.access(a);
                if !linked.contains(&id) {
                    unresolved.push(Unresolved::at(
                        id,
                        "memory effect lacks its AMD issue association",
                    ));
                }
                if effect_issues.get(&id).is_some_and(|issue| {
                    last_wave_record.get(&(a.attributed.0 / 64)) != Some(issue)
                }) {
                    unresolved.push(Unresolved::at(
                        id,
                        "AMD memory effect is recorded after the wave progressed beyond its issue",
                    ));
                }
                if a.ordering != OrderingKind::Ordinary
                    || a.proxy != Proxy::Generic
                    || a.space == Space::Tensor
                    || !matches!(a.footprint, Footprint::Bytes { .. })
                {
                    unresolved.push(Unresolved::at(
                        id,
                        "memory effect is outside the AMD ordinary byte-memory contract",
                    ));
                }
                if matches!(a.footprint,Footprint::Bytes{addr,width} if addr.checked_add(u64::from(width)).is_none())
                {
                    unresolved.push(Unresolved::at(
                        id,
                        "AMD footprint end exceeds the representable byte-iterator range",
                    ));
                }
            }
            EventKind::Exit => {}
            _ => unresolved.push(Unresolved::at(
                id,
                "unrecognized or PTX-only event in native AMD trace",
            )),
        }
    }
    PropertyResult::from_findings(
        AmdTraceEvidence {
            waves,
            instructions,
            issues,
            effects: graph.amd_memory_effects().len(),
        },
        Vec::new(),
        unresolved,
    )
}

struct InstructionRecord {
    event: EventId,
    pc: Option<volta_graph::ProgramPoint>,
    exec_mask: u64,
    reads: Vec<AmdRegister>,
    writes: Vec<AmdRegister>,
    expected: Option<&'static str>,
    semantic: bool,
}
fn check_missing_semantic(instruction: &InstructionRecord, unresolved: &mut Vec<Unresolved>) {
    if instruction.expected.is_some() && !instruction.semantic {
        unresolved.push(Unresolved::at(
            instruction.event,
            "instruction is missing its required AMD semantic event",
        ));
    }
}
fn claim_instruction<'a>(
    graph: &EventGraph,
    record: &volta_graph::AmdEvent,
    instructions: &'a mut HashMap<u32, InstructionRecord>,
    kind: &str,
    unresolved: &mut Vec<Unresolved>,
) -> Option<&'a InstructionRecord> {
    let Some(instruction) = instructions.get_mut(&record.wave) else {
        unresolved.push(Unresolved::at(
            record.event,
            "AMD semantic event lacks its instruction/register record",
        ));
        return None;
    };
    if instruction.pc != graph.event(record.event).pc
        || instruction.semantic
        || instruction
            .expected
            .is_some_and(|expected| expected != kind)
    {
        unresolved.push(Unresolved::at(
            record.event,
            "AMD semantic event has a duplicate, mismatched, or missing instruction record",
        ));
        return None;
    }
    instruction.semantic = true;
    Some(instruction)
}

#[derive(Clone)]
struct MatrixOperation {
    event: EventId,
    tick: u64,
    kind: AmdMatrixKind,
    destinations: Vec<AmdRegister>,
    accumulator: Vec<AmdRegister>,
}
impl MatrixOperation {
    fn ordinary_wait(&self, architecture: AmdArchitecture) -> u64 {
        // LLVM 22.1.0 GCNHazardRecognizer.cpp at reachable release commit
        // 4434dabb69916856b824f68a64b029c67175e532 (source SHA in
        // docs/gfx942-graph-contract.md), GFX940_XDL_N_PassWriteVgprVALUMemExpReadWaitStates and VALUWaw:
        // CDNA3 XDL uses passes+3; CDNA4 adds a state for >2-pass forms.
        // The modeled legacy SGEMM forms use passes+2 on both targets.
        self.kind.passes()
            + if self.kind.is_xdl() {
                3 + u64::from(architecture == AmdArchitecture::Gfx950 && self.kind.passes() != 2)
            } else {
                2
            }
    }
}

/// Select the native contract from provenance, including for direct callers.
pub fn check_amdgpu(graph: &EventGraph) -> AmdReport {
    check_with_scope(graph, false)
}

/// A construction prefix is never full-execution or acceptance evidence.
fn check_prefix(graph: &EventGraph) -> AmdReport {
    check_with_scope(graph, true)
}

fn check_with_scope(graph: &EventGraph, prefix: bool) -> AmdReport {
    let Some(architecture) = AmdArchitecture::from_targets(&graph.provenance().targets) else {
        return AmdReport {
            trace: unknown("AMD checks require one recognized gfx942/gfx950/gfx1250 target"),
            wait_readiness: unknown("AMD target selection did not pass"),
            barrier: unknown("AMD target selection did not pass"),
            memory_conflict: unknown("AMD target selection did not pass"),
        };
    };
    // Each generation owns its completion and synchronization contract;
    // gfx1250 wave32 must never inherit legacy CDNA counter/barrier semantics.
    if architecture == AmdArchitecture::Gfx1250 {
        return if prefix {
            gfx1250::check_prefix(graph)
        } else {
            gfx1250::check(graph)
        };
    }
    let trace = trace_contract(graph, architecture, prefix);
    if trace.verdict() != Verdict::Pass {
        return AmdReport {
            trace,
            wait_readiness: unknown("AMD trace contract did not pass"),
            barrier: unknown("AMD trace contract did not pass"),
            memory_conflict: unknown("AMD trace contract did not pass"),
        };
    }
    let waves = (graph.threads() as usize).div_ceil(64);
    let mut clocks = vec![vec![0; waves * CLASSES]; waves];
    let mut sequences = vec![0; waves * CLASSES];
    let mut operations = Vec::<Operation>::new();
    let mut issue_index = HashMap::new();
    let mut pending = vec![Vec::<usize>::new(); waves];
    let mut hazards = Vec::new();
    let mut waits = 0;
    let mut barriers = HashMap::<u32, HashMap<u32, (EventId, Vec<u32>)>>::new();
    let mut releases = HashSet::new();
    let mut next_generation = vec![0u32; waves];
    let mut barrier_violations = Vec::new();
    let mut barrier_unresolved = Vec::new();
    let mut barrier_arrivals = 0;
    let mut ticks = vec![0u64; waves];
    let mut duration = vec![0u16; waves];
    let mut matrices = vec![Vec::<MatrixOperation>::new(); waves];
    let mut valu_writes = vec![Vec::<(EventId, u64, AmdRegister)>::new(); waves];
    let mut trans_writes = vec![Vec::<(EventId, u64, AmdRegister)>::new(); waves];
    let mut shifted_writes = vec![Vec::<(EventId, u64, AmdRegister)>::new(); waves];
    let mut scalar_valu_writes = vec![HashMap::<u16, (EventId, u64)>::new(); waves];
    // EXEC scheduling is a dependency on a prior VALU instruction, not on the
    // currently reaching scalar value. A later SALU EXEC restore does not erase it.
    let mut lane_exec_valu_writes = vec![[None::<(EventId, u64)>; 2]; waves];
    // New lane/private consumers require scheduling history even when a SALU
    // later changes the same scalar value. Keep old consumers' policy intact.
    let mut private_lane_valu_writes = vec![HashMap::<u16, (EventId, u64)>::new(); waves];
    let mut readlane_scalar_writes = vec![HashMap::<u16, (EventId, u64)>::new(); waves];
    let mut matrix_at = HashMap::new();
    let mut dma_at = HashSet::new();
    let mut m0_salu_writes = vec![None::<(EventId, u64)>; waves];
    let mut last_instruction = HashMap::new();
    for record in graph.amd_events() {
        if matches!(record.kind, AmdEventKind::Instruction { .. }) {
            last_instruction.insert(record.wave, record.event);
        } else if let AmdEventKind::MatrixIssue {
            kind,
            destinations,
            sources_ab,
            accumulator,
        } = &record.kind
        {
            matrix_at.insert(
                last_instruction[&record.wave],
                (*kind, destinations, sources_ab, accumulator),
            );
        } else if matches!(&record.kind, AmdEventKind::MemoryIssue { class, .. } if buffer_lds::width(*class).is_some())
        {
            dma_at.insert(last_instruction[&record.wave]);
        }
    }
    let mut blocked_at = vec![None; waves];
    for record in graph.amd_events() {
        let wave = record.wave as usize;
        if let Some(generation) = blocked_at[wave]
            && !matches!(record.kind,AmdEventKind::BarrierRelease{generation:g} if g==generation)
        {
            barrier_violations.push(AmdBarrierViolation {
                generation,
                event: record.event,
                wave: record.wave,
                reason: "wave progressed before its barrier release".into(),
            });
        }
        match &record.kind {
            AmdEventKind::Instruction {
                reads,
                writes,
                scalar_mask_reads,
                ..
            } => {
                ticks[wave] = ticks[wave].saturating_add(u64::from(duration[wave]));
                duration[wave] = 1;
                let tick = ticks[wave];
                let instruction_kind = graph
                    .event(record.event)
                    .pc
                    .map(|pc| graph.provenance().instructions[pc.0 as usize].kind)
                    .unwrap_or("");
                // LLVM GCNHazardRecognizer::checkReadM0Hazards requires one
                // independent state from the latest SALU definition to LDS DMA.
                if dma_at.contains(&record.event)
                    && let Some((issue, issued)) = m0_salu_writes[wave]
                    && tick.saturating_sub(issued).saturating_sub(1) < 1
                {
                    hazards.push(AmdRegisterHazard {
                        issue,
                        use_event: record.event,
                        wave: record.wave,
                        register: AmdRegister {
                            file: AmdRegisterFile::Scalar,
                            index: 124,
                            lanes: u64::MAX,
                        },
                        kind: AmdRegisterHazardKind::SaluM0ReadBeforeReady,
                    });
                }
                if instruction_kind == "scalar ALU"
                    && writes
                        .iter()
                        .any(|r| r.file == AmdRegisterFile::Scalar && r.index == 124)
                {
                    m0_salu_writes[wave] = Some((record.event, tick));
                }
                // CDNA4 §4.5 Table11: VALU-produced scalar data needs two
                // independent states before a VALU constant-data read, five
                // before VMEM uses it. Carry/condition inputs are interlocked.
                // VCC low/high occupy scalar106/107 in this instrumentation;
                // source aliases do not make them invisible to these checks.
                // EXEC126/127 has a separate four-state VALU→MFMA rule in
                // CDNA4 §7.6 Table38, including writes that leave EXEC unchanged.
                scalar_valu_writes[wave].retain(|_, (_, issued)| tick.saturating_sub(*issued) <= 5);
                private_lane_valu_writes[wave]
                    .retain(|_, (_, issued)| tick.saturating_sub(*issued) <= 5);
                readlane_scalar_writes[wave]
                    .retain(|_, (_, issued)| tick.saturating_sub(*issued) <= 5);
                let scalar_wait = if matches!(
                    instruction_kind,
                    "vector memory" | "scratch load" | "scratch store"
                ) {
                    5
                } else if matches!(
                    instruction_kind,
                    "vector ALU" | "trans ALU" | "shifted ALU" | "readlane" | "writelane" | "MFMA"
                ) {
                    2
                } else {
                    0
                };
                for &register in reads.iter().filter(|r| {
                    r.file == AmdRegisterFile::Scalar
                        && !scalar_mask_reads.iter().any(|mask| mask.overlaps(**r))
                }) {
                    let mut definition = scalar_valu_writes[wave].get(&register.index).copied();
                    if matches!(
                        instruction_kind,
                        "writelane" | "scratch load" | "scratch store"
                    ) {
                        definition = definition
                            .into_iter()
                            .chain(private_lane_valu_writes[wave].get(&register.index).copied())
                            .max_by_key(|(_, tick)| *tick);
                    }
                    if scalar_wait != 0 {
                        definition = definition
                            .into_iter()
                            .chain(readlane_scalar_writes[wave].get(&register.index).copied())
                            .max_by_key(|(_, tick)| *tick);
                    }
                    if let Some((issue, issued)) = definition {
                        let required = if matrix_at.contains_key(&record.event)
                            && matches!(register.index, 126 | 127)
                        {
                            4
                        } else {
                            scalar_wait
                        };
                        if tick.saturating_sub(issued).saturating_sub(1) < required {
                            hazards.push(AmdRegisterHazard {
                                issue,
                                use_event: record.event,
                                wave: record.wave,
                                register,
                                kind: AmdRegisterHazardKind::ValuScalarReadBeforeReady,
                            });
                        }
                    }
                }
                // CDNA4 Table11: literal lane moves ignore EXEC for their data
                // operand, but do not waive the explicit EXEC scheduling rule.
                if matches!(instruction_kind, "readlane" | "writelane") {
                    for index in [126, 127] {
                        if let Some(&(issue, issued)) =
                            lane_exec_valu_writes[wave][usize::from(index - 126)].as_ref()
                            && tick.saturating_sub(issued).saturating_sub(1) < 4
                        {
                            hazards.push(AmdRegisterHazard {
                                issue,
                                use_event: record.event,
                                wave: record.wave,
                                register: AmdRegister {
                                    file: AmdRegisterFile::Scalar,
                                    index,
                                    lanes: u64::MAX,
                                },
                                kind: AmdRegisterHazardKind::ValuLaneExecBeforeReady,
                            });
                        }
                    }
                    if instruction_kind == "readlane" {
                        for &(issue, issued, written) in &valu_writes[wave] {
                            if tick.saturating_sub(issued).saturating_sub(1) < 1 {
                                for &register in reads.iter().filter(|r| {
                                    r.file == AmdRegisterFile::Vector && r.overlaps(written)
                                }) {
                                    hazards.push(AmdRegisterHazard {
                                        issue,
                                        use_event: record.event,
                                        wave: record.wave,
                                        register,
                                        kind: AmdRegisterHazardKind::ValuLaneReadBeforeReady,
                                    });
                                }
                            }
                        }
                    }
                }
                trans_writes[wave].retain(|(_, written, _)| tick.saturating_sub(*written) < 2);
                shifted_writes[wave].retain(|(_, written, _)| tick.saturating_sub(*written) < 2);
                if matches!(
                    instruction_kind,
                    "vector ALU" | "shifted ALU" | "readlane" | "writelane"
                ) {
                    for (issue, _, written) in &trans_writes[wave] {
                        for &register in reads.iter().filter(|r| r.overlaps(*written)) {
                            hazards.push(AmdRegisterHazard {
                                issue: *issue,
                                use_event: record.event,
                                wave: record.wave,
                                register,
                                kind: AmdRegisterHazardKind::ValuReadBeforeReady,
                            });
                        }
                    }
                }
                if matches!(
                    instruction_kind,
                    "vector ALU" | "trans ALU" | "shifted ALU" | "readlane" | "writelane" | "MFMA"
                ) {
                    for (issue, _, written) in &shifted_writes[wave] {
                        for &register in reads.iter().filter(|r| r.overlaps(*written)) {
                            hazards.push(AmdRegisterHazard {
                                issue: *issue,
                                use_event: record.event,
                                wave: record.wave,
                                register,
                                kind: AmdRegisterHazardKind::ValuReadBeforeReady,
                            });
                        }
                        // LLVM consumesDstSelForwardingOperand also considers
                        // destination overlap: preserving/ECC paths can read
                        // the preceding shifted result through a WAW operand.
                        for &register in writes.iter().filter(|r| r.overlaps(*written)) {
                            hazards.push(AmdRegisterHazard {
                                issue: *issue,
                                use_event: record.event,
                                wave: record.wave,
                                register,
                                kind: AmdRegisterHazardKind::ValuDestinationBeforeReady,
                            });
                        }
                    }
                }
                if instruction_kind == "shifted ALU" {
                    shifted_writes[wave]
                        .extend(writes.iter().copied().map(|r| (record.event, tick, r)));
                }
                if instruction_kind == "trans ALU" {
                    trans_writes[wave]
                        .extend(writes.iter().copied().map(|r| (record.event, tick, r)));
                }
                // CDNA4 §7.6 Table 38; LLVM GCNHazardRecognizer.cpp at
                // 846473237377990d00b9c353f6a2c86116b52ea5, checkMAIHazards90A
                // and checkMAIVALUHazards. Time excludes the issuing instruction.
                matrices[wave]
                    .retain(|m| tick.saturating_sub(m.tick) <= m.ordinary_wait(architecture));
                let next_matrix = matrix_at.get(&record.event);
                for operation in &matrices[wave] {
                    let independent = tick.saturating_sub(operation.tick).saturating_sub(1);
                    for &register in reads
                        .iter()
                        .filter(|r| operation.destinations.iter().any(|d| d.overlaps(**r)))
                    {
                        let wait = if let Some((kind, _, ab, accumulator)) = next_matrix {
                            if ab.iter().any(|r| r.overlaps(register)) {
                                operation.ordinary_wait(architecture)
                            } else if accumulator.iter().any(|r| r.overlaps(register)) {
                                if (**accumulator == operation.destinations
                                    && kind.passes() == operation.kind.passes())
                                    || (!operation.kind.is_xdl() && kind.is_xdl())
                                {
                                    // Complete SrcC forwarding, including a different output range,
                                    // or a non-XDL producer feeding an XDL consumer.
                                    0
                                } else if operation.kind.is_xdl() {
                                    // Partial/different-pass SrcC forwarding. LLVM's
                                    // GFX940_XDL_N_PassWritesVGPROverlapped*SrcC
                                    // tables distinguish CDNA3 from CDNA4.
                                    operation.kind.passes()
                                        + 1
                                        + u64::from(architecture == AmdArchitecture::Gfx950)
                                } else {
                                    operation.kind.passes()
                                }
                            } else {
                                operation.ordinary_wait(architecture)
                            }
                        } else {
                            operation.ordinary_wait(architecture)
                        };
                        if independent < wait {
                            hazards.push(AmdRegisterHazard {
                                issue: operation.event,
                                use_event: record.event,
                                wave: record.wave,
                                register,
                                kind: AmdRegisterHazardKind::MatrixReadBeforeReady,
                            });
                        }
                    }
                    // MFMA destination scheduling is internally interlocked; legacy
                    // VALU and memory destinations have the explicit WAW/WAR rules.
                    if next_matrix.is_none() {
                        for &register in writes {
                            let kind =
                                if operation.destinations.iter().any(|d| d.overlaps(register))
                                    && independent < operation.ordinary_wait(architecture)
                                {
                                    Some(AmdRegisterHazardKind::MatrixDestinationReuse)
                                } else if operation.kind.is_xdl()
                                    && operation.accumulator.iter().any(|r| r.overlaps(register))
                                    && independent < operation.kind.passes() - 1
                                {
                                    Some(AmdRegisterHazardKind::MatrixSourceReuse)
                                } else {
                                    None
                                };
                            if let Some(kind) = kind {
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
                valu_writes[wave].retain(|(_, written, _)| tick.saturating_sub(*written) < 3);
                if private_memory::valu(instruction_kind) {
                    valu_writes[wave]
                        .extend(writes.iter().copied().map(|r| (record.event, tick, r)));
                }
                for &op in &pending[wave] {
                    let op = &operations[op];
                    for &r in reads {
                        if op.destinations.iter().any(|d| d.overlaps(r)) {
                            hazards.push(AmdRegisterHazard {
                                issue: op.event,
                                use_event: record.event,
                                wave: record.wave,
                                register: r,
                                kind: AmdRegisterHazardKind::ReadBeforeReady,
                            });
                        }
                    }
                    for &r in writes {
                        let kind = if op.destinations.iter().any(|d| d.overlaps(r)) {
                            Some(AmdRegisterHazardKind::DestinationReuse)
                        } else if op.held_sources.iter().any(|d| d.overlaps(r)) {
                            let wide = matches!(
                                op.class,
                                AmdMemoryClass::VectorStoreWide | AmdMemoryClass::FlatStoreWide
                            );
                            // CDNA4 §4.5 Table 11, VMEM FLAT/GLOBAL 3/4-DWORD store
                            // -> data-register VALU overwrite: two independent states.
                            // LLVM GCNHazardRecognizer.cpp (846473237377990d00b9c353f6a2c86116b52ea5)
                            // PreEmitNoopsCommon invokes checkVALUHazards only for VALU;
                            // checkSOFFSETWindowVALUHazardsHelper requires two states for
                            // gfx950 wide FLAT stores. VM/LDS load destinations return
                            // asynchronously and do not take this VALU issue-time rule.
                            let valu = matches!(
                                instruction_kind,
                                "vector ALU"
                                    | "trans ALU"
                                    | "shifted ALU"
                                    | "readlane"
                                    | "writelane"
                                    | "MFMA"
                            );
                            (!wide || (valu && tick.saturating_sub(op.issued_tick) < 3))
                                .then_some(AmdRegisterHazardKind::SourceReuse)
                        } else {
                            None
                        };
                        if let Some(kind) = kind {
                            hazards.push(AmdRegisterHazard {
                                issue: op.event,
                                use_event: record.event,
                                wave: record.wave,
                                register: r,
                                kind,
                            });
                        }
                    }
                }
                for register in writes.iter().filter(|r| r.file == AmdRegisterFile::Scalar) {
                    scalar_valu_writes[wave].remove(&register.index);
                    if private_memory::valu(instruction_kind) {
                        scalar_valu_writes[wave].insert(register.index, (record.event, tick));
                        private_lane_valu_writes[wave].insert(register.index, (record.event, tick));
                        if instruction_kind == "readlane" {
                            readlane_scalar_writes[wave]
                                .insert(register.index, (record.event, tick));
                        }
                        if matches!(register.index, 126 | 127) {
                            lane_exec_valu_writes[wave][usize::from(register.index - 126)] =
                                Some((record.event, tick));
                        }
                    }
                }
            }
            AmdEventKind::MatrixIssue {
                kind,
                destinations,
                sources_ab,
                accumulator,
            } => {
                for (issue, _, written) in &valu_writes[wave] {
                    for &register in sources_ab
                        .iter()
                        .chain(accumulator)
                        .filter(|r| r.overlaps(*written))
                    {
                        hazards.push(AmdRegisterHazard {
                            issue: *issue,
                            use_event: record.event,
                            wave: record.wave,
                            register,
                            kind: AmdRegisterHazardKind::MatrixOperandBeforeReady,
                        });
                    }
                }
                matrices[wave].push(MatrixOperation {
                    event: record.event,
                    tick: ticks[wave],
                    kind: *kind,
                    destinations: destinations.clone(),
                    accumulator: accumulator.clone(),
                });
            }
            AmdEventKind::Delay { cycles } => {
                duration[wave] = *cycles;
            }
            AmdEventKind::MemoryIssue {
                class,
                counters,
                destinations,
                held_sources,
            } => {
                let slot = wave * CLASSES + class_index(*class);
                sequences[slot] += 1;
                let mut clock = clocks[wave].clone();
                // Only FIFO classes carry earlier operations of the same class into an
                // effect. The effect's own issue does not certify its own completion.
                if ordered(*class) {
                    clock[slot] = sequences[slot] - 1;
                }
                issue_index.insert(record.event, operations.len());
                pending[wave].push(operations.len());
                operations.push(Operation {
                    event: record.event,
                    wave: record.wave,
                    class: *class,
                    sequence: sequences[slot],
                    issued_tick: ticks[wave],
                    remaining: counters.clone(),
                    destinations: destinations.clone(),
                    held_sources: held_sources.clone(),
                    clock,
                    arrival_clock: None,
                });
            }
            AmdEventKind::Wait {
                vmcnt,
                lgkmcnt,
                expcnt,
            } => {
                waits += 1;
                for (counter, threshold) in [
                    (AmdCounter::Vm, *vmcnt),
                    (AmdCounter::Lgkm, *lgkmcnt),
                    (AmdCounter::Exp, *expcnt),
                ] {
                    let Some(threshold) = threshold else { continue };
                    // CDNA4 SOPP S_WAITCNT uses 6/4/3-bit VM/LGKM/EXP fields.
                    // LLVM AMDGPUWaitcntUtils::HardwareLimits and
                    // SIGfx6CacheControl::insertWait select the all-ones field
                    // for a counter that is not being waited on. Logical issue
                    // histories may exceed that field width; do not turn this
                    // don't-care encoding into a completion proof by counting
                    // that history. Implicit hardware throttling is not modeled.
                    let maximum = match counter {
                        AmdCounter::Vm => 63,
                        AmdCounter::Lgkm => 15,
                        AmdCounter::Exp => 7,
                        _ => unreachable!("legacy Wait contains only VM/LGKM/EXP"),
                    };
                    if threshold == maximum {
                        continue;
                    }
                    let tracked: Vec<usize> = pending[wave]
                        .iter()
                        .copied()
                        .filter(|&o| operations[o].remaining.iter().any(|(c, _)| *c == counter))
                        .collect();
                    // CDNA3/CDNA4 ISA §4.4 orders LDS instructions within their
                    // own type, independently of unordered scalar/FLAT/message
                    // completions on LGKM. If the total remaining count is <= n,
                    // the LDS stream alone is also <= n. Its older FIFO prefix
                    // is therefore complete even when other types are pending.
                    // This never identifies an individual SMEM/FLAT result.
                    // LLVM SIInsertWaitcnts.cpp at
                    // 5fd50ec8e07abb94e203933017e8d6e40fb51824 documents this
                    // per-event-type improvement in its opening TODO.
                    let tracked = if counter == AmdCounter::Lgkm && threshold != 0 {
                        tracked
                            .into_iter()
                            .filter(|&o| {
                                matches!(
                                    operations[o].class,
                                    AmdMemoryClass::Lds | AmdMemoryClass::CrossLane
                                )
                            })
                            .collect::<Vec<_>>()
                    } else {
                        tracked
                    };
                    let homogeneous = tracked.first().is_some_and(|&first| {
                        ordered(operations[first].class)
                            && tracked.iter().all(|&o| {
                                class_index(operations[o].class)
                                    == class_index(operations[first].class)
                            })
                    });
                    if threshold != 0 && !homogeneous {
                        continue;
                    }
                    let Some(mut remaining) = tracked
                        .iter()
                        .flat_map(|&o| {
                            operations[o]
                                .remaining
                                .iter()
                                .filter(|(c, _)| *c == counter)
                                .map(|(_, n)| u32::from(*n))
                        })
                        .try_fold(0u32, u32::checked_add)
                    else {
                        // An unrepresentable logical history supplies no
                        // completion proof; retain every pending obligation.
                        continue;
                    };
                    for o in tracked {
                        let charge: u32 = operations[o]
                            .remaining
                            .iter()
                            .filter(|(c, _)| *c == counter)
                            .map(|(_, n)| u32::from(*n))
                            .sum();
                        if remaining.saturating_sub(charge) < u32::from(threshold) {
                            break;
                        }
                        operations[o].remaining.retain(|(c, _)| *c != counter);
                        remaining -= charge;
                    }
                }
                pending[wave].retain(|&o| {
                    if operations[o].remaining.is_empty() {
                        let op = &operations[o];
                        let slot = wave * CLASSES + class_index(op.class);
                        clocks[wave][slot] = clocks[wave][slot].max(op.sequence);
                        false
                    } else {
                        true
                    }
                });
            }
            AmdEventKind::BarrierArrive { generation, .. } => {
                barrier_arrivals += 1;
                let arrivals = barriers.entry(*generation).or_default();
                let mut reason = None;
                if *generation != next_generation[wave] {
                    reason = Some("wave arrived at an unexpected barrier generation");
                }
                if arrivals.contains_key(&record.wave) {
                    reason = Some("duplicate wave arrival");
                }
                if arrivals
                    .values()
                    .any(|(event, _)| graph.event(*event).pc != graph.event(record.event).pc)
                {
                    reason = Some("waves reached different barrier instructions");
                }
                if let Some(reason) = reason {
                    barrier_violations.push(AmdBarrierViolation {
                        generation: *generation,
                        event: record.event,
                        wave: record.wave,
                        reason: reason.into(),
                    });
                }
                arrivals.insert(record.wave, (record.event, clocks[wave].clone()));
                blocked_at[wave] = Some(*generation);
            }
            AmdEventKind::BarrierRelease { generation } => {
                let arrivals = barriers.get(generation);
                let mut missing = Vec::new();
                let mut exited = Vec::new();
                for w in 0..waves {
                    let wave_exited =
                        (w * 64..((w + 1) * 64).min(graph.threads() as usize)).all(|t| {
                            graph
                                .program_order(volta_graph::ThreadIx(t as u32))
                                .last()
                                .is_some_and(|&e| {
                                    e < record.event && graph.event(e).kind == EventKind::Exit
                                })
                        });
                    if wave_exited {
                        exited.push(w);
                    } else if !arrivals.is_some_and(|a| a.contains_key(&(w as u32))) {
                        missing.push(w);
                    }
                }
                let own = arrivals.is_some_and(|a| a.contains_key(&record.wave));
                let matched_pc =
                    arrivals
                        .and_then(|a| a.get(&record.wave))
                        .is_some_and(|(event, _)| {
                            graph.event(*event).pc == graph.event(record.event).pc
                        });
                if !missing.is_empty()
                    || !own
                    || !matched_pc
                    || blocked_at[wave] != Some(*generation)
                    || !releases.insert((*generation, record.wave))
                {
                    barrier_violations.push(AmdBarrierViolation {
                        generation: *generation,
                        event: record.event,
                        wave: record.wave,
                        reason: format!(
                            "invalid barrier release; missing waves {missing:?}, own arrival {own}"
                        ),
                    });
                } else {
                    for (_, clock) in arrivals.unwrap().values() {
                        join(&mut clocks[wave], clock);
                    }
                    // CDNA4 §4.3 / S_BARRIER counts a terminated wave as
                    // satisfied. S_ENDPGM (§12.5, printed142) implicitly waits
                    // for every counter, so its completed effects also precede
                    // this surviving-wave release. No exit or trace order by
                    // itself manufactures completion for a missing drain.
                    for exited_wave in exited {
                        if !pending[exited_wave].is_empty() {
                            barrier_unresolved.push(Unresolved::at(
                                record.event,
                                "terminated wave lacks its implicit completed-memory drain",
                            ));
                        } else {
                            let frontier = clocks[exited_wave].clone();
                            join(&mut clocks[wave], &frontier);
                        }
                    }
                    next_generation[wave] = generation + 1;
                    blocked_at[wave] = None;
                }
            }
            // The trace contract already validates membership, roles and bounds.
            // Their parent MemoryIssue/Instruction supplies counter/timeline state.
            AmdEventKind::PrivateMemoryAccess { .. }
            | AmdEventKind::LaneRead { .. }
            | AmdEventKind::LaneWrite { .. } => {}
            _ => unreachable!("unsupported target events are rejected by the trace contract"),
        }
    }
    if matches!(graph.terminal(), Terminal::Completed) {
        for (&generation, arrivals) in &barriers {
            for (&wave, (event, _)) in arrivals {
                if !releases.contains(&(generation, wave)) {
                    barrier_violations.push(AmdBarrierViolation {
                        generation,
                        event: *event,
                        wave,
                        reason: "completed trace leaves a barrier arrival unreleased".into(),
                    });
                }
            }
        }
    }
    let wait_readiness = PropertyResult::from_findings(
        AmdWaitEvidence {
            issues: operations.len(),
            waits,
        },
        hazards,
        Vec::new(),
    );
    let barrier = PropertyResult::from_findings(
        AmdBarrierEvidence {
            generations: barriers.len(),
            arrivals: barrier_arrivals,
            releases: releases.len(),
        },
        barrier_violations,
        barrier_unresolved,
    );
    let memory_conflict = if barrier.verdict() == Verdict::Pass {
        memory_conflicts(graph, &operations, &issue_index)
    } else {
        unknown("AMD barrier validation did not pass")
    };
    AmdReport {
        trace,
        wait_readiness,
        barrier,
        memory_conflict,
    }
}

struct EffectNode {
    event: EventId,
    operation: usize,
    mode: volta_graph::AccessMode,
    children: Vec<usize>,
    scoped: bool,
    subtree_scoped: bool,
    subtree_writes: bool,
    subtree_wave: Option<u32>,
}

/// Ordered read roots can hide older conflicting writes in the overlap forest.
/// Check the relevant ancestors too; subtree summaries keep pure reads,
/// single-wave traffic and fully scoped publication paths constant-time.
fn missing_global_publication(
    root: &EffectNode,
    nodes: &[EffectNode],
    operations: &[Operation],
    later_wave: u32,
    later_mode: volta_graph::AccessMode,
    later_scoped: bool,
    completed_read_before_write: bool,
) -> Option<EventId> {
    let mut pending = vec![root];
    while let Some(node) = pending.pop() {
        if node.subtree_wave == Some(later_wave)
            || (later_mode == volta_graph::AccessMode::Read && !node.subtree_writes)
            || (later_scoped && node.subtree_scoped)
        {
            continue;
        }
        if operations[node.operation].wave != later_wave
            && ConflictKind::of(node.mode, later_mode).is_some()
            && !(completed_read_before_write && node.mode == volta_graph::AccessMode::Read)
            && (!later_scoped || !node.scoped)
        {
            return Some(node.event);
        }
        pending.extend(node.children.iter().map(|&child| &nodes[child]));
    }
    None
}

fn memory_conflicts(
    graph: &EventGraph,
    operations: &[Operation],
    issue_index: &HashMap<EventId, usize>,
) -> MemoryConflictResult {
    memory_conflicts_with_clock(graph, operations, issue_index, false)
}

fn memory_conflicts_with_clock(
    graph: &EventGraph,
    operations: &[Operation],
    issue_index: &HashMap<EventId, usize>,
    arrivals_only: bool,
) -> MemoryConflictResult {
    let links: HashMap<EventId, usize> = graph
        .amd_memory_effects()
        .iter()
        .map(|l| (l.effect, issue_index[&l.issue]))
        .collect();
    // CDNA3/CDNA4 §9.1.10.2: SC0=1, SC1=0 is group scope.
    // LLVM AMDGPUUsage.rst's global.load/store.b128 tables explicitly group
    // gfx942 and gfx950 together with the same workgroup SC0 policy.
    // LLVM SIMemoryLegalizer.cpp at846473237377990d00b9c353f6a2c86116b52ea5
    // enableLoad/StoreCacheBypass selects SC0 for workgroup accesses;
    // insertAcquire needs no CU-cache invalidate when TG_SPLIT is disabled,
    // since every wave then shares the CU/L1. The scope supplies visibility,
    // never completion or HB. No FLAT/SMEM/device/system protocol is inferred.
    let single_cu = graph
        .provenance()
        .assumptions
        .contains(&volta_graph::Assumption::SingleCuWorkgroup);
    let gfx1250 = AmdArchitecture::from_targets(&graph.provenance().targets)
        == Some(AmdArchitecture::Gfx1250);
    // LLVM AMDGPUUsage.md at5fd50ec8, Memory Model GFX125x: explicit
    // SCOPE_SYS forwards both accesses through smaller-scope caches. This
    // supplies visibility only after the existing completion and HB checks.
    // B0 communicating-waves profile: the selected workgroup's waves share one
    // workgroup-coherent vector cache, so every GLOBAL effect is published to
    // the other waves once completion and happens-before hold.
    let workgroup_coherent = gfx1250
        && graph
            .provenance()
            .assumptions
            .contains(&volta_graph::Assumption::Gfx1250B0WorkgroupCoherentProfile);
    let scoped: HashSet<EventId> = graph
        .amd_memory_effects()
        .iter()
        .filter(|link| {
            if gfx1250 {
                link.scope == AmdMemoryScope::Gfx1250System
                    || (workgroup_coherent
                        && matches!(graph.event(link.effect).kind, EventKind::Access(a)
                            if graph.access(a).space == Space::Global))
            } else {
                single_cu && link.scope == AmdMemoryScope::Workgroup
            }
        })
        .map(|link| link.effect)
        .collect();
    // B0 litmus profile: lanes of one range-checked buffer store that write
    // the same bytes are ordered by lane (the higher lane's data overrides
    // during write combining); the interpreter writes lanes in ascending order.
    let assumptions = &graph.provenance().assumptions;
    let lane_ordered_stores: HashSet<usize> = if assumptions
        .contains(&volta_graph::Assumption::Gfx1250B0MultiwaveProfile)
        || assumptions.contains(&volta_graph::Assumption::Gfx1250B0LitmusBarrierProfile)
    {
        // B0 multiwave and litmus/2 profiles: the same write-combining
        // order for every vector-memory store of the selected profile.
        operations
            .iter()
            .enumerate()
            .filter(|(_, operation)| operation.class == AmdMemoryClass::VectorStore)
            .map(|(index, _)| index)
            .collect()
    } else if assumptions.contains(&volta_graph::Assumption::Gfx1250B0LitmusProfile) {
        graph
            .amd_events()
            .iter()
            .filter_map(|record| match &record.kind {
                AmdEventKind::BufferRangeCheck { issue, .. } => issue_index.get(issue).copied(),
                _ => None,
            })
            .filter(|&operation| operations[operation].class == AmdMemoryClass::VectorStore)
            .collect()
    } else {
        HashSet::new()
    };
    let mut written = written_bytes(graph);
    // Native trace validation admits only the selected RMW profile. Keep its
    // writes in this domain without changing PTX's ordinary-only domain M.
    written.extend(
        graph
            .accesses()
            .values()
            .filter(|a| {
                a.ordering == OrderingKind::Atomic && a.mode == volta_graph::AccessMode::Write
            })
            .flat_map(bytes),
    );
    let mut atomic_unresolved = Vec::new();
    let mut roots = HashMap::<ByteLocation, Vec<usize>>::new();
    let mut nodes = Vec::<EffectNode>::new();
    let mut conflicts = Vec::new();
    let mut reported = HashSet::new();
    let mut unresolved = Vec::new();
    for (event, ev) in graph.events().iter() {
        let EventKind::Access(a) = ev.kind else {
            continue;
        };
        let access = graph.access(a);
        let operation = links[&event];
        let later = &operations[operation];
        let later_scoped = scoped.contains(&event);
        let mut reaches = |node: &EffectNode, forest: &[EffectNode]| {
            let earlier = &operations[node.operation];
            let atomic_self = node.operation == operation
                && (later.class == AmdMemoryClass::GlobalAtomicAddF32NoReturn
                    || lane_ordered_stores.contains(&operation));
            let clock = if arrivals_only {
                later.arrival_clock.as_ref().unwrap_or(&later.clock)
            } else {
                &later.clock
            };
            let ordered = atomic_self
                || clock[earlier.wave as usize * CLASSES + class_index(earlier.class)]
                    >= earlier.sequence;
            if ordered
                && access.space == Space::Global
                && unresolved.is_empty()
                && let Some(earlier) = missing_global_publication(
                    node,
                    forest,
                    operations,
                    later.wave,
                    access.mode,
                    later_scoped,
                    gfx1250 && access.mode == volta_graph::AccessMode::Write,
                )
            {
                unresolved.push(Unresolved::between(earlier, event,
                        if gfx1250 {
                            "gfx1250 ordered cross-wave GLOBAL visibility requires explicit SCOPE_SYS at both endpoints"
                        } else {
                            "native AMD cache visibility for an ordered cross-wave GLOBAL conflict requires workgroup-scoped endpoints and a single-CU workgroup premise"
                        }));
            }
            ordered
        };
        for location in bytes(access) {
            if !written.contains(&location) {
                continue;
            }
            let previous = roots.remove(&location).unwrap_or_default();
            let mut children = Vec::new();
            let mut kept = Vec::new();
            let mut unordered = Vec::new();
            for node in previous {
                if reaches(&nodes[node], &nodes) {
                    children.push(node)
                } else {
                    kept.push(node);
                    unordered.push(node)
                }
            }
            while let Some(n) = unordered.pop() {
                if let Some(kind) = ConflictKind::of(nodes[n].mode, access.mode) {
                    let both_atomic = later.class == AmdMemoryClass::GlobalAtomicAddF32NoReturn
                        && operations[nodes[n].operation].class
                            == AmdMemoryClass::GlobalAtomicAddF32NoReturn;
                    if both_atomic && reported.insert((nodes[n].event, event)) {
                        atomic_unresolved.push(Unresolved::between(nodes[n].event, event,
                            "concurrent atomic RMW serialization is outside the selected single-issuer profile"));
                    } else if !both_atomic
                        && reported.insert((nodes[n].event, event))
                        && conflicts.len() < CONFLICT_CAP
                    {
                        conflicts.push(MemoryConflict {
                            kind,
                            location,
                            earlier: nodes[n].event,
                            later: event,
                        });
                    }
                }
                for &child in &nodes[n].children {
                    if !reaches(&nodes[child], &nodes) {
                        unordered.push(child);
                    }
                }
            }
            let index = nodes.len();
            let mut subtree_scoped = later_scoped;
            let mut subtree_writes = access.mode == volta_graph::AccessMode::Write;
            let mut subtree_wave = Some(later.wave);
            for &child in &children {
                subtree_scoped &= nodes[child].subtree_scoped;
                subtree_writes |= nodes[child].subtree_writes;
                if nodes[child].subtree_wave != subtree_wave {
                    subtree_wave = None;
                }
            }
            nodes.push(EffectNode {
                event,
                operation,
                mode: access.mode,
                children,
                scoped: later_scoped,
                subtree_scoped,
                subtree_writes,
                subtree_wave,
            });
            kept.push(index);
            roots.insert(location, kept);
        }
    }
    conflicts.sort_by_key(|c| (c.later, c.earlier));
    unresolved.extend(atomic_unresolved);
    PropertyResult::from_findings(
        MemoryConflictEvidence {
            effects: links.len(),
            written_bytes: written.len(),
        },
        conflicts,
        unresolved,
    )
}

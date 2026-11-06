//! Independently validated rejection of an attempted gfx1250 global access
//! whose footprint touches a configured guard or leaves every configured
//! object. This never establishes complete execution, outputs or acceptance.
use crate::{Unresolved, Verdict};
use volta_graph::{
    AbortCause, AmdArchitecture, AmdEvent, AmdEventKind, AmdGlobalAccess, EventGraph, EventId,
    GLOBAL_BOUNDS_ABORT_REASON, Provenance, Terminal,
};

/// Which configured boundary the first offending lane violates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlobalBoundsRegion {
    /// A byte lies in this guard; `offset` is its byte offset in the guard.
    Guard { name: String, offset: u64 },
    /// The footprint is not contained in any configured object.
    OutsideObjects,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalBoundsViolation {
    pub event: EventId,
    pub wave: u32,
    pub access: AmdGlobalAccess,
    /// Lowest offending lane, its footprint start and width.
    pub lane: u8,
    pub address: u64,
    pub bytes: u64,
    pub region: GlobalBoundsRegion,
    /// Every offending lane, in increasing order.
    pub lanes: Vec<u8>,
}

/// Untrusted exact-prefix identity. The frontend remains trusted to compute
/// lane addresses; replay validates them against the recorded layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalBoundsCertificate {
    pub provenance: Provenance,
    pub amd_records: usize,
    pub graph_events: usize,
    pub event: EventId,
    pub fault_record: AmdEvent,
}

/// Identify a candidate only. No verdict is implied until replay succeeds.
pub fn global_bounds_candidate(graph: &EventGraph) -> Option<GlobalBoundsCertificate> {
    let last = graph.amd_events().last()?;
    matches!(last.kind, AmdEventKind::GlobalAccessFault { .. }).then(|| GlobalBoundsCertificate {
        provenance: graph.provenance().clone(),
        amd_records: graph.amd_events().len(),
        graph_events: graph.events().len(),
        event: last.event,
        fault_record: last.clone(),
    })
}

fn classify(
    graph: &EventGraph,
    access: AmdGlobalAccess,
    address: u64,
    bytes: u64,
) -> Option<GlobalBoundsRegion> {
    let layout = graph.global_region_layout();
    let Some(end) = address.checked_add(bytes) else {
        return Some(GlobalBoundsRegion::OutsideObjects);
    };
    if let Some(guard) = layout
        .iter()
        .find(|r| r.guard && address < r.base.saturating_add(r.size) && r.base < end)
    {
        return Some(GlobalBoundsRegion::Guard {
            name: guard.name.clone(),
            offset: address.max(guard.base) - guard.base,
        });
    }
    let speculative = matches!(access, AmdGlobalAccess::Prefetch(s) if s.speculative_prefetch());
    (!speculative
        && !layout
            .iter()
            .any(|r| !r.guard && r.base <= address && end <= r.base.saturating_add(r.size)))
    .then_some(GlobalBoundsRegion::OutsideObjects)
}

pub fn validate_global_bounds_prefix(
    graph: &EventGraph,
    certificate: &GlobalBoundsCertificate,
) -> Result<GlobalBoundsViolation, Vec<Unresolved>> {
    let fail = |reason| vec![Unresolved::global(reason)];
    if AmdArchitecture::from_targets(&graph.provenance().targets) != Some(AmdArchitecture::Gfx1250)
        || graph.provenance() != &certificate.provenance
        || graph.amd_events().len() != certificate.amd_records
        || graph.events().len() != certificate.graph_events
    {
        return Err(fail(
            "global bounds certificate does not identify this exact gfx1250 prefix",
        ));
    }
    if let Err(defects) = graph.validate() {
        return Err(defects
            .into_iter()
            .map(|defect| Unresolved {
                events: defect
                    .event
                    .filter(|event| graph.events().get(*event).is_some())
                    .into_iter()
                    .collect(),
                reason: defect.message,
            })
            .collect());
    }
    let record = graph
        .amd_events()
        .last()
        .filter(|r| r.event == certificate.event && **r == certificate.fault_record)
        .ok_or_else(|| fail("global bounds certificate has no terminal fault record"))?;
    let AmdEventKind::GlobalAccessFault {
        access,
        bytes_per_lane,
        lanes,
    } = &record.kind
    else {
        return Err(fail("global bounds certificate is not an attempted access"));
    };
    let source_pc = graph
        .events()
        .get(record.event)
        .map(|event| event.pc)
        .ok_or_else(|| fail("global fault record has an invalid graph event ID"))?;
    if !matches!(graph.terminal(), Terminal::Aborted { cause: AbortCause::Error, reason, thread: Some(thread), pc }
        if reason == GLOBAL_BOUNDS_ABORT_REASON && record.wave.checked_mul(32) == Some(thread.0) && *pc == source_pc)
    {
        return Err(fail(
            "global bounds requires the matching typed access abort, without completed execution",
        ));
    }
    if graph.global_region_layout().is_empty() {
        return Err(fail(
            "global bounds requires a configured object and guard layout",
        ));
    }
    let bytes = u64::from(*bytes_per_lane);
    let offending: Vec<_> = lanes
        .iter()
        .filter_map(|&(lane, address)| {
            classify(graph, *access, address, bytes).map(|region| (lane, address, region))
        })
        .collect();
    let Some((lane, address, region)) = offending.first().cloned() else {
        return Err(fail(
            "every recorded lane footprint lies inside configured objects",
        ));
    };
    // The trace contract checks that the record lists exactly the active lanes
    // of its instruction; readiness and reachability establish the addresses.
    let replay = super::gfx1250::check_prefix(graph);
    if replay.trace.verdict() != Verdict::Pass {
        return Err(replay.trace.unresolved().to_vec());
    }
    if replay.wait_readiness.verdict() != Verdict::Pass {
        return Err(fail(
            "address source readiness is not established for this prefix",
        ));
    }
    if replay.barrier.verdict() != Verdict::Pass {
        return Err(fail(
            "faulting instruction execution/barrier reachability is not established",
        ));
    }
    Ok(GlobalBoundsViolation {
        event: record.event,
        wave: record.wave,
        access: *access,
        lane,
        address,
        bytes,
        region,
        lanes: offending.iter().map(|(lane, ..)| *lane).collect(),
    })
}

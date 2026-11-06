//! Independently validated rejection of a supported attempted tensor LDS access.
//! This never establishes complete execution, valid outputs or acceptance.
use crate::{Unresolved, Verdict};
use volta_graph::{
    AbortCause, AccessMode, AmdArchitecture, AmdEvent, AmdEventKind, Assumption, EventGraph,
    EventId, ProgramPoint, Provenance, TENSOR_BOUNDS_ABORT_REASON, Terminal,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorBoundsViolation {
    pub event: EventId,
    pub wave: u32,
    pub access: AccessMode,
    pub address: u64,
    pub bytes: u64,
    pub configured_lds_bytes: u64,
}

/// Untrusted exact-prefix identity. The frontend remains trusted to capture
/// descriptor values/register roles; this is not independent source decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorBoundsCertificate {
    pub provenance: Provenance,
    pub amd_records: usize,
    pub graph_events: usize,
    pub event: EventId,
    pub fault_record: AmdEvent,
}

/// Identify a candidate only. No verdict is implied until replay succeeds.
pub fn tensor_bounds_candidate(graph: &EventGraph) -> Option<TensorBoundsCertificate> {
    let last = graph.amd_events().last()?;
    matches!(last.kind, AmdEventKind::TensorAccessFault { .. }).then(|| TensorBoundsCertificate {
        provenance: graph.provenance().clone(),
        amd_records: graph.amd_events().len(),
        graph_events: graph.events().len(),
        event: last.event,
        fault_record: last.clone(),
    })
}

fn fault_source_pc(
    graph: &EventGraph,
    event: EventId,
) -> Result<Option<ProgramPoint>, Vec<Unresolved>> {
    graph
        .events()
        .get(event)
        .map(|event| event.pc)
        .ok_or_else(|| {
            vec![Unresolved::global(
                "tensor fault record has an invalid graph event ID",
            )]
        })
}

pub fn validate_tensor_bounds_prefix(
    graph: &EventGraph,
    certificate: &TensorBoundsCertificate,
) -> Result<TensorBoundsViolation, Vec<Unresolved>> {
    let fail = |reason| vec![Unresolved::global(reason)];
    if AmdArchitecture::from_targets(&graph.provenance().targets) != Some(AmdArchitecture::Gfx1250)
        || graph.provenance() != &certificate.provenance
        || graph.amd_events().len() != certificate.amd_records
        || graph.events().len() != certificate.graph_events
    {
        return Err(fail(
            "tensor bounds certificate does not identify this exact gfx1250 prefix",
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
        .ok_or_else(|| fail("tensor bounds certificate has no terminal fault record"))?;
    let AmdEventKind::TensorAccessFault {
        load, descriptor, ..
    } = &record.kind
    else {
        return Err(fail(
            "tensor bounds certificate is not an attempted tensor access",
        ));
    };
    let source_pc = fault_source_pc(graph, record.event)?;
    if !matches!(graph.terminal(), Terminal::Aborted { cause: AbortCause::Error, reason, thread: Some(thread), pc }
        if reason == TENSOR_BOUNDS_ABORT_REASON && record.wave.checked_mul(32) == Some(thread.0) && *pc == source_pc)
    {
        return Err(fail(
            "tensor bounds requires the matching typed access abort, without completed execution",
        ));
    }
    let limits: Vec<_> = graph
        .provenance()
        .assumptions
        .iter()
        .filter_map(|a| {
            if let Assumption::ConfiguredLdsBytes(bytes) = a {
                Some(*bytes)
            } else {
                None
            }
        })
        .collect();
    if limits.len() != 1
        || limits[0] > AmdArchitecture::Gfx1250.lds_capacity()
        || graph.shared_allocation_layout().iter().any(|a| {
            a.offset
                .checked_add(a.size)
                .is_none_or(|end| end > limits[0])
        })
    {
        return Err(fail(
            "tensor bounds requires exactly one valid configured LDS extent",
        ));
    }
    let footprint = descriptor.decode().map_err(fail)?.ok_or_else(|| {
        fail("a predicate-disabled tensor descriptor cannot violate access bounds")
    })?;
    if footprint.shared_end <= limits[0] {
        return Err(fail(
            "tensor descriptor does not exceed the configured LDS extent",
        ));
    }
    let replay = super::gfx1250::check_prefix(graph);
    if replay.trace.verdict() != Verdict::Pass {
        return Err(replay.trace.unresolved().to_vec());
    }
    if replay.wait_readiness.verdict() != Verdict::Pass {
        return Err(fail(
            "tensor descriptor source readiness is not established for this prefix",
        ));
    }
    if replay.barrier.verdict() != Verdict::Pass {
        return Err(fail(
            "tensor fault instruction execution/barrier reachability is not established",
        ));
    }
    Ok(TensorBoundsViolation {
        event: record.event,
        wave: record.wave,
        access: if *load {
            AccessMode::Write
        } else {
            AccessMode::Read
        },
        address: footprint.shared,
        bytes: footprint.bytes,
        configured_lds_bytes: limits[0],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volta_graph::GraphBuilder;

    #[test]
    fn malformed_fault_event_id_uses_checked_lookup() {
        let graph = GraphBuilder::new(Provenance {
            kernel: "malformed_fault".into(),
            block_dim: (1, 1, 1),
            unit: volta_graph::ThreadUnit::CtaThread,
            grid_dim: (1, 1, 1),
            workgroup_id: (0, 0, 0),
            targets: vec!["gfx1250".into()],
            assumptions: vec![],
            instructions: vec![],
        })
        .finish(Terminal::Aborted {
            thread: None,
            pc: None,
            cause: AbortCause::Error,
            reason: TENSOR_BOUNDS_ABORT_REASON.into(),
        });
        for event in [EventId(0), EventId(u32::MAX)] {
            assert!(fault_source_pc(&graph, event).is_err());
        }
    }
}

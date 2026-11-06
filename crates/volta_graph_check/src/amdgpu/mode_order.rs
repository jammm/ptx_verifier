//! Ordering of gfx1250 runtime MODE accesses under the B0 MODE profiles.
//!
//! The profile does not model when a MODE read by S_GETREG_B32 completes
//! relative to a later MODE write (S_SETREG_B32, S_SETREG_IMM32_B32,
//! S_ROUND_MODE or S_DENORM_MODE). Such a pair is inside the modelled subset
//! only when an S_WAIT_ALU whose SA_SDST field is zero separates them. Any
//! other pair is reported as unresolved: neither a failure nor accepted.
//!
//! A write followed by a read, and a write followed by a write, are in program
//! order and need no wait in this profile.
use crate::Verdict;
use volta_graph::{AmdEventKind, AmdModeOp, EventGraph, EventId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeOrderKind {
    /// GETREG MODE then a MODE write with no SA_SDST==0 wait between.
    UnresolvedWriteAfterRead,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModeOrderFinding {
    pub kind: ModeOrderKind,
    pub wave: u32,
    /// The pending S_GETREG_B32's MODE access event.
    pub read: EventId,
    /// The later MODE write's access event.
    pub write: EventId,
    pub write_op: AmdModeOp,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModeOrderReport {
    /// MODE access records checked.
    pub accesses: usize,
    pub unresolved: Vec<ModeOrderFinding>,
    /// Records the check could not interpret; any entry makes it inconclusive.
    pub malformed: Vec<String>,
}

impl ModeOrderReport {
    pub fn verdict(&self) -> Verdict {
        if !self.unresolved.is_empty() || !self.malformed.is_empty() {
            Verdict::Unknown
        } else {
            Verdict::Pass
        }
    }
}

/// Check every wave's MODE accesses in recorded (program) order. Each access
/// must directly follow its own instruction record (same wave and pc).
pub fn check_mode_order(graph: &EventGraph) -> ModeOrderReport {
    let mut report = ModeOrderReport::default();
    // Per wave: the S_GETREG_B32 accesses not yet ordered by a wait.
    let mut pending: std::collections::BTreeMap<u32, Vec<EventId>> = Default::default();
    // Per wave: pc of the last instruction record.
    let mut last_instruction: std::collections::BTreeMap<u32, Option<u32>> = Default::default();
    for record in graph.amd_events() {
        let pc = graph.event(record.event).pc.map(|pc| pc.0);
        match &record.kind {
            AmdEventKind::Instruction { .. } => {
                last_instruction.insert(record.wave, pc);
            }
            AmdEventKind::AluWait { wait } if wait.sa_sdst() == 0 => {
                pending.remove(&record.wave);
            }
            AmdEventKind::ModeAccess {
                op, offset, width, ..
            } => {
                report.accesses += 1;
                if pc.is_none() || last_instruction.get(&record.wave).copied().flatten() != pc {
                    report.malformed.push(format!(
                        "MODE access event {} does not follow its instruction record",
                        record.event.0
                    ));
                    continue;
                }
                if !matches!((offset, width), (0 | 4, 4)) {
                    report.malformed.push(format!(
                        "MODE access event {} selects bits outside FP_ROUND/FP_DENORM",
                        record.event.0
                    ));
                    continue;
                }
                let reads = pending.entry(record.wave).or_default();
                match op {
                    AmdModeOp::Getreg => reads.push(record.event),
                    AmdModeOp::Setreg
                    | AmdModeOp::SetregImm32
                    | AmdModeOp::RoundMode
                    | AmdModeOp::DenormMode => {
                        for &read in reads.iter() {
                            report.unresolved.push(ModeOrderFinding {
                                kind: ModeOrderKind::UnresolvedWriteAfterRead,
                                wave: record.wave,
                                read,
                                write: record.event,
                                write_op: *op,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }
    report
}

//! gfx1250 B0 WMMA spacing under the WMMA and multiwave profiles.
//!
//! A slot is a V_NOP or an independent vector-ALU instruction: one that
//! touches no register of the earlier WMMA. Transcendental, shifted and WMMA
//! instructions are not slots. For an F16/BF16 WMMA:
//!
//! * a later WMMA reading its D as A or B needs 5 slots (certified failure);
//! * a later VALU reading its D, writing its D, or writing its A, B or C
//!   needs 4 slots (certified failure);
//! * two WMMA with identical operands and no vector-ALU instruction between
//!   them are a certified failure;
//! * a later WMMA reading its D as C, writing its D, or writing its A, B or C
//!   is not covered by the documented table; fewer than 5 slots (the largest
//!   listed requirement) is reported as below a conservative policy, which is
//!   neither a failure nor accepted.
//!
//! [`check_wmma_hazards_types`] (profile `gfx1250-b0-wmma/2`) applies the same
//! rules per WMMA type (F16/BF16 5 and 4, FP8 16x16x64 2 and 1, FP8 16x16x128
//! 4 and 3, IU8 9 and 8; the policy count is the type's WMMA-to-WMMA count)
//! and counts only instructions issued with a nonzero EXEC as slots. Whether
//! an instruction issued with EXEC = 0 provides a slot is not documented: a
//! dependency that meets its count only if such instructions counted is
//! reported as unresolved, never as a failure or as accepted.
//!
//! Profile `gfx1250-b0-wmma/3` adds the BF8 combinations (16x16x64 2 and 1,
//! 16x16x128 4 and 3), F8F6F4 (both sources FP4: 2 and 1, otherwise 4 and 3)
//! and F4 32x16x128 (4 and 3), scaled or not. The block-scale operands of the
//! 4-dword scaled forms are named by no documented row: a scale operand that
//! reads an earlier WMMA's D, or a VALU that writes an earlier WMMA's scale
//! registers, below that WMMA's count is unresolved, never a failure.
//!
//! Profile `gfx1250-b0-wmma/4` adds the 16-bit-output forms (F16/BF16 inputs
//! 5 and 4, FP8/BF8 16x16x64 2 and 1, 16x16x128 4 and 3) and SWMMAC (3 and
//! 2; IU8 5 and 4). The SWMMAC sparsity index is a documented source (the
//! table names A/B/Index and A/B/C/Index), so it is checked with A and B.
use crate::Verdict;
use std::collections::{BTreeMap, HashMap};
use volta_graph::{
    AmdEventKind, AmdRegister, AmdSmallFloat, AmdSwmmacInput, AmdWmmaKind, EventGraph, EventId,
};

/// Slots after an F16/BF16 WMMA before a dependent WMMA (A/B) or the policy.
pub const XDL_SLOTS: u64 = 5;
/// Slots after an F16/BF16 WMMA before a dependent VALU.
pub const VALU_SLOTS: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WmmaHazardKind {
    /// Later WMMA reads the earlier D as A or B.
    MatrixABReadBeforeReady,
    /// Later VALU reads the earlier D.
    ValuReadBeforeReady,
    /// Later VALU writes the earlier D.
    ValuDestinationReuse,
    /// Later VALU writes the earlier A, B or C.
    ValuSourceReuse,
    /// Identical operands, no vector-ALU instruction between.
    SameArgumentsBackToBack,
    /// Policy: later WMMA reads the earlier D as C.
    UnlistedAccumulatorChain,
    /// Policy: later WMMA writes the earlier D.
    UnlistedDestinationReuse,
    /// Policy: later WMMA writes the earlier A, B or C.
    UnlistedSourceReuse,
    /// Unresolved (types profile): the dependency meets its count only if
    /// instructions issued with EXEC = 0 counted as slots.
    Exec0SlotsUnresolved,
    /// Policy (`wmma/3`): a later WMMA's block-scale operand reads the earlier
    /// D, or a later VALU writes the earlier block-scale registers.
    ScaleSourceBelowPolicy,
}

impl WmmaHazardKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::MatrixABReadBeforeReady => "WMMA_AB_READ_BEFORE_READY",
            Self::ValuReadBeforeReady => "WMMA_VALU_READ_BEFORE_READY",
            Self::ValuDestinationReuse => "WMMA_VALU_DESTINATION_REUSE",
            Self::ValuSourceReuse => "WMMA_VALU_SOURCE_REUSE",
            Self::SameArgumentsBackToBack => "WMMA_SAME_ARGUMENTS_BACK_TO_BACK",
            Self::UnlistedAccumulatorChain
            | Self::UnlistedDestinationReuse
            | Self::UnlistedSourceReuse => "WMMA_UNLISTED_CHAIN_BELOW_POLICY",
            Self::Exec0SlotsUnresolved => "WMMA_EXEC0_SLOTS_UNRESOLVED",
            Self::ScaleSourceBelowPolicy => "WMMA_SCALE_SOURCE_BELOW_POLICY",
        }
    }
    /// Unresolved rather than a documented failure.
    pub const fn policy(self) -> bool {
        matches!(
            self,
            Self::UnlistedAccumulatorChain
                | Self::UnlistedDestinationReuse
                | Self::UnlistedSourceReuse
                | Self::Exec0SlotsUnresolved
                | Self::ScaleSourceBelowPolicy
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WmmaHazardFinding {
    pub kind: WmmaHazardKind,
    pub wave: u32,
    /// The earlier WMMA's instruction record.
    pub first: EventId,
    /// The later instruction's record.
    pub second: EventId,
    pub register: Option<AmdRegister>,
    pub slots: u64,
    pub required: u64,
    /// Instructions issued with EXEC = 0 that would have been slots (types
    /// profile only; never counted).
    pub exec0: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WmmaHazardReport {
    pub wmmas: usize,
    pub violations: Vec<WmmaHazardFinding>,
    /// Unlisted dependencies below the conservative policy.
    pub policy: Vec<WmmaHazardFinding>,
    /// Records the check could not interpret; any entry makes it inconclusive.
    pub malformed: Vec<String>,
}

impl WmmaHazardReport {
    pub fn verdict(&self) -> Verdict {
        if !self.violations.is_empty() {
            Verdict::Fail
        } else if !self.policy.is_empty() || !self.malformed.is_empty() {
            Verdict::Unknown
        } else {
            Verdict::Pass
        }
    }
}

struct Open {
    event: EventId,
    d: Vec<AmdRegister>,
    ab: Vec<AmdRegister>,
    c: Vec<AmdRegister>,
    /// Block-scale operand registers (scaled forms only).
    scales: Vec<AmdRegister>,
    slots: u64,
    valu_since: u64,
    /// Slots required before a dependent WMMA (also the policy count) and
    /// before a dependent VALU.
    xdl: u64,
    valu: u64,
    /// EXEC = 0 V_NOP or vector-ALU instructions since the WMMA.
    exec0: u64,
}

/// Documented (WMMA, VALU) slot counts of a WMMA type under the types
/// profile; `None` for a type outside it.
pub const fn type_slots(kind: AmdWmmaKind) -> Option<(u64, u64)> {
    match kind {
        AmdWmmaKind::F16_16x16x32 | AmdWmmaKind::Bf16_16x16x32 => Some((5, 4)),
        AmdWmmaKind::Fp8Fp8_16x16x64 => Some((2, 1)),
        AmdWmmaKind::Fp8Fp8_16x16x128 => Some((4, 3)),
        AmdWmmaKind::Iu8_16x16x64 => Some((9, 8)),
        // `wmma/3` (the only profile that issues these kinds).
        AmdWmmaKind::Fp8Bf8_16x16x64
        | AmdWmmaKind::Bf8Fp8_16x16x64
        | AmdWmmaKind::Bf8Bf8_16x16x64 => Some((2, 1)),
        AmdWmmaKind::Fp8Bf8_16x16x128
        | AmdWmmaKind::Bf8Fp8_16x16x128
        | AmdWmmaKind::Bf8Bf8_16x16x128
        | AmdWmmaKind::F4_32x16x128 => Some((4, 3)),
        AmdWmmaKind::F8f6f4_16x16x128 {
            a: AmdSmallFloat::Fp4,
            b: AmdSmallFloat::Fp4,
        } => Some((2, 1)),
        AmdWmmaKind::F8f6f4_16x16x128 { .. } => Some((4, 3)),
        // `wmma/4` (the only profile that issues these kinds).
        AmdWmmaKind::Half16x16x32 { .. } => Some((5, 4)),
        AmdWmmaKind::F16Out8 { k128: false, .. } => Some((2, 1)),
        AmdWmmaKind::F16Out8 { k128: true, .. } => Some((4, 3)),
        AmdWmmaKind::Swmmac {
            input: AmdSwmmacInput::Iu8,
            ..
        } => Some((5, 4)),
        AmdWmmaKind::Swmmac { .. } => Some((3, 2)),
        _ => None,
    }
}

impl Open {
    fn touches(&self, register: AmdRegister) -> bool {
        self.d
            .iter()
            .chain(&self.ab)
            .chain(&self.c)
            .chain(&self.scales)
            .any(|r| r.overlaps(register))
    }
}

fn first_overlap(a: &[AmdRegister], b: &[AmdRegister]) -> Option<AmdRegister> {
    b.iter().copied().find(|r| a.iter().any(|x| x.overlaps(*r)))
}

/// Check every wave's WMMA spacing in recorded (execution) order.
pub fn check_wmma_hazards(graph: &EventGraph) -> WmmaHazardReport {
    check(graph, false)
}

/// [`check_wmma_hazards`] for the WMMA types profile: per-type counts, and
/// instructions issued with EXEC = 0 are never slots.
pub fn check_wmma_hazards_types(graph: &EventGraph) -> WmmaHazardReport {
    check(graph, true)
}

fn check(graph: &EventGraph, types: bool) -> WmmaHazardReport {
    let mut report = WmmaHazardReport::default();
    // Each WMMA issue follows its own instruction record in the same wave.
    let mut last_instruction = HashMap::<u32, EventId>::new();
    let mut issues = HashMap::<
        EventId,
        (
            AmdWmmaKind,
            &Vec<AmdRegister>,
            // A and B, then the SWMMAC index (a documented A/B/Index source).
            Vec<AmdRegister>,
            &Vec<AmdRegister>,
            &Vec<AmdRegister>,
        ),
    >::new();
    for record in graph.amd_events() {
        match &record.kind {
            AmdEventKind::Instruction { .. } => {
                last_instruction.insert(record.wave, record.event);
            }
            AmdEventKind::WmmaIssue {
                kind,
                destinations,
                sources_ab,
                accumulator,
                scales,
                index,
            } => match last_instruction.get(&record.wave) {
                Some(&instruction)
                    if graph.event(instruction).pc == graph.event(record.event).pc =>
                {
                    let ab = sources_ab.iter().chain(index).copied().collect();
                    issues.insert(instruction, (*kind, destinations, ab, accumulator, scales));
                }
                _ => report.malformed.push(format!(
                    "WMMA issue e{} without its instruction record",
                    record.event.0
                )),
            },
            _ => {}
        }
    }
    report.wmmas = issues.len();
    let mut open = BTreeMap::<u32, Vec<Open>>::new();
    for record in graph.amd_events() {
        let AmdEventKind::Instruction {
            reads,
            writes,
            exec_mask,
            ..
        } = &record.kind
        else {
            continue;
        };
        // Types profile: an instruction issued with EXEC = 0 is never a slot.
        let exec0 = types && *exec_mask == 0;
        let Some(kind) = graph
            .event(record.event)
            .pc
            .and_then(|pc| graph.provenance().instructions.get(pc.0 as usize))
            .map(|info| info.kind)
        else {
            report.malformed.push(format!(
                "instruction e{} without provenance",
                record.event.0
            ));
            continue;
        };
        let wave = open.entry(record.wave).or_default();
        let mut found = |kind: WmmaHazardKind, first: &Open, register, required| {
            // Short of the count, but met if EXEC = 0 instructions counted.
            let kind = if first.slots < required && first.slots + first.exec0 >= required {
                WmmaHazardKind::Exec0SlotsUnresolved
            } else {
                kind
            };
            let finding = WmmaHazardFinding {
                kind,
                wave: record.wave,
                first: first.event,
                second: record.event,
                register,
                slots: first.slots,
                required,
                exec0: first.exec0,
            };
            if kind.policy() {
                report.policy.push(finding);
            } else {
                report.violations.push(finding);
            }
        };
        let wmma = issues.get(&record.event);
        if let Some(&(wmma_kind, d2, ref ab2, c2, scales2)) = wmma {
            if types && type_slots(wmma_kind).is_none() {
                report.malformed.push(format!(
                    "WMMA e{} is outside the type table of this profile",
                    record.event.0
                ));
            } else if !types
                && !matches!(
                    wmma_kind,
                    AmdWmmaKind::F16_16x16x32 | AmdWmmaKind::Bf16_16x16x32
                )
            {
                report.malformed.push(format!(
                    "WMMA e{} is outside the F16/BF16 table of this profile",
                    record.event.0
                ));
            }
            for first in wave.iter() {
                if first.valu_since == 0 && first.d == *d2 && first.ab == *ab2 && first.c == *c2 {
                    // Only EXEC = 0 instructions between: not back-to-back in
                    // the instruction stream, not shown to be separated.
                    let kind = if first.exec0 > 0 {
                        WmmaHazardKind::Exec0SlotsUnresolved
                    } else {
                        WmmaHazardKind::SameArgumentsBackToBack
                    };
                    found(kind, first, None, 1);
                }
                if first.slots >= first.xdl {
                    continue;
                }
                let sources: Vec<_> = first
                    .ab
                    .iter()
                    .chain(&first.c)
                    .chain(&first.scales)
                    .copied()
                    .collect();
                for (kind, register) in [
                    (
                        WmmaHazardKind::MatrixABReadBeforeReady,
                        first_overlap(&first.d, ab2),
                    ),
                    (
                        WmmaHazardKind::ScaleSourceBelowPolicy,
                        first_overlap(&first.d, scales2),
                    ),
                    (
                        WmmaHazardKind::UnlistedAccumulatorChain,
                        first_overlap(&first.d, c2),
                    ),
                    (
                        WmmaHazardKind::UnlistedDestinationReuse,
                        first_overlap(&first.d, d2),
                    ),
                    (
                        WmmaHazardKind::UnlistedSourceReuse,
                        first_overlap(&sources, d2),
                    ),
                ] {
                    if register.is_some() {
                        found(kind, first, register, first.xdl);
                    }
                }
            }
        } else if matches!(kind, "vector ALU" | "trans ALU" | "shifted ALU") {
            for first in wave.iter().filter(|first| first.slots < first.valu) {
                let sources: Vec<_> = first.ab.iter().chain(&first.c).copied().collect();
                for (kind, register) in [
                    (
                        WmmaHazardKind::ValuReadBeforeReady,
                        first_overlap(&first.d, reads),
                    ),
                    (
                        WmmaHazardKind::ValuDestinationReuse,
                        first_overlap(&first.d, writes),
                    ),
                    (
                        WmmaHazardKind::ValuSourceReuse,
                        first_overlap(&sources, writes),
                    ),
                    (
                        WmmaHazardKind::ScaleSourceBelowPolicy,
                        first_overlap(&first.scales, writes),
                    ),
                ] {
                    if register.is_some() {
                        found(kind, first, register, first.valu);
                    }
                }
            }
        }
        let vector = wmma.is_some() || matches!(kind, "vector ALU" | "trans ALU" | "shifted ALU");
        for first in wave.iter_mut() {
            if vector && !exec0 {
                first.valu_since += 1;
            }
            if wmma.is_none()
                && kind == "vector ALU"
                && !reads.iter().chain(writes).any(|r| first.touches(*r))
            {
                if exec0 {
                    first.exec0 += 1;
                } else {
                    first.slots += 1;
                }
            }
        }
        wave.retain(|first| first.slots < first.xdl);
        if let Some(&(wmma_kind, d, ref ab, c, scales)) = wmma {
            let (xdl, valu) = if types {
                type_slots(wmma_kind).unwrap_or((XDL_SLOTS, VALU_SLOTS))
            } else {
                (XDL_SLOTS, VALU_SLOTS)
            };
            wave.push(Open {
                event: record.event,
                d: d.clone(),
                ab: ab.clone(),
                c: c.clone(),
                scales: scales.clone(),
                slots: 0,
                valu_since: 0,
                xdl,
                valu,
                exec0: 0,
            });
        }
    }
    report
}

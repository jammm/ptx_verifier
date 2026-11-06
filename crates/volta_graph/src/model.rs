//! The graph's node, edge and metadata types.

use volta_expr::ExprId;

use crate::ids::{
    AccessId, AsyncOpId, EventId, GenerationId, ProgramPoint, SyncObjectId, ThreadIx, ValueId,
};

// =========================================================================
// Events
// =========================================================================

/// Where an event happens. Only `Thread` events are in a program order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecContext {
    /// Issued by (or performed synchronously for) this thread, in its program order.
    Thread(ThreadIx),
    /// An effect or the completion of an asynchronous operation.
    Async(AsyncOpId),
    /// A synchronization generation's completion.
    Generation(GenerationId),
}

/// One event of the candidate trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub ctx: ExecContext,
    pub kind: EventKind,
    /// The instruction the event belongs to; `None` for generation completions.
    pub pc: Option<ProgramPoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// A native AMD wave event; indexes [`crate::EventGraph::amd_events`].
    Amd(u32),
    /// A memory effect: synchronous in a `Thread` context, an async op's in an `Async` one.
    Access(AccessId),
    /// The thread contributes to a generation; how is the generation's [`Contribution`].
    Contribute(GenerationId),
    /// The thread starts a wait on an mbarrier phase of parity `parity`, contributing nothing.
    WaitStart {
        object: SyncObjectId,
        parity: Parity,
    },
    /// The thread is released by a completed generation.
    Resume(GenerationId),
    /// A generation completes (a `Generation` context).
    GenerationComplete(GenerationId),
    /// The thread issues an asynchronous operation; the op's effects are `Async` events.
    AsyncIssue(AsyncOpId),
    /// An asynchronous operation completes (an `Async` context).
    AsyncComplete(AsyncOpId),
    /// The thread seals its uncommitted ops of `family` into a new group (`commit_group`).
    GroupCommit(GroupFamily),
    Fence(FenceKind),
    /// A synchronization object's lifetime begins (`mbarrier.init`).
    ObjectInit(SyncObjectId),
    /// A synchronization object's lifetime ends (`mbarrier.inval`).
    ObjectInval(SyncObjectId),
    /// Tensor Memory columns `[start_col, start_col + num_cols)` become allocated.
    TmemAlloc {
        start_col: u16,
        num_cols: u16,
    },
    /// Tensor Memory columns `[start_col, start_col + num_cols)` are freed.
    TmemDealloc {
        start_col: u16,
        num_cols: u16,
    },
    /// The CTA gives up its right to allocate Tensor Memory (`tcgen05.relinquish_alloc_permit`).
    TmemRelinquish,
    /// The thread exits; nothing follows it in its program order.
    Exit,
}

/// The commit-group mechanism a `GroupCommit` belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupFamily {
    /// `cp.async.commit_group`, waited on by `cp.async.wait_group`.
    CpAsync,
    /// `wgmma.commit_group`, waited on by `wgmma.wait_group`.
    Wgmma,
    /// `cp.async.bulk.commit_group`, waited on by `cp.async.bulk.wait_group`.
    Bulk,
}

/// An mbarrier phase parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Parity {
    Even,
    Odd,
}

impl Parity {
    pub fn of_phase(index: u32) -> Parity {
        if index.is_multiple_of(2) {
            Parity::Even
        } else {
            Parity::Odd
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceKind {
    /// `fence.proxy.async`: a generic/async proxy fence over `space` (`None` = every space).
    ProxyAsync { space: Option<Space> },
    /// `tcgen05.fence::before_thread_sync`: the thread's prior async `tcgen05` ops, before its
    /// next thread synchronization.
    Tcgen05BeforeThreadSync,
    /// `tcgen05.fence::after_thread_sync`: the thread's prior thread synchronization, before its
    /// later async `tcgen05` ops.
    Tcgen05AfterThreadSync,
    /// `fence.proxy.tensormap::generic.release`, or the release `tensormap.cp_fenceproxy` ends
    /// with: the thread's prior generic-proxy writes, before an acquire ordered after it.
    TensormapRelease,
    /// `fence.proxy.tensormap::generic.acquire [addr], 128`: a release ordered before it, before
    /// the thread's later tensor-map-proxy reads of the tensor map at global `addr`.
    TensormapAcquire { addr: u64 },
}

// =========================================================================
// Memory effects
// =========================================================================

/// A memory space with cross-thread effects (local memory is thread-private).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Space {
    Global,
    Shared,
    /// Tensor Memory, addressed by (lane, column) cells of 32 bits.
    Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessMode {
    Read,
    Write,
}

/// The memory-model kind of an effect. Selected native AMD RMW profiles record
/// `Atomic`; consumers must validate that profile before applying its rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderingKind {
    Ordinary,
    Atomic,
    Reduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Proxy {
    Generic,
    Async,
    /// The tensor-map proxy, through which `cp.async.bulk.tensor` reads its tensor map.
    TensorMap,
}

/// The canonical physical locations an effect touches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Footprint {
    /// Bytes `[addr, addr + width)` of a byte-addressed space; never wraps.
    Bytes { addr: u64, width: u32 },
    /// Tensor Memory lanes `[lane, lane + lanes)` x columns `[col, col + cols)`.
    Tensor {
        lane: u8,
        lanes: u8,
        col: u16,
        cols: u16,
    },
}

impl Footprint {
    /// Whether two footprints share a location (callers compare effects of one space).
    pub fn overlaps(&self, other: &Footprint) -> bool {
        match (*self, *other) {
            (Footprint::Bytes { addr: a, width: aw }, Footprint::Bytes { addr: b, width: bw }) => {
                let (a, b) = (u128::from(a), u128::from(b));
                a < b + u128::from(bw) && b < a + u128::from(aw)
            }
            (
                Footprint::Tensor {
                    lane: al,
                    lanes: an,
                    col: ac,
                    cols: acn,
                },
                Footprint::Tensor {
                    lane: bl,
                    lanes: bn,
                    col: bc,
                    cols: bcn,
                },
            ) => {
                let lanes = u32::from(al) < u32::from(bl) + u32::from(bn)
                    && u32::from(bl) < u32::from(al) + u32::from(an);
                let cols = u32::from(ac) < u32::from(bc) + u32::from(bcn)
                    && u32::from(bc) < u32::from(ac) + u32::from(acn);
                lanes && cols
            }
            _ => false,
        }
    }
}

/// Which operand of its instruction an effect is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperandRole {
    /// The memory operand of an ordinary load/store or warp-cooperative access.
    Plain,
    /// An async copy's source read.
    Source,
    /// An async copy's destination write.
    Destination,
    /// A matrix-multiply `A` operand read (shared memory or Tensor Memory).
    MmaA,
    /// A matrix-multiply `B` operand read.
    MmaB,
    /// A matrix-multiply accumulator (`D`) read or write in Tensor Memory.
    Accumulator,
    /// A block-scaled matrix multiply's `scale_A` read in Tensor Memory.
    MmaScaleA,
    /// A block-scaled matrix multiply's `scale_B` read in Tensor Memory.
    MmaScaleB,
    /// The Tensor Memory cells of a `tcgen05.ld`/`.st`.
    TmemData,
    /// The 128-byte tensor map a `cp.async.bulk.tensor` reads.
    TensorMap,
}

/// A value stored by a write, as expressions of the producer's arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredValue {
    Scalar(ExprId),
    /// Two packed lanes (low first).
    Pair(ExprId, ExprId),
    /// Four packed byte lanes (lowest first).
    Quad([ExprId; 4]),
    /// A value without an expression form (e.g. an mbarrier handle).
    Opaque,
}

/// One memory effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemAccess {
    pub space: Space,
    pub mode: AccessMode,
    pub ordering: OrderingKind,
    pub proxy: Proxy,
    pub footprint: Footprint,
    pub role: OperandRole,
    /// The thread the effect is attributed to: the issuer, or the lane a collective op
    /// performs it for.
    pub attributed: ThreadIx,
    /// The stored value, for writes.
    pub value: Option<ValueId>,
}

// =========================================================================
// Synchronization objects and generations
// =========================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncObjectKind {
    /// One of the CTA's hardware named barriers (`bar.sync`, `bar.arrive`); an AMD
    /// workgroup's one `s_barrier` is barrier 0.
    NamedBarrier { id: u8 },
    /// One lifetime (`init` to `inval`) of an mbarrier object at a shared address; a
    /// re-initialization is a new object.
    Mbarrier { addr: u64, expected_arrivals: u64 },
    /// The warp-cooperative op instances of one warp.
    WarpCollective { warp: u32 },
    /// The warpgroup-cooperative op instances (`wgmma.mma_async`) of one warpgroup.
    WarpgroupCollective { warpgroup: u32 },
    /// One thread's `cp.async.wait_group` completions; an AMD wave's `s_waitcnt lgkmcnt`
    /// completions of its DS instructions.
    AsyncGroupWait { thread: ThreadIx },
    /// One thread's `cp.async.bulk.wait_group` completions.
    BulkGroupWait { thread: ThreadIx },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncObject {
    pub kind: SyncObjectKind,
    pub init: Option<EventId>,
    pub inval: Option<EventId>,
}

/// How many arrivals complete a named barrier's generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BarrierArrivals {
    /// Every live thread of the CTA (`bar.sync a`); exited threads count as arrived.
    FullCta,
    /// Exactly this many threads (`bar.sync a, b`, `bar.arrive a, b`).
    Count(u32),
}

/// Which end of a collective instance a generation is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollectiveSide {
    /// Members arrive before the op's effects.
    Entry,
    /// Members leave after the op's effects.
    Exit,
}

/// Whether a collective orders its members' memory effects with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemberOrdering {
    /// `bar.warp.sync`, which "guarantee\[s\] memory ordering among threads participating in
    /// barrier" (PTX ISA 9.7.15.2): each member's contribution joins.
    Synchronizing,
    /// Every other collective (`shfl.sync`, `ldmatrix`, `tcgen05.alloc`, ...): the members
    /// only wait for each other, so their contributions join nothing.
    ExecutionOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationKind {
    Barrier {
        arrivals: BarrierArrivals,
    },
    /// The phase whose index is the generation's `index`.
    MbarrierPhase,
    /// The phase before an mbarrier's phase 0, at index 0: complete at init, with no
    /// contributions. An odd-parity wait on a fresh object returns through it, the PTX ISA's
    /// "immediately preceding phase".
    MbarrierPreInit,
    /// One side of a collective instance over the `mask` lanes, at `pc`.
    Collective {
        pc: ProgramPoint,
        mask: u32,
        side: CollectiveSide,
        ordering: MemberOrdering,
    },
    /// A `cp.async.wait_group n` (or `wait_all`, `n = 0`) completing the groups past `n`; an
    /// AMD `s_waitcnt lgkmcnt(n)` completing the DS instructions past its `n` most recent.
    AsyncGroupWait {
        keep_pending: u32,
    },
    /// A `cp.async.bulk.wait_group{.read} n` completing the bulk groups past `n`.
    BulkGroupWait {
        keep_pending: u32,
        scope: BulkWaitScope,
    },
}

/// What a `cp.async.bulk.wait_group` waits for: with `.read`, only that the
/// groups' source reads are done (PTX ISA 9.7.9.25.4.3), not their writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BulkWaitScope {
    SourceReads,
    Complete,
}

/// How a contribution takes part in its generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContributionKind {
    /// The thread waits at this generation until it completes (`bar.sync`, a collective).
    Blocking,
    /// A non-blocking arrival of `count` (`bar.arrive`, `mbarrier.arrive`,
    /// `cp.async.mbarrier.arrive`, `tcgen05.commit`), raising the expected transaction bytes by
    /// `expect_tx`.
    Arrival {
        count: u64,
        expect_tx: u64,
        ordering: ArrivalOrdering,
    },
    /// An exited thread (its `Exit` event), counted as arrived.
    Exited,
    /// An asynchronous operation's completion (its `AsyncComplete` event), with the
    /// transaction bytes it completes on an mbarrier.
    AsyncCompletion { op: AsyncOpId, tx_bytes: u64 },
    /// An `mbarrier.complete_tx` of `bytes`: `.relaxed`, so it orders nothing.
    CompleteTx { bytes: u64 },
    /// An `mbarrier.expect_tx` of `bytes`, an announcement with no arrival: `.relaxed`, so it
    /// orders nothing.
    ExpectTx { bytes: u64 },
}

/// What an arrival releases of the thread it is attributed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrivalOrdering {
    /// The thread's own arrive (`bar.arrive`, `mbarrier.arrive`): a release of everything
    /// before it in the thread's program order.
    Release,
    /// An arrive-on the system performs once the thread's tracked async operations complete
    /// (`cp.async.mbarrier.arrive`, `tcgen05.commit`): ordered with those operations' accesses
    /// alone (PTX ISA, asynchronous operations), so it releases nothing else of the thread.
    AsyncTracked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Contribution {
    pub event: EventId,
    pub kind: ContributionKind,
}

/// What the producer observed when a generation completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GenerationEvidence {
    /// Named barriers: arrivals beyond the count present when the generation fired.
    pub surplus_arrivals: u32,
    /// mbarrier phases: net outstanding transaction bytes at completion.
    pub pending_tx: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    pub object: SyncObjectId,
    /// The logical generation number within the object, from 0.
    pub index: u32,
    pub kind: GenerationKind,
    pub contributions: Vec<Contribution>,
    /// `Resume` events of the threads it released.
    pub releases: Vec<EventId>,
    pub completion: Option<EventId>,
    pub evidence: GenerationEvidence,
}

// =========================================================================
// Asynchronous operations
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AsyncOpKind {
    /// `cp.async` (global to shared), completed by `wait_group` or `cp.async.mbarrier.arrive`.
    CpAsync,
    /// `cp.async.bulk` (global to shared), completed on an mbarrier.
    BulkCopy,
    /// `cp.async.bulk.tensor` (TMA), completed on an mbarrier.
    TensorCopy,
    /// A TMA store's (`cp.async.bulk.tensor` shared to global) shared-memory
    /// reads, completed by any `cp.async.bulk.wait_group` of its bulk group.
    TensorStoreRead,
    /// The same TMA store's global writes, completed only by a
    /// `cp.async.bulk.wait_group` without `.read`.
    TensorStoreWrite,
    /// A `clusterlaunchcontrol.try_cancel` response write, completed on an mbarrier.
    ClcResponse,
    /// `tcgen05.mma`, tracked by later `tcgen05.commit`s of its issuer.
    Tcgen05Mma,
    /// `tcgen05.ld`, completed by the warp's `tcgen05.wait::ld`.
    Tcgen05Ld,
    /// `tcgen05.st`, completed by the warp's `tcgen05.wait::st`.
    Tcgen05St,
    /// One member's share of a `wgmma.mma_async`; `instance` names the warpgroup op.
    WgmmaMma { instance: u32 },
    /// An AMD DS instruction's LDS accesses, completed by its wave's `s_waitcnt lgkmcnt`. The
    /// LDS performs one wave's DS instructions in issue order: each one's effects follow those
    /// of the wave's earlier ones, and its completion implies theirs.
    Ds,
}

impl AsyncOpKind {
    pub fn name(self) -> &'static str {
        match self {
            AsyncOpKind::CpAsync => "cp.async",
            AsyncOpKind::BulkCopy => "cp.async.bulk",
            AsyncOpKind::TensorCopy => "cp.async.bulk.tensor",
            AsyncOpKind::TensorStoreRead => "cp.async.bulk.tensor (store, shared reads)",
            AsyncOpKind::TensorStoreWrite => "cp.async.bulk.tensor (store, global writes)",
            AsyncOpKind::ClcResponse => "clusterlaunchcontrol.try_cancel",
            AsyncOpKind::Tcgen05Mma => "tcgen05.mma",
            AsyncOpKind::Tcgen05Ld => "tcgen05.ld",
            AsyncOpKind::Tcgen05St => "tcgen05.st",
            AsyncOpKind::WgmmaMma { .. } => "wgmma.mma_async",
            AsyncOpKind::Ds => "ds",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsyncOp {
    pub kind: AsyncOpKind,
    pub issuer: ThreadIx,
    pub pc: ProgramPoint,
    /// The `AsyncIssue` event, in the issuer's program order.
    pub issue: EventId,
    pub effects: Vec<AccessId>,
    pub completion: Option<EventId>,
    /// Generations listing this op's completion as a contribution (tentative tracking).
    pub tracking: Vec<GenerationId>,
}

// =========================================================================
// Terminal state and provenance
// =========================================================================

/// How the candidate execution ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminal {
    /// Every thread exited.
    Completed,
    /// No thread could proceed; every live thread is listed with its wait condition.
    Stuck { blocked: Vec<BlockedThread> },
    /// The producer stopped the execution (unsupported or invalid behavior, a budget).
    Aborted {
        thread: Option<ThreadIx>,
        pc: Option<ProgramPoint>,
        cause: AbortCause,
        reason: String,
    },
}

/// Why the producer stopped an execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortCause {
    /// Behavior the producer does not model.
    Unsupported,
    /// Anything else: invalid behavior, a launch the producer cannot run, a budget.
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockedThread {
    pub thread: ThreadIx,
    pub pc: ProgramPoint,
    pub on: BlockedOn,
}

/// A blocked thread's wait condition, with the state that disables it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockedOn {
    /// An AMD scalar S_BARRIER rendezvous (wave-level generation, no PTX sync object).
    AmdBarrier {
        generation: u32,
        missing_waves: Vec<u32>,
    },
    AmdNamedBarrier {
        object: crate::AmdNamedBarrier,
        phase: u32,
        expected: u16,
        arrived: u16,
    },
    /// Waiting in a named barrier generation.
    Barrier { generation: GenerationId },
    /// Waiting at a collective instance's entry for the missing live lanes.
    Collective {
        generation: GenerationId,
        missing: Vec<ThreadIx>,
    },
    /// Waiting for an mbarrier phase of parity `parity` to complete.
    Mbarrier {
        object: SyncObjectId,
        parity: Parity,
        state: MbarrierState,
    },
}

/// An mbarrier's forming phase when the execution stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MbarrierState {
    pub completed_phases: u32,
    pub arrived: u64,
    pub expected: u64,
    pub pending_tx: i64,
}

/// A modeling assumption the producer executed under, recorded so verdicts can cite it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Assumption {
    /// Recognize the compiler's exact rounded f32 log2(e) multiply/exp2 idiom as exp.
    CompilerExp2Idiom,
    /// Base2 operations use the exact reciprocal of rounded f32 log2(e) as ln(2),
    /// matching the existing PTX mathematical abstraction.
    CompilerBase2Conversion,
    /// The analyzed AMD execution has XNACK replay disabled, including generic
    /// gfx950 code that can also be loaded on a replay-enabled system.
    XnackReplayDisabled,
    /// Exactly one AMD workgroup is analyzed; no statement about other workgroups.
    SingleWorkgroupAnalyzed,
    /// Every wave of the analyzed AMD workgroup resides on the same CU:
    /// threadgroup splitting is disabled.
    SingleCuWorkgroup,
    /// Workgroup clustering and cluster-dependent ABI fields are disabled.
    UnclusteredWorkgroup,
    /// Number of shader named barrier objects allocated by the native kernel
    /// descriptor (four times NAMED_BAR_CNT), not the number used by a trace.
    AllocatedNamedBarriers(u8),
    /// Native LDS object boundaries are supplied explicitly by configuration.
    ConfiguredSharedAllocationLayout,
    /// Symbolic arithmetic uses mathematical reals, not bit-exact floating-point.
    MathematicalRealArithmetic,
    /// Addresses, branch conditions and synchronization parameters are concrete: the
    /// kernel is a structured CTA under the launch configuration.
    StructuredCta,
    /// One CTA (block (0, 0, 0)) is analyzed; the others run the same code.
    SingleCtaAnalyzed,
    /// Every cluster holds exactly one CTA.
    SingleCtaCluster,
    /// No `clusterlaunchcontrol.try_cancel` request ever succeeds.
    NeverStealClusterLaunchControl,
    /// An exited thread counts as arrived at full-CTA barriers and warp collectives
    /// (the paper's Sync' rule).
    ExitedThreadsArrive,
    /// Asynchronous operations compute their data at issue; only their completion is
    /// asynchronous.
    EagerAsyncData,
    /// Static descriptor LDS plus configured dynamic LDS for native execution.
    /// A caller/frontend premise, not an inferred hardware allocation.
    ConfiguredLdsBytes(u64),
    /// Native AMD execution uses default scheduling; expert-mode transitions
    /// and its separate register dependency counters are outside this profile.
    AmdDefaultScheduling,
    /// The caller/frontend establishes a valid dispatch with this fixed number
    /// of backed private bytes per physical work item, consistent with the
    /// selected descriptor and metadata. Not inferred from an access or a frame.
    FixedPrivateAllocation(u32),
    /// Only shared-memory (LDS) accesses are recorded: global memory is not checked.
    SharedMemoryOnly,
    /// gfx1250 B0 m1 integer profile and its SCHED_MODE 0 dependency
    /// profile: S_MAX SCC is strict greater-than, as measured on hardware (the guide prints >=).
    Gfx1250B0M1Profile,
    /// gfx1250 B0 m2 MODE profile: MODE FP_ROUND/FP_DENORM state
    /// starts from the kernel descriptor and is changed only by the selected
    /// MODE instructions; no ALU result in the profile depends on MODE.
    Gfx1250B0M2ModeProfile,
    /// Global objects and forbidden guard regions are supplied by configuration.
    ConfiguredGlobalRegionLayout,
    /// gfx1250 B0 litmus profile: untyped raw-buffer accesses are range checked
    /// per DWORD against the documented rule, and configured objects start
    /// with caller-supplied exact words.
    Gfx1250B0LitmusProfile,
    /// gfx1250 B0 litmus/2 profile: a STATE_PRIV read returns only the named
    /// barrier ID and a clear completion bit, and lanes of one global store
    /// that write the same bytes are ordered by lane.
    Gfx1250B0LitmusBarrierProfile,
    /// gfx1250 B0 WMMA profile: a BF16 WMMA result element has exact bits
    /// only when every partial sum of its terms is exactly representable.
    Gfx1250B0WmmaProfile,
    /// gfx1250 B0 multiwave profile: XNACK replay in single-group mode (the
    /// wave-start mode), lane-ordered same-address global stores and opaque
    /// per-wave evidence reads.
    Gfx1250B0MultiwaveProfile,
    /// gfx1250 B0 communicating-waves profile: every wave of the selected
    /// workgroup shares one workgroup-coherent vector cache, so an ordered
    /// cross-wave GLOBAL conflict is visible at any memory scope.
    Gfx1250B0WorkgroupCoherentProfile,
}

impl Assumption {
    pub fn describe(self) -> &'static str {
        match self {
            Assumption::FixedPrivateAllocation(_) => {
                "the dispatch supplies the declared fixed private allocation for each physical work item; descriptor and metadata agree"
            }
            Assumption::Gfx1250B0M1Profile => {
                "gfx1250 B0 m1 integer profile (SCHED_MODE 0, strict-greater S_MAX SCC); other steppings and targets are outside this result"
            }
            Assumption::Gfx1250B0M2ModeProfile => {
                "gfx1250 B0 m2 MODE profile: entry FP_ROUND/FP_DENORM come from the kernel descriptor, only the selected MODE instructions change them, and no modeled ALU result depends on MODE"
            }
            Assumption::Gfx1250B0LitmusProfile => {
                "gfx1250 B0 litmus profile: untyped raw-buffer DWORDs are range checked by the documented rule (out-of-range loads read zero, stores write nothing); initial object words are launch premises"
            }
            Assumption::Gfx1250B0LitmusBarrierProfile => {
                "gfx1250 B0 litmus/2 profile: a STATE_PRIV read after S_WAIT_IDLE defines only NAMED_BARRIER_ID and a clear NAMED_BARRIER_COMPLETE, read only when the wave has consumed every completion of its joined barrier; a JOIN with an unconsumed completion is outside the profile; lanes of one global store that write the same bytes are ordered by lane"
            }
            Assumption::Gfx1250B0WmmaProfile => {
                "gfx1250 B0 WMMA profile: a BF16 WMMA result element is reported as exact binary32 bits only when its inputs are normal or zero and every partial sum of its terms, in any order, is an exactly representable normal binary32 value; other results stay mathematical reals without bits"
            }
            Assumption::Gfx1250B0MultiwaveProfile => {
                "gfx1250 B0 multiwave profile: every wave stays in the wave-start single-group XNACK replay mode (the first non-memory instruction after memory instructions waits for XCNT to drain), lanes of one global store that write the same bytes are ordered by lane, and shader-cycle and HW_ID1 reads are opaque"
            }
            Assumption::Gfx1250B0WorkgroupCoherentProfile => {
                "gfx1250 B0 communicating-waves profile: the waves of the selected workgroup share one vector-memory cache that is coherent at workgroup scope, so a GLOBAL store completed (STOREcnt) before a barrier is visible to a later GLOBAL load of another wave of the workgroup at any scope; nothing is claimed across workgroups"
            }
            Assumption::ConfiguredGlobalRegionLayout => {
                "global object and guard-region boundaries are caller-supplied launch premises"
            }
            Assumption::AmdDefaultScheduling => {
                "native AMD execution uses default scheduling; expert scheduling and mode transitions are outside this result"
            }
            Assumption::ConfiguredLdsBytes(_) => {
                "native LDS extent is the descriptor's static allocation plus configured dynamic bytes"
            }
            Assumption::ConfiguredSharedAllocationLayout => {
                "LDS object bounds use the explicitly configured named allocation layout; these source boundaries are caller-supplied premises"
            }
            Assumption::CompilerExp2Idiom => {
                "the compiler's exact rounded f32 log2(e) multiply/exp2 idiom is modeled as mathematical exp"
            }
            Assumption::CompilerBase2Conversion => {
                "general base2 exp/log use the exact reciprocal of rounded f32 log2(e) for ln(2), matching the PTX mathematical abstraction"
            }
            Assumption::XnackReplayDisabled => {
                "the analyzed AMD execution has XNACK memory replay disabled; replay-enabled executions are outside this result"
            }
            Assumption::SingleWorkgroupAnalyzed => {
                "only the selected AMD workgroup is analyzed; other workgroups and inter-workgroup conflicts are outside this result"
            }
            Assumption::SingleCuWorkgroup => {
                "all waves of the analyzed AMD workgroup share one CU; threadgroup splitting is disabled"
            }
            Assumption::UnclusteredWorkgroup => {
                "workgroup clustering is disabled; cluster-dependent execution and memory protocols are outside this result"
            }
            Assumption::AllocatedNamedBarriers(4) => {
                "the native kernel descriptor allocates four shader named barrier objects"
            }
            Assumption::AllocatedNamedBarriers(8) => {
                "the native kernel descriptor allocates eight shader named barrier objects"
            }
            Assumption::AllocatedNamedBarriers(12) => {
                "the native kernel descriptor allocates twelve shader named barrier objects"
            }
            Assumption::AllocatedNamedBarriers(16) => {
                "the native kernel descriptor allocates sixteen shader named barrier objects"
            }
            Assumption::AllocatedNamedBarriers(_) => {
                "the recorded native named barrier allocation is outside the supported descriptor profile"
            }
            Assumption::MathematicalRealArithmetic => {
                "arithmetic is modeled over mathematical reals, not bit-exact floating-point"
            }
            Assumption::StructuredCta => {
                "addresses, branch conditions and synchronization parameters are concrete"
            }
            Assumption::SingleCtaAnalyzed => {
                "CTA (0, 0, 0) is analyzed and stands for every CTA of the grid"
            }
            Assumption::SingleCtaCluster => "every cluster holds exactly one CTA",
            Assumption::NeverStealClusterLaunchControl => {
                "no clusterlaunchcontrol.try_cancel request succeeds"
            }
            Assumption::ExitedThreadsArrive => {
                "an exited thread counts as arrived at full-CTA barriers and warp collectives"
            }
            Assumption::EagerAsyncData => {
                "asynchronous operations compute their data at issue; only completion is \
                 asynchronous"
            }
            Assumption::SharedMemoryOnly => {
                "only shared-memory (LDS) accesses are recorded; global memory is not checked \
                 for races"
            }
        }
    }
}

/// What one graph thread ([`crate::ThreadIx`]) executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadUnit {
    /// One CTA thread, by its linearized index.
    CtaThread,
    /// One wave of `lanes` work-items issuing every instruction together: thread `w` is the
    /// wave of work-items `w * lanes` to `(w + 1) * lanes - 1`.
    Wave { lanes: u32 },
}

/// The launch, target and model the graph was produced under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    pub kernel: String,
    pub block_dim: (u32, u32, u32),
    pub unit: ThreadUnit,
    pub grid_dim: (u32, u32, u32),
    /// The analyzed block/workgroup coordinate; native AMD verifies only this group.
    pub workgroup_id: (u32, u32, u32),
    pub targets: Vec<String>,
    pub assumptions: Vec<Assumption>,
    /// Per [`ProgramPoint`]: the instruction kind and its source byte span.
    pub instructions: Vec<InstructionInfo>,
}

impl Provenance {
    /// The graph's threads: the block's work-items, grouped into waves for `ThreadUnit::Wave`.
    pub fn threads(&self) -> u32 {
        let work_items = self.block_dim.0 * self.block_dim.1 * self.block_dim.2;
        match self.unit {
            ThreadUnit::CtaThread => work_items,
            ThreadUnit::Wave { lanes } => work_items.div_ceil(lanes),
        }
    }

    pub fn assumes(&self, assumption: Assumption) -> bool {
        self.assumptions.contains(&assumption)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstructionInfo {
    pub kind: &'static str,
    pub span: Option<(usize, usize)>,
}

# volta_graph_check

Checks the event graph of one candidate execution without running symbolic
analysis or equivalence checking. `check_graph` validates the representation,
selects the target-specific properties, and returns typed findings and a
`GraphReport`. Its conclusion is `Verified`, `Rejected` or `Unknown`.
Summaries retain witness event IDs for source-linked diagnostics, including AMD
register hazards, barrier violations and unresolved trace obligations.

The graph's recorded trace order is not a synchronization proof. The checker
must validate tentative barrier, completion and memory associations before
using them to establish ordering. An incomplete execution cannot become
`Verified` merely because checks on its prefix pass.

Native AMD producers use a supported `gfx` target and its explicit trace
contract. An unnormalized `amdgcn-...` target is inconclusive rather than falling
through to PTX checks. A completed placeholder graph without AMD access events
and trace premises supplies no native safety proof.

The experimental `volta_amd_analysis` producer instead records a shared graph
with explicit wave64 units, a valid `amdgcn-amd-amdhsa--gfx950` target (including
recognized target features), `SharedMemoryOnly`, and no native sideband records.
This profile uses the generic graph checks for its LDS accesses, waits and
barriers; it does not establish global-memory safety. Only this wave-unit
profile exempts LDS effects of distinct DS instructions issued by the same
wave from memory conflicts. Cross-wave effects and overlapping writers within
one DS instruction remain checked. Physical-work-item/native AMD graphs retain
their target-specific rules. Missing or conflicting profile metadata stays
inconclusive.

`amdgpu::early_memory_conflicts` supplies rejection-only certificates for two
different active physical lanes' overlapping ordinary writes within one memory
issue on gfx950 or gfx1250. GLOBAL denotes the memory space: already-supported
concrete raw-buffer stores are included alongside GLOBAL and direct LDS/DS
stores. Atomics, reductions, tensor transfers, mixed FLAT operations and opaque
descriptor profiles cannot supply these certificates.

An incremental observer proposes at most one pair per issue. The validator
checks exact prefix identity and pair membership, then independently replays
the existing target-selected trace, readiness, barrier and memory checks. All
three prerequisite properties must pass; the exact pair and its first shared
byte must appear in the retained memory findings. A pair excluded by the
ordinary findings cap is refused. Discovery does not duplicate counter or
happens-before rules. Later waits cannot order the same issue's sibling effects.

The trusted producer calls the observer only after a complete successful
instruction batch and preflights physical addresses against its allocations.
Validation checks recorded roles, active-lane membership, that selected effects
precede the next AMD record, LDS limits when
recorded, and configured allocation layouts. It does not decode raw ISA,
independently recover absent GLOBAL/CDNA allocation extents, or infer every
machine instruction's component count from source spans. Only gfx1250 Shared
certificates require the existing recorded LDS-size premise. Unstarted waves
may be absent in a prefix; none receives a fabricated END or completion.
The ordinary graph conclusion for an observer-stopped artifact stays Unknown.
Only the pipeline's privately validated prefix result can report this rejection;
it supplies no outputs, equivalence proof or early acceptance. Default complete
execution and full diagnostic enumeration retain their existing behavior.

## PTX checks

The PTX graph branch of the pipeline runs ten checks over one `volta_graph::EventGraph`
that decide whether the recorded candidate execution is free of data races,
proxy and tensor-ordering hazards, and synchronization misuse. Seven are the
algorithms of the Volta v2 design's data-race freedom check (Algorithms 1-7);
three more enforce PTX ISA rules the design leaves out. The crate depends only
on `volta_graph`. It never runs symbolic execution or equivalence checking.

## High-level logic

```
EventGraph ──validate──► typed edge projections (G0, G_HB, D, E_pipeline)
                               │
                               ▼
     the ten checks, in prerequisite order, each returning
     Pass(evidence) | Fail(witnesses) | Unknown(unresolved)
                               │
                               ▼
     GraphReport::conclusion() = Verified | Rejected { failed, deadlock } | Unknown
```

The event graph stores raw, tentative facts: "this arrival joined generation
`g3`", "this TMA completion counted toward phase 2". A check may not trust a
tentative fact until an earlier check has validated it. So the checks form a
chain of **certificates**: a passing check returns a typed evidence value,
and only that value can build the stronger projection the next checks use.
The type system enforces the order. `ValidatedGraph` (`G_HB`) has no public
constructor except `from_certificates(named_epochs, mbarrier_epochs,
mbarrier_ordering)`. `assuming_epochs` exists for isolated tests.

## Building the ordering relation (`relation.rs`)

The graph's edges are grouped into typed sets and never used untyped:

```
E = E_PO ∪ E_causal ∪ E_sync ∪ E_pipeline
```

- **`E_PO`** program order: each thread's `program_order` sequence.
- **`E_causal`**: an async op's `Issue -> each effect -> Complete`. The
  issuer's program order continues from the issue, so nothing orders the
  effects before the issuer's *later* events.
- **`E_sync`**: each generation's joins (contribution -> completion) and
  releases (completion -> `Resume`).
- **`E_pipeline`** (`pipeline.rs`): implicitly pipelined `tcgen05` pairs (see
  below). No projection includes it; only the tensor-order check reads it.

Two **projections** select which `E_sync` joins count. `Admission::joins`
decides each contribution kind:

| contribution                                    | `G0` (execution order) | `G_HB` (memory order) |
|-------------------------------------------------|:----:|:----:|
| `Blocking`, `Arrival { Release }`               | yes  | yes  |
| `Arrival { AsyncTracked }` (`tcgen05.commit`, `cp.async.mbarrier.arrive`) | yes | no |
| `Exited` (Sync' only)                           | yes  | no   |
| `AsyncCompletion` into an mbarrier phase        | no (not yet validated) | yes |
| `AsyncCompletion` into a wait collective / wait-group | yes | yes |
| `CompleteTx`, `ExpectTx` (relaxed)              | no   | no   |
| members of an `ExecutionOnly` collective        | no   | no   |

- **`IndependentGraph` (`G0`)** is what the barrier-epoch and mbarrier-ordering
  checks run on. It holds every join this execution produced except the mbarrier
  phase completion joins, which those checks are still validating. In `G0` an
  async completion also carries what preceded its op's issue.
- **`ValidatedGraph` (`G_HB`)** adds the validated mbarrier completion joins
  and follows the PTX memory model for async ops: an async arrive-on or
  completion is ordered only with *that op's own* memory effects, not with the
  issuer's earlier accesses. So it joins nothing of the thread, and a completion
  carries only its op's effects. An exited thread is also not a barrier
  participant here. `G_HB` is also Algorithm 5's `D`;
  `ValidatedGraph::successors()` returns its explicit successor lists.

### Reachability by vector-clock sweep

Every edge runs forward in trace order (checked by `EventGraph::validate`), so
trace order is a topological order. `sweep(graph, admission, visit)` walks the
events once in trace order, keeping one vector clock per thread
(`clock[t] = index of the last event of t that reaches here`):

- a thread event increments its own entry;
- a joining `Contribute` merges its thread's clock into the generation's clock;
- a `GenerationComplete` merges the admitted exited threads and (in `G0`) the
  issue clocks of the async ops it tracks;
- a `Resume` merges the generation's clock into the thread's;
- an async event's clock is its op's issue clock.

At each event the callback gets an `HbView`. `thread_hb(ThreadPoint)` answers
"does thread event `(t, i)` reach here" as `clock[t] >= i`.
`effects_hb(op)` answers it for an async op's effects. Every path out of an
effect goes through the op's completion and then a release of a generation that
joins it, so the effects reach the visited event iff one of those **observers**
(`Resume` events) does.

## The checks

`suite::check_graph(&graph)` runs them in `Property::ALL` order. A check whose
prerequisites did not pass is `Unknown`. No check turns an unsupported case into
`Pass`.

| #  | `Property`             | Module                    | Runs on | Question it answers |
|----|------------------------|---------------------------|---------|---------------------|
| 1  | `NamedBarrierEpoch`    | `barrier_epoch.rs`        | `G0`    | Is generation `k` of each named barrier separated from `k+1`? Every synchronization of `k` must reach the program predecessor of every contribution of `k+1` (Weft's criterion). |
| 2  | `MbarrierEpoch`        | `barrier_epoch.rs`        | `G0`    | The same over mbarrier phases (pre-init = phase `-1`). A wait of phase `k+1` instead needs one release of `k` before it. |
| 3  | `MbarrierOrdering`     | `mbarrier_ordering.rs`    | `G0`    | Does each phase's arrival/transaction count balance (`D = n_k`, `T = 0`), and is every `expect_tx` announcement ordered after an arrival of its phase? |
| 4  | `MemoryConflict`       | `memory_conflict.rs`      | `G_HB`  | Is every overlapping pair of effects in domain `M` (global/shared) with a write ordered? A per-byte dominance forest reports each unordered pair once. |
| 5  | `ProxyOrder`           | `proxy_order.rs`          | `G_HB`  | Does every ordered generic/async cross-proxy pair have a covering fence (`fence.proxy.async`, or the implicit fence of an observed completion)? Does each tensor-map write reach a TMA's map read through a release/acquire pair? |
| 6  | `TensorOrder`          | `tensor_order.rs`, `sync_search.rs` | `D`, `E_pipeline` | Is every conflicting Tensor Memory pair of distinct ops pipelined, or connected by a `D` path through a source-thread `before_thread_sync` (or tracking commit) and a target-thread `after_thread_sync`? (Algorithm 6, a 3-state DFS.) |
| 7  | `ResourceLifetime`     | `resource_lifetime.rs`    | `G_HB`  | Is every TMEM use inside an allocation that is freed exactly once, with alloc -> use -> dealloc ordered? |
| 8  | `AllocationOrder`      | `allocation_order.rs`     | `G_HB`  | Are the `tcgen05.alloc`s totally ordered, with non-increasing column counts? |
| 9  | `AllocationPermit`     | `allocation_permit.rs`    | `G_HB`  | Does every allocation reach every `relinquish_alloc_permit`? |
| 10 | `MbarrierReleaseOrder` | `mbarrier_release_order.rs` | `G0`  | Do each mbarrier's waits observe its phases in sequence 0, 1, 2, ... (optionally from pre-init), with each phase's releases reaching the next phase's? |

Prerequisites (`Property::prerequisites`): 1, 2, 3 and 10 need nothing; 4 and
6-9 need 1, 2, 3 (to build `G_HB`); 5 also needs 4.

### Supporting modules

- `domain.rs`: the memory domains obligations are split by. `M` is ordinary
  global/shared effects, sync or async, including `tcgen05` shared-memory
  operands. `A` is Tensor Memory effects of `tcgen05` ops. Atomics are in
  neither; they are outside the contract.
- `pipeline.rs`: `E_pipeline`, the pairs the PTX ISA (9.7.18.6.2) pipelines
  without explicit sync. These are same-thread `tcgen05.mma` chains on one
  accumulator, and same-warp `st->ld`, `st->st`, `ld->st` pairs separated by
  the earlier op's own `tcgen05.wait`. Each edge keeps its rule, roles and
  covered cells. `pipelinable` is the same pairing without the one-thread and
  wait preconditions; `sync_search` uses it to choose where the search starts.
- `sync_search.rs`: Algorithm 6. A DFS over `D` from the source op's issue
  (pipelinable pairs) or completion (all others), with state
  `0 -before fence-> 1 -after fence-> 2`. It accepts only at the target's issue
  in state 2, and memoizes `(vertex, state)`.
- `result.rs`: `PropertyResult<E, V> = Pass(E) | Fail(Vec<V>) |
  Unknown(Vec<Unresolved>)`, `Property`, `Verdict`.
- `suite.rs`: `check_graph`, `GraphReport` (per-check results and summaries),
  `GraphConclusion`, `Deadlock` (from a `Terminal::Stuck`).

## Conclusion

- **`Verified`**: the trace completed and every applicable check passed: ten
  PTX properties or four native AMD properties. Inapplicable PTX fields in an
  AMD report do not participate in that conclusion.
- **`Rejected { failed, deadlock }`**: some check failed (with its typed
  witnesses), or the trace is stuck (a deadlock).
- **`Unknown`**: nothing failed, but some obligation is unresolved or the trace
  was aborted. The checks still run on a partial graph.

## Tests

`tests/` holds graph-level unit tests built with `GraphBuilder`. End-to-end PTX
tests for each check live in `volta_pipeline/tests/<check>_tests.rs` and assert on
the check's typed witness.

## Native AMD checks

For supported `gfx942`, `gfx950` and `gfx1250` targets, `check_graph` selects four AMD properties:
trace completeness, wait/register readiness, workgroup barrier generations, and
memory-conflict freedom. `Property::ALL` retains the ten PTX properties for API
compatibility; consumers must enumerate `report.applicable_properties()` or
`report.summaries()`. `report.amd` retains typed register and barrier witnesses.
Empty/missing AMD instrumentation, malformed graphs, conflicting targets, foreign
events, unsupported memory effects, and missing instruction/issue associations
cannot produce `Verified`. An aborted prefix remains incomplete even if its checked
obligations pass.

The separate [tensor-bounds prefix validator](../../docs/tensor-bounds.md)
can certify an invalid attempted gfx1250 transfer before memory issue. It
checks the supported descriptor, configured LDS extent and prefix premises;
it does not upgrade the ordinary graph conclusion or establish complete outputs.

### gfx942 and gfx950

AMD ordering is independent of PTX certificates. The following VM/LGKM/MFMA
profile applies to gfx942/gfx950; see the separate gfx1250 section for its
wave32 counters, replay and WMMA rules. Ordinary VM and LDS FIFO pipelines, waits
that guarantee completion, and validated workgroup rendezvous create completed-effect
frontiers. Candidate trace order and a barrier alone do not complete memory operations.
The checker reuses the existing byte-overlap domain and conflict forest, with clocks
bounded by wave count and instruction class. SMEM charges one LGKM count for a dword,
two for wider loads; FLAT retires only after both VM and LGKM charges clear. Nonzero
LGKM waits do not assume FIFO for scalar/mixed classes. Cross-lane DS permutations
share the LDS completion FIFO while recording no LDS memory footprint. Register reads, pending destination overwrites and held
store-source overwrites have distinct witnesses, respecting physical lane slices.
Ordinary DS, cross-lane, and narrow store sources use hardware operand handling;
they are not held until a counter wait. Three/four-dword Global/Flat store data
requires two independent states before a VALU overwrite (CDNA4 Table 11).
VM/LDS destination writes occur on asynchronous return and use hardware dependency
handling; their issue events do not act as immediate overwrites. A trans-VALU result similarly needs one independent
state before a non-trans VALU consumes it. Minimal failing and corrected controls
cover these timing rules.

VALU-produced scalar registers (including VCC low/high) require two independent
states before ordinary VALU data use, and five before vector-memory data/address
use. Explicit carry/condition-mask roles require no software bubble; recording
those roles never bypasses asynchronous register readiness checks. Later scalar
writes replace earlier producers. Scalar-mask roles must be actual scalar reads
of the corresponding instruction.
VALU writes of EXEC require four states before MFMA, including writes that leave
the active mask unchanged. MFMA traces must include both implicit EXEC reads.

Selected **gfx950** MUBUF-to-LDS DMA uses distinct `BufferLoadToLds32` and
`BufferLoadToLds128` classes. Each active physical lane records one or four
paired DWORD Global reads and Shared writes, with no register destination and
exactly one VM charge. The contract checks the descriptor-quad/offset/M0 source
roles, initialized M0, complete active-lane/component coverage, a common
DWORD-aligned M0 base at most `0xffff`, and whole LDS ranges within the declared
extent and any configured subobject. The final lane contribution does not wrap
at 64 KiB. Address/descriptor decoding and complete-instruction publication
remain trusted producer premises; these checks do not reconstruct raw ISA.

DMA shares the ordinary counter-wide pending VM inventory but has a separate,
initially unordered completion domain. Mixed or DMA-only partial VM waits retire
no contributor; ordinary-only FIFO behavior stays unchanged. VM0 completes DMA;
LGKM/EXP waits and barriers alone do not. FLAT still needs both remaining
charges. The model snapshots address, resource and M0 operands at issue, with their
normal producer-readiness checks and no invented EXP/X source lease.

A SALU write to M0 needs **one independent state** before an active DMA reads it,
including a scalar write performed under EXEC0. This pins LLVM
[`checkReadM0Hazards`](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp#L1938-L1943)
and the [GFX9-only DMA hazard predicate](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/GCNSubtarget.h#L566-L568).
An intervening instruction or `s_nop 0` supplies that state; the DMA itself does
not. Missing spacing reports `SaluM0ReadBeforeReady`. Concrete descriptor values,
resource OOB/suppression and the public M0-width discrepancy retain the native
profile's separate refusal boundaries. The [direct graph controls](tests/gfx950_buffer_lds.rs)
exercise both issue orders, three-contributor partial waits, mixed DMA widths,
FLAT counter orders, all-ones waits and cross-wave publication.

The MFMA timing model pins the dense, single-block F32 16x16x4/32x32x2,
F16/BF16 16x16x16/16x16x32/32x32x8/32x32x16, and OCP FP8
16x16x128/32x32x64 forms (including unit-scaled forms). SGEMM ordinary RAW/WAW
requires `passes + 2` independent states; XDL requires `passes + 4` on gfx950.
VALU operand production requires two states. A complete accumulator can forward
without a bubble when the pass counts match; partial accumulator overlap and matrix-to-A/B dependencies have
separate rules. XDL accumulator-source overwrites require `passes - 1` states.
S_NOP contributes its encoded immediate plus one and must have a pinned source
instruction. Other matrix forms, general DPP/special-register software hazards,
and global cache visibility policies are not inferred from this model.
Scoped cross-wave GLOBAL publication on gfx942/gfx950 requires completed effects,
workgroup scope at both endpoints, and the explicit `SingleCuWorkgroup` premise.
Unscoped publication remains inconclusive even when execution order is known;
conflicting unordered accesses still fail. No FLAT/SMEM/device/system visibility
protocol is inferred. This legacy cache certificate is rejected on gfx1250.
Disjoint global outputs and LDS communication use the ordinary native checks.

References: [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
(5-August-2025, sections 4.3–4.5, 7.6 Table 38, 8.3, 10.2 and 12.12), and LLVM
[`GCNHazardRecognizer.cpp`](https://github.com/llvm/llvm-project/blob/846473237377990d00b9c353f6a2c86116b52ea5/llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp)
and [`SIInstrInfo.cpp`](https://github.com/llvm/llvm-project/blob/846473237377990d00b9c353f6a2c86116b52ea5/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp). The former's `checkMAIHazards90A`
and `checkMAIVALUHazards` cross-check the per-form timing rules; the latter
counts S_NOP states. `SISchedule.td` pins the gfx950 pass counts, including the
FP8-dependent doubled pass counts for F8F6F4 instructions.

Focused regression suite: `cargo test -p volta_graph_check --test amdgpu_tests`.
The PTX tests remain in the same package and are required independently.

## gfx1250 checks

The target-selected wave32 checker validates trace completeness before readiness,
barrier-generation and memory-conflict checks. X replay-source leases are
separate from completion counters. Shader split signals publish completed
effects; missing data waits remain hazards. WMMA operand roles and dependencies
use gfx1250 layouts. Missing waves, wrong masks/register classes, forged memory
associations and incomplete traces stay inconclusive or fail their typed
obligation. Advanced barrier/cache/tensor modes outside the documented subset
cannot inherit gfx942/gfx950 certificates. Extending a graph profile does not
establish a hardware result. Numerical agreement remains separate from these
graph obligations.

`s_wait_alu` events are claimed by their instruction and change no counter.
GLOBAL prefetches join the VMEM replay group: their X lease holds every address
source, so an overwrite before X retires is a `SourceReuse` hazard; they must
have exactly one footprint listing the active lanes and no effects. When a
graph carries a global region layout, every completed GLOBAL effect and every
prefetch footprint is checked against it (guards for every scope, objects for
non-speculative scopes). `global_bounds` validates an attempted-access fault
against the layout and replays the prefix (trace, readiness, reachability)
before the pipeline certifies a guard, out-of-object or non-speculative
prefetch rejection with the lowest offending lane.
The gfx1250 trace check accepts a `ModeAccess` only under the B0 m2 profile,
right after its own scalar instruction record. `mode_order` reports a MODE read
followed by any MODE write without an intervening SA_SDST==0 `s_wait_alu` as
unresolved: that ordering is outside the profile. A `BufferRangeCheck` is accepted only under the B0 litmus profile,
once per buffer issue and for exactly its active lanes; the check recomputes
every mask and requires effects for exactly the in-range DWORDs. Under that
profile, lanes of one range-checked buffer store that write the same bytes are
ordered by lane (the higher lane's data is written last). Under the B0
multiwave profile the same order applies to every GLOBAL store, and each wave's
first non-memory instruction after memory instructions drains XCNT
(single-group replay). `wmma_hazards` (WMMA and multiwave profiles) checks
F16/BF16 WMMA spacing in slots: documented WMMA-to-WMMA A/B and WMMA-to-VALU
dependencies and same-operand back-to-back WMMA fail with witnesses; WMMA
dependencies outside the published table (D as C, WAW, WAR) below 5 slots are
unresolved.

Per-lane async GLOBAL/LDS transfers charge ASYNCCNT alone and require an explicit
`AmdDefaultScheduling` premise. The model snapshots addresses at issue; ordinary
instruction reads still require ready producers, but later ALU overwrites do not
need Load/X completion. A matching async wait completes the transfer; Load,
Store, DS and X waits alone do not. Partial async waits cannot identify a FIFO
prefix across mixed load/store completion streams. The pinned LLVM
[async encoding](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/FLATInstructions.td#L393-L446),
[VMEM classification](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L4765)
and [event selection](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/AMDGPUHWEvents.cpp#L119-L139)
establish this distinction even with XNACK enabled. Expert scheduling and its
separate register dependency counters remain unsupported.

For ordinary gfx1250 GLOBAL and resolved-global FLAT accesses, explicit
`SCOPE_SYS` at both endpoints supplies cache visibility under the pinned
[LLVM GFX125x memory model](https://github.com/ROCm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/docs/AMDGPUUsage.md#memory-model-gfx125x).
Completed effects and a validated publication path remain required. Missing
either scope stays inconclusive; a missing completion or rendezvous remains a
conflict. The target-specific annotation cannot certify legacy AMD targets,
scalar caches, LDS, atomics, tensor effects or another workgroup.
A completed Global read that happens before a later overwrite needs no cache
visibility from that future write. Earlier writes hidden beneath read nodes in
the overlap forest still require the original visibility contract.
The logical launch grid is preserved in provenance. A grid larger than one
requires `SingleWorkgroupAnalyzed`: the result covers only the selected workgroup,
including its launch-coordinate inputs, and makes no claim about other groups.
The unclustered premise and full selected-block trace remain required.

Tensor descriptor SGPRs are captured by the issuing instruction. They remain
ordinary instruction reads, so descriptor values returned by an unfinished KM
load fail readiness at issue. The Tensor memory issue has no completion-held
register sources; a later SGPR overwrite does not complete its memory effects.
Tensor waits still govern LDS/global buffer lifetime, and completed effects
still require publication before another wave consumes them. Load, DS and X
waits cannot replace Tensor completion. The selected contract follows pinned
LLVM's [TDM operand definitions](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/MIMGInstructions.td#L2291),
[Tensor-only event classification](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/AMDGPUHWEvents.cpp#L140)
and [register score handling](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp#L985).
Independent graph tests cover descriptor overwrite, pending-KM inputs, LDS
source/destination reuse, publication and malformed held-source records.

An exiting wave drops its automatic workgroup-barrier participation, so an
otherwise independent surviving wave can still finish. The pinned LLVM
[execution-barrier description](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/docs/AMDGPUExecutionSynchronization.md)
distinguishes drops from arrivals, and the
[memory model](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/docs/AMDGPUUsage.md#memory-model-gfx125x)
specifies arrive-release/wait-acquire ordering. These sources do not establish
that END's implicit drop publishes the exiting wave's completed LDS operations.
The checker therefore reports a typed inconclusive memory result only when a
conflicting pair needs that unpinned edge, including dependencies carried through
later phases. An explicit arrival publishes only its captured completed frontier;
later operations do not inherit that publication. Disjoint accesses, read/read
pairs and exit-only progress still verify, and unrelated established conflicts
still fail. This is a conservative proof limit, not an architectural race claim.

Genuine global FLAT accesses use `FlatGlobalLoad/Store`, with strictly matching
resolved Global effects and `flat memory` provenance. They
retain every source, including EXEC, and charge Load or Store plus Ds and X.
Both data counters must drain before a load destination becomes ready. A
same-wave store/load to the same address retains effect order even while a
separate conservative counter marker remains; this does not certify data
completion or cross-wave global cache publication. Global and LDS frontiers
are separate. `FlatLdsLoad/Store` require resolved Shared effects and the same
conservative dual data completion. They hold every source through their own
X token, but remain outside both scalar and GLOBAL replay groups: group changes
and later GLOBAL destination aliases cannot release their source leases.
An explicit own Load0/Store0 releases that operation's source token while its
independent Ds token remains. Explicit X0 releases sources without completing
either data token. Ds0 alone does not release this conservative source token.
Partial Load/Store/Ds waits retire only the oldest guaranteed target FIFO prefix;
dual data tokens still must both retire. Nonzero X waits with pending LDS FLAT
source tokens remain inconclusive. Old CDNA FLAT events cannot certify this
target. The frontend must reject unresolved apertures.
These distinctions follow the pinned LLVM
[FLAT event classification](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/AMDGPUHWEvents.cpp#L119),
[address-space classification](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L4765)
and [FLAT counter definition](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/FLATInstructions.td#L76),
with explicit Load/Store waits and their X relationship from the
[GFX125x memory model](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/docs/AMDGPUUsage.md#memory-model-gfx125x).
The pinned [target ordering helper](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/GCNSubtarget.h#L396)
and [wait-prefix construction](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp#L1369)
establish the GFX1250 FLAT data-counter FIFO. A generic FLAT destination may also
retire the required older GLOBAL translation prefix, even when its effect
resolves to Shared. Encoding class and resolved effect space are separate: a
Shared address does not establish a compiler LOCAL MMO. Actual pre-wait MIR
retains generic pointer MMOs for the native ABI controls. The
[VMEM translation dependency](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp#L2596)
and its [FLAT classification](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp#L2671)
support this asymmetric rule. It never retires a later GLOBAL translation or an
older LDS source token, completes data, or establishes memory happens-before.
An ordinary VALU overwrite still fails the source-lease check.

The dual data tokens and retained LDS source token are conservative model
requirements; they do not assert that a resolved LDS access uses VMEM_GROUP.
Actual compiled static/dynamic LDS controls use Store0, then Ds0 before
publication, and Load0/X0, then Ds0 before result consumption. Opaque or mixed
apertures, arbitrary numeric shared-base values and unpinned partial waits cannot
be promoted into this concrete-aperture profile.

A later VMEM destination that aliases an older VMEM replay source establishes
that older translation prefix, following LLVM's in-order translation dependency
rule. This does not complete either operation's data, retire later translations,
or permit immediate ALU overwrites and LDS destinations to bypass source leases.

The initial atomic profile is device-scoped `global_atomic_add_f32`, no return,
one active lane. It requires an exact paired 4-byte Atomic read/write and retains
all sources through X completion. Older non-RMW memory translations must drain
before issuance; explicit X0, completing data waits, and architectural drains
can establish this. Back-to-back RMWs do not create that earlier-translation
prerequisite. Only a validated RMW's own read/write pair avoids self-conflict;
atomic overlap with ordinary accesses retains typed conflict checks. Unordered
atomic/atomic serialization and returning/multi-lane profiles remain inconclusive.
These rules follow LLVM's [hardware-event classification](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/AMDGPUHWEvents.cpp),
[wait insertion](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp),
and GFX125x memory-model replay requirements in AMDGPUUsage.md at that revision.

Shader named barriers are reconstructed independently in `gfx1250/named.rs`.
The gfx1250 named-object checker requires execution order from initialization
to every other wave's use, and from earlier uses to reinitialization. Absent
order admits a potential uninitialized/reset-object protocol hazard and yields
a typed `AmdBarrierViolation`. Reinitialization with live joins remains outside
the selected profile; it does not gain an ordering certificate from this rule.

The checker validates descriptor capacity, initialized M0 roles, object
incarnations, operation counts, last-joined WAIT resolution, matching wait/release
source points, join-before-contribution and phase separation. Counts refer to
arrival operations, not distinct waves or PTX lane participants. A separate
execution frontier validates initialization and reuse without manufacturing a
memory completion. Completed memory at actual arrivals can publish; paths that
need a drop's memory frontier remain Unknown because that publication is unpinned.
NULL LEAVE preserves a NULL join; an unjoined LEAVE produces no drop. Named END
never supplies a missing contribution. Fresh lifetimes cannot consume stale
signals or waits from an earlier initialization.
NULL WAIT requires an explicit NULL join; an initial or cleared last-joined
selector is not inferred from its encoded operand.

This selected profile follows LLVM's pinned
[named barrier execution model](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/docs/AMDGPUExecutionSynchronization.md#named-barriers-model-extensions)
and [instruction sequences](https://github.com/llvm/llvm-project/blob/5fd50ec8e07abb94e203933017e8d6e40fb51824/llvm/docs/AMDGPUExecutionSynchronization.md#code-sequences-1).
Unordered initialization/reuse produces typed potential protocol violations;
outstanding joins at reinitialization and missing native sideband state remain
inconclusive. Signal-time expected-count updates
retain their exact 7-bit nonzero domain, phase/lifetime and M0 source. Different
setters and drops require execution order across phases; identical explicit
updates may commute. Equality with existing arrivals remains inconclusive.
`_isfirst` and clustered objects are outside this profile. The native allocated
controls and independent malformed graphs are separate tests; no GPU execution
of a harmful mutation is required or claimed.

The bounded gfx950 private profile requires exact scratch DWORD roles, VM charges,
one access per active physical work item and the fixed-private allocation premise.
Its conservative private visibility/data-reuse restriction requires an explicit
source VM0 before store-data overwrite, load-destination use/overwrite or END.
Any nonzero VM wait field, including the all-ones no-wait value, while private
accesses are pending is also outside this zero-only profile. These cases make
the trace UNKNOWN; they do not establish a hardware hazard or a distinct private
completion FIFO. A synthesized END drain does not satisfy this restriction.
Ordinary-only partial waits and remaining FLAT counter obligations are unchanged.
Private accesses do not become Global/Shared conflict effects. Dedicated literal
readlane/writelane records retain their EXEC-independent lane masks and the public
CDNA4 scheduling delays; absent records, invalid lanes or mismatched roles refuse.

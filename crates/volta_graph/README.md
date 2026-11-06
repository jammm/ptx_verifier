# volta_graph

The typed event graph of **one candidate execution** of a kernel. The analysis
(`volta_ptx_analysis`, `volta_amdgpu_analysis` or the experimental
`volta_amd_analysis`) records it while it interprets the kernel; the graph checks
(`volta_graph_check`) and the viewer (`volta_viz`) read it. Neither side
depends on the other: this crate (plus `volta_expr`, for stored values) is
their whole contract.

The graph is a record of what happened in the one schedule the interpreter
ran: which thread did what, which barrier generation each arrival joined,
which async operation each wait observed. It asserts no happens-before
relation. Each check decides for itself which edges order which events.

```
src/
├── ids.rs       typed indices (EventId, ThreadIx, AccessId, GenerationId, ...)
├── model.rs     node, edge and metadata types
├── graph.rs     EventGraph: the finished, immutable graph and its accessors
├── builder.rs   GraphBuilder: append-only construction in trace order
├── validate.rs  EventGraph::validate: representation invariants
└── dump.rs      EventGraph::dump: deterministic PTX v1 / native AMD v2 diagnostic text
```

## The data structure

`EventGraph` is a set of dense, id-indexed tables (`IdVec`s) that point into
each other:

```rust
pub struct EventGraph {
    amd_events:    Vec<AmdEvent>,                    // native wave instruction/semantic records
    amd_memory_effects: Vec<AmdMemoryEffect>,        // issues associated with byte effects
    shared_allocation_layout: Vec<SharedAllocation>, // explicit caller-supplied LDS bounds
    provenance:    Provenance,                       // launch, targets, assumptions, pc -> source span
    events:        IdVec<EventId, Event>,            // every event, in trace order
    accesses:      IdVec<AccessId, MemAccess>,       // memory effects (pointed to by Access events)
    values:        IdVec<ValueId, StoredValue>,      // values written by writes
    objects:       IdVec<SyncObjectId, SyncObject>,  // barriers, mbarrier lifetimes, collectives
    generations:   IdVec<GenerationId, Generation>,  // one "round" of a sync object
    async_ops:     IdVec<AsyncOpId, AsyncOp>,        // cp.async, TMA, tcgen05.mma/ld/st, ...
    program_order: IdVec<ThreadIx, Vec<EventId>>,    // index: each thread's events in order
    terminal:      Terminal,                         // Completed / Stuck / Aborted
}
```

`Provenance::unit` identifies what one `ThreadIx` means. PTX and native AMD
use `CtaThread`, one physical work item per graph thread. The experimental AMD
producer uses `Wave { lanes: 64 }`, one graph thread per wave, and records the
`SharedMemoryOnly` premise because it checks LDS accesses alone. Partial waves
round up the graph-thread count. Native AMD sideband records require physical
work-item units; they cannot be interpreted as wave-unit records.

### Events

An `EventId` *is* the event's position in the trace: event `e5` happened
after `e4` in the schedule the interpreter ran. Each `Event` has:

- `ctx: ExecContext`, the context it happened in:
  - `Thread(t)`: issued by thread `t`. Only these events are in a program
    order (`program_order[t]`).
  - `Async(op)`: an effect or the completion of async operation `op`. It is in
    no thread's program order: an async copy's write is not "after" the
    issuer's next instruction.
  - `Generation(g)`: the completion of synchronization generation `g`.
- `kind: EventKind`, what happened: `Access`, `Contribute`, `WaitStart`,
  `Resume`, `GenerationComplete`, `AsyncIssue`, `AsyncComplete`,
  `GroupCommit`, `Fence`, `ObjectInit`/`ObjectInval`,
  `TmemAlloc`/`TmemDealloc`/`TmemRelinquish`, `Amd(index)`, `Exit`.
- `pc: Option<ProgramPoint>`, the lowered instruction it belongs to (`None`
  for generation completions). `Provenance::instructions[pc]` maps it back to
  the PTX or native AMD source span.

### Memory effects

An `Access(AccessId)` event points to a `MemAccess`:

| field        | meaning                                                                   |
|--------------|---------------------------------------------------------------------------|
| `space`      | `Global`, `Shared` or `Tensor` (local memory is thread-private and never recorded) |
| `mode`       | `Read` / `Write`                                                          |
| `ordering`   | `Ordinary`, or `Atomic` for the selected native no-return F32 add profile |
| `proxy`      | `Generic`, `Async` (TMA, `cp.async.bulk`, `tcgen05`) or `TensorMap`       |
| `footprint`  | `Bytes { addr, width }` or a Tensor Memory `lane x column` rectangle      |
| `role`       | which operand: `Plain`, `Source`, `Destination`, `MmaA`, `MmaB`, `Accumulator`, ... |
| `attributed` | the thread the effect counts for (the issuer, or the lane a warp op acts for) |
| `value`      | for writes, a `ValueId` into `values` (`StoredValue` = arena `ExprId`s)   |

### Synchronization: objects, generations, contributions, releases

Synchronization is modeled uniformly, for every kind of barrier, as:

```
SyncObject ──has many──► Generation ◄──contributions── Contribute / Exit / AsyncComplete events
                             │
                             ├── completion: GenerationComplete event
                             └── releases:   Resume events of the threads it woke
```

- A `SyncObject` (`SyncObjectKind`) is one thing threads synchronize on: a
  named barrier `bar 0..15`, **one lifetime** of an mbarrier (a re-init
  creates a new object), the warp collectives of one warp, the
  warpgroup collectives of one warpgroup, or one thread's `cp.async` wait
  groups.
- A `Generation` is one round of that object: a named barrier's k-th firing, an
  mbarrier's k-th phase, one warp op instance. It stores its `index`, its
  `kind` (`GenerationKind`), its `contributions`, its `releases` (the `Resume`
  events), its `completion` event, and `evidence` (surplus arrivals, pending
  transaction bytes).
- A `Contribution` is an event plus how it takes part (`ContributionKind`):
  - `Blocking`: the thread waits here (`bar.sync`, a warp collective)
  - `Arrival { count, expect_tx, ordering }`: a non-blocking arrive. `ordering`
    is `Release` for a thread's own arrive, `AsyncTracked` for an arrive-on
    the hardware performs (`cp.async.mbarrier.arrive`, `tcgen05.commit`)
  - `Exited`: an exited thread counts as arrived (the paper's Sync' rule)
  - `AsyncCompletion { op, tx_bytes }`: an async op's completion counts
    toward this generation
  - `CompleteTx` / `ExpectTx`: relaxed transaction-count updates, which order
    nothing

Special generation kinds:

- `MbarrierPreInit`: the phase "before phase 0" of a fresh mbarrier. It
  completes at `mbarrier.init` with no contributions, so an odd-parity wait on
  a fresh barrier (CUTLASS's first producer wait) has a phase to return
  through.
- `Collective { side: Entry | Exit, ordering }`: every warp op instance is
  two generations, `Entry` (the lanes rendezvous before the op's effects) and
  `Exit` (they leave after). `ordering` is `Synchronizing` only for
  `bar.warp.sync`, which orders its members' memory. Every other collective is
  `ExecutionOnly`: the lanes wait for each other, nothing more.

### Asynchronous operations

An `AsyncOp` (`cp.async`, `cp.async.bulk`, TMA, `tcgen05.mma/ld/st`, a CLC
response, ...) is

```
AsyncIssue (issuer's program order) ──spawn──► effect events (Async ctx) ──► AsyncComplete (Async ctx)
                                                                                 │
                                                          contributes to ──► tracking generations
```

`AsyncOp { kind, issuer, pc, issue, effects, completion, tracking }`. `tracking`
lists the generations whose `contributions` include this op's completion: an
mbarrier phase (TMA, `tcgen05.commit`), a warp's `tcgen05.wait` collective,
or a thread's `cp.async.wait_group`.

### Terminal state and provenance

- `Terminal::Completed`: every thread exited.
- `Terminal::Stuck { blocked }`: deadlock. Each live thread with its
  `BlockedOn` condition (the barrier generation, the missing lanes of a
  collective, or the mbarrier's forming-phase state).
- `Terminal::Aborted { cause, reason, .. }`: the analysis stopped (an
  unsupported feature, or an execution error). The graph is partial but still
  valid.
- `Provenance` records the kernel, launch dims, targets, the `Assumption`s the
  analysis ran under (structured CTA, single CTA analyzed, Sync', eager async
  data, ...) so verdicts can cite them.

## How the graph is built

`GraphBuilder` is append-only. Every method pushes events at the end of the
trace, so **every edge automatically points forward** (from a smaller
`EventId` to a larger one). The builder keeps the cross-indexes consistent
(`program_order`, `Generation::contributions/releases/completion`,
`AsyncOp::effects/completion/tracking`, `SyncObject::init/inval`) but decides
nothing about the execution. The interpreter decides; the builder writes down.

The producer is `volta_ptx_analysis/src/eval/record.rs`: the interpreter calls
`record_*` hooks at each effect and sync point, and those translate to builder
calls:

| Kernel behavior                    | Builder calls                                                    |
|------------------------------------|------------------------------------------------------------------|
| `ld`/`st` to global or shared      | `thread_access(t, pc, MemAccess)`                                |
| `bar.sync` / `bar.arrive`          | first arrival: `sync_object(NamedBarrier)` + `open_generation(Barrier)`; each arrival: `contribute(Blocking \| Arrival)` |
| the barrier fires                  | `contribute_exited` for exited threads (full-CTA only), `complete_generation`, then `resume` for each waiter |
| warp op (`shfl.sync`, `ldmatrix`, `mma.sync`, ...) | per lane `contribute(Blocking)` into the `Entry` generation; at fire: `complete_generation(Entry)`, `resume` each lane, `open_generation(Exit)`; the op's effects; then each lane `contribute` + `complete_generation(Exit)` + `resume` |
| `mbarrier.init`                    | `sync_object(Mbarrier)` + `object_init`, the `MbarrierPreInit` generation opened and completed at once, phase 0 opened |
| `mbarrier.arrive{.expect_tx}`      | `contribute(Arrival)` into the forming phase; if the phase completed, `complete_generation` and open the next phase |
| `mbarrier.try_wait.parity` loop    | `wait_start` when the thread blocks, `resume` into the latest completed phase when it wakes |
| TMA / `cp.async.bulk`              | `async_issue`, `async_effect` per read/write (data lands eagerly), `track(op, phase, bytes)` on the forming mbarrier phase |
| `tcgen05.mma` + `tcgen05.commit`   | `async_issue` + `async_effect`s on Tensor Memory; at commit, `track` each unobserved MMA of the thread into the forming phase |
| `tcgen05.ld/st` + `tcgen05.wait`   | `async_issue` + effects; the wait (a warp collective) `track`s the warp's pending ops into its `Exit` generation |
| `cp.async` + `cp.async.wait_group` | `async_issue` + effects; the wait opens an `AsyncGroupWait` generation, `track`s the completed ops, completes it, resumes the thread |
| fences                             | `fence(t, pc, FenceKind)`                                        |
| `tcgen05.alloc/dealloc/relinquish` | `tmem_alloc` / `tmem_dealloc` / `tmem_relinquish`                |
| thread exits                       | `exit`                                                           |
| end of run                         | `finish(Terminal)`                                               |

`track(op, generation, bytes)` completes `op` now if it has not completed yet
(`async_complete`), adds `generation` to `op.tracking`, and adds an
`AsyncCompletion` contribution to the generation.

### Worked example

Two threads each store to shared memory, meet at `bar.sync 0`, then read the
other's slot and write it to global memory:

```ptx
st.shared.u32 [buf + 4*tid], tid;
bar.sync 0;
ld.shared.u32 %r1, [buf + 4*(tid^1)];
st.global.u32 [out + 4*(tid^1)], %r1;
```

`volta analyze ex.ptx -k ex -b 2 ... --dump-graph ex.graph` writes:

```
object o0 named-barrier 0 init=- inval=-
generation g0 o0#0 barrier full-cta complete=e4 surplus=0 pending_tx=0
  contrib e1 blocking
  contrib e3 blocking
  release e5 e6
e0 t0 pc4 write ordinary generic shared Plain by t0 [0x0,+4) = n2
e1 t0 pc5 contribute g0
e2 t1 pc4 write ordinary generic shared Plain by t1 [0x4,+4) = n8
e3 t1 pc5 contribute g0
e4 g0 - complete g0
e5 t0 pc5 resume g0
e6 t1 pc5 resume g0
e7 t0 pc9 read ordinary generic shared Plain by t0 [0x4,+4)
e8 t0 pc13 write ordinary generic global Plain by t0 [0x1004,+4) = n8
e9 t0 pc14 exit
e10 t1 pc9 read ordinary generic shared Plain by t1 [0x0,+4)
e11 t1 pc13 write ordinary generic global Plain by t1 [0x1000,+4) = n2
e12 t1 pc14 exit
terminal completed
```

As a picture (program order down each lane, `E_sync` edges through `g0`):

```
 t0                    g0                     t1
 e0 write sh[0x0]                             e2 write sh[0x4]
 e1 contribute ──────►  e4 complete  ◄─────── e3 contribute
 e5 resume     ◄──────       │      ───────►  e6 resume
 e7 read sh[0x4]                              e10 read sh[0x0]
 e8 write gl[0x1004]                          e11 write gl[0x1000]
 e9 exit                                      e12 exit
```

The round-robin scheduler ran `t0` until it blocked at the barrier, then `t1`;
the barrier fired (`e4`) and released both. `n2`/`n8` are `ExprId`s in the
analysis's arena. The check for `e0 -> e10` (write then read of `0x0` by
different threads) finds the path `e0 -> e1 -> e4 -> e6 -> e10`.

## Edge kinds

The graph stores edges implicitly, as the cross-references above:

| edge          | encoded by                                                         |
|---------------|--------------------------------------------------------------------|
| program order | consecutive entries of `program_order[t]`                          |
| contribution  | `Generation::contributions[i].event` -> `Generation::completion`   |
| release       | `Generation::completion` -> each `Generation::releases[i]`         |
| spawn         | `AsyncOp::issue` -> each effect event                              |
| completion    | each effect -> `AsyncOp::completion` -> each `tracking` generation |

`volta_graph_check::relation` turns them into the typed projections
(`G0`, `G_HB`). `volta_graph_check::pipeline` adds `E_pipeline` (implicitly
pipelined `tcgen05` pairs). Neither is stored here.

## Tentative associations

Some links are what this schedule produced, not facts that hold for every
schedule: which generation an arrival or wait landed in, which phase tracks an
async completion, and read-from (the last write before a read in trace order).
The checks must validate these before they rely on them. For example,
`MbarrierEpoch` and `MbarrierOrdering` certify the mbarrier phase joins before
`G_HB` admits them.

## Invariants (`validate.rs`)

`EventGraph::validate()` checks the representation only, never a property of
the execution: contexts and ids in range, thread events only in thread
contexts, program order consistent with `events`, each async op's effects after
its issue and before its completion, contributions before their generation's
completion and releases after it, one generation per `(object, index, slot)`,
consistent terminal state. Most importantly it checks that **every edge runs
forward in trace order**, which makes the relation acyclic and makes trace
order a topological order. Consumers may assume this only after `validate()`
returns `Ok`; it returns every `GraphDefect` it found.

## Text dump (`dump.rs`)

`EventGraph::dump()` / `write_dump()` print the PTX version 1 format shown
above, or version 2 with native AMD records and associations. Both include the
header, assumptions, objects, generations, async ops, events and terminal state
in deterministic order. These dumps are diagnostics, not replay inputs.
Equal graphs dump to identical bytes, so diffs of dumps show trace changes.

## Native AMD records

The gfx942/gfx950/gfx1250 producers use the same `EventGraph` and `GraphBuilder`.
Their additive `AmdEventKind` records retain target-selected EXEC masks, individual register/lane slices,
memory counter charges, waits, matrix operand roles, independent S_NOP states, and
workgroup barrier generations. `EventKind::Amd(index)` indexes `amd_events()`;
`amd_memory_effects()` associates existing byte-footprint memory events with their
issuing wave instruction. Those effects are pending in the AMD interpretation even
though their data is calculated eagerly. Their event positions do not certify completion.

Record one `Instruction` before each issued instruction and its semantic event,
one `MemoryIssue` per wave instruction, and an association for every memory effect
(including kernarg/dispatch reads). EXEC=0 vector instructions produce an instruction
record but no memory issue. Cross-lane DS permutations have target-selected counter charges and pending
registers with no fabricated LDS footprint. Legacy MFMA uses `MatrixIssue`;
gfx1250 WMMA uses `WmmaIssue`. Register ranges use one entry per 32-bit register;
u64 lane masks use 64 physical bits on gfx942/gfx950 and only 32 on gfx1250.

`AmdMemoryScope::Gfx1250System` preserves explicit `SCOPE_SYS` on ordinary
gfx1250 GLOBAL and resolved-global FLAT effects. It is distinct from legacy
CDNA workgroup scope. Visibility requires both endpoints; the scope does not
create completion, synchronization or coverage beyond the selected workgroup.

The selected gfx1250 `GlobalAtomicAddF32NoReturn` profile records device-scoped
F32 RMWs with one active lane, no destination register, and Store/X charges.
Its read then write use paired aligned 4-byte Global effects with `Atomic`
ordering and the same attributed lane/address. It retains every source, including
implicit EXEC, through translation completion. This tag grants no publication
edge for other addresses or other instructions.

The gfx1250 `FlatGlobalLoad/Store` and `FlatLdsLoad/Store` classes retain genuine
FLAT provenance and a producer-resolved segment. They conservatively charge Load
or Store, plus Ds and X; instruction sources include implicit EXEC. The LDS
classes remain outside the GLOBAL replay group: group switches and later GLOBAL
destination aliases cannot retire their source leases. Explicit waits retain
their own counter meanings, and both data tokens must complete for readiness
and publication.
A producer must resolve the aperture before selecting a class; an unknown
pointer segment does not become global merely because its address is concrete.

Gfx1250 shader named barriers use `AmdNamedBarrier { id, incarnation }` plus an
explicit phase on joins, signals, waits, releases and leaves. Every successful
reinitialization advances the incarnation. INIT records its expected arrival
count, WAIT retains both the encoded selector and resolved last-joined object,
and the descriptor's allocated capacity is an explicit provenance premise.
`NamedBarrierSignalWithCount` records a single arrival and its explicit
7-bit M0 expected-count update. M0 is scalar register 124 in instruction
dependency roles. Named deadlocks retain
object, incarnation, phase and actual expected/arrived counts in `BlockedOn`.
These records describe candidate associations; they do not certify publication.

gfx1250 `AluWait` records an `s_wait_alu` DEPCTR encoding; it names ALU and
VMEM-source dependency counters only and never completes a memory counter.
`Prefetch` memory issues charge only X and have no destinations or effects; the
following `PrefetchFootprint` lists every active lane's byte address and its
`AmdCacheScope`. Scopes wider than CU are speculative. A profile may record
`global_region_layout()` (objects and forbidden guards, with the
`ConfiguredGlobalRegionLayout` premise); `GlobalAccessFault` then records an
attempted GLOBAL access with every active lane's address before an abort with
`GLOBAL_BOUNDS_ABORT_REASON`. It has no effects and certifies nothing alone.
`ModeAccess` (B0 m2 profile and its WMMA/multiwave supersets) records one executed MODE read or write: the
operation (`AmdModeOp`), bit field and exact value; it follows its instruction
record and has no register effects.
`BufferRangeCheck` (B0 litmus profile only) follows the effects of one untyped
raw-buffer issue: descriptor base, `num_records`, stride, OOB_SELECT, DWORD
count and every active lane's byte offset and in-range DWORD mask.
`AmdBufferRangeCheck::expected_in_range` is the documented rule.

Text dumps without native AMD events retain `volta-graph 1`; graphs with AMD events use
`volta-graph 2` and include their sidecar associations. The text dump is a diagnostic
format, not an executable replay input. AMD rendezvous deadlocks use
`BlockedOn::AmdBarrier` without inventing a PTX synchronization object.

The gfx950 fixed-private profile records DWORD scratch accesses separately from
Global/Shared effects. `FixedPrivateAllocation` is a valid-dispatch premise for
backed bytes per physical work item, checked against producer descriptor and
metadata; it is not inferred from a descriptor alone. Literal-lane markers
identify the selected physical lane even when EXEC is zero.

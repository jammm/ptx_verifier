//! One kernel run as the viewer's JSON document (`volta-viz 1`): the graph's events placed in
//! lanes, its typed edges, the checks' findings resolved to events, and the PTX source the
//! events' program points map back to.
//!
//! A large trace holds tens of thousands of near-identical events, so events are rows of
//! indices into the document's string and detail-pair tables, and each async op is listed once
//! (`[issue, completion, effects...]`) rather than as two edges per effect.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use volta_expr::ExprArena;
use volta_graph::{
    AbortCause, AccessMode, ArrivalOrdering, AsyncOpId, BarrierArrivals, BlockedOn, BulkWaitScope,
    CollectiveSide, ContributionKind, EventGraph, EventId, EventKind, ExecContext, FenceKind,
    Footprint, GenerationId, GenerationKind, GroupFamily, MemberOrdering, OrderingKind, Parity,
    Proxy, Space, StoredValue, SyncObjectId, SyncObjectKind, Terminal, ThreadIx, ThreadUnit,
};
use volta_graph_check::allocation_order::AllocationOrderViolation;
use volta_graph_check::allocation_permit::AllocationPermitViolation;
use volta_graph_check::barrier_epoch::EpochViolation;
use volta_graph_check::domain::ConflictKind;
use volta_graph_check::mbarrier_release_order::ReleaseOrderViolation;
use volta_graph_check::pipeline::pipeline_edges;
use volta_graph_check::proxy_order::MissingFence;
use volta_graph_check::resource_lifetime::LifetimeViolation;
use volta_graph_check::{GraphConclusion, GraphReport, Property, PropertyResult, Verdict};

use crate::json::Json;
use crate::store::RunId;

mod display;
use display::DisplayEvents;

/// The longest stored-value expression a document carries; longer ones are cut.
const MAX_VALUE_BYTES: usize = 400;

/// What the analysis produced for one run.
pub enum Outcome<'a> {
    /// The execution ran (to completion or not) and recorded `graph`, whose stored values are
    /// expressions of `arena`. `report` is the graph checks' result, when the run checked them.
    Executed {
        graph: &'a EventGraph,
        arena: &'a ExprArena,
        report: Option<&'a GraphReport>,
    },
    /// The execution never started: lowering rejected the kernel (`Unsupported`), or there is
    /// no such kernel or the configuration is invalid (`Error`).
    NotStarted { cause: AbortCause, error: String },
}

/// One kernel run to visualize.
pub struct Run<'a> {
    pub id: &'a RunId,
    /// The PTX file the source was read from, as the producer names it; `None` for source a
    /// test built itself.
    pub ptx_file: Option<&'a str>,
    pub source: &'a str,
    pub outcome: Outcome<'a>,
}

/// The checks whose failure is a race condition: an unordered conflicting access, or a
/// cross-proxy pair no proxy fence orders.
const RACE_CHECKS: [Property; 5] = [
    Property::AmdMemoryConflict,
    Property::AmdWaitReadiness,
    Property::MemoryConflict,
    Property::ProxyOrder,
    Property::TensorOrder,
];

/// A run's outcome: what its checks concluded, or why it never got a conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    RaceFree,
    RaceCondition,
    CheckFailed,
    Deadlock,
    Unsupported,
    Unknown,
    Error,
    /// Executed, but the run did not check the graph.
    Unchecked,
}

impl Status {
    fn of(outcome: &Outcome) -> Status {
        match outcome {
            Outcome::NotStarted { cause, .. } => Status::stopped(*cause),
            Outcome::Executed { report: None, .. } => Status::Unchecked,
            Outcome::Executed {
                graph,
                report: Some(report),
                ..
            } => Status::checked(graph, report),
        }
    }

    /// A failed race check outranks a deadlock, and an inconclusive trace that aborted is
    /// classified by why it stopped.
    fn checked(graph: &EventGraph, report: &GraphReport) -> Status {
        match report.conclusion() {
            GraphConclusion::Verified => Status::RaceFree,
            GraphConclusion::Rejected { failed, deadlock } => {
                if failed.iter().any(|s| RACE_CHECKS.contains(&s.property)) {
                    Status::RaceCondition
                } else if deadlock.is_some() {
                    Status::Deadlock
                } else {
                    Status::CheckFailed
                }
            }
            GraphConclusion::Unknown { .. } => match graph.terminal() {
                Terminal::Aborted { cause, .. } => Status::stopped(*cause),
                Terminal::Completed | Terminal::Stuck { .. } => Status::Unknown,
            },
        }
    }

    fn stopped(cause: AbortCause) -> Status {
        match cause {
            AbortCause::Unsupported => Status::Unsupported,
            AbortCause::Error => Status::Error,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Status::RaceFree => "race-free",
            Status::RaceCondition => "race-condition",
            Status::CheckFailed => "check-failed",
            Status::Deadlock => "deadlock",
            Status::Unsupported => "unsupported",
            Status::Unknown => "unknown",
            Status::Error => "error",
            Status::Unchecked => "unchecked",
        }
    }
}

/// The run's index entry: what the test list shows without loading the document.
pub fn summary(run: &Run) -> Json {
    let (kernel, terminal, threads, events) = match &run.outcome {
        Outcome::Executed { graph, .. } => (
            Json::str(graph.provenance().kernel.clone()),
            Json::str(terminal_name(graph.terminal())),
            Json::uint(graph.threads()),
            Json::uint(graph.events().len() as u64),
        ),
        Outcome::NotStarted { .. } => (Json::Null, Json::Null, Json::Null, Json::Null),
    };
    Json::Obj(vec![
        ("key", Json::str(run.id.key())),
        ("suite", Json::str(run.id.suite.clone())),
        ("test", Json::str(run.id.test.clone())),
        ("run", Json::uint(run.id.ordinal.get())),
        ("ptx", Json::opt(run.ptx_file, Json::str)),
        ("kernel", kernel),
        ("status", Json::str(Status::of(&run.outcome).name())),
        ("terminal", terminal),
        ("threads", threads),
        ("events", events),
    ])
}

/// The run's browser document, with a bounded display selection for large graphs.
pub fn document(run: &Run) -> Json {
    document_with_limit(run, Some(display::EVENT_LIMIT))
}

/// Complete archival document, without the browser's display projection.
pub fn complete_document(run: &Run) -> Json {
    document_with_limit(run, None)
}

fn document_with_limit(run: &Run, limit: Option<usize>) -> Json {
    let amd_source = match &run.outcome {
        Outcome::Executed { graph, .. } => amd_graph(graph),
        // Before execution there is no graph/launch provenance. Retain only
        // the source's syntax family; do not invent an analyzed target/launch.
        Outcome::NotStarted { .. } => run
            .source
            .lines()
            .any(|line| line.split_whitespace().next() == Some(".amdgcn_target")),
    };
    let mut fields = vec![
        ("format", Json::str("volta-viz 1")),
        ("meta", summary(run)),
        ("source", Json::str(run.source)),
        (
            "source_language",
            Json::str(if amd_source { "amdgpu" } else { "ptx" }),
        ),
    ];
    if matches!(run.outcome, Outcome::NotStarted { .. })
        && let Some((target, architecture)) = declared_native_target(run.source)
    {
        // Source declaration only: these fields do not assert that parsing,
        // launch validation or symbolic execution started successfully.
        fields.push(("source_target", Json::str(target)));
        fields.push(("source_wave_size", Json::uint(architecture.wave_size())));
        fields.push(("source_wave_kind", Json::str("wave")));
    }
    match &run.outcome {
        Outcome::NotStarted { error, .. } => fields.push(("error", Json::str(error.clone()))),
        Outcome::Executed {
            graph,
            arena,
            report,
        } => fields.extend(graph_fields(graph, arena, *report, run.source, limit)),
    }
    Json::Obj(fields)
}

fn amd_graph(graph: &EventGraph) -> bool {
    matches!(graph.provenance().unit, ThreadUnit::Wave { .. })
        || volta_graph::AmdArchitecture::from_targets(&graph.provenance().targets).is_some()
        || !graph.amd_events().is_empty()
}

fn graph_wave_size(graph: &EventGraph) -> u32 {
    if let ThreadUnit::Wave { lanes } = graph.provenance().unit {
        return lanes;
    }
    volta_graph::AmdArchitecture::from_targets(&graph.provenance().targets)
        .map_or(if amd_graph(graph) { 64 } else { 32 }, |architecture| {
            architecture.wave_size()
        })
}

fn amd_mask(graph: &EventGraph, mask: u64) -> String {
    format!(
        "{mask:#0width$x}",
        width = (graph_wave_size(graph) / 4 + 2) as usize
    )
}

fn declared_native_target(source: &str) -> Option<(&str, volta_graph::AmdArchitecture)> {
    let mut declarations = source
        .lines()
        .filter(|line| line.split_whitespace().next() == Some(".amdgcn_target"));
    let line = declarations.next()?;
    if declarations.next().is_some() {
        return None;
    }
    let token = line.split_whitespace().nth(1)?;
    let target = token
        .strip_prefix('"')?
        .strip_suffix('"')?
        .strip_prefix("amdgcn-amd-amdhsa--")?;
    volta_graph::AmdArchitecture::from_target(target).map(|architecture| (target, architecture))
}

/// Each distinct string and `(key, value)` detail pair of a document, stored once.
#[derive(Default)]
struct Interner {
    strings: Vec<String>,
    string_ids: HashMap<String, u32>,
    pairs: Vec<(u32, u32)>,
    pair_ids: HashMap<(u32, u32), u32>,
}

impl Interner {
    fn string(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.string_ids.get(s) {
            return id;
        }
        let id = self.strings.len() as u32;
        self.strings.push(s.to_string());
        self.string_ids.insert(s.to_string(), id);
        id
    }

    fn pair(&mut self, key: &str, value: &str) -> u32 {
        let pair = (self.string(key), self.string(value));
        let next = self.pairs.len() as u32;
        let id = *self.pair_ids.entry(pair).or_insert(next);
        if id == next {
            self.pairs.push(pair);
        }
        id
    }
}

/// `[lane, pc or -1, category, label, detail pairs...]`, all but the lane and pc interned.
fn event_row(
    graph: &EventGraph,
    arena: &ExprArena,
    lanes: &Lanes,
    interner: &mut Interner,
    id: EventId,
    scope: Option<volta_graph::AmdMemoryScope>,
) -> Json {
    let ev = graph.event(id);
    let (category, label, mut info) = describe(graph, arena, id);
    if let Some(scope) = scope {
        info.push(("AMD memory scope", format!("{scope:?}")));
    }
    let mut row = vec![
        Json::uint(lanes.lane_of(graph, ev.ctx) as u64),
        ev.pc.map_or(Json::Int(-1), |pc| Json::uint(pc.0)),
        Json::uint(interner.string(category)),
        Json::uint(interner.string(&label)),
    ];
    row.extend(info.iter().map(|(k, v)| Json::uint(interner.pair(k, v))));
    Json::Arr(row)
}

fn graph_fields(
    graph: &EventGraph,
    arena: &ExprArena,
    report: Option<&GraphReport>,
    source: &str,
    limit: Option<usize>,
) -> Vec<(&'static str, Json)> {
    let lanes = Lanes::of(graph);
    let finding_limit = if limit.is_some_and(|limit| graph.events().len() > limit) {
        display::FINDING_LIMIT
    } else {
        usize::MAX
    };
    let mut checks = Json::opt(report, |r| checks(graph, r, finding_limit));
    let mut terminal = terminal(graph);
    let view = DisplayEvents::of(graph, &lanes, &mut checks, &mut terminal, limit);
    let mut interner = Interner::default();
    let amd_scopes: HashMap<_, _> = graph
        .amd_memory_effects()
        .iter()
        .filter(|effect| view.index(effect.effect).is_some())
        .map(|effect| (effect.effect, effect.scope))
        .collect();
    let events: Vec<Json> = view
        .ids()
        .map(|id| {
            event_row(
                graph,
                arena,
                &lanes,
                &mut interner,
                id,
                amd_scopes.get(&id).copied(),
            )
        })
        .collect();
    let lines = LineIndex::new(source);
    let p = graph.provenance();
    let mut fields = vec![
        ("launch", {
            let mut launch = vec![
                ("wave_size", Json::uint(graph_wave_size(graph))),
                (
                    "thread_unit",
                    Json::str(match p.unit {
                        ThreadUnit::CtaThread => "work_item",
                        ThreadUnit::Wave { .. } => "wave",
                    }),
                ),
                (
                    "wave_kind",
                    Json::str(if amd_graph(graph) { "wave" } else { "warp" }),
                ),
                (
                    "source_language",
                    Json::str(if amd_graph(graph) { "amdgpu" } else { "ptx" }),
                ),
                ("block", dims(p.block_dim)),
                ("grid", dims(p.grid_dim)),
                ("workgroup_id", dims(p.workgroup_id)),
                (
                    "targets",
                    Json::Arr(p.targets.iter().map(|t| Json::str(t.clone())).collect()),
                ),
                (
                    "assumptions",
                    Json::Arr(
                        p.assumptions
                            .iter()
                            .map(|a| Json::str(a.describe()))
                            .collect(),
                    ),
                ),
            ];
            if !graph.shared_allocation_layout().is_empty() {
                launch.push((
                    "shared_allocations",
                    Json::Arr(
                        graph
                            .shared_allocation_layout()
                            .iter()
                            .map(|region| {
                                Json::Obj(vec![
                                    ("name", Json::str(region.name.clone())),
                                    ("offset", Json::uint(region.offset)),
                                    ("size", Json::uint(region.size)),
                                ])
                            })
                            .collect(),
                    ),
                ));
            }
            Json::Obj(launch)
        }),
        (
            "instructions",
            Json::Arr(
                p.instructions
                    .iter()
                    .map(|i| {
                        Json::Obj(vec![
                            ("kind", Json::str(i.kind)),
                            (
                                "line",
                                Json::opt(i.span.and_then(|(lo, _)| lines.line(lo)), |l| {
                                    Json::uint(l as u64)
                                }),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("lanes", lanes.to_json()),
        (
            "strings",
            Json::Arr(interner.strings.into_iter().map(Json::Str).collect()),
        ),
        (
            "pairs",
            Json::Arr(
                interner
                    .pairs
                    .iter()
                    .flat_map(|&(k, v)| [Json::uint(k), Json::uint(v)])
                    .collect(),
            ),
        ),
        ("events", Json::Arr(events)),
        ("edges", edges(graph, &view)),
        ("asyncOps", async_ops(graph, &view)),
        ("terminal", terminal),
        ("checks", checks),
        ("conclusion", Json::opt(report, |r| conclusion(graph, r))),
    ];
    fields.extend(view.fields());
    fields
}

fn dims((x, y, z): (u32, u32, u32)) -> Json {
    Json::Arr(vec![Json::uint(x), Json::uint(y), Json::uint(z)])
}

/// 1-based line numbers of source byte offsets.
struct LineIndex {
    starts: Vec<usize>,
    len: usize,
}

impl LineIndex {
    fn new(source: &str) -> LineIndex {
        let starts = std::iter::once(0)
            .chain(source.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        LineIndex {
            starts,
            len: source.len(),
        }
    }

    fn line(&self, offset: usize) -> Option<usize> {
        (offset <= self.len).then(|| self.starts.partition_point(|&s| s <= offset))
    }
}

// =========================================================================
// Lanes
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LaneKind {
    Thread,
    /// The effects and completions of one thread's asynchronous operations.
    Async,
    /// The generation completions of one synchronization object.
    Object,
}

struct Lane {
    kind: LaneKind,
    label: String,
    /// The thread a thread or async lane belongs to.
    thread: Option<ThreadIx>,
}

/// Each thread's lane, followed by its async lane if it issues any; then one lane per
/// synchronization object with a completed generation.
struct Lanes {
    lanes: Vec<Lane>,
    thread: BTreeMap<ThreadIx, usize>,
    asynchronous: BTreeMap<ThreadIx, usize>,
    object: BTreeMap<SyncObjectId, usize>,
}

impl Lanes {
    fn of(graph: &EventGraph) -> Lanes {
        let issuers: BTreeSet<ThreadIx> = graph.async_ops().values().map(|op| op.issuer).collect();
        let mut lanes = Lanes {
            lanes: Vec::new(),
            thread: BTreeMap::new(),
            asynchronous: BTreeMap::new(),
            object: BTreeMap::new(),
        };
        for t in (0..graph.threads()).map(ThreadIx) {
            lanes.thread.insert(t, lanes.lanes.len());
            lanes.lanes.push(Lane {
                kind: LaneKind::Thread,
                label: thread_name(graph, t),
                thread: Some(t),
            });
            if issuers.contains(&t) {
                lanes.asynchronous.insert(t, lanes.lanes.len());
                lanes.lanes.push(Lane {
                    kind: LaneKind::Async,
                    label: format!("async {}", thread_name(graph, t)),
                    thread: Some(t),
                });
            }
        }
        let completing: BTreeSet<SyncObjectId> = graph
            .generations()
            .values()
            .filter(|g| g.completion.is_some())
            .map(|g| g.object)
            .collect();
        for object in completing {
            lanes.object.insert(object, lanes.lanes.len());
            lanes.lanes.push(Lane {
                kind: LaneKind::Object,
                label: object_name(graph, object),
                thread: None,
            });
        }
        lanes
    }

    fn lane_of(&self, graph: &EventGraph, ctx: ExecContext) -> usize {
        match ctx {
            ExecContext::Thread(t) => self.thread[&t],
            ExecContext::Async(op) => self.asynchronous[&graph.async_op(op).issuer],
            ExecContext::Generation(g) => self.object[&graph.generation(g).object],
        }
    }

    fn to_json(&self) -> Json {
        Json::Arr(
            self.lanes
                .iter()
                .map(|lane| {
                    Json::Obj(vec![
                        (
                            "kind",
                            Json::str(match lane.kind {
                                LaneKind::Thread => "thread",
                                LaneKind::Async => "async",
                                LaneKind::Object => "object",
                            }),
                        ),
                        ("label", Json::str(lane.label.clone())),
                        ("thread", Json::opt(lane.thread, |t| Json::uint(t.0))),
                    ])
                })
                .collect(),
        )
    }
}

// =========================================================================
// Edges
// =========================================================================

/// The typed edges besides program order (implicit in each thread lane's sequence) and the
/// async ops' spawn and completion edges (see `async_ops`), as flat `[from, to, ...]` lists
/// per kind. An execution-only collective's member edges get their own `-execution-only` kinds,
/// drawn dashed: they order nothing, unlike its async completions and the releases observing them.
fn edges(graph: &EventGraph, view: &DisplayEvents) -> Json {
    let mut by_kind: BTreeMap<&'static str, Vec<Json>> = BTreeMap::new();
    let mut push = |kind, from: EventId, to: EventId| {
        let (Some(from), Some(to)) = (view.index(from), view.index(to)) else {
            return;
        };
        by_kind
            .entry(kind)
            .or_default()
            .extend([Json::uint(from), Json::uint(to)]);
    };
    for g in graph.generations().values() {
        let Some(completion) = g.completion else {
            continue;
        };
        let execution_only = matches!(
            g.kind,
            GenerationKind::Collective {
                ordering: MemberOrdering::ExecutionOnly,
                ..
            }
        );
        let completes_op = |kind| matches!(kind, ContributionKind::AsyncCompletion { .. });
        for c in &g.contributions {
            let kind = if execution_only && !completes_op(c.kind) {
                "contribute-execution-only"
            } else {
                "contribute"
            };
            push(kind, c.event, completion);
        }
        let observes = g.contributions.iter().any(|c| completes_op(c.kind));
        let release = if execution_only && !observes {
            "release-execution-only"
        } else {
            "release"
        };
        for &r in &g.releases {
            push(release, completion, r);
        }
    }
    for e in pipeline_edges(graph).edges(graph) {
        push("pipeline", e.from, e.to);
    }
    for association in graph.amd_memory_effects() {
        if graph.events().get(association.issue).is_some()
            && graph.events().get(association.effect).is_some()
        {
            push("amd-issue", association.issue, association.effect);
        }
    }
    let mut amd_arrivals = HashMap::<u32, Vec<EventId>>::new();
    let mut split_arrivals = HashMap::<(i16, u32), Vec<EventId>>::new();
    let mut split_waits = HashMap::new();
    let mut named_arrivals = HashMap::<(volta_graph::AmdNamedBarrier, u32), Vec<EventId>>::new();
    let mut named_waits = HashMap::new();
    for record in graph.amd_events() {
        match record.kind {
            volta_graph::AmdEventKind::NamedBarrierSignal { object, phase, .. }
            | volta_graph::AmdEventKind::NamedBarrierSignalWithCount { object, phase, .. }
            | volta_graph::AmdEventKind::NamedBarrierLeave { object, phase }
                if object.id != 0 =>
            {
                named_arrivals
                    .entry((object, phase))
                    .or_default()
                    .push(record.event);
            }
            volta_graph::AmdEventKind::NamedBarrierWait { object, phase, .. } => {
                named_waits.insert((object, phase, record.wave), record.event);
            }
            volta_graph::AmdEventKind::NamedBarrierRelease { object, phase } => {
                if let Some(arrivals) = named_arrivals.get(&(object, phase)) {
                    for &arrival in arrivals {
                        push("amd-barrier", arrival, record.event);
                    }
                }
                if let Some(&wait) = named_waits.get(&(object, phase, record.wave)) {
                    push("amd-barrier", wait, record.event);
                }
            }
            volta_graph::AmdEventKind::SplitBarrierSignal {
                barrier,
                generation,
                ..
            } => split_arrivals
                .entry((barrier, generation))
                .or_default()
                .push(record.event),
            volta_graph::AmdEventKind::SplitBarrierWait {
                barrier,
                generation,
            } => {
                split_waits.insert((barrier, generation, record.wave), record.event);
            }
            volta_graph::AmdEventKind::SplitBarrierRelease {
                barrier,
                generation,
            } => {
                if let Some(arrivals) = split_arrivals.get(&(barrier, generation)) {
                    for &arrival in arrivals {
                        push("amd-barrier", arrival, record.event);
                    }
                }
                if let Some(&wait) = split_waits.get(&(barrier, generation, record.wave)) {
                    push("amd-barrier", wait, record.event);
                }
            }
            volta_graph::AmdEventKind::BarrierArrive { generation, .. } => amd_arrivals
                .entry(generation)
                .or_default()
                .push(record.event),
            volta_graph::AmdEventKind::BarrierRelease { generation } => {
                if let Some(arrivals) = amd_arrivals.get(&generation) {
                    for &arrival in arrivals {
                        push("amd-barrier", arrival, record.event);
                    }
                }
            }
            _ => {}
        }
    }
    Json::Obj(
        by_kind
            .into_iter()
            .map(|(kind, list)| (kind, Json::Arr(list)))
            .collect(),
    )
}

/// Each async op as `[issue, completion or -1, effect events...]`: the issue spawns every
/// effect, and every effect precedes the completion.
fn async_ops(graph: &EventGraph, view: &DisplayEvents) -> Json {
    if graph.async_ops().is_empty() {
        return Json::Arr(Vec::new());
    }
    let mut access_event = BTreeMap::new();
    for (id, ev) in graph.events().iter() {
        if let EventKind::Access(a) = ev.kind
            && let Some(displayed) = view.index(id)
        {
            access_event.insert(a, displayed);
        }
    }
    Json::Arr(
        graph
            .async_ops()
            .values()
            .filter(|op| view.index(op.issue).is_some())
            // The v1 decoder draws issue -> completion for truly effectless
            // ops. Omitted effects must not create that synthetic direct edge.
            .filter(|op| {
                op.effects.is_empty() || op.effects.iter().any(|a| access_event.contains_key(a))
            })
            .map(|op| {
                let mut row = vec![
                    Json::uint(view.index(op.issue).expect("selected async issue")),
                    op.completion.map_or(Json::Int(-1), |c| {
                        Json::uint(view.index(c).expect("selected async completion"))
                    }),
                ];
                row.extend(
                    op.effects
                        .iter()
                        .filter_map(|a| access_event.get(a))
                        .map(|&id| Json::uint(id)),
                );
                Json::Arr(row)
            })
            .collect(),
    )
}

// =========================================================================
// Event descriptions
// =========================================================================

type Info = Vec<(&'static str, String)>;

/// The event's category (its node style), one-line label and detail rows.
fn describe(graph: &EventGraph, arena: &ExprArena, id: EventId) -> (&'static str, String, Info) {
    let ev = graph.event(id);
    let mut info: Info = vec![("context", context_name(graph, ev.ctx))];
    let (category, label) = match ev.kind {
        EventKind::Amd(index) => {
            let Some(record) = graph.amd_event(index) else {
                return ("issue", "invalid AMD event".into(), info);
            };
            info.push(("wave", record.wave.to_string()));
            match &record.kind {
                volta_graph::AmdEventKind::PrivateMemoryAccess {
                    issue,
                    thread,
                    offset,
                    width,
                    kind,
                } => {
                    info.push(("issue", format!("{issue:?}")));
                    info.push(("thread", thread.0.to_string()));
                    info.push(("private offset", offset.to_string()));
                    info.push(("bytes", width.to_string()));
                    ("access", format!("private {kind:?}"))
                }
                volta_graph::AmdEventKind::LaneRead { lane } => {
                    ("instruction", format!("read physical lane {lane}"))
                }
                volta_graph::AmdEventKind::LaneWrite { lane } => {
                    ("instruction", format!("write physical lane {lane}"))
                }
                volta_graph::AmdEventKind::TensorAccessFault {
                    load,
                    descriptor,
                    sources,
                } => {
                    info.push((
                        "transfer",
                        if *load {
                            "global to LDS"
                        } else {
                            "LDS to global"
                        }
                        .into(),
                    ));
                    info.push(("descriptor", format!("{descriptor:?}")));
                    info.push(("scalar tuple starts", format!("{sources:?}")));
                    (
                        "instruction",
                        "attempted tensor access; aborted before issue".into(),
                    )
                }
                volta_graph::AmdEventKind::WmmaIssue {
                    kind,
                    destinations,
                    sources_ab,
                    accumulator,
                    scales,
                    index,
                } => {
                    info.push(("matrix form", format!("{kind:?}")));
                    if !scales.is_empty() {
                        info.push(("block scales", format!("{scales:?}")));
                    }
                    if !index.is_empty() {
                        info.push(("sparsity index", format!("{index:?}")));
                    }
                    info.push(("destinations", format!("{destinations:?}")));
                    info.push(("A/B sources", format!("{sources_ab:?}")));
                    info.push(("accumulator", format!("{accumulator:?}")));
                    ("issue", format!("wave {} WMMA {kind:?}", record.wave))
                }
                volta_graph::AmdEventKind::DelayAlu {
                    instid0,
                    instskip,
                    instid1,
                } => {
                    info.push(("dependency 0", format!("{instid0:?}")));
                    info.push(("instruction skip", instskip.to_string()));
                    info.push(("dependency 1", format!("{instid1:?}")));
                    (
                        "wait",
                        format!("s_delay_alu {instid0:?} skip {instskip} {instid1:?}"),
                    )
                }
                volta_graph::AmdEventKind::WaitCounters { thresholds } => {
                    info.push(("counter thresholds", format!("{thresholds:?}")));
                    ("wait", format!("native wait {thresholds:?}"))
                }
                volta_graph::AmdEventKind::SplitBarrierSignal {
                    barrier,
                    generation,
                    exec_mask,
                } => {
                    info.push(("barrier object", barrier.to_string()));
                    info.push(("generation", generation.to_string()));
                    info.push(("EXEC", format!("{exec_mask:#x}")));
                    (
                        "sync",
                        format!("s_barrier_signal {barrier} phase {generation}"),
                    )
                }
                volta_graph::AmdEventKind::SplitBarrierWait {
                    barrier,
                    generation,
                } => {
                    info.push(("barrier object", barrier.to_string()));
                    info.push(("generation", generation.to_string()));
                    (
                        "wait",
                        format!("s_barrier_wait {barrier} phase {generation}"),
                    )
                }
                volta_graph::AmdEventKind::SplitBarrierRelease {
                    barrier,
                    generation,
                } => {
                    info.push(("barrier object", barrier.to_string()));
                    info.push(("generation", generation.to_string()));
                    (
                        "resume",
                        format!("barrier {barrier} phase {generation} released"),
                    )
                }
                volta_graph::AmdEventKind::NamedBarrierInit { object, expected } => {
                    info.push((
                        "named object",
                        format!("{} incarnation {}", object.id, object.incarnation),
                    ));
                    info.push(("expected arrivals", expected.to_string()));
                    ("sync", format!("named init {} count {expected}", object.id))
                }
                volta_graph::AmdEventKind::NamedBarrierJoin { object, phase }
                | volta_graph::AmdEventKind::NamedBarrierSignal { object, phase, .. }
                | volta_graph::AmdEventKind::NamedBarrierSignalWithCount {
                    object, phase, ..
                }
                | volta_graph::AmdEventKind::NamedBarrierWait { object, phase, .. }
                | volta_graph::AmdEventKind::NamedBarrierRelease { object, phase }
                | volta_graph::AmdEventKind::NamedBarrierLeave { object, phase } => {
                    info.push((
                        "named object",
                        format!("{} incarnation {}", object.id, object.incarnation),
                    ));
                    info.push(("phase", phase.to_string()));
                    let operation = match record.kind {
                        volta_graph::AmdEventKind::NamedBarrierJoin { .. } => "join",
                        volta_graph::AmdEventKind::NamedBarrierSignal { exec_mask, .. } => {
                            info.push(("EXEC", format!("{exec_mask:#x}")));
                            "signal"
                        }
                        volta_graph::AmdEventKind::NamedBarrierSignalWithCount {
                            exec_mask,
                            expected,
                            ..
                        } => {
                            info.push(("EXEC", format!("{exec_mask:#x}")));
                            info.push(("updated expected arrivals", expected.to_string()));
                            "signal"
                        }
                        volta_graph::AmdEventKind::NamedBarrierWait { encoded, .. } => {
                            info.push(("encoded WAIT selector", encoded.to_string()));
                            "wait"
                        }
                        volta_graph::AmdEventKind::NamedBarrierRelease { .. } => "release",
                        _ => "leave",
                    };
                    (
                        if operation == "wait" {
                            "wait"
                        } else if operation == "release" {
                            "resume"
                        } else {
                            "sync"
                        },
                        format!("named {operation} {} phase {phase}", object.id),
                    )
                }
                volta_graph::AmdEventKind::MatrixIssue {
                    kind,
                    destinations,
                    sources_ab,
                    accumulator,
                } => {
                    info.push(("matrix form", format!("{kind:?}")));
                    info.push(("destinations", format!("{destinations:?}")));
                    info.push(("A/B sources", format!("{sources_ab:?}")));
                    info.push(("accumulator", format!("{accumulator:?}")));
                    ("issue", format!("wave {} MFMA {kind:?}", record.wave))
                }
                volta_graph::AmdEventKind::PrefetchFootprint {
                    issue,
                    scope,
                    lanes,
                } => {
                    info.push(("issue", format!("{issue:?}")));
                    info.push(("scope", scope.name().into()));
                    info.push(("lanes", lanes.len().to_string()));
                    ("instruction", "prefetch address footprint (no data)".into())
                }
                volta_graph::AmdEventKind::GlobalAccessFault {
                    access,
                    bytes_per_lane,
                    lanes,
                } => {
                    info.push(("access", format!("{access:?}")));
                    info.push(("bytes per lane", bytes_per_lane.to_string()));
                    info.push(("lanes", format!("{lanes:x?}")));
                    (
                        "instruction",
                        "attempted global access outside objects; aborted before issue".into(),
                    )
                }
                volta_graph::AmdEventKind::ModeAccess {
                    op,
                    offset,
                    width,
                    value,
                } => {
                    info.push(("bits", format!("MODE[{}:{offset}]", offset + width - 1)));
                    info.push(("value", format!("{value:#x}")));
                    (
                        "instruction",
                        format!(
                            "{} {}",
                            op.name(),
                            if op.writes() {
                                "writes MODE"
                            } else {
                                "reads MODE"
                            }
                        ),
                    )
                }
                volta_graph::AmdEventKind::BufferRangeCheck { issue, check } => {
                    info.push(("issue", issue.0.to_string()));
                    info.push(("num_records", check.num_records.to_string()));
                    info.push(("stride", check.stride.to_string()));
                    info.push(("OOB_SELECT", u8::from(check.oob_select).to_string()));
                    let suppressed = check
                        .lanes
                        .iter()
                        .map(|l| check.dwords - l.in_range.count_ones() as u8)
                        .map(u64::from)
                        .sum::<u64>();
                    (
                        "instruction",
                        format!(
                            "buffer range check: {suppressed} out-of-range DWORDs (loads read zero, stores dropped)"
                        ),
                    )
                }
                volta_graph::AmdEventKind::AluWait { wait } => {
                    info.push(("encoded", format!("{:#06x}", wait.encoded)));
                    ("wait", "s_wait_alu (ALU dependency counters only)".into())
                }
                volta_graph::AmdEventKind::Delay { cycles } => {
                    info.push(("independent wait states", cycles.to_string()));
                    (
                        "wait",
                        format!("s_nop {} ({cycles} wait states)", cycles.saturating_sub(1)),
                    )
                }
                volta_graph::AmdEventKind::Instruction {
                    exec_mask,
                    reads,
                    writes,
                    scalar_mask_reads,
                } => {
                    info.push(("EXEC", amd_mask(graph, *exec_mask)));
                    info.push(("reads", format!("{reads:?}")));
                    info.push(("writes", format!("{writes:?}")));
                    info.push(("carry/condition masks", format!("{scalar_mask_reads:?}")));
                    ("instruction", format!("wave {} instruction", record.wave))
                }
                volta_graph::AmdEventKind::MemoryIssue {
                    class,
                    counters,
                    destinations,
                    held_sources,
                } => {
                    info.push(("class", format!("{class:?}")));
                    info.push(("counter charges", format!("{counters:?}")));
                    info.push(("pending destinations", format!("{destinations:?}")));
                    info.push(("held sources", format!("{held_sources:?}")));
                    ("issue", format!("wave {} {class:?} issue", record.wave))
                }
                volta_graph::AmdEventKind::Wait {
                    vmcnt,
                    lgkmcnt,
                    expcnt,
                } => {
                    info.push(("vmcnt", format!("{vmcnt:?}")));
                    info.push(("lgkmcnt", format!("{lgkmcnt:?}")));
                    info.push(("expcnt", format!("{expcnt:?}")));
                    (
                        "wait",
                        format!("s_waitcnt vm={vmcnt:?} lgkm={lgkmcnt:?} exp={expcnt:?}"),
                    )
                }
                volta_graph::AmdEventKind::BarrierArrive {
                    generation,
                    exec_mask,
                } => {
                    info.push(("generation", generation.to_string()));
                    info.push(("EXEC", amd_mask(graph, *exec_mask)));
                    ("sync", format!("s_barrier arrive #{generation}"))
                }
                volta_graph::AmdEventKind::BarrierRelease { generation } => {
                    info.push(("generation", generation.to_string()));
                    ("resume", format!("s_barrier release #{generation}"))
                }
            }
        }
        EventKind::Access(a) => {
            let a = graph.access(a);
            let mode = match a.mode {
                AccessMode::Read => "read",
                AccessMode::Write => "write",
            };
            info.push(("space", space_name(a.space).to_string()));
            info.push(("footprint", footprint(a.footprint)));
            info.push((
                "proxy",
                match a.proxy {
                    Proxy::Generic => "generic",
                    Proxy::Async => "async",
                    Proxy::TensorMap => "tensormap",
                }
                .to_string(),
            ));
            info.push((
                "ordering",
                match a.ordering {
                    OrderingKind::Ordinary => "ordinary",
                    OrderingKind::Atomic => "atomic",
                    OrderingKind::Reduction => "reduction",
                }
                .to_string(),
            ));
            info.push(("operand", format!("{:?}", a.role)));
            info.push(("attributed to", thread_name(graph, a.attributed)));
            if let Some(v) = a.value {
                info.push(("value", stored_value(arena, graph.value(v))));
            }
            (
                mode,
                format!("{mode} {} {}", space_name(a.space), footprint(a.footprint)),
            )
        }
        EventKind::Contribute(g) => {
            let how = graph
                .generation(g)
                .contributions
                .iter()
                .find(|c| c.event == id)
                .map_or("contribution".to_string(), |c| contribution(c.kind));
            info.push(("generation", generation_name(graph, g)));
            info.push(("contribution", how.clone()));
            ("sync", format!("{how} → {}", generation_name(graph, g)))
        }
        EventKind::WaitStart { object, parity } => {
            info.push(("object", object_name(graph, object)));
            info.push(("parity", parity_bit(parity).to_string()));
            (
                "wait",
                format!(
                    "wait {} parity {}",
                    object_name(graph, object),
                    parity_bit(parity)
                ),
            )
        }
        EventKind::Resume(g) => {
            info.push(("generation", generation_name(graph, g)));
            ("resume", format!("resume ← {}", generation_name(graph, g)))
        }
        EventKind::GenerationComplete(g) => {
            let genr = graph.generation(g);
            info.push(("generation", generation_name(graph, g)));
            info.push(("kind", generation_kind(graph, genr.kind)));
            info.push(("contributions", genr.contributions.len().to_string()));
            info.push(("releases", genr.releases.len().to_string()));
            info.push((
                "surplus arrivals",
                genr.evidence.surplus_arrivals.to_string(),
            ));
            info.push(("pending tx", genr.evidence.pending_tx.to_string()));
            (
                "generation",
                format!("{} complete", generation_name(graph, g)),
            )
        }
        EventKind::AsyncIssue(op) => {
            async_info(graph, op, &mut info);
            ("issue", format!("issue {}", async_name(graph, op)))
        }
        EventKind::AsyncComplete(op) => {
            async_info(graph, op, &mut info);
            (
                "async-complete",
                format!("complete {}", async_name(graph, op)),
            )
        }
        EventKind::GroupCommit(family) => (
            "commit",
            match family {
                GroupFamily::CpAsync => "cp.async.commit_group",
                GroupFamily::Wgmma => "wgmma.commit_group",
                GroupFamily::Bulk => "cp.async.bulk.commit_group",
            }
            .to_string(),
        ),
        EventKind::Fence(kind) => (
            "fence",
            match kind {
                FenceKind::ProxyAsync { space: Some(s) } => {
                    format!("fence.proxy.async {}", space_name(s))
                }
                FenceKind::ProxyAsync { space: None } => "fence.proxy.async".to_string(),
                FenceKind::Tcgen05BeforeThreadSync => {
                    "tcgen05.fence::before_thread_sync".to_string()
                }
                FenceKind::Tcgen05AfterThreadSync => "tcgen05.fence::after_thread_sync".to_string(),
                FenceKind::TensormapRelease => "fence.proxy.tensormap::generic.release".to_string(),
                FenceKind::TensormapAcquire { addr } => {
                    format!("fence.proxy.tensormap::generic.acquire [{addr:#x}]")
                }
            },
        ),
        EventKind::ObjectInit(o) => ("lifetime", format!("init {}", object_name(graph, o))),
        EventKind::ObjectInval(o) => ("lifetime", format!("inval {}", object_name(graph, o))),
        EventKind::TmemAlloc {
            start_col,
            num_cols,
        } => (
            "tmem",
            format!("tcgen05.alloc cols [{start_col}, +{num_cols})"),
        ),
        EventKind::TmemDealloc {
            start_col,
            num_cols,
        } => (
            "tmem",
            format!("tcgen05.dealloc cols [{start_col}, +{num_cols})"),
        ),
        EventKind::TmemRelinquish => ("tmem", "tcgen05.relinquish_alloc_permit".to_string()),
        EventKind::Exit => ("exit", "exit".to_string()),
    };
    (category, label, info)
}

fn async_info(graph: &EventGraph, op: AsyncOpId, info: &mut Info) {
    let a = graph.async_op(op);
    info.push(("operation", async_name(graph, op)));
    info.push(("issuer", thread_name(graph, a.issuer)));
    info.push(("effects", a.effects.len().to_string()));
    if !a.tracking.is_empty() {
        let tracking: Vec<String> = a
            .tracking
            .iter()
            .map(|&g| generation_name(graph, g))
            .collect();
        info.push(("tracked by", tracking.join(", ")));
    }
}

fn context_name(graph: &EventGraph, ctx: ExecContext) -> String {
    match ctx {
        ExecContext::Thread(t) => format!("thread {}", thread_name(graph, t)),
        ExecContext::Async(op) => format!("async {}", async_name(graph, op)),
        ExecContext::Generation(g) => format!("generation {}", generation_name(graph, g)),
    }
}

fn async_name(graph: &EventGraph, op: AsyncOpId) -> String {
    format!("{} a{}", graph.async_op(op).kind.name(), op.0)
}

fn object_name(graph: &EventGraph, object: SyncObjectId) -> String {
    match &graph.object(object).kind {
        SyncObjectKind::NamedBarrier { id } => format!("barrier {id}"),
        SyncObjectKind::Mbarrier {
            addr,
            expected_arrivals,
        } => format!("mbarrier {addr:#x} (count {expected_arrivals})"),
        SyncObjectKind::WarpCollective { warp } => format!("warp {warp} collectives"),
        SyncObjectKind::WarpgroupCollective { warpgroup } => {
            format!("warpgroup {warpgroup} collectives")
        }
        SyncObjectKind::AsyncGroupWait { thread }
            if matches!(graph.provenance().unit, ThreadUnit::Wave { .. }) =>
        {
            format!("LGKM waits {}", thread_name(graph, *thread))
        }
        SyncObjectKind::AsyncGroupWait { thread } => format!("cp.async groups t{}", thread.0),
        SyncObjectKind::BulkGroupWait { thread } => format!("bulk groups t{}", thread.0),
    }
}

/// Physical native work items retain `w:l`; wave-unit traces use one `w` lane.
fn thread_name(graph: &EventGraph, t: ThreadIx) -> String {
    match graph.provenance().unit {
        ThreadUnit::CtaThread if amd_graph(graph) => format!(
            "w{}:l{}",
            t.0 / graph_wave_size(graph),
            t.0 % graph_wave_size(graph)
        ),
        ThreadUnit::CtaThread => format!("t{}", t.0),
        ThreadUnit::Wave { .. } => format!("w{}", t.0),
    }
}

fn generation_name(graph: &EventGraph, g: GenerationId) -> String {
    let genr = graph.generation(g);
    match genr.kind {
        GenerationKind::MbarrierPreInit => format!("{} pre-init", object_name(graph, genr.object)),
        _ => format!("{} #{}", object_name(graph, genr.object), genr.index),
    }
}

fn generation_kind(graph: &EventGraph, kind: GenerationKind) -> String {
    match kind {
        GenerationKind::Barrier {
            arrivals: BarrierArrivals::FullCta,
        } => "barrier of the full CTA".to_string(),
        GenerationKind::Barrier {
            arrivals: BarrierArrivals::Count(n),
        } => format!("barrier of {n} threads"),
        GenerationKind::MbarrierPhase => "mbarrier phase".to_string(),
        GenerationKind::MbarrierPreInit => "mbarrier phase before init".to_string(),
        GenerationKind::Collective {
            pc,
            mask,
            side,
            ordering,
        } => format!(
            "collective {} at pc {}, mask {mask:#010x}{}",
            match side {
                CollectiveSide::Entry => "entry",
                CollectiveSide::Exit => "exit",
            },
            pc.0,
            match ordering {
                MemberOrdering::Synchronizing => "",
                MemberOrdering::ExecutionOnly => ", execution only",
            }
        ),
        GenerationKind::AsyncGroupWait { keep_pending }
            if matches!(graph.provenance().unit, ThreadUnit::Wave { .. }) =>
        {
            format!("lgkmcnt({keep_pending})")
        }
        GenerationKind::AsyncGroupWait { keep_pending } => {
            format!("cp.async.wait_group {keep_pending}")
        }
        GenerationKind::BulkGroupWait {
            keep_pending,
            scope,
        } => match scope {
            BulkWaitScope::SourceReads => format!("cp.async.bulk.wait_group.read {keep_pending}"),
            BulkWaitScope::Complete => format!("cp.async.bulk.wait_group {keep_pending}"),
        },
    }
}

fn contribution(kind: ContributionKind) -> String {
    match kind {
        ContributionKind::Blocking => "blocking arrival".to_string(),
        ContributionKind::Arrival {
            count,
            expect_tx,
            ordering,
        } => {
            let tx = match expect_tx {
                0 => String::new(),
                bytes => format!(", expect_tx {bytes}"),
            };
            let how = match ordering {
                ArrivalOrdering::Release => "",
                ArrivalOrdering::AsyncTracked => " once its tracked ops complete",
            };
            format!("arrive ×{count}{tx}{how}")
        }
        ContributionKind::Exited => "exited".to_string(),
        ContributionKind::AsyncCompletion { op, tx_bytes } => {
            format!("completion of a{} ({tx_bytes} tx bytes)", op.0)
        }
        ContributionKind::CompleteTx { bytes } => format!("complete_tx {bytes}"),
        ContributionKind::ExpectTx { bytes } => format!("expect_tx {bytes}"),
    }
}

fn parity_bit(p: Parity) -> u8 {
    match p {
        Parity::Even => 0,
        Parity::Odd => 1,
    }
}

fn space_name(s: Space) -> &'static str {
    match s {
        Space::Global => "global",
        Space::Shared => "shared",
        Space::Tensor => "tmem",
    }
}

fn footprint(f: Footprint) -> String {
    match f {
        Footprint::Bytes { addr, width } => format!("[{addr:#x}, +{width})"),
        Footprint::Tensor {
            lane,
            lanes,
            col,
            cols,
        } => format!("lanes [{lane}, +{lanes}) × cols [{col}, +{cols})"),
    }
}

fn stored_value(arena: &ExprArena, value: StoredValue) -> String {
    let render = |e| arena.display_expr_prefix(e, MAX_VALUE_BYTES);
    match value {
        StoredValue::Scalar(e) => render(e),
        StoredValue::Pair(lo, hi) => format!("({}, {})", render(lo), render(hi)),
        StoredValue::Quad(lanes) => {
            let lanes: Vec<String> = lanes.into_iter().map(render).collect();
            format!("({})", lanes.join(", "))
        }
        StoredValue::Opaque => "opaque".to_string(),
    }
}

// =========================================================================
// Terminal state, checks and conclusion
// =========================================================================

fn terminal_name(terminal: &Terminal) -> &'static str {
    match terminal {
        Terminal::Completed => "completed",
        Terminal::Stuck { .. } => "stuck",
        Terminal::Aborted { .. } => "aborted",
    }
}

/// A thread's last event: where a stuck or aborted thread stopped.
fn last_event(graph: &EventGraph, thread: ThreadIx) -> Vec<Json> {
    graph
        .program_order(thread)
        .last()
        .map(|e| Json::uint(e.0))
        .into_iter()
        .collect()
}

fn terminal(graph: &EventGraph) -> Json {
    let mut fields = vec![("kind", Json::str(terminal_name(graph.terminal())))];
    let notes: Vec<Json> = match graph.terminal() {
        Terminal::Completed => Vec::new(),
        Terminal::Stuck { blocked } => blocked
            .iter()
            .map(|b| {
                let on = match &b.on {
                    BlockedOn::AmdNamedBarrier { object, phase, expected, arrived } => format!("waits at AMD named barrier {} incarnation {} phase {phase}: {arrived}/{expected} arrivals", object.id, object.incarnation),
                    BlockedOn::AmdBarrier {
                        generation,
                        missing_waves,
                    } => format!("waits at AMD barrier #{generation} for waves {missing_waves:?}"),
                    BlockedOn::Barrier { generation } => {
                        format!("waits in {}", generation_name(graph, *generation))
                    }
                    BlockedOn::Collective {
                        generation,
                        missing,
                    } => {
                        let missing: Vec<String> =
                            missing.iter().map(|t| format!("t{}", t.0)).collect();
                        format!(
                            "waits in {} for {}",
                            generation_name(graph, *generation),
                            missing.join(", ")
                        )
                    }
                    BlockedOn::Mbarrier {
                        object,
                        parity,
                        state,
                    } => format!(
                        "waits for {} parity {} ({} phases completed, {}/{} arrived, {} tx \
                         pending)",
                        object_name(graph, *object),
                        parity_bit(*parity),
                        state.completed_phases,
                        state.arrived,
                        state.expected,
                        state.pending_tx
                    ),
                };
                finding(
                    format!("t{} at pc {} {on}", b.thread.0, b.pc.0),
                    last_event(graph, b.thread),
                )
            })
            .collect(),
        Terminal::Aborted {
            thread, pc, reason, ..
        } => {
            let at = match (thread, pc) {
                (Some(t), Some(pc)) => format!("{} at pc {}: ", thread_name(graph, *t), pc.0),
                (Some(t), None) => format!("{}: ", thread_name(graph, *t)),
                (None, _) => String::new(),
            };
            let events = thread.map_or(Vec::new(), |t| last_event(graph, t));
            vec![finding(format!("{at}{reason}"), events)]
        }
    };
    fields.push(("notes", Json::Arr(notes)));
    Json::Obj(fields)
}

fn finding(text: String, events: Vec<Json>) -> Json {
    Json::Obj(vec![
        ("text", Json::Str(text)),
        ("events", Json::Arr(events)),
    ])
}

fn event_list(events: &[EventId]) -> Vec<Json> {
    events.iter().map(|e| Json::uint(e.0)).collect()
}

/// A check's findings: each violation (`Fail`) or unresolved obligation (`Unknown`), with the
/// events it concerns.
fn findings<E, V>(
    limit: usize,
    result: &PropertyResult<E, V>,
    violation: impl Fn(&V) -> (String, Vec<EventId>),
) -> Vec<Json> {
    match result {
        PropertyResult::Pass(_) => Vec::new(),
        PropertyResult::Fail(violations) => violations
            .iter()
            .take(limit)
            .map(|v| {
                let (text, events) = violation(v);
                finding(text, event_list(&events))
            })
            .collect(),
        PropertyResult::Unknown(unresolved) => unresolved
            .iter()
            .take(limit)
            .map(|u| finding(u.reason.clone(), event_list(&u.events)))
            .collect(),
    }
}

fn epoch_violation(graph: &EventGraph, v: &EpochViolation) -> (String, Vec<EventId>) {
    let mut events = vec![v.synchronization, v.operation];
    events.extend(v.predecessor);
    (
        format!(
            "e{} of {} is not separated from synchronization e{} of {}",
            v.operation.0,
            generation_name(graph, v.next),
            v.synchronization.0,
            generation_name(graph, v.generation)
        ),
        events,
    )
}

fn conflict_kind(kind: ConflictKind) -> &'static str {
    match kind {
        ConflictKind::WriteRead => "write → read",
        ConflictKind::WriteWrite => "write → write",
        ConflictKind::ReadWrite => "read → write",
    }
}

fn lifetime_violation(v: &LifetimeViolation) -> (String, Vec<EventId>) {
    match v {
        LifetimeViolation::DoubleFree { alloc, deallocs } => {
            let mut events = vec![*alloc];
            events.extend(deallocs);
            (
                format!("allocation e{} is freed {} times", alloc.0, deallocs.len()),
                events,
            )
        }
        LifetimeViolation::NeverDeallocated { alloc } => (
            format!("allocation e{} is never freed", alloc.0),
            vec![*alloc],
        ),
        LifetimeViolation::UnmatchedDeallocation { dealloc } => (
            format!("e{} frees columns no allocation provides", dealloc.0),
            vec![*dealloc],
        ),
        LifetimeViolation::Uncovered { effect, alloc } => {
            let mut events = vec![*effect];
            events.extend(alloc);
            (
                format!("no allocation covers the footprint of e{}", effect.0),
                events,
            )
        }
        LifetimeViolation::UseNotBeforeDeallocation {
            effect,
            alloc,
            dealloc,
        } => (
            format!(
                "use e{} is not ordered before the deallocation e{}",
                effect.0, dealloc.0
            ),
            vec![*alloc, *effect, *dealloc],
        ),
        LifetimeViolation::AllocationNotBeforeUse { effect, alloc } => (
            format!(
                "allocation e{} is not ordered before the use e{}",
                alloc.0, effect.0
            ),
            vec![*alloc, *effect],
        ),
    }
}

fn allocation_order_violation(v: &AllocationOrderViolation) -> (String, Vec<EventId>) {
    match *v {
        AllocationOrderViolation::Unordered { earlier, later } => (
            format!("allocations e{} and e{} are unordered", earlier.0, later.0),
            vec![earlier, later],
        ),
        AllocationOrderViolation::IncreasingColumnCount {
            earlier,
            earlier_cols,
            later,
            later_cols,
        } => (
            format!(
                "allocation e{} takes {} columns, more than the {} of e{} before it",
                later.0, later_cols, earlier_cols, earlier.0
            ),
            vec![earlier, later],
        ),
    }
}

fn missing_fence(missing: MissingFence) -> &'static str {
    match missing {
        MissingFence::ProxyAsync => "missing fence.proxy.async",
        MissingFence::TensormapRelease => "missing fence.proxy.tensormap::generic.release",
        MissingFence::TensormapAcquire => "missing fence.proxy.tensormap::generic.acquire",
    }
}

fn allocation_permit_violation(v: &AllocationPermitViolation) -> (String, Vec<EventId>) {
    (
        format!(
            "allocation e{} is not ordered before relinquish e{}",
            v.allocation.0, v.relinquish.0
        ),
        vec![v.allocation, v.relinquish],
    )
}

fn release_order_violation(
    graph: &EventGraph,
    v: &ReleaseOrderViolation,
) -> (String, Vec<EventId>) {
    match *v {
        ReleaseOrderViolation::LateStart { phase, release } => (
            format!(
                "the first release e{} is of {}: phase 0 released no thread",
                release.0,
                generation_name(graph, phase)
            ),
            vec![release],
        ),
        ReleaseOrderViolation::SkippedPhase {
            previous,
            phase,
            release,
        } => (
            format!(
                "release e{} of {} follows {}: the phase between released no thread",
                release.0,
                generation_name(graph, phase),
                generation_name(graph, previous)
            ),
            vec![release],
        ),
        ReleaseOrderViolation::Unordered { earlier, later } => (
            format!(
                "release e{} is not ordered before e{}, a release of the next phase",
                earlier.0, later.0
            ),
            vec![earlier, later],
        ),
    }
}

fn checks(graph: &EventGraph, report: &GraphReport, limit: usize) -> Json {
    Json::Arr(
        report
            .applicable_properties()
            .iter()
            .map(|&property| {
                let summary = report.summary(property);
                let findings = match property {
                    Property::AmdTrace => report
                        .amd
                        .as_ref()
                        .map(|a| {
                            findings(limit, &a.trace, |u| (u.reason.clone(), u.events.clone()))
                        })
                        .unwrap_or_default(),
                    Property::AmdWaitReadiness => report
                        .amd
                        .as_ref()
                        .map(|a| {
                            findings(limit, &a.wait_readiness, |h| {
                                (
                                    format!(
                                        "wave {} {:?}: {:?}{} {} is pending from e{} at e{}",
                                        h.wave,
                                        h.kind,
                                        h.register.file,
                                        h.register.index,
                                        if h.register.is_uniform() {
                                            "uniform".into()
                                        } else {
                                            format!("lanes {}", amd_mask(graph, h.register.lanes))
                                        },
                                        h.issue.0,
                                        h.use_event.0
                                    ),
                                    vec![h.issue, h.use_event],
                                )
                            })
                        })
                        .unwrap_or_default(),
                    Property::AmdBarrier => report
                        .amd
                        .as_ref()
                        .map(|a| {
                            findings(limit, &a.barrier, |v| {
                                (
                                    format!(
                                        "wave {} barrier #{}: {}",
                                        v.wave, v.generation, v.reason
                                    ),
                                    vec![v.event],
                                )
                            })
                        })
                        .unwrap_or_default(),
                    Property::AmdMemoryConflict => report
                        .amd
                        .as_ref()
                        .map(|a| {
                            findings(limit, &a.memory_conflict, |c| {
                                (
                                    format!(
                                        "{} conflict at {} {:#x}: e{} and e{} lack AMD ordering",
                                        conflict_kind(c.kind),
                                        space_name(c.location.space),
                                        c.location.addr,
                                        c.earlier.0,
                                        c.later.0
                                    ),
                                    vec![c.earlier, c.later],
                                )
                            })
                        })
                        .unwrap_or_default(),
                    Property::NamedBarrierEpoch => {
                        findings(limit, &report.named_barrier_epoch, |v| {
                            epoch_violation(graph, v)
                        })
                    }
                    Property::MbarrierEpoch => {
                        findings(limit, &report.mbarrier_epoch, |v| epoch_violation(graph, v))
                    }
                    Property::MbarrierOrdering => findings(limit, &report.mbarrier_ordering, |v| {
                        (
                            format!(
                                "phase {} of the mbarrier does not balance: {} of {} \
                                 arrival(s), {} transaction byte(s) net",
                                graph.generation(v.phase).index,
                                v.arrivals,
                                v.expected_arrivals,
                                v.net_tx
                            ),
                            graph
                                .generation(v.phase)
                                .contributions
                                .iter()
                                .map(|c| c.event)
                                .collect(),
                        )
                    }),
                    Property::MemoryConflict => findings(limit, &report.memory_conflict, |c| {
                        (
                            format!(
                                "{} conflict at {} {:#x}: e{} and e{} are unordered",
                                conflict_kind(c.kind),
                                space_name(c.location.space),
                                c.location.addr,
                                c.earlier.0,
                                c.later.0
                            ),
                            vec![c.earlier, c.later],
                        )
                    }),
                    Property::ProxyOrder => findings(limit, &report.proxy_order, |v| {
                        (
                            format!(
                                "{} conflict at {} {:#x}: no proxy fence orders generic e{} \
                                 before e{} ({})",
                                conflict_kind(v.kind),
                                space_name(v.location.space),
                                v.location.addr,
                                v.generic.0,
                                v.asynchronous.0,
                                missing_fence(v.missing)
                            ),
                            vec![v.generic, v.asynchronous],
                        )
                    }),
                    Property::TensorOrder => findings(limit, &report.tensor_order, |c| {
                        (
                            format!(
                                "{} conflict at tmem lane {}, column {}: e{} and e{} are \
                                 unordered",
                                conflict_kind(c.kind),
                                c.cell.lane,
                                c.cell.col,
                                c.earlier.0,
                                c.later.0
                            ),
                            vec![c.earlier, c.later],
                        )
                    }),
                    Property::ResourceLifetime => {
                        findings(limit, &report.resource_lifetime, lifetime_violation)
                    }
                    Property::AllocationOrder => {
                        findings(limit, &report.allocation_order, allocation_order_violation)
                    }
                    Property::AllocationPermit => findings(
                        limit,
                        &report.allocation_permit,
                        allocation_permit_violation,
                    ),
                    Property::MbarrierReleaseOrder => {
                        findings(limit, &report.mbarrier_release_order, |v| {
                            release_order_violation(graph, v)
                        })
                    }
                };
                let mut fields = vec![
                    ("name", Json::str(property.name())),
                    ("algorithm", Json::str(property.algorithm())),
                    (
                        "verdict",
                        Json::str(match summary.verdict {
                            Verdict::Pass => "pass",
                            Verdict::Fail => "fail",
                            Verdict::Unknown => "unknown",
                        }),
                    ),
                    ("evidence", Json::opt(summary.evidence, Json::Str)),
                    ("findings", Json::Arr(findings)),
                ];
                if limit != usize::MAX {
                    fields.push(("totalFindings", Json::uint(summary.findings.len() as u64)));
                }
                Json::Obj(fields)
            })
            .collect(),
    )
}

fn conclusion(graph: &EventGraph, report: &GraphReport) -> Json {
    let reasons = match report.conclusion() {
        GraphConclusion::Verified => Vec::new(),
        GraphConclusion::Rejected { failed, deadlock } => {
            let mut reasons: Vec<String> = failed
                .iter()
                .map(|s| format!("{} failed", s.property.name()))
                .collect();
            if let Some(d) = deadlock {
                reasons.push(format!("deadlock: {} thread(s) blocked", d.blocked.len()));
            }
            reasons
        }
        GraphConclusion::Unknown { reasons } => reasons,
    };
    Json::Obj(vec![
        ("status", Json::str(Status::checked(graph, report).name())),
        (
            "reasons",
            Json::Arr(reasons.into_iter().map(Json::Str).collect()),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;
    use volta_graph::{
        GraphBuilder, InstructionInfo, MemAccess, OperandRole, ProgramPoint, Provenance,
    };
    use volta_graph_check::check_graph;

    fn field<'a>(value: &'a Json, key: &str) -> &'a Json {
        let Json::Obj(fields) = value else {
            panic!("object required")
        };
        &fields.iter().find(|(name, _)| *name == key).unwrap().1
    }

    #[test]
    fn large_display_preserves_full_rejection_or_unknown_and_archival_events() {
        for ordering in [OrderingKind::Ordinary, OrderingKind::Atomic] {
            let mut builder = GraphBuilder::new(Provenance {
                unit: volta_graph::ThreadUnit::CtaThread,
                kernel: "large_display".into(),
                block_dim: (2, 1, 1),
                grid_dim: (1, 1, 1),
                workgroup_id: (0, 0, 0),
                targets: vec!["Sm100a".into()],
                assumptions: vec![],
                instructions: vec![InstructionInfo {
                    kind: "Store",
                    span: None,
                }],
            });
            for n in 0..50_021 {
                builder.fence(
                    ThreadIx(n % 2),
                    ProgramPoint(0),
                    FenceKind::ProxyAsync { space: None },
                );
            }
            let mut arena = ExprArena::new();
            let value = builder.value(StoredValue::Scalar(arena.int(7)));
            for t in 0..2 {
                builder.thread_access(
                    ThreadIx(t),
                    ProgramPoint(0),
                    MemAccess {
                        space: Space::Global,
                        mode: AccessMode::Write,
                        ordering,
                        proxy: Proxy::Generic,
                        footprint: Footprint::Bytes { addr: 0, width: 4 },
                        role: OperandRole::Plain,
                        attributed: ThreadIx(t),
                        value: Some(value),
                    },
                );
            }
            for t in 0..2 {
                builder.exit(ThreadIx(t), ProgramPoint(0));
            }
            let graph = builder.finish(Terminal::Completed);
            let report = check_graph(&graph);
            let expected = if ordering == OrderingKind::Atomic {
                "unknown"
            } else {
                "race-condition"
            };
            let before = report.clone();
            let id = RunId {
                suite: "display".into(),
                test: "large".into(),
                ordinal: NonZeroU32::MIN,
            };
            let run = Run {
                id: &id,
                ptx_file: None,
                source: "complete source",
                outcome: Outcome::Executed {
                    graph: &graph,
                    arena: &arena,
                    report: Some(&report),
                },
            };
            let projected = document(&run);
            assert_eq!(
                field(field(&projected, "meta"), "status"),
                &Json::str(expected),
                "{:?}",
                report.conclusion()
            );
            assert_eq!(field(&projected, "source"), &Json::str("complete source"));
            let Json::Arr(events) = field(&projected, "events") else {
                panic!()
            };
            assert_eq!(events.len(), display::EVENT_LIMIT);
            let Json::Arr(original_ids) = field(&projected, "eventIds") else {
                panic!()
            };
            assert_eq!(original_ids.len(), events.len());
            assert!(original_ids.contains(&Json::uint(50_021u32)));
            assert!(original_ids.contains(&Json::uint(50_022u32)));
            assert_eq!(
                field(field(&projected, "display"), "totalEvents"),
                &Json::uint(graph.events().len() as u64)
            );
            let Json::Arr(complete) = field(&complete_document(&run), "events").clone() else {
                panic!()
            };
            assert_eq!(complete.len(), graph.events().len());
            assert_eq!(report.conclusion(), before.conclusion());
        }
    }

    #[test]
    fn offsets_map_to_one_based_lines() {
        let lines = LineIndex::new("ab\ncd\n\nef");
        assert_eq!(lines.line(0), Some(1));
        assert_eq!(lines.line(2), Some(1));
        assert_eq!(lines.line(3), Some(2));
        assert_eq!(lines.line(6), Some(3));
        assert_eq!(lines.line(7), Some(4));
        assert_eq!(lines.line(10), None);
    }
}

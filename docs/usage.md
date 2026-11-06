# Using GPU Kernel Verifier

Run these examples from the repository root. `volta` means the locally built
`target/release/volta` executable, or the same executable installed on your PATH.
Use `volta COMMAND --help` for the complete option list.

## Describe the launch

The analyzer needs the kernel's geometry, memory allocations and arguments.
Array sizes are logical bounds, not actual GPU allocations. Symbolic input
values remain unknown; addresses and control flow must be concrete.

Create `copy.json` for the native fixture used below:

```json
{
  "block": 65,
  "grid": [1, 1, 1],
  "arrays": [
    {"name": "x", "base": "0x10000", "dtype": "f32", "shape": [65], "kind": "in"},
    {"name": "out", "base": "0x30000", "dtype": "f32", "shape": [65], "kind": "out"}
  ],
  "params": [{"ptr": "x"}, {"ptr": "out"}, {"int": 65}]
}
```

`block` is required. Geometry accepts a number or one to three dimensions;
omitted dimensions are one. Array kinds are `in`, `out`, `inout` and `index`.
`index` supplies concrete element values equal to their indices. Use `inout`
when the computation depends on the initial output, such as atomic accumulation.
An array can use `elem_width`/`len` instead of `dtype`/`shape`. Omitted bases
are placed consecutively at 64 KiB boundaries, starting at `0x10000`.
`dtype` is a size shorthand, not a typed-array assertion; instructions and their
consumers determine how raw words are interpreted.

Arguments are positional. Both backends accept `ptr`, `int`, `float` and `sym`.
PTX also accepts `struct` fields for a declared by-value aggregate, including
host tensor maps. Native AMD instead accepts `bytes` with the exact ABI size,
including padding. Each entry has one value form; an optional `name` is
descriptive and does not change its position. See the
[parameter guide](launch-parameters.md) for field layouts, tensor maps and
which forms the CLI and Python interfaces expose.

Other launch fields are `description`, `dynamic_shared_bytes`, `globals`,
`max_instructions` and native `shared_allocations`. The latter declares disjoint
LDS byte ranges, for example `[{"name":"tile","offset":0,"size":256}]`.
These ranges are caller-supplied subobject premises; PTX rejects them. Unknown
JSON fields, invalid geometry and overlapping allocation ranges are errors.
An optional `kernel` sets the default entry for `analyze` and `verify`. Omit that
field for `compare`, which selects entries with `--kernel1` and `--kernel2`.

Native AMD analysis selects one workgroup. Its `workgroup_id` defaults to
`[0,0,0]`; set exactly three coordinates, for example `"workgroup_id": [1,2,3]`
inside `"grid": [2,3,4]`, or use `--workgroup-id 1,2,3` with launch flags.
Each coordinate must lie inside the grid. Local thread and wave indices remain
local to this group. Other workgroups and inter-workgroup conflicts are outside
the result. PTX currently rejects nonzero selection. Gfx1250's Y/Z coordinates
must fit its 16-bit architected fields.

Output indices remain absolute within the declared arrays. A selected tile that
writes only part of an array cannot pass a whole-array specification, including
with sampling. Selection never rebases outputs or fills missing ones.

`--config FILE` replaces launch flags and cannot be combined with `--block`,
`--grid`, `--array`, `--param`, `--global`, `--dyn-shared` or
`--shared-allocation`, `--workgroup-id`, `--early-readiness` or
`--early-memory-conflicts`.
Backend and target selection remain separate options.
See [launch_file.rs](../crates/volta_ptx_analysis/src/launch_file.rs) for the
shared JSON schema and validation rules.

## Parse and analyze

```sh
volta parse tests/gfx1250/fixtures/native-copy-add.s \
  --backend amdgpu --target gfx1250
volta analyze tests/gfx1250/fixtures/native-copy-add.s --config copy.json \
  --backend amdgpu --target gfx1250 --kernel copy --print-outputs 2
```

Parsing checks the supported source grammar. Analysis symbolically executes the
selected entry and applies the target's graph checks. The output includes array
element counts, instruction profiles, individual graph results and a final
`GRAPH VERIFICATION` status. It does not compare the computation with a reference.
`--print-outputs 0` suppresses expression rendering while retaining complete
execution and graph checks; `--no-profile` suppresses the instruction profile.

For PTX, omit `--backend` and `--target`, and provide its own source and matching
launch. Native AMD defaults to gfx950 if its target is omitted; selecting the
target explicitly makes a source mismatch an error.

Native gfx1250 source execution also offers opt-in
[early scalar readiness rejection](early-readiness.md). Set `early_readiness`
to `true` inside launch JSON, or pass `--early-readiness` with launch flags.
A certified finding prints `EARLY_READINESS: CERTIFIED_REJECTION`, its typed
witness and `VERDICT: REJECT` with exit code 1. The result contains an incomplete
prefix and no output expressions or equivalence claim. Ordinary execution is
the default; every acceptance still requires complete execution and checking.

Native gfx950/gfx1250 source execution also offers opt-in
[early same-instruction memory-conflict rejection](early-memory-conflicts.md).
Set `early_memory_conflicts` in launch JSON, or pass `--early-memory-conflicts`.
A validated finding prints `EARLY_MEMORY_CONFLICTS: CERTIFIED_REJECTION`, the
typed write/write pair and prefix counts, then `VERDICT: REJECT` with exit 1.
It returns no complete outputs or equivalence result; default execution remains
unchanged. Both early options may be enabled for gfx1250.

A supported gfx1250 tensor transfer outside the configured LDS extent uses
[certified tensor bounds rejection](tensor-bounds.md), without an opt-in flag.
It prints `TENSOR_BOUNDS: CERTIFIED_REJECTION` and `VERDICT: REJECT`, exits 1,
and returns no outputs or equivalence result. Unsupported descriptor modes and
unrelated aborts remain inconclusive.

## Verify a mathematical specification

```sh
volta verify tests/gfx1250/fixtures/native-copy-add.s \
  tests/gfx1250/specs/copy.spec --config copy.json --dim N=65 \
  --backend amdgpu --target gfx1250 --kernel copy --sample 0
```

The specification can be a file or inline text:

```text
dim N;
array x[N];
array out[N];
out[i] = x[i];
```

Bind every declared dimension with `--dim NAME=VALUE`. Every computed output
array must have the correct launch kind and complete index set. Equal element
counts alone do not establish equal footprints. The
[specification guide](../crates/volta_spec/README.md) describes reductions,
layouts and the expression language.

`--sample 0` checks all elements and is the default. A sample that omits elements
cannot produce full acceptance. `--verify-numeric` adds random f64 oracle
confirmation; it does not turn mathematical-real equivalence into a physical floating-point
proof. `--iterations N` repeats the solver with fresh sessions, and
`--recycle-terms N` controls term-table recycling (`0` disables recycling).

## Compare kernels and replay expressions

This identity comparison uses the same native copy source on both sides:

```sh
volta compare tests/gfx1250/fixtures/native-copy-add.s \
  tests/gfx1250/fixtures/native-copy-add.s --config copy.json \
  --kernel1 copy --kernel2 copy --check-array out \
  --backend amdgpu --target gfx1250 --dump-vcs copy.vcdump
volta compare --from-dump copy.vcdump --check-array out
```

Replace the second source/entry with the kernel under comparison. Repeat
`--check-array` for every output you want compared. Arrays, arguments and globals
are shared; `--block2` and `--grid2` can describe a different second launch.
These overrides must preserve the intended logical computation.

```sh
volta analyze tests/gfx1250/fixtures/native-copy-add.s --config copy.json \
  --backend amdgpu --target gfx1250 --kernel copy \
  --emit-exprs copy.vexpr --dump-graph copy.graph
volta compare --exprs copy.vexpr copy.vexpr --check-array out
```

`.vexpr` files contain symbolic outputs; `.vcdump` files contain the expressions
and output contracts for a comparison. Replay skips kernel execution and reports
equivalence only. A `.graph` dump is diagnostic text, not an executable replay
format. Use a recorded run store with `volta viz STORE -o page.html` to browse
events, target geometry, source links and typed findings.

## Check a gfx1250 B0 program

`check-program` evaluates one gfx1250 m1 integer kernel (an
fuzzer candidate, twin or calibration control) under profile
`gfx1250-b0-m1-int/1` and writes a JSON report:

```sh
volta --backend amdgpu --target gfx1250 --no-log-file check-program \
  --request cand.request.json --report cand.report.json
```

The request (`ptxv-m1-request/1`) and report (`ptxv-m1-report/1`) schemas are
frozen in [tests/gfx1250/m1/schema](../tests/gfx1250/m1/schema). The request
names the source and its sha256, the echoed object hash, the role (`cand`,
`twin` or `control`), enabled pending operations, the one-wave launch, the
`out` object, the backing allocation, a `pre` and a `post` guard and the
`raw-u32` output contract. Relative paths are resolved against the request's
directory. Unknown fields or values outside the schema are `ERROR`; a source
hash mismatch is `ERROR` with reason `SOURCE_HASH_MISMATCH` and no analysis.

The report keeps the parser status and refused operations, execution status,
seven checks (`trace_completeness`, `wait_readiness`, `barrier_generations`,
`memory_conflicts`, `object_bounds`, `prefetch_policy`, `store_completion`,
each with typed witnesses or unresolved reasons), kernarg/store/prefetch memory
events, every output word as an exact u32 or a null with its reason, the
modeling assumptions and the verifier identity (version, build revision, dirty
flag and executable sha256). The report is written atomically and is
authoritative; the exit status equals its class:

| Class | Exit | Meaning |
| --- | --- | --- |
| `PASS` | 0 | Parser and profile supported, complete execution, every check passes, every output word exact |
| `REJECT` | 1 | A check failed with a certified typed witness |
| `UNKNOWN_UNSUPPORTED` | 2 | Refused operation or form, parser/layout refusal, aborted execution, unresolved check or partial outputs |
| `ERROR` | 3 | Invalid request or invocation, unreadable or mismatched source |
| `TIMEOUT` | 4 | The request's instruction budget was exhausted |

Precedence is REJECT > ERROR > TIMEOUT > UNKNOWN_UNSUPPORTED > PASS. A PASS
states that the required checks passed under the recorded model and launch
premises; it is not a claim about other steppings or about device results. See
the [gfx1250 m1 profile](../tests/gfx1250/CONTRACT.md#m1-integer-profile)
and its [fixtures](../tests/gfx1250/m1/README.md).

A `ptxv-m2-request/1` request selects the m2 MODE profile
`gfx1250-b0-m2-mode/1` instead. Its `ptxv-m2-report/1` has the same fields plus
an eighth check, `mode_order`, and a `mode` record (entry FP_ROUND/FP_DENORM,
every MODE access with its value, final fields). The schemas are in
[tests/gfx1250/m2/schema](../tests/gfx1250/m2/schema); see the
[m2 profile](../tests/gfx1250/CONTRACT.md#m2-mode-profile)
and its [fixtures](../tests/gfx1250/m2/README.md).

A `ptxv-litmus-request/1` request selects the litmus profile
`gfx1250-b0-litmus/1` for fuzzer litmus kernels. It names contiguous
objects (`out`, `data`) with their exact initial words and the static LDS size
the descriptor must declare; its `ptxv-litmus-report/1` has the seven m1 checks,
the final object images and a `litmus` record of every buffer range check and
async LDS copy. The schemas are in
[tests/gfx1250/litmus/schema](../tests/gfx1250/litmus/schema); see the
[litmus profile](../tests/gfx1250/CONTRACT.md#litmus-profile) and its
[fixtures](../tests/gfx1250/litmus/README.md). A `ptxv-litmus-request/2`
request selects `gfx1250-b0-litmus/2`, which adds the two-wave named-barrier
JOIN template (`launch_file.block` `[32, 1, 1]` or `[64, 1, 1]`); see
[litmus/2](../tests/gfx1250/CONTRACT.md#litmus2-named-barrier-join-read-back)
and its [fixtures](../tests/gfx1250/litmus2/README.md).

`ptxv-wmma-request/1` and `ptxv-mw-request/1` select `gfx1250-b0-wmma/1` (one
wave with BF16 WMMA) and `gfx1250-b0-mw/1` (2..8 waves: `launch_file.block` is
`[32*W, 1, 1]`; `outputs` declares the `opaque` and `not_written` word ranges as
`[first, count]` pairs). Their reports add the ninth check `wmma_hazards`. The
schemas are in [tests/gfx1250/mw_wmma/schema](../tests/gfx1250/mw_wmma/schema);
see the [WMMA and multiwave profiles](../tests/gfx1250/CONTRACT.md#wmma-and-multiwave-profiles).
`ptxv-wmma-request/2` selects `gfx1250-b0-wmma/2` (F16, IU8 and FP8 WMMA and an
EXEC=0 spacer; schemas in [tests/gfx1250/wmma2/schema](../tests/gfx1250/wmma2/schema));
see the [WMMA types profile](../tests/gfx1250/CONTRACT.md#wmma-types-profile).
`ptxv-wmma-request/3` selects `gfx1250-b0-wmma/3` (adds the BF8 combinations,
F8F6F4, F4 32x16x128 and their block-scaled forms; schemas in
[tests/gfx1250/wmma3/schema](../tests/gfx1250/wmma3/schema)); see the
[WMMA scale profile](../tests/gfx1250/CONTRACT.md#wmma-scale-profile).
`ptxv-wmma-request/4` selects `gfx1250-b0-wmma/4` (adds the 16-bit-output WMMA
forms and SWMMAC; schemas in [tests/gfx1250/wmma4/schema](../tests/gfx1250/wmma4/schema));
see the [WMMA 16-bit output and SWMMAC profile](../tests/gfx1250/CONTRACT.md#wmma-16-bit-output-and-swmmac-profile).
`ptxv-mw-request/2` selects `gfx1250-b0-mw/2` (the multiwave profile with the
`/4` WMMA forms in any wave; schemas in [tests/gfx1250/mw2/schema](../tests/gfx1250/mw2/schema));
see the [multiwave WMMA-types profile](../tests/gfx1250/CONTRACT.md#multiwave-wmma-types-profile).
`ptxv-mwpred-request/1` selects `gfx1250-b0-mwpred/1` (every workgroup of a
non-communicating grid is analysed and composed; schemas in [tests/gfx1250/mwpred/schema](../tests/gfx1250/mwpred/schema));
see the [multi-workgroup predecessor profile](../tests/gfx1250/CONTRACT.md#multi-workgroup-predecessor-profile).
`ptxv-comms-request/1` selects `gfx1250-b0-comms/1` (waves of a workgroup
exchange data through LDS and GLOBAL memory, ordered by completion waits and
barriers; grids of isolated workgroups; schemas in [tests/gfx1250/comms/schema](../tests/gfx1250/comms/schema));
see the [communicating-waves profile](../tests/gfx1250/CONTRACT.md#communicating-waves-profile).

## Read results and logs

| Operation | Success / failure / inconclusive | Exit codes |
| --- | --- | --- |
| Ordinary `analyze` | `VERIFIED` / `REJECTED` / `UNKNOWN` | 0 / 1 / 2 |
| `verify`, source `compare` | `ACCEPT` / `REJECT` / `UNKNOWN` | 0 / 1 / 2 |
| Certified readiness, memory-conflict or tensor-bounds finding, any source command | `VERDICT: REJECT` with incomplete prefix | 1 |
| Expression/VC replay | Equivalence result without a graph verdict | 0 / 1 / 2 |
| `check-program` | Report class `PASS` / `REJECT` / `UNKNOWN_UNSUPPORTED` / `ERROR` / `TIMEOUT` | 0 / 1 / 2 / 3 / 4 |

A completed trace with passing graph checks is distinct from a proof of output
equivalence. Unsupported instructions, uninitialized values, budgets and missing
synchronization evidence must be understood from their diagnostics; they are
not successful results. Ordinary graph reports list the modeling assumptions
used by the result. Early output identifies the witness; its full provenance
and trust boundary are described in the [early-readiness](early-readiness.md),
[memory-conflict](early-memory-conflicts.md) and [tensor-bounds](tensor-bounds.md)
guides.

The CLI writes per-invocation logs to `volta-logs/`. Use `--log-dir DIR` to move
them or `--no-log-file` to disable them. Stage logging is available with:

```sh
cargo build --release -p volta_cli --features logging
target/release/volta --log-level info --help
```

## Python

Install the local bindings into a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install ./crates/volta_py
```

Run from the repository root after activating that environment:

```python
from pathlib import Path
import volta

source = Path("tests/gfx1250/fixtures/native-copy-add.s").read_text()
spec = Path("tests/gfx1250/specs/copy.spec").read_text()
config = volta.Config(block=(65, 1, 1), backend="amdgpu", target="gfx1250")
config.add_array(volta.ArrayDef("x", 0x10000, 4, 65, volta.ArrayKind.Input))
config.add_array(volta.ArrayDef("out", 0x30000, 4, 65, volta.ArrayKind.Output))
config.add_param(volta.Param.array_ptr("x"))
config.add_param(volta.Param.array_ptr("out"))
config.add_param(volta.Param.int(65))

volta.parse(source, backend="amdgpu", target="gfx1250")
result = volta.analyze(source, config, kernel="copy")
assert result.race_verified and not result.unresolved
check = volta.verify(source, spec, config, kernel="copy", dims={"N": 65}, sample=0)
assert check.race_verified and check.equivalent
assert check.elements_checked == check.elements_total == 65
```

Python exposes `parse`, `analyze` and `verify`; kernel comparison and artifact
replay use the CLI. Configuration has typed argument constructors, globals and
native shared-allocation premises. `DataRaceError` and `DeadlockError` derive
from `VoltaError`. Inconclusive graph checks emit a warning and leave
`race_verified` false. Output strings are formatted on access and can be much
larger than their underlying expression DAG. See the
[bindings guide](../crates/volta_py/README.md) for result fields and tests.

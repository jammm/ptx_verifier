# volta_cli

The `volta` binary parses command-line flags into a launch configuration,
drives `volta_pipeline`, and prints results with exit codes. It holds no
verification logic of its own. `src/main.rs` handles startup and dispatch;
`src/cli.rs` defines the command tree, and `src/commands/` contains handlers.
Shared launch, source, specification and output helpers remain separate.

| command | does | exit |
|---------|------|------|
| `volta parse <file>` | syntax check | 0 / 1 |
| `volta analyze <file> -k K ...` | analysis + target-selected graph checks: prints the symbolic outputs, a per-instruction-kind profile (`--no-profile`), each check's verdict, any deadlock and the first failure; `GRAPH VERIFICATION: VERIFIED \| REJECTED \| UNKNOWN` | 0 / 1 / 2 |
| `volta compare <ref> <opt> --kernel1 .. --kernel2 .. --check-array A` | both kernels through both branches, then `decide`: `VERDICT: ACCEPT \| REJECT \| UNKNOWN` | 0 / 1 / 2 |
| `volta verify <file> <spec> -k K --dim N=..` | one kernel against a `.spec` (file or inline string), through both branches and `decide` | 0 / 1 / 2 |
| `volta solve <file> --spec SPEC --dim N=..` | canonical output checking, with graph checks when recorded | 0 / 1 / 2 |
| `volta extract lean <file> --spec SPEC -o challenge.lean` | applicable graph gates, then mathematical obligations for Lean | 0 / 1 |
| `volta viz <store> -o page.html` | the `volta_viz` page over a run store | |
| `volta check-program --request R --report P` | one gfx1250 kernel under the B0 profile its request names (`src/commands/check_program.rs`): validates the frozen request, checks the source hash, applies the profile, executes and writes the matching report atomically | 0 PASS / 1 REJECT / 2 UNKNOWN_UNSUPPORTED / 3 ERROR / 4 TIMEOUT |

## Flow of a command

`solve` and `extract lean` infer PTX, experimental AMD gfx950 or Trainium NKI
from `.ptx`, `.s` and `.py` when `--backend` is omitted. The extension-selected
AMD interpreter checks wave-level LDS accesses, waits and barriers; its
`SharedMemoryOnly` assumption explicitly excludes global-memory certification.
NKI records no race graph, so its mathematical `ACCEPT` supplies no race-freedom
evidence. Explicit `--backend amdgpu
--target gfx942|gfx950|gfx1250` uses the native checked frontend instead,
including its complete outputs and independently validated prefix rejections.
`--backend ptx` selects PTX explicitly; `--target` without AMD selection is
rejected. Existing `parse`/`analyze`/`verify`/`compare` default to PTX.

Lean extraction accepts a spec from `--spec` or launch JSON, and optionally
`--tolerance 1/1000`. `-o`/`--output` selects the module; otherwise it uses the
kernel filename's stem. Rejected or incomplete checked graphs, including
certified native prefixes, prevent successful extraction. Exit code 0 means proof
obligations were written, not that Lean proved them. `solve` and extraction use
`volta_pipeline`'s shared frontend/solver APIs.

```text
source + backend/target + launch flags ──► KernelModule + AnalysisConfig
                                                  │
                                    verify_source_kernel
                                      │           │
                        graph report ─┘           └─ symbolic outputs
                             │                            │
                analyze: graph verdict         compare/verify: equivalence
                                                        + graph evidence
                                                              │
                                                           decide
```

- **Launch flags** (shared by `analyze`/`compare`/`verify` via the flattened
  `LaunchArgs`): `-b/--block`, `-g/--grid`,
  `--array name:base:elem_width:len:in|out|inout|index`,
  `--param int:N|float:X|sym:name|ptr:array` (in declaration order), globals,
  dynamic shared size. `--config FILE` replaces them all with a JSON launch
  file (`volta_ptx_analysis`'s `launch_file` format), whose `"kernel"` is the
  default `--kernel` of `analyze`/`verify`. `compare` adds `--block2/--grid2`
  for the optimized kernel. Native JSON may include `shared_allocations` byte
  ranges such as `[{"name":"tile","offset":0,"size":256}]`, matching
  `--shared-allocation NAME:OFFSET:SIZE` and Python LDS premises. Names/ranges
  are validated strictly; native accesses must fit the supplied objects and
  aggregate LDS capacity, and PTX rejects this native-only option.
- **Equivalence flags**: `--sample N`, `--verify-numeric`,
  `--recycle-terms N`, `--iterations N`.
- **Artifacts**: `analyze --emit-exprs FILE` (`.vexpr`) and `--dump-graph FILE`
  (`volta-graph 2` with native AMD events, otherwise version 1). These graph dumps are
  diagnostic text, not replay inputs. `compare --dump-vcs FILE` executes and
  checks both kernels before saving their expressions. `compare --exprs a.vexpr b.vexpr` and `--from-dump FILE` skip execution and run equivalence
  only; success prints `EQUIVALENT` without a graph-verification verdict.
- **Logs**: every run writes a `volta_common::run_log` file (`--log-dir`,
  `--no-log-file`). Build with `--features logging` and pass
  `--log-level info` to narrate the stages.

```sh
cargo run --release -p volta_cli -- analyze relu.ptx -k relu -b 128 \
  --array x:0x10000:4:128:in --array out:0x20000:4:128:out \
  --param ptr:x --param ptr:out

cargo run --release -p volta_cli -- analyze relu.ptx --config relu.json
```

See [usage](../../docs/usage.md) for launch formats, examples and replay commands.

## Native gfx1250

Use `--backend amdgpu --target gfx1250` with `parse`, `analyze`, `verify`,
`compare`, `solve` and `extract lean`. The same native launch JSON, exact output
footprints and graph rules apply. Numeric confirmation, expression/VC replay,
profiles, logs and the graph viewer remain available through their respective
commands. Source targets
must match explicit selection; omitted AMD targets remain gfx950. Native graph
failures exit 1, inconclusive checks exit 2, and success exits 0. Expression-only
replay reports equivalence without claiming any graph checks. See the
[gfx1250 compiler corpus](../../tests/gfx1250/README.md) for the wave32/ABI subset
and hardware reproduction workflows.

`gfx1250_original_workload_tests` exercises original workload shapes
and storage dtypes: exact output coordinates, complete unsampled mathematical
specs, graph/expression export, pair comparison, VC/expression replay and a
full-size mismatch. `VOLTA_GFX1250_ORIGINAL_FULL=1` requests the separate heavy
full-workload tests and refuses missing registrations. See the
[original-shape workflows](../../tests/gfx1250/full-parity/README.md).
Registering a workload or replaying its expressions does not establish graph
safety; analysis must complete and the applicable graph checks must pass.

Graph-check witnesses include the source instruction when a source span is available.

`check-program` is the machine interface for fuzzer m1 kernels.
Its report, not printed text, is authoritative; the exit status equals the
report class. The build script embeds `git rev-parse HEAD` and a dirty flag
(`unknown` outside a checkout) and the report adds the executable's sha256,
computed by the in-tree `sha256.rs`. See [usage](../../docs/usage.md#check-a-gfx1250-b0-program)
and the [fixtures](../../tests/gfx1250/m1/README.md). A `ptxv-m2-request/1`
selects the m2 MODE profile `gfx1250-b0-m2-mode/1` and gets a
`ptxv-m2-report/1` with the `mode_order` check
([fixtures](../../tests/gfx1250/m2/README.md)). A `ptxv-litmus-request/1`
selects the litmus profile `gfx1250-b0-litmus/1` for fuzzer litmus
kernels and gets a `ptxv-litmus-report/1` with object images and every buffer
range check ([fixtures](../../tests/gfx1250/litmus/README.md));
`ptxv-litmus-request/2` selects `gfx1250-b0-litmus/2`, which adds the two-wave
named-barrier JOIN template ([fixtures](../../tests/gfx1250/litmus2/README.md)). A
`ptxv-wmma-request/1` or `ptxv-mw-request/1` selects `gfx1250-b0-wmma/1` or
`gfx1250-b0-mw/1` (fuzzer WMMA bodies and multiwave experiments) and gets
the m2 report plus the `wmma_hazards` check
([fixtures](../../tests/gfx1250/mw_wmma/README.md)); `ptxv-wmma-request/2`
selects `gfx1250-b0-wmma/2` (F16, IU8 and FP8 WMMA, EXEC=0 spacers;
[fixtures](../../tests/gfx1250/wmma2/README.md)); `ptxv-wmma-request/3` selects
`gfx1250-b0-wmma/3` (adds BF8 combinations, F8F6F4, F4 32x16x128 and the
block-scaled forms; [fixtures](../../tests/gfx1250/wmma3/README.md));
`ptxv-wmma-request/4` selects `gfx1250-b0-wmma/4` (adds the 16-bit-output WMMA
forms and SWMMAC; [fixtures](../../tests/gfx1250/wmma4/README.md));
`ptxv-mw-request/2` selects `gfx1250-b0-mw/2` (the multiwave profile with the
`/4` WMMA forms in any wave; [fixtures](../../tests/gfx1250/mw2/README.md));
`ptxv-mwpred-request/1` selects `gfx1250-b0-mwpred/1` (every workgroup of a
non-communicating grid; [fixtures](../../tests/gfx1250/mwpred/README.md));
`ptxv-comms-request/1` selects `gfx1250-b0-comms/1` (waves of a workgroup
communicating through LDS and GLOBAL memory, and grids of isolated workgroups;
[fixtures](../../tests/gfx1250/comms/README.md)).

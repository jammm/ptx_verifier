# gfx1250 m1 integer profile fixtures

Real fuzzer-generated m1 kernels and controls for `volta check-program`
under profile `gfx1250-b0-m1-int/1` (dependency profile
`gfx1250-b0-sched0-m1/1`). The [model contract](../CONTRACT.md#m1-integer-profile)
describes what the profile checks; [usage](../../../docs/usage.md#check-program)
describes the command and its report.

| Path | Content |
| --- | --- |
| `schema/` | Frozen interface mirroring the fuzzer (schema ids, op lists and allowlist hash identical): request and report schemas and `m1_int_subset.json` |
| `kernels/` | `cand`/`twin` sources as checked (each with `.size kernel, .-kernel` after the first `s_endpgm`) of the seeds, the 41 op probes, two seeded mutants and the fixed-text signed S_MAX SCC calibration controls (`role: control`, `s_max_i32` enabled), plus harmful negatives, refusal, undefined-state and parser controls that are not small edits of `kernels/seed-scc_chain-cand.s` |
| `request-template.json` | The request shared by every case; `source` and `object` are filled in per case |
| `cases.json` | One case per line: program and role (or a derived case), base kernel, line edits of `seed-scc_chain-cand`, a merge patch over the template, source and object hashes |
| `expected-words.json` | Independent reference: every output word of every program (fuzzer reference outputs plus canary tags, computed when the probe corpus was built) |
| `provenance.json` | Toolchain identity and launch premises; per program and role the clang-24 object hash, `llvm-readobj --file-headers` and `llvm-objdump -d` (repeated `s_code_end` padding as one `s_code_end xN` line); per derived case its edit and assembler acceptance |

Programs are the two integer seeds, all 41 calibration op probes and two
seeded mutants of a fuzzer probe corpus (AMD clang 24.0.0git
`9fa7f0cb`). The checked source applies one transformation, the one the m1
renderer applies: `.size kernel, .-kernel` on the line after the first
`  s_endpgm`. Both source hashes are recorded. Requests use the launch of the
m1 contract: `out` at `0x10100` with `W` words, backing allocation at
`0x10000`, a 256-byte pre guard, a 1024-byte post guard and a budget of
100000 instructions.

## Run

```sh
cargo test --release -p volta_cli --test gfx1250_m1_check_program
cargo test --release -p volta_amdgpu_analysis --test gfx1250_m1_forms
```

Tests need no ROCm. Each report is validated against the frozen schema and its
exit status against its class. Positives must PASS with every word equal to the
reference, and the verifier's encoded size, padding and `s_wait_alu` encodings
must equal the clang-24 object disassembly.

## Provenance

The fixtures are committed data, produced by a generator that lives with the fuzzer (it imports the fuzzer to render programs and reference words); expected classes and check results are written by that generator, never taken from verifier output.
Toolchain paths in the recorded provenance are written relative to
`$ROCM_PATH`.

clang 24 has no `SCOPE_WGP` spelling; the
non-speculative prefetch control uses `scope:SCOPE_CU`.

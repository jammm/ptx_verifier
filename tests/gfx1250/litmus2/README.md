# gfx1250 B0 litmus/2 profile fixtures

Real fuzzer-generated litmus kernels for `volta check-program` under profile
`gfx1250-b0-litmus/2` (dependency profile `gfx1250-b0-sched0-litmus/2`). The
profile is `gfx1250-b0-litmus/1` (raw buffers, async loads to LDS; see
`../litmus`) plus the two-wave named-barrier JOIN template `barjoin`: one
workgroup of two waves (the second exits through a forward branch), named
barrier INIT/JOIN/SIGNAL/WAIT, `s_wait_idle` and the read of the wave's
STATE_PRIV register. A STATE_PRIV read defines only the joined named barrier
id (bits 8:4) and a clear completion bit (bit 3). The profile analyses a JOIN
or a STATE_PRIV read only after the wave has waited on every completion of its
joined barrier, because the order of a barrier's completion notification
against a later JOIN is not settled; earlier forms are refused.

| Path | Content |
| --- | --- |
| `schema/` | Frozen interface mirroring the fuzzer (schema ids, op lists and allowlist hash identical): `ptxv-litmus-request/2`, `ptxv-litmus-report/2` and `litmus_b0_t2_subset.json` |
| `kernels/` | Genuine fuzzer-generated kernels: the seeds and controls, a subset of the generated positives that still uses every instruction form (mnemonic and matrix format) of the full generated set, every program a test names, and each kernel that no edit of at most six lines reaches |
| `request-template.json` | The request shared by the profile's cases; `source` and `object` are filled in per case |
| `cases.json` | One case per line: base kernel, line edits for the negative and refusal variants, a merge patch over the template, source and object hashes, profile, role and the expected result written by the generator |
| `expected-words.json` | Every object word of each positive case: the template's documented-contract image, computed by the fuzzer |
| `provenance.json` | Toolchain identity, and per case the clang-24 assembly result, object hash and `llvm-objdump -d` (repeated `s_code_end` padding as one `s_code_end xN` line) |

Cases:

* Positives (PASS, exact object images), candidate and twin: the waited
  `barjoin` seed, both barjoin contract controls, a barjoin parameter grid
  (barrier pairs, idle gaps 0-16, with and without the NULL join), the
  litmus/1 seeds and all seven litmus/1 contract controls under /2, and the
  barjoin kernel without `s_wait_kmcnt` (the `s_wait_idle` before the only
  kernarg use drains KMcnt).
* Harmful negatives (REJECT with a certified witness): a kernarg SGPR read
  before `s_wait_kmcnt` (`READ_BEFORE_READY`) and the result store moved into
  the post guard (`GUARD_ACCESS`).
* Refusals (UNKNOWN_UNSUPPORTED, never PASS): the unwaited JOIN forms (direct
  form, and with the NULL join and 12 idle instructions), a STATE_PRIV read
  without a directly preceding `s_wait_idle`, a read of another hardware
  register, a barrier id beyond the descriptor allocation, a read while the
  joined barrier has an unconsumed completion, a wait that never completes,
  and the unqualified `asyncstore` and `bufswz` templates.

`crates/volta_cli/tests/gfx1250_litmus2_check_program.rs` runs every case,
validates each report against the frozen schema, checks that litmus/1 still
refuses the barjoin kernel and that malformed requests are rejected.
The fixtures are committed data, produced by a generator that lives with the fuzzer (it imports the fuzzer to render programs and reference words); expected classes and check results are written by that generator, never taken from verifier output.

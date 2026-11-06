# gfx1250 B0 communicating-waves profile fixtures

Real fuzzer-generated communicating-waves experiments for `volta check-program`
under profile `gfx1250-b0-comms/1`: one workgroup of 2..8 waves whose waves
exchange exact data through LDS or GLOBAL memory. A producer stores its data,
waits for the store to complete and signals a barrier (the workgroup barrier,
or a named barrier with two members); the consumer waits on the barrier, loads
the data and stores what it received. Grids run the same experiment in every
workgroup on its own slice of the output (workgroup id from TTMP9); the
verifier analyses every workgroup and composes the results only when the
workgroups are isolated (check `workgroup_isolation`). Expected output words
are the fuzzer's own channel table; the placement words are declared opaque.

| Path | Content |
| --- | --- |
| `schema/` | Interface shared with the fuzzer: `ptxv-comms-request/1`, `ptxv-comms-report/1`, `comms_b0.json` |
| `kernels/` | Genuine fuzzer-generated kernels: the seeds and controls, a subset of the generated positives that still uses every instruction form (mnemonic and matrix format) of the full generated set, every program a test names, and each kernel that no edit of at most six lines reaches |
| `request-template.json` | The request shared by the profile's cases; `source` and `object` are filled in per case |
| `cases.json` | One case per line: base kernel, line edits for the negative and refusal variants, a merge patch over the template, source and object hashes, profile, role and the expected result written by the generator |
| `expected-words.json` | Every output word of each positive case (null where the request declares the word opaque or unwritten) |
| `provenance.json` | Toolchain identity, and per case the assembly result, object hash and `llvm-objdump -d` (repeated `s_code_end` padding as one `s_code_end xN` line) |

Cases: positives (the five fuzzer seeds, two twin forms with drained fillers
and longer gaps, a grid of 4 and a grid of 2 workgroups); harmful negatives
(an LDS or a GLOBAL producer that signals before its store completed:
`MEMORY_CONFLICT`; every workgroup of a grid writing the same slice:
`CROSS_WORKGROUP_WRITE_OVERLAP`; evidence stored into the post guard:
`GUARD_ACCESS`); unknowns (the round barrier removed, so the consumer reads LDS
no wave wrote yet; a workgroup reading another workgroup's slice, whose bytes
its own analysis never saw written); refusals (named barrier 3, a scalar load
of communicated data, the `gfx1250-b0-mw/1` profile on a communicating
kernel); malformed requests (no `lds_bytes`, a grid above 64, `lds_bytes` on a
`gfx1250-b0-mw/1` request).

The fixtures are committed data, produced by a generator that lives with the fuzzer (it imports the fuzzer to render programs and reference words); expected classes and check results are written by that generator, never taken from verifier output.

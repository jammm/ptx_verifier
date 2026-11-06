# gfx1250 B0 multi-workgroup predecessor profile fixtures

Real fuzzer-generated predecessor grids for `volta check-program` under profile
`gfx1250-b0-mwpred/1`: a fixed kernel launched as a grid of workgroups whose
every wave runs the same integer or BF16 WMMA body and stores one 16-byte
record (HW_ID1, the SE/XCC id message return, the lane-0 result, a tag). The
verifier analyses every workgroup of the grid with its id in TTMP9 and composes
the results only when the workgroups are isolated (pairwise disjoint written
bytes, no workgroup reading bytes another workgroup writes; check
`workgroup_isolation`). Expected
output words are the fuzzer's own CPU reference; the placement words are
declared opaque.

| Path | Content |
| --- | --- |
| `schema/` | Interface shared with the fuzzer: `ptxv-mwpred-request/1`, `ptxv-mwpred-report/1`, `mw_pred_b0.json` |
| `kernels/` | Genuine fuzzer-generated kernels: the seeds and controls, a subset of the generated positives that still uses every instruction form (mnemonic and matrix format) of the full generated set, every program a test names, and each kernel that no edit of at most six lines reaches |
| `request-template.json` | The request shared by the profile's cases; `source` and `object` are filled in per case |
| `cases.json` | One case per line: base kernel, line edits for the negative and refusal variants, a merge patch over the template, source and object hashes, profile, role and the expected result written by the generator |
| `expected-words.json` | Every output word of each positive case (null where the request declares the word opaque) |
| `provenance.json` | Toolchain identity, and per case the clang-24 assembly result, object hash and `llvm-objdump -d` (repeated `s_code_end` padding as one `s_code_end xN` line) |

Cases: positives (the integer and the WMMA predecessor at grids 1, 2 and 4 with
32 waves, the WMMA predecessor at 3 workgroups of 8 waves); harmful negatives
(every workgroup writing the same records: `CROSS_WORKGROUP_WRITE_OVERLAP`;
every wave of a workgroup writing the same record: `MEMORY_CONFLICT`; the WMMA
result read without its spacing: `WMMA_VALU_READ_BEFORE_READY`; records stored
into the post guard: `GUARD_ACCESS`); refusals (the message return not followed
by `s_wait_kmcnt 0`, an opaque value used as an address, the `gfx1250-b0-mw/1`
profile on a predecessor kernel); malformed requests (a grid above 1024, a
`workgroup_id`, no `workgroups`).

The fixtures are committed data, produced by a generator that lives with the fuzzer (it imports the fuzzer to render programs and reference words); expected classes and check results are written by that generator, never taken from verifier output.

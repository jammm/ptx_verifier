# gfx1250 B0 litmus profile fixtures

Real fuzzer-generated litmus kernels for `volta check-program` under profile
`gfx1250-b0-litmus/1` (dependency profile `gfx1250-b0-sched0-litmus/1`). The
profile checks the qualified litmus templates: untyped raw-buffer loads and
stores across `num_records` and the OOB_SELECT stride bound (`bufload`,
`bufstore`) and scalar-base async loads to LDS below a 64 KiB displacement
(`asynclds`). It models the documented range-check and async address rules;
forms they do not settle are refused, and the premises are
listed as report assumptions.

| Path | Content |
| --- | --- |
| `schema/` | Frozen interface mirroring the fuzzer (schema ids, op lists and allowlist hash identical): `ptxv-litmus-request/1`, `ptxv-litmus-report/1` and `litmus_b0_subset.json` |
| `kernels/` | Genuine fuzzer-generated kernels: the seeds and controls, a subset of the generated positives that still uses every instruction form (mnemonic and matrix format) of the full generated set, every program a test names, and each kernel that no edit of at most six lines reaches |
| `request-template.json` | The request shared by the profile's cases; `source` and `object` are filled in per case |
| `cases.json` | One case per line: base kernel, line edits for the negative and refusal variants, a merge patch over the template, source and object hashes, profile, role and the expected result written by the generator |
| `expected-words.json` | Every object word of each positive case: the template's documented-contract image, computed by the fuzzer |
| `provenance.json` | Toolchain identity, and per case the clang-24 assembly result, object hash and `llvm-objdump -d` (repeated `s_code_end` padding as one `s_code_end xN` line) |

Cases:

* Positives (PASS, exact object images), candidate and twin: the three seeds,
  the litmus contract controls and boundary parameter points (DWORDs partly
  and fully out of range, OOB_SELECT 1 with the stride below and above
  `num_records`, async displacements up to 0xfffc), plus the twin of a 64 KiB
  async displacement, which carries the offset in its address registers.
* Harmful negatives (REJECT with a certified witness): a missing
  `s_wait_loadcnt` or `s_wait_kmcnt` (`READ_BEFORE_READY`), a descriptor
  base in the post guard and a store into it (`GUARD_ACCESS`), the envelope
  case whose only in-range DWORD lies in its object while the unclamped
  footprint crosses into the guard (`GUARD_ACCESS`), and an async copy whose
  LDS writes race a DS load or a DS store (`MEMORY_CONFLICT`).
* Refusals (UNKNOWN_UNSUPPORTED, never PASS): unbound (`num_records` 0),
  swizzled, stride-scaled, typed and bounds-disabled resources, an unaligned
  buffer offset, indexed addressing, an async displacement of 64 KiB, an
  unaligned async LDS footprint, LDS beyond the descriptor allocation, a
  request whose LDS size disagrees with the descriptor, and the unqualified
  `asyncstore` and `bufswz` templates.

`crates/volta_cli/tests/gfx1250_litmus_check_program.rs` runs every case,
validates each report against the frozen schema, checks the m1
profile still refuses litmus operations and rejects malformed requests.
The fixtures are committed data, produced by a generator that lives with the fuzzer (it imports the fuzzer to render programs and reference words); expected classes and check results are written by that generator, never taken from verifier output.

# gfx1250 B0 WMMA and multiwave profile fixtures

Real fuzzer-generated kernels for `volta check-program` under profiles
`gfx1250-b0-wmma/1` (one wave: the m2 operations plus
`v_wmma_f32_16x16x32_bf16`) and `gfx1250-b0-mw/1` (the fuzzer's multiwave
aggressor/victim wrapper for 2..8 waves of one workgroup). Expected output
words are the fuzzer's own CPU reference, computed per wave with its lane base;
the multiwave hardware-evidence words (HW_ID1, shader cycles) are declared
opaque by the request and never compared.

| Path | Content |
| --- | --- |
| `schema/` | Frozen interface mirroring the fuzzer (schema ids, op lists and allowlist hash identical): `ptxv-{wmma,mw}-request/1`, `ptxv-{wmma,mw}-report/1`, `wmma_b0.json` and `mw_b0.json` |
| `kernels/` | Genuine fuzzer-generated kernels: the seeds and controls, a subset of the generated positives that still uses every instruction form (mnemonic and matrix format) of the full generated set, every program a test names, and each kernel that no edit of at most six lines reaches |
| `request-template.json` | The request shared by the profile's cases; `source` and `object` are filled in per case |
| `cases.json` | One case per line: base kernel, line edits for the negative and refusal variants, a merge patch over the template, source and object hashes, profile, role and the expected result written by the generator |
| `expected-words.json` | Every output word of each positive case (null where the request declares the word opaque or unwritten) |
| `provenance.json` | Toolchain identity, and per case the clang-24 assembly result, object hash and `llvm-objdump -d` (repeated `s_code_end` padding as one `s_code_end xN` line) |

Cases:

* Positives (PASS, exact words), candidate and twin: an integer/WMMA chain at
  the documented spacing, integer seeds, six fuzzer mutants that keep a WMMA,
  the twin form of the calibration WMMA probe, a MODE body with its pending
  operations enabled; the multiwave seeds (integer victims, WMMA aggressors)
  and a geometry sweep of 2..8 waves, aggressors first or alternating; a MODE
  victim with MODE enabled.
* Harmful negatives (REJECT with a certified witness): a WMMA reading the
  previous D as A after 4 slots, a VALU reading D or overwriting A within the
  4-slot window, a same-argument back-to-back WMMA (single wave and inside a
  multiwave aggressor), a missing `s_wait_kmcnt` (`READ_BEFORE_READY`), the
  evidence store moved into the post guard (`GUARD_ACCESS`) and every wave
  storing its evidence at the same address (`MEMORY_CONFLICT`).
* Refusals (UNKNOWN_UNSUPPORTED, never PASS): a D-to-C chain or the
  calibration probe's candidate form below the 5-slot policy
  (`WMMA_UNLISTED_CHAIN_BELOW_POLICY`, witness on an UNKNOWN check); WMMA data
  outside the exact-representable regime (a magnitude sum reaching 2^24 times
  the common quantum, a denormal BF16 input, D read as BF16 A); the F16 WMMA
  form; a floating-point victim; an opaque evidence value used as an address;
  a field-selected HW_ID1 read; pending MODE operations; undeclared evidence
  words.

`crates/volta_cli/tests/gfx1250_mw_wmma_check_program.rs` runs every case,
validates each report against the frozen schema, checks the m2
profile still refuses these kernels and rejects malformed requests.
The fixtures are committed data, produced by a generator that lives with the fuzzer (it imports the fuzzer to render programs and reference words); expected classes and check results are written by that generator, never taken from verifier output.

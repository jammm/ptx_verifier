# gfx1250 B0 WMMA scale profile fixtures

Real fuzzer-generated kernels for `volta check-program` under profile
`gfx1250-b0-wmma/3`: the `gfx1250-b0-wmma/2` operations plus the BF8
combinations (`v_wmma_f32_16x16x64_*`, `v_wmma_f32_16x16x128_*` with a BF8
operand), `v_wmma_f32_16x16x128_f8f6f4`, `v_wmma_f32_32x16x128_f4` and the
16-byte block-scaled forms `v_wmma_scale[16]_f32_16x16x128_f8f6f4` and
`v_wmma_scale[16]_f32_32x16x128_f4`. Expected output words are the fuzzer's own
CPU reference.

| Path | Content |
| --- | --- |
| `schema/` | Frozen interface mirroring the fuzzer (schema ids, op lists and allowlist hash identical): `ptxv-wmma-request/3`, `ptxv-wmma-report/3`, `wmma_scale_b0.json` |
| `kernels/` | Genuine fuzzer-generated kernels: the seeds and controls, a subset of the generated positives that still uses every instruction form (mnemonic and matrix format) of the full generated set, every program a test names, and each kernel that no edit of at most six lines reaches |
| `request-template.json` | The request shared by the profile's cases; `source` and `object` are filled in per case |
| `cases.json` | One case per line: base kernel, line edits for the negative and refusal variants, a merge patch over the template, source and object hashes, profile, role and the expected result written by the generator |
| `expected-words.json` | Every output word of each positive case |
| `provenance.json` | Toolchain identity, and per case the clang-24 assembly result, object hash and `llvm-objdump -d` (repeated `s_code_end` padding as one `s_code_end xN` line) |

Profile rules (beyond `gfx1250-b0-wmma/2`):

* Spacing, WMMA to dependent WMMA A/B and WMMA to dependent VALU: BF8 16x16x64
  2 and 1, BF8 16x16x128 4 and 3, F8F6F4 with both formats FP4 2 and 1, other
  F8F6F4 4 and 3, F4 32x16x128 4 and 3 slots, scaled or not; below them a
  certified failure. Unlisted WMMA-to-WMMA dependencies below the WMMA count
  are unresolved.
* A block-scaled WMMA is one 16-byte instruction; its load-scale half is never
  a slot. Scale operands are named by no documented row: a scale operand that
  reads an earlier WMMA's D, or a VALU that writes an earlier WMMA's scale
  registers, below the count is unresolved.
* Results are evaluated only for uniform A/B tiles and uniform scale tuples
  (layout-independent); exact bits for normal-or-zero elements, normal
  power-of-two scales, the exact-representable regime and concrete C bits.

Cases: positives (seeds, fuzzer mutants that keep a new form, the twin form of
each calibration probe, each spacing class's D-to-C chain at its count, scale
lanes 16..31, a VALU rewriting a scale register at the count); harmful
negatives (each spacing class one slot short for a D-as-A read and for a VALU
read, a scaled consumer four slots after a BF16 WMMA, same-argument
back-to-back); refusals (D-as-C one slot below the policy, a VALU rewriting a
scale register in flight, non-uniform BF8, FP6 and scale tiles, an E8M0 NaN
scale, E4M3 scale bytes with bit 7 set, a non-power-of-two scale, FP4
denormals, reuse hints, inline and SGPR scale operands, partially overlapping
32x16 C/D, a standalone load-scale instruction, and the `gfx1250-b0-wmma/2`
profile on the new kernels).

The fixtures are committed data, produced by a generator that lives with the fuzzer (it imports the fuzzer to render programs and reference words); expected classes and check results are written by that generator, never taken from verifier output.

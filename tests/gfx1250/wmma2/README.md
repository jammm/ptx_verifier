# gfx1250 B0 WMMA types profile fixtures

Real fuzzer-generated kernels for `volta check-program` under profile
`gfx1250-b0-wmma/2`: the `gfx1250-b0-wmma/1` operations plus
`v_wmma_f32_16x16x32_f16`, `v_wmma_i32_16x16x64_iu8`,
`v_wmma_f32_16x16x64_fp8_fp8` and `v_wmma_f32_16x16x128_fp8_fp8`, and an EXEC=0
spacer (`s_mov_b32 exec_lo, 0`; `v_nop` or `v_mov_b32 v81, 0`;
`s_mov_b32 exec_lo, -1`). Expected output words are the fuzzer's own CPU
reference.

| Path | Content |
| --- | --- |
| `schema/` | Frozen interface mirroring the fuzzer (schema ids, op lists and allowlist hash identical): `ptxv-wmma-request/2`, `ptxv-wmma-report/2`, `wmma_types_b0.json` |
| `kernels/` | Genuine fuzzer-generated kernels: the seeds and controls, a subset of the generated positives that still uses every instruction form (mnemonic and matrix format) of the full generated set, every program a test names, and each kernel that no edit of at most six lines reaches |
| `request-template.json` | The request shared by the profile's cases; `source` and `object` are filled in per case |
| `cases.json` | One case per line: base kernel, line edits for the negative and refusal variants, a merge patch over the template, source and object hashes, profile, role and the expected result written by the generator |
| `expected-words.json` | Every output word of each positive case |
| `provenance.json` | Toolchain identity, and per case the clang-24 assembly result, object hash and `llvm-objdump -d` (repeated `s_code_end` padding as one `s_code_end xN` line) |

Profile rules (beyond `gfx1250-b0-wmma/1`):

* Spacing per WMMA type, WMMA to dependent WMMA A/B and WMMA to dependent VALU:
  F16/BF16 5 and 4, FP8 16x16x64 2 and 1, FP8 16x16x128 4 and 3, IU8 9 and 8
  slots; below them a certified failure. Unlisted WMMA to WMMA dependencies
  (D as C, WAW, WAR) below the type's WMMA count are unresolved (conservative
  policy).
* A slot is a V_NOP or an independent vector-ALU instruction issued with a
  nonzero EXEC. Whether an instruction issued with EXEC = 0 provides a slot is
  not settled, so it is never counted: a dependency that meets its count only by
  counting such instructions is unresolved (`WMMA_EXEC0_SLOTS_UNRESOLVED`),
  never a failure and never accepted.
* FP8 and IU8 are evaluated only for uniform A and B tiles (no 8-bit layout is
  assumed); IU8 without NEG modifiers is unsigned and is evaluated for bytes
  below 0x80 and results inside int32. F32 results carry exact bits only in
  the exact-representable regime with inputs normal in their own format.
* EXEC writes: only the two spacer forms.

Cases: positives (seeds, fuzzer mutants that keep a new form, the twin form of
each calibration probe, each type's D-to-C chain at its count, with and without
EXEC=0 instructions on top); harmful negatives (each type one slot short for a
D-as-A read and for a VALU read, same-argument back-to-back); refusals
(D-as-C one slot below the policy, spacing short by one and covered only by
EXEC=0 instructions in both spacer forms, a D-as-A read covered only by them,
same arguments separated only by them, a WMMA under EXEC = 0, other EXEC writes,
non-uniform or 0x80 IU8 tiles, a non-uniform FP8 tile, FP8 and F16 denormal
inputs, a float D read as IU8 C, IU8 overflow and NEG modifiers, and the
`gfx1250-b0-wmma/1` profile on the new kernels).

The fixtures are committed data, produced by a generator that lives with the fuzzer (it imports the fuzzer to render programs and reference words); expected classes and check results are written by that generator, never taken from verifier output.

# gfx1250 B0 multiwave WMMA-types profile fixtures

Real fuzzer-generated multiwave experiments for `volta check-program` under profile
`gfx1250-b0-mw/2`: the multiwave wrapper of `gfx1250-b0-mw/1` (one workgroup of
2..8 wave32 waves, opaque hardware-evidence reads, per-wave output slices) with
the WMMA forms of `gfx1250-b0-wmma/4` in any wave, every WMMA rule applied per
wave. Expected output words are the fuzzer's own CPU reference, computed per
wave with its lane base; the hardware-evidence words are declared opaque and
never compared.

| Path | Content |
| --- | --- |
| `schema/` | Frozen interface mirroring the fuzzer (schema ids, op lists and allowlist hash identical): `ptxv-mw-request/2`, `ptxv-mw-report/2`, `mw_types_b0.json` |
| `kernels/` | Genuine fuzzer-generated kernels: the seeds and controls, a subset of the generated positives that still uses every instruction form (mnemonic and matrix format) of the full generated set, every program a test names, and each kernel that no edit of at most six lines reaches |
| `request-template.json` | The request shared by the profile's cases; `source` and `object` are filled in per case |
| `cases.json` | One case per line: base kernel, line edits for the negative and refusal variants, a merge patch over the template, source and object hashes, profile, role and the expected result written by the generator |
| `expected-words.json` | Every output word of each positive case (null where the request declares the word opaque or unwritten) |
| `provenance.json` | Toolchain identity, and per case the clang-24 assembly result, object hash and `llvm-objdump -d` (repeated `s_code_end` padding as one `s_code_end xN` line) |

The 16-bit C/D tiles of the fuzzer's multiwave prologue are seeded from the
lane (`v0 & 31`), so F16 results stay in the 11-bit integer regime in every
wave; the F32 C/D tiles are seeded from the work-item id.

Cases: positives (the multiwave WMMA-types seeds in candidate and twin form, a
geometry sweep of 2..7 waves, fuzzer mutants that keep a WMMA-types form, the
fully drained multiwave probe of every certified form, each spacing class's
chain at its count inside aggressor waves 1 and 3); harmful negatives (each
spacing class one slot short for a VALU read and, where the count exceeds one,
for a D-as-A read; a same-argument back-to-back WMMA; a cross-wave memory
conflict; evidence stored into the post guard; a missing kernarg wait);
refusals (D-as-C one slot below the policy, the same chain met only by EXEC = 0
instructions, BF16 and BF16F32 results including their probes, 16-bit C/D
tiles seeded from the work-item id, and the `gfx1250-b0-mw/1` profile on these
kernels).

The fixtures are committed data, produced by a generator that lives with the fuzzer (it imports the fuzzer to render programs and reference words); expected classes and check results are written by that generator, never taken from verifier output.

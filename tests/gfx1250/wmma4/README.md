# gfx1250 B0 WMMA 16-bit output and SWMMAC profile fixtures

Real fuzzer-generated kernels for `volta check-program` under profile
`gfx1250-b0-wmma/4`: the `gfx1250-b0-wmma/3` operations plus the 16-bit-output
WMMA forms (F16, BF16, BF16F32) and every SWMMAC form. Expected output words are
the fuzzer's own CPU reference.

| Path | Content |
| --- | --- |
| `schema/` | Frozen interface mirroring the fuzzer (schema ids, op lists and allowlist hash identical): `ptxv-wmma-request/4`, `ptxv-wmma-report/4`, `wmma_sparse16_b0.json` |
| `kernels/` | Genuine fuzzer-generated kernels: the seeds and controls, a subset of the generated positives that still uses every instruction form (mnemonic and matrix format) of the full generated set, every program a test names, and each kernel that no edit of at most six lines reaches |
| `request-template.json` | The request shared by the profile's cases; `source` and `object` are filled in per case |
| `cases.json` | One case per line: base kernel, line edits for the negative and refusal variants, a merge patch over the template, source and object hashes, profile, role and the expected result written by the generator |
| `expected-words.json` | Every output word of each positive case |
| `provenance.json` | Toolchain identity, and per case the clang-24 assembly result, object hash and `llvm-objdump -d` (repeated `s_code_end` padding as one `s_code_end xN` line) |

Profile rules (beyond `gfx1250-b0-wmma/3`): spacing per form (16-bit outputs
5/4, 2/1, 4/3; SWMMAC 3/2, IU8 5/4), the SWMMAC sparsity index as a documented
source, F16 results certified in an 11-bit integer regime, BF16 and BF16F32
results never certified, SWMMAC evaluated for uniform packed A, a B uniform in
each K half and uniform valid index words.

Cases: positives (seeds, fuzzer mutants that keep a new form, the twin form of
each certified calibration probe, each spacing class's chain at its count, the
tier-2 chain seed); harmful negatives (each spacing class one slot short for a
D-as-A read and for a VALU read, a D read as a SWMMAC index); refusals (D-as-C
one slot below the policy, BF16 and BF16F32 results including their probes,
F16 results beyond 2^11, invalid and non-uniform index words, a B that is not
half-uniform, an `index_key` modifier, and the `gfx1250-b0-wmma/3` profile on
the new kernels).

The fixtures are committed data, produced by a generator that lives with the fuzzer (it imports the fuzzer to render programs and reference words); expected classes and check results are written by that generator, never taken from verifier output.

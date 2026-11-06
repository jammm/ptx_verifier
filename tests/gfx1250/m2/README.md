# gfx1250 m2 MODE profile fixtures

Real fuzzer-generated kernels and controls for `volta check-program` under profile
`gfx1250-b0-m2-mode/1` (dependency profile `gfx1250-b0-sched0-m2/1`), the
m1 integer profile plus runtime MODE state for FP_ROUND (MODE[3:0])
and FP_DENORM (MODE[7:4]). The profile checks MODE state and the ordering of
MODE reads and writes; it does not model floating-point rounding or denormal
behaviour, and every floating-point operation stays outside it.

| Path | Content |
| --- | --- |
| `schema/` | Frozen interface mirroring the fuzzer (schema ids, op lists and allowlist hash identical): `ptxv-m2-request/1`, `ptxv-m2-report/1` and `m2_mode_subset.json` |
| `kernels/` | Genuine fuzzer-generated kernels: the seeds and controls, a subset of the generated positives that still uses every instruction form (mnemonic and matrix format) of the full generated set, every program a test names, and each kernel that no edit of at most six lines reaches |
| `request-template.json` | The request shared by the profile's cases; `source` and `object` are filled in per case |
| `cases.json` | One case per line: base kernel, line edits for the negative and refusal variants, a merge patch over the template, source and object hashes, profile, role and the expected result written by the generator |
| `expected-words.json` | Every output word of each positive case: the fuzzer's reference outputs plus canary tags, or a control's written values |
| `provenance.json` | Toolchain identity, and per case the clang-24 assembly result, object hash and `llvm-objdump -d` (repeated `s_code_end` padding as one `s_code_end xN` line) |

Cases:

* Positives (PASS, exact words): the MODE seeds `mode_mix` and
  `mode_getreg_wait` (the fuzzer seed that reads MODE, waits on SA_SDST and
  then writes ROUND_MODE, found by shape), the integer seed `scc_chain`, the
  calibration probes of the five MODE operations, candidate and twin, and the
  nine MODE calibration controls in candidate (unfenced) and twin (fenced)
  form.
* A MODE read followed by a MODE write (`s_round_mode`, `s_denorm_mode` or a
  SETREG to MODE) without an intervening `s_wait_alu` whose SA_SDST field is
  0: UNKNOWN (`mode_order` unresolved), never PASS or REJECT. The `unk-mode-*`
  cases cover the candidate, the twin, a distance, other fields, another wait
  field, a NOP and an intervening SETREG. The same programs with
  `s_wait_alu 0x0` or `0xfffe` pass.
* Refusals (UNKNOWN, not executed): MODE fields and forms outside the profile
  (`MODE_FORM_NOT_IN_PROFILE`), `s_set_vgpr_msb` (`EXCLUDED_OP`), FP VALU
  (`FLOATING_POINT`), MODE operations not enabled by the request
  (`PENDING_HARDWARE_CONTROL`), and MODE under the m1 profile
  (`MODE_ACCESS`), also inside an integer body.
* Undefined MODE state (missing descriptor control, SETREG from an undefined
  SGPR): UNKNOWN, execution aborted.

## Run

```sh
cargo test --release -p volta_cli --test gfx1250_m2_check_program
cargo test --release -p volta_amdgpu_analysis --test gfx1250_m1_forms
```

## Provenance

The fixtures are committed data, produced by a generator that lives with the fuzzer (it imports the fuzzer to render programs and reference words); expected classes and check results are written by that generator, never taken from verifier output.

# Implemented gfx1250 contract

This document defines the selected native execution and checking model.
[Full-workload mappings](full-parity/README.md) and smaller mechanism fixtures
use the same target model with different input/output contracts. CPU analysis
is usable without ROCm once built. [Device runners](hardware/README.md) provide
separate numerical checks.

| Boundary | Implemented envelope |
| --- | --- |
| Target | Bare `gfx1250` as `amdgcn-amd-amdhsa--gfx1250` or the assembler's `amdgcn-amd-amdhsa-unknown-gfx1250`; no strict/neighbor alias or feature-suffix fallback |
| Compiler | ROCm 7.2 AMD LLVM 22, revision `7b800a19466229b8479a78de19143dc33c3ab9b5` |
| Object/metadata | Tested AMDHSA code-object 6, metadata 1.2 |
| Launch | One selected unclustered workgroup, default `(0,0,0)`, configured nonzero grid retained as ABI input, 1..1024 workitems |
| Geometry | Wave32; only low 32 bits of stored EXEC/VCC masks are physical lanes |
| Registers | s0..s105, logical VGPR allocation 1024, encoded operands v0..v255, TTMP0..15; no AGPR namespace |
| LDS | Aggregate static plus dynamic capacity up to 327,680 bytes; explicit subobjects optional |
| Inputs | Concrete addresses/control, symbolic mathematical data, exact by-value metadata bytes |
| Replay | Hardwired XNACK; independent X-count address-source leases and data completion |
| Global visibility | Ordered cross-wave write dependencies require explicit `SCOPE_SYS` at both endpoints, exposed by the source profile for resolved-global FLAT; ordinary GLOBAL scope modifiers are unsupported except on `global_prefetch_b8`. Completed read-before-write needs completion and publication; no cross-workgroup claim |
| Completion | Load/store/DS/KM/X/async/tensor classes; tensor threshold 0 only |
| Barriers | Unnamed shader split signal/wait/release; selected allocated named shader INIT/JOIN/SIGNAL/WAIT/LEAVE with explicit M0 7-bit signal-time count updates, object/phase/incarnation and completed-effect publication |
| Matrices | WMMA F32 16x16x4, F16/BF16 16x16x32; F32 accumulation, complete wave32 |
| Tensor | Plain bounded 2D global/LDS transfers in both directions; staged buffer reuse |
| Arithmetic | Existing mathematical reals; exact concrete signed/unsigned 64-bit products and mad |

Register bounds are assembler/ABI probes rather than blanket access guarantees.
Higher banked VGPR encodings are explicitly unsupported. The logical allocation
limit is separate from the supported operand spelling.
Initial ABI state includes only defined architected TTMP fields, descriptor
user-SGPR inputs, packed local IDs and recognized hidden launch values. Runtime
facilities and padding remain undefined unless metadata/configuration defines
them. Argument count/layout, required dimensions, descriptor resources and
static/dynamic LDS bounds are checked before execution.

Unsupported or inconclusive modes include clusters/multiple workgroups,
`tg_split`, expert scheduling, `_isfirst`, reinitialization with live named joins,
privileged barriers, arbitrary raw symbolic format
reinterpretation, legacy CDNA cache visibility certificates, advanced tensor
padding/multicast/gather/iteration/atomic-barrier modes and nonzero tensor waits.
They are not projected into a weaker selected-workgroup safety claim. Positive
tests require a complete verified execution; deliberate unsupported controls
remain a distinct test category.

The signal-time count field supports explicit M0 bits22:16 with nonzero values
1–127; zero preserves the current count and NULL remains a no-op. Equality
between a new expected count and existing arrivals remains inconclusive because
the pinned public rules disagree. Unordered conflicting setters and setter/drop
pairs also remain inconclusive; identical explicit setters may commute.

Each instruction records target-selected source spans, physical masks and
register accesses. Semantic records identify exact counter charges, memory
footprints, sources, destinations and instruction associations. Complete traces
must end each wave. Missing or forged records prevent positive graph
verification. Waits can establish completion; a barrier or ordinary trace order
cannot fabricate it. Only completed memory effects participate in split-barrier
publication, and replay-source retention is a separate readiness obligation.

The source pins in `fixtures/semantic-references.json` provide reachable public
LLVM ABI, counter, execution, memory and DMA contracts plus independent FlyDSL
layout formulas. Compiler probes in `fixtures/boundary-assembler.json` do not
certify semantic execution. Full native tests use independent dense/basis formulas and typed
assemble-valid harmful mutations. Exact integers are compared without f64
coercion. All positive tests require every configured output, not a sample.

The twenty portable workload ports use bounded complete computations on decoded
F32 mathematical inputs. Their dimensions and dtype/shape exclusions are explicit
in `portable-cases.json`; BitDelta uses concrete index-generated packed words.
These ports are family coverage, not original PTX binary/shape/dtype equivalence.
The separate original32 source registry retains all original configured extents,
storage widths and output coordinates through explicit native mappings.
Its [stage table](full-parity/stage-support.md) identifies parser, lowering,
value, event and checker obligations; source registration alone proves none of
the required full-workload gates.
WMMA raw finite F16/BF16 encodings have separate native tests. No physical IEEE
rounding, NaN, denormal, approximation-error or whole-device execution claim is
made. Hardware numerical agreement is supporting evidence for the tested programs;
it does not replace the graph obligations or expand the selected instruction model.

## M1 integer profile

`volta check-program` selects profile `gfx1250-b0-m1-int/1` with dependency
profile `gfx1250-b0-sched0-m1/1` for one-wave fuzzer integer kernels. It
is opt-in: without it the rules above, and gfx942/gfx950, are unchanged. The
profile admits the frozen allowlist in [m1/schema](m1/schema/m1_int_subset.json)
(40 body operations, the wrapper mnemonics with their exact operand forms,
`s_max_i32` only when the request enables it and `s_cmp_eq_u32` only in
calibration controls). Every other mnemonic of the entry, including
unreachable code and forms the parser refuses, is reported as a refused
operation (MODE access, floating point, excluded operation, pending hardware
control, not in profile or operand form) and the program is unsupported.

| Rule | Profile behavior |
| --- | --- |
| Source layout | Directive and `amdhsa.target` use one of the two spellings without features; code object 6, metadata 1.2; an explicit `.size kernel, .-kernel` ends right after `s_endpgm`; `.p2alignl`/`.fill` padding after it is retained and never executed |
| S_MAX SCC | `s_max_u32`/`s_max_i32` set SCC for strict `>`, as measured on hardware (the CDNA5 guide prints `>=`, which the default contract keeps); `s_min` keeps strict `<`. Only this profile changes it |
| `s_wait_alu` | Raw unsigned16 or `depctr_<field>(n)` lists, decoded with LLVM's DEPCTR layout (`hold_cnt` raw). It is recorded as an ALU-only wait and never completes KM, load, store, DS or X counts; SCHED_MODE 0 interlocks ALU dependencies |
| Kernarg | `s_load` results are ready only after `s_wait_kmcnt 0` |
| `global_prefetch_b8` | Scalar-base form with optional offset and scope. No data, no destination, no load count: one X-count lease holding its address sources (overwriting them before the lease retires is a readiness failure) and a one-byte-per-lane footprint. `SCOPE_SE`/`DEV`/`SYS` are speculative and exempt from object bounds; `SCOPE_CU` (the default) must lie inside an object. Temporal-hint modifiers are unsupported |
| Guards and objects | `out` is the only object; `pre`/`post` guards are forbidden regions. Every GLOBAL load, store and prefetch footprint is checked before issue; a byte in a guard (any scope) or a footprint outside every object records an attempted-access fault and aborts. The checker replays the prefix and certifies a guard, out-of-object or non-speculative-prefetch rejection with the lowest offending lane; the checker also validates every completed global effect against the layout |
| Raw-u32 outputs | Every word of `out` is its exact stored bits (untyped, unsigned and two's-complement signed integers, concrete predicate masks, literals) or null with `NOT_WRITTEN`, `SYMBOLIC`, `FLOATING_POINT_DERIVED`, `UNDEFINED_BITS` or `OPAQUE` |
| Entry state | `s[0:1]`, packed lane IDs in `v0` and full EXEC; SCC, VCC and other registers are undefined and reading them aborts as unsupported |

Store completion at `s_endpgm` may remain outstanding. The profile makes no
claim about A0 or other steppings, about termination in general or about device
results. Fixtures, the independent output reference and harmful negatives are
in [m1/](m1/README.md).

## M2 MODE profile

`gfx1250-b0-m2-mode/1` (dependency profile `gfx1250-b0-sched0-m2/1`, request
`ptxv-m2-request/1`) is the m1 profile plus runtime MODE state for
FP_ROUND (MODE[3:0]) and FP_DENORM (MODE[7:4]). Every m1 rule above
applies unchanged; the m1 profile itself still refuses every MODE
operation. The frozen allowlist is [m2/schema](m2/schema/m2_mode_subset.json).

| Rule | Profile behavior |
| --- | --- |
| Operations | `s_getreg_b32 sD, hwreg(HW_REG_WAVE_MODE, 0\|4, 4)`, `s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0\|4, 4), sS`, `s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 0\|4, 4), imm` and `s_round_mode`/`s_denorm_mode imm`, with `imm` 0..15. They are pending operations: refused unless the request enables them after their hardware controls pass. Any other field, width, hwreg or hwreg spelling (including aliases that assemble identically) and larger immediates are `MODE_FORM_NOT_IN_PROFILE`; `s_set_vgpr_msb` is an excluded operation (`EXCLUDED_OP`). `s_wait_alu 0xfffe` is accepted in addition to the m1 encodings |
| State | Entry fields come from the descriptor's `float_round_mode_32/16_64` and `float_denorm_mode_32/16_64` (2 bits each); a missing control leaves its field undefined and reading it aborts. A read returns the exact zero-extended field; SETREG from an undefined or symbolic SGPR aborts. No instruction of the profile computes a MODE-dependent result, so outputs stay exact raw words; floating-point rounding and denormal behaviour are not modeled |
| Events | Each executed MODE operation records a `ModeAccess` event (operation, field, value) after its instruction record |
| `mode_order` | The profile does not model when a MODE read completes relative to a later MODE write. A MODE read followed by any MODE write (`s_setreg_b32`, `s_setreg_imm32_b32`, `s_round_mode`, `s_denorm_mode`; any field, any distance) is inside the profile only when an `s_wait_alu` whose SA_SDST field is 0 lies between them; otherwise the pair is unresolved (UNKNOWN), never PASS or FAIL. A MODE write followed by a read or write is in program order; this rests on hardware readback controls |
| Report | `ptxv-m2-report/1`: the m1 report plus the eighth check `mode_order` (`NOT_APPLICABLE` without MODE operations) and a `mode` record of entry fields, every access and final fields |

MODE support establishes state readback and access order only; it makes no claim
about floating-point results. Fixtures and controls are in [m2/](m2/README.md).


## Litmus profile

`gfx1250-b0-litmus/1` (dependency profile `gfx1250-b0-sched0-litmus/1`, request
`ptxv-litmus-request/1`) checks fuzzer litmus kernels of the `bufload`,
`bufstore` and `asynclds` templates. It shares the m1 source layout,
guard, bounds-certificate, kernarg, prefetch and raw-word rules but has its own
allowlist ([litmus/schema](litmus/schema/litmus_b0_subset.json)); operations of
the other litmus templates (async stores from LDS, `s_wait_idle`, named
barriers, swizzled buffers) are refused by name.

| Rule | Profile behavior |
| --- | --- |
| Objects | Contiguous `out`/`data` arrays start with the exact words the launcher writes (`fill_u32` or `index_base_u32 + i`); unwritten words report those values. LDS is undefined at entry and the request's LDS size must equal the descriptor |
| Buffer resource | Base 56:0, `num_records` 101:57 in bytes, stride 121:108, OOB_SELECT 125. Reserved or type bits, stride scale, swizzle, `num_records` 0 (unbound) or all ones (unchecked) are refused |
| Buffer address | Untyped B32-B128 with a VGPR offset (`offen`) or none (`off`), null SGPR offset: offset = immediate + VGPR, no index. Offsets not DWORD aligned are refused |
| Range check | DWORD `c` is out of range when `offset + 4c + 4 > num_records` or, with OOB_SELECT, `offset + 4c + 4 > stride`. Out-of-range load DWORDs read zero and out-of-range store DWORDs write nothing; counters are charged per instruction |
| Envelope | Every lane's unclamped footprint (all DWORDs, as if in range) must lie in one object and outside the guards before issue; otherwise the access is certified as a guard or out-of-object rejection |
| Same-address stores | Lanes of one buffer store that write the same bytes are ordered by lane; the highest lane's data remains |
| Async loads to LDS | `global_load_async_to_lds_b32/b64/b128 vLDS, vOFF, s[n:n+1]`: global `SADDR + VADDR + IOFFSET`, LDS `VDST + IOFFSET`, completed by ASYNCcnt. IOFFSET must be 0..0xffff; footprints must be naturally aligned and the LDS footprint inside the allocation, otherwise the copy is refused |
| Report | `ptxv-litmus-report/1`: the seven m1 checks, `outputs` with contract `raw-u32-image` (object images in request order), and a `litmus` record with every buffer range check (masks, in- and out-of-range DWORD counts) and async copy |

A pass states that the documented rule holds for the recorded launch;
device results that differ remain visible to the caller's own comparison.
Fixtures are in [litmus/](litmus/README.md).

### Litmus/2: named-barrier JOIN read-back

`gfx1250-b0-litmus/2` (dependency profile `gfx1250-b0-sched0-litmus/2`, request
`ptxv-litmus-request/2`, allowlist
[litmus2/schema](litmus2/schema/litmus_b0_t2_subset.json)) is `gfx1250-b0-litmus/1`
plus the fuzzer's `barjoin` template. Every litmus/1 rule applies unchanged;
async stores from LDS, global loads and swizzled buffers stay refused by name.

| Rule | Profile behavior |
| --- | --- |
| Geometry | One workgroup of 32 or 64 workitems (one or two wave32 waves); named barriers are the descriptor's `.amdhsa_named_barrier_count` x 4, higher ids are refused |
| Named barriers | `s_barrier_init m0` (member count from M0[22:16]), `s_barrier_join 0..16`, `s_barrier_signal 1..16`, `s_barrier_wait 1..16` with the existing object/phase model |
| Unconsumed completion | A JOIN while the wave's joined barrier completed a phase the wave has not waited on is refused: when that completion's notification reaches the wave relative to the later JOIN is not settled |
| `s_wait_idle` | Every memory counter reaches zero (ALU dependency counters are interlocked under SCHED_MODE 0) |
| STATE_PRIV read | `s_getreg_b32 sD, hwreg(HW_REG_WAVE_STATE_PRIV)` (whole register) directly after `s_wait_idle`: bits 8:4 = joined named barrier id, bit 3 = 0, every other bit undefined. Refused before any JOIN, without the preceding `s_wait_idle`, or with an unconsumed completion |
| Masking | `s_and_b32` with an immediate inside the defined bits gives an exact word |
| Same-address stores | Lanes of one global store that write the same bytes are ordered by lane |
| Report | `ptxv-litmus-report/2`: the litmus/1 report with rules J1-J7 and assumption `P-BJ-CONSUMED` |

A JOIN or STATE_PRIV read before the wave has waited on every completion of
its joined barrier is refused.
Fixtures are in [litmus2/](litmus2/README.md).

## WMMA and multiwave profiles

`gfx1250-b0-wmma/1` (dependency profile `gfx1250-b0-sched0-wmma/1`, request
`ptxv-wmma-request/1`) is the m2 profile plus
`v_wmma_f32_16x16x32_bf16` for one-wave fuzzer kernels.
`gfx1250-b0-mw/1` (dependency profile `gfx1250-b0-sched0-mw/1`, request
`ptxv-mw-request/1`) checks the fuzzer's multiwave aggressor/victim kernels:
one workgroup of 2..8 wave32 waves executing the same body operations plus
that family's wrapper. Every m1/m2 rule applies unchanged; the
m1/m2 and litmus profiles still refuse these operations. The frozen
allowlists are [wmma_b0.json](mw_wmma/schema/wmma_b0.json) and
[mw_b0.json](mw_wmma/schema/mw_b0.json).

| Rule | Profile behavior |
| --- | --- |
| WMMA operands | Four 8-VGPR tuples (D, A, B, C), no modifiers; EXEC must be all ones (otherwise execution is refused). Other WMMA forms are refused |
| WMMA results | Computed exactly over the rationals. An element is reported as binary32 bits only when every input is normal or zero, every term (C and the 32 products) is a multiple of one power of two q >= 2^-126, the magnitude sum is below 2^24 q and the result is nonzero: every partial sum is then exactly representable, independent of order, rounding and denormal handling. Other elements have no bits (`FLOATING_POINT_DERIVED`) |
| `wmma_hazards` | For an F16/BF16 WMMA, with slots counted as V_NOPs or vector-ALU instructions touching none of its registers (transcendental and WMMA instructions are not slots): a later WMMA reading its D as A/B within 5 slots, a VALU reading or writing its D or writing its A/B/C within 4 slots, and two WMMA with identical operands and no vector-ALU instruction between are certified failures. A later WMMA reading its D as C, writing its D or writing its A/B/C within 5 slots is unresolved (UNKNOWN, with witnesses): the profile neither accepts nor rejects these by the published dependency rules |
| Multiwave geometry | Block 32*W, W = 2..8; all waves of the selected workgroup execute; each wave must end at `s_endpgm` |
| Evidence reads | `s_get_shader_cycles_u64` and `s_getreg_b32 sD, hwreg(HW_REG_WAVE_HW_ID1)` (whole register) write fresh opaque values: they may be moved and stored, every other use aborts as unsupported. A field-selected HW_ID1 read is a refused form |
| XNACK replay | Each wave stays in the wave-start single-group replay mode: the first non-memory instruction after memory instructions drains XCNT, which releases the source leases of those memory instructions |
| Same-address stores | Lanes of one GLOBAL store that write the same bytes are ordered by lane |
| Outputs | `raw-u32-mw`: the request declares the `out` words that must hold an opaque value and those that must stay unwritten; every other word must be exact. Status `COMPLETE_AS_DECLARED` when only declared words are undefined |
| Report | `ptxv-{wmma,mw}-report/1`: the m2 report plus the ninth check `wmma_hazards`, every wave in `execution.waves` |

No claim is made about floating-point equivalence outside the exact regime,
about earlier kernels on the device (no predecessor premise), about
cross-workgroup behavior or about the hardware-ID and cycle values, which the
caller judges from device results. Fixtures are in [mw_wmma/](mw_wmma/README.md).

## WMMA types profile

`gfx1250-b0-wmma/2` (dependency profile `gfx1250-b0-sched0-wmma2/1`, request
`ptxv-wmma-request/2`) is `gfx1250-b0-wmma/1` plus `v_wmma_f32_16x16x32_f16`,
`v_wmma_i32_16x16x64_iu8`, `v_wmma_f32_16x16x64_fp8_fp8`,
`v_wmma_f32_16x16x128_fp8_fp8` (16-VGPR A and B) and an EXEC=0 spacer whose
only EXEC writes are `s_mov_b32 exec_lo, 0` and `s_mov_b32 exec_lo, -1`. The
frozen allowlist is [wmma_types_b0.json](wmma2/schema/wmma_types_b0.json);
`gfx1250-b0-wmma/1` is unchanged and still refuses these forms.

| Rule | Profile behavior |
| --- | --- |
| Spacing | Per type, a later WMMA reading D as A/B and a VALU reading/writing D or writing A/B/C need F16/BF16 5 and 4, FP8 16x16x64 2 and 1, FP8 16x16x128 4 and 3, IU8 9 and 8 slots (certified failures below). Unlisted WMMA-to-WMMA dependencies below the type's WMMA count are unresolved |
| EXEC = 0 | A slot is a V_NOP or an independent vector-ALU instruction issued with a nonzero EXEC. Instructions issued with EXEC = 0 are never counted; a dependency met only by counting them is unresolved (`WMMA_EXEC0_SLOTS_UNRESOLVED`), never a failure and never accepted. A WMMA under EXEC = 0 is not executed |
| F16 results | As BF16, with inputs normal in F16 (magnitude at least 2^-14) or zero |
| FP8, IU8 results | Evaluated only for uniform A and B tiles (every byte of every register in every lane equal), so the result is C + K*a*b independent of the 8-bit layout. FP8 (E4M3) results carry bits in the exact-representable regime with inputs normal in E4M3 (at least 2^-6); IU8 without NEG modifiers is unsigned and is evaluated for bytes below 0x80, concrete C words and results inside int32. Anything else is unsupported |
| Report | `ptxv-wmma-report/2`: as `/1`, with assumptions `A-WMMA-UNIFORM-8BIT` and `A-WMMA-EXEC0-SLOTS` |

Fixtures are in [wmma2/](wmma2/README.md).

## WMMA scale profile

`gfx1250-b0-wmma/3` (dependency profile `gfx1250-b0-sched0-wmma3/1`, request
`ptxv-wmma-request/3`) is `gfx1250-b0-wmma/2` plus
`v_wmma_f32_16x16x64_{fp8_bf8,bf8_fp8,bf8_bf8}`,
`v_wmma_f32_16x16x128_{fp8_bf8,bf8_fp8,bf8_bf8}`, `v_wmma_f32_16x16x128_f8f6f4`
(formats in `matrix_{a,b}_fmt`), `v_wmma_f32_32x16x128_f4` (32x16 C and D: 16
VGPRs) and the 16-byte block-scaled forms `v_wmma_scale[16]_f32_16x16x128_f8f6f4`
and `v_wmma_scale[16]_f32_32x16x128_f4` (two scale operands, formats in
`matrix_{a,b}_scale_fmt`, lanes in `matrix_{a,b}_scale`). The frozen allowlist
is [wmma_scale_b0.json](wmma3/schema/wmma_scale_b0.json); `gfx1250-b0-wmma/2` is
unchanged and refuses these forms.

| Rule | Profile behavior |
| --- | --- |
| Spacing | BF8 16x16x64 2 and 1, BF8 16x16x128 4 and 3, F8F6F4 with both formats FP4 2 and 1, other F8F6F4 4 and 3, F4 32x16x128 4 and 3 slots (scaled or not); certified failures below, unlisted WMMA-to-WMMA dependencies below the WMMA count unresolved |
| Scaled forms | One 16-byte instruction: its load-scale half is never a slot. No documented row names scale operands, so a scale operand reading an earlier WMMA's D, or a VALU writing an earlier WMMA's scale registers, below the count is unresolved (`WMMA_SCALE_SOURCE_BELOW_POLICY`) |
| Operand forms | Tuple widths by format (FP8/BF8 16, FP6/BF6 12, FP4 8; 32x128 FP4 A 16), VGPR scale operands (one, or an even pair for SCALE16), legal data/scale format pairs, no reuse hints, no NEG or OP_SEL, 32x16 C equal to D or disjoint; anything else is refused |
| Results | Evaluated only for uniform A and B tiles (one byte, nibble or packed 6-bit code everywhere) and uniform scale tuples, so the result is C + K*a*b*sa*sb independent of the matrix and scale layouts. Exact bits for elements normal or zero in their own format, normal power-of-two scales, the exact-representable regime and C words that are themselves concrete bits; other uniform values give reals; non-uniform tiles, NaN/Inf, E4M3 scale bytes with bit 7 set and non-VGPR scales are unsupported |
| Report | `ptxv-wmma-report/3`: as `/2`, with assumptions `A-WMMA-UNIFORM-SMALL`, `A-WMMA-F6-PACKING`, `A-WMMA-BLOCK-SCALE`, `A-WMMA-VOP3PX2` and `A-WMMA-32X16-LAYOUT` |

Fixtures are in [wmma3/](wmma3/README.md).

## WMMA 16-bit output and SWMMAC profile

`gfx1250-b0-wmma/4` (dependency profile `gfx1250-b0-sched0-wmma4/1`, request
`ptxv-wmma-request/4`) is `gfx1250-b0-wmma/3` plus `v_wmma_f16_16x16x32_f16`,
`v_wmma_bf16_16x16x32_bf16`, `v_wmma_bf16f32_16x16x32_bf16`,
`v_wmma_f16_16x16x{64,128}_{fp8,bf8}_{fp8,bf8}` and every SWMMAC form
(`v_swmmac_{f32,f16,bf16,bf16f32}_16x16x64_*`,
`v_swmmac_{f32,f16}_16x16x128_{fp8,bf8}_{fp8,bf8}`,
`v_swmmac_i32_16x16x128_iu8`). The frozen allowlist is
[wmma_sparse16_b0.json](wmma4/schema/wmma_sparse16_b0.json);
`gfx1250-b0-wmma/3` is unchanged and refuses these forms.

| Rule | Profile behavior |
| --- | --- |
| Spacing | F16/BF16-input 16-bit outputs 5 and 4, FP8/BF8 16x16x64 F16 output 2 and 1, 16x16x128 4 and 3, SWMMAC (16-bit and FP8/BF8 inputs) 3 and 2, SWMMAC IU8 5 and 4 slots; certified failures below, unlisted WMMA-to-WMMA dependencies below the WMMA count unresolved |
| SWMMAC | C is the destination. The sparsity index is a documented source: a D read as an index, or a VALU writing an index register, below the count is a certified failure |
| Operand forms | 16-bit D 4 VGPRs (BF16F32 C 8); SWMMAC D 4 or 8, packed A 8, B 16, index one VGPR (16-bit inputs) or an even pair (8-bit); no modifiers; anything else is refused |
| Results | Uniform tiles only. F16 results carry bits while every partial sum is an integer below 2^11 (exact for any accumulator of at least F16 precision) and C halves are concrete; BF16 and BF16F32 results are never certified. SWMMAC needs uniform packed A, a B uniform within each K half (registers 0-7 and 8-15) and uniform index words whose pairs all satisfy idx0 < idx1: D = C + (K/4)*a*(b0+b1); otherwise unsupported |
| Report | `ptxv-wmma-report/4`: as `/3`, with assumptions `A-WMMA-F16-OUTPUT-REGIME`, `A-WMMA-16BIT-LAYOUT` and `A-WMMA-SPARSE-B-HALVES` |

Fixtures are in [wmma4/](wmma4/README.md).

## Multiwave WMMA-types profile

`gfx1250-b0-mw/2` (dependency profile `gfx1250-b0-sched0-mw2/1`, request
`ptxv-mw-request/2`) is the multiwave profile `gfx1250-b0-mw/1` with the
operations of `gfx1250-b0-wmma/4` in any wave of the selected workgroup. The
frozen allowlist is [mw_types_b0.json](mw2/schema/mw_types_b0.json);
`gfx1250-b0-mw/1` is unchanged and refuses these forms.

| Rule | Profile behavior |
| --- | --- |
| Geometry and wrapper | As `gfx1250-b0-mw/1`: one workgroup of 2..8 wave32 waves, opaque shader-cycle and HW_ID1 reads, single-group replay draining, lane-ordered same-address global stores, no communication between waves |
| WMMA | As `gfx1250-b0-wmma/4`, applied to each wave's own instruction stream; nothing about other waves of the same SIMD is modelled |
| Report | `ptxv-mw-report/2`: the `ptxv-mw-report/1` shape (`raw-u32-mw` outputs with declared opaque and unwritten words), the multiwave assumptions followed by the `/4` WMMA assumptions |

Fixtures are in [mw2/](mw2/README.md).

## Multi-workgroup predecessor profile

`gfx1250-b0-mwpred/1` (dependency profile `gfx1250-b0-sched0-mwpred/1`, request
`ptxv-mwpred-request/1`) checks a fixed kernel launched as a grid of
workgroups. The frozen allowlist is
[mw_pred_b0.json](mwpred/schema/mw_pred_b0.json): the `gfx1250-b0-wmma/1` body
and the multiwave wrapper plus the SE/XCC id message return.

| Rule | Profile behavior |
| --- | --- |
| Geometry | Grid [G, 1, 1] with G = 1..1024, block of 1..32 wave32 waves, `workgroups: "all"`: every workgroup (k, 0, 0) is analysed with its id in TTMP9 |
| Composition | The tenth check `workgroup_isolation` passes only when the bytes written by different workgroups are pairwise disjoint and no workgroup reads bytes another workgroup writes. Overlapping writes are a certified failure (`CROSS_WORKGROUP_WRITE_OVERLAP`); a read of bytes another workgroup writes stays unknown (`CROSS_WORKGROUP_READ_NOT_MODELLED`): communication between workgroups is not modelled. The other nine checks are merged over the grid (worst result, witnesses tagged with their workgroup) |
| Outputs | Each composed word comes from its single writer; declared opaque and unwritten words as in the multiwave profile |
| Evidence | `s_getreg_b32 hwreg(HW_REG_WAVE_HW_ID1)` and `s_sendmsg_rtn_b32 sD, sendmsg(MSG_RTN_GET_SE_AID_ID)` write opaque values; the message return must be followed immediately by `s_wait_kmcnt 0`; `s_mov_b32 sD, ttmp9` reads the workgroup id |
| Report | `ptxv-mwpred-report/1`: the multiwave report shape with ten checks and a per-workgroup summary `workgroups` |

Fixtures are in [mwpred/](mwpred/README.md).

## Communicating-waves profile

`gfx1250-b0-comms/1` (dependency profile `gfx1250-b0-sched0-comms/1`, request
`ptxv-comms-request/1`) checks programs whose waves exchange data through LDS
or GLOBAL memory, alone or as a grid of workgroups that do not communicate.
The frozen allowlist is [comms_b0.json](comms/schema/comms_b0.json): the
integer body plus `ds_store`/`ds_load` and `global_store`/`global_load`
b32/b64/b128, the three completion waits and `s_barrier_init`/`join`/`signal`/
`wait`, with the multiwave wrapper.

| Rule | Profile behavior |
| --- | --- |
| Geometry | Grid [G, 1, 1] with G = 1..64, block of 2..8 wave32 waves, `workgroups: "all"`: every workgroup (k, 0, 0) is analysed with its id in TTMP9; the request declares the static LDS allocation (`lds_bytes`, equal to the descriptor's group segment size) |
| Ordering | Transfers between waves are checked by happens-before: a store whose completion the producer waited for (`s_wait_dscnt 0`, `s_wait_storecnt 0`) before a barrier signal is seen by a load after the matching barrier wait; an unordered cross-wave conflict is a certified failure (`memory_conflicts`); a read of LDS no wave wrote is unknown |
| Visibility | Inside the analysed workgroup an ordered GLOBAL write and read are visible at any scope (assumption `Gfx1250B0WorkgroupCoherentProfile`: the workgroup's waves share one workgroup-coherent vector cache); the default contract still requires `SCOPE_SYS` at both endpoints |
| Barriers | The workgroup barrier (-1) and named barriers 1..2: `s_barrier_init m0` with the member count in M0, `s_barrier_join` once per wave, signal and wait; a join while the joined barrier has an unconsumed completion stays unknown |
| Scalar memory | The only scalar load is the kernarg pointer load `s_load_b64 s[2:3], s[0:1], 0x0`; any other scalar load is refused (scalar loads are not kept coherent with vector stores) |
| Composition | `workgroup_isolation` as in the multi-workgroup predecessor profile: overlapping writes are a certified failure, a read of bytes another workgroup writes stays unknown; communication between workgroups is not modelled |
| Report | `ptxv-comms-report/1`: the multiwave report shape without the MODE record, eight checks (the seven integer checks merged over the grid, then `workgroup_isolation`) and the per-workgroup summary `workgroups` |

Fixtures are in [comms/](comms/README.md).

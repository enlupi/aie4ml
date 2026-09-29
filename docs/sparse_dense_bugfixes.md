# Two Bug Fixes on `sparse_dense_support`

Branch `sparse_dense_support`. Two defects found while auditing the data types the dense and
sparse paths actually accept, and what was changed to fix them. Both were found by probing the
lowering pipeline and the toolchain directly rather than by reading the tables, which is the
only reason either surfaced: neither is visible in the IR-only test suite, and one of them is
the reason the other stayed hidden.

Paths written `$AIEDOCS` refer to the local AIE documentation tree at
`/eos/home-e/enlupi/workspace/repo/docs/aie-ml/2025.2` (installed version 2025.2). They are
outside this repository.

---

## Bug 1 — AIE-MLv2 had a sparse `int16 x int8` mode that does not exist

**File:** `src/aie4ml/op_impls/families/matmul/common.py`

### What was wrong

`SPARSE_MICROTILE_OPTIONS['AIE-MLV2']` listed a microtile for the `('int16', 'int8')` precision
pairing:

```python
'AIE-MLV2': {
    ('int8', 'int8'): [(4, 16, 8)],
    ('int16', 'int8'): [(8, 16, 8)],   # <- removed
    ('int16', 'int16'): [(2, 8, 8)],
},
```

There is no such sparse mode on AIE-MLv2. Two independent sources agree:

1. **The AI Engine API sparse modes table**
   (`$AIEDOCS/aie_api/group__group__mmul_html.md`, "Supported Sparse Matrix Multiplication
   Modes", line 512). Its columns are `8b x 4b | 8b x 8b | 16b x 8b | 16b x 16b | bfloat16 x
   bfloat16`. Splitting the `AIE-MLv2` row on its pipes gives six cells — one label plus five
   columns — and the `16b x 8b` cell is **empty**:

   | Arch. | 8b x 4b | 8b x 8b | 16b x 8b | 16b x 16b | bf16 x bf16 |
   | --- | --- | --- | --- | --- | --- |
   | AIE-ML/XDNA1 | 4x32x8 | 4x16x8, 8x16x8ᵃ, 4x16x16ᵃᵇ | **2x16x8**, 4x16x8ᵃ | 2x8x8, 4x8x8ᵃ, 2x8x16ᵃᵇ | 4x16x4, 4x16x8ᵃᵇ |
   | XDNA2 | | 4x16x8, 8x16x8 | 2x16x8, 4x16x8 | 2x8x8, 4x8x8 | |
   | AIE-MLv2 | | 4x16x8, **8x16x8** | *(empty)* | 2x8x8, 4x8x8 | 4x16x8 |

   Note where `8x16x8` actually lives on the `AIE-MLv2` row: in the **`8b x 8b`** column. The
   removed entry was a misread of the neighbouring column. `2x16x8`/`4x16x8` are the `16b x 8b`
   shapes, and they belong to `AIE-ML/XDNA1` and `XDNA2` — not to `AIE-MLv2`.

2. **UG1603** (`$AIEDOCS/ug1603/...-2025.2.txt`, line 4606), which states the rule directly:

   > "For AIE-ML v2, both matrices should have the same data type: int8xint8, int16xint16, or
   > bfloat16xbfloat16."

   `int16 x int8` is a mixed-dtype pairing, so it is excluded by that rule regardless of shape.

### Why it mattered

This was not a dormant table row. The sparse variants sit at `plevel = 20`, above the dense
ones, and selection is weight-driven and **on by default**. So any AIE-MLv2 model with int16
activations, constant int8 weights, and an already-2:4 weight matrix would be compiled sparse
automatically, with no directive anywhere.

Verified before the fix, on `vek385_base`:

```
int16 act x int8 wts  [auto]  SPARSE  variant=dense.sparse.b.r.v1  microtile=(8, 16, 8)
```

and the resulting project does not compile. `make x86com` fails with 8 errors, the root being
that no sparse overload exists for that instantiation:

```
dense_bias_relu.cpp:243:40: note: in instantiation of function template specialization
  'aie::mmul<8, 16, 8, short, signed char, acc32>::to_vector<short>' requested here
...
ERROR: [aiecompiler 77-753] ... Compilation Failed
```

**This broke the "sparsity can never break a build" guarantee** documented in
`docs/sparse_dense_implementation.md` §1 and §4.3. That guarantee rests on
`selection_is_speculative()` returning `True` under `'auto'`, so that a variant which offers
itself and then fails to `resolve()` hands the node to the next candidate. But this failure is
not a resolve failure — `resolve()` succeeds, `validate_config()` passes (the microtile is in
the variant's own table, which is exactly the table that was wrong), and the project is written
out. The failure happens in `v++`, long after the resolver has finished and far outside the
scope of any fallback it can offer.

The generation gate is worth being precise about: `select_generation_key` maps the `vek385_base`
device to `'AIE-MLV2'`, so the `AIE-MLv2` row of the table is the one that governs. `XDNA2` does
have `16b x 8b` sparse modes, but it is a separate architecture and a separate row; its entries
do not transfer.

### The fix

The entry was deleted. Nothing else changed — no comment was added in its place, because a
pairing the hardware does not have needs no explanation at the point where it is absent; the
table is a transcription of the vendor table and now matches it.

```diff
 'AIE-MLV2': {
     ('int8', 'int8'): [(4, 16, 8)],
-    ('int16', 'int8'): [(8, 16, 8)],
     ('int16', 'int16'): [(2, 8, 8)],
 },
```

`sparse_capability()` already handles an absent pairing correctly, so removing the row was
sufficient and no logic needed touching.

Pinned by **`test_aie_mlv2_has_no_mixed_dtype_sparse_pairing`** in `tests/test_sparse_dense.py`,
which asserts the AIE-MLv2 table offers no mixed-dtype pairing at all rather than only checking
the one row that was wrong. It is a table assertion rather than a lowering test because the suite
deliberately targets Vek280 only, so a lowering test would be the first thing in `tests/` to need
another device. Verified to fail when the entry is put back (`AssertionError: assert ('int16',
'int8') not in {...}`) and to pass once it is removed again — a regression test that was never
seen to fail proves nothing.

After the fix, on `vek385_base`:

| Directive | Before | After |
| --- | --- | --- |
| `auto` (default) | sparse, microtile `(8,16,8)`, **fails in `v++`** | falls back to **dense**, `(4,4,8)`, silently — as the default promises |
| `'2of4'` | sparse, **fails in `v++`** | `NotImplementedError: ... no sparse aie::mmul overload for this precision pairing on 'AIE-MLV2'` |

AIE-ML is untouched and still selects sparse `(2,16,8)` for the same pairing, which is correct —
`2x16x8` is unfootnoted in the `AIE-ML/XDNA1` row.

### Why no test caught it

`tests/aie_platform.py` hardcodes `TARGET_DEVICE = 'Vek280'`, and says so deliberately:

> "The tests are written against AIE-ML microtiles and VEK280 array geometry, so the deduction
> is deliberately confined to that device rather than picking whatever is newest."

Every test module routes through `resolve_aie_part()`, and `grep -rn "vek385\|MLV2" tests/`
returns nothing. **AIE-MLv2 is untested for both the dense and the sparse path.** That is a
defensible scoping decision, but it means the entire AIE-MLv2 column of both microtile tables is
unexercised, and a transcription error in it is invisible. The other AIE-MLv2 rows were
re-checked against the vendor table by hand while fixing this one and are correct.

---

## Bug 2 — `make` reported success for failed compiles and simulations

**File:** `src/aie4ml/templates/firmware/Makefile.jinja`

### What was wrong

Six recipes in the emitted Makefile pipe their tool into `tee` so the log is kept:

| Line | Recipe | Command |
| --- | --- | --- |
| 125 | `compile` | `v++ --compile ... --target=hw ... |& tee log` |
| 130 | `x86com` | `v++ --compile ... --target=x86sim ... |& tee log` |
| 133 | `aiesim` | `aiesimulator --pkg-dir=... |& tee -a log` |
| 136 | `x86sim` | `x86simulator --pkg-dir=... |& tee -a log` |
| 139 | `profile` | `aiesimulator --online -wdb -text --profile |& tee -a log` |
| 142 | `trace` | `aiesimulator --online -wdb -text |& tee -a log` |

A shell pipeline's exit status is the status of its **last** command. `tee` succeeds whenever it
can write its file, so every one of these recipes exited `0` no matter what the compiler or
simulator did. `make` then reported success.

That defeats the check one layer up. `AIEModel.build()` (`src/aie4ml/model.py:95`) is written
correctly:

```python
result = subprocess.run(cmd, cwd=output_dir, env=env, ...)
if result.returncode != 0:
    raise RuntimeError(f'Make target "{make_target}" failed for project ...')
```

but it never saw a nonzero code to raise on. The observable symptom, from the Bug 1
investigation above, was this pair of lines in one run — the compiler reporting failure, and
aie4ml reporting success for the same build:

```
ERROR: [aiecompiler 77-753] ... Compilation Failed
(WARNING:0, CRITICAL-WARNING:0, ERROR:1)
COMPILE: SUCCEEDED
```

### Why it mattered

A build that cannot compile looked like a build that worked. Concretely:

- **`build()` and `compile()` silently did nothing.** Any caller — a user, a notebook, a script
  — got a clean return and an incomplete project.
- **Failures surfaced far from their cause.** Tests that call `build()` then `predict()` still
  failed, but at the point where `predict()` could not find a simulator output file, several
  layers from the compiler error that caused it. `tests/aie_platform.py`'s docstring already
  complains about exactly this shape of problem for the stale-platform case: *"`v++` rejects a
  platform it does not have with an error that surfaces far from its cause — a missing simulator
  output file, several layers up."* The `tee` bug was one of the reasons that class of error had
  to be diagnosed that way.
- **It weakened every simulator test in the suite.** "x86sim passes" was a weaker claim than it
  looked, since a compile error did not fail the build step. This mattered most for the sparse
  work, where §5.3 of the implementation notes already establishes that x86sim alone is
  insufficient — the suite's evidentiary value was lower still than that caveat assumed.

### The fix

`SHELL := /bin/bash` was already set on line 1, so the fix is one line next to it:

```diff
 SHELL := /bin/bash
+# The recipes below pipe into `tee`, so without pipefail the exit status is tee's (always 0)
+# and a failed v++ or simulator run still makes `make` succeed.
+.SHELLFLAGS := -o pipefail -c
```

`pipefail` makes a pipeline return the status of the rightmost command that failed, so a
nonzero `v++` or simulator status now propagates through `tee`, out of `make`, and into the
`RuntimeError` that `build()` was always ready to raise. Setting it via `.SHELLFLAGS` covers all
six recipes at once rather than rewriting each pipeline, and it also makes the existing `|&`
usage safe rather than incidental — `|&` is itself a bashism that only worked because `SHELL`
was already `bash`.

### Verification

A project was emitted, `#error deliberate_failure_to_test_pipefail` was appended to the emitted
`dense_bias_relu.cpp`, and `build(make_target='x86com')` was called:

```
MAKEFILE has .SHELLFLAGS pipefail: True
build() RAISED RuntimeError: Make target "x86com" failed for project "p"
```

Before the fix the same probe returned `0` and raised nothing.

---

## Files changed

| File | Change |
| --- | --- |
| `src/aie4ml/op_impls/families/matmul/common.py` | Bug 1: removed the AIE-MLv2 `('int16','int8')` sparse entry |
| `src/aie4ml/templates/firmware/Makefile.jinja` | Bug 2: added `.SHELLFLAGS := -o pipefail -c` |
| `tests/test_sparse_dense.py` | Added `test_aie_mlv2_has_no_mixed_dtype_sparse_pairing` |
| `docs/support.md` | Sparse precision pairings are per generation; recorded what is actually verified (required by `AGENTS.md` §10) |
| `docs/sparse_dense_bugfixes.md` | This file |

## Test results

Staged rather than run as one invocation, because running all of them at once was killed with
SIGKILL (exit 137) — memory pressure from concurrent simulator builds.

| Suite | Result |
| --- | --- |
| IR-only (`-m "not requires_vitis"`) | **44 passed** in 16s (43 + the new regression test) |
| `test_sparse_dense.py -m requires_vitis` (x86sim + aiesim differential) | **2 passed** in 4m32s |
| `test_float_sim.py`, `test_hls4ml_dense.py`, `test_hls4ml_fanout.py -m requires_vitis` | **10 passed** in 5m26s |

**56 passed, 0 failed, 0 skipped** across the three runs.

The Vitis-gated runs matter more than usual here: with `pipefail` now in effect, any build step
that was previously failing silently would start failing loudly. A clean run is therefore
evidence that the existing suite was not resting on a swallowed error.

`ruff` is **not installed** in the `aie4ml_env` conda environment or on `PATH`, so the
`ruff check` / `ruff format` step of the `AGENTS.md` §10 checklist could not be run. The Python
change is a single deleted line and the test addition follows the surrounding style, but this is
unverified rather than clean.

Bug 2 has **no regression test**. Pinning it needs a build that fails for a real toolchain
reason, so the test would be `requires_vitis`, would take minutes, and would work by corrupting
an emitted kernel — a fragile shape of test. The one-line `.SHELLFLAGS` is instead documented
here and commented in the Makefile itself.

---

## Open: AIE-MLv2 sparse produces incorrect results

Found while trying to verify two additional AIE-MLv2 microtiles. **Not fixed** — deferred to a
dedicated AIE-MLv2 study, and the table entries are retained for it.

### The measurement

One x86simulator run on `vek385_base`, all four builds fed the identical 2:4 weight matrix and
input, with AIE-ML dense as the reference (itself verified in x86sim *and* aiesim):

| Build | Microtile | vs AIE-ML dense |
| --- | --- | --- |
| AIE-ML dense | `(4,8,8)` | reference |
| AIE-ML sparse | `(4,16,8)` | **EQUAL** |
| AIE-MLv2 dense | `(8,8,8)` | **EQUAL** |
| AIE-MLv2 sparse | `(4,16,8)` | **DIFFERS — 2001 / 2048 elements** |

This localizes the fault precisely: the AIE-MLv2 *dense* path is correct, so the device entry,
placement, DMA and transport are all fine. Only the AIE-MLv2 *sparse* path is wrong, and it is
wrong for `(4,16,8)` — the **pre-existing** entry, not one of the two shapes being evaluated.

### Suspected cause

The compressed chunk geometry is generation-specific and aie4ml implements only AIE-ML's:

- aie4ml packs with `SPARSE_CHUNK_DATA_BYTES = 64` and a 64-bit mask, taken from the installed
  `aie_ml` model (`aietools/data/aie_ml/lib/me_native.h`, `mask_length_sparse = 64`). That
  geometry is confirmed correct on AIE-ML by the bit-exact aiesim differential.
- UG1603 says of AIE-MLv2: *"For AIE-ML v2, data chunks are 512-bit wide, and the mask and
  alignment is on 64 bits."* — i.e. a different chunking from the one aie4ml emits.
- The installed `aie2ps` model (`aietools/data/aie2ps/lib/me_native.h`) has sparse FIFO types
  but **no `mask_length_sparse` and no `sparse_read8b`/`sparse_read16b`**, so nothing there
  corroborates aie4ml's layout for that architecture.

A wrong chunk geometry is exactly the failure mode §3 of the implementation notes warns about:
it compiles and simulates cleanly while silently producing wrong numbers.

### Correction: the simulator is the leading explanation, not the packer

The chunk-geometry theory above is still open, but it is **no longer the best explanation of the
measurement**, and the measurement is not evidence about the hardware.

`aietools/data/aie2ps/lib/me_native.h` contains **no sparse decompression whatsoever**: no
`decompress_sparse`, no `sparse_read8b`/`sparse_read16b`, no `sparse_pop`, no `sparse_fill`. The
`aie_ml` model has all of them, and they are what x86simulator executes to expand a compressed
tile. The `aie2ps` model has only the sparse FIFO *types* and the `status_sparse_of` flag.

So x86simulator cannot execute a sparse kernel correctly on AIE-MLv2 no matter what the ROM
contains, and a 2001/2048 mismatch is the expected outcome of running one there. **x86sim cannot
validate AIE-MLv2 sparse at all** — which also means the earlier verdict "AIE-MLv2 sparse
produces incorrect results" overstated what was measured. The honest statement is that the path
is unverified and, on this installation, unverifiable.

This inverts the usual §5.3 relationship. On AIE-ML, x86sim is sound but blind to FIFO
contention, so aiesim is the stronger of two usable simulators. On AIE-MLv2, x86sim is not usable
for sparse at all, so aiesim is the *only* option rather than the better one.

aiesimulator is unavailable here: the hardware `compile` target fails with
`ERROR: [aiecompiler 77-6572] Could not load Device Part xc2ve3858-ssva2112-2MP-e-S`, identically
for a plain dense build. The VE3858 device is **partially** installed — comparing it against the
working `xcve2802` path by path:

| Path (under `/opt/Xilinx/2026.1`) | `xcve2802` | `xc2ve3858` |
| --- | --- | --- |
| `data/ips/parts/versal/<part>` (incl. `ssva2112.json`) | present | present |
| `data/versal/me/regdb/<part>` | present | present |
| `data/parts/xilinx/versal/devint/versalaiedge/<part>` | present | **absent** |
| `SharedData/.../vault/versal/data/device_cfg/<part>` | present | **absent** |
| `SharedData/.../vault/versal/data/device_pkgs/<part>` | present | **absent** |
| `data/versal/me/code/<part>` | present | **absent** |
| `data/versal/me/clock/<part>` | present | **absent** |

The AI Engine `me/code` and `me/clock` databases are the notable ones: `regdb` is installed for
both parts, so a probe that checks only register specs looks fine. The fix is to install full
VE3858 / Versal AI Edge Gen 2 device support for this Vitis.

`repro_mlv2/` in the repository root reproduces all of the above in two scripts.

### Why this matters more than an ordinary unsupported-configuration bug

**Sparsity is opt-out.** A Dense layer on a VEK385 whose trained weights are already 2:4 takes
this path with no directive anywhere, so the default outcome on that target is silently wrong
arithmetic. That is a stronger failure than the `int16 x int8` bug in §1, which at least failed
loudly in `v++`.

Mitigations, in increasing order of intrusiveness, for the study to choose from:

1. Document it (done — `docs/support.md`), and set `sparsity: {'mode': 'dense'}` on AIE-MLv2.
2. Have `sparse_capability()` decline AIE-MLv2 pending verification, so those layers compile
   dense automatically. Behaviour-preserving for AIE-ML, and restores "never silently wrong".
3. Derive the chunk geometry per generation and implement the AIE-MLv2 layout, then verify on a
   machine that has the VEK385 part.

Option 2 is one line and is what the branch's own "numerically free" claim requires; it is left
undone only because retaining the current behaviour was an explicit decision.

## Related issues found in the same audit, deliberately **not** fixed

Each is real and reproduced, but out of scope for these two fixes.

1. **The hls4ml `sparsity` shorthand is unreachable.**
   `frontends/hls4ml/backend.py:68` declares
   `ConfigurableAttribute('sparsity', value_type=dict, default={})`, so hls4ml's own attribute
   validation rejects a bare mode string — *"Expected <class 'dict'>, got <class 'str'>"* —
   before `frontends/hls4ml/utils.py:117-129`, which accepts either form, is ever reached. The
   string branch is dead on that path and `docs/sparse_dense_implementation.md` advertises a
   shorthand that does not work. The ONNX path handles both forms correctly. **Worth fixing
   next**: with sparsity opt-out, a directive that fails to arrive yields the opposite of what
   was asked.

2. **The hls4ml backend's default part is stale.**
   `frontends/hls4ml/backend.py:176` defaults to `part='xilinx_vek280_base_202520_1'`; Vitis
   2026.1 installs only `xilinx_vek280_base_202610_1`. Left alone because changing it retargets
   every hls4ml user. The tests already deduce the part via `tests/aie_platform.py`.

3. **Dense `int8 x int16` is declared but unreachable.**
   `MICROTILE_OPTIONS` lists `('int8','int16')` for both AIE-ML and AIE-MLV2, but
   `_SUPPORTED_INT_WIDTH_COMBOS` (`common.py:255`) is `{(8,8), (16,8), (16,16)}` and omits
   `(8,16)`, so `bitwidths_supported()` declines first. Confirmed on both parts: *"no dense
   variant matches"*. Fixing it means either enabling a never-tested path or deleting table
   rows — a judgment call, not a transcription error.

4. **The whole `'AIE'` block of `MICROTILE_OPTIONS` is dead.**
   It declares five pairings, but `bitwidths_supported()` hard-rejects any generation outside
   `('AIE-ML', 'AIE-MLV2')`, and both the dense and dynamic-matmul variants share that gate
   (`matmul.py:57`). `MICROTILE_OPTIONS` has no consumer other than `dense.py:94`.

5. **Sparse `('int16','int16')` is declared but unreachable** — by design, for now.
   `sparse_capability()` (`common.py:196`) declines non-8-bit weights, so both generations'
   `('int16','int16')` rows cannot be selected. **Kept deliberately**: unlike Bug 1 this is a
   real hardware mode (`2x8x8`, unfootnoted, on AIE-ML/XDNA1, XDNA2 *and* AIE-MLv2; named
   explicitly in UG1603's AIE-MLv2 sentence), so the row is correct and the gap is in aie4ml's
   packer. The remaining work is narrower than
   `docs/sparse_dense_implementation.md` §7 implies:

   - `sparse_tile_bytes()` already **is** `aie::sparse_vector::memory_bytes()` for any width —
     `(Elems/sparsity_ratio())*type_bits + (Elems*type_bits)/8`, per
     `/opt/Xilinx/2026.1/Vitis/aietools/include/aie_api/sparse_vector.hpp:147-159`. Its
     `element_bits != 8` guard is gratuitous. For int16 at `(2,8,8)`: 64 + 16 = 80 bytes.
   - `_compress_sparse_tile()` is the real work: the mask is one bit **per byte**, so a 16-bit
     sample spans two mask bits, and `sparse_read16b` (`me_native.h:1057`) steps a 64-bit mask
     by 8 bits per group of four, giving **32 samples per chunk** rather than
     `SPARSE_CHUNK_ELEMS = 64`.
   - Then relax the four 8-bit guards: `sparse_capability()`, `validate_config()`'s `rhs_bits`,
     `_pack_sparse()`'s `itemsize != 1` reinterpret check, and the kernel's
     `static_assert(sizeof(weight_t) == 1)` (`dense_bias_relu.cpp:16`).

6. **`int16 x int8` sparse has no simulation coverage**, on any generation. Its two tests are
   IR-only: the packing round-trip (`test_packed_rom_decodes_back_to_the_models_own_weights`,
   id `int16_acts`) and microtile selection
   (`test_int16_activations_use_the_int16_sparse_microtile`). A lossless-decode test proves the
   ROM is right; it does not prove the kernel *reads* it correctly at `microtile_m = 2`, which
   changes the L1 DMA tiling that §5.2 of the implementation notes calls out as easy to get
   wrong and hard to see. `uint8 x int8` sparse is reachable and has no test at all.

Also unchanged: the 7 pre-existing `ruff check` findings and the repo's `ruff format` drift, both
of which predate this branch.

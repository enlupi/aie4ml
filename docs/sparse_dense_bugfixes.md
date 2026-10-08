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

### Resolution (Vitis 2026.1.1): option 3, and the packer *was* wrong

This section corrects the one above. The simulator was not the problem: aie4ml was packing
AIE-MLv2 sparse weights in a layout that the AIE-MLv2 hardware models do not read. That layout is
the one the vendor documentation describes. The layout that works was found by experiment and
contradicts the documentation, so this section records what the docs say, what the hardware
models do instead, and how that was established.

#### What changed in the environment

With Vitis 2026.1.1 (`~/AMD/2026.1.1/Vitis`) both blockers described above are gone:

- **The part loads.** `make compile` for `vek385_base` succeeds (0 errors, ~4–5 min and ~3.7 GB
  peak on a tiny layer), and `aiesimulator` runs (~65 s, ~2.5 GB, `AIEMLv2sim` license).
  `probe_part.sh` is misleading on this install. It still reports `devint/versalaiedge/`,
  `me/code` and `me/clock` as absent for `xc2ve3858`, but the part lives under
  `devint/versalaiedge2/`, and AIE-MLv2 evidently does not need the `me/code` and `me/clock`
  databases.
- **x86sim now models AIE-MLv2 sparse.** The 2026.1.1 `aietools/data/aie2ps/lib/me_native.h`
  declares native FIFO pops for every `*_sparse` type (`FIFO_POP_FILL_NATIVE(v128int8_sparse, …)`
  and so on). On every sparse probe below, its output is bit-identical to aiesim. On this install,
  x86sim is therefore a valid oracle for this path, taking ~1 min against ~6 min for
  compile + aiesim. The earlier finding that x86sim cannot validate AIE-MLv2 sparse still holds
  for older installs.

#### The symptom

The layer was deliberately tiny: batch 4, 32 → 16, int8 x int8, one tile, microtile `(4,16,8)`.
Built both ways on VEK385 and run through aiesim:

- **Dense** agrees with a NumPy fixed-point reference to within 1 LSB at 2 of 64 elements. That
  residue is a rounding detail of the quick reference; MLv2 dense was already known to be
  bit-identical to AIE-ML dense.
- **Sparse** differs from dense in **58 of 64** elements (errors up to 4.25). aiesim also reports a
  hardware error event on the tile:
  ```
  [CRITICAL WARNING:151] … tile_7_3.cm.event_trace.error_check Error event asserted: sparsity_overflow (ID: 70)
  ```

`sparsity_overflow` is raised by the core, not by aie4ml. It means the hardware was handed a mask
it cannot decompress. That points at the byte layout of the weight ROM, not at the simulator.

#### What the documentation says

Both vendor sources describe the same layout. For AIE-MLv2 it is a **64-bit mask covering 512
bits (64 bytes) of uncompressed data**.

**UG1603** (local mirror
`docs/vendor/aie-ml/2025.2/ug1603/ug1603-ai-engine-ml-kernel-graph-en-us-2025.2.txt`,
~line 4540, "Sparse Matrix Multiplication"):

> The sparse data can be compressed in 256-bit chunks and stored in data memory. The first 32
> bits are mask bits. Each mask bit corresponds to one byte in the original data sequentially.
> If the byte is zero, the mask bit is 0. If the byte is non-zero, the corresponding bit is 1,
> and the byte is appended to the compressed data. After the 256 bits data is compressed, guard
> bits are added to make sure that the compressed data is 32-bit aligned.
>
> Note: For AIE-ML v2, data chunks are 512-bit wide, and the mask and alignment is on 64 bits.

**AI Engine API**, "Sparse Data Format" (`docs/vendor/aie-ml/2025.2/aie_api/group__group__memory_html.md`
~line 478). The same text is still in the 2026.1.1 install at
`aietools/include/aie_api/aie_doc.hpp:874–883`, so it was not revised for this release.

> Loading sparse data from memory will interpret the first 64 bits as a mask. If a mask bit is not
> set, then 8 bits will be initialised to zero at its corresponding position. If the bit is set,
> then the 8 bits will be loading from the incompressible data that follows the mask. Hence, masks
> describe the layout of 512 bit after decompression.
>
> Each mask must be aligned to a 32b boundary.

The page's own example builds exactly that: an 8-byte mask at `data[0..7]`, its data from
`data[8]`, then the next 8-byte mask. The API page makes no distinction between generations.

**What aie4ml emitted before the fix.** It used exactly this layout, on both generations, as
`SPARSE_CHUNK_DATA_BYTES = 64`. For one 16×8 int8 tile (128 bytes column-major, 80 compressed):

```
offset  0..7    64-bit mask   -> columns 0-3 (bit i = uncompressed byte i)
offset  8..39   32 kept bytes of columns 0-3, ascending position
offset 40..47   64-bit mask   -> columns 4-7
offset 48..79   32 kept bytes of columns 4-7
```

This is verified correct on AIE-ML (aiesim differential, `test_sparse_matches_dense_in_simulation`).

#### What the AIE-MLv2 hardware models actually read

```
offset 10c+0..1   16-bit mask   -> column c's 16 uncompressed bytes      (c = 0..7)
offset 10c+2..9   8 data slots  -> 2 fixed slots per group of 4 bytes
```

That is, one **2-byte mask per 16 uncompressed bytes**, not one 8-byte mask per 64. Within each
4-bit group of the mask, the two data slots follow the aie_api partial-decompression table
(`group__group__memory_html.md` ~line 482). For example, mask nibble `0011` puts A in the low slot
and B in the high slot, `0100` puts C in the high slot, and `1000` puts D in the high slot.

What is *unchanged* from the docs:

- one mask bit per byte, LSB-first within each mask byte;
- column-major order across the tile;
- the slot semantics above;
- the tile total: 8 × (2 + 8) = 80 bytes = `aie::sparse_vector<int8,128>::memory_bytes()`.

aie4ml always sets exactly 2 bits per nibble (`sparse_mask_along_k`), so "low slot, high slot" is
the same as "ascending position". The data bytes therefore come out in the same order as before;
**only the positions of the masks move.**

#### The discrepancy, precisely

| | UG1603 / AI Engine API | AIE-MLv2 models (x86sim and aiesim, 2026.1.1) |
| --- | --- | --- |
| Uncompressed bytes per mask | 64 (512 bits) | **16 (128 bits)** |
| Mask size | 64 bits | **16 bits** |
| Compressed chunk | 8 + 32 = 40 bytes | **2 + 8 = 10 bytes** |
| Mask alignment | 32-bit (API) / 64-bit (UG1603) | **16-bit is enough.** Masks at offsets 10, 30, 50, 70 decode correctly |
| Bytes per 16×8 int8 tile | 80 | 80 |

Two properties made this hard to see:

- **Every size and stride check passes either way.** The tile size is identical, so
  `sparse_tile_bytes()`, `SPARSE_TILE_BYTES`, the kernel's pointer strides and the L1 DMA tiling
  are all correct under both layouts. Only the positions of bytes inside a tile differ.
- **The documented layout fails silently in effect.** It compiles, runs, returns `Sim result: 0`,
  and produces plausible-looking wrong numbers. The only warning is a single `sparsity_overflow`
  line among the simulator's `CRITICAL WARNING`s.

One microtile-dependent ambiguity remains. Every AIE-MLv2 sparse microtile has exactly 16 bytes
per tile column: `(4,16,8)` and `(8,16,8)` hold 16 int8 values per column, and `(2,8,8)` and
`(4,8,8)` hold 8 int16 values. "One mask per 16 bytes" and "one mask per tile column" are
therefore indistinguishable for every shape aie4ml can emit. The code states the first
(`SPARSE_CHUNK_DATA_BYTES_BY_GENERATION['AIE-MLV2'] = 16`). A future microtile whose column is not
16 bytes would have to re-measure.

#### What the headers do say (and do not)

The installed `aie_api` headers do not document the MLv2 layout. They do show that AIE-MLv2
handles sparse B differently from AIE-ML at exactly this point, which is why the docs were not
taken as final.

- **`detail/aie2/mmul_8_8.hpp` (AIE-ML)** shuffles the loaded sparse vector before multiplying:
  `::mul_4x16_16x8(a, …, ::shuffle(b, T8_8x8), …)`. This is the "internal transpose of the
  partially decompressed data" that the `mmul` docs mention.
- **`detail/aie2p/mmul_8_8.hpp` (`__AIE_ARCH__` 21/22, which includes AIE-MLv2)** passes the
  sparse vector straight to the native transposed intrinsic: `::mul_8x16_16x8T(a.grow<128>(), …, b, …)`
  for `4x16x8`, and `::mul_8x16_16x8T(a, …, b, …)` for `8x16x8`. There is no shuffle.
- **`detail/aie2/sparse_vector_native_types.hpp`** maps the load pointer to
  `v128int8_sparse_compress` on AIE-ML (`__AIE_ARCH__ == 20`) and to `v128int8_sparse_unaligned`
  on later architectures. The load goes through a different path.
- **`aie2ps/lib/isg/me_primitives6.cpp`, `shift_combine` mode 0 (the plain FIFO load)** returns
  the raw 512 loaded bits. There is no decompression at load time.
- **`aie2ps/lib/isg/me_primitives4.cpp`, the `mac_bf` bodies** read the sparse operand register
  as a 256-bit mask (`mask_b[255:0]`) plus 1024 bits of data (`mask_b[1279:256]`), and hand the
  mask to the multiplier (`vmac_bf(…, b, sparse_mask, …)`). Decompression happens inside the
  multiplier, not in the load unit.

The function that turns memory bytes into that register, `fifo_pop_qx`, is only present
compiled, inside `libnative.a`. Its source is not shipped, so the byte layout could not be read
off the model directly. It had to be measured.

#### Method

The approach: make the hardware report the weight matrix it actually used, then fit a layout to
it. Scripts and outputs lived in the session scratchpad (`mlv2_probe.py`, `mlv2_identity.py`,
`rom_probe.py`, `h1.py`) and are not in the repository.

**1. Trust the dense path first.** MLv2 dense agrees with the NumPy reference (above), so
placement, DMA, PLIO, I/O layout and quantization are sound on this target. Any sparse difference
comes from the sparse ROM or the sparse kernel.

**2. Read the effective weights back with an identity input.** Build the sparse layer with
K = batch = 32 and N = 16. Set every fractional width to 0 (input, weight and output), so every
quantization shift is 0 and the bias rounds to 0. Then feed `X = I₃₂`. Every product is then an
exact integer and

```
y[m, n] = Σ_k I[m, k] · W_eff[k, n] = W_eff[m, n]
```

so one simulation returns **the whole effective weight matrix**, `W_eff`: the matrix the
multiplier used, as opposed to the one aie4ml meant. Weight magnitudes stay ≤ 127, so int8
output saturation never hides anything.

**3. Make every weight name its own source.** The 2:4 weights were built so each value encodes
where it came from. With `g = k // 4` the group index along K, the two kept values of group
`(g, n)` are `+(16g + n + 1)` for the first and `−(16g + n + 1)` for the second (capped at 127).
Any value in the readback then identifies its original `(k, n)`. Result, packer as it was:

- 289 of 512 outputs were wrong, with `sparsity_overflow` raised in aiesim.
- The output at `[0, 0]` was 102, i.e. group 6, from the *second* K-tile, a different 80-byte tile
  in the ROM. The fault is therefore not a permutation inside a tile. The hardware reads bytes at
  different offsets from the ones the packer assumed.
- Many outputs decoded to no weight at all, consistent with garbage masks.

**4. Rule out the A operand.** The same identity probe with the `(8,16,8)` microtile uses the
native `mul_8x16_16x8T`, with no `grow<128>` and no emulated half-accumulator. It gave
**byte-identical** output to `(4,16,8)`. Both microtiles share the same 16×8 B tile and so the
same ROM, while A is handled differently. The A side is read consistently; the fault is in how
the B stream is parsed.

**5. Confirm x86sim is a faithful oracle.** The identity probe under x86sim matched the aiesim
output exactly (0 of 512 differ). From here on each experiment was one x86sim run (~1 min)
instead of a hardware compile plus aiesim (~6 min).

**6. Feed the hardware hand-built ROMs.** After `run_pipeline()`, the probe overwrote
`packed_weights` with chosen bytes and emitted the project, still with `X = I`. The decisive
ROM ("E1") put `0x33` in every position the old layout calls a mask (bits 0,1 of every nibble,
i.e. "keep the first two of every group of 4"). Every data position got a distinct id,
1, 2, 3, … Output column 0 came back as

```
rows 0,1,4,5,8,9,12,13 = 51 51 51 51 51 51 1 2        (51 = 0x33)
```

This is decisive:
- **The *pattern* of rows** is exactly what mask bytes 0–1 (`0x33 0x33`) describe for 16 values.
- **The *values*** are ROM bytes 2..9: six mask bytes read as data, then the first two data bytes.
- So the hardware takes bytes 0–1 as the mask of column 0 and bytes 2–9 as its 8 data slots.

Column 1 then starts at byte 10. Under the model, bytes 10–11 (ids 3, 4, i.e. `0x03 0x04`) are its
mask. That gives nibble 0 = `0011` and nibble 2 = `0100`, so column 1 should be nonzero only at
rows 0, 1 and 10, with row 10 taking the *high* slot of group 2 (data slot 5 = id 10). The
readback had exactly that: rows 0, 1 = 5, 6 and row 10 = 10. This confirmed the fixed-slot
semantics, not just the mask position.

**7. Formalise and test the model against everything.** A decoder for "2-byte mask + 8 slots per
16 bytes, slots per the partial-decompression table" (`h1.py`) was run over every ROM tried. It
compared only the "well-defined" outputs, those whose mask nibble has at most two bits set. A
3-of-4 nibble has no entry in the table and is what raises `sparsity_overflow`, so the hardware's
result there is unspecified.

| Probe | Simulator | Model predicts |
| --- | --- | --- |
| E1 hand-built ROM, `(4,16,8)` | x86sim | **304 / 304** well-defined outputs |
| aie4ml's own (old-layout) ROM, positional weights, `(4,16,8)` | aiesim | **160 / 160** well-defined outputs |

The second row matters most: it is the cycle-accurate model, fed a ROM that was not designed for
the experiment.

**8. Implement, then verify end to end.** The packer change is only the chunk size (see "The fix"
below). Every probe was re-run against the new packer:

| Check | Before | After |
| --- | --- | --- |
| `(4,16,8)` sparse vs dense, aiesim, random input and fractional quantization | 58/64 differ, `sparsity_overflow` | **bit-exact**, no event |
| `(4,16,8)` identity readback, x86sim | 289/512 wrong | **exact** |
| `(8,16,8)` identity readback, aiesim | 289/512 wrong | **exact**, no event |
| `test_aie_mlv2_sparse_matches_dense_in_aiesim` (pytest, aiesim) | — | **passed** (8 min 34 s, 3.7 GB peak) |

After the change, the packer's ROM also decodes back to the model's weights under the new model, with
every nibble well-defined (`test_aie_mlv2_packs_sixteen_byte_sparse_chunks`). That test was seen
to fail with the chunk size put back to 64.

#### The fix

Files: `common.py`, `sparse.py` and `config.py` under `op_impls/families/matmul/`.

- `SPARSE_CHUNK_DATA_BYTES_BY_GENERATION = {'AIE-ML': 64, 'AIE-MLV2': 16}`. The value is resolved
  into the new `DenseConfig.sparse_chunk_bytes` at `resolve()` time, because the pack pass has no
  device, and is passed to `pack_mmul_rhs_sparse_matrix` → `_compress_sparse_tile`.
- The kernel, strides, DMA tiling and `sparse_tile_bytes()` are untouched.
- AIE-ML ROMs are **byte-identical** to before for int8 x int8, int16 x int8 and int16 x int16. This
  was checked against a `HEAD` worktree over 81 920 bytes.
- AIE-MLv2 takes 8- and 16-bit weights (`SPARSE_ELEMENT_BITS_BY_GENERATION`). int16 x int16 was
  first held back as unmeasured, then measured and enabled (next section).

#### int16 x int16 on AIE-MLv2

The first version of the fix restricted AIE-MLv2 to 8-bit weights. The int8 measurement says
nothing about 16-bit samples, and the reason recorded for holding them back was that a kept int16
sample sets two adjacent mask bits, "a pattern the partial-decompression table does not define
within a nibble". **That reason was only partly right.**
- If the two kept samples of a group of 4 are not adjacent, each nibble is `0011` or `1100`, and
  both are valid table entries.
- Only an *adjacent* kept pair gives `1111`, which the byte-level table cannot express.

So 16-bit data cannot be decoded through the byte table alone, and it evidently has its own
per-sample rule, as on AIE-ML (`sparse_read16b`). That rule had to be measured, not inferred.

**Method.** The same identity readback, at int16 x int16:
- K = batch = 32, N = 16, all fractional widths 0, `X = I₃₂`, so `y = W_eff` exactly.
- 16-bit precision allows every weight to be **globally unique**. The kept value at `(k, n)` is
  `±(16k + n + 1)`, with magnitudes up to 512. Every output therefore names its source directly;
  unlike the int8 probe, no group-level decoding was needed.
- The weights cover **all six** kept-pair patterns of a group of 4, `(0,1) (0,2) (0,3) (1,2) (1,3)
  (2,3)`, 19–25 times each, including the three adjacent pairs whose nibble is `1111`.
- The MLv2 `{8}` restriction was lifted temporarily, and the packer run with no other change:
  16-byte chunks, two mask bits per int16 sample, the kept samples in ascending order.

**Results:**

| Probe | Result |
| --- | --- |
| `(2,8,8)` identity readback, x86sim | **exact**, 512/512 |
| `(4,8,8)` identity readback, x86sim | **exact**, 512/512 |
| Negative control: `(2,8,8)` packed with the AIE-ML 64-byte chunk, x86sim | **267/512 wrong** |
| `(2,8,8)` identity readback, aiesim | **exact**, 512/512, no `sparsity_overflow` |
| `(4,8,8)` identity readback, aiesim | **exact**, 512/512, no `sparsity_overflow` |
| `test_aie_mlv2_sparse_matches_dense_in_aiesim[int16xint16]`: MLv2 sparse in aiesim vs **AIE-ML** dense in x86sim, random input, fractional quantization | **passed**, bit-exact (4 min 45 s) |

The dense reference for the last row is AIE-ML, not AIE-MLv2, because AIE-MLv2 dense int16 x int16
does not compile (item 7 under "Related issues"). The integer arithmetic is exact on both
generations, so the outputs must still agree bit for bit.

The negative control is what makes the exact readbacks meaningful: the probe does detect a wrong
layout at this width. The aiesim ROMs were also decoded straight from the emitted
`src/weights/weights_d0_aie.h`, and both equal the weight matrix under the 16-byte layout. This
rules out the runs having picked up a temporary edit made while they were in progress.

So **no code beyond the int8 fix was needed**. The 16-byte chunk, with the AIE-ML rule of two
mask bits per 16-bit sample, is the AIE-MLv2 int16 layout. The change is
`SPARSE_ELEMENT_BITS_BY_GENERATION['AIE-MLV2'] = {8, 16}`, plus tests:
- `test_aie_mlv2_packs_sixteen_byte_sparse_chunks` is parametrized over all four MLv2 cases, and
  was seen to fail in the decode (not just the config check) for all four with the chunk put
  back to 64;
- `test_aie_mlv2_int16_weights_select_the_int16_sparse_microtile`;
- `test_aie_mlv2_sparse_matches_dense_in_aiesim` is parametrized over int8 (MLv2 dense reference)
  and int16 (AIE-ML dense reference).

#### Limits of this result

- **Both oracles are AMD models.** x86sim and aiesim are different simulators, but both are
  vendor-authored and could share a modelling error that silicon does not. The documentation
  and the models disagree, and only real hardware settles which one describes the chip.
  **Not run on silicon.**
- **Shapes.** The simulated layers are single-tile. Cascades, `cas_num > 1` and the `outer`
  contract use the same per-tile packing code, which is covered on AIE-ML, but no AIE-MLv2
  simulation exercised them.
- **Column ≠ 16 bytes.** See the ambiguity above; it is irrelevant for the current microtile
  table.
- **Worth reporting to AMD.** The docs describe a layout their own simulators reject. The identity
  readback above is a self-contained reproducer.

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

5. **Sparse `('int16','int16')` is declared but unreachable** — *resolved since*, on both
   generations. As originally recorded, `sparse_capability()` declined non-8-bit weights, although
   `2x8x8` is a real hardware mode on AIE-ML, XDNA2 and AIE-MLv2. The packer work listed here
   has since been done: two mask bits per 16-bit sample, 32 samples per AIE-ML chunk, and the
   8-bit guards relaxed to `SPARSE_ELEMENT_BITS = {8, 16}`.
   - **AIE-ML:** int16 x int16 resolves to `(2,8,8)` and is verified against dense in x86sim and
     aiesim (`test_sparse_matches_dense_in_simulation`).
   - **AIE-MLv2:** enabled after the separate measurement in "int16 x int16 on AIE-MLv2" above.

6. **`int16 x int8` sparse has no simulation coverage** — *partly resolved since*.
   - **AIE-ML:** it is now in `SPARSE_SIM_CASES` and is verified against dense in x86sim, at
     `microtile_m = 2` (`test_sparse_matches_dense_in_simulation[x86-int16xint8]`). It is not run
     in aiesim, because it shares its weight width and decompression path with int8 x int8.
   - **AIE-MLv2:** does not apply; the pairing does not exist there (Bug 1).
   - `uint8 x int8` is still not simulated separately. It folds onto the int8 x int8 row via
     `legality_format`, so it resolves to the same microtile and kernel.

7. **AIE-MLv2 *dense* int16 x int16 does not compile.** Full write-up, root cause and options in
   `docs/issue_int16_dense_accumulator.md`. Found while building a dense reference
   for the MLv2 int16 sparse test. This is the dense path; nothing on the sparse side causes it.
   - `MICROTILE_OPTIONS['AIE-MLV2'][('int16','int16')]` is `[(8, 2, 8)]`, and aie4ml infers
     `acc64` for int16 x int16.
   - On AIE-MLv2, `detail/aie2p/mmul_16_16.hpp` defines `mmul_16_16<8, 2, 8, …>` (and `<4, 2, 8>`)
     **only for a 32-bit accumulator**. The acc64-capable dense shapes are `<2, 4, 8>` and
     `<4, 4, 8>`.
   - So `v++` fails with `implicit instantiation of undefined template
     'aie::detail::mmul_16_16<8, 2, 8, short, short, 64>'`.

   **User-visible effect on VEK385:** an int16 x int16 Dense layer compiles only if its weights
   are already 2:4, because it then takes the sparse path. A layer with ordinary weights, or with
   `sparsity: {'mode': 'dense'}`, fails in `v++`, as does any other fall-back to dense. That fall-
   back is exactly what the "sparsity never breaks a build" guarantee relies on, so the guarantee
   does not hold for this pairing on AIE-MLv2 until the dense table is fixed.

   **Likely fix, not done:** list `(4, 4, 8)` / `(2, 4, 8)` for AIE-MLv2 int16 x int16, verified in
   aiesim like any dense change. The other AIE-MLv2 16-bit dense rows (`int16 x int8`,
   `int8 x int16`) were not checked for the same accumulator mismatch.

Also unchanged: the 7 pre-existing `ruff check` findings and the repo's `ruff format` drift, both
of which predate this branch.

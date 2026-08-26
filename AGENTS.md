# AGENTS.md — `aie4ml`

Guidance for AI coding agents (Claude Code and similar) working in this repository.
Read this file **before** editing code, and re-read the relevant section before touching an
unfamiliar subsystem.

---

## 1. What this project is

`aie4ml` is an **end-to-end compiler** that lowers quantized neural-network graphs into
**AMD AI Engine (AIE)** firmware: ADF graphs, C++ kernels, packed weights, a build `Makefile`,
and (for hardware targets) PL data movers and a host application. The emitted project is a
standalone Vitis AIE project that can be compiled and simulated with `v++`, `x86simulator`
and `aiesimulator`.

- **Hardware targets:** AIE-ML (VEK280 / `xcve2802`) and AIE-MLv2 (VEK385).
- **Frontends:** ONNX (recommended, operator-level) and `hls4ml` (optional, MLP-oriented).
- **Toolchain:** AMD Vitis 2025.2 + a valid AIE tools license. Python ≥ 3.10.
- **Package layout:** `src/` layout, `setuptools_scm` versioning, entry points for
  `hls4ml.backends` (`AIE = aie4ml.plugin:register`) and the `aie4ml-report` CLI.

Authoritative capability list: **`docs/support.md`**. If a change alters what is supported,
update that file in the same commit.

---

## 2. AI Engine documentation — read it, don't guess

A local mirror of the AMD AI Engine documentation lives at:

```
docs/vendor/aie-ml/2025.2/
├── ug1603/     # AI Engine-ML Kernel and Graph Programming Guide (UG1603)
│               #   - the original PDF
│               #   - an extracted plain-text (.txt) version — grep this, it is much faster
└── aie_api/    # AI Engine API User Guide, mirrored as Markdown, split by topic
```

**Use these directories as the source of truth** for anything touching:

- `aie::` API types and functions (`aie_api/`)
- ADF graph constructs, connections, buffer/DMA semantics, tiling parameters,
  constraints, RTP, cascade, memory tiles, PLIO (`ug1603/`)

### 2.1 HARD RULE: never invent an API

> **Do not write any `aie::*`, `adf::*`, or ADF/kernel intrinsic call unless you have located it
> in `aie_api/` or `ug1603/` in this session and confirmed its exact name, template parameters,
> argument order, and return type.**

Concretely:

1. **Grep before writing.** Search `aie_api/` (Markdown) and `ug1603/*.txt` for the symbol.
   Only then use it.
2. **Do not extrapolate by analogy.** The existence of `aie::max` does not imply `aie::argmax`;
   the existence of `aie::shuffle_down` does not imply `aie::shuffle_up_fill`. Overload sets are
   narrower than they look, and many operations are restricted by element type, vector width, or
   AIE generation.
3. **Vector widths and types are constrained.** A given `aie::vector<T, N>` is only legal for
   certain `(T, N)` pairs, and available `aie::mmul<M, K, N, TA, TB, TAcc>` shapes differ between
   AIE-ML and AIE-MLv2. Check `MICROTILE_OPTIONS` in
   `src/aie4ml/op_impls/families/matmul/common.py` alongside the docs.
4. **If you cannot find it, say so.** State plainly that the call could not be verified in the
   local docs and propose an alternative built from confirmed primitives. Never emit a plausible-
   looking placeholder — it will compile-fail deep inside `aiecompiler` with an opaque message,
   or worse, silently produce wrong numerics.
5. **Prefer copying an existing, working pattern** from `src/aie4ml/templates/nnet_utils/` over
   inventing a new one. Those kernels are known to compile and to be bit-exact against reference.
6. The same discipline applies to **`v++` / `aiecompiler` flags**, `aie.cfg` keys, and
   `.cfg`/`system.cfg` connectivity syntax: verify against UG1603 or the existing templates.

---

## 3. Repository layout

```
src/aie4ml/
├── model.py              # AIEModel: write() / build() / compile() / predict() / report()
├── pipeline.py           # HLS4ML_FLOW_SPEC + DEFAULT_PIPELINE (the pass order)
├── writer.py             # AIEProjectEmitter: renders the whole output project
├── report.py             # Post-build metric collection (+ `aie4ml-report` CLI)
├── simulation.py         # I/O layout, quantize/dequantize, stimulus files, sim invocation
├── system_plan.py        # PL data movers, host wiring, PLIO connectivity plan
├── device_catalog.py     # loads aie_devices.json
├── aie_devices.json      # device catalog (see §7)
├── aie_types.py          # AIEDataType, QuantIntent, FloatIntent, rounding/saturation modes
├── serialization.py      # dump/load of aie_pipeline.json
├── plugin.py             # hls4ml backend entry point
├── ir/
│   ├── context.py        # AIEBackendContext, DeviceSpec, ProjectConfig, BackendPolicies, traits
│   ├── graph.py          # LogicalIR / ExecutionIR / PhysicalIR, OpNode, TensorVar, contracts
│   └── __init__.py       # public IR exports
├── passes/
│   ├── base.py           # AIEPass base + run_aie_passes
│   ├── placement.py      # PlaceKernels: geometric, graph-aware kernel placement
│   ├── pack.py           # PackKernelArtifacts
│   ├── transport/        # fanout, classify, memtile legalization, materialization
│   └── utils.py          # sanitize_identifier, misc helpers
├── op_impls/
│   ├── base.py           # OpImplVariant, OpImplFootprint
│   ├── registry.py       # register_variant, OpImplRegistry, plevel-ordered candidates
│   ├── family_registry.py# FamilyResolver, @family_resolver
│   ├── common_types.py   # PortMap, PortBinding
│   ├── utils/            # tiling, tensor_view, io, precision helpers
│   └── families/         # matmul (dense+matmul), elementwise, layernorm, softmax
├── frontends/
│   ├── common.py         # register_default_traits
│   ├── onnx/             # importer, context, registry, handlers/, quantize, utils
│   └── hls4ml/           # backend, lower, writer adapters
└── templates/
    ├── firmware/         # Jinja: app.cpp, top_graph.h, parameters.h, graph_plan.h,
    │   ├── variants/     #        aie.cfg, Makefile, system.cfg
    │   ├── pl/           #        benchmark/ and deployment/ HLS data movers
    │   └── host/         #        aarch64 host application
    └── nnet_utils/       # the actual AIE C++ kernels (dense_bias_relu, matmul,
                          # layer_norm, elementwise_add, softmax, ...)
docs/support.md           # operator + feature support matrix (keep in sync!)
tests/                    # pytest; markers: aie_ir, requires_vitis
scripts/check_onnx_bit_exact.py   # ONNX Runtime vs AIE x86-sim bit-exactness checker
tutorials/tutorial_1.ipynb, tutorial_2.ipynb
```

---

## 4. Architecture: the compilation flow

```
ONNX / Keras(hls4ml)
        │  frontend handlers
        ▼
   LogicalIR         OpNode + TensorVar; semantic ops, traits, per-node directives
        │  DEFAULT_PIPELINE passes
        ▼
  ExecutionIR        ExecutionEntry per node: chosen OpImplVariant, resolved config,
        │            PortMap, io_views, io_route, packed artifacts
        ▼
  PhysicalIR         placements {col,row} + plan (buffers, transport units, memtiles)
        │  AIEProjectEmitter
        ▼
  Emitted project    src/*.h, app.cpp, aie.cfg, Makefile, weights, (PL + host)
        │  make
        ▼
  v++ / x86simulator / aiesimulator  →  report()
```

### 4.1 The pass pipeline

`src/aie4ml/pipeline.py` defines the canonical order. **Order matters** — several passes assume
invariants established upstream:

| # | Pass name | Class | Purpose |
|---|---|---|---|
| 1 | `force_float` | `ForceFloatMode` | Optional whole-graph float compute override |
| 2 | `fold_apply_alpha` | `FoldApplyAlpha` | Fold hls4ml alpha scaling |
| 3 | `fold_bias` | `FoldBias` | Fold a trailing Add into Dense bias |
| 4 | `fold_scale` | `FoldScale` | Fold power-of-two output scale into shifts |
| 5 | `fuse` | `FuseActivationCasts` | Fuse ReLU/casts into the producer |
| 6 | `fold_views` | `FoldViewOps` | Fold Split/Slice/Concat/Permute into views |
| 7 | `resolve` | `Resolve` | Variant selection + config resolution → ExecutionIR |
| 8 | `pack` | `PackKernelArtifacts` | Tile/pack weights for `aie::mmul` layouts |
| 9 | `memory_collect` | `CollectMemoryEntries` | Build transport `EdgeEntry` list |
| 10 | `fanout_legalize` | `LegalizeFanoutEntries` | Split fanout into single-consumer legs |
| 11 | `transport_classify` | `ClassifyTransportEntries` | Decide `direct` vs `memtile` per leg |
| 12 | `placement` | `PlaceKernels` | Geometric placement on the AIE array |
| 13 | `memtile_legalize` | `LegalizeMemtilePortLimits` | Shard legs to fit memtile port limits |
| 14 | `memory_plan` | `MaterializeMemoryPlan` | Emit concrete buffers + BD descriptors |
| 15 | `compact_batch` | `CompactBufferRank` | Rank compaction of execution buffers |

Passes subclass `AIEPass`, set `self.name`, and implement
`transform(model_or_ctx) -> bool` (returns `True` if the IR changed). Always obtain the context
via `get_backend_context(model_or_ctx)` — the same pass runs both standalone and wrapped as an
hls4ml `ModelOptimizerPass`.

### 4.2 Op implementation variants

Each operator family has:

- a **`FamilyResolver`** (`@family_resolver('dense')`) that structurally validates the node and
  dispatches to variants;
- one or more **`OpImplVariant`** subclasses registered with `@register_variant`, ordered by
  descending `plevel`. The first whose `matches(node, device)` returns `True` wins.

A variant owns its whole lifecycle: `matches` → `resolve` → `validate_config` →
`build_template_params` / `build_ports` / `footprint` / `pack` / `get_artifacts`, plus the
staging descriptors (`describe_input_staging`, `describe_output_staging`,
`output_staging_contract`, `output_port_count`).

`@register_variant` **enforces** that `param_template` has a matching
`templates/firmware/variants/<name>/parameters.h.jinja`. Create the template first, then register.

Registered families: `matmul` (dense + dynamic matmul, inner/outer contracts),
`elementwise` (add), `layernorm` (linear + tiled), `softmax` (exp + HCCS surrogate,
each in linear + tiled layouts).

### 4.3 Core concepts

- **Parallelism contract** (`ParallelismConfig`): `cas_num` × `cas_length`, plus
  `contract ∈ {'inner', 'outer'}`.
  - `cas_length` splits the **reduction** axis (K) across a cascade chain.
  - `cas_num` splits either the **feature** axis N (`inner`, default) or the **row** axis M
    (`outer`, which keeps each tile's output whole so a consumer can read it directly).
- **Microtiling**: the intrinsic `aie::mmul<M,K,N,...>` shape. Legal shapes are
  generation- and dtype-dependent (`MICROTILE_OPTIONS`).
- **Transport**: every tensor edge is realized as a **direct** AIE connection or **one**
  memory-tile stage. Multi-stage relay transport is *not* implemented and fails explicitly.
- **Views**: Split/Slice/Concat/Permute are folded into `io_view` traits — they instantiate
  **no kernel**. Permute handles only the last two axes. Chained folded views are unsupported.
- **Precision**: `QuantIntent` / `FloatIntent` → `AIEDataType` (`format`, `frac`, `rounding`,
  `saturation`). Per-tensor static quantization only; per-channel activation quantization is not
  supported. Accumulator tag is inferred (`acc32` / `acc48` / `acc64` / `accfloat`).
- **Placement**: rectangular footprints with port faces and keepouts, solved by a bounded
  branch-and-bound DFS with a disjoint-fanout fast path. Objective: `|Δcol| + lam·|Δrow| + mu·Σrow`.

---

## 5. Emitted project & build targets

`AIEProjectEmitter.emit()` writes into `output_dir`:

```
aie_pipeline.json      # serialized IR + physical plan (report() and reload depend on this)
aie.cfg                # v++ AIE config
Makefile
app.cpp                # ADF top-level: PLIO create + graph instantiation
src/graph_plan.h, src/parameters.h, src/top_graph.h
src/kernels/           # copy of templates/nnet_utils + any custom_sources
src/weights/           # generated artifact headers
data/                  # simulator stimulus / golden files
pl/, host/             # only when target == 'hardware'
```

Makefile targets (`Makefile.jinja`): `compile` (`v++ --mode aie --target=hw`),
`x86com` / `x86sim`, `aiesim`, `profile` (adds `--profile`, needed for per-kernel cycles),
`trace`, `analyze`, and for hardware: `kernels`, `xsa_hw`, `xsa_hw_emu`, `host`,
`package_hw`, `package_hw_emu`.

`AIEModel` API:

```python
model.run_pipeline()   # passes only
model.write()          # run_pipeline + emit
model.build(target)    # make <target>  (default 'all')
model.compile()        # make x86com
model.predict(X, simulator='x86'|'aie', quantize_in=True, dequantize_out=True)
model.report()         # or aie4ml.report.report(model_or_project_dir)
```

Frontend entry points:

```python
from aie4ml.frontends.onnx import from_onnx, lower_onnx_model
# hls4ml: hls4ml.converters.convert_from_keras_model(..., backend='aie')
```

---

## 6. Directives and configuration

Per-node/per-layer overrides (ONNX `layer_directives`, hls4ml `cfg['LayerName'][name]`):

- `parallelism`: `{'cas_num': int, 'cas_length': int, 'contract': 'inner'|'outer'}`
- `microtiling`: `{'microtile_m', 'microtile_k', 'microtile_n'}`
- `placement`: `{'col': int, 'row': int}`
- `io_route`: `{'inputs': {tensor: 'auto'|'direct'|'memtile'|'plio'}, 'outputs': {...}}`
- `layout`: `'linear'` | `'tiled'` (LayerNorm, Softmax)
- `approximation`: `'exp'` (default) | `'hccs'` (Softmax; HCCS additionally requires
  `hccs: {'B', 'S', 'Dmax'}` and a QAT-calibrated model)

`AIEConfig` keys: `Part`, `Device`, `Generation`, `Columns`, `Rows`, `ColumnStart`, `RowStart`,
`PLIOWidthBits`, `PLClockFreqMHz`, `BatchSize`, `Iterations`, `Target` (`'aie'`/`'hardware'`),
`PLMemory`, `EnablePLTiming`, `PLDataMoverMode` (`'benchmark'`/`'external_stream'`),
`Memory.BankMemBytes`, `MaxMemTileInPorts`, `MaxMemTileOutPorts`, optional `ComputeDtype`.

---

## 7. Device catalog

`src/aie4ml/aie_devices.json` — entries keyed by platform name:

| Key | Device | Generation | Cols × Rows | ColStart/RowStart |
|---|---|---|---|---|
| `xilinx_vek280_base_202520_1` | Vek280 | `AIE-ML` | 38 × 8 | 7 / 0 |
| `xilinx_vek280_base_202510_1` | Vek280 | `AIE-ML` | 38 × 8 | 7 / 0 |
| `vek385_base` | Vek385 | `AIE-MLV2` | 36 × 4 | 7 / 0 |

Generation strings are compared **exactly** as `'AIE-ML'` / `'AIE-MLV2'` (uppercase V). Adding a
device means adding a catalog entry with all required keys (`DeviceName`, `Generation`, `Columns`,
`Rows`, `ColumnStart`, `RowStart`, `PLIOWidthBits`, `PLClockFreqMHz`, `Memory.BankMemBytes`,
`MaxMemTileInPorts`, `MaxMemTileOutPorts`) — missing keys raise at config time.

> **Environment note:** the VEK280 part `xcve2802` must actually be installed in the Vitis/Vivado
> device set. Check `/opt/Xilinx/<version>/Vivado/installed_devices.txt` before assuming
> `aiecompiler` failures are a code bug — a missing device family produces an opaque internal
> data exception.

---

## 8. Development workflow

```bash
pip install -e .              # editable install
pip install onnx onnxruntime  # ONNX frontend + bit-exactness checker
pip install hls4ml            # only for the hls4ml path

ruff check . && ruff format .  # line-length 120, single quotes, py310 target
pytest -m "aie_ir and not requires_vitis"   # no Vitis needed: conversion + lowering
pytest -m "requires_vitis"                  # needs XILINX_VITIS in the environment

python scripts/check_onnx_bit_exact.py model.onnx --part xilinx_vek280_base_202520_1
```

Test markers: `aie_ir` (exercises the backend IR), `requires_vitis` (skipped unless
`XILINX_VITIS` is set).

---

## 9. Conventions

- Apache-2.0 SPDX header on new Python files:
  `# Copyright 2025 D. Danopoulos, aie4ml` / `# SPDX-License-Identifier: Apache-2.0`
- `from __future__ import annotations` at the top of modules with type hints.
- Ruff: 120 cols, 4-space indent, **single quotes**, rules `E, F, I, W`.
- Frozen dataclasses for value objects (configs, contracts, descriptors); `__post_init__`
  validation is the norm — keep it.
- **Fail loudly and specifically.** Errors are prefixed with the node name:
  `raise ValueError(f'{node.name}: <what went wrong and what to do instead>')`. Never silently
  degrade or fall back to a different numeric behaviour.
- Unsupported cases raise `NotImplementedError` with a note on what would be needed — do not
  emit approximate code paths for them.

---

## 10. Checklist before finishing a change

- [ ] Every `aie::*` / ADF construct used was **verified** against `aie_api/` or `ug1603/`.
- [ ] New variant? `param_template` directory + `parameters.h.jinja` exists and is registered.
- [ ] New pass? Inserted at the correct position in `HLS4ML_FLOW_SPEC`, with its assumed
      upstream invariants documented in the class docstring.
- [ ] Kernel change? Re-checked bit-exactness (`scripts/check_onnx_bit_exact.py` or the
      `test_*_sim` tests) — numerics changes must be intentional and called out.
- [ ] Support surface changed? `docs/support.md` updated.
- [ ] `ruff check` and `ruff format` clean; `pytest -m "aie_ir and not requires_vitis"` passes.
- [ ] Anything you could not verify is stated explicitly in your summary rather than guessed.

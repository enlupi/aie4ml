# AIE-ML Hardware Sparse (2:4) Dense — Implementation Notes

Branch `sparse_dense_support`. This document lists every change made to add AIE-ML hardware
structured-sparse matrix multiplication to the `matmul` family, why each one was made, and which
source established the behaviour it relies on.

---

## 1. What it does

AIE-ML's hardware sparse `aie::mmul` path is **on by default, and driven by the weights**. aie4ml
inspects every Dense layer's trained weight matrix; a layer that is *already* 2:4-structured along
the reduction axis is compiled sparse, a layer that is not is compiled dense. The directive exists
to *opt out*:

```python
# ONNX frontend -- force the ordinary dense path on a layer that would otherwise go sparse
layer_directives = {'d0': {'sparsity': {'mode': 'dense'}}}   # or the shorthand 'sparsity': 'dense'

# hls4ml frontend
cfg['LayerName']['d0']['sparsity'] = {'mode': 'dense'}
```

Three modes:

| Mode | Meaning |
| --- | --- |
| `auto` | **Default.** Sparse where the layer's weights already carry the 2:4 pattern, dense otherwise. Never fails a build over sparsity. |
| `dense` | Opt out. Force the dense path even on a layer whose weights qualify. Checked before anything is measured, so an opted-out layer costs nothing. |
| `2of4` | Opt in explicitly. Same as `auto` when the weights qualify, but a layer that cannot take the path is a hard **error** rather than a silent fallback, so a deliberate request cannot be quietly downgraded. |

The division of authority matters: **the mode can only veto or insist, never conjure.** What
decides whether the sparse path is on the table is the weight pattern; the directive cannot
produce one. The one place the mode is consulted *first* is the explicit opt-out, purely as a
short-circuit — the outcome no longer depends on the weights, so there is no reason to spend
~0.5 ms/layer measuring them (`test_opting_out_skips_the_weight_inspection_entirely`). A caller
who opts out is asserting they know what their model looks like.

For a qualifying layer the weight matrix is compressed into the hardware format at build time and
consumed by `aie::sparse_vector_input_buffer_stream` inside the kernel. The weight ROM shrinks to
**0.625×** dense (4 kept bytes + 1 mask byte per 8 weights), and the reduction runs on a wider
microtile.

### Why there is no pruning here

**aie4ml never prunes, and deliberately owns no code that could.** Deciding which weights to zero
is a model-accuracy decision made with the loss function in hand — that is the training
pipeline's job, not a compiler's. aie4ml only *detects* a pattern that is already present.

That is what makes the default defensible. Because the weight matrix is used exactly as handed
over, taking the sparse path is **numerically free**: the compressed ROM decodes back to the same
integers the dense ROM would have held (`test_packed_rom_decodes_back_to_the_models_own_weights`),
and there is no accuracy cost for a model to opt out of. The only thing the directive changes is
storage format and kernel path.

Two things about the pattern the hardware needs, which a training recipe has to match:

- The group is **4 consecutive input features, per output column** — the compressed layout is
  column-major, so pruning must zero along the *input-feature* axis in groups of four, one output
  column at a time, **not** whole output columns.
- At most **2 nonzeros per group** (UG1603: "every four consecutive bytes require at least two
  zeros"). A group with 3 nonzeros disqualifies the layer, however sparse the layer is overall.

A layer that is genuinely sparse but in the wrong layout is the one case where silence would be
unhelpful: it was trained for a saving the hardware cannot take. So when a layer has ≥50% zero
weights (`SUBSTANTIAL_SPARSITY_FRACTION`) and still fails the structural check,
`report_unusable_sparsity` logs one warning naming the layer, the zero fraction, the worst group,
and the fix. An ordinary dense layer says nothing.

### Three properties that make "automatic" defensible

1. **It changes no numerics.** Detection, not transformation. See above.
2. **It cannot break a build.** `sparse_eligibility` is a total predicate, and anything it fails
   to predict is caught by a fallback: a variant selected automatically that then fails to
   resolve hands the node to the next candidate (§4.3). Verified by
   `test_speculative_resolve_failure_falls_back_to_dense`, and the opt-out path's x86sim output
   is byte-identical to the pre-sparse committed kernel.
3. **It cannot silently ignore you.** An explicit `2of4` is never downgraded — not by a weight
   pattern it cannot use, not by ineligibility, not by a resolve failure.

---

## 2. Sources

Everything below is grounded in one of these. Nothing about the AIE API was inferred or invented —
`AGENTS.md` forbids it, and the compressed-format details in particular are not guessable.

Paths below written `$AIEDOCS` are the local AIE documentation tree at
`/eos/home-e/enlupi/workspace/repo/docs/aie-ml/2025.2` (the installed version is 2025.2, not the
2025.6 named in the original request). They are outside this repository.

| Source | What it established |
| --- | --- |
| `$AIEDOCS/ug1603/ug1603-ai-engine-ml-kernel-graph-en-us-2025.2.txt` §"Sparse Matrix Multiplication" (line 4516ff) | The sparsity requirement — "for data types that are smaller or equal to 8 bits, to be considered sparse, every four consecutive bytes require at least two zeros" (4526) — and the ordering rule: "the sparse matrix is compressed or decompressed column-wise (column by column)" (4604). Also the partial-decompression table, which only defines outcomes for mask patterns of 0, 1 or 2 set bits. |
| `$AIEDOCS/aie_api/group__group__mmul_html.md` | The "Supported Sparse Matrix Multiplication Modes" table (which `(M,K,N)` shapes have a sparse overload, per generation and dtype pair, and which are footnoted as emulated or needing extra data manipulation); the sparse `aie::mmul::mul/mac(VecA, SparseVectorOrOp)` signatures; the requirement that "sparse matrix multiplications require that the sparse data be stored in column major layout" (511); and the `gemm_int8xint8_sparse` worked example. |
| the same file's `gemm_int8xint8_sparse` body | The `tsB >> Ybuff0 >> Ybuff1` idiom — **one** stream popped twice, not two streams. This turned out to be load-bearing (§5.3). |
| `$AIEDOCS/aie_api/group__group__memory_html.md` | `aie::sparse_vector` and `aie::sparse_vector_input_buffer_stream` semantics, and `memory_bytes()` — the basis for `sparse_tile_bytes()`. |
| `/opt/Xilinx/2026.1/Vitis/aietools/data/aie_ml/lib/me_native.h` | The **installed** intrinsic model: `mask_length_sparse = 64` (line 761) at one bit per byte, `sparse_read`/`decompress_sparse`/`sparse_fill`/`sparse_pop`, and `fifo_buffer_stream`'s uninitialized `idx_`. Used in preference to UG1603's prose, which describes 32-bit masks over 256-bit chunks for AIE-ML — the installed model is what actually executes. |
| `sparse_matmul_design.html` | The intended shape of the integration, and the pre-identified hazards — including the registry collision written up there as **E1**, which is real and is fixed in §4.2. |
| `AGENTS.md` | Variant lifecycle, `plevel` dispatch, the `@register_variant` template requirement, the directive surface, and the landing checklist (`docs/support.md` in the same commit, `ruff`, `pytest`). |

The dense-vs-sparse microtile distinction is worth stating explicitly because it drove a lot of the
design: on AIE-ML, dense int8×int8 `(4,16,8)` carries footnotes *a* (emulated via multiple intrinsic
calls) and *b* (requires additional data manipulation), while the **sparse** table lists `(4,16,8)`
with no footnote. Sparse-capable shapes are a distinct set, not a filtered view of the dense ones.

---

## 3. Prototype first, then integration

The work was done in two passes, because a wrong compressed byte layout **compiles and simulates
cleanly while silently producing wrong numbers**. There is no way to get it right by reading alone.

The first pass modified only the already-emitted `tutorials/proj_aie` project — no aie4ml changes —
until it was bit-exact against a NumPy model of the whole pipeline in *both* `x86sim` and
`aiesim`. Candidate orderings (row-major; column-major; each split into two K-halves) were each
built and simulated; only plain column-major over the whole tile is correct.

Only then were the kernels promoted into aie4ml. The aie4ml packer's output was checked
byte-identical against that proven-good ROM before anything else was trusted.

Measured on the prototype (`qfc1`, int8×int8, cycle-accurate `aiesim`):

| Configuration | Cycles/call | Weight ROM per chain |
| --- | --- | --- |
| K=8 dense (original) | 634 | 16384 B |
| K=16 dense | 614 | 16384 B |
| **K=16 sparse** | **501** | **10240 B** (−37.5%) |

≈21% fewer cycles than the original. These are prototype numbers; the compiler path was verified
for *correctness* in both simulators but cycles were not re-measured through it.

---

## 4. Changes to `src/aie4ml`

### 4.1 New: the sparse variants

**`op_impls/families/matmul/sparse.py`** (new, 320 lines)

`_DenseSparseVariantBase(_DenseVariantBase)` plus two registered variants:

| `variant_id` | Contract |
| --- | --- |
| `dense.sparse.b.r.v1` | `inner` (output features partitioned across chains) |
| `dense.sparse.b.r.row.v1` | `outer` (rows partitioned across chains) |

Reasoning, point by point:

- **Subclasses `_DenseVariantBase`** rather than reimplementing. The compressed ROM reaches the
  kernel through the *identical* `parameter`-port mechanism the dense ROM already uses, so ports,
  graph, footprint and staging descriptors are all unchanged. Only packing, the microtile table and
  three `parameters.h` constants differ. Duplicating the rest would have created two things to keep
  in sync for no benefit.

- **`plevel = 20`**, above the dense variants' `10`, so `FamilyResolver`'s descending-plevel scan
  tries these first. Under the opt-out default *both* variants match an eligible node; sparse wins
  on plevel and dense remains as the fallback candidate.

- **`matches()` short-circuits the opt-out, then asks the weights.** `'dense'` returns
  immediately, before `sparse_eligibility()` walks the matrix. Otherwise the weight pattern
  decides: if it qualifies the node is claimed; if it does not, an explicit `'2of4'` still claims
  it — so `resolve()` can fail with a *specific* reason rather than falling through to dense and
  silently ignoring the directive — and `'auto'` declines, reporting an unusable sparsity pattern
  if there was one.

- **`selection_is_speculative()` returns True under `'auto'`.** This is what makes the default
  safe: nobody asked for sparsity, so a resolve failure must not break the build.

- **`_reject_unsupported()` and `validate_config()`** reject, loudly: a non-AIE-ML generation, float
  precisions, unsupported bitwidth pairings, non-constant weights, non-8-bit weights, weights that
  are not already 2:4, a microtile with no sparse overload, `microtile_k` not a multiple of 4 or not
  dividing the K slice, and an odd number of N-tiles per chain. Each carries a message naming the
  constraint and what to do. Weights that are not 2:4 raise `ValueError`, not
  `NotImplementedError`: the request contradicts the model rather than hitting a compiler gap, and
  only retraining would change that.

- **`_pack_sparse()`** mirrors the contract handling in the dense variants' own `pack()`: `inner`
  asks the packer for `cas_num` chains, since chain *c* owns weight columns
  `[c*N_slice, (c+1)*N_slice)`; `outer` asks for one chain and replicates it, because there
  `N_slice` is the whole of N. Factored onto the shared base since only that one argument differs.

  One subtlety: the compressed stream is raw bytes, but it is emitted as an array of the weight's
  own storage type and passed through the same `weight_t (&)[WTS_ELEMS]` parameter. So it is
  **reinterpreted** (`packed_W.view(rom_dtype)`), not converted — emitting a byte value of 150 into
  an `int8_t` array is a compile error (*"constant expression evaluates to 150 which cannot be
  narrowed"*). Guarded by a 1-byte-itemsize check.

**`templates/firmware/variants/dense_bias_relu_sparse/parameters.h.jinja`** (new)

Its own variant template, as `@register_variant` requires. Identical to the dense one except
`USE_SPARSE = true`, `SPARSE_TILE_BYTES = {{P.sparse_tile_bytes}}`, and the sparse-capable microtile.

### 4.2 Fixed: the microtile-table collision (design doc's E1)

`_supported_microtile_options` now asks **the variant doing the resolving** for its own table:

```python
def _supported_microtile_options(variant, op_type, generation, lhs_dtype, rhs_dtype):
    query = (legality_format(lhs_dtype.format), legality_format(rhs_dtype.format))
    if variant is not None:
        return list(variant.microtiling_options(generation, query))
    return get_op_impl_registry().supported_microtilings(op_type, generation, query)
```

and `_resolve_tile_cfg(..., variant=self)` is passed `self` from all three variants' `resolve()`.

**Why.** `OpImplRegistry.supported_microtilings` is keyed only on `op_type`, and answered from the
first candidate variant that had a table — regardless of which variant would actually be selected.
Sparse needs `(4,16,8)`; dense picks `(4,8,8)`. With sparse registered at a higher `plevel`, every
dense node would have been offered the sparse table.

I first fixed this by filtering the registry's candidates through `matches()`. **That is not
sufficient once sparsity is the default**, and the reason is worth recording: under `'auto'` both
variants match an eligible node, so the filtered lookup still returns the sparse table — including
to the *dense* variant on the fallback path, which would then resolve a dense kernel onto a sparse
microtile. Asking the resolving variant directly removes the ambiguity at its root instead of
guessing at it, so `supported_microtilings` went back to its original signature with a docstring
saying not to use it from inside a variant.

This resolves E1 without inventing a separate `op_type`, which was the alternative the design doc
floated. A new `op_type` would have needed a parallel resolver, parallel frontend plumbing and
parallel traits registration, all to express "the same operator with a different weight encoding".

Pinned by `test_dense_directive_opts_out`, which asserts the opted-out layer gets a *dense*
microtile.

### 4.3 New: speculative selection and fallback

`OpImplVariant.selection_is_speculative(node, device)` (default `False`) and a fallback loop in
`FamilyResolver.resolve()`:

```python
for variant in get_op_impl_registry().candidates(self.op_type):
    if not variant.matches(node, device):
        continue
    try:
        config = variant.resolve(node, device, directives)
    except Exception as exc:
        if not variant.selection_is_speculative(node, device):
            raise
        log.warning(...)          # naming the node and the reason
        continue
    return config, variant
```

**Why this is required and not belt-and-braces.** `sparse_eligibility` is a capability check and
deliberately does not try to predict everything `resolve()` could reject. Two of the sparse
kernel's shape constraints hold structurally — `_parallelism_candidate` aligns the K slice up to
`2 * microtile_k` and the N slice up to `2 * microtile_n`, which is exactly what
`validate_config` needs — but **tile memory does not**. The sparse microtile has a wider K, so a
shape that fits at K=8 could leave no valid parallelism at K=16 and `_resolve_parallelism` raises
`no valid parallelism fits tile memory`. Under an opt-in directive that error is correct. Under an
opt-out default it would break a graph the user never asked to change.

The default `False` means every other family's behaviour is bit-for-bit unchanged: a resolve
failure propagates exactly as before. The loop also stays lazy — `matches()` is still only called
until a candidate resolves — so no variant's `matches()` is invoked on a node an earlier variant
already claimed.

I scanned ten shapes from 64×64 up to 1024×4096 looking for a real case that trips this; none did
(the two that failed, failed for dense too). The mechanism is therefore tested by monkeypatching
the sparse `resolve()` to raise, which also lets the companion test assert that an explicit
`'2of4'` request does *not* fall back.

### 4.4 Fixed: the RTP update count in `app.cpp`

**`templates/firmware/app.cpp.jinja`**

```diff
-  {{ L.struct_name }}::IN_FEAT_SLICE * {{ L.struct_name }}::OUT_FEAT_SLICE
+  {{ spec.array[0][0]|length }}
```

The host-side `dut.update(...)` element count was computed as the dense product. A compressed ROM is
smaller than `K_slice * N_slice`, so this would have over-run the parameter port. Taking the length
from the packed artifact itself is correct for both paths and cannot drift.

### 4.5 Detection and packing helpers

**`op_impls/families/matmul/common.py`** (+348 lines)

| Added | Purpose |
| --- | --- |
| `SPARSE_MICROTILE_OPTIONS` | Sparse-capable `(M,K,N)` per generation and dtype pair, transcribed from the AIE API table. Footnoted (emulated / extra-data-manipulation) shapes are **omitted rather than guessed at**. |
| `SPARSITY_MODES`, `DEFAULT_SPARSITY_MODE` | The directive's accepted values (`auto`/`dense`/`2of4`) and the opt-out default. Sits next to the existing `DEFAULT_CONTRACT` and is labelled the same way — policy, not a law. |
| `sparse_eligibility(node, device)` | The gate in front of the default. Returns `(eligible, reason)` and **never raises**: it is asked of every Dense node, including ones that could never be sparse, so a `matches()` built on it stays total. Checks capability first — generation, integer precisions, constant weights, 8-bit weight storage, a sparse overload for the precision pairing, any pinned microtiling directive — then the decisive question, whether the weights already carry the pattern. Capability is cheap and comes first; the weight walk is last because it is the expensive one. |
| `sparse_capability(node, device)` | Everything *except* the weights: generation, precisions, constant weights, 8-bit storage, a sparse overload, any pinned microtile. Split out so a caller can tell "this target has no sparse path" from "these weights do not carry the pattern" — `report_unusable_sparsity` only has something useful to say in the second case. |
| `sparse_microtiling_options(generation, node)` | The node's sparse-capable microtiles, or `[]`. Shared by `sparse_eligibility` and the variants' `microtiling_options`. |
| `SPARSITY_RATIO`, `SPARSE_CHUNK_DATA_BYTES`, `sparse_chunk_elems()` | 2-of-4; and **64 uncompressed *bytes*** per mask+data chunk, taken from `me_native.h`'s `mask_length_sparse` rather than UG1603's prose. Stating the chunk in bytes rather than elements is what makes it width-independent — see §5.4. |
| `SPARSE_ELEMENT_BITS` | The weight storage widths the packer implements, `{8, 16}`. |
| `requested_sparsity(node)` | Reads and validates the directive, defaulting to `DEFAULT_SPARSITY_MODE`. Raising here means an unknown mode fails at resolve time with the node name attached. |
| `sparse_tile_bytes(k, n)` | Mirrors `aie::sparse_vector<T, K*N>::memory_bytes()`. Constant per shape — which is what lets the kernel address tiles by a compile-time stride. For `(16,8)`: 64 + 16 = **80 bytes**. |
| `sparse_mask_along_k(W)` | Derives the **mask** for an already-2:4 matrix; does not touch the values. A group with more nonzeros than the format can hold **raises** rather than being thinned — the replacement for the pruning function this module used to own. A group with fewer than two nonzeros is topped up with explicitly-stored zeros, which costs nothing numerically and is what keeps every tile the same size. |
| `is_structured_sparse_2of4(W)` / `worst_group_nonzeros(W)` | Whether the pattern is encodable, and by how much it misses. ≤2 nonzeros per group of 4 is an **upper bound**, not an exact quota — that reading comes from UG1603's "at least two zeros" together with the partial-decompression table defining only 0/1/2-set-bit patterns. A K that is not a multiple of 4 is zero-padded, matching what the packer does to its edge slab. |
| `quantized_weight_matrix(node)` | The node's constant weights as the `[K, N]` integers the kernel will multiply, or `None` for anything not analysable. The quantized domain is the right one to ask about sparsity in: a weight that quantizes to zero *is* a zero to the kernel, and quantization can only add zeros, never remove them. |
| `analyze_weight_sparsity(node)` | The measurement the whole feature turns on: `structured_2of4`, `zero_fraction`, `worst_group_nonzeros`. Memoized in `node.artifacts['weight_sparsity']` — `matches()` can be asked more than once per node and this walks the whole matrix; weight data does not change after import. |
| `SUBSTANTIAL_SPARSITY_FRACTION`, `report_unusable_sparsity(node)` | The diagnostic for a layer that was plainly trained for sparsity (≥50% zeros) but not in the 2:4 layout. Warns once per node, via the memo's own `warned` flag, so asking twice does not say it twice. |
| `_compress_sparse_tile(values, kept)` | One microtile → bytes. Column-major over the whole tile; each 64-element chunk is an 8-byte mask (one bit per byte, LSB-first within each mask byte, mask bytes ascending) followed by the kept bytes in ascending position order. |
| `pack_mmul_rhs_sparse_matrix(...)` | Full ROM, shaped `(cas_num, cas_length, tiles * tile_bytes)`. Compresses the weights **as given** — nothing is pruned — and raises out of `sparse_mask_along_k` if a slab is not 2:4, which by then is the invariant asserting itself rather than the expected way to find out. Zero-fills partial edge tiles exactly as the dense packer does. |

The tile *ordering* in `pack_mmul_rhs_sparse_matrix` — by (n, n+1) microtile **pair**, then K-tile,
then the two columns of the pair — is not arbitrary; see §5.3.

Grouping along K per output column is worth restating for anyone writing a pruning recipe upstream
of aie4ml — which is now the only place a pruning recipe lives: because the layout is
column-major, you must zero along the **input-feature** axis in groups of four, one output column
at a time — *not* whole output columns.

### 4.6 Config and the dense variant

**`config.py`** — `DenseConfig` gained `sparsity: str = 'dense'` and `sparse_tile_bytes: int = 0`.
Defaults mean the dense variants construct unchanged.

**`dense.py`** —
- `matches()` now additionally requires `requested_sparsity(node) != '2of4'` — it declines a node
  that explicitly asked for sparsity, but stays available under `'auto'` as the fallback candidate
  behind the higher-plevel sparse variant.
- New `weight_rom_elems()`, surfaced as the `wts_elems` template param: the dense product for a
  dense ROM, `tile_bytes × tiles` for a compressed one. The kernel declares
  `weight_t (&wts)[ConfigT::WTS_ELEMS]`, so this must agree with the packed artifact's length
  exactly or the ADF parameter port is mis-sized. Lives on the shared base because both paths need
  it.

**`matmul.py`** — `matches()` raises `NotImplementedError` only for an explicit `'2of4'`. A dynamic
matmul streams its B operand at runtime, so it cannot be pre-compressed; there is no sparse matmul
variant and there cannot be one under this design. Raising beats returning `False`, which would fall
through to the dense matmul and ignore the directive.

Narrowing this from "any non-dense mode" to "`2of4` only" was **required** by the inversion: with
`'auto'` as the default, the original condition would have raised on every dynamic matmul in every
model. `test_dynamic_matmul_unaffected_by_the_auto_default` guards it.

**`__init__.py`** — exports the new helpers and variants.

### 4.7 Kernel

**`templates/nnet_utils/dense_bias_relu/dense_bias_relu.{h,cpp}`**

The verified prototype kernel, promoted. The sparse path is a `if constexpr (ConfigT::USE_SPARSE)`
branch in all four stage specializations:

```cpp
static constexpr int N_PAIR_STRIDE = ConfigT::SPARSE_TILE_BYTES * 2 * (colA / K);
aie::sparse_vector_input_buffer_stream<weight_t, MMUL::size_B> sB(pB + (j / 2) * N_PAIR_STRIDE);
aie::sparse_vector<weight_t, MMUL::size_B> B0 = sB.pop();
aie::sparse_vector<weight_t, MMUL::size_B> B1 = sB.pop();
```

`wts`'s array bound became `ConfigT::WTS_ELEMS` in all four kernel classes (was the
`IN_FEAT_SLICE * OUT_FEAT_SLICE` product), and `static_assert`s were added for the ROM size limit,
1-byte weights when sparse, and `IN_FEAT_SLICE % K == 0` when sparse.

`dense_bias_relu_graph.h` is unchanged and reused by both variants.

The **dense** path through this rewritten kernel was checked byte-identical to the committed kernel
(§6), so the promotion is behaviour-preserving for existing users.

**`templates/firmware/variants/dense_bias_relu/parameters.h.jinja`** — gained `USE_SPARSE = false`,
`SPARSE_TILE_BYTES = 0`, `WTS_ELEMS`, since the two variants now share one kernel source.

### 4.8 Frontends

**`frontends/onnx/utils.py`** — `normalize_directives` handles `sparsity`, accepting either
`{'mode': ...}` or a bare mode string (`'mode'` is the only key the 2:4 format has to
parameterise), and validating against `SPARSITY_MODES`. Since sparsity is opt-out, the value
users will actually write here is `'dense'`.

**`frontends/hls4ml/utils.py`** — same in `extract_layer_directives`, guarded by `if sparsity_cfg:`
so the default empty dict does not emit a directive.

**`frontends/hls4ml/backend.py`** — `ConfigurableAttribute('sparsity', value_type=dict, default={})`
on the Dense attrs, without which hls4ml drops the key before aie4ml ever sees it.

### 4.9 Docs

- **`docs/support.md`** — a Sparse Dense / GEMM row and a sparsity-contract bullet.
  `AGENTS.md` §1 requires support-surface changes in the same commit.
- **`AGENTS.md`** — the `sparsity` directive in §6, and the sparse variants in the registered-families
  listing.
- **this file**.

---

## 5. The three findings that were load-bearing

Documented in-code as well, because each is invisible from the API surface.

### 5.1 Element order inside a compressed tile is plain column-major

Over the *whole* tile — column 0's K values, then column 1's, and so on — exactly as UG1603 says.
Row-major, and either ordering split into two K-halves, were each built and simulated; all mismatch.
I had briefly implemented K-half chunking on a wrong reading and had to revert it.

### 5.2 The L1 input DMA tiling must track the microtile

`graph_plan.h`'s L1 input `read_access` needs `tiling_dimension = {microtile_k, microtile_m}` with
dim-0 traversal `{stride = microtile_k, wrap = IN_FEAT_SLICE / microtile_k}`.

On the compiler path aie4ml emits this from the same resolved microtile as `parameters.h`, so the
two cannot drift — confirmed by inspecting the emitted project and pinned by
`test_emitted_parameters_and_dma_tiling_track_the_microtile`. The hand-fix needed in the prototype
was purely an artifact of editing an already-emitted project.

Worth recording anyway, because the failure mode is nasty: K=8 and K=16 staging *coincide* for row 0
/ features 0–7, so a probe of the first K-group appears to pass while everything else is
mis-tiled. This cost real time.

### 5.3 One live sparse stream, not two

Both B operands of a K-tile must come from a **single** live
`aie::sparse_vector_input_buffer_stream`, popped twice — matching `gemm_int8xint8_sparse`'s
`tsB >> Ybuff0 >> Ybuff1`. The (n, n+1) pair-major ROM ordering in
`pack_mmul_rhs_sparse_matrix` exists to suit exactly that access pattern.

Two concurrently-live streams (one per column) are **bit-exact under x86simulator and wrong under
aiesimulator**. `me_native.h` shows why: x86sim models `sparse_fill()` as pure bookkeeping
(`compr_bytes += 32`, pointer untouched), so inter-stream contention is invisible there, while on
the AIE core the sparse decompression FIFO is real shared state the two streams corrupt for each
other. Rebuilding a stream per tile is also unsafe: `fifo_buffer_stream`'s constructor leaves `idx_`
uninitialized while `pop()` branches on it.

**Corollary: for sparse work, x86sim passing is not sufficient.** This is why the differential test
runs under both simulators (§6), for both weight widths — the FIFO is shared state keyed on the
*sparse* operand, so each distinct weight width is a distinct risk, while a change of activation
width is not.

### 5.4 The mask is one bit per *byte*, which is what makes widths uniform

The rule that generalises the whole format: the mask carries one bit per byte of uncompressed
data. So an int8 sample owns one mask bit and an int16 sample owns two adjacent bits, both set
when the sample is stored — `me_native.h`'s `decompress_sparse` tests each field with `bool()`,
and `sparse_read8b`/`sparse_read16b` differ only in how far they step the same 64-bit mask
(4 bits per group of four int8s, 8 bits per group of four int16s).

Stated that way, a chunk is *always* an 8-byte mask plus 32 data bytes covering 64 bytes of
original data, for any width. Two consequences worth recording:

- `sparse_tile_bytes` needed no formula change at all when int16 was added, only a widened guard.
  It already was `aie::sparse_vector::memory_bytes()`
  (`(Elems/sparsity_ratio())*type_bits + (Elems*type_bits)/8`) for any width.
- int8 `(16,8)` and int16 `(8,8)` both come to **80 bytes** — the halved element count is exactly
  offset by the doubled element width. A pleasant coincidence, and a trap: a width mix-up cannot
  be caught by a tile-size check, which is why the round-trip test asserts decoded *values* for
  both widths rather than only sizes.

The one place bytes-versus-elements does bite is pointer arithmetic. `SPARSE_TILE_BYTES` is a byte
count but `pB` is a `weight_t*`, so the kernel derives
`SPARSE_TILE_ELEMS = SPARSE_TILE_BYTES / sizeof(weight_t)` and strides by that; likewise
`weight_rom_elems()` divides the ROM byte count by the weight size, because `_pack_sparse`
reinterprets the same bytes as `weight_t` and the two must agree or the ADF parameter port is
mis-sized. For a 1-byte weight both reduce to what they were before.

---

## 6. Verification

`tests/test_sparse_dense.py` — 37 tests: 35 IR-only plus a differential test parameterised over
both simulators. `_build` defaults to 2:4-structured weights (`_weights_2of4`), since selection
is weight-driven; `_weights_dense` and `_weights_unstructured_sparse` are the fixtures for the
layers that must *not* go sparse. The platform to compile against is deduced from the installed
Vitis (`tests/aie_platform.py`), not hardcoded.

| Check | Result |
| --- | --- |
| **The weights decide**: already-2:4 layer goes sparse with no directive; an ordinary dense layer goes dense *without a word*; a ≥50%-zero layer that is not 2:4 goes dense **and warns**, once | pass |
| **Opting out costs nothing**: `mode: 'dense'` leaves `node.artifacts['weight_sparsity']` unpopulated, i.e. the matrix is never walked; an `auto` layer is measured and the result cached | pass |
| **No pruning exists**: `prune_to_2of4_along_k` is gone from the package and no name in `common` contains "prune"; `sparse_mask_along_k` refuses to thin an overfull group | pass |
| **Adoption changes nothing**: full ROM decoded by an independent decoder reproduces the model's own quantized weights element for element; `pack()` runs silently; `node.inputs[1].data` is untouched | pass |
| **Every packing geometry decodes losslessly, and the padding decodes to zero** — `inner` (N split, `cas_num=2`), `outer` (**K cascade**, `cas_length=2`, the axis 2:4 groups live on), `inner` and `outer` with slices overshooting K and N (**zero-filled edge tiles**), and **int16 x int8** on its own `(2,16,8)` microtile | pass |
| **`outer` geometry**: `cas_length > 1`, `k_slice % microtile_k == 0`, `microtile_k % 4 == 0` — the alignment the no-straddling-a-stage-boundary guarantee rests on; all chains byte-identical | pass |
| **Multi-layer graphs**: a sparse and a dense Dense layer in one graph get their own variant, `param_template` and ROM, and one emitted `parameters.h` carries `USE_SPARSE = true` and `= false` exactly once each; two sparse layers do not share a ROM | pass |
| **Opt-out and opt-in**: `mode: 'dense'` forces dense with a dense microtile; explicit `2of4` still works; `contract='outer'` → `.row.v1` | pass |
| **Never breaks a build**: ineligible layer (bfloat16) falls back silently; pinned non-sparse microtile falls back silently; a speculative resolve failure falls back to dense *and logs it* | pass |
| **Never silently ignores you**: explicit `2of4` raises on non-2:4 weights (`ValueError`, naming the pattern) and on an ineligible layer; explicit `2of4` does not fall back on a resolve failure | pass |
| **Dynamic matmul unaffected** by the `auto` default (the regression that would have broken every model) | pass |
| `sparse_eligibility` is total and explains itself (eligible / wrong generation / runtime weights / weights not 2:4) | pass |
| Packed ROM is 5/8 of dense, 1-byte dtype, `(cas_num, cas_length, …)`-shaped; report says `pruned_by_aie4ml: False` | pass |
| Emitted `parameters.h` (`USE_SPARSE`, `SPARSE_TILE_BYTES = 80`, `WTS_ELEMS`), `app.cpp` count, and `graph_plan.h` tiling all track the microtile | pass |
| Mask derivation + compress round-trip against an independent decoder; sparser-than-2:4 groups topped up to a constant tile size | pass |
| **Differential sparse vs dense, `x86sim`** | bit-identical (re-run after the no-pruning change, Vitis 2026.1) |
| **Differential sparse vs dense, `aiesimulator`** | bit-identical (re-run after the no-pruning change, Vitis 2026.1) |
| **Opt-out path vs committed `HEAD` kernel, `x86sim`** | byte-identical output — no regression |
| Full suite (Vitis 2026.1, conda env `aie4ml_env`) | **55 passed, 0 skipped** in 9m55s — the whole repo, both simulators |
| `ruff check` / `ruff format` on changed files | clean |

The differential test is simpler than it used to be, because neither path alters the weights any
more: both variants are handed the identical 2:4-structured matrix, so any output difference is
attributable to the compressed layout or the sparse kernel and nothing else. Both simulators,
together, take ~4.5 min.

Note the dense side of that test now passes `mode: 'dense'` explicitly — under the opt-out default,
omitting the directive would build two sparse projects and compare them to each other. That is the
kind of silent test-degradation an inverted default invites, so it is called out in the code.

Also checked by hand: the aie4ml packer's output is **byte-identical** to the prototype's
proven-good ROM.

`tests/test_hls4ml_*.py` — the pre-existing end-to-end dense tests — now run in full. They used to
fail under Vitis 2026.1 because the hls4ml backend's default part is `xilinx_vek280_base_202520_1`,
which 2026.1 does not ship, so `v++` rejected the platform before compiling anything. The tests now
deduce the part from the installed Vitis and pass it explicitly, so the suite follows a toolchain
upgrade on its own; the backend default itself is still stale (§8).

### Running the tests

```bash
# IR-only (fast); needs hls4ml for the frontend tests
/eos/user/e/enlupi/miniconda3/envs/aie4ml_env/bin/python -m pytest -q -m "not requires_vitis"

# Vitis-gated. XILINX_VITIS is not set by default in this shell even though Vitis is on PATH;
# setting it is enough, sourcing settings64.sh is not required.
XILINX_VITIS=/opt/Xilinx/2026.1/Vitis/ \
  /eos/user/e/enlupi/miniconda3/envs/aie4ml_env/bin/python -m pytest -q -m requires_vitis
```

The conda env `aie4ml_env` has `hls4ml` 1.3.0 and an editable aie4ml install; the older
`/tmp/aie4mlenv` venv works for everything except the hls4ml frontend tests. The sparse
differential test takes ~4.5 min for both simulators.

## 7. Limits

- **int8 and int16 weights only.** `sparse_tile_bytes` and `_compress_sparse_tile` raise for other
  widths. 4-bit sparse weights exist in the AIE API table (`8b x 4b`, AIE-ML only) but aie4ml has
  no 4-bit dense path to build on; 32-bit has no sparse mode on any generation.
- **Three reachable pairings**, all verified against the dense path on identical weights in
  x86simulator, and the two distinct *weight* widths additionally in aiesimulator:
  int8×int8 `(4,16,8)`, int16×int8 `(2,16,8)` (AIE-ML only), int16×int16 `(2,8,8)`.
  uint8×int8 folds onto the int8×int8 row via `legality_format` and is reachable but untested.
- **AIE-MLv2 is not exercised by the test suite** (`tests/aie_platform.py` pins Vek280), so its
  rows are transcribed from the AIE API table but unverified here.
- **No sparse dynamic matmul**, by construction (§4.5).
- **No bfloat16 sparse.** The AIE API table does list modes for it (`4x16x4` unfootnoted on
  AIE-ML, `4x16x8` on AIE-MLv2), and bfloat16 is 2 bytes so the mask layout is the same as
  int16's. What is missing is a float path through detection: `sparse_capability` rejects floats
  outright, and `quantized_weight_matrix` reasons in the quantized integer domain, whose
  justification ("a weight that quantizes to zero *is* a zero to the kernel") has no bfloat16
  analogue — it would have to test the bfloat16 bit pattern instead.
- **The pattern must be exactly 2:4, in the hardware's grouping.** Sparsity that is unstructured,
  or structured along the wrong axis, or 3-of-4 in even one group, is unusable — the layer is
  compiled dense. A model with a lot of zeros can therefore get no benefit at all; the warning
  says so, but only retraining fixes it. This is a hardware-format constraint, not a policy
  choice: there is no encoding for a group of 3.
- **Detection is per layer and all-or-nothing.** A layer is sparse or it is not; there is no
  partial adoption of the qualifying columns of a mostly-unstructured layer.
- **Opting out is per layer.** There is no model-wide switch; a `sparsity: {'mode': 'dense'}`
  directive has to be set on each Dense layer that should stay dense. Since adoption is now
  numerically free, the reason to opt out is a performance or memory trade-off rather than
  accuracy.
- **Cycles were measured on the prototype only** (§3), not re-measured through the compiler path.

## 8. Pre-existing issues left alone

**The hls4ml backend's default part is stale for Vitis 2026.1.**
`frontends/hls4ml/backend.py:176` defaults to `part='xilinx_vek280_base_202520_1'`, but 2026.1
installs `xilinx_vek280_base_202610_1` (`/opt/Xilinx/2026.1/Vitis/base_platforms/`). Any hls4ml
model that does not pass an explicit part fails in `v++` with *"Invalid platform file ...
xilinx_vek280_base_202520_1.xpfm"*. The tests now sidestep it by deducing the part
(`tests/aie_platform.py`), which is why they pass; the default is untouched because changing it
changes the default target for every hls4ml user, which is not this branch's call to make.

**The hls4ml `sparsity` shorthand is unreachable.** `frontends/hls4ml/utils.py:117-129` accepts
either `{'mode': ...}` or a bare mode string, and this document advertises the shorthand — but
`backend.py:68` declares `ConfigurableAttribute('sparsity', value_type=dict, default={})`, so
hls4ml's own attribute validation raises *"Expected <class 'dict'>, got <class 'str'>"* before
aie4ml's parser is reached. The string branch is dead code on that path. Verified end-to-end
through `convert_from_keras_model`: no directive on 2:4 weights gives
`('dense.sparse.b.r.v1', '2of4')`, `{'mode': 'dense'}` gives `('dense.b.r.v1', 'dense')`, and the
bare string raises. The ONNX path (`frontends/onnx/utils.py:88`) handles the string, the dict, an
unknown mode and a bad type correctly. Predates this work in the sense that the attribute
declaration and the parser were written together and never tested against each other; worth
noting that with sparsity opt-out, a directive that silently fails to arrive now yields the
*opposite* of what was asked, so this passthrough deserves a test either way.

`ruff check .` still reports 7 findings that predate this work — trailing whitespace in
`src/aie4ml/passes/placement.py:336-337`, and import ordering / an ambiguous name in
`tutorials/tutorial_{1,2}.ipynb` and `tutorials/tutorial_3.py`. The repo is also not uniformly
`ruff format`-clean (10 files under `src/` and `tests/` drift, 18 repo-wide). Left untouched to keep
this diff to its subject; the notebooks in particular are under active edit. `src/aie4ml/aie_devices.json` shows as modified but predates
this branch's work.

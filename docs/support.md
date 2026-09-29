# Operator and Feature Support


## Operators

| Operator / feature | Current support | Notes and limitations |
| --- | --- | --- |
| Dense / GEMM | Int8/int16, BF16, FP8 | Read-only RTP weights; optional bias and fused ReLU; configurable cascade parallelism. |
| Sparse Dense / GEMM | **AIE-ML only** (int8 x int8, int16 x int8, int16 x int16); AIE-MLv2 is incorrect -- see below | **Detected automatically, numerically free.** Every Dense layer's trained weights are inspected; a layer that is *already* 2:4-structured along the reduction axis (at most 2 nonzeros per 4 consecutive input features, per output column) is stored in the AIE-ML hardware-compressed format (0.625x the dense ROM) and run on the sparse `aie::mmul` path. **aie4ml never prunes** -- creating the pattern is the training pipeline's job -- so compilation never changes a weight, and the sparse path is chosen on merit rather than paid for in accuracy. A layer that is not already 2:4, or that misses a capability requirement (AIE-ML or AIE-MLv2, integer precisions, 8- or 16-bit constant weights, a sparse-capable microtile), is compiled dense; if it nonetheless has >=50% zero weights, a warning names the layer and says the pattern is unusable. `sparsity: {'mode': 'dense'}` opts a qualifying layer out; an explicit `2of4` request on a layer that cannot support it is an error. |
| Dynamic MatMul | Int8/int16, BF16, FP8 | Rank-2 GEMM ABI. A rank-2 RHS may be broadcast across compacted independent LHS axes; batched RHS MatMul with non-broadcast leading axes is rejected. |
| Elementwise Add | Quantized integer | Exact-shape inputs for residual and elementwise connections; broadcasting is not supported. |
| LayerNorm | Quantized integer | Last-axis normalization using integer mean/variance and reciprocal-square-root approximation. Requires supported static quantization and kernel legality constraints. |
| Softmax | Int8 to uint8/int16 | Accurate integer exponential or an opt-in surrogate (needs explicit parameters + QAT). Linear or microtile layout; the exp variant is not yet perf-optimized. |
| Transpose / Permute | Memtile-backed view | Permutation of the final two axes only. |
| Split / Slice | Direct or per-slice memtile | No Split/Slice kernel. A slice must be an exact union of complete producer-port regions; crossing producer ports, graph-boundary slices, and chained views are not supported. |
| Concat | Direct or per-input memtile | No Concat kernel. Each consumer port must read entirely from one input slice; graph-input-backed Concat and chained views are not supported. |
| Fanout / branching | Direct or per-consumer memtile | One producer tensor may feed multiple independently planned consumer transport legs. |
| Constant scale | Quantized integer | Constant power-of-two output scaling is folded into Dense/MatMul output shifts. Arbitrary constant scaling is not yet supported. |
| Activation | ReLU | Fused into Dense; no standalone activation kernel. |

## Frontends

| Frontend | Current support | Notes and limitations |
| --- | --- | --- |
| ONNX | Recommended operator-level frontend | Supports explicit graphs composed from supported operators and quantized Q/DQ boundaries. |
| hls4ml | Optional MLP-oriented frontend | Intended primarily for Dense-style pipelines. Install `hls4ml` separately when using this path. |


## Tensor and View Contracts

- AIE execution buffers remain 2-D. Rank-preserving logical views are compacted only where doing so preserves operator
  semantics.
- Dynamic MatMul rejects a non-broadcast batched RHS. Such operations must be lowered into parallel rank-2 MatMul
  subgraphs or use a future batched-MatMul implementation.
- Permute supports the final two axes only.
- Split/Slice and Concat are folded view operations and do not instantiate AIE kernels.
- Chained folded views are not currently supported.
- Per-tensor static quantization is the primary supported quantization contract. Per-channel activation quantization is
  not supported.
- Hardware sparsity is a weight-only, structured 2:4 contract along the reduction axis. Unstructured sparsity,
  activation sparsity, and runtime-variable sparsity patterns are not supported.
- The sparsity pattern must come from training. aie4ml detects it and never creates it: there is no pruning step in
  the compiler, so a weight it was given is a weight the kernel computes.
- Hardware sparsity is applied opportunistically: aie4ml inspects each Dense layer's weights and never fails a build
  over sparsity. If a layer whose weights qualify turns out not to be resolvable as sparse, the dense implementation
  is used and the fallback is logged.
- Sparse precision pairings are per generation, and narrower than the dense ones. Weights must be 8- or 16-bit.
  int8 x int8 and int16 x int16 are sparse-capable on AIE-ML and AIE-MLv2; **int16 x int8 only on AIE-ML**, because
  AIE-MLv2 requires both sparse operands to share a data type (UG1603). Float precisions have no sparse path in
  aie4ml: bfloat16 has a sparse `aie::mmul` mode but is not wired up, while float32 and fp8 have no sparse mode at
  all on any generation.
- int8 x int8, int16 x int8 and int16 x int16 are each verified **on AIE-ML** in x86simulator against the dense path on
  identical weights; int8 x int8 and int16 x int16 -- the two distinct weight widths -- are additionally verified in
  aiesimulator.
- **Hardware sparsity on AIE-MLv2 is unverified and currently unverifiable; do not rely on it.** AIE-MLv2 *dense* is
  bit-identical to AIE-ML dense, so only the sparse path is in question. AIE-MLv2 *sparse* differs from that reference
  in 2001 of 2048 output elements, but that measurement is **not evidence about the hardware**: it came from
  x86simulator, and the installed `aie2ps` intrinsic model
  (`aietools/data/aie2ps/lib/me_native.h`) contains **no sparse decompression at all** -- no `decompress_sparse`, no
  `sparse_read8b`/`sparse_read16b`, no `sparse_pop`/`sparse_fill`, where the `aie_ml` model has all of them. So
  x86simulator cannot execute a sparse kernel correctly on AIE-MLv2 whatever the ROM contains, and cannot be used to
  validate this path. aiesimulator would be the only trustworthy check and is unavailable here: the hardware compile
  fails with `Could not load Device Part xc2ve3858-ssva2112-2MP-e-S`, identically for dense, so it is a missing device
  part rather than anything about sparsity. A second, independent doubt remains open on the aie4ml side: the packer
  emits the AIE-ML chunk geometry (64-byte chunks, 64-bit mask) while UG1603 states that "for AIE-ML v2, data chunks
  are 512-bit wide, and the mask and alignment is on 64 bits". Since sparsity is opt-out, a VEK385 Dense layer with
  already-2:4 weights takes this path **by default**, so set `sparsity: {'mode': 'dense'}` on AIE-MLv2 targets pending
  a study. The AIE-MLv2 rows of `SPARSE_MICROTILE_OPTIONS` are retained for that study and are not a statement of
  support.

## Transport

- Internal transport is realized as either a direct AIE connection or one memory-tile stage.
- Fanout creates independently planned transport legs for each consumer.
- Split/Slice and Concat may use direct connections when producer and consumer port regions align exactly; otherwise
  each legal leg may use a memory tile.
- Graph boundaries may expose multiple ports for partitioned tensors.
- Multi-stage relay transport is not implemented. Topologies requiring an additional relay stage fail explicitly.
- Internal AIE-to-PL-to-AIE bridge points and complete Versal system-link generation are not yet implemented.

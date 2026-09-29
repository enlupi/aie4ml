# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Dense variants that drive AIE-ML's hardware sparse (2:4) `aie::mmul` overload.

Selection is driven by the weights, not by a directive. Every Dense layer that has not opted out
is inspected (`sparse_eligibility` -> `analyze_weight_sparsity`), and a layer whose trained
weights are *already* 2:4-structured along the reduction axis takes this path; one that is not is
compiled by the ordinary dense variants. The `sparsity` directive can only veto or insist:
`{'mode': 'dense'}` is the opt-out, checked first so an opted-out layer costs nothing to skip,
and `{'mode': '2of4'}` demands this path, turning a layer that cannot take it from a silent
fallback into a hard error.

**aie4ml never prunes.** Creating the 2:4 pattern belongs to the training pipeline -- it is a
model-accuracy decision, made with the loss function in hand -- so the compiler only detects a
pattern that is already there. That makes this path numerically free: the kernel computes exactly
the weights the model was handed over with, which is why it can be on by default.

Everything except weight packing is shared with the dense path -- same kernel
source, same graph, same ports, same footprint -- because the compressed weight ROM reaches the
kernel through the identical `parameter`-port mechanism the dense weight ROM already uses. What
changes is the microtile table (sparse-capable shapes are a distinct set, see
`SPARSE_MICROTILE_OPTIONS`), the packer (`pack_mmul_rhs_sparse_matrix`), and three extra
compile-time constants in `parameters.h` (`USE_SPARSE`, `SPARSE_TILE_BYTES`, `WTS_ELEMS`).

Verified bit-exact against a NumPy reference, and against the dense kernel fed the same weights,
in both x86simulator and aiesimulator.

`pack()` records the measured sparsity in `inst.artifacts['sparsity_report']`. A model that was
not trained with 2:4-structured sparsity simply never reaches here; when it has substantial but
unstructured sparsity, `report_unusable_sparsity` says so rather than letting the lost saving
pass unnoticed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ....ir.graph import OpImplInstance, OpNode, input_tensor_for_role
from ...registry import register_variant
from ...utils import ParallelismConfig, parse_directives
from ...utils.precision import (
    aie_rounding_token,
    infer_accumulator_tag,
    resolve_accumulator_output_shift,
)
from .common import (
    SPARSE_ELEMENT_BITS,
    SPARSE_MICROTILE_OPTIONS,
    SPARSITY_RATIO,
    describe_inner_lhs_staging,
    describe_inner_output_staging,
    describe_outer_lhs_staging,
    describe_outer_output_staging,
    is_structured_sparse_2of4,
    np_bias_dtype_for_spec,
    np_dtype_for_spec,
    pack_mmul_rhs_sparse_matrix,
    pack_vector_by_n_slice,
    report_unusable_sparsity,
    requested_contract,
    requested_sparsity,
    select_generation_key,
    sparse_eligibility,
    sparse_tile_bytes,
)
from .config import DenseConfig, DenseFlags
from .dense import _DenseVariantBase
from .resolver import (
    _build_matmul_io_views,
    _resolve_bias_dtype,
    _resolve_numeric,
    _resolve_output_scale_shift,
    _resolve_parallelism,
    _resolve_tile_cfg,
)


class _DenseSparseVariantBase(_DenseVariantBase):
    """Shared behaviour of the sparse dense variants, independent of the contract."""

    param_template = 'dense_bias_relu_sparse'
    # Above the dense variants so FamilyResolver's descending-plevel scan tries these first;
    # they only match when the sparsity directive asks for it, so the dense path is unaffected.
    plevel = 20

    def matches(self, node: OpNode, device) -> bool:
        if requested_contract(node) != self.contract:
            return False

        mode = requested_sparsity(node)
        if mode == 'dense':
            # An explicit opt-out settles it before anything is measured. Inspecting the weights
            # of a layer whose outcome the user has already vetoed would only cost time (the
            # analysis walks the whole matrix), and a caller who opts out owns that choice.
            return False

        # Otherwise the weights decide: whether this layer arrived from training already 2:4
        # structured is what puts the sparse path on the table at all (`sparse_eligibility` ends
        # in `analyze_weight_sparsity`). The directive can veto, never conjure.
        eligible, reason = sparse_eligibility(node, device)
        if eligible:
            return True
        if mode == '2of4':
            # An explicit request claims the node even when it cannot be honoured, so `resolve()`
            # can fail with a specific reason instead of falling through to the dense variant and
            # silently ignoring the directive.
            return True
        # Only 'auto' is left: nobody opted out and nobody asked. If the layer was plainly
        # trained sparse but not in the layout the hardware needs, that is worth one line.
        report_unusable_sparsity(node, device)
        return False

    def selection_is_speculative(self, node: OpNode, _device) -> bool:
        # Under 'auto' nobody asked for sparsity, so a resolve() failure must fall back to dense
        # rather than break the build. An explicit '2of4' is not speculative and will raise.
        return requested_sparsity(node) == 'auto'

    def microtiling_options(self, generation: str, query) -> List[Tuple[int, int, int]]:
        return list(SPARSE_MICROTILE_OPTIONS.get(select_generation_key(generation), {}).get(tuple(query), []))

    def resolve(self, node: OpNode, device, directives=None) -> DenseConfig:
        io_route, _, _ = parse_directives(directives)
        self._reject_unsupported(node, device)

        precision = _resolve_numeric(node, device)
        precision['bias'] = _resolve_bias_dtype(node, precision)
        microtiling = _resolve_tile_cfg(node, device, precision['lhs'], precision['rhs'], variant=self)
        tiling = _resolve_parallelism(node, device, microtiling, precision, self.contract)
        io_views = _build_matmul_io_views(node, microtiling, tiling)

        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        lhs_perm = io_views[lhs_tensor.name].perm

        shift = resolve_accumulator_output_shift(lhs_tensor.precision, node.outputs[0].precision, rhs_tensor.precision)
        shift += _resolve_output_scale_shift(node, is_float=False)

        fused_act = node.traits.get('fused_activation')
        use_relu = ((fused_act.data.get('activation') if fused_act else '') or '').lower() == 'relu'

        return DenseConfig(
            precision=precision,
            parallelism=ParallelismConfig(
                cas_length=tiling.cas_length, cas_num=tiling.cas_num, contract=tiling.contract
            ),
            microtiling=microtiling,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag=infer_accumulator_tag(device, None, None, precision['acc']),
            rounding_mode=aie_rounding_token(precision['output']),
            flags=DenseFlags(
                use_relu=use_relu,
                transpose_lhs=bool(lhs_perm is not None and lhs_perm[-1] != (len(lhs_perm) - 1)),
                use_bias=bool(node.metadata.get('use_bias')),
            ),
            sparsity='2of4',
            sparse_tile_bytes=sparse_tile_bytes(
                microtiling.microtile_k, microtiling.microtile_n, int(precision['rhs'].width)
            ),
        )

    def _reject_unsupported(self, node: OpNode, device) -> None:
        """Fail loudly, and specifically, for anything the sparse path cannot do.

        Only reachable for an explicit `'2of4'` request: under `'auto'` an ineligible node never
        reaches `resolve()` because `matches()` already declined it.
        """
        eligible, reason = sparse_eligibility(node, device)
        if eligible:
            return
        # NotImplementedError for a capability gap, ValueError for a request that contradicts
        # itself, mirroring how the rest of the family separates the two. Weights that are not
        # 2:4 are the latter: the hardware and the compiler are both willing, the model is not,
        # and no aie4ml feature would change that -- only retraining would.
        contradiction = 'pinned microtiling' in reason or 'not 2:4-structured' in reason
        error = ValueError if contradiction else NotImplementedError
        raise error(
            f'{node.name}: sparsity mode "2of4" was requested explicitly but this node cannot '
            f'use the hardware sparse path: {reason}. Remove the directive to let aie4ml choose, '
            f'or set the mode to "dense" to compile it dense.'
        )

    def validate_config(self, node: OpNode, config: DenseConfig, device) -> None:
        super().validate_config(node, config, device)

        rhs_bits = int(config.precision['rhs'].width)
        if rhs_bits not in SPARSE_ELEMENT_BITS:
            raise NotImplementedError(
                f'{node.name}: sparse weight packing is implemented for '
                f'{sorted(SPARSE_ELEMENT_BITS)}-bit weights, got {rhs_bits}-bit; the mask '
                'granularity and chunking would need re-deriving.'
            )

        microtile = (config.microtiling.microtile_m, config.microtiling.microtile_k, config.microtiling.microtile_n)
        allowed = self.microtiling_options(
            str(getattr(device, 'generation', '')),
            (config.precision['lhs'].format, config.precision['rhs'].format),
        )
        if allowed and microtile not in allowed:
            raise ValueError(
                f'{node.name}: no sparse aie::mmul overload for microtile {microtile} on '
                f'Generation={getattr(device, "generation", None)!r} and '
                f'(input={config.precision["lhs"].format!r}, weight={config.precision["rhs"].format!r}); '
                f'allowed: {allowed}.'
            )

        # The compressed layout groups four consecutive K values, so a group must never straddle
        # a microtile or cascade-stage boundary. _resolve_parallelism aligns K slices to
        # 2 * microtile_k, so this holds structurally -- assert it rather than assume it.
        lhs_view = config.io_views[input_tensor_for_role(node, 'lhs').name]
        k_slice = int(lhs_view.tile_inner)
        if config.microtiling.microtile_k % 4 or k_slice % config.microtiling.microtile_k:
            raise ValueError(
                f'{node.name}: sparse 2:4 grouping needs microtile_k ({config.microtiling.microtile_k}) '
                f'to be a multiple of 4 and to divide the per-cascade K slice ({k_slice}).'
            )

        output_view = config.io_views[node.outputs[0].name]
        n_slice = int(output_view.tile_inner)
        if (n_slice // config.microtiling.microtile_n) % 2:
            raise ValueError(
                f'{node.name}: the kernel consumes N microtiles in pairs, so the per-chain N slice '
                f'({n_slice}) must contain an even number of {config.microtiling.microtile_n}-wide tiles.'
            )

    def _pack_sparse(self, inst: OpImplInstance, *, n_slice_scale: int = 1):
        """Quantize and compress the already-2:4 weights; shared by both contracts.

        Nothing is pruned: the compressed ROM decodes back to exactly these weights, so this is
        a pure change of storage format.

        `n_slice_scale` is 1 for the 'inner' contract, where cas_num slices N, and cas_num for
        'outer', where every row group holds the full N and the packers are asked for a single
        chain that is then replicated.
        """
        p = inst.config
        W, b, n_in, n_out = self._quantize_weight_bias(inst)
        lhs_view = p.io_views[inst.node.inputs[0].name]
        output_view = p.io_views[inst.node.outputs[0].name]
        cas_num = int(p.parallelism.cas_num)

        packed_W = pack_mmul_rhs_sparse_matrix(
            W,
            K=n_in,
            N=n_out,
            K_slice=lhs_view.tile_inner,
            N_slice=output_view.tile_inner,
            microtile_k=p.microtiling.microtile_k,
            microtile_n=p.microtiling.microtile_n,
            cas_length=p.parallelism.cas_length,
            cas_num=1 if n_slice_scale > 1 else cas_num,
            dtype=np_dtype_for_spec(p.precision['rhs']),
            layer_name=inst.name,
        )
        if n_slice_scale > 1:
            packed_W = np.repeat(packed_W, cas_num, axis=0)

        # The compressed stream is raw bytes, but it is emitted as an array of the weight's own
        # storage type (weight_t) and handed to the kernel through the same
        # `weight_t (&)[WTS_ELEMS]` parameter. Reinterpret rather than convert: the bit pattern
        # is what matters, and emitting 0..255 into an int8_t array would not compile.
        #
        # For a 2-byte weight_t this halves the trailing dimension, which is why the kernel's
        # pointer strides are expressed in weight_t elements (SPARSE_TILE_ELEMS) rather than in
        # bytes -- see dense_bias_relu.cpp. sparse_tile_bytes() is always even, so the view is
        # exact and never truncates.
        rom_dtype = np_dtype_for_spec(p.precision['rhs'])
        itemsize = np.dtype(rom_dtype).itemsize
        if itemsize * 8 not in SPARSE_ELEMENT_BITS:
            raise NotImplementedError(
                f'{inst.name}: sparse weight ROM needs a storage type of '
                f'{sorted(b // 8 for b in SPARSE_ELEMENT_BITS)} bytes, got {np.dtype(rom_dtype)}.'
            )
        if packed_W.shape[-1] % itemsize:
            raise AssertionError(
                f'{inst.name}: compressed ROM is {packed_W.shape[-1]} bytes, not a multiple of the '
                f'{itemsize}-byte weight storage type.'
            )
        packed_W = packed_W.view(rom_dtype)

        packed_B = None
        if b is not None:
            packed_B = pack_vector_by_n_slice(
                b,
                N=n_out,
                N_slice=output_view.tile_inner,
                cas_num=1 if n_slice_scale > 1 else cas_num,
                dtype=np_bias_dtype_for_spec(p.precision['bias']),
            )
            if n_slice_scale > 1:
                packed_B = np.repeat(packed_B, cas_num, axis=0)

        # Report what the model brought, measured on the [K, N] matrix the kernel multiplies --
        # not on anything this pass did to it, because it did nothing but re-encode.
        W_kn = np.asarray(W).reshape((-1, n_in, n_out))[-1]
        total = int(W_kn.size)
        nonzero = int(np.count_nonzero(W_kn))
        report = {
            'mode': '2of4',
            'sparsity_ratio': SPARSITY_RATIO,
            'nonzero': nonzero,
            'elements': total,
            'zero_fraction': (total - nonzero) / total if total else 0.0,
            'is_structured_sparse_2of4': bool(is_structured_sparse_2of4(W_kn)),
            'pruned_by_aie4ml': False,
            'weight_bytes_dense': total * int(W_kn.dtype.itemsize),
            'weight_bytes_sparse': int(packed_W.nbytes),
        }
        return {'packed_weights': packed_W, 'packed_bias': packed_B, 'sparsity_report': report}

    def get_artifacts(self, inst: OpImplInstance):
        artifacts = super().get_artifacts(inst)
        # The compressed ROM is a raw byte stream, not an array of weight_t values. It is handed
        # to the kernel through the same `weight_t (&)[WTS_ELEMS]` parameter, which is why this
        # only works while weight_t is exactly one byte wide (enforced in validate_config).
        for spec in artifacts:
            if spec.get('name') == 'weights':
                spec['sparsity'] = '2of4'
        return artifacts


@register_variant
class DenseSparseOpImplVariant(_DenseSparseVariantBase):
    """Sparse dense with the output features partitioned across cascade chains ('inner')."""

    variant_id = 'dense.sparse.b.r.v1'
    contract = 'inner'

    def kernel_outer_extent(self, lhs_view):
        return lhs_view.compacted_full_outer

    def lhs_port_count(self, config: DenseConfig) -> int:
        return int(config.parallelism.cas_length)

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
        return describe_inner_lhs_staging(config.io_views[tensor_name], port, buf_dims)

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        return describe_inner_output_staging(config.io_views[tensor_name], port, buf_dims)

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        # 'inner': cas_num slices the columns, so chain c owns weight/bias columns
        # [c*N_slice, (c+1)*N_slice) -- exactly what the packers lay out.
        return self._pack_sparse(inst)


@register_variant
class DenseSparseRowWiseOpImplVariant(_DenseSparseVariantBase):
    """Sparse dense with the rows partitioned across cascade chains ('outer')."""

    variant_id = 'dense.sparse.b.r.row.v1'
    contract = 'outer'

    def kernel_outer_extent(self, lhs_view):
        return lhs_view.compacted_tile_outer

    def lhs_port_count(self, config: DenseConfig) -> int:
        return int(config.parallelism.cas_length) * int(config.parallelism.cas_num)

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
        return describe_outer_lhs_staging(config.io_views[tensor_name], config.parallelism, port, buf_dims)

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        return describe_outer_output_staging(config.io_views[tensor_name], port, buf_dims)

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        # 'outer': N_slice is the whole N, so pack a single chain and give every row group that
        # same copy, mirroring DenseRowWiseOpImplVariant.
        return self._pack_sparse(inst, n_slice_scale=int(inst.config.parallelism.cas_num))

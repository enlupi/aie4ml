from __future__ import annotations

from ....aie_types import FloatIntent
from ....ir.graph import OpImplInstance, OpNode, input_role, input_tensor_for_role
from ...base import OpImplFootprint
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from ...utils import ParallelismConfig, parse_directives
from ...utils.precision import (
    aie_rounding_token,
    infer_accumulator_tag,
    resolve_accumulator_output_shift,
)
from .common import (
    bitwidths_supported,
    describe_inner_lhs_staging,
    describe_inner_output_staging,
    describe_inner_rhs_staging,
    describe_outer_lhs_staging,
    describe_outer_output_staging,
    describe_outer_rhs_staging,
    requested_contract,
    requested_sparsity,
)
from .config import MatmulConfig, MatmulFlags
from .dense import _BaseDenseMatmulVariant
from .resolver import (
    _build_matmul_io_views,
    _resolve_numeric,
    _resolve_output_scale_shift,
    _resolve_parallelism,
    _resolve_tile_cfg,
)


class _MatmulVariantBase(_BaseDenseMatmulVariant):
    """Everything the matmul variants share, independent of which contract they implement."""

    op_type = 'matmul'
    graph_header = 'matmul_graph.h'
    graph_name = 'matmul_graph'
    param_template = 'matmul'
    plevel = 10

    def matches(self, node: OpNode, device) -> bool:
        if requested_sparsity(node) == '2of4':
            # No sparse matmul variant exists: the sparse path needs its B operand packed into
            # the compressed layout ahead of time, which only works for a kernel-resident weight
            # ROM (dense), not a dynamically streamed B. An explicit request fails here rather
            # than quietly running dense and ignoring the directive. The 'auto' default is
            # silent, because a dynamic matmul simply never qualifies.
            raise NotImplementedError(
                f'{node.name}: sparsity is only supported for the dense family (kernel-resident '
                'weights); a dynamic matmul streams its B operand at runtime, so it cannot be '
                'pre-compressed. Remove the sparsity directive or fold the operand into a Dense.'
            )
        return requested_contract(node) == self.contract and bitwidths_supported(node, device)

    def resolve(self, node: OpNode, device, directives=None) -> MatmulConfig:
        io_route, _, _ = parse_directives(directives)
        precision = _resolve_numeric(node, device)
        microtiling = _resolve_tile_cfg(node, device, precision['lhs'], precision['rhs'], variant=self)
        tiling = _resolve_parallelism(node, device, microtiling, precision, self.contract)
        io_views = _build_matmul_io_views(node, microtiling, tiling)

        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        lhs_perm = io_views[lhs_tensor.name].perm
        rhs_perm = io_views[rhs_tensor.name].perm
        is_float = isinstance(rhs_tensor.precision, FloatIntent)

        shift = (
            0
            if is_float
            else resolve_accumulator_output_shift(lhs_tensor.precision, node.outputs[0].precision, rhs_tensor.precision)
        )
        shift += _resolve_output_scale_shift(node, is_float=is_float)

        return MatmulConfig(
            precision=precision,
            parallelism=ParallelismConfig(
                cas_length=tiling.cas_length, cas_num=tiling.cas_num, contract=tiling.contract
            ),
            microtiling=microtiling,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag=infer_accumulator_tag(device, None, None, precision['acc']),
            rounding_mode='conv_even' if is_float else aie_rounding_token(precision['output']),
            flags=MatmulFlags(
                transpose_lhs=bool(lhs_perm is not None and lhs_perm[-1] != (len(lhs_perm) - 1)),
                transpose_rhs=bool(rhs_perm is not None and rhs_perm[-1] != (len(rhs_perm) - 1)),
            ),
        )

    def validate_config(self, node: OpNode, config: MatmulConfig, _device) -> None:
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        rhs_view = config.io_views[rhs_tensor.name]
        rhs_perm = rhs_view.perm
        if rhs_perm is not None:
            rank = len(rhs_perm)
            identity = list(range(rank))
            swapped = list(range(rank))
            if rank >= 2:
                swapped[-2], swapped[-1] = swapped[-1], swapped[-2]
            if list(rhs_perm) not in [identity, swapped]:
                raise ValueError(f'{node.name}: matmul RHS does not support io_view permutation {rhs_perm}.')
        rhs_is_float = isinstance(rhs_tensor.precision, FloatIntent)
        if not rhs_is_float and not bool(config.precision['rhs'].signed):
            raise ValueError(f'{node.name}: matmul RHS must use a signed integer precision.')

    def pack(self, _inst: OpImplInstance):
        return {}

    def get_artifacts(self, _inst: OpImplInstance):
        return []

    def footprint(self, _node, config) -> OpImplFootprint:
        return OpImplFootprint(
            width=config.parallelism.cas_length,
            height=config.parallelism.cas_num,
            extras={'keepout_left': 1},
        )

    def build_ports(self, node: OpNode, config: MatmulConfig):
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        return PortMap(
            inputs={
                lhs_tensor.name: PortBinding(group='inA', count=self.lhs_port_count(config)),
                rhs_tensor.name: PortBinding(
                    group='inB',
                    count=int(config.parallelism.cas_length) * int(config.parallelism.cas_num),
                ),
            },
            outputs={node.outputs[0].name: PortBinding(group='outC', count=int(config.parallelism.cas_num))},
        )


@register_variant
class MatmulOpImplVariant(_MatmulVariantBase):
    """Matmul with the output features partitioned across cascade chains ('inner' contract)."""

    variant_id = 'matmul.v1'
    contract = 'inner'

    def kernel_outer_extent(self, lhs_view):
        return lhs_view.compacted_full_outer

    def lhs_port_count(self, config: MatmulConfig) -> int:
        # One lhs slice per cascade column, multicast across every chain.
        return int(config.parallelism.cas_length)

    def describe_input_staging(self, node, config, tensor_name, port, buf_dims=None, _producer=None):
        view = config.io_views[tensor_name]
        if input_role(node, tensor_name) == 'rhs':
            return describe_inner_rhs_staging(view, config.parallelism, port, buf_dims)
        return describe_inner_lhs_staging(view, port, buf_dims)

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        return describe_inner_output_staging(config.io_views[tensor_name], port, buf_dims)


@register_variant
class MatmulRowWiseOpImplVariant(_MatmulVariantBase):
    """Matmul with the rows partitioned across cascade chains ('outer' contract)."""

    variant_id = 'matmul.row.v1'
    contract = 'outer'

    def kernel_outer_extent(self, lhs_view):
        return lhs_view.compacted_tile_outer

    def lhs_port_count(self, config: MatmulConfig) -> int:
        # Every (chain, column) tile reads its own row slice, so the lhs port array is per-tile.
        return int(config.parallelism.cas_length) * int(config.parallelism.cas_num)

    def describe_input_staging(self, node, config, tensor_name, port, buf_dims=None, _producer=None):
        view = config.io_views[tensor_name]
        if input_role(node, tensor_name) == 'rhs':
            return describe_outer_rhs_staging(view, config.parallelism, port, buf_dims)
        return describe_outer_lhs_staging(view, config.parallelism, port, buf_dims)

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        return describe_outer_output_staging(config.io_views[tensor_name], port, buf_dims)

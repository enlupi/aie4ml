# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AIE-ML hardware sparse (2:4) dense variants.

The weights decide. Every Dense layer is inspected, and one whose *trained* weights are already
2:4-structured along the reduction axis is compiled sparse with no directive at all; one that is
not is compiled dense with its numerics untouched. `sparsity: {'mode': 'dense'}` is the opt-out
consulted afterwards, and `{'mode': '2of4'}` is the explicit demand that errors instead of
falling back. aie4ml never prunes -- there is no code here that manufactures the pattern -- so
these tests are largely about the detection gate, the two ways an automatic feature can
misbehave (taking a node it cannot support, breaking a graph that used to build), and the
losslessness of the re-encoding.

`_build` therefore defaults to 2:4-structured weights, as if from a sparsity-aware training run;
`_weights_dense` and `_weights_unstructured_sparse` are the fixtures for the layers that must
*not* go sparse.

The IR-level tests need no toolchain. The differential simulation test is marked
`requires_vitis`: a wrong compressed byte layout compiles and simulates cleanly while silently
producing wrong numbers, so it is the only check that actually pins the format down.
"""

import os
from pathlib import Path

import pytest
from aie_platform import resolve_aie_part

pytestmark = pytest.mark.aie_ir

# Deduced from the installed Vitis rather than hardcoded -- see tests/aie_platform.py.
PART = resolve_aie_part()
IN_FEAT, OUT_FEAT, BATCH = 128, 256, 8
W_FRAC, X_FRAC, Y_FRAC = 6, 5, 3
SPARSE_MICROTILE = (4, 16, 8)


def _require_vitis():
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')


def _qi(width, frac, signed=True):
    from aie4ml.aie_types import QuantIntent, RoundingMode, SaturationMode

    return QuantIntent(
        width=width, frac=frac, signed=signed, rounding=RoundingMode.RND_CONV, saturation=SaturationMode.SAT
    )


def _weights_2of4(seed=7, in_feat=IN_FEAT, out_feat=OUT_FEAT):
    """Weights as a 2:4-structured sparsity-aware training run would hand them over.

    Exactly 2 nonzeros in every group of 4 consecutive input features, per output column -- the
    grouping the compressed column-major format uses. Magnitudes stay well clear of the W_FRAC
    quantization step so no kept value rounds to zero and changes the pattern under the compiler.
    """
    np = pytest.importorskip('numpy')
    rng = np.random.default_rng(seed)
    groups = in_feat // 4
    within = np.argsort(rng.random((groups, 4, out_feat)), axis=1)[:, :2, :]
    mask = np.zeros((groups, 4, out_feat), dtype=bool)
    np.put_along_axis(mask, within, True, axis=1)
    values = rng.uniform(0.05, 0.5, size=(in_feat, out_feat)) * rng.choice([-1.0, 1.0], size=(in_feat, out_feat))
    return np.where(mask.reshape(in_feat, out_feat), values, 0.0)


def _weights_dense(seed=3, in_feat=IN_FEAT, out_feat=OUT_FEAT):
    """An ordinary trained layer: no zeros to speak of, so nothing for the sparse path to use."""
    np = pytest.importorskip('numpy')
    return np.random.default_rng(seed).normal(0, 0.2, size=(in_feat, out_feat))


def _weights_unstructured_sparse(seed=9, in_feat=IN_FEAT, out_feat=OUT_FEAT):
    """Heavily pruned (>=50% zeros) but *not* 2:4: one group per column holds 3 nonzeros.

    This is the case worth warning about -- the model paid for sparsity in training and the
    hardware cannot use it, because the zeros are not laid out in groups of four along K.
    """
    np = pytest.importorskip('numpy')
    rng = np.random.default_rng(seed)
    groups = in_feat // 4
    within = np.argsort(rng.random((groups, 4, out_feat)), axis=1)[:, :1, :]
    mask = np.zeros((groups, 4, out_feat), dtype=bool)
    np.put_along_axis(mask, within, True, axis=1)
    mask[0, :, :] = False
    mask[0, :3, :] = True  # exactly one group per column breaks the 2-per-4 budget
    values = rng.uniform(0.05, 0.5, size=(in_feat, out_feat)) * rng.choice([-1.0, 1.0], size=(in_feat, out_feat))
    return np.where(mask.reshape(in_feat, out_feat), values, 0.0)


def _build(
    tmp_path,
    *,
    sparsity=None,
    contract=None,
    microtile=None,
    parallelism=None,
    weights=None,
    part=PART,
    in_feat=None,
    out_feat=None,
    act_width=8,
    wt_width=8,
):
    """Lower a single quantized Dense layer to ExecutionIR and return (ctx, model).

    `act_width` sets the activation storage width on both the input and the output, `wt_width` the
    weight storage width; the sparse packer implements 8 and 16. `in_feat`/`out_feat` override the
    module defaults so a shape whose slices do not divide evenly can be exercised.
    """
    np = pytest.importorskip('numpy')
    from aie4ml.frontends.common import register_default_traits
    from aie4ml.frontends.onnx.utils import create_context
    from aie4ml.ir.graph import OpNode, TensorVar, set_input_roles
    from aie4ml.model import AIEModel

    ctx = create_context(
        {'Part': part, 'AIEConfig': {'BatchSize': BATCH, 'Iterations': 1}}, tmp_path, 'sparse_probe', None, None
    )
    graph = ctx.ir.logical
    rng = np.random.default_rng(3)
    n_in = IN_FEAT if in_feat is None else int(in_feat)
    n_out = OUT_FEAT if out_feat is None else int(out_feat)
    # Default to sparsity-aware-trained weights: the sparse path is now weight-driven, so a
    # random dense matrix would send every test down the dense variant instead.
    W = _weights_2of4(in_feat=n_in, out_feat=n_out) if weights is None else weights
    b = rng.normal(0, 0.1, size=(n_out,))

    x = TensorVar(name='x', shape=(BATCH, n_in), precision=_qi(act_width, X_FRAC))
    graph.add_tensor(x)
    graph.mark_graph_input('x')
    graph.add_tensor(TensorVar(name='w', shape=W.shape, precision=_qi(wt_width, W_FRAC), data=W))
    graph.add_tensor(TensorVar(name='bi', shape=b.shape, precision=_qi(wt_width, W_FRAC), data=b))
    wt, bt = graph.tensors['w'], graph.tensors['bi']

    op = OpNode(name='d0_aie', op_type='dense', dialect=ctx.device.dialect)
    op.metadata.update(
        {
            'n_in': n_in,
            'n_out': n_out,
            'use_bias': True,
            'layer_class': 'Dense',
            'source_class': 'Gemm',
            'source_layer': 'd0',
            'input_roles': ['lhs', 'rhs', 'bias'],
        }
    )
    if sparsity:
        op.directives['sparsity'] = {'mode': sparsity}
    if contract or parallelism:
        op.directives['parallelism'] = {**({'contract': contract} if contract else {}), **(parallelism or {})}
    if microtile:
        op.directives['microtiling'] = dict(zip(('microtile_m', 'microtile_k', 'microtile_n'), microtile))

    for src in (x, wt, bt):
        src.consumers.append(op)
    op.inputs.extend([x, wt, bt])
    y = TensorVar(name='y', shape=(BATCH, n_out), precision=_qi(act_width, Y_FRAC), producer=op)
    graph.add_tensor(y)
    op.outputs.append(y)
    graph.add_node(op)
    graph.mark_graph_output('y')
    set_input_roles(op, op.inputs, ['lhs', 'rhs', 'bias'])

    register_default_traits(ctx)
    model = AIEModel.from_context(ctx)
    model.run_pipeline()
    return ctx, model


def _entry(ctx):
    return next(iter(ctx.ir.execution.instances.values()))


def test_already_2of4_layer_goes_sparse_with_no_directive(tmp_path):
    """The default: a layer whose trained weights are already 2:4 is compiled sparse."""
    ctx, _ = _build(tmp_path)
    entry = _entry(ctx)
    assert entry.variant.variant_id == 'dense.sparse.b.r.v1'
    assert entry.variant.param_template == 'dense_bias_relu_sparse'
    assert entry.config.sparsity == '2of4'
    mt = entry.config.microtiling
    assert (mt.microtile_m, mt.microtile_k, mt.microtile_n) == SPARSE_MICROTILE


def test_dense_weights_compile_dense_without_a_word(tmp_path, caplog):
    """An ordinary layer is not sparse, so it is not a sparse layer. Nothing to report."""
    import logging

    with caplog.at_level(logging.DEBUG):
        ctx, _ = _build(tmp_path, weights=_weights_dense())
    entry = _entry(ctx)
    assert entry.variant.variant_id == 'dense.b.r.v1'
    assert entry.config.sparsity == 'dense'
    assert not [r for r in caplog.records if 'sparse' in r.getMessage().lower()]


def test_unstructured_sparse_weights_compile_dense_and_warn(tmp_path, caplog):
    """>=50% zeros in the wrong layout: dense, but say so -- the trained saving is being lost."""
    import logging

    with caplog.at_level(logging.WARNING):
        ctx, _ = _build(tmp_path, weights=_weights_unstructured_sparse())

    entry = _entry(ctx)
    assert entry.variant.variant_id == 'dense.b.r.v1'
    assert entry.config.sparsity == 'dense'

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any('not 2:4-structured' in m and 'compiled dense' in m for m in warnings)
    assert any('d0_aie' in m and 'holds up to 3 nonzeros' in m for m in warnings)
    # And it is said once, not once per candidate variant asked.
    assert len([m for m in warnings if 'not 2:4-structured' in m]) == 1


def test_explicit_2of4_on_weights_that_are_not_sparse_is_an_error(tmp_path):
    """aie4ml will not prune to satisfy the directive; it says why and stops."""
    with pytest.raises(ValueError, match='not 2:4-structured'):
        _build(tmp_path, sparsity='2of4', weights=_weights_dense())

    with pytest.raises(ValueError, match='does not prune|sparsity-aware training'):
        _build(tmp_path, sparsity='2of4', weights=_weights_unstructured_sparse())


def test_aie4ml_has_no_pruning_entry_point():
    """Guards the boundary: creating the pattern is training's job, not the compiler's."""
    import aie4ml.op_impls.families.matmul as matmul
    from aie4ml.op_impls.families.matmul import common

    assert not hasattr(matmul, 'prune_to_2of4_along_k')
    assert not [name for name in dir(common) if 'prune' in name.lower()]


def test_dense_directive_opts_out(tmp_path):
    """`sparsity: {'mode': 'dense'}` forces the dense path on a layer that qualifies."""
    ctx, _ = _build(tmp_path, sparsity='dense')
    entry = _entry(ctx)
    assert entry.variant.variant_id == 'dense.b.r.v1'
    assert entry.config.sparsity == 'dense'
    assert entry.config.sparse_tile_bytes == 0
    # It must also get the *dense* microtile table, not the sparse one. Asking the resolved
    # variant for its own options is what keeps these separate.
    mt = entry.config.microtiling
    assert (mt.microtile_m, mt.microtile_k, mt.microtile_n) != SPARSE_MICROTILE


def test_opting_out_skips_the_weight_inspection_entirely(tmp_path):
    """`mode: 'dense'` settles the outcome, so nothing walks the weight matrix.

    The check order is an efficiency contract, not just an implementation detail: a caller who
    opts out has asserted they know their model, and pays nothing for the analysis.
    """
    ctx, _ = _build(tmp_path, sparsity='dense')
    node = _entry(ctx).node
    assert 'weight_sparsity' not in node.artifacts

    # ... whereas a layer left to `auto` is measured, and the measurement is cached on the node.
    ctx, _ = _build(tmp_path / 'auto')
    assert _entry(ctx).node.artifacts['weight_sparsity']['structured_2of4']


def test_explicit_2of4_still_selects_sparse(tmp_path):
    ctx, _ = _build(tmp_path, sparsity='2of4')
    assert _entry(ctx).variant.variant_id == 'dense.sparse.b.r.v1'


def test_outer_contract_selects_row_wise_sparse_variant(tmp_path):
    ctx, _ = _build(tmp_path, contract='outer')
    assert _entry(ctx).variant.variant_id == 'dense.sparse.b.r.row.v1'


def test_ineligible_layer_falls_back_silently(tmp_path):
    """An automatic feature must never break a graph. No directive, no error, no sparsity."""
    from aie4ml.aie_types import FloatFormat, FloatIntent

    ctx = _build_float(tmp_path, FloatIntent(width=16, format=FloatFormat.BF16), sparsity=None)
    entry = _entry(ctx)
    assert entry.variant.variant_id == 'dense.b.r.v1'
    assert entry.config.sparsity == 'dense'


def test_pinned_non_sparse_microtile_falls_back_silently(tmp_path):
    """A microtiling directive with no sparse overload disqualifies the node rather than losing."""
    ctx, _ = _build(tmp_path, microtile=(4, 8, 8))
    entry = _entry(ctx)
    assert entry.variant.variant_id == 'dense.b.r.v1'
    mt = entry.config.microtiling
    assert (mt.microtile_m, mt.microtile_k, mt.microtile_n) == (4, 8, 8)


def test_dynamic_matmul_unaffected_by_the_auto_default(tmp_path):
    """A dynamic matmul can never be sparse, so `auto` must not raise on one.

    Guards a specific regression: while sparsity was opt-in, `MatmulOpImplVariant.matches()`
    raised for any non-dense mode. With 'auto' as the default that would have failed every
    dynamic matmul in every model.
    """
    from aie4ml.op_impls.families.matmul import MatmulOpImplVariant, requested_sparsity

    ctx, _ = _build(tmp_path, sparsity='dense')
    node = _entry(ctx).node
    node.directives.pop('sparsity', None)
    assert requested_sparsity(node) == 'auto'
    # Must return a bool, not raise. An explicit request still raises, below.
    assert MatmulOpImplVariant().matches(node, ctx.device) in (True, False)

    node.directives['sparsity'] = {'mode': '2of4'}
    with pytest.raises(NotImplementedError, match='only supported for the dense family'):
        MatmulOpImplVariant().matches(node, ctx.device)


def test_packed_rom_is_smaller_and_the_report_claims_no_pruning(tmp_path):
    np = pytest.importorskip('numpy')
    from aie4ml.op_impls.families.matmul import sparse_tile_bytes

    ctx, _ = _build(tmp_path, sparsity='2of4')
    entry = _entry(ctx)
    packed = entry.artifacts['packed_weights']
    report = entry.artifacts['sparsity_report']

    tile_bytes = sparse_tile_bytes(*SPARSE_MICROTILE[1:])
    assert tile_bytes == 80  # (16*8)/2 kept bytes + (16*8)/8 mask bytes
    cas_num, cas_length = entry.config.parallelism.cas_num, entry.config.parallelism.cas_length
    assert packed.shape[:2] == (cas_num, cas_length)
    assert np.dtype(packed.dtype).itemsize == 1

    # 0.625 of dense: 5 bytes per 8 weights (4 kept values + 1 mask byte).
    assert report['weight_bytes_sparse'] < report['weight_bytes_dense']
    assert report['weight_bytes_sparse'] == report['weight_bytes_dense'] * 5 // 8
    assert report['is_structured_sparse_2of4']
    assert report['pruned_by_aie4ml'] is False
    # The model arrived exactly 2:4, and re-encoding it kept every nonzero.
    assert report['nonzero'] == report['elements'] // 2
    assert report['zero_fraction'] == 0.5


def _decode_rom(entry):
    """Decode a whole packed ROM back to a (padded) [K, N] matrix, independently of the packer.

    Mirrors nothing from `common.py` on purpose: mask bits LSB-first over a column-major
    microtile, tiles ordered by (n, n+1) pair then K-tile then column of the pair. Slices come
    from the resolved io_views rather than from K/N, so a shape whose slices are padded past the
    real matrix decodes to a padded array the caller can check the edges of.

    Under the 'outer' contract every cascade chain holds the *whole* of N (`cas_num` splits rows,
    not columns), so the packer builds one chain and replicates it -- hence `n_base` is 0 for
    every chain there, and all chains must be byte-identical.
    """
    np = pytest.importorskip('numpy')
    from aie4ml.op_impls.families.matmul import sparse_tile_bytes

    cfg = entry.config
    packed = np.asarray(entry.artifacts['packed_weights']).view(np.uint8)
    microtile_k, microtile_n = cfg.microtiling.microtile_k, cfg.microtiling.microtile_n
    cas_num, cas_length = cfg.parallelism.cas_num, cfg.parallelism.cas_length
    outer = cfg.parallelism.contract == 'outer'

    k_slice = int(cfg.io_views[entry.node.inputs[0].name].tile_inner)
    n_slice = int(cfg.io_views[entry.node.outputs[0].name].tile_inner)
    element_bits = int(cfg.precision['rhs'].width)
    tile_bytes = sparse_tile_bytes(microtile_k, microtile_n, element_bits)
    tiles_per_k, tiles_per_n = k_slice // microtile_k, n_slice // microtile_n

    if outer:
        for chain in range(1, cas_num):
            np.testing.assert_array_equal(packed[chain], packed[0])

    decoded = np.zeros((cas_length * k_slice, n_slice if outer else cas_num * n_slice), dtype=np.int64)
    for chain in range(cas_num):
        for cas in range(cas_length):
            blob, off = packed[chain, cas], 0
            for pair in range(tiles_per_n // 2):
                for kt in range(tiles_per_k):
                    for nt in (2 * pair, 2 * pair + 1):
                        tile = _decode_sparse_tile(
                            blob[off : off + tile_bytes], microtile_k, microtile_n, element_bits
                        )
                        off += tile_bytes
                        k0 = cas * k_slice + kt * microtile_k
                        n0 = (0 if outer else chain * n_slice) + nt * microtile_n
                        decoded[k0 : k0 + microtile_k, n0 : n0 + microtile_n] = tile
            assert off == blob.size, f'chain {chain} stage {cas}: decoded {off} of {blob.size} bytes'
    return decoded


@pytest.mark.parametrize(
    'contract,in_feat,out_feat,act_width,wt_width',
    [
        # cas_num=2: N split across two chains, K whole.
        ('inner', 128, 256, 8, 8),
        # cas_length=2: K split across a cascade -- the reduction axis the 2:4 groups live on,
        # so a group straddling a stage boundary would corrupt exactly here.
        ('outer', 128, 256, 8, 8),
        # Neither axis divides its slice: k_slice=160 > K and 3*96=288 > N, so most edge tiles
        # are partly or wholly zero-fill.
        ('inner', 132, 260, 8, 8),
        # The same, on the row-wise variant.
        ('outer', 132, 72, 8, 8),
        # int16 activations pick the (2,16,8) microtile instead of (4,16,8).
        ('inner', 128, 256, 16, 8),
        # int16 *weights*: the (2,8,8) microtile, two mask bits per sample, and a ROM that is
        # reinterpreted as int16 rather than int8 -- the three things a width change touches.
        ('inner', 128, 256, 16, 16),
        ('outer', 128, 256, 16, 16),
    ],
    ids=[
        'inner_split_n',
        'outer_cascade_k',
        'inner_padded_edges',
        'outer_padded_edges',
        'int16_acts',
        'int16_weights_inner',
        'int16_weights_outer',
    ],
)
def test_packed_rom_decodes_back_to_the_models_own_weights(
    tmp_path, contract, in_feat, out_feat, act_width, wt_width
):
    """The central guarantee, across every packing geometry: storage changes, numerics do not.

    Also pins the zero-fill: everything past the real [K, N] must decode to zero, because the
    kernel reads the padded slice and a stray nonzero there would silently contribute to a dot
    product.
    """
    np = pytest.importorskip('numpy')
    from aie4ml.op_impls.families.matmul import quantized_weight_matrix

    ctx, _ = _build(
        tmp_path,
        sparsity='2of4',
        contract=contract,
        in_feat=in_feat,
        out_feat=out_feat,
        act_width=act_width,
        wt_width=wt_width,
    )
    entry = _entry(ctx)
    assert entry.config.sparsity == '2of4'
    assert int(entry.config.precision['rhs'].width) == wt_width

    decoded = _decode_rom(entry)
    expected = quantized_weight_matrix(entry.node).astype(np.int64)
    assert expected.shape == (in_feat, out_feat)

    np.testing.assert_array_equal(decoded[:in_feat, :out_feat], expected)
    # The padding, which the kernel does read.
    assert not decoded[in_feat:, :].any(), 'K padding decoded to a nonzero'
    assert not decoded[:, out_feat:].any(), 'N padding decoded to a nonzero'


def test_outer_contract_cascades_the_reduction_axis(tmp_path):
    """The 'outer' contract splits K across a cascade, which is where 2:4 groups are at risk.

    `_resolve_parallelism` aligns the K slice to `2 * microtile_k`, so a group of four never
    straddles a stage boundary -- assert the geometry that guarantee depends on actually holds,
    rather than trusting the alignment silently.
    """
    ctx, _ = _build(tmp_path, contract='outer')
    entry = _entry(ctx)
    cfg = entry.config

    assert entry.variant.variant_id == 'dense.sparse.b.r.row.v1'
    assert cfg.parallelism.cas_length > 1, 'expected a real K cascade to exercise'
    k_slice = int(cfg.io_views[entry.node.inputs[0].name].tile_inner)
    assert k_slice % cfg.microtiling.microtile_k == 0
    assert cfg.microtiling.microtile_k % 4 == 0
    # Every chain holds the whole of N under 'outer', so the ROM is a replication.
    assert int(cfg.io_views[entry.node.outputs[0].name].tile_inner) == OUT_FEAT


def test_padded_shapes_still_go_sparse_and_size_the_rom_from_the_slices(tmp_path):
    """A shape whose slices overshoot K and N is packed, not rejected."""
    from aie4ml.op_impls.families.matmul import sparse_tile_bytes

    ctx, _ = _build(tmp_path, in_feat=132, out_feat=260)
    entry = _entry(ctx)
    cfg = entry.config
    assert entry.variant.variant_id == 'dense.sparse.b.r.v1'

    k_slice = int(cfg.io_views[entry.node.inputs[0].name].tile_inner)
    n_slice = int(cfg.io_views[entry.node.outputs[0].name].tile_inner)
    assert k_slice * cfg.parallelism.cas_length >= 132, 'K should be padded up, not truncated'
    assert n_slice * cfg.parallelism.cas_num >= 260

    tiles = (k_slice // cfg.microtiling.microtile_k) * (n_slice // cfg.microtiling.microtile_n)
    expected_bytes = tiles * sparse_tile_bytes(cfg.microtiling.microtile_k, cfg.microtiling.microtile_n)
    assert entry.artifacts['packed_weights'].shape[-1] == expected_bytes


def test_int16_activations_use_the_int16_sparse_microtile(tmp_path):
    """int16 x int8 is a distinct row of SPARSE_MICROTILE_OPTIONS, with a narrower M."""
    from aie4ml.op_impls.families.matmul import SPARSE_MICROTILE_OPTIONS

    ctx, _ = _build(tmp_path, act_width=16)
    entry = _entry(ctx)
    mt = entry.config.microtiling
    shape = (mt.microtile_m, mt.microtile_k, mt.microtile_n)

    assert entry.variant.variant_id == 'dense.sparse.b.r.v1'
    assert entry.config.precision['lhs'].width == 16
    assert entry.config.precision['rhs'].width == 8
    assert shape in SPARSE_MICROTILE_OPTIONS['AIE-ML'][('int16', 'int8')]
    assert shape != SPARSE_MICROTILE, 'int16 activations must not reuse the int8 microtile'


def test_int16_weights_use_the_int16x16_sparse_microtile(tmp_path):
    """int16 weights are a distinct row again, with a narrower K -- and a 2-byte weight ROM."""
    from aie4ml.op_impls.families.matmul import SPARSE_MICROTILE_OPTIONS, sparse_tile_bytes

    ctx, _ = _build(tmp_path, act_width=16, wt_width=16)
    entry = _entry(ctx)
    mt = entry.config.microtiling
    shape = (mt.microtile_m, mt.microtile_k, mt.microtile_n)

    assert entry.variant.variant_id == 'dense.sparse.b.r.v1'
    assert entry.config.precision['rhs'].width == 16
    assert shape in SPARSE_MICROTILE_OPTIONS['AIE-ML'][('int16', 'int16')]
    # microtile_k drops from 16 to 8, but the compressed tile is still 80 bytes: the halved
    # element count is exactly offset by the doubled element width.
    assert entry.config.sparse_tile_bytes == sparse_tile_bytes(mt.microtile_k, mt.microtile_n, 16) == 80
    # The ROM is handed over as weight_t, so it must be an int16 array of half as many elements
    # as it has bytes -- and WTS_ELEMS has to agree or the ADF parameter port is mis-sized.
    packed = entry.artifacts['packed_weights']
    assert packed.dtype.itemsize == 2
    assert packed.nbytes == packed.size * 2


def test_aie_mlv2_has_no_mixed_dtype_sparse_pairing():
    """AIE-MLv2 sparse needs both operands in the same dtype, so int16 x int8 must not be offered.

    UG1603: "For AIE-ML v2, both matrices should have the same data type: int8xint8, int16xint16,
    or bfloat16xbfloat16." The AI Engine API sparse modes table agrees -- its AIE-MLv2 row leaves
    the 16b x 8b cell empty, and the 8x16x8 shape that once sat here belongs to the neighbouring
    8b x 8b column.

    Pinned as a table assertion rather than a lowering test because the suite deliberately targets
    Vek280 only (see tests/aie_platform.py), and because this is the shape of mistake that comes
    back: a wrong row here selects a sparse kernel that resolves and validates cleanly, then fails
    in `v++` -- too late for the speculative-resolve fallback to keep the build alive.
    """
    from aie4ml.op_impls.families.matmul import SPARSE_MICROTILE_OPTIONS

    mlv2 = SPARSE_MICROTILE_OPTIONS['AIE-MLV2']
    assert ('int16', 'int8') not in mlv2
    assert ('int8', 'int16') not in mlv2
    for lhs_fmt, rhs_fmt in mlv2:
        assert lhs_fmt == rhs_fmt, f'AIE-MLv2 sparse pairing {(lhs_fmt, rhs_fmt)} is mixed-dtype'


def _decode_sparse_tile(blob, microtile_k, microtile_n, element_bits=8):
    """Independent reference decoder for one compressed (K, N) microtile.

    Written from UG1603 and me_native.h rather than from the encoder, so a shared misreading
    cannot cancel out. The mask is one bit per *byte* of uncompressed data, so an int8 sample owns
    one mask bit and an int16 sample owns two adjacent ones; a sample is stored iff its first bit
    is set. Each chunk covers SPARSE_CHUNK_DATA_BYTES of original data as an 8-byte mask plus a
    constant SPARSE_CHUNK_DATA_BYTES/2 data bytes.
    """
    np = pytest.importorskip('numpy')
    from aie4ml.op_impls.families.matmul.common import SPARSE_CHUNK_DATA_BYTES, sparse_chunk_elems

    itemsize = element_bits // 8
    chunk_elems = sparse_chunk_elems(element_bits)
    mask_bytes = SPARSE_CHUNK_DATA_BYTES // 8
    flat = np.zeros(microtile_k * microtile_n, dtype=np.int64)
    off = 0
    for chunk in range(flat.size // chunk_elems):
        mask = blob[off : off + mask_bytes]
        off += mask_bytes
        taken = 0
        for local in range(chunk_elems):
            bit = local * itemsize
            if not ((int(mask[bit // 8]) >> (bit % 8)) & 1):
                continue
            lo = off + taken * itemsize
            raw = int.from_bytes(bytes(blob[lo : lo + itemsize]), 'little')
            if raw >= 1 << (element_bits - 1):
                raw -= 1 << element_bits
            flat[chunk * chunk_elems + local] = raw
            taken += 1
        assert taken * itemsize == SPARSE_CHUNK_DATA_BYTES // 2, 'every group of 4 must keep exactly 2'
        off += SPARSE_CHUNK_DATA_BYTES // 2
    assert off == len(blob)
    return flat.reshape(microtile_n, microtile_k).T


def _build_two_layer(tmp_path, kinds, *, part=PART):
    """Lower a chain of Dense layers, one per entry of `kinds` ('sparse' or 'dense' weights).

    A single-node graph cannot catch the ways two differently-implemented Dense layers interfere:
    two `param_template`s in one `parameters.h`, two weight ROMs of different sizes, and a
    transport edge between a sparse producer and a dense consumer.
    """
    np = pytest.importorskip('numpy')
    from aie4ml.frontends.common import register_default_traits
    from aie4ml.frontends.onnx.utils import create_context
    from aie4ml.ir.graph import OpNode, TensorVar, set_input_roles
    from aie4ml.model import AIEModel

    ctx = create_context(
        {'Part': part, 'AIEConfig': {'BatchSize': BATCH, 'Iterations': 1}}, tmp_path, 'mixed_probe', None, None
    )
    graph = ctx.ir.logical
    widths = [IN_FEAT] + [OUT_FEAT if i % 2 == 0 else IN_FEAT for i in range(len(kinds))]

    prev = TensorVar(name='x', shape=(BATCH, widths[0]), precision=_qi(8, X_FRAC))
    graph.add_tensor(prev)
    graph.mark_graph_input('x')

    for i, kind in enumerate(kinds):
        n_in, n_out = widths[i], widths[i + 1]
        make = _weights_2of4 if kind == 'sparse' else _weights_dense
        W = make(seed=i, in_feat=n_in, out_feat=n_out)
        graph.add_tensor(TensorVar(name=f'w{i}', shape=W.shape, precision=_qi(8, W_FRAC), data=W))
        graph.add_tensor(TensorVar(name=f'b{i}', shape=(n_out,), precision=_qi(8, W_FRAC), data=np.zeros(n_out)))
        wt, bt = graph.tensors[f'w{i}'], graph.tensors[f'b{i}']

        op = OpNode(name=f'd{i}_aie', op_type='dense', dialect=ctx.device.dialect)
        op.metadata.update(
            {
                'n_in': n_in,
                'n_out': n_out,
                'use_bias': True,
                'layer_class': 'Dense',
                'source_class': 'Gemm',
                'source_layer': f'd{i}',
                'input_roles': ['lhs', 'rhs', 'bias'],
            }
        )
        for src in (prev, wt, bt):
            src.consumers.append(op)
        op.inputs.extend([prev, wt, bt])
        y = TensorVar(name=f'y{i}', shape=(BATCH, n_out), precision=_qi(8, Y_FRAC), producer=op)
        graph.add_tensor(y)
        op.outputs.append(y)
        graph.add_node(op)
        set_input_roles(op, op.inputs, ['lhs', 'rhs', 'bias'])
        prev = y

    graph.mark_graph_output(prev.name)
    register_default_traits(ctx)
    model = AIEModel.from_context(ctx)
    model.run_pipeline()
    return ctx, model


def test_mixed_graph_compiles_each_layer_on_its_own_merits(tmp_path):
    """Per-layer detection: in one graph, the 2:4 layer goes sparse and its neighbour does not."""
    np = pytest.importorskip('numpy')
    from aie4ml.op_impls.families.matmul import quantized_weight_matrix

    ctx, model = _build_two_layer(tmp_path, ['sparse', 'dense'])
    instances = ctx.ir.execution.instances
    assert [i.variant.variant_id for i in instances.values()] == ['dense.sparse.b.r.v1', 'dense.b.r.v1']
    assert [i.param_template for i in instances.values()] == ['dense_bias_relu_sparse', 'dense_bias_relu']

    sparse_entry, dense_entry = list(instances.values())
    assert sparse_entry.config.sparsity == '2of4'
    assert dense_entry.config.sparsity == 'dense'
    assert 'sparsity_report' in sparse_entry.artifacts
    assert 'sparsity_report' not in dense_entry.artifacts

    # The sparse layer's own ROM still decodes to its own weights, with a dense layer downstream.
    decoded = _decode_rom(sparse_entry)
    expected = quantized_weight_matrix(sparse_entry.node).astype(np.int64)
    np.testing.assert_array_equal(decoded[: expected.shape[0], : expected.shape[1]], expected)

    # And both templates survive into one emitted project.
    model.write()
    params = (Path(ctx.project_config.output_dir) / 'src' / 'parameters.h').read_text()
    assert params.count('USE_SPARSE = true') == 1
    assert params.count('USE_SPARSE = false') == 1


def test_all_sparse_graph_gives_every_layer_its_own_rom(tmp_path):
    """Two sparse layers of different shapes must not share or overwrite each other's weights."""
    np = pytest.importorskip('numpy')
    from aie4ml.op_impls.families.matmul import quantized_weight_matrix

    ctx, _ = _build_two_layer(tmp_path, ['sparse', 'sparse'])
    entries = list(ctx.ir.execution.instances.values())
    assert all(e.config.sparsity == '2of4' for e in entries)

    for entry in entries:
        decoded = _decode_rom(entry)
        expected = quantized_weight_matrix(entry.node).astype(np.int64)
        np.testing.assert_array_equal(decoded[: expected.shape[0], : expected.shape[1]], expected)

    roms = [e.artifacts['packed_weights'] for e in entries]
    assert roms[0].shape != roms[1].shape or not np.array_equal(roms[0], roms[1])


def test_emitted_parameters_and_dma_tiling_track_the_microtile(tmp_path):
    """graph_plan.h's L1 input tiling is generated from the microtile, so it cannot drift."""
    ctx, model = _build(tmp_path, sparsity='2of4')
    model.write()
    out = Path(ctx.project_config.output_dir)

    params = (out / 'src' / 'parameters.h').read_text()
    assert 'static constexpr bool USE_SPARSE = true;' in params
    assert 'static constexpr int SPARSE_TILE_BYTES = 80;' in params
    _, k, n = SPARSE_MICROTILE
    assert f'int M = {SPARSE_MICROTILE[0]}, K = {k}, N = {n};' in params

    entry = _entry(ctx)
    wts_elems = entry.artifacts['packed_weights'].shape[-1]
    assert f'static constexpr int WTS_ELEMS = {wts_elems};' in params
    # app.cpp's RTP update count comes from the artifact, not a dense-only product.
    assert str(wts_elems) in (out / 'app.cpp').read_text()

    plan = (out / 'src' / 'graph_plan.h').read_text()
    assert f'.tiling_dimension = {{ {k}, {SPARSE_MICROTILE[0]} }}' in plan
    assert f'.stride={k}, .wrap={IN_FEAT // k}' in plan


def test_float_weights_rejected(tmp_path):
    from aie4ml.aie_types import FloatFormat, FloatIntent

    with pytest.raises(NotImplementedError, match='float precisions'):
        _build_float(tmp_path, FloatIntent(width=16, format=FloatFormat.BF16))


def _build_float(tmp_path, prec, sparsity='2of4'):
    """Same graph as _build but with float precisions throughout."""
    np = pytest.importorskip('numpy')
    from aie4ml.frontends.common import register_default_traits
    from aie4ml.frontends.onnx.utils import create_context
    from aie4ml.ir.graph import OpNode, TensorVar, set_input_roles
    from aie4ml.model import AIEModel

    ctx = create_context({'Part': PART, 'AIEConfig': {'BatchSize': BATCH, 'Iterations': 1}}, tmp_path, 'p', None, None)
    graph = ctx.ir.logical
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.2, size=(IN_FEAT, OUT_FEAT))
    b = rng.normal(0, 0.1, size=(OUT_FEAT,))
    x = TensorVar(name='x', shape=(BATCH, IN_FEAT), precision=prec)
    graph.add_tensor(x)
    graph.mark_graph_input('x')
    graph.add_tensor(TensorVar(name='w', shape=W.shape, precision=prec, data=W))
    graph.add_tensor(TensorVar(name='bi', shape=b.shape, precision=prec, data=b))
    wt, bt = graph.tensors['w'], graph.tensors['bi']
    op = OpNode(name='d0_aie', op_type='dense', dialect=ctx.device.dialect)
    op.metadata.update(
        {
            'n_in': IN_FEAT,
            'n_out': OUT_FEAT,
            'use_bias': True,
            'layer_class': 'Dense',
            'input_roles': ['lhs', 'rhs', 'bias'],
        }
    )
    if sparsity:
        op.directives['sparsity'] = {'mode': sparsity}
    for src in (x, wt, bt):
        src.consumers.append(op)
    op.inputs.extend([x, wt, bt])
    y = TensorVar(name='y', shape=(BATCH, OUT_FEAT), precision=prec, producer=op)
    graph.add_tensor(y)
    op.outputs.append(y)
    graph.add_node(op)
    graph.mark_graph_output('y')
    set_input_roles(op, op.inputs, ['lhs', 'rhs', 'bias'])
    register_default_traits(ctx)
    model = AIEModel.from_context(ctx)
    model.run_pipeline()
    return ctx


def test_non_sparse_microtile_rejected(tmp_path):
    with pytest.raises(ValueError, match='no sparse overload'):
        _build(tmp_path, sparsity='2of4', microtile=(4, 8, 8))


def test_unknown_sparsity_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match='unknown sparsity mode'):
        _build(tmp_path, sparsity='4of8')


def test_speculative_resolve_failure_falls_back_to_dense(tmp_path, monkeypatch, caplog):
    """The safety net for automatic adoption.

    `sparse_eligibility` is a capability check and deliberately does not try to predict
    everything `resolve()` could reject -- notably tile memory, since the sparse microtile has a
    wider K than the dense one. So a resolve failure on an *automatically* selected variant has
    to fall through to the next candidate instead of failing the build. Simulated here by making
    the sparse resolve raise, because no real shape in the scan I ran triggers it.
    """
    import logging

    from aie4ml.op_impls.families.matmul.sparse import _DenseSparseVariantBase

    def _boom(self, node, device, directives=None):
        raise ValueError('simulated: no valid parallelism fits tile memory')

    monkeypatch.setattr(_DenseSparseVariantBase, 'resolve', _boom)
    with caplog.at_level(logging.WARNING):
        ctx, _ = _build(tmp_path)

    entry = _entry(ctx)
    assert entry.variant.variant_id == 'dense.b.r.v1'
    assert entry.config.sparsity == 'dense'
    # And it must say so rather than silently downgrading.
    assert any('selected automatically but could not be resolved' in r.getMessage() for r in caplog.records)


def test_explicit_request_does_not_fall_back(tmp_path, monkeypatch):
    """The same failure with '2of4' asked for explicitly must propagate, not downgrade."""
    from aie4ml.op_impls.families.matmul.sparse import _DenseSparseVariantBase

    def _boom(self, node, device, directives=None):
        raise ValueError('simulated: no valid parallelism fits tile memory')

    monkeypatch.setattr(_DenseSparseVariantBase, 'resolve', _boom)
    with pytest.raises(ValueError, match='simulated'):
        _build(tmp_path, sparsity='2of4')


def test_adoption_is_silent_and_changes_no_weight(tmp_path, caplog):
    """Sparsity-aware-trained weights make the sparse path free -- and quiet."""
    import logging

    np = pytest.importorskip('numpy')
    from aie4ml.op_impls.families.matmul import quantized_weight_matrix

    weights = _weights_2of4(seed=21)
    with caplog.at_level(logging.WARNING):
        ctx, _ = _build(tmp_path, weights=weights)

    entry = _entry(ctx)
    assert entry.variant.variant_id == 'dense.sparse.b.r.v1'
    assert not caplog.records

    # The layer's own weights are untouched: the sparsity the report describes is the model's.
    report = entry.artifacts['sparsity_report']
    quantized = quantized_weight_matrix(entry.node)
    assert report['nonzero'] == int(np.count_nonzero(quantized))
    assert report['elements'] == quantized.size
    np.testing.assert_array_equal(np.asarray(entry.node.inputs[1].data), weights)


def test_eligibility_reasons(tmp_path):
    """sparse_eligibility is total and explains itself -- matches() depends on both."""
    from aie4ml.op_impls.families.matmul import sparse_eligibility

    ctx, _ = _build(tmp_path, sparsity='dense')
    node, device = _entry(ctx).node, ctx.device
    assert sparse_eligibility(node, device) == (True, '')

    # A non-AIE-ML target cannot, whatever else is true.
    class _Gen1:
        generation = 'AIE'
        rows, columns = 4, 4

    eligible, reason = sparse_eligibility(node, _Gen1())
    assert not eligible and 'no sparse aie::mmul support' in reason

    # Runtime weights cannot be pre-compressed.
    rhs = node.inputs[1]
    saved, rhs.data = rhs.data, None
    node.artifacts.pop('weight_sparsity', None)
    try:
        eligible, reason = sparse_eligibility(node, device)
        assert not eligible and 'compile-time constants' in reason
    finally:
        rhs.data = saved
        node.artifacts.pop('weight_sparsity', None)

    # And the decisive one: capable target, capable precisions, wrong weights.
    rhs.data = _weights_unstructured_sparse()
    node.artifacts.pop('weight_sparsity', None)
    try:
        eligible, reason = sparse_eligibility(node, device)
        assert not eligible
        assert 'not 2:4-structured' in reason and 'aie4ml does not prune' in reason
    finally:
        rhs.data = saved
        node.artifacts.pop('weight_sparsity', None)


@pytest.mark.parametrize(
    'element_bits,microtile_k,microtile_n',
    [(8, 16, 8), (16, 8, 8)],
    ids=['int8', 'int16'],
)
def test_mask_and_compress_roundtrip(element_bits, microtile_k, microtile_n):
    """The derived mask and the column-major mask/data layout are mutually consistent.

    Both widths are checked because the mask granularity differs: one bit per int8 sample, two
    adjacent bits per int16 sample, since the mask is defined per *byte* of uncompressed data.
    Both land on the same 80-byte tile, which is a useful coincidence to pin -- it means a width
    mix-up cannot be caught by a size check alone.
    """
    np = pytest.importorskip('numpy')
    from aie4ml.op_impls.families.matmul import is_structured_sparse_2of4, sparse_mask_along_k, sparse_tile_bytes
    from aie4ml.op_impls.families.matmul.common import _compress_sparse_tile

    dtype = np.int8 if element_bits == 8 else np.int16
    scale = 64 if element_bits == 8 else 16384
    values = (_weights_2of4(seed=11, in_feat=microtile_k, out_feat=microtile_n) * scale).astype(dtype)
    kept = sparse_mask_along_k(values)

    assert is_structured_sparse_2of4(values)
    # Exactly 2 stored per group of 4 down K, which is what makes the tile size constant.
    assert kept.reshape(microtile_k // 4, 4, microtile_n).sum(axis=1).max() == 2
    assert kept.sum() == (microtile_k // 4) * 2 * microtile_n

    blob = _compress_sparse_tile(values, kept, element_bits=element_bits)
    assert len(blob) == sparse_tile_bytes(microtile_k, microtile_n, element_bits) == 80

    decoded = _decode_sparse_tile(blob, microtile_k, microtile_n, element_bits)
    np.testing.assert_array_equal(decoded, values)


def test_no_sparsity_warning_on_a_target_without_a_sparse_path(tmp_path, caplog):
    """The warning is advice about the weights; on AIE1 there is nothing to advise."""
    import logging

    from aie4ml.op_impls.families.matmul import report_unusable_sparsity

    ctx, _ = _build(tmp_path, sparsity='dense', weights=_weights_unstructured_sparse())
    node = _entry(ctx).node

    class _Gen1:
        generation = 'AIE'
        rows, columns = 4, 4

    with caplog.at_level(logging.WARNING):
        report_unusable_sparsity(node, _Gen1())
    assert not caplog.records

    # On a capable target the same node does get told.
    with caplog.at_level(logging.WARNING):
        report_unusable_sparsity(node, ctx.device)
    assert any('not 2:4-structured' in r.getMessage() for r in caplog.records)


def test_mask_refuses_to_thin_an_overfull_group():
    """The packer's last line of defence: it encodes, it does not decide what to drop."""
    np = pytest.importorskip('numpy')
    from aie4ml.op_impls.families.matmul import sparse_mask_along_k

    tile = np.zeros((16, 8), dtype=np.int8)
    tile[:3, 0] = [7, -8, 9]  # three nonzeros in one group of four
    with pytest.raises(ValueError, match='not 2:4-structured'):
        sparse_mask_along_k(tile)


def test_mask_tops_up_sparser_groups_with_stored_zeros():
    """A group with fewer than 2 nonzeros still stores 2 slots, so tiles stay a constant size."""
    np = pytest.importorskip('numpy')
    from aie4ml.op_impls.families.matmul import sparse_mask_along_k

    tile = np.zeros((16, 8), dtype=np.int8)
    tile[1, 0] = 5  # a single nonzero; the rest of the matrix is all zero
    kept = sparse_mask_along_k(tile)

    assert kept.reshape(4, 4, 8).sum(axis=1).max() == 2
    assert kept.sum() == (16 // 4) * 2 * 8
    assert kept[1, 0]  # the real value is always among the stored slots


# Every sparse precision pairing aie4ml can reach on AIE-ML, with the microtile each one is
# expected to resolve to. uint8 x int8 is deliberately absent: it folds onto the int8 x int8 row
# via legality_format and resolves to the same microtile and the same kernel, so simulating it
# would re-run an already-covered path rather than cover a new one.
SPARSE_SIM_CASES = [
    ('int8xint8', 8, 8, (4, 16, 8)),
    ('int16xint8', 16, 8, (2, 16, 8)),
    ('int16xint16', 16, 16, (2, 8, 8)),
]

# The aiesim-only hazard is contention in the sparse decompression FIFO (see
# pack_mmul_rhs_sparse_matrix), which depends on the *sparse* operand -- the weights -- and on the
# single-live-stream discipline in the shared kernel. So the two distinct weight widths are the
# two cases that need the cycle-accurate run; int16 x int8 shares its weight width and its
# sparse_read8b decompression path with int8 x int8, so x86sim covers what is new about it.
SPARSE_AIESIM_CASES = [c for c in SPARSE_SIM_CASES if c[0] in ('int8xint8', 'int16xint16')]


@pytest.mark.requires_vitis
@pytest.mark.parametrize(
    'simulator,case',
    [('x86', c) for c in SPARSE_SIM_CASES] + [('aie', c) for c in SPARSE_AIESIM_CASES],
    ids=[f'x86-{c[0]}' for c in SPARSE_SIM_CASES] + [f'aie-{c[0]}' for c in SPARSE_AIESIM_CASES],
)
def test_sparse_matches_dense_in_simulation(tmp_path, simulator, case):
    """Differential test: the same weights down both paths must give bit-identical outputs.

    Both variants are handed the identical 2:4-structured weight matrix, and neither alters it,
    so any difference is attributable to the compressed layout or the sparse kernel.

    Run for every reachable precision pairing, because each resolves to a different microtile and
    the microtile drives both the L1 input DMA tiling and the compressed layout. int16 weights
    additionally exercise the two-mask-bits-per-sample layout and the int16 reinterpretation of
    the ROM, neither of which the int8-weight rows touch.

    x86sim models `sparse_fill()` as pure bookkeeping, so it cannot see contention between two
    concurrently-live sparse input streams -- a kernel written that way is bit-exact under x86sim
    and wrong on the core. Only the cycle-accurate `aie` run catches that, so both weight widths
    are also run there.
    """
    _require_vitis()
    np = pytest.importorskip('numpy')

    _, act_width, wt_width, microtile = case
    weights = _weights_2of4(seed=5)
    X = np.random.default_rng(5).normal(0, 0.5, size=(BATCH, IN_FEAT))

    outputs = {}
    # Neither side pins a microtile: each resolves its own, which is how a user's project is
    # actually built. Pinning the sparse shape on both sides only ever worked for int8 x int8,
    # where (4,16,8) happens to appear in the dense table too -- the sparse shapes for the int16
    # rows do not, so a shared pin makes the dense build unresolvable. The comparison does not
    # need a shared microtile: tiling changes the order of exact integer accumulations, not the
    # result, so the two paths must still agree bit for bit.
    #
    # 'dense' must be the explicit opt-out: these weights qualify, so passing no directive would
    # build *two* sparse projects and compare them to each other.
    for label, sparsity in (('dense', 'dense'), ('sparse', '2of4')):
        ctx, model = _build(
            tmp_path / label,
            sparsity=sparsity,
            weights=weights,
            act_width=act_width,
            wt_width=wt_width,
        )
        entry = _entry(ctx)
        assert (entry.config.sparsity == '2of4') == (label == 'sparse')
        if label == 'sparse':
            mt = entry.config.microtiling
            assert (mt.microtile_m, mt.microtile_k, mt.microtile_n) == microtile
        model.write()
        model.build(make_target='x86com' if simulator == 'x86' else 'compile')
        outputs[label] = np.asarray(model.predict(X, simulator=simulator))

    np.testing.assert_array_equal(outputs['sparse'], outputs['dense'])

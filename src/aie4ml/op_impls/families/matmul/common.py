from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ....aie_types import FLOAT_FORMATS, FloatIntent, legality_format
from ....ir.graph import STAGING_CONTRACTS, input_tensor_for_role
from ....quant_utils import apply_rounding, dtype_for_precision, handle_overflow
from ...utils import AxisPlan, TensorView, build_staging_descriptor, canonical_buffer_axes
from ...utils.precision import resolve_exact_storage_dtype

log = logging.getLogger(__name__)

# Keys are canonical format-string pairs (lhs_format, rhs_format).
# Integer formats: 'int8', 'int16' (sign-agnostic — both int8_t and uint8_t map here).
# Float formats: 'bfloat16', 'float32', 'fp8_e4m3'.
MICROTILE_OPTIONS: Dict[str, Dict[Tuple[str, str], List[Tuple[int, int, int]]]] = {
    'AIE': {
        ('int8', 'int8'): [(2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 8, 8), (4, 16, 4), (4, 16, 8), (8, 8, 4)],
        ('int16', 'int8'): [(4, 4, 4), (4, 4, 8), (4, 8, 4), (8, 4, 4)],
        ('int8', 'int16'): [(4, 4, 8), (4, 4, 4), (8, 8, 1)],
        ('int16', 'int16'): [(4, 4, 8), (2, 4, 8), (4, 2, 8), (4, 4, 4), (8, 8, 1)],
        ('float32', 'float32'): [(2, 4, 4)],
    },
    'AIE-ML': {
        ('int8', 'int8'): [(4, 8, 8), (2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 16, 4), (4, 16, 8), (8, 8, 4), (8, 8, 8)],
        ('int16', 'int8'): [(4, 4, 8), (2, 8, 8), (4, 4, 4), (4, 8, 4), (8, 4, 4), (8, 4, 8)],
        ('int8', 'int16'): [(4, 4, 4), (4, 4, 8)],
        ('int16', 'int16'): [(4, 4, 4), (2, 4, 8), (4, 2, 8), (4, 4, 8), (8, 1, 8), (8, 2, 8)],
        ('bfloat16', 'bfloat16'): [(4, 8, 4)],
        ('float32', 'float32'): [(4, 8, 4)],
    },
    'AIE-MLV2': {
        ('int8', 'int8'): [(8, 8, 8), (4, 8, 8)],
        ('int16', 'int8'): [(4, 4, 8), (8, 2, 8)],
        ('int8', 'int16'): [(4, 4, 8), (8, 2, 8)],
        ('int16', 'int16'): [(8, 2, 8)],
        ('bfloat16', 'bfloat16'): [(4, 8, 8)],
        ('float32', 'float32'): [(4, 8, 4)],
        ('fp8_e4m3', 'fp8_e4m3'): [(8, 8, 8)],
    },
}


# Microtile shapes with an `aie::mmul` overload taking a *sparse* B operand, from the AI Engine
# API "Supported Sparse Matrix Multiplication Modes" table (aie_api group__group__mmul).
# Deliberately narrower than MICROTILE_OPTIONS: sparse-capable shapes are a distinct set, not a
# filtered view of the dense ones, and every entry here carries an int8-family K that is a
# multiple of 4 (required by the 2:4 group structure -- see is_structured_sparse_2of4).
#
# Only ('int8','int8') on AIE-ML is exercised end-to-end (bit-exact in x86simulator and
# aiesimulator). The remaining rows are transcribed from the same table but unverified here;
# shapes the table footnotes as emulated ('a') or needing extra data manipulation ('b') are
# omitted rather than guessed at.
SPARSE_MICROTILE_OPTIONS: Dict[str, Dict[Tuple[str, str], List[Tuple[int, int, int]]]] = {
    'AIE-ML': {
        ('int8', 'int8'): [(4, 16, 8)],
        ('int16', 'int8'): [(2, 16, 8)],
        ('int16', 'int16'): [(2, 8, 8)],
    },
    'AIE-MLV2': {
        ('int8', 'int8'): [(4, 16, 8), (8, 16, 8)],
        ('int16', 'int16'): [(2, 8, 8), (4, 8, 8)],
    },
}

# Structured-sparsity selectors a `sparsity` directive may ask for. The mode can only ever veto
# or insist -- what decides whether the sparse path is on the table is the weights themselves
# (see sparse_eligibility), never the directive.
#   'auto'  -- the default. Use the hardware sparse path on every Dense layer whose weights
#              already carry the 2:4 pattern, otherwise the ordinary dense path. Adoption is
#              opportunistic: if the sparse variant cannot resolve for this node after all, the
#              family resolver falls back to dense rather than failing the build.
#   'dense' -- opt out. Force the dense path even on a layer whose weights qualify. Short-circuits
#              the weight inspection, since the outcome no longer depends on it: a caller who
#              opts out is asserting they know what their model looks like.
#   '2of4'  -- opt *in* explicitly. Same path as 'auto' when the weights qualify, but a layer
#              that does not is a hard error instead of a silent fallback, so a deliberate
#              request cannot be quietly downgraded.
SPARSITY_MODES = ('auto', 'dense', '2of4')

# Policy, not a law (compare DEFAULT_CONTRACT below): sparse is opt-out, so a layer whose weights
# already satisfy the hardware's 2:4 pattern is compiled sparse unless its directive says
# otherwise. This is numerically free -- aie4ml never prunes, so the weights the kernel computes
# are exactly the weights it was given.
DEFAULT_SPARSITY_MODE = 'auto'

# The sparse format stores exactly one kept value per `SPARSITY_RATIO` positions.
SPARSITY_RATIO = 2

# A layer with at least this fraction of zero weights was plainly trained for sparsity. If such a
# layer still cannot use the hardware path -- because the zeros are not laid out 2:4 along the
# reduction axis -- that is worth telling the user about, since the saving they trained for is
# being lost. Below this fraction the layer is just an ordinary dense layer and nothing is said.
SUBSTANTIAL_SPARSITY_FRACTION = 0.5

# Bytes of *uncompressed* data covered by one mask+data chunk. Taken from the installed AIE-ML
# intrinsic model (aietools/data/aie_ml/lib/me_native.h: `mask_length_sparse = 64` bits at one
# bit per byte), *not* from UG1603's prose, which describes 32-bit masks over 256-bit chunks for
# AIE-ML; the installed model is what actually executes.
#
# Stating this in bytes rather than elements is what makes it width-independent. The mask is one
# bit per *byte*, so a chunk is always an 8-byte mask plus 32 data bytes, covering 64 bytes of
# original data -- 64 int8 samples, or 32 int16 samples. me_native.h shows both: `sparse_read8b`
# steps the 64-bit mask 4 bits at a time over 4 int8 samples (16 x 4 = 64 samples), while
# `sparse_read16b` steps it 8 bits at a time over 4 int16 samples (8 x 4 = 32 samples), each
# sample taking two mask bits (`lmask[i] = (local_mask >> i*2) & 0x3`).
SPARSE_CHUNK_DATA_BYTES = 64

# Weight storage widths the packer implements. Both have a sparse `aie::mmul` overload and a
# decompression path in me_native.h (`sparse_read8b` / `sparse_read16b`). 4-bit sparse weights
# exist in the AIE API table (`8b x 4b`, AIE-ML only) but aie4ml has no 4-bit dense path to build
# on, and 32-bit has no sparse mode on any generation.
SPARSE_ELEMENT_BITS = frozenset({8, 16})


def sparse_chunk_elems(element_bits: int = 8) -> int:
    """Elements one mask+data chunk covers, for a given element width."""
    if element_bits % 8:
        raise NotImplementedError(f'sparse element width must be a whole number of bytes, got {element_bits}-bit.')
    return SPARSE_CHUNK_DATA_BYTES // (element_bits // 8)


def requested_sparsity(node) -> str:
    """The structured sparsity asked of this node; selects the family's sparse variants.

    Defaults to DEFAULT_SPARSITY_MODE ('auto'), so a Dense layer with no `sparsity` directive
    takes the hardware sparse path whenever its weights already qualify. `'dense'` is the opt-out.
    """
    mode = str((node.directives.get('sparsity', {}) or {}).get('mode', DEFAULT_SPARSITY_MODE))
    if mode not in SPARSITY_MODES:
        raise ValueError(f'{node.name}: unknown sparsity mode {mode!r}; expected one of {sorted(SPARSITY_MODES)}.')
    return mode


def sparse_eligibility(node, device) -> Tuple[bool, str]:
    """Whether this node can run on the hardware sparse path, and why not when it cannot.

    Returns `(eligible, reason)` with `reason` empty when eligible. Never raises: this is the
    gate in front of the opt-out default, so it is asked of every Dense node including ones that
    could never be sparse, and a variant's `matches()` must stay total.

    Two independent things are checked, in this order:

    1. **Capability** (`sparse_capability`) -- the target generation, the precisions, whether the
       weights are compile-time constants, and whether a sparse `aie::mmul` overload exists for
       the resulting microtile.
    2. **The weights themselves** -- whether the trained weight matrix is *already* 2:4
       structured along the reduction axis (`analyze_weight_sparsity`). aie4ml does not prune;
       producing the sparsity pattern is training's job, not the compiler's. So a layer whose
       weights do not already carry the pattern is simply not eligible, and is compiled dense
       with its numerics untouched.

    Capability first because it is cheap and because it decides whether the weights are even
    worth walking. Because of (2), taking the sparse path is always numerically free: the kernel
    computes exactly the weights the model was handed over with.

    Deliberately does not try to predict everything `resolve()` could reject. The two shape
    constraints the sparse kernel needs -- `microtile_k` dividing the per-cascade K slice, and an
    even number of N microtiles per chain -- hold structurally, because `_parallelism_candidate`
    aligns the K slice up to `2 * microtile_k` and the N slice up to `2 * microtile_n`. What it
    cannot rule out is tile memory: the sparse microtile has a wider K, so a shape that fits at
    K=8 may leave no valid parallelism at K=16. That case is caught by the resolver's
    speculative-fallback path (see `OpImplVariant.selection_is_speculative`) rather than
    predicted here.
    """
    capable, reason = sparse_capability(node, device)
    if not capable:
        return False, reason

    analysis = analyze_weight_sparsity(node)
    if analysis is None:
        return False, 'weight values could not be analysed for the 2:4 pattern'
    if not analysis['structured_2of4']:
        return False, (
            f'the weights are not 2:4-structured along the reduction axis '
            f'(a group of 4 holds up to {analysis["worst_group_nonzeros"]} nonzeros, '
            f'{100.0 * analysis["zero_fraction"]:.1f}% of all weights are zero); aie4ml does not '
            f'prune, so the pattern has to come from sparsity-aware training'
        )

    return True, ''


def sparse_capability(node, device) -> Tuple[bool, str]:
    """Everything except the weights: could this target and this layer's shape do sparse at all?

    Split out from `sparse_eligibility` so callers can tell a target that has no sparse path from
    a layer whose weights simply do not carry the pattern -- `report_unusable_sparsity` only has
    something useful to say in the second case. Never raises, for the same reason.
    """
    generation = str(getattr(device, 'generation', '') or '')
    if select_generation_key(generation) not in SPARSE_MICROTILE_OPTIONS:
        return False, f'target generation {generation!r} has no sparse aie::mmul support'

    try:
        lhs = input_tensor_for_role(node, 'lhs')
        rhs = input_tensor_for_role(node, 'rhs')
    except Exception as exc:  # a malformed node is the family resolver's problem, not ours
        return False, f'lhs/rhs roles are not resolvable ({exc})'

    if isinstance(lhs.precision, FloatIntent) or isinstance(rhs.precision, FloatIntent):
        return False, 'float precisions are not wired up on the sparse path'
    if not rhs.is_parameter:
        return False, 'weights are not compile-time constants, so they cannot be pre-compressed'
    if not bitwidths_supported(node, device):
        return False, f'lhs/rhs storage widths are not a supported pairing on {generation!r}'

    try:
        rhs_width = int(resolve_exact_storage_dtype(rhs.precision, namespace='rhs', layer_name=node.name).width)
    except Exception as exc:
        return False, f'weight storage width is not resolvable ({exc})'
    if rhs_width not in SPARSE_ELEMENT_BITS:
        return False, (
            f'weight packing is implemented for {sorted(SPARSE_ELEMENT_BITS)}-bit weights, got {rhs_width}-bit'
        )

    options = sparse_microtiling_options(generation, node)
    if not options:
        return False, f'no sparse aie::mmul overload for this precision pairing on {generation!r}'

    pinned = node.directives.get('microtiling', {}) or {}
    keys = ('microtile_m', 'microtile_k', 'microtile_n')
    if all(key in pinned for key in keys):
        shape = tuple(int(pinned[key]) for key in keys)
        if shape not in options:
            return False, f'pinned microtiling {shape} has no sparse overload (allowed: {options})'

    return True, ''


def sparse_microtiling_options(generation: str, node) -> List[Tuple[int, int, int]]:
    """Sparse-capable microtiles for this node's precision pairing, or [] if there are none."""
    try:
        lhs = input_tensor_for_role(node, 'lhs')
        rhs = input_tensor_for_role(node, 'rhs')
        query = (
            legality_format(resolve_exact_storage_dtype(lhs.precision, namespace='lhs', layer_name=node.name).format),
            legality_format(resolve_exact_storage_dtype(rhs.precision, namespace='rhs', layer_name=node.name).format),
        )
    except Exception:
        return []
    return list(SPARSE_MICROTILE_OPTIONS.get(select_generation_key(generation), {}).get(query, []))


def sparse_tile_bytes(microtile_k: int, microtile_n: int, element_bits: int = 8) -> int:
    """Compressed bytes one (K, N) microtile occupies.

    Mirrors `aie::sparse_vector<T, K*N>::memory_bytes()`, i.e.
    `(Elems / sparsity_ratio) * type_bits + Elems * type_bits / 8` bits: the kept values plus a
    one-bit-per-byte mask. Because the mask always has exactly SPARSITY_RATIO bits set per group
    of 4 (see sparse_mask_along_k), this is a constant, which is what lets the kernel address
    tiles by a compile-time stride.
    """
    elems = int(microtile_k) * int(microtile_n)
    if int(element_bits) not in SPARSE_ELEMENT_BITS:
        raise NotImplementedError(
            f'sparse weight packing is implemented for {sorted(SPARSE_ELEMENT_BITS)}-bit weights, got '
            f'{element_bits}-bit; the mask granularity and chunking would have to be re-derived.'
        )
    data_bits = (elems // SPARSITY_RATIO) * element_bits
    mask_bits = (elems * element_bits) // 8
    return (data_bits + mask_bits) // 8


def select_generation_key(generation: str) -> str:
    norm = (generation or '').upper()
    for key in sorted(MICROTILE_OPTIONS.keys(), key=len, reverse=True):
        if key in norm:
            return key
    return 'AIE'


_SUPPORTED_INT_WIDTH_COMBOS = frozenset({(8, 8), (16, 8), (16, 16)})


def bitwidths_supported(node, device) -> bool:
    """Whether this device can run the node's lhs/rhs storage widths."""
    if device.generation not in ('AIE-ML', 'AIE-MLV2'):
        return False
    lhs = input_tensor_for_role(node, 'lhs')
    rhs = input_tensor_for_role(node, 'rhs')
    if isinstance(lhs.precision, FloatIntent):
        return True
    lhs_p = resolve_exact_storage_dtype(lhs.precision, namespace='lhs', layer_name=node.name)
    rhs_p = resolve_exact_storage_dtype(rhs.precision, namespace='rhs', layer_name=node.name)
    return (lhs_p.width, rhs_p.width) in _SUPPORTED_INT_WIDTH_COMBOS


# Policy, not a law: the contract used when nothing asks for one. A global partition
# optimizer will write the directive instead, at which point this is only the fallback for
# graphs it did not run on.
DEFAULT_CONTRACT = 'inner'


def requested_contract(node) -> str:
    """The parallelism contract asked of this node; selects between the family's variants."""
    contract = str((node.directives.get('parallelism', {}) or {}).get('contract', DEFAULT_CONTRACT))
    if contract not in STAGING_CONTRACTS:
        raise ValueError(
            f'{node.name}: unknown parallelism contract {contract!r}; expected one of {sorted(STAGING_CONTRACTS)}.'
        )
    return contract


def describe_inner_lhs_staging(view: TensorView, port: int, buf_dims=None):
    """LHS staging for the 'inner' contract: the port selects a K-chain; the rows stay whole."""
    microtile_m = int(view.microtile.outer)
    microtile_k = int(view.microtile.inner)
    in_slice = view.tile_inner
    outer_slice = view.tile_outer
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)
    return build_staging_descriptor(
        view,
        access='read',
        plans={
            inner_dim: AxisPlan(microtile_k, microtile_k, in_slice // microtile_k, port * in_slice),
            outer_dim: AxisPlan(microtile_m, microtile_m, outer_slice // microtile_m),
        },
        order=traversal_dims,
        io_tiling_overrides={inner_dim: view.tile_raw_inner},
        buf_dims=buf_dims,
        boundary_shape='logical',
    )


def describe_inner_output_staging(view: TensorView, port: int, buf_dims=None):
    """Output staging for the 'inner' contract: the port selects an N-slice; the rows stay whole."""
    microtile_m = int(view.microtile.outer)
    microtile_n = int(view.microtile.inner)
    out_slice = view.tile_inner
    outer_slice = view.tile_outer
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)
    return build_staging_descriptor(
        view,
        access='write',
        plans={
            inner_dim: AxisPlan(microtile_n, microtile_n, out_slice // microtile_n, port * out_slice),
            outer_dim: AxisPlan(microtile_m, microtile_m, outer_slice // microtile_m),
        },
        order=traversal_dims,
        io_tiling_overrides={inner_dim: view.tile_raw_inner},
        buf_dims=buf_dims,
    )


def describe_outer_lhs_staging(view: TensorView, parallelism, port: int, buf_dims=None):
    """LHS staging for the 'outer' contract: the port selects a (row-group, K-chain) tile."""
    microtile_m = int(view.microtile.outer)
    microtile_k = int(view.microtile.inner)
    in_slice = view.tile_inner
    outer_slice = view.tile_outer
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)

    cas_length = max(1, int(parallelism.cas_length))
    row_group = int(port) // cas_length
    k_chain = int(port) % cas_length

    return build_staging_descriptor(
        view,
        access='read',
        plans={
            inner_dim: AxisPlan(microtile_k, microtile_k, in_slice // microtile_k, k_chain * in_slice),
            outer_dim: AxisPlan(microtile_m, microtile_m, outer_slice // microtile_m, row_group * outer_slice),
        },
        order=traversal_dims,
        io_tiling_overrides={inner_dim: view.tile_raw_inner, outer_dim: view.tile_raw_outer},
        buf_dims=buf_dims,
        slice_dim=outer_dim,
        boundary_shape='logical',
    )


def describe_outer_output_staging(view: TensorView, port: int, buf_dims=None):
    """Output staging for the 'outer' contract: the port selects a row-group; N stays whole."""
    microtile_m = int(view.microtile.outer)
    microtile_n = int(view.microtile.inner)
    out_slice = view.tile_inner
    outer_slice = view.tile_outer
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)
    return build_staging_descriptor(
        view,
        access='write',
        plans={
            inner_dim: AxisPlan(microtile_n, microtile_n, out_slice // microtile_n),
            outer_dim: AxisPlan(microtile_m, microtile_m, outer_slice // microtile_m, int(port) * outer_slice),
        },
        order=traversal_dims,
        io_tiling_overrides={inner_dim: view.tile_raw_inner, outer_dim: view.tile_raw_outer},
        buf_dims=buf_dims,
        slice_dim=outer_dim,
    )


def describe_outer_rhs_staging(view: TensorView, parallelism, port: int, buf_dims=None):
    """RHS staging for the 'outer' contract: the port selects a K-chain; every row group shares it."""
    microtile_k = int(view.microtile.outer)
    microtile_n = int(view.microtile.inner)
    k_slice = view.tile_outer
    n_slice = view.tile_inner
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)

    k_chain = int(port) % max(1, int(parallelism.cas_length))

    return build_staging_descriptor(
        view,
        access='read',
        plans={
            inner_dim: AxisPlan(microtile_n, microtile_n, max(1, n_slice // microtile_n)),
            outer_dim: AxisPlan(microtile_k, microtile_k, max(1, k_slice // microtile_k), k_chain * k_slice),
        },
        order=traversal_dims,
        io_tiling_overrides={inner_dim: view.tile_raw_inner, outer_dim: view.tile_raw_outer},
        buf_dims=buf_dims,
        boundary_shape='logical',
        extras={
            'packing': 'mmul_rhs',
            'packing_microtile_k': microtile_k,
            'packing_microtile_n': microtile_n,
        },
    )


def describe_inner_rhs_staging(view: TensorView, parallelism, port: int, buf_dims=None):
    """RHS staging for the 'inner' contract: the port selects an (N-slice, K-chain) tile."""
    microtile_k = int(view.microtile.outer)
    microtile_n = int(view.microtile.inner)
    # The rhs view encodes both K and N slices: outer dim = K, inner dim = N.
    k_slice = view.tile_outer
    n_slice = view.tile_inner
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)

    row = int(port) // int(parallelism.cas_length)
    col = int(port) % int(parallelism.cas_length)

    return build_staging_descriptor(
        view,
        access='read',
        plans={
            inner_dim: AxisPlan(microtile_n, microtile_n, max(1, n_slice // microtile_n), row * n_slice),
            outer_dim: AxisPlan(microtile_k, microtile_k, max(1, k_slice // microtile_k), col * k_slice),
        },
        order=traversal_dims,
        io_tiling_overrides={inner_dim: view.tile_raw_inner, outer_dim: view.tile_raw_outer},
        buf_dims=buf_dims,
        boundary_shape='logical',
        extras={
            'packing': 'mmul_rhs',
            'packing_microtile_k': microtile_k,
            'packing_microtile_n': microtile_n,
        },
    )


def np_dtype_for_spec(spec) -> np.dtype:
    fmt = getattr(spec, 'format', '') or ''
    if fmt == 'bfloat16':
        return np.uint16
    if fmt in ('float32', 'accfloat'):
        return np.float32
    if fmt == 'fp8_e4m3':
        return np.uint8
    if fmt.startswith('uint'):
        return np.uint8 if int(spec.width) <= 8 else np.uint16
    return np.int8 if int(spec.width) <= 8 else np.int16


def np_bias_dtype_for_spec(spec) -> np.dtype:
    fmt = getattr(spec, 'format', '') or ''
    if fmt in FLOAT_FORMATS:
        return np.float32
    return np.int16 if int(spec.width) <= 16 else np.int32


def pack_as_float(array: np.ndarray, fmt) -> np.ndarray:
    """Cast weight/bias data to the float storage format required by mmul kernels."""
    if array is None:
        return None
    from ....aie_types import FloatFormat

    fmt_value = getattr(fmt, 'value', fmt)

    def _float32_to_fp8_scalar(f: np.float32) -> np.uint8:
        h = int(np.float32(f).view(np.uint32))
        h = (h + 0x00080000) & 0xFFFFFFFF
        e = (h & 0x7F800000) >> 23
        m = h & 0x007FFFFF
        sign = (h & 0x80000000) >> 24
        if e > 135:
            result = sign | 0x7F
        elif e > 120:
            result = sign | (((e - 120) << 3) & 0x78) | (m >> 20)
        elif e > 116:
            result = sign | (((0x00780000 + m) >> (140 - e)) + 1) >> 1
        else:
            result = sign
        return np.uint8(result & 0xFF)

    if fmt_value == FloatFormat.BF16.value:
        f32 = np.asarray(array, dtype=np.float32)
        return (f32.view(np.uint32) >> 16).astype(np.uint16)
    if fmt_value == FloatFormat.FP8_E4M3.value:
        vfloat32_to_fp8 = np.vectorize(_float32_to_fp8_scalar, otypes=[np.uint8])
        f32 = np.asarray(array, dtype=np.float32)
        return np.asarray(vfloat32_to_fp8(f32), dtype=np.uint8)
    return np.asarray(array, dtype=np.float32)


def quantize_to_int(
    array: np.ndarray,
    frac_bits: int,
    target_bits: int,
    signed: bool = True,
    rounding_mode=None,
    saturation_mode=None,
) -> np.ndarray:
    """Quantize float weight/bias data to fixed-point integers for mmul kernels."""
    if array is None:
        return None
    scale = 1 << frac_bits if frac_bits > 0 else 1
    scaled = np.asarray(array, dtype=np.float64) * scale
    rounded = apply_rounding(scaled, rounding_mode)
    integers = rounded.astype(np.int64)
    processed = handle_overflow(integers, target_bits, signed, saturation_mode)
    dtype = dtype_for_precision(target_bits, signed)
    return processed.astype(dtype, copy=False)


def pack_mmul_rhs_matrix(
    W,
    *,
    K: int,
    N: int,
    K_slice: int,
    N_slice: int,
    microtile_k: int,
    microtile_n: int,
    cas_length: int,
    cas_num: int,
    order: str = 'C',
    dtype=None,
):
    assert microtile_k > 0 and microtile_n > 0
    assert K_slice % microtile_k == 0
    assert N_slice % microtile_n == 0

    W = np.asarray(W)
    if dtype is not None:
        W = W.astype(dtype, copy=False)
    if W.ndim < 2:
        raise ValueError('W must have at least 2 dimensions')
    W_kn = W.reshape((-1, K, N))[-1]

    tiles_per_k = K_slice // microtile_k
    tiles_per_n = N_slice // microtile_n
    elements_per_tile = microtile_k * microtile_n
    flat_len = tiles_per_k * tiles_per_n * elements_per_tile

    packed = np.zeros((cas_num, cas_length, flat_len), dtype=W_kn.dtype)
    tile_buf = np.zeros((microtile_k, microtile_n), dtype=W_kn.dtype)

    for chain in range(cas_num):
        n_base = chain * N_slice
        for cas in range(cas_length):
            flat = packed[chain, cas]
            tile_idx = 0
            for k_tile in range(tiles_per_k):
                gk = cas * K_slice + k_tile * microtile_k
                real_k = max(0, min(microtile_k, K - gk))
                for n_tile in range(tiles_per_n):
                    tile_buf.fill(0)
                    gn = n_base + n_tile * microtile_n
                    real_n = max(0, min(microtile_n, N - gn))
                    if real_k > 0 and real_n > 0:
                        tile_buf[:real_k, :real_n] = W_kn[gk : gk + real_k, gn : gn + real_n]
                    start = tile_idx * elements_per_tile
                    flat[start : start + elements_per_tile] = tile_buf.ravel(order=order)
                    tile_idx += 1

    return packed


def sparse_mask_along_k(W_kn: np.ndarray) -> np.ndarray:
    """Which positions of an already-2:4 weight matrix the compressed format physically stores.

    **This does not modify the weights and does not prune.** Producing a 2:4 sparsity pattern is
    the training pipeline's job, not the compiler's, so the matrix handed in is expected to
    satisfy the pattern already and is used verbatim; a group of 4 with more nonzeros than the
    format can hold raises rather than being silently thinned.

    Returns `kept`: which positions carry a mask bit. Exactly SPARSITY_RATIO per group of 4,
    which may include an explicitly stored zero when fewer than two values in the group were
    nonzero. Storing a redundant zero costs nothing numerically and is what keeps every
    compressed tile the same size, so the kernel can address tiles by a compile-time stride.

    Grouping is along K, per output column, because the compressed layout is column-major (a
    "column" of a K x N weight matrix is fixed N, varying K) -- see UG1603 "Sparse Matrix
    Multiplication" and the aie_api mmul note that sparse B must be stored column-major. That
    means sparsity-aware training upstream of aie4ml must zero along the *input-feature* axis in
    groups of four, one output column at a time -- not whole output columns.
    """
    if W_kn.ndim != 2:
        raise ValueError(f'expected a 2-D [K, N] weight matrix, got shape {W_kn.shape}.')
    K, N = W_kn.shape
    if K % 4 != 0:
        raise ValueError(f'K={K} must be a multiple of 4 for 2:4 structured sparsity.')

    nonzero = W_kn != 0
    per_group = nonzero.reshape(K // 4, 4, N).sum(axis=1)
    worst = int(per_group.max()) if per_group.size else 0
    if worst > SPARSITY_RATIO:
        raise ValueError(
            f'weights are not {SPARSITY_RATIO}:4-structured along the reduction axis: a group of 4 '
            f'consecutive K values holds {worst} nonzeros. aie4ml does not prune weights -- the '
            f'pattern must come from sparsity-aware training.'
        )

    # Every nonzero is stored; groups holding fewer than SPARSITY_RATIO nonzeros are topped up
    # with explicit zeros, lowest position first, purely to keep the tile size constant.
    kept = nonzero.copy()
    deficit = SPARSITY_RATIO - per_group
    for g in np.argwhere(deficit > 0):
        g0, n = int(g[0]) * 4, int(g[1])
        need = int(deficit[g[0], g[1]])
        for idx in range(4):
            if need == 0:
                break
            if not kept[g0 + idx, n]:
                kept[g0 + idx, n] = True
                need -= 1
    return kept


def is_structured_sparse_2of4(W_kn: np.ndarray) -> bool:
    """Whether every group of 4 consecutive K values has at most 2 nonzeros.

    UG1603 states the requirement as "every four consecutive bytes require at least two zeros"
    for <=8-bit types (and four consecutive *samples* for wider ones). Read together with the
    partial-decompression table, which only defines outcomes for mask patterns of 0, 1 or 2 set
    bits, that is an upper bound of 2 nonzeros per group -- not an exact quota.

    A K that is not a multiple of 4 is zero-padded, which is exactly what the packer does to the
    trailing edge slab: group boundaries land on multiples of 4 in global K coordinates (K slices
    are multiples of `microtile_k`, itself a multiple of 4), so the tail group of a short matrix
    really is padded with stored zeros rather than being ill-defined.
    """
    return worst_group_nonzeros(W_kn) <= SPARSITY_RATIO


def worst_group_nonzeros(W_kn: np.ndarray) -> int:
    """The largest number of nonzeros any group of 4 consecutive K values holds, per column.

    `SPARSITY_RATIO` or fewer means the matrix is encodable as-is; more means it is not, and by
    how much. Trailing K is zero-padded to a multiple of 4, as the packer pads its edge slab.
    """
    W_kn = np.asarray(W_kn)
    if W_kn.ndim != 2:
        raise ValueError(f'expected a 2-D [K, N] weight matrix, got shape {W_kn.shape}.')
    K, N = W_kn.shape
    nonzero = W_kn != 0
    pad = (-K) % 4
    if pad:
        nonzero = np.vstack([nonzero, np.zeros((pad, N), dtype=bool)])
    groups = nonzero.reshape((K + pad) // 4, 4, N).sum(axis=1)
    return int(groups.max()) if groups.size else 0


def quantized_weight_matrix(node) -> Optional[np.ndarray]:
    """This node's constant weights as the [K, N] integers the kernel will actually multiply.

    Returns `None` -- rather than raising -- for anything that is not an analysable integer
    constant, because the callers are total predicates asked of every Dense node.

    The quantized domain is the right one to ask about sparsity in: a weight that quantizes to
    zero *is* a zero to the kernel and is legitimately part of the pattern, and quantization can
    only ever add zeros, never remove them.
    """
    try:
        rhs = input_tensor_for_role(node, 'rhs')
    except Exception:
        return None
    if rhs.data is None or isinstance(rhs.precision, FloatIntent) or rhs.precision is None:
        return None
    try:
        W = quantize_to_int(
            rhs.data,
            rhs.precision.frac,
            rhs.precision.width,
            signed=rhs.precision.signed,
            rounding_mode=rhs.precision.rounding,
            saturation_mode=rhs.precision.saturation,
        )
        W = np.asarray(W)
        if W.ndim < 2:
            return None
        return W.reshape((-1, W.shape[-2], W.shape[-1]))[-1]
    except Exception:
        return None


def analyze_weight_sparsity(node) -> Optional[Dict[str, Any]]:
    """Measure the sparsity the *model* arrived with; `None` if the weights are not analysable.

    This is the check that decides whether a Dense layer takes the hardware sparse path, so it
    reports what is there rather than what could be arranged: `structured_2of4` says whether the
    trained pattern is directly encodable, `zero_fraction` how sparse the layer is overall, and
    `worst_group_nonzeros` how far off a non-encodable layer is.

    Memoized in `node.artifacts` because `matches()` may be asked more than once per node and
    this walks the whole weight matrix. Weight data does not change after import, so the cache
    is valid for the node's lifetime.
    """
    cached = node.artifacts.get('weight_sparsity')
    if cached is not None:
        return cached or None

    W_kn = quantized_weight_matrix(node)
    if W_kn is None:
        node.artifacts['weight_sparsity'] = {}
        return None

    elements = int(W_kn.size)
    nonzero = int(np.count_nonzero(W_kn))
    worst = worst_group_nonzeros(W_kn)
    analysis = {
        'elements': elements,
        'nonzero': nonzero,
        'zeros': elements - nonzero,
        'zero_fraction': (elements - nonzero) / elements if elements else 0.0,
        'worst_group_nonzeros': worst,
        'structured_2of4': worst <= SPARSITY_RATIO,
    }
    node.artifacts['weight_sparsity'] = analysis
    return analysis


def report_unusable_sparsity(node, device) -> None:
    """Warn, once per node, about trained sparsity the hardware path cannot use.

    Only fires for a layer that is genuinely sparse (>= SUBSTANTIAL_SPARSITY_FRACTION zeros) yet
    not 2:4-structured along the reduction axis, on a target that could otherwise have used the
    sparse path. Such a model was trained for a saving it is not getting, and the fix -- structure
    the pruning 2:4 along the input-feature axis -- is not something the user can guess from
    silence. An ordinary dense layer, or a target with no sparse path at all, says nothing.
    """
    if not sparse_capability(node, device)[0]:
        return
    analysis = analyze_weight_sparsity(node)
    if analysis is None or analysis['structured_2of4']:
        return
    if analysis['zero_fraction'] < SUBSTANTIAL_SPARSITY_FRACTION or analysis.get('warned'):
        return
    analysis['warned'] = True
    log.warning(
        '%s: %.1f%% of the weights are zero, but the pattern is not 2:4-structured along the '
        'reduction axis (a group of 4 consecutive input features holds up to %d nonzeros, the '
        'hardware format allows %d), so the AIE-ML sparse path cannot encode it; compiled dense. '
        'To use it, prune 2:4 along the input-feature axis per output column during training.',
        node.name,
        100.0 * analysis['zero_fraction'],
        analysis['worst_group_nonzeros'],
        SPARSITY_RATIO,
    )


def _compress_sparse_tile(values: np.ndarray, kept: np.ndarray, *, element_bits: int = 8) -> bytes:
    """Compress one (K, N) microtile into AIE-ML sparse bytes.

    Column-major element order over the whole tile -- column 0's K values, then column 1's, ...
    -- exactly as UG1603 ("compressed or decompressed column-wise") and the aie_api mmul note
    ("sparse data be stored in column major layout") specify. Each chunk is an 8-byte mask
    followed by the kept values in ascending position order, only for set bits.

    The mask is one bit per *byte* of uncompressed data, which is the single rule that makes this
    width-independent: an 8-bit sample takes one mask bit, a 16-bit sample takes two adjacent mask
    bits (both set when the sample is stored, matching how me_native.h's `decompress_sparse` tests
    each 2-bit field with `bool()`). Bits run LSB-first within each mask byte, mask bytes
    ascending. So one chunk always covers SPARSE_CHUNK_DATA_BYTES of original data and always
    emits 8 mask bytes plus SPARSE_CHUNK_DATA_BYTES/SPARSITY_RATIO data bytes -- a constant, so
    no guard padding is needed to keep the next mask 32-bit aligned, and the kernel can address
    tiles by a compile-time stride.
    """
    K, N = values.shape
    elems = K * N
    itemsize = int(element_bits) // 8
    chunk_elems = sparse_chunk_elems(element_bits)
    if elems % chunk_elems != 0:
        raise NotImplementedError(
            f'microtile {K}x{N} has {elems} elements, not a multiple of the {chunk_elems}-element '
            f'sparse chunk for {element_bits}-bit data; the mask/guard-bit layout for a partial '
            'chunk is not implemented.'
        )
    flat_vals = values.T.reshape(-1)
    flat_kept = kept.T.reshape(-1)

    mask_bytes_per_chunk = SPARSE_CHUNK_DATA_BYTES // 8
    data_bytes_per_chunk = SPARSE_CHUNK_DATA_BYTES // SPARSITY_RATIO
    twos_complement = (1 << int(element_bits)) - 1

    out = bytearray()
    for chunk in range(elems // chunk_elems):
        base = chunk * chunk_elems
        mask = bytearray(mask_bytes_per_chunk)
        data = bytearray()
        for local in range(chunk_elems):
            pos = base + local
            if not flat_kept[pos]:
                continue
            for bit in range(local * itemsize, (local + 1) * itemsize):
                mask[bit // 8] |= 1 << (bit % 8)
            data += (int(flat_vals[pos]) & twos_complement).to_bytes(itemsize, 'little')
        if len(data) != data_bytes_per_chunk:
            raise AssertionError(
                f'chunk {chunk} stored {len(data)} bytes, expected {data_bytes_per_chunk}: '
                'every group of 4 must keep exactly 2 values.'
            )
        out += mask
        out += data
    return bytes(out)


def pack_mmul_rhs_sparse_matrix(
    W,
    *,
    K: int,
    N: int,
    K_slice: int,
    N_slice: int,
    microtile_k: int,
    microtile_n: int,
    cas_length: int,
    cas_num: int,
    dtype=None,
    layer_name: str = '',
):
    """Pack the RHS for the sparse `aie::mmul` overload, mirroring pack_mmul_rhs_matrix.

    Returns `packed`, a uint8 array shaped `(cas_num, cas_length, tiles * sparse_tile_bytes)`
    holding the compressed ROM. The weights are compressed exactly as given -- nothing is pruned,
    so the kernel computes the model's own numerics -- and a matrix that is not already
    2:4-structured along K raises out of `sparse_mask_along_k` rather than being thinned to fit.
    Eligibility has normally been checked long before this point (`sparse_eligibility`), so the
    raise here is the invariant asserting itself, not the expected way to find out.

    Tiles are ordered by the (n, n+1) microtile *pair* the kernel processes together, then by
    K-tile, then by the two columns of the pair. The kernel reads both B operands of a K-tile as
    two consecutive pops of a *single* live `aie::sparse_vector_input_buffer_stream`; that is the
    shape aie_api's own gemm_int8xint8_sparse example uses, and using one stream rather than one
    per column is load-bearing -- two concurrently-live streams are bit-exact under
    x86simulator but wrong under aiesimulator, because the sparse decompression FIFO is real
    shared state on the AIE core that x86sim models only as bookkeeping.
    """
    prefix = f'{layer_name}: ' if layer_name else ''
    if microtile_k <= 0 or microtile_n <= 0:
        raise ValueError(f'{prefix}microtile_k and microtile_n must be positive.')
    if K_slice % microtile_k or N_slice % microtile_n:
        raise ValueError(
            f'{prefix}K_slice={K_slice} and N_slice={N_slice} must be multiples of the microtile '
            f'({microtile_k}, {microtile_n}).'
        )
    if microtile_k % 4:
        raise ValueError(
            f'{prefix}microtile_k={microtile_k} must be a multiple of 4: 2:4 sparsity groups four '
            'consecutive K values, and a group straddling a microtile boundary cannot be encoded.'
        )

    tiles_per_k = K_slice // microtile_k
    tiles_per_n = N_slice // microtile_n
    if tiles_per_n % 2:
        raise ValueError(
            f'{prefix}N_slice={N_slice} gives {tiles_per_n} microtiles along N; the kernel consumes '
            'them in pairs, so this must be even.'
        )

    W = np.asarray(W)
    if dtype is not None:
        W = W.astype(dtype, copy=False)
    if W.ndim < 2:
        raise ValueError(f'{prefix}W must have at least 2 dimensions.')
    W_kn = W.reshape((-1, K, N))[-1]

    element_bits = int(W_kn.dtype.itemsize) * 8
    tile_bytes = sparse_tile_bytes(microtile_k, microtile_n, element_bits)
    flat_len = tiles_per_k * tiles_per_n * tile_bytes
    packed = np.zeros((cas_num, cas_length, flat_len), dtype=np.uint8)

    for chain in range(cas_num):
        n_base = chain * N_slice
        for cas in range(cas_length):
            k_base = cas * K_slice
            # Pad the slice so a partial edge tile is encoded as explicit zeros, exactly as the
            # dense packer zero-fills its tile buffer.
            slab = np.zeros((K_slice, N_slice), dtype=W_kn.dtype)
            real_k = max(0, min(K_slice, K - k_base))
            real_n = max(0, min(N_slice, N - n_base))
            if real_k > 0 and real_n > 0:
                slab[:real_k, :real_n] = W_kn[k_base : k_base + real_k, n_base : n_base + real_n]

            try:
                kept = sparse_mask_along_k(slab)
            except ValueError as exc:
                raise ValueError(f'{prefix}{exc}') from exc
            values = slab

            blob = bytearray()
            for pair in range(tiles_per_n // 2):
                for kt in range(tiles_per_k):
                    for nt in (2 * pair, 2 * pair + 1):
                        ks = slice(kt * microtile_k, (kt + 1) * microtile_k)
                        ns = slice(nt * microtile_n, (nt + 1) * microtile_n)
                        blob += _compress_sparse_tile(values[ks, ns], kept[ks, ns], element_bits=element_bits)
            if len(blob) != flat_len:
                raise AssertionError(f'{prefix}packed {len(blob)} bytes, expected {flat_len}.')
            packed[chain, cas] = np.frombuffer(bytes(blob), dtype=np.uint8)

    return packed


def pack_vector_by_n_slice(
    v,
    *,
    N: int,
    N_slice: int,
    cas_num: int,
    dtype=None,
):
    v = np.asarray(v)
    if dtype is not None:
        v = v.astype(dtype, copy=False)
    if v.ndim > 1:
        v = v.reshape((-1,))[:N]
    if v.shape[0] != N:
        raise ValueError(f'Vector length mismatch: got {v.shape[0]}, expected {N}')

    packed = np.zeros((cas_num, N_slice), dtype=v.dtype)
    for chain in range(cas_num):
        n_base = chain * N_slice
        real = max(0, min(N_slice, N - n_base))
        if real > 0:
            packed[chain, :real] = v[n_base : n_base + real]
    return packed

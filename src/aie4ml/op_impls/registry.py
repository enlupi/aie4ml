from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .base import OpImplVariant

_TEMPLATE_ROOT = Path(__file__).resolve().parent.parent / 'templates' / 'firmware' / 'variants'


class OpImplRegistry:
    def __init__(self):
        self._variants: dict[str, list[OpImplVariant]] = {}

    def register(self, variant: OpImplVariant) -> None:
        self._variants.setdefault(variant.op_type, []).append(variant)

    def candidates(self, op_type: str) -> list[OpImplVariant]:
        """Return registered variants for op_type sorted by descending plevel."""
        return sorted(self._variants.get(op_type, []), key=lambda v: v.plevel, reverse=True)

    def supported_microtilings(self, op_type: str, generation: str, query) -> List[Tuple[int, int, int]]:
        """Legal microtile shapes for `op_type`, from the highest-plevel variant that answers.

        Only meaningful where every variant of `op_type` shares one table. Variants of a family
        can disagree -- the sparse dense variants need shapes with a sparse `aie::mmul` overload,
        a different set from the dense ones -- so code inside a variant's `resolve()` should ask
        that variant directly (see `_supported_microtile_options`) rather than call this.
        """
        for variant in self.candidates(op_type):
            try:
                return variant.microtiling_options(generation, query)
            except NotImplementedError:
                continue
        return []


_GLOBAL_OP_IMPL_REGISTRY = OpImplRegistry()


def get_op_impl_registry() -> OpImplRegistry:
    return _GLOBAL_OP_IMPL_REGISTRY


def register_variant(cls):
    if cls.param_template:
        expected = _TEMPLATE_ROOT / cls.param_template / 'parameters.h.jinja'
        if not expected.exists():
            raise FileNotFoundError(
                f'{cls.__name__}: param_template={cls.param_template!r} has no template at {expected}. '
                'Create the template before registering the variant.'
            )
    _GLOBAL_OP_IMPL_REGISTRY.register(cls())
    return cls

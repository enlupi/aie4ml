from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .base import OpImplVariant

log = logging.getLogger(__name__)


class FamilyResolver:
    """Thin structural validator + variant dispatcher for one op type."""

    op_type: ClassVar[str] = ''

    def validate_structure(self, _node: Any, _device: Any) -> None:
        raise NotImplementedError

    def resolve(self, node: Any, device: Any, directives: Optional[Dict[str, Any]] = None) -> Tuple[Any, OpImplVariant]:
        from .registry import get_op_impl_registry

        self.validate_structure(node, device)
        for variant in get_op_impl_registry().candidates(self.op_type):
            if not variant.matches(node, device):
                continue
            try:
                config = variant.resolve(node, device, directives)
            except Exception as exc:
                # A variant that offered itself (rather than being asked for) must not be able to
                # break a graph a lower-priority variant would have compiled, so let the next
                # candidate try. Anything explicitly requested propagates.
                if not variant.selection_is_speculative(node, device):
                    raise
                log.warning(
                    '%s: %s was selected automatically but could not be resolved (%s); '
                    'falling back to the next matching implementation.',
                    node.name,
                    variant.variant_id or type(variant).__name__,
                    exc,
                )
                continue
            return config, variant
        raise ValueError(f'{node.name}: no {self.op_type} variant matches (generation={device.generation!r}).')


class FamilyResolverRegistry:
    def __init__(self):
        self._resolvers: dict[str, FamilyResolver] = {}

    def register(self, op_type: str, resolver: FamilyResolver) -> None:
        self._resolvers[op_type] = resolver

    def get(self, op_type: str) -> FamilyResolver:
        resolver = self._resolvers.get(op_type)
        if resolver is None:
            raise NotImplementedError(f'No family resolver registered for op_type={op_type!r}.')
        return resolver


_GLOBAL_FAMILY_RESOLVER_REGISTRY = FamilyResolverRegistry()


def get_family_resolver_registry() -> FamilyResolverRegistry:
    return _GLOBAL_FAMILY_RESOLVER_REGISTRY


def family_resolver(*op_types: str):
    def decorator(cls):
        instance = cls()
        for op_type in op_types:
            _GLOBAL_FAMILY_RESOLVER_REGISTRY.register(str(op_type), instance)
        return cls

    return decorator

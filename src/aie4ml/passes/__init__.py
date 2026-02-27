"""Optimizer passes specific to the AIE backend."""

from .fanout_legalize import LegalizeFanoutEntries
from .fold_transpose import FoldTransposeViews
from .fuse_activation import FuseActivationCasts
from .lower import LowerToAieIr
from .memory_plan import BuildMemoryPlan, CollectMemoryEntries, MaterializeMemoryPlan
from .memtile_legalize import LegalizeMemtilePortLimits
from .pack import PackKernelArtifacts
from .placement import PlaceKernels
from .quant import IntegerQuantizer
from .resolve import Resolve

__all__ = [
    'LowerToAieIr',
    'IntegerQuantizer',
    'FuseActivationCasts',
    'FoldTransposeViews',
    'LegalizeFanoutEntries',
    'LegalizeMemtilePortLimits',
    'Resolve',
    'PackKernelArtifacts',
    'PlaceKernels',
    'CollectMemoryEntries',
    'MaterializeMemoryPlan',
    'BuildMemoryPlan',
]

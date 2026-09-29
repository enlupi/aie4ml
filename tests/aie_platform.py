# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Which AIE platform the tests compile against, deduced from the installed Vitis.

The platform is deduced from the *installed* Vitis rather than hardcoded, because the base
platform names are version-stamped (`xilinx_vek280_base_202610_1` ships with 2026.1,
`..._202520_1` with 2025.2) and `v++` rejects a platform it does not have with an error that
surfaces far from its cause — a missing simulator output file, several layers up. Deducing it
means the suite follows a toolchain upgrade on its own.

Every test that needs a part calls `resolve_aie_part()` and passes the result through
explicitly, including on the hls4ml path where `create_initial_config` would otherwise apply its
own default. That default is a fixed string and will be stale for some installed Vitis; the tests
should be exercising aie4ml, not that default.

Lives outside `conftest.py` so test modules can import it directly; `conftest.py` only uses it
for the run header.
"""

from __future__ import annotations

import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

# The tests are written against AIE-ML microtiles and VEK280 array geometry, so the deduction is
# deliberately confined to that device rather than picking whatever is newest.
TARGET_DEVICE = 'Vek280'


def vitis_root() -> Path | None:
    """The installed Vitis, from `XILINX_VITIS` or by locating `v++` on `PATH`."""
    env = os.environ.get('XILINX_VITIS')
    if env and Path(env).is_dir():
        return Path(env)
    vpp = shutil.which('v++')
    if vpp:
        # .../Vitis/bin/v++ -> .../Vitis
        root = Path(vpp).resolve().parent.parent
        if root.is_dir():
            return root
    return None


def installed_platforms() -> set[str]:
    """Base platform names Vitis actually has, or an empty set if it is not installed."""
    root = vitis_root()
    if root is None:
        return set()
    base = root / 'base_platforms'
    if not base.is_dir():
        return set()
    return {entry.name for entry in base.iterdir() if entry.is_dir()}


def _version_key(name: str) -> tuple[int, ...]:
    """Sort key from a version-stamped platform name; unstamped names sort oldest."""
    match = re.search(r'(\d{4})(\d)0_(\d+)$', name)
    return tuple(int(g) for g in match.groups()) if match else (0, 0, 0)


@lru_cache(maxsize=1)
def resolve_aie_part() -> str:
    """The newest catalog platform for TARGET_DEVICE that the installed Vitis also has.

    Falls back to the newest catalog entry when Vitis is absent, so the IR-only tests — which
    never invoke `v++` and only need a well-formed device entry — still run on a machine with no
    toolchain at all.
    """
    from aie4ml.device_catalog import load_device_catalog

    catalog = load_device_catalog()
    supported = sorted(
        (name for name, info in catalog.items() if info.get('DeviceName') == TARGET_DEVICE),
        key=_version_key,
        reverse=True,
    )
    if not supported:
        raise RuntimeError(f'No {TARGET_DEVICE} entry in the aie4ml device catalog.')

    installed = installed_platforms()
    if not installed:
        return supported[0]

    for name in supported:
        if name in installed:
            return name

    raise RuntimeError(
        f"None of the catalog's {TARGET_DEVICE} platforms {supported} is installed under "
        f'{vitis_root()}/base_platforms (found {sorted(installed)}). Install a matching base '
        f'platform, or add the installed one to src/aie4ml/aie_devices.json.'
    )

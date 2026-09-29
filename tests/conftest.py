# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Pytest hooks. The platform-deduction logic itself lives in `aie_platform.py`."""

from __future__ import annotations

from aie_platform import resolve_aie_part, vitis_root


def pytest_report_header(config):
    """Say which platform the run resolved to — the first thing to check on a platform error."""
    del config
    root = vitis_root()
    try:
        part = resolve_aie_part()
    except RuntimeError as exc:  # keep collection alive; the tests themselves will report
        return f'aie4ml: part resolution FAILED ({exc})'
    return f'aie4ml: part={part} vitis={root or "not installed (IR-only tests)"}'

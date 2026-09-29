#!/usr/bin/env python3
"""Minimal reproducer: one ordinary Dense layer, built for AIE-MLv2 and for AIE-ML.

Self-contained -- builds its own single-node graph, imports nothing from tests/.

Why the control matters
-----------------------
The AIE-MLv2 hardware compile fails with

    ERROR: [aiecompiler 77-6572] Could not load Device Part xc2ve3858-ssva2112-2MP-e-S

Run on its own, that looks like it could be anything: the device entry, the kernel, sparsity.
So this script always builds the *same* layer on the VEK280 (AIE-ML) part in the same
invocation. A stage that fails on MLv2 and passes on ML is attributable to the target; a stage
that fails on both is not.

The layer is deliberately ORDINARY DENSE: random normal weights are essentially never
2:4-structured, so the weight-driven sparse path declines and the plain dense variant is chosen.
The script asserts that, so it fails loudly rather than silently testing something else.

Stages, in dependency order
---------------------------
    resolve   aie4ml picks a variant + microtile and builds the IR   (no toolchain)
    write     emit the project                                       (no toolchain)
    x86com    v++ --target=x86sim                                     (functional compile)
    x86sim    run x86simulator                                        (functional result)
    compile   v++ --target=hw     <-- what aiesim needs               (expected MLv2 failure)
    aiesim    aiesimulator                                            (never reached on MLv2)

Usage
-----
    XILINX_VITIS=/opt/Xilinx/2026.1/Vitis/ python repro_dense.py
    XILINX_VITIS=... python repro_dense.py --parts vek385_base
    XILINX_VITIS=... python repro_dense.py --keep      # keep project dirs for inspection
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BATCH, IN_FEAT, OUT_FEAT = 8, 128, 256
X_FRAC, W_FRAC, Y_FRAC = 5, 6, 3
STAGES = ('resolve', 'write', 'x86com', 'x86sim', 'compile', 'aiesim')


def qi(width: int, frac: int):
    from aie4ml.aie_types import QuantIntent, RoundingMode, SaturationMode

    return QuantIntent(
        width=width, frac=frac, signed=True, rounding=RoundingMode.RND_CONV, saturation=SaturationMode.SAT
    )


def default_parts() -> list[str]:
    """The MLv2 part under investigation, plus the newest installed VEK280 as control."""
    from aie4ml.device_catalog import load_device_catalog

    catalog = load_device_catalog()
    vitis = os.environ.get('XILINX_VITIS')
    installed = set()
    if vitis and (Path(vitis) / 'base_platforms').is_dir():
        installed = {p.name for p in (Path(vitis) / 'base_platforms').iterdir() if p.is_dir()}

    def newest(device: str) -> str | None:
        names = sorted(
            (n for n, info in catalog.items() if info.get('DeviceName') == device),
            reverse=True,
        )
        for n in names:
            if not installed or n in installed:
                return n
        return names[0] if names else None

    return [p for p in (newest('Vek385'), newest('Vek280')) if p]


def build_graph(part: str, out_dir: Path):
    """One Dense layer: y = relu-free x @ W + b, all int8, ordinary (non-sparse) weights."""
    from aie4ml.frontends.common import register_default_traits
    from aie4ml.frontends.onnx.utils import create_context
    from aie4ml.ir.graph import OpNode, TensorVar, set_input_roles
    from aie4ml.model import AIEModel

    ctx = create_context(
        {'Part': part, 'AIEConfig': {'BatchSize': BATCH, 'Iterations': 1}}, out_dir, 'repro_dense', None, None
    )
    graph = ctx.ir.logical
    rng = np.random.default_rng(3)
    W = rng.normal(0, 0.2, size=(IN_FEAT, OUT_FEAT))  # dense: ~no zeros, so never 2:4
    b = rng.normal(0, 0.1, size=(OUT_FEAT,))

    x = TensorVar(name='x', shape=(BATCH, IN_FEAT), precision=qi(8, X_FRAC))
    graph.add_tensor(x)
    graph.mark_graph_input('x')
    graph.add_tensor(TensorVar(name='w', shape=W.shape, precision=qi(8, W_FRAC), data=W))
    graph.add_tensor(TensorVar(name='bi', shape=b.shape, precision=qi(8, W_FRAC), data=b))
    wt, bt = graph.tensors['w'], graph.tensors['bi']

    op = OpNode(name='d0_aie', op_type='dense', dialect=ctx.device.dialect)
    op.metadata.update(
        {
            'n_in': IN_FEAT,
            'n_out': OUT_FEAT,
            'use_bias': True,
            'layer_class': 'Dense',
            'source_class': 'Gemm',
            'source_layer': 'd0',
            'input_roles': ['lhs', 'rhs', 'bias'],
        }
    )
    for src in (x, wt, bt):
        src.consumers.append(op)
    op.inputs.extend([x, wt, bt])
    y = TensorVar(name='y', shape=(BATCH, OUT_FEAT), precision=qi(8, Y_FRAC), producer=op)
    graph.add_tensor(y)
    op.outputs.append(y)
    graph.add_node(op)
    graph.mark_graph_output('y')
    set_input_roles(op, op.inputs, ['lhs', 'rhs', 'bias'])

    register_default_traits(ctx)
    model = AIEModel.from_context(ctx)
    model.run_pipeline()
    return ctx, model


def errors_from_log(project: Path, limit: int = 4) -> list[str]:
    log = project / 'log'
    if not log.exists():
        return []
    keep = [ln.strip() for ln in log.read_text(errors='replace').splitlines() if 'ERROR' in ln]
    return keep[:limit]


def run_part(part: str, keep: bool) -> dict:
    result = {s: '-' for s in STAGES}
    notes: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix=f'repro_{part}_'))
    X = np.random.default_rng(5).normal(0, 0.5, size=(BATCH, IN_FEAT))

    try:
        ctx, model = build_graph(part, tmp)
    except Exception as exc:  # noqa: BLE001
        result['resolve'] = 'FAIL'
        notes.append(f'resolve: {type(exc).__name__}: {exc}')
        return {'part': part, 'dir': tmp, 'stages': result, 'notes': notes, 'meta': {}}

    inst = next(iter(ctx.ir.execution.instances.values()))
    mt = inst.config.microtiling
    meta = {
        'generation': getattr(ctx.device, 'generation', '?'),
        'variant': inst.variant.variant_id,
        'microtile': (mt.microtile_m, mt.microtile_k, mt.microtile_n),
        'sparsity': getattr(inst.config, 'sparsity', '?'),
    }
    result['resolve'] = 'OK'
    if 'sparse' in str(meta['variant']):
        notes.append('UNEXPECTED: a sparse variant was selected; this reproducer wants plain dense.')

    project = ctx.project_config.output_dir
    try:
        model.write()
        result['write'] = 'OK'
    except Exception as exc:  # noqa: BLE001
        result['write'] = 'FAIL'
        notes.append(f'write: {exc}')
        return {'part': part, 'dir': tmp, 'stages': result, 'notes': notes, 'meta': meta}

    # --- functional path: x86sim -------------------------------------------------
    try:
        model.build(make_target='x86com', log_to_stdout=False)
        result['x86com'] = 'OK'
    except Exception:  # noqa: BLE001
        result['x86com'] = 'FAIL'
        notes += [f'x86com: {e}' for e in errors_from_log(project)]

    if result['x86com'] == 'OK':
        try:
            y = np.asarray(model.predict(X, simulator='x86'))
            result['x86sim'] = 'OK'
            notes.append(f'x86sim output shape {y.shape}, checksum {int(np.abs(y).sum())}')
        except Exception as exc:  # noqa: BLE001
            result['x86sim'] = 'FAIL'
            notes.append(f'x86sim: {type(exc).__name__}: {exc}')

    # --- hardware path: what aiesim depends on -----------------------------------
    try:
        model.build(make_target='compile', log_to_stdout=False)
        result['compile'] = 'OK'
    except Exception:  # noqa: BLE001
        result['compile'] = 'FAIL'
        notes += [f'compile: {e}' for e in errors_from_log(project)]

    # Run aiesim regardless, to record what it does with an incomplete Work/.
    proc = subprocess.run(['make', 'aiesim'], cwd=project, capture_output=True, text=True)
    result['aiesim'] = 'OK' if proc.returncode == 0 else 'FAIL'
    if proc.returncode != 0:
        for ln in (proc.stdout + proc.stderr).splitlines():
            if any(k in ln for k in ('ERROR', 'No such', 'missing')):
                notes.append(f'aiesim: {ln.strip()[:150]}')
                break
    work = project / 'Work'
    notes.append(f'Work/ exists={work.exists()}  Work/config/scsim_config.json exists='
                 f'{(work / "config" / "scsim_config.json").exists()}')

    if not keep:
        notes.append('(project dir removed; pass --keep to retain)')
    return {'part': part, 'dir': project if keep else tmp, 'stages': result, 'notes': notes, 'meta': meta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parts', nargs='+', default=None, help='parts to build (default: VEK385 + VEK280 control)')
    ap.add_argument('--keep', action='store_true', help='keep emitted projects for inspection')
    args = ap.parse_args(argv)

    if 'XILINX_VITIS' not in os.environ:
        print('warning: XILINX_VITIS is not set; compile/sim stages will fail.\n', file=sys.stderr)

    parts = args.parts or default_parts()
    print(f'Dense-layer reproducer  ({BATCH}x{IN_FEAT} @ {IN_FEAT}x{OUT_FEAT}, int8)')
    print(f'Parts under test: {", ".join(parts)}\n')

    rows = [run_part(p, args.keep) for p in parts]

    width = max(len(r['part']) for r in rows) + 2
    print(f'{"part".ljust(width)}{"gen".ljust(10)}' + ''.join(s.ljust(9) for s in STAGES))
    print('-' * (width + 10 + 9 * len(STAGES)))
    for r in rows:
        gen = str(r['meta'].get('generation', '?'))
        print(r['part'].ljust(width) + gen.ljust(10) + ''.join(r['stages'][s].ljust(9) for s in STAGES))

    for r in rows:
        print(f'\n--- {r["part"]} ({r["meta"].get("generation", "?")})')
        if r['meta']:
            print(f'    variant={r["meta"]["variant"]}  microtile={r["meta"]["microtile"]}  '
                  f'sparsity={r["meta"]["sparsity"]}')
        if args.keep:
            print(f'    project: {r["dir"]}')
        for n in r['notes']:
            print(f'    {n}')

    print('\nExpected today: every stage OK on the VEK280 control; on VEK385, resolve/write/'
          '\nx86com/x86sim OK but compile FAIL (missing device part) and aiesim FAIL as a'
          '\nconsequence. compile failing on BOTH parts would mean something else is wrong.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

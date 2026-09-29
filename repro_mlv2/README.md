# AIE-MLv2 dense-layer reproducer

Two standalone scripts for investigating why nothing can be simulated in `aiesimulator` on
AIE-MLv2 (VEK385) in this installation. Self-contained: they import nothing from `tests/` and
change nothing in `src/`. Delete the directory when you are done with it.

The headline finding is that **this is not a sparsity problem**. An ordinary dense layer fails
in exactly the same way, at exactly the same stage, with a byte-identical error. That matters
because the AIE-MLv2 *sparse* path is separately under suspicion, and it is easy to mistake this
environment failure for evidence about sparsity — a mistake that was actually made while
investigating it.

## Scripts

### `./probe_part.sh` — environment only, no build, ~1 second

Compares the Vitis device databases for the failing part (`xc2ve3858`, VEK385, AIE-MLv2) against
the working one (`xcve2802`, VEK280, AIE-ML), path by path.

Needs no license and compiles nothing. `XILINX_VITIS` is honoured, defaulting to
`/opt/Xilinx/2026.1/Vitis`.

What it shows, as measured on this machine:

| Path (under the Vitis data root) | `xcve2802` | `xc2ve3858` |
| --- | --- | --- |
| `data/ips/parts/versal/<part>` (incl. `ssva2112.json`) | present | present |
| `data/versal/me/regdb/<part>` | present | present |
| `data/parts/xilinx/versal/devint/versalaiedge/<part>` | present | **absent** |
| `SharedData/.../vault/versal/data/device_cfg/<part>` | present | **absent** |
| `SharedData/.../vault/versal/data/device_pkgs/<part>` | present | **absent** |
| `data/versal/me/code/<part>` | present | **absent** |
| `data/versal/me/clock/<part>` | present | **absent** |

So the VE3858 device is *partially* installed. The base platform is there, the IP-level part and
package JSONs are there, and the AI Engine **register** DB is there — but the devint part
definition, the SharedData device cfg/pkgs, and the AI Engine **code** and **clock** databases
are not. That is what `aiecompiler 77-6572` cannot load.

> A warning about method, learned the hard way: do **not** diagnose this with a bare
> `find -maxdepth N`. `ssva2112.json` sits at depth 6, so a shallower search reports it missing
> and you conclude the package database is absent when it is present. The script does a
> path-by-path comparison against a known-good part instead.

The script also reports one extra fact relevant to the sparse question: the `aie_ml` intrinsic
model has ~30 sparse decompression symbols (`decompress_sparse`, `sparse_read8b/16b`,
`sparse_pop`, `sparse_fill`) while the `aie2ps` model has **zero**. So `x86simulator` cannot
expand a compressed tile on AIE-MLv2 either, which is why `aiesimulator` is the only viable
check there rather than merely the stronger one.

### `python repro_dense.py` — builds an ordinary dense layer, ~2 minutes

One int8 Dense layer (`8x128 @ 128x256`), built for VEK385 **and** for VEK280 in the same run,
reporting each stage independently:

```
resolve   aie4ml picks a variant + microtile, builds the IR      (no toolchain)
write     emit the project                                       (no toolchain)
x86com    v++ --target=x86sim                                     functional compile
x86sim    run x86simulator                                        functional result
compile   v++ --target=hw      <-- what aiesim needs              expected MLv2 failure
aiesim    aiesimulator                                            never reached on MLv2
```

```bash
XILINX_VITIS=/opt/Xilinx/2026.1/Vitis/ python repro_dense.py
XILINX_VITIS=/opt/Xilinx/2026.1/Vitis/ python repro_dense.py --parts vek385_base
XILINX_VITIS=/opt/Xilinx/2026.1/Vitis/ python repro_dense.py --keep   # retain projects
```

Run it with the `aie4ml` environment active (here: `/eos/user/e/enlupi/miniconda3/envs/aie4ml_env`).

**The VEK280 control is not optional.** It is in the default run on purpose. A stage that fails
on VEK385 and passes on VEK280 is attributable to the target; a stage that fails on both is not,
and three wrong conclusions were reached during this investigation by omitting exactly that
control. If you narrow to `--parts vek385_base`, you give up that attribution.

The weights are random normal, so they are essentially never 2:4-structured and the
weight-driven sparse path declines — you get the plain dense variant. The script asserts this and
warns loudly if a sparse variant is somehow selected, so it cannot quietly test something else.

## Expected result today

| part | gen | resolve | write | x86com | x86sim | compile | aiesim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `xilinx_vek280_base_202610_1` | AIE-ML | OK | OK | OK | OK | OK | OK |
| `vek385_base` | AIE-MLV2 | OK | OK | OK | OK | **FAIL** | **FAIL** |

On VEK385 the observed failure sequence is:

```
make compile            -> ERROR: [aiecompiler 77-6572] Could not load Device Part
                                  xc2ve3858-ssva2112-2MP-e-S
make aiesim  (exit 2)   -> [ERROR]: AIE Simulator config Json file missing,
                                    file: ./Work/config/scsim_config.json
predict(simulator='aie')-> RuntimeError: Make target "profile" failed
```

Three things worth noting in that trace:

- **`Work/` exists but `Work/config/scsim_config.json` does not.** The compile got far enough to
  create the directory before rejecting the part, so the presence of `Work/` is not evidence that
  a compile succeeded.
- **`predict(simulator='aie')` reports a `profile` failure**, three make targets away from the
  real cause. If you only ever call `predict`, this looks like a profiling problem.
- **`resolve` and `x86sim` both succeed**, and AIE-MLv2 dense output is bit-identical to AIE-ML
  dense. aie4ml's AIE-MLv2 support is fine as far as it can be exercised; only `--target=hw` is
  blocked.

If `compile` fails on **both** parts, something other than the missing part is wrong — check
`XILINX_VITIS`, the license, and that `xilinx_vek280_base_202610_1` matches your installed Vitis
(the platform names are version-stamped).

## Fixing it

Install full VE3858 / Versal AI Edge Gen 2 device support for Vitis 2026.1, then re-run
`probe_part.sh` and confirm the five **absent** rows above have become present. After that,
`repro_dense.py` should show `compile` and `aiesim` OK for `vek385_base` — which is the
prerequisite for investigating the AIE-MLv2 sparse layout question at all, since x86simulator
cannot validate that path.

Whether installing the part is sufficient has not been verified here; it is necessary, but there
may be more behind it.

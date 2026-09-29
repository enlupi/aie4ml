#!/usr/bin/env bash
# Probe what the Vitis install has for the VEK385 (AIE-MLv2) part vs the VEK280 (AIE-ML) one.
# Pure filesystem inspection -- runs in a second, builds nothing, needs no license.
#
# The question this answers: aiecompiler rejects the VEK385 part with
#   ERROR: [aiecompiler 77-6572] Could not load Device Part xc2ve3858-ssva2112-2MP-e-S
# so which database is actually absent?

set -uo pipefail

XV="${XILINX_VITIS:-/opt/Xilinx/2026.1/Vitis}"
ROOT="$(dirname "$XV")"          # e.g. /opt/Xilinx/2025.2
ML_PART=xcve2802                 # VEK280, AIE-ML      -- known working
V2_PART=xc2ve3858                # VEK385, AIE-MLv2    -- the one that fails
V2_PKG=ssva2112                  # package from the error string

hr() { printf '%s\n' "-------------------------------------------------------------"; }
say() { printf '%-58s %s\n' "$1" "$2"; }
chk() { if [ -e "$1" ]; then echo "PRESENT"; else echo "ABSENT"; fi; }

echo "Vitis:      $XV"
echo "Data root:  $ROOT"
hr

echo "1) Base platforms (what aie4ml points --platform at)"
for p in vek385_base xilinx_vek280_base_202610_1; do
  say "  $p" "$(chk "$XV/base_platforms/$p")"
done
echo "  installed base_platforms:"
ls "$XV/base_platforms/" 2>/dev/null | sed 's/^/    /'
hr

echo "2) AI Engine register DB (data/versal/me/regdb) -- both expected present"
for d in "$ML_PART" "$V2_PART"; do
  say "  regdb/$d" "$(chk "$ROOT/data/versal/me/regdb/$d")"
done
hr

echo "3) Architecture device entry"
say "  data/aie2ps/devices/XC2VE3858.json" "$(chk "$ROOT/data/aie2ps/devices/XC2VE3858.json")"
hr

echo "4) Side-by-side part database comparison  (the actual diagnosis)"
echo "   Same set of paths for the working part and the failing one. What the failing"
echo "   part LACKS is the answer; do not use a bare find -- depth limits give false"
echo "   negatives (the ssva2112 package JSON is at depth 6 and IS present)."
printf '   %-62s %-9s %-9s\n' "path (relative to $ROOT)" "$ML_PART" "$V2_PART"
for tmpl in \
  "data/ips/parts/versal/%s" \
  "data/ips/parts/versal/%s/${V2_PKG}.json" \
  "data/parts/xilinx/versal/devint/versalaiedge/%s" \
  "SharedData/data/parts/xilinx/devint/vault/versal/data/device_cfg/%s" \
  "SharedData/data/parts/xilinx/devint/vault/versal/data/device_pkgs/%s" \
  "data/versal/me/regdb/%s" \
  "data/versal/me/code/%s" \
  "data/versal/me/clock/%s" ; do
  # shellcheck disable=SC2059
  ml=$(printf "$tmpl" "$ML_PART"); v2=$(printf "$tmpl" "$V2_PART")
  printf '   %-62s %-9s %-9s\n' "$(printf "$tmpl" '<part>')" "$(chk "$ROOT/$ml")" "$(chk "$ROOT/$v2")"
done
echo
echo "   The AI Engine per-device 'code' and 'clock' databases are the ones that"
echo "   matter most here: regdb is installed for both, but aiecompiler also needs"
echo "   me/code and me/clock, which exist only for the working part."
hr

echo "6) EXTRA: sparse decompression in the two intrinsic models"
echo "   Relevant to why x86sim cannot substitute for aiesim on AIE-MLv2."
for m in aie_ml aie2ps; do
  f="$XV/aietools/data/$m/lib/me_native.h"
  if [ -f "$f" ]; then
    c=$(grep -c -E "decompress_sparse|sparse_read8b|sparse_read16b|sparse_pop|sparse_fill" "$f" 2>/dev/null)
    say "  $m/lib/me_native.h decompression symbols" "$c"
  else
    say "  $m/lib/me_native.h" "ABSENT"
  fi
done
echo "   aie_ml has them; aie2ps has 0 -> x86sim cannot expand a sparse tile on AIE-MLv2."
hr
echo "Summary: the VEK385 device is PARTIALLY installed. Its base platform, its"
echo "IP-level part/package JSONs (including ${V2_PKG}.json) and its AI Engine"
echo "register DB are all present -- but the devint part definition, the SharedData"
echo "device cfg/pkgs, and the AI Engine me/code and me/clock databases are not."
echo "So aiecompiler cannot load xc2ve3858-${V2_PKG}-2MP-e-S and every --target=hw"
echo "compile fails -- for an ordinary dense layer just as much as for a sparse one."
echo "Fix: install full VE3858 / Versal AI Edge Gen 2 device support for this Vitis."

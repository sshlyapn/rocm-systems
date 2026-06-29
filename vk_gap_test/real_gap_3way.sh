#!/usr/bin/env bash
# 3-way per-kernel inter-kernel gap on gfx1201:
#   A) stock ROCm 7.2.0 default   (/opt/rocm libamdhip64.so.7.2.70200, no PM4)
#   B) develop AQL                (patched 7.14 lib, HIP_PM4_GRAPH unset = PM4 off)
#   C) develop PM4                (patched 7.14 lib, HIP_PM4_GRAPH=1)
#
# gap/k = (e2e_ms*1000 - K*busy_us) / (K-1)   [us per interval between kernels]
# busy_us = median rocprof kernel duration measured under the SAME runtime
#           (PM4 invisible to rocprof, so C reuses B's busy).

set -u
cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0
K=2000; KRP=300; BIN=./hip_gap_graph.x
OURLIB=/home/sshliapn/code/rocm-libraries/projects/clr/build-gap/hipamd/lib
ROCRGAP=/home/sshliapn/code/rocm-libraries/projects/rocr-runtime/install-gap/lib
STOCK=/opt/rocm/lib

e2e_of(){ local lib="$1"; shift
  env -u HIP_PM4_GRAPH LD_LIBRARY_PATH="$lib" "$@" "$BIN" "$K" "$SPIN" "$N" 2>&1 \
    | grep -oE "E2E_WALL_MS = [0-9.]+" | grep -oE "[0-9.]+$"; }

busy_of(){ local lib="$1"; shift
  local d=/tmp/rp3_$$; rm -rf "$d"; mkdir -p "$d"
  env -u HIP_PM4_GRAPH LD_LIBRARY_PATH="$lib" "$@" rocprofv3 --kernel-trace \
      --output-format csv -d "$d" -o kt -- "$BIN" "$KRP" "$SPIN" "$N" >/dev/null 2>&1
  local csv; csv=$(find "$d" -name "*kernel_trace*.csv" | head -1)
  python3 - "$csv" <<'PY'
import sys,csv,statistics
d=[(int(r["End_Timestamp"])-int(r["Start_Timestamp"]))/1000.0
   for r in csv.DictReader(open(sys.argv[1])) if "gap_kernel" in r["Kernel_Name"]]
print(f"{statistics.median(d):.3f}" if d else "0")
PY
  rm -rf "$d"; }

gpk(){ python3 -c "e,b,K=$1,$2,$3; print(f'{(e*1000-K*b)/(K-1):.3f}')"; }  # us/interval

printf "%-6s %-7s | %-9s %-9s %-9s | %-9s %-9s %-9s\n" \
  spin n e2e7.2 e2edevAQL e2ePM4 gpk7.2 gpkdevAQL gpkPM4
echo "---------------------------------------------------------------------------------"
for cfg in "0 4096" "0 65536" "2000 65536"; do
  set -- $cfg; SPIN=$1; N=$2
  B72=$(busy_of "$STOCK"); BDEV=$(busy_of "$OURLIB:$ROCRGAP:$STOCK")
  E72=$(e2e_of "$STOCK")
  EDEV=$(e2e_of "$OURLIB:$ROCRGAP:$STOCK")
  EPM4=$(e2e_of "$OURLIB:$ROCRGAP:$STOCK" HIP_PM4_GRAPH=1)
  G72=$(gpk "$E72" "$B72" "$K"); GDEV=$(gpk "$EDEV" "$BDEV" "$K"); GPM4=$(gpk "$EPM4" "$BDEV" "$K")
  printf "%-6s %-7s | %-9s %-9s %-9s | %-9s %-9s %-9s\n" \
    "$SPIN" "$N" "$E72" "$EDEV" "$EPM4" "$G72" "$GDEV" "$GPM4"
done

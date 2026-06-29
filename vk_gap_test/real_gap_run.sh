#!/usr/bin/env bash
# Real inter-kernel gap on gfx1201, artifact-free.
#
# Method:
#   period      = real wall time per dispatch, from hipEvent around hipGraphLaunch
#                 (hip_gap_graph prints "period(span/K)" which equals wall_us/K; trustworthy).
#   busy_real   = true kernel duration from rocprofv3 --kernel-trace (CP hardware
#                 timestamps, no in-kernel min/max leakage).
#   gap_real    = period - busy_real, for both AQL and PM4 (kernel is byte-identical,
#                 same clocks, so busy_real is shared).
#   gap_rp(AQL) = rocprof's own Start[k+1]-End[k] (CP-side), cross-check for AQL only.
#
# PM4 is invisible to rocprofv3 (raw IB submission bypasses the traced AQL queue),
# so busy_real is taken from the AQL trace and reused for PM4.

set -u
cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0

# Patched PM4 CLR (RUNPATH already points to its matching install-gap ROCr).
# Apples-to-apples: SAME lib for both columns, toggling only HIP_PM4_GRAPH.
OURLIB=/home/sshliapn/code/rocm-libraries/projects/clr/build-gap/hipamd/lib
DEVLIB=$OURLIB   # AQL baseline = identical runtime with PM4 feature OFF

K=2000        # for wall-period (fast, accurate per-dispatch wall)
KRP=300       # for rocprof trace (smaller; keeps trace + overhead bounded)
BIN=./hip_gap_graph.x

e2e_of() {  # $1=LD_LIBRARY_PATH  $2..=pm4env ; reads K spin n from globals
            # echoes the directly measured total GPU wall time (ms) for the whole graph
  local lib="$1"; shift
  env -u HIP_PM4_GRAPH LD_LIBRARY_PATH="$lib" "$@" "$BIN" "$K" "$SPIN" "$N" 2>&1 \
    | grep -oE "E2E_WALL_MS = [0-9.]+" | grep -oE "[0-9.]+$"
}

busy_real_of() {  # rocprof kernel-trace under DEVLIB (AQL). echoes "busy gap_rp"
  local d=/tmp/rpf_$$_${SPIN}_${N}
  rm -rf "$d"; mkdir -p "$d"
  env LD_LIBRARY_PATH="$DEVLIB" rocprofv3 --kernel-trace --output-format csv \
      -d "$d" -o kt -- "$BIN" "$KRP" "$SPIN" "$N" >"$d/out.txt" 2>&1
  local csv; csv=$(find "$d" -name "*kernel_trace*.csv" | head -1)
  python3 - "$csv" <<'PY'
import sys, csv, statistics
rows=[]
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if "gap_kernel" in r["Kernel_Name"]:
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"])))
rows.sort()
dur=[(e-s)/1000.0 for s,e in rows]                       # us
gap=[(rows[i+1][0]-rows[i][1])/1000.0 for i in range(len(rows)-1)]
gap=[g for g in gap if g>0]                              # drop warmup/timed boundary negatives
def med(x): return statistics.median(x) if x else 0.0
print(f"{med(dur):.3f} {med(gap):.3f}")
PY
  rm -rf "$d"
}

echo "K (dispatches per graph) = $K   busy = pure rocprof kernel time (per kernel)"
echo "kern_ms = K * busy (total kernel compute; identical kernels => same for AQL & PM4)"
echo "e2e_ms  = directly measured total GPU wall (one event pair around hipGraphLaunch)"
echo "gap_ms     = e2e_ms - kern_ms (total time that is NOT kernel compute)"
echo "gap/k_us   = total gap / (K-1) intervals between kernels, in microseconds"
printf "%-6s %-7s | %-8s %-9s | %-9s %-9s %-9s | %-9s %-9s %-9s | %-7s\n" \
  spin n busy_us kern_ms e2eAQL gapAQL_ms gapAQL/k e2ePM4 gapPM4_ms gapPM4/k spdup
echo "----------------------------------------------------------------------------------------------------------------------"

for cfg in "0 4096" "0 65536" "200 65536" "2000 65536" "2000 16384"; do
  set -- $cfg; SPIN=$1; N=$2
  read BUSY GAPRP < <(busy_real_of)
  EAQL=$(e2e_of "$DEVLIB")                       # ms, PM4 off (var unset)
  EPM4=$(e2e_of "$OURLIB" HIP_PM4_GRAPH=1)       # ms, PM4 on
  python3 - "$SPIN" "$N" "$K" "$BUSY" "$EAQL" "$EPM4" <<'PY'
import sys
spin,n,K=sys.argv[1],sys.argv[2],int(sys.argv[3])
busy,eaql,epm4=map(float,sys.argv[4:7])
kern=K*busy/1000.0          # total kernel compute, ms (same kernels both paths)
gaql=eaql-kern; gpm4=epm4-kern
intervals=K-1               # number of gaps between K kernels
gaql_k=gaql*1000.0/intervals; gpm4_k=gpm4*1000.0/intervals   # per-interval gap, us
spd=eaql/epm4 if epm4>0 else 0
print(f"{spin:<6} {n:<7} | {busy:<8.3f} {kern:<9.3f} | {eaql:<9.4f} {gaql:<9.3f} {gaql_k:<9.3f} | {epm4:<9.4f} {gpm4:<9.3f} {gpm4_k:<9.3f} | {spd:<7.4f}")
PY
done

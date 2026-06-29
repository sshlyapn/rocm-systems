#!/usr/bin/env bash
# Block-count (occupancy) sweep of the AQL per-kernel gap, to look for the
# "gap step" seen on RDNA3 (~2.9 us at <=16 blocks, ~6.75 us at >=64 blocks).
#
# gap/k = (e2e_ms*1000 - K*busy_us) / (K-1)   [us per interval]
# busy   = median rocprof kernel duration (develop AQL, PM4 off).
# Also reports PM4 per-kernel gap for reference.

set -u
cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0
K=2000; KRP=300; BIN=./hip_gap_graph.x
SPIN=${1:-1000}     # fixed compute; pick to get tens-of-us busy like the RDNA3 table
OURLIB=/home/sshliapn/code/rocm-libraries/projects/clr/build-gap/hipamd/lib
ROCRGAP=/home/sshliapn/code/rocm-libraries/projects/rocr-runtime/install-gap/lib
STOCK=/opt/rocm/lib
LIB=$OURLIB:$ROCRGAP:$STOCK

e2e_of(){ env -u HIP_PM4_GRAPH LD_LIBRARY_PATH="$LIB" "$@" "$BIN" "$K" "$SPIN" "$N" 2>&1 \
    | grep -oE "E2E_WALL_MS = [0-9.]+" | grep -oE "[0-9.]+$"; }

busy_of(){ local d=/tmp/rpstep_$$; rm -rf "$d"; mkdir -p "$d"
  env -u HIP_PM4_GRAPH LD_LIBRARY_PATH="$LIB" rocprofv3 --kernel-trace \
      --output-format csv -d "$d" -o kt -- "$BIN" "$KRP" "$SPIN" "$N" >/dev/null 2>&1
  local csv; csv=$(find "$d" -name "*kernel_trace*.csv" | head -1)
  python3 - "$csv" <<'PY'
import sys,csv,statistics
d=[(int(r["End_Timestamp"])-int(r["Start_Timestamp"]))/1000.0
   for r in csv.DictReader(open(sys.argv[1])) if "gap_kernel" in r["Kernel_Name"]]
print(f"{statistics.median(d):.3f}" if d else "0")
PY
  rm -rf "$d"; }

echo "K=$K  spin=$SPIN  (AQL = PM4 off; gap/k = per-interval, us)"
printf "%-8s %-7s | %-8s | %-9s %-9s | %-9s %-9s\n" \
  n blocks busy_us e2eAQL_ms gapAQL/k e2ePM4_ms gapPM4/k
echo "-----------------------------------------------------------------------"
for N in 1024 4096 16384 65536 262144; do
  BLK=$(( (N+255)/256 ))
  B=$(busy_of)
  EA=$(e2e_of)
  EP=$(e2e_of HIP_PM4_GRAPH=1)
  python3 - "$N" "$BLK" "$B" "$EA" "$EP" "$K" <<'PY'
import sys
n,blk=sys.argv[1],sys.argv[2]; b,ea,ep=map(float,sys.argv[3:6]); K=int(sys.argv[6])
ga=(ea*1000-K*b)/(K-1); gp=(ep*1000-K*b)/(K-1)
print(f"{n:<8} {blk:<7} | {b:<8.3f} | {ea:<9.4f} {ga:<9.3f} | {ep:<9.4f} {gp:<9.3f}")
PY
done

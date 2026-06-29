#!/usr/bin/env bash
# Clean PM4 vs AQL inter-kernel gap using rocprofv3 for the authoritative per-kernel
# busy. The gap is the small difference of two large numbers (e2e - K*busy), so it
# needs the TRUE chained HW dispatch duration -- only rocprof's per-dispatch HW
# timestamps qualify (in-kernel envelope / isolated dispatch are off by more than the
# gap). rocprof cannot profile the PM4 path (a profiler forces AQL fallback), so:
#
#   busy     = median(End-Start) from rocprofv3 --kernel-trace (AQL run; busy is
#              dispatch-path independent)
#   AQL gap  = median(Start[k+1]-End[k]) straight from the same rocprof trace
#   PM4 gap  = (e2e_PM4 - K*busy)/(K-1), e2e_PM4 from the benchmark hipEvent (no prof)
#   AQL gap' = (e2e_AQL - K*busy)/(K-1), cross-check vs rocprof's direct AQL gap
#
# Run with clocks pinned (host: sudo gpu_pin_freq.sh pin 0 high) for stable numbers.
set -u
cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0
OURLIB=/home/sshliapn/code/rocm-libraries/projects/clr/build-gap/hipamd/lib
BIN=./hip_gap_graph.x
K=${K:-50}
SPIN=${SPIN:-200000}
N=${N:-16384}

med() { sort -n | awk '{a[NR]=$1} END{ if(NR==0){print 0} else {print a[int((NR+1)/2)]} }'; }

# 1) rocprofv3 kernel trace on the AQL run -> per-kernel busy + direct AQL gap.
rm -rf rp_run
LD_LIBRARY_PATH=/opt/rocm/lib rocprofv3 --kernel-trace -f csv -d rp_run -o kt -- \
    "$BIN" "$K" "$SPIN" "$N" >/dev/null 2>&1
KT=$(find rp_run -name '*kernel_trace*.csv' | head -1)
if [ -z "$KT" ]; then echo "ERROR: no rocprof csv produced"; exit 1; fi

# Start=$(NF-12), End=$(NF-11): indexed from the end so the comma-laden quoted kernel
# name does not shift the numeric trailing fields. Sort by start, drop 2 warmups.
awk -F, 'NR>1{print $(NF-12)","$(NF-11)}' "$KT" | sort -t, -k1,1n | tail -n +3 > rp_pairs.txt
busy_us=$(awk -F, '{print ($2-$1)/1000.0}' rp_pairs.txt | med)
rp_aql_gap_us=$(awk -F, '{if(pe>0) print ($1-pe)/1000.0; pe=$2}' rp_pairs.txt | med)

# 2) benchmark e2e (no profiler) for AQL and PM4, fed the rocprof busy.
run_e2e() {  # $1=mode -> "e2e_us gap_hw_us"
  local env_pm4=(); [ "$1" = "PM4" ] && env_pm4=(HIP_PM4_GRAPH=1)
  local best_e2e="" best_line=""
  for r in 1 2 3 4 5; do
    local s; s=$(env -u HIP_PM4_GRAPH KERN_US="$busy_us" LD_LIBRARY_PATH="$OURLIB" \
                 "${env_pm4[@]}" "$BIN" "$K" "$SPIN" "$N" 2>/dev/null | grep '^SUMMARY')
    local e; e=$(echo "$s" | sed -n 's/.*e2e_wall_us=\([0-9.]*\).*/\1/p')
    [ -z "$e" ] && continue
    if [ -z "$best_e2e" ] || awk "BEGIN{exit !($e<$best_e2e)}"; then best_e2e="$e"; best_line="$s"; fi
  done
  local e2e gap
  e2e=$(echo "$best_line"  | sed -n 's/.*e2e_wall_us=\([0-9.]*\).*/\1/p')
  gap=$(echo "$best_line"  | sed -n 's/.*gap_hw_us=\([0-9.]*\).*/\1/p')
  echo "$e2e $gap"
}
read aql_e2e aql_gap < <(run_e2e AQL)
read pm4_e2e pm4_gap < <(run_e2e PM4)

printf "\nrocprofv3 + benchmark gap (K=%s spin=%s n=%s, blocks=%s)\n" \
  "$K" "$SPIN" "$N" "$((N/256))"
printf "  per-kernel busy (rocprof median End-Start) : %.3f us\n" "$busy_us"
printf "  rocprof AQL gap  (direct Start[k+1]-End[k]) : %.3f us\n" "$rp_aql_gap_us"
echo   "  ----------------------------------------------------------"
printf "  %-5s | %-12s | %-11s\n" path e2e_us gap_us
printf "  %-5s | %-12.2f | %-11.3f\n" AQL "$aql_e2e" "$aql_gap"
printf "  %-5s | %-12.2f | %-11.3f\n" PM4 "$pm4_e2e" "$pm4_gap"
echo

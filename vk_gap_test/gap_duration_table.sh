#!/usr/bin/env bash
# Inter-kernel gap vs kernel DURATION (compute length), up to ~3000 us.
#
# Same decomposition as gap_table.sh, all from in-kernel readsteadycounter
# timestamps (no rocprofv3). Here the swept variable is the per-kernel compute
# duration (set via the spin loop count), at a fixed grid (blocks). This shows
# how the absolute inter-kernel gap behaves as kernels get longer, and how PM4
# replay compares to AQL across the whole range.
#
# us-per-spin is auto-calibrated once on this GPU/grid, then the spin needed for
# each target duration is derived. The achieved median kernel time is reported
# (med_kern_us), so you can see the true landed duration.
set -u
cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0

OURLIB=/home/sshliapn/code/rocm-libraries/projects/clr/build-gap/hipamd/lib
BIN=./hip_gap_graph.x
BLOCKS=${BLOCKS:-64}                 # fixed grid: n = BLOCKS*256
N=$(( BLOCKS * 256 ))
REPS=${REPS:-3}                      # best-of (min-e2e) per point
# Target per-kernel compute durations in microseconds.
DUR_LIST=${DUR_LIST:-"10 50 100 250 500 1000 2000 3000"}

run_raw() {  # $1=mode $2=K $3=spin -> best (min-e2e) SUMMARY line
  local mode="$1" K="$2" spin="$3" env_pm4=()
  [ "$mode" = "PM4" ] && env_pm4=(HIP_PM4_GRAPH=1)
  local best="" best_e2e=""
  for ((r = 0; r < REPS; r++)); do
    local s; s=$(env -u HIP_PM4_GRAPH LD_LIBRARY_PATH="$OURLIB" "${env_pm4[@]}" \
                 "$BIN" "$K" "$spin" "$N" 2>/dev/null | grep '^SUMMARY')
    [ -z "$s" ] && continue
    local e; e=$(echo "$s" | sed -n 's/.*e2e_us=\([0-9.]*\).*/\1/p')
    if [ -z "$best_e2e" ] || awk "BEGIN{exit !($e < $best_e2e)}"; then best="$s"; best_e2e="$e"; fi
  done
  echo "$best"
}

# K is scaled down for long kernels so each run stays well under ~0.3 s of GPU time.
k_for() {  # $1=dur_us -> dispatches
  awk "BEGIN{k=int(300000/$1); if(k<60)k=60; if(k>500)k=500; print k}"
}

# --- auto-calibrate us-per-spin (probe at a mid spin) ---
PROBE_SPIN=50000
probe=$(run_raw AQL 80 "$PROBE_SPIN")
probe_us=$(echo "$probe" | sed -n 's/.*med_kern_us=\([0-9.]*\).*/\1/p')
US_PER_SPIN=$(awk "BEGIN{print $probe_us/$PROBE_SPIN}")
echo "calibration: $probe_us us @ spin=$PROBE_SPIN  =>  ${US_PER_SPIN} us/spin  (blocks=$BLOCKS, n=$N)"
echo

spin_for() { awk "BEGIN{s=int($1/$US_PER_SPIN); if(s<1)s=1; print s}"; }

print_table() {  # $1=mode
  local mode="$1"
  echo "== $mode  (blocks=$BLOCKS, n=$N)  [pure in-kernel timestamps, no rocprof] =="
  printf "%-9s | %-3s | %-11s %-7s | %-12s %-12s %-11s %-9s\n" \
    target_us K spin med_kern e2e_us tot_comp_us tot_gap_us gap_us
  echo "----------+-----+-----------+---------+-----------------------------------------------------"
  for d in $DUR_LIST; do
    local K spin line
    K=$(k_for "$d"); spin=$(spin_for "$d")
    line=$(run_raw "$mode" "$K" "$spin")
    eval "$(echo "$line" | sed 's/SUMMARY //; s/ /;/g' | tr ';' '\n' | sed 's/^/L_/')" 2>/dev/null
    printf "%-9s | %-3s | %-11s %-7.2f | %-12.2f %-12.2f %-11.2f %-9.3f\n" \
      "$d" "$K" "$spin" "${L_med_kern_us:-0}" "${L_e2e_us:-0}" \
      "${L_total_compute_us:-0}" "${L_total_gap_us:-0}" "${L_gap_us:-0}"
  done
  echo
}

run_raw AQL 80 1000 >/dev/null 2>&1   # warmup
print_table AQL
print_table PM4

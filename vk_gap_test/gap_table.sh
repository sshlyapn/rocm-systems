#!/usr/bin/env bash
# Inter-kernel gap decomposition table, derived purely from in-kernel GPU
# timestamps (readsteadycounter atomicMin/atomicMax) -- NO rocprofv3.
#
# For a serialized K-kernel chain the GPU end-to-end span telescopes exactly:
#   e2e_gpu = he[K-1]-hs[0] = sum(busy) + sum(gap)
# so every column below comes from the same kernel timestamps and is self
# consistent: total_gap = e2e - total_compute, gap = total_gap / (K-1).
#
# Columns:
#   med_kern   median per-kernel compute time (us)
#   e2e        pure GPU end-to-end span of the whole graph (us)
#   tot_comp   total compute = sum of all kernels (us)
#   tot_gap    total gap = e2e - tot_comp (us)
#   gap        per-interval gap = tot_gap / (K-1) (us)   <-- the "gap" figure
set -u
cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0

OURLIB=/home/sshliapn/code/rocm-libraries/projects/clr/build-gap/hipamd/lib
BIN=./hip_gap_graph.x
K=${K:-2000}        # dispatches per graph (large => stable medians)
SPIN=${SPIN:-0}     # per-thread spin iters (0 => gap-dominated, pure dispatch)
BLOCKS_LIST=${BLOCKS_LIST:-"4 16 64 256 1024"}

REPS=${REPS:-5}     # repeats per point; keep the least host-perturbed (min e2e)

run_one() {  # $1=mode(AQL|PM4) $2=blocks -> echoes the best (min-e2e) SUMMARY line
  local mode="$1" blocks="$2" n=$(( $2 * 256 ))
  local env_pm4=()
  [ "$mode" = "PM4" ] && env_pm4=(HIP_PM4_GRAPH=1)
  local best=""; local best_e2e=""
  for ((r = 0; r < REPS; r++)); do
    local s; s=$(env -u HIP_PM4_GRAPH LD_LIBRARY_PATH="$OURLIB" "${env_pm4[@]}" \
                 "$BIN" "$K" "$SPIN" "$n" 2>/dev/null | grep '^SUMMARY')
    [ -z "$s" ] && continue
    local e; e=$(echo "$s" | sed -n 's/.*e2e_us=\([0-9.]*\).*/\1/p')
    if [ -z "$best_e2e" ] || awk "BEGIN{exit !($e < $best_e2e)}"; then
      best="$s"; best_e2e="$e"
    fi
  done
  echo "$best"
}

print_table() {  # $1=mode
  local mode="$1"
  echo "== $mode (K=$K dispatches, spin=$SPIN)  [pure in-kernel timestamps, no rocprof] =="
  printf "%-7s | %-10s %-10s %-11s %-10s %-9s\n" blocks med_kern_us e2e_us tot_comp_us tot_gap_us gap_us
  echo "--------+-------------------------------------------------------------------"
  for b in $BLOCKS_LIST; do
    local line; line=$(run_one "$mode" "$b")
    # shellcheck disable=SC2086
    eval "$(echo "$line" | sed 's/SUMMARY //; s/ /;/g' | tr ';' '\n' | sed 's/^/L_/')" 2>/dev/null
    printf "%-7s | %-10.4f %-10.4f %-11.4f %-10.4f %-9.4f\n" \
      "$b" "${L_med_kern_us:-0}" "${L_e2e_us:-0}" "${L_total_compute_us:-0}" \
      "${L_total_gap_us:-0}" "${L_gap_us:-0}"
  done
  echo
}

# warmup the runtime/clocks once
run_one AQL 16 >/dev/null 2>&1

print_table AQL
print_table PM4

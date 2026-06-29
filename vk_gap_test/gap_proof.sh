#!/usr/bin/env bash
# STANDALONE proof that PM4 graph replay beats AQL dispatch -- no rocprof, no
# profiler, no busy/gap split. The only metric is the whole-graph end-to-end wall
# time measured with a single hipEvent pair around hipGraphLaunch (GPU timeline).
#
# busy is identical for AQL and PM4 (same kernels/grid), so it cancels exactly:
#   saving/interval = (e2e_AQL - e2e_PM4) / (K-1)
# With spin=0 the kernels are ~0.4 us (<< gap), so e2e/K is itself ~ the absolute
# per-dispatch cost and the comparison is gap-dominated.
#
# Run with clocks pinned for stable numbers (host: sudo gpu_pin_freq.sh pin 0 high).
set -u
cd "$(dirname "$0")"
export HIP_VISIBLE_DEVICES=0
OURLIB=/home/sshliapn/code/rocm-libraries/projects/clr/build-gap/hipamd/lib
BIN=./hip_gap_graph.x
K=${K:-2000}
REPS=${REPS:-7}
# "K spin n" triples to sweep: gap-dominated small kernels first, then larger.
CASES=${CASES:-"0:4096 0:16384 200:16384 2000:16384 20000:16384"}

e2e_of() {  # $1=mode $2=spin $3=n -> best (min) e2e_wall_us
  local env_pm4=(); [ "$1" = "PM4" ] && env_pm4=(HIP_PM4_GRAPH=1)
  local best=""
  for ((r=0;r<REPS;r++)); do
    local v; v=$(env -u HIP_PM4_GRAPH LD_LIBRARY_PATH="$OURLIB" "${env_pm4[@]}" \
                 "$BIN" "$K" "$2" "$3" 2>/dev/null | sed -n 's/.*e2e_wall_us=\([0-9.]*\).*/\1/p')
    [ -z "$v" ] && continue
    if [ -z "$best" ] || awk "BEGIN{exit !($v<$best)}"; then best="$v"; fi
  done
  echo "$best"
}

env -u HIP_PM4_GRAPH LD_LIBRARY_PATH="$OURLIB" "$BIN" "$K" 0 4096 >/dev/null 2>&1  # warmup

printf "STANDALONE AQL-vs-PM4 e2e proof (K=%s dispatches, best-of-%s, hipEvent only)\n\n" "$K" "$REPS"
printf "%-6s %-7s | %-12s %-12s | %-11s %-11s | %-8s\n" \
  spin n e2e_AQL_us e2e_PM4_us aql/K_us pm4/K_us speedup
echo "-------------------+---------------------------+-------------------------+---------"
for c in $CASES; do
  spin=${c%%:*}; n=${c##*:}
  a=$(e2e_of AQL "$spin" "$n"); p=$(e2e_of PM4 "$spin" "$n")
  awk -v s="$spin" -v n="$n" -v a="$a" -v p="$p" -v K="$K" 'BEGIN{
    printf "%-6s %-7s | %-12.2f %-12.2f | %-11.4f %-11.4f | %-7.3fx\n",
      s, n, a, p, a/K, p/K, a/p;
  }'
done
echo
echo "saving/interval = (e2e_AQL - e2e_PM4)/(K-1):"
for c in $CASES; do
  spin=${c%%:*}; n=${c##*:}
  a=$(e2e_of AQL "$spin" "$n"); p=$(e2e_of PM4 "$spin" "$n")
  awk -v s="$spin" -v n="$n" -v a="$a" -v p="$p" -v K="$K" 'BEGIN{
    printf "  spin=%-6s n=%-7s : %.4f us/interval saved\n", s, n, (a-p)/(K-1);
  }'
done

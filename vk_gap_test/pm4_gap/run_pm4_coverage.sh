#!/usr/bin/env bash
# Build and run the HIP_PM4_GRAPH coverage test INSIDE hipvk-isolated-sshliapn.
# Runs the binary three ways against the patched runtime and diffs checksums:
#   baseline (AQL)            HIP_PM4_GRAPH unset
#   PM4 graph                 HIP_PM4_GRAPH=1
#   PM4 graph + scratch       HIP_PM4_GRAPH=1 HIP_PM4_GRAPH_SCRATCH=1
# Every per-scenario checksum must match the baseline. Plain ASCII only.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

ARCH="${ARCH:-gfx1100}"
CLR_LIB=/home/sshliapn/code/rocm-systems/projects/clr/build-gap/hipamd/lib
HIPCC=/opt/rocm/bin/hipcc
M="${M:-16384}"

echo "=== build (arch=$ARCH) ==="
"$HIPCC" -O2 -std=c++17 --offload-arch="$ARCH" -mwavefrontsize64 -c wave64_kernel.hip -o wave64_kernel.o
"$HIPCC" -O2 -std=c++17 --offload-arch="$ARCH" -c hip_pm4_coverage.cpp -o hip_pm4_coverage.o
"$HIPCC" -O2 -std=c++17 --offload-arch="$ARCH" hip_pm4_coverage.o wave64_kernel.o -o hip_pm4_coverage
echo "built: $HERE/hip_pm4_coverage"

export LD_LIBRARY_PATH="$CLR_LIB:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

run() {  # label  env...
    local label="$1"; shift
    echo "--- $label ---"
    env "$@" ./hip_pm4_coverage "$M" | grep -E 'checksum=|^#'
}

run baseline                 > cov_baseline.txt
run pm4        HIP_PM4_GRAPH=1                            > cov_pm4.txt
run pm4scratch HIP_PM4_GRAPH=1 HIP_PM4_GRAPH_SCRATCH=1    > cov_pm4scratch.txt

cat cov_baseline.txt

echo "=== diff: baseline vs PM4 (scenario checksums must be identical) ==="
fail=0
# Compare only the "name checksum=" lines.
join -j1 \
  <(grep 'checksum=' cov_baseline.txt | sort) \
  <(grep 'checksum=' cov_pm4scratch.txt | sort) \
  | while read -r name a b; do
      if [ "$a" = "$b" ]; then s="OK   "; else s="FAIL "; fi
      printf "  %s %-12s base=%s pm4=%s\n" "$s" "$name" "${a#checksum=}" "${b#checksum=}"
  done

if diff <(grep 'checksum=' cov_baseline.txt | sort) \
        <(grep 'checksum=' cov_pm4scratch.txt | sort) >/dev/null; then
    echo "RESULT: ALL SCENARIOS BIT-EXACT"
else
    echo "RESULT: MISMATCH (see FAIL rows above)"
    fail=1
fi
exit $fail

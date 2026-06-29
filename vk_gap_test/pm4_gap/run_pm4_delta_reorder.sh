#!/usr/bin/env bash
# Build and run the HIP_PM4_GRAPH coverage test INSIDE hipvk-isolated-sshliapn,
# validating the register delta-encoding (HIP_PM4_GRAPH_DELTA) and the IB-reorder
# (HIP_PM4_GRAPH_REORDER) levers. Every flag combination must produce checksums
# bit-identical to the baseline AQL path. Plain ASCII only.
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

run() {  # outfile  env...
    local out="$1"; shift
    env "$@" ./hip_pm4_coverage "$M" | grep 'checksum=' | sort > "$out"
}

# Baseline: AQL path (no PM4 graph at all).
run dr_baseline.txt

# All combos enable scratch too (HIP_PM4_GRAPH_SCRATCH=1) so the scratch scenario
# also exercises the PM4 path under each lever.
declare -A combos=(
  [pm4]="HIP_PM4_GRAPH=1 HIP_PM4_GRAPH_SCRATCH=1"
  [delta]="HIP_PM4_GRAPH=1 HIP_PM4_GRAPH_SCRATCH=1 HIP_PM4_GRAPH_DELTA=1"
  [reorder]="HIP_PM4_GRAPH=1 HIP_PM4_GRAPH_SCRATCH=1 HIP_PM4_GRAPH_REORDER=1"
  [both]="HIP_PM4_GRAPH=1 HIP_PM4_GRAPH_SCRATCH=1 HIP_PM4_GRAPH_DELTA=1 HIP_PM4_GRAPH_REORDER=1"
)

fail=0
for name in pm4 delta reorder both; do
    run "dr_${name}.txt" ${combos[$name]}
    if diff -q dr_baseline.txt "dr_${name}.txt" >/dev/null; then
        echo "OK    ${name}: all scenarios bit-exact vs baseline"
    else
        echo "FAIL  ${name}: MISMATCH vs baseline"
        join -j1 dr_baseline.txt "dr_${name}.txt" | while read -r s a c; do
            [ "$a" = "$c" ] || printf "    %-12s base=%s %s=%s\n" "$s" "${a#checksum=}" "$name" "${c#checksum=}"
        done
        fail=1
    fi
done

echo "=== baseline checksums ==="
cat dr_baseline.txt
exit $fail

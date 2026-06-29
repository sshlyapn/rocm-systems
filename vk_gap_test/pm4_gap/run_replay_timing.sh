#!/usr/bin/env bash
# Isolated host (CPU) cost of issuing one hipGraph replay: AQL graph path vs the
# PM4-IB path, swept over chain length N. Shows O(N) (AQL) vs O(1) (PM4). Run
# inside the docker container. Plain ASCII only.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

ARCH="${ARCH:-gfx1100}"
CLR_LIB=/home/sshliapn/code/rocm-systems/projects/clr/build-gap/hipamd/lib
HIPCC=/opt/rocm/bin/hipcc
M="${M:-256}"
R="${R:-4000}"
NS="${NS:-1 4 16 64 128 256}"

echo "=== build (arch=$ARCH) ==="
"$HIPCC" -O2 -std=c++17 --offload-arch="$ARCH" hip_replay_timing.cpp -o hip_replay_timing
echo "built: $HERE/hip_replay_timing"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

echo
echo "=== A) default HIP (stock /opt/rocm, AQL graph replay) ==="
for n in $NS; do env -u LD_LIBRARY_PATH ./hip_replay_timing "$n" "$M" "$R"; done

echo
echo "=== B) PM4-IB replay (patched lib + HIP_PM4_GRAPH=1 + DELTA=1) ==="
for n in $NS; do
    env LD_LIBRARY_PATH="$CLR_LIB" HIP_PM4_GRAPH=1 HIP_PM4_GRAPH_DELTA=1 \
        ./hip_replay_timing "$n" "$M" "$R"
done

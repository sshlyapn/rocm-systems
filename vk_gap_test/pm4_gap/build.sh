#!/usr/bin/env bash
# Build the PM4 gap microbench. Run INSIDE the hipvk-isolated-sshliapn container.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

KFDINC=/home/sshliapn/code/rocm-systems/projects/rocr-runtime/libhsakmt/tests/kfdtest/include
ROCMINC=/opt/rocm/include
LLVMBIN=/opt/rocm/llvm/bin

echo "=== assemble gfx1100 shader ==="
"$LLVMBIN/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1100 \
    -c gap_kernel.s -o gap_kernel.o
"$LLVMBIN/llvm-objcopy" -O binary --only-section=.text gap_kernel.o gap_kernel.bin
echo "shader .text bytes: $(stat -c %s gap_kernel.bin)"

echo "=== compile pm4_gap ==="
g++ -O2 -std=c++17 -Wall \
    -I"$KFDINC" -I"$ROCMINC" \
    pm4_gap.cpp \
    /opt/rocm/lib/libhsakmt.a \
    $(pkg-config --libs libdrm_amdgpu libdrm) \
    -lnuma -lpthread -lrt \
    -o pm4_gap
echo "built: $HERE/pm4_gap"

echo "=== compile real hipcc kernel + unbundle gfx1100 code object ==="
/opt/rocm/bin/hipcc --genco --offload-arch=gfx1100 realkern.hip -o realkern.co
"$LLVMBIN/clang-offload-bundler" --type=o --unbundle --input=realkern.co \
    --output=k_gfx1100.co --targets=hipv4-amdgcn-amd-amdhsa--gfx1100
echo "real code object: k_gfx1100.co"

echo "=== compile pm4_real (real-kernel PM4 dispatch, ABI replication) ==="
g++ -O2 -std=c++17 -Wall \
    -I"$KFDINC" -I"$ROCMINC" \
    pm4_real.cpp \
    /opt/rocm/lib/libhsakmt.a \
    $(pkg-config --libs libdrm_amdgpu libdrm) \
    -lnuma -lpthread -lrt \
    -o pm4_real
echo "built: $HERE/pm4_real"

echo "=== decode-layer kernels + unbundle ==="
/opt/rocm/bin/hipcc --genco --offload-arch=gfx1100 layer_kernels.hip -o layer.co
"$LLVMBIN/clang-offload-bundler" --type=o --unbundle --input=layer.co \
    --output=layer_gfx1100.co --targets=hipv4-amdgcn-amd-amdhsa--gfx1100
echo "decode-layer code object: layer_gfx1100.co (+ layer.co for hipModuleLoad)"

echo "=== compile pm4_layer (decode-layer chain via PM4) ==="
g++ -O2 -std=c++17 -Wall \
    -I"$KFDINC" -I"$ROCMINC" \
    pm4_layer.cpp \
    /opt/rocm/lib/libhsakmt.a \
    $(pkg-config --libs libdrm_amdgpu libdrm) \
    -lnuma -lpthread -lrt \
    -o pm4_layer
echo "built: $HERE/pm4_layer"

echo "=== compile hip_layer (HIP/AQL reference) ==="
/opt/rocm/bin/hipcc -O2 -std=c++17 hip_layer.cpp -o hip_layer
echo "built: $HERE/hip_layer"

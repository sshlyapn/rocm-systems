// wave64_kernel.hip
//
// Compiled with -mwavefrontsize64 so its kernel descriptor clears the
// WAVEFRONT_SIZE32 bit. Used by hip_pm4_coverage.cpp to exercise the PM4-graph
// wave-size path (CS_W32_EN must be cleared for this kernel).
//
// Plain ASCII only in comments.

#include <hip/hip_runtime.h>

__global__ void k_w64_madd(float* y, const float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = y[i] * 1.5f + w[i];
    }
}

void launch_w64_madd(float* y, const float* w, int n, hipStream_t s) {
    int blocks = (n + 255) / 256;
    hipLaunchKernelGGL(k_w64_madd, dim3(blocks), dim3(256), 0, s, y, w, n);
}

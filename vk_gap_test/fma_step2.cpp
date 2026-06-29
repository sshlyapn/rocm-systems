// Probe v2: per-lane (VALU) dependent FMA loop, WORK FMAs per iteration.
// Compare the doubling threshold (in iterations) for WORK=1 vs WORK=4 to tell
// apart an iteration-count threshold from an execution-time (~16 us) threshold.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef WORK
#define WORK 1
#endif

#define CK(x) do{ hipError_t e=(x); if(e){fprintf(stderr,"err %s\n",hipGetErrorString(e));exit(1);} }while(0)

__global__ void k(float* d, uint32_t spin) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < 256) {
        float v = d[gid];  // per-lane distinct -> VALU, like the real gap_kernel
        for (uint32_t i = 0; i < spin; ++i) {
#pragma unroll
            for (int w = 0; w < WORK; ++w) v = v * 1.0000001f + 1.0f;
        }
        d[gid] = v;
    }
}

int main(int argc, char** argv) {
    uint32_t spin = argc > 1 ? atoi(argv[1]) : 1000;
    float* d; CK(hipMalloc(&d, 256 * sizeof(float)));
    CK(hipMemset(d, 0, 256 * sizeof(float)));
    dim3 grid(1), block(256);
    for (int i = 0; i < 20; i++) k<<<grid, block>>>(d, spin);
    CK(hipDeviceSynchronize());
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const int N = 200;
    float best = 1e30f;
    for (int r = 0; r < 5; r++) {
        CK(hipEventRecord(a));
        for (int i = 0; i < N; i++) k<<<grid, block>>>(d, spin);
        CK(hipEventRecord(b));
        CK(hipEventSynchronize(b));
        float ms; CK(hipEventElapsedTime(&ms, a, b));
        float us = ms * 1000.0f / N;
        if (us < best) best = us;
    }
    printf("WORK=%d spin=%u per_launch_us=%.3f\n", WORK, spin, best);
    return 0;
}

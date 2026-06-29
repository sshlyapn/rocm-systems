// Minimal probe: dependent FMA spin loop, identical body to gap_kernel.
// Lets us set the INITIAL accumulator value to separate an iteration-count
// threshold from a value-magnitude threshold for the busy-time doubling.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do{ hipError_t e=(x); if(e){fprintf(stderr,"err %s\n",hipGetErrorString(e));exit(1);} }while(0)

__global__ void k(float* d, uint32_t spin, float init) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < 256) {
        float v = init + (float)gid * 0.0f;  // init, keep dependence on arg
        for (uint32_t i = 0; i < spin; ++i) v = v * 1.0000001f + 1.0f;
        d[gid] = v;
    }
}

int main(int argc, char** argv) {
    uint32_t spin = argc > 1 ? atoi(argv[1]) : 1000;
    float    init = argc > 2 ? atof(argv[2]) : 0.0f;
    float* d; CK(hipMalloc(&d, 256 * sizeof(float)));
    dim3 grid(1), block(256);
    // warmup
    for (int i = 0; i < 20; i++) k<<<grid, block>>>(d, spin, init);
    CK(hipDeviceSynchronize());
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const int N = 200;
    float best = 1e30f;
    for (int r = 0; r < 5; r++) {
        CK(hipEventRecord(a));
        for (int i = 0; i < N; i++) k<<<grid, block>>>(d, spin, init);
        CK(hipEventRecord(b));
        CK(hipEventSynchronize(b));
        float ms; CK(hipEventElapsedTime(&ms, a, b));
        float us = ms * 1000.0f / N;
        if (us < best) best = us;
    }
    printf("spin=%u init=%.1f per_launch_us=%.3f\n", spin, init, best);
    return 0;
}

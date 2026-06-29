// HIP isolation test: does the inter-kernel gap collapse with concurrency?
//
// Launches K kernels round-robin across N streams. Each stream uses its OWN
// buffer region, so kernels on DIFFERENT streams are independent (may overlap),
// while kernels on the SAME stream stay ordered + RAW-dependent. If the
// per-dispatch PERIOD (total_span / K) drops as N grows, the single-stream gap
// is serialization/dispatch-path overhead that the GPU can hide - i.e. it is NOT
// an unavoidable per-dispatch hardware cost.
//
// Also reports achieved concurrency = sum(kernel_busy) / total_span.
//
// Usage: hip_gap_multi [K] [spin] [n] [nstreams]
// Build: hipcc -O2 --offload-arch=gfx1100 hip_gap_multi.cpp -o hip_gap_multi

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIPCHECK(x)                                                                       \
    do {                                                                                  \
        hipError_t _e = (x);                                                              \
        if (_e != hipSuccess) {                                                           \
            fprintf(stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

__device__ __forceinline__ unsigned long long gpu_realtime() {
    return (unsigned long long)__builtin_readsteadycounter();
}

__global__ void gap_kernel(float * data, uint32_t spin, uint32_t n,
                           unsigned long long * tstart, unsigned long long * tend, uint32_t kidx) {
    if (threadIdx.x == 0) {
        atomicMin(&tstart[kidx], gpu_realtime());
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = data[gid];
        for (uint32_t i = 0; i < spin; ++i) v = v * 1.0000001f + 1.0f;
        data[gid] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMax(&tend[kidx], gpu_realtime());
    }
}

int main(int argc, char ** argv) {
    int      K        = argc > 1 ? atoi(argv[1]) : 3000;
    uint32_t spin     = argc > 2 ? (uint32_t)atoi(argv[2]) : 0;
    uint32_t n        = argc > 3 ? (uint32_t)atoi(argv[3]) : 4096;
    int      nstreams = argc > 4 ? atoi(argv[4]) : 1;

    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    printf("device : %s (%s)  K=%d spin=%u n=%u nstreams=%d\n", prop.name, prop.gcnArchName, K, spin, n, nstreams);

    size_t region = std::max(n, 64u);
    float * data;
    HIPCHECK(hipMalloc(&data, region * (size_t)nstreams * sizeof(float)));
    HIPCHECK(hipMemset(data, 0, region * (size_t)nstreams * sizeof(float)));
    unsigned long long *tstart, *tend;
    HIPCHECK(hipMalloc(&tstart, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMalloc(&tend, (size_t)K * sizeof(unsigned long long)));

    std::vector<hipStream_t> streams(nstreams);
    for (int s = 0; s < nstreams; s++) HIPCHECK(hipStreamCreateWithFlags(&streams[s], hipStreamNonBlocking));

    uint32_t blocks = (n + 255) / 256, threads = 256;
    auto reset_ts = [&]() {
        HIPCHECK(hipMemset(tstart, 0xFF, (size_t)K * sizeof(unsigned long long)));
        HIPCHECK(hipMemset(tend, 0x00, (size_t)K * sizeof(unsigned long long)));
    };
    auto launch_all = [&]() {
        for (int k = 0; k < K; k++) {
            int s = k % nstreams;
            gap_kernel<<<blocks, threads, 0, streams[s]>>>(data + (size_t)s * region, spin, n, tstart, tend,
                                                           (uint32_t)k);
        }
        for (int s = 0; s < nstreams; s++) HIPCHECK(hipStreamSynchronize(streams[s]));
    };

    reset_ts();
    launch_all();  // warmup

    reset_ts();
    hipEvent_t e0, e1;
    HIPCHECK(hipEventCreate(&e0));
    HIPCHECK(hipEventCreate(&e1));
    HIPCHECK(hipEventRecord(e0, streams[0]));
    launch_all();
    HIPCHECK(hipEventRecord(e1, streams[0]));
    HIPCHECK(hipEventSynchronize(e1));
    float wall_ms = 0;
    HIPCHECK(hipEventElapsedTime(&wall_ms, e0, e1));

    std::vector<unsigned long long> hs(K), he(K);
    HIPCHECK(hipMemcpy(hs.data(), tstart, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(he.data(), tend, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));

    unsigned long long tmin = ~0ULL, tmax = 0;
    double sum_busy_ticks = 0;
    for (int k = 0; k < K; k++) {
        tmin = std::min(tmin, hs[k]);
        tmax = std::max(tmax, he[k]);
        sum_busy_ticks += (double)(he[k] - hs[k]);
    }
    unsigned long long span_ticks = tmax - tmin;
    double ns_per_tick = (double)wall_ms * 1e6 / (double)span_ticks;

    double period_us = (double)span_ticks * ns_per_tick / 1000.0 / K;
    double busy_us   = sum_busy_ticks * ns_per_tick / 1000.0 / K;
    double concurrency = sum_busy_ticks / (double)span_ticks;

    printf("  calib %.4f ns/tick | period(span/K) = %.3f us | mean kernel busy = %.3f us | concurrency = %.2fx\n",
           ns_per_tick, period_us, busy_us, concurrency);
    return 0;
}

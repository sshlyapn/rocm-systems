// HIP isolation test: toggle the AQL "barrier bit" on the kernel dispatch packet.
//
// Default in-stream launch sets HSA_PACKET_HEADER_BARRIER on each dispatch packet,
// which forces the CP to wait for the previous packet to fully complete before
// processing the next (full per-dispatch serialization). hipExtLaunchKernel with
// hipExtAnyOrderLaunch clears that bit (rocclr/.../rocvirtual.cpp:4364-4367), so
// the CP can issue dispatches back-to-back.
//
// Same binary, same session, same clock state -> a clean A/B that isolates how
// much of the inter-kernel gap is the barrier-bit serialization vs unavoidable
// per-dispatch cost. (With any-order + RAW the result is racy; we only measure timing.)
//
// Usage: hip_gap_anyorder [K] [spin] [n] [anyorder 0|1]
// Build: hipcc -O2 --offload-arch=gfx1100 hip_gap_anyorder.cpp -o hip_gap_anyorder

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
    if (threadIdx.x == 0) atomicMin(&tstart[kidx], gpu_realtime());
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = data[gid];
        for (uint32_t i = 0; i < spin; ++i) v = v * 1.0000001f + 1.0f;
        data[gid] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(&tend[kidx], gpu_realtime());
}

int main(int argc, char ** argv) {
    int      K        = argc > 1 ? atoi(argv[1]) : 4000;
    uint32_t spin     = argc > 2 ? (uint32_t)atoi(argv[2]) : 0;
    uint32_t n        = argc > 3 ? (uint32_t)atoi(argv[3]) : 4096;
    int      anyorder = argc > 4 ? atoi(argv[4]) : 0;

    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    int flags = anyorder ? hipExtAnyOrderLaunch : 0;
    printf("device : %s (%s)  K=%d spin=%u n=%u  anyorder=%d (barrier bit %s)\n", prop.name, prop.gcnArchName,
           K, spin, n, anyorder, anyorder ? "CLEARED" : "SET");

    float * data;
    HIPCHECK(hipMalloc(&data, (size_t)std::max(n, 64u) * sizeof(float)));
    HIPCHECK(hipMemset(data, 0, (size_t)std::max(n, 64u) * sizeof(float)));
    unsigned long long *tstart, *tend;
    HIPCHECK(hipMalloc(&tstart, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMalloc(&tend, (size_t)K * sizeof(unsigned long long)));

    hipStream_t stream;
    HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    uint32_t blocks = (n + 255) / 256, threads = 256;

    auto reset_ts = [&]() {
        HIPCHECK(hipMemset(tstart, 0xFF, (size_t)K * sizeof(unsigned long long)));
        HIPCHECK(hipMemset(tend, 0x00, (size_t)K * sizeof(unsigned long long)));
    };

    std::vector<uint32_t> kidx(K);
    for (int k = 0; k < K; k++) kidx[k] = (uint32_t)k;
    auto launch_all = [&]() {
        for (int k = 0; k < K; k++) {
            void * args[] = { &data, &spin, &n, &tstart, &tend, &kidx[k] };
            HIPCHECK(hipExtLaunchKernel((void *)gap_kernel, dim3(blocks), dim3(threads), args, 0, stream, nullptr,
                                        nullptr, flags));
        }
        HIPCHECK(hipStreamSynchronize(stream));
    };

    reset_ts();
    launch_all();  // warmup
    reset_ts();
    hipEvent_t e0, e1;
    HIPCHECK(hipEventCreate(&e0));
    HIPCHECK(hipEventCreate(&e1));
    HIPCHECK(hipEventRecord(e0, stream));
    launch_all();
    HIPCHECK(hipEventRecord(e1, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    float wall_ms = 0;
    HIPCHECK(hipEventElapsedTime(&wall_ms, e0, e1));

    std::vector<unsigned long long> hs(K), he(K);
    HIPCHECK(hipMemcpy(hs.data(), tstart, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(he.data(), tend, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));

    unsigned long long tmin = ~0ULL, tmax = 0;
    double sum_busy = 0;
    for (int k = 0; k < K; k++) {
        tmin = std::min(tmin, hs[k]);
        tmax = std::max(tmax, he[k]);
        sum_busy += (double)(he[k] - hs[k]);
    }
    unsigned long long span = tmax - tmin;
    double ns_per_tick = (double)wall_ms * 1e6 / (double)span;
    printf("  calib %.4f ns/tick | period(span/K) = %.3f us | mean busy = %.3f us | concurrency = %.2fx\n",
           ns_per_tick, (double)span * ns_per_tick / 1000.0 / K, sum_busy * ns_per_tick / 1000.0 / K,
           sum_busy / (double)span);
    return 0;
}

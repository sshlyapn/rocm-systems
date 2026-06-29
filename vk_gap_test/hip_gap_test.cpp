// HIP inter-kernel gap microbenchmark - mirror of the Vulkan vk_gap_test.
//
// Launches K dependent compute kernels into one stream. In a HIP stream,
// consecutive kernels are ordered and (by default) carry acquire/release cache
// fences between them - the HIP/HSA analog of the Vulkan SHADER_WRITE->SHADER_READ
// barrier. Each kernel reads+writes data[gid] (RAW dependency across kernels).
//
// Gap is measured on the GPU timeline using the on-die wall clock
// s_memrealtime() (same fixed-frequency counter PAL's profiler uses):
//   tstart[k] = earliest block start time of kernel k (atomicMin)
//   tend[k]   = latest  block end   time of kernel k (atomicMax)
//   kernel[k] = tend[k]   - tstart[k]
//   gap[k]    = tstart[k+1] - tend[k]   (CP relaunch + cache fences + drain)
// Ticks are converted to us via self-calibration against a hipEvent wall timer.
//
// Usage: hip_gap_test [K] [spin] [n]
// Build: hipcc -O2 --offload-arch=gfx1100 hip_gap_test.cpp -o hip_gap_test

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// RDNA3 (gfx11) has no s_memrealtime; the fixed-freq global wall clock is read
// via s_sendmsg_rtn(GET_REALTIME). 0x83 = MSG_RTN_GET_REALTIME.
__device__ __forceinline__ unsigned long long gpu_realtime() {
    // Portable LLVM steady counter. On gfx11 lowers to s_sendmsg_rtn(GET_REALTIME),
    // on CDNA to s_memrealtime - both the fixed-freq global wall clock.
    return (unsigned long long)__builtin_readsteadycounter();
}

#define HIPCHECK(x)                                                                       \
    do {                                                                                  \
        hipError_t _e = (x);                                                              \
        if (_e != hipSuccess) {                                                           \
            fprintf(stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

__global__ void gap_kernel(float * data, uint32_t spin, uint32_t n,
                           unsigned long long * tstart, unsigned long long * tend, uint32_t kidx) {
    if (threadIdx.x == 0) {
        unsigned long long t = gpu_realtime();
        atomicMin(&tstart[kidx], t);
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = data[gid];
        for (uint32_t i = 0; i < spin; ++i) {
            v = v * 1.0000001f + 1.0f;
        }
        data[gid] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long t = gpu_realtime();
        atomicMax(&tend[kidx], t);
    }
}

int main(int argc, char ** argv) {
    int      K    = argc > 1 ? atoi(argv[1]) : 3000;
    uint32_t spin = argc > 2 ? (uint32_t)atoi(argv[2]) : 0;
    uint32_t n    = argc > 3 ? (uint32_t)atoi(argv[3]) : 4096;

    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    printf("device : %s  (gcnArch %s)\n", prop.name, prop.gcnArchName);
    printf("config : K=%d kernels  spin=%u  n=%u (%u blocks of 256)\n", K, spin, n, (n + 255) / 256);

    float * data;
    HIPCHECK(hipMalloc(&data, (size_t)std::max(n, 64u) * sizeof(float)));
    HIPCHECK(hipMemset(data, 0, (size_t)std::max(n, 64u) * sizeof(float)));
    unsigned long long *tstart, *tend;
    HIPCHECK(hipMalloc(&tstart, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMalloc(&tend, (size_t)K * sizeof(unsigned long long)));

    hipStream_t stream;
    HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    uint32_t blocks  = (n + 255) / 256;
    uint32_t threads = 256;

    auto reset_ts = [&]() {
        HIPCHECK(hipMemset(tstart, 0xFF, (size_t)K * sizeof(unsigned long long))); // = ULLONG_MAX
        HIPCHECK(hipMemset(tend, 0x00, (size_t)K * sizeof(unsigned long long)));
    };

    // warmup (clock ramp, code cache) - not timed
    reset_ts();
    for (int k = 0; k < K; k++) {
        gap_kernel<<<blocks, threads, 0, stream>>>(data, spin, n, tstart, tend, (uint32_t)k);
    }
    HIPCHECK(hipStreamSynchronize(stream));

    // timed
    reset_ts();
    hipEvent_t e0, e1;
    HIPCHECK(hipEventCreate(&e0));
    HIPCHECK(hipEventCreate(&e1));
    HIPCHECK(hipEventRecord(e0, stream));
    for (int k = 0; k < K; k++) {
        gap_kernel<<<blocks, threads, 0, stream>>>(data, spin, n, tstart, tend, (uint32_t)k);
    }
    HIPCHECK(hipEventRecord(e1, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    float wall_ms = 0;
    HIPCHECK(hipEventElapsedTime(&wall_ms, e0, e1));

    std::vector<unsigned long long> hs(K), he(K);
    HIPCHECK(hipMemcpy(hs.data(), tstart, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(he.data(), tend, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));

    // -------- CP-boundary pass: hipEvent before/after each kernel ----------
    // hipEventRecord inserts a CP barrier packet that writes a GPU timestamp -
    // the true analog of Vulkan's vkCmdWriteTimestamp(TOP/BOTTOM_OF_PIPE). This
    // puts the post-dispatch cache flush INSIDE "kernel" (like Vulkan), so
    // gap_cp = afterEvent[k] -> beforeEvent[k+1] is the apples-to-apples CP turnaround.
    std::vector<hipEvent_t> be(K), af(K);
    for (int k = 0; k < K; k++) {
        HIPCHECK(hipEventCreate(&be[k]));
        HIPCHECK(hipEventCreate(&af[k]));
    }
    for (int k = 0; k < K; k++) {
        HIPCHECK(hipEventRecord(be[k], stream));
        gap_kernel<<<blocks, threads, 0, stream>>>(data, spin, n, tstart, tend, (uint32_t)k);
        HIPCHECK(hipEventRecord(af[k], stream));
    }
    HIPCHECK(hipStreamSynchronize(stream));
    std::vector<double> kern_cp, gap_cp;
    for (int k = 0; k < K; k++) {
        float ms = 0;
        HIPCHECK(hipEventElapsedTime(&ms, be[k], af[k]));
        kern_cp.push_back(ms * 1000.0);
        if (k < K - 1) {
            HIPCHECK(hipEventElapsedTime(&ms, af[k], be[k + 1]));
            gap_cp.push_back(ms * 1000.0);
        }
    }

    // self-calibrate ticks->us using the wall-clock event over the whole batch
    unsigned long long span_ticks = he[K - 1] - hs[0];
    double ns_per_tick = (double)wall_ms * 1e6 / (double)span_ticks;
    auto to_us = [&](long long ticks) { return (double)ticks * ns_per_tick / 1000.0; };

    std::vector<double> kern, gap;
    for (int k = 0; k < K; k++) {
        kern.push_back(to_us((long long)(he[k] - hs[k])));
        if (k < K - 1) gap.push_back(to_us((long long)((long long)hs[k + 1] - (long long)he[k])));
    }

    int skip = std::min(50, K / 10);
    auto stats = [&](std::vector<double> v) {
        std::vector<double> s(v.begin() + std::min((size_t)skip, v.size()), v.end());
        std::sort(s.begin(), s.end());
        double sum = 0;
        for (double x : s) sum += x;
        double mean = s.empty() ? 0 : sum / s.size();
        return std::vector<double>{ mean, s.empty() ? 0 : s[s.size() / 2],
                                    s.empty() ? 0 : s[(size_t)(s.size() * 0.05)],
                                    s.empty() ? 0 : s[(size_t)(s.size() * 0.95)] };
    };
    auto ks = stats(kern), gs = stats(gap);
    auto kc = stats(kern_cp), gc = stats(gap_cp);

    printf("calib  : wall %.3f ms over %llu ticks -> %.4f ns/tick (100MHz wall clock)\n",
           wall_ms, span_ticks, ns_per_tick);
    printf("\n  [A] wave-timeline  (in-kernel realtime: first-wave-start -> last-wave-end)\n");
    printf("      kernel : median %7.3f us   (pure shader busy, excludes CP prelude + post-flush)\n", ks[1]);
    printf("      GAP    : median %7.3f us   (last-wave-end -> next-first-wave-start: flush+CP+acquire)\n", gs[1]);
    printf("\n  [B] CP-boundary    (hipEvent before/after = Vulkan TOP/BOTTOM_OF_PIPE analog)\n");
    printf("      kernel : median %7.3f us   (includes post-dispatch cache flush, like Vulkan)\n", kc[1]);
    printf("      GAP    : median %7.3f us   (CP turnaround between dispatches, apples-to-apples)\n", gc[1]);
    printf("\n  >>> per-dispatch PERIOD (reference-point invariant) = %.3f us/kernel\n",
           to_us((long long)span_ticks) / K);
    return 0;
}

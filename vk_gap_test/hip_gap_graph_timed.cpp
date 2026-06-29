// HIP inter-dispatch gap microbench, DURATION-isolated variant.
//
// Builds a K-node linear-chain HIP graph (node k depends on node k-1). Each
// kernel isolates DURATION from workload: every block reads one float, busy
// waits on the fixed-frequency steady counter (__builtin_readsteadycounter)
// until a target_us deadline (no memory traffic, no compute in the wait loop),
// writes one float, and records per-kernel start/end via atomicMin/atomicMax
// for optional in-kernel timing. The authoritative timing source is rocprofv3
// kernel-trace (Start/End); the in-kernel atomics are a cross-check.
//
// The steady-counter frequency is self-calibrated against a hipEvent wall timer
// at startup so target_us is accurate regardless of the actual counter rate.
//
// Note: measured kernel busy != target_us. With n=4096 there are 16 blocks that
// run in a few sequential waves, so busy is a small multiple of target_us.
// Always report measured busy (rocprof End-Start), not target_us.
//
// Usage: hip_gap_graph_timed [K] [target_us] [n]
// Build: hipcc -O2 --offload-arch=gfx1201 hip_gap_graph_timed.cpp -o hip_gap_graph_timed.x

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Portable LLVM steady counter. On gfx11/gfx12 this lowers to the fixed
// frequency global wall clock (s_sendmsg_rtn GET_REALTIME / s_memrealtime).
__device__ __forceinline__ unsigned long long gpu_realtime() {
    return (unsigned long long)__builtin_readsteadycounter();
}

#define HIPCHECK(x)                                                                                \
    do {                                                                                           \
        hipError_t _e = (x);                                                                       \
        if (_e != hipSuccess) {                                                                    \
            fprintf(stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(_e), __FILE__,          \
                    __LINE__);                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// Duration-isolated kernel: busy-wait on the steady counter until target_ticks
// have elapsed since this block's entry. One float in, one float out.
__global__ void gap_kernel(float* data, unsigned long long target_ticks, uint32_t n,
                           unsigned long long* tstart, unsigned long long* tend, uint32_t kidx) {
    unsigned long long t0 = gpu_realtime();
    if (threadIdx.x == 0) atomicMin(&tstart[kidx], t0);
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (gid < n) ? data[gid] : 0.0f;
    while (gpu_realtime() - t0 < target_ticks) {
        // spin on the wall clock only -- no memory/compute work
    }
    if (gid < n) data[gid] = v + 1.0f;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long t1 = gpu_realtime();
        atomicMax(&tend[kidx], t1);
    }
}

// Calibration kernel: bracket a fixed compute loop with counter reads. Timed
// with hipEvent on the host to recover ticks-per-us (not circular: the loop is
// compute, the reference is the hipEvent wall clock).
__global__ void calib_kernel(unsigned long long* out, int iters) {
    unsigned long long t0 = gpu_realtime();
    float v = 1.0f;
    for (int i = 0; i < iters; ++i) v = v * 1.0000001f + 1.0f;
    unsigned long long t1 = gpu_realtime();
    if (v < 0.0f) out[2] = (unsigned long long)v;  // defeat dead-code elimination
    out[0] = t0;
    out[1] = t1;
}

int main(int argc, char** argv) {
    int      K         = argc > 1 ? atoi(argv[1]) : 300;
    double   target_us = argc > 2 ? atof(argv[2]) : 20.0;
    uint32_t n         = argc > 3 ? (uint32_t)atoi(argv[3]) : 4096;

    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));

    // ---- calibrate steady-counter frequency against hipEvent wall time ----
    unsigned long long* dcal;
    HIPCHECK(hipMalloc(&dcal, 3 * sizeof(unsigned long long)));
    hipEvent_t c0, c1;
    HIPCHECK(hipEventCreate(&c0));
    HIPCHECK(hipEventCreate(&c1));
    calib_kernel<<<1, 1>>>(dcal, 1 << 22);  // warmup
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipEventRecord(c0));
    calib_kernel<<<1, 1>>>(dcal, 1 << 24);
    HIPCHECK(hipEventRecord(c1));
    HIPCHECK(hipEventSynchronize(c1));
    float calib_ms = 0;
    HIPCHECK(hipEventElapsedTime(&calib_ms, c0, c1));
    unsigned long long hcal[3] = {0, 0, 0};
    HIPCHECK(hipMemcpy(hcal, dcal, 3 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    double ticks            = (double)(hcal[1] - hcal[0]);
    double ticks_per_us     = ticks / ((double)calib_ms * 1000.0);
    unsigned long long ttgt = (unsigned long long)(target_us * ticks_per_us + 0.5);

    printf("device : %s  (gcnArch %s)\n", prop.name, prop.gcnArchName);
    printf("calib  : %.0f ticks over %.3f ms -> %.3f ticks/us (%.1f MHz steady counter)\n", ticks,
           calib_ms, ticks_per_us, ticks_per_us);
    printf("config : K=%d  target_us=%.1f (=%llu ticks)  n=%u (%u blocks of 256)\n", K, target_us,
           ttgt, n, (n + 255) / 256);

    float* data;
    HIPCHECK(hipMalloc(&data, (size_t)std::max(n, 64u) * sizeof(float)));
    HIPCHECK(hipMemset(data, 0, (size_t)std::max(n, 64u) * sizeof(float)));
    unsigned long long *tstart, *tend;
    HIPCHECK(hipMalloc(&tstart, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMalloc(&tend, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMemset(tstart, 0xFF, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMemset(tend, 0x00, (size_t)K * sizeof(unsigned long long)));

    uint32_t blocks = (n + 255) / 256, threads = 256;

    // ---- build the K-node linear-chain graph ----
    hipGraph_t graph;
    HIPCHECK(hipGraphCreate(&graph, 0));
    std::vector<hipGraphNode_t> nodes(K);
    for (int k = 0; k < K; k++) {
        uint32_t            kidx = (uint32_t)k;
        void*               args[] = {&data, &ttgt, &n, &tstart, &tend, &kidx};
        hipKernelNodeParams p      = {};
        p.func                     = (void*)gap_kernel;
        p.gridDim                  = dim3(blocks, 1, 1);
        p.blockDim                 = dim3(threads, 1, 1);
        p.sharedMemBytes           = 0;
        p.kernelParams             = args;
        p.extra                    = nullptr;
        hipGraphNode_t deps[1];
        size_t         ndeps = 0;
        if (k > 0) {
            deps[0] = nodes[k - 1];
            ndeps   = 1;
        }
        HIPCHECK(hipGraphAddKernelNode(&nodes[k], graph, ndeps ? deps : nullptr, ndeps, &p));
    }

    hipGraphExec_t exec;
    HIPCHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    hipStream_t stream;
    HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    HIPCHECK(hipGraphLaunch(exec, stream));  // warmup
    HIPCHECK(hipStreamSynchronize(stream));
    // reset atomic timestamp buffers so the cross-check sees only the timed pass
    HIPCHECK(hipMemset(tstart, 0xFF, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMemset(tend, 0x00, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipStreamSynchronize(stream));

    // Dedicated capture launch for the in-kernel atomics. Read them back BEFORE any
    // further launch contaminates the atomicMin/atomicMax slots. This is a
    // profiler-free direct gap measurement: under rocprofv3 the runtime falls back to
    // AQL (pm4TracingArmed), so a kernel-trace cannot observe the real PM4 path -- only
    // this in-kernel / e2e route can.
    HIPCHECK(hipGraphLaunch(exec, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    std::vector<unsigned long long> hs(K), he(K);
    HIPCHECK(hipMemcpy(hs.data(), tstart, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(he.data(), tend, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));

    // Profiler-free whole-graph timing: per-dispatch PERIOD = e2e / K.
    hipEvent_t g0, g1;
    HIPCHECK(hipEventCreate(&g0));
    HIPCHECK(hipEventCreate(&g1));
    double best_ms = 1e30;
    double best_ms_host = 1e30;
    for (int r = 0; r < 5; r++) {
        auto h0 = std::chrono::high_resolution_clock::now();
        HIPCHECK(hipEventRecord(g0, stream));
        HIPCHECK(hipGraphLaunch(exec, stream));
        HIPCHECK(hipEventRecord(g1, stream));
        HIPCHECK(hipStreamSynchronize(stream));
        auto h1 = std::chrono::high_resolution_clock::now();
        double host_ms =
            std::chrono::duration<double, std::milli>(h1 - h0).count();
        best_ms_host = std::min(best_ms_host, host_ms);
        float ms = 0;
        HIPCHECK(hipEventElapsedTime(&ms, g0, g1));
        best_ms = std::min(best_ms, (double)ms);
    }
    double period_us = best_ms * 1000.0 / K;
    double period_us_host = best_ms_host * 1000.0 / K;

    auto to_us = [&](long long t) { return (double)t / ticks_per_us; };
    std::vector<double> busy, gap;
    for (int k = 0; k < K; k++) {
        if (hs[k] == ~0ull || he[k] == 0ull) continue;
        busy.push_back(to_us((long long)(he[k] - hs[k])));
        if (k < K - 1 && hs[k + 1] != ~0ull)
            gap.push_back(to_us((long long)((long long)hs[k + 1] - (long long)he[k])));
    }
    auto med = [&](std::vector<double> v) {
        if (v.empty()) return 0.0;
        size_t skip = std::min((size_t)50, v.size() / 10);
        std::vector<double> s(v.begin() + skip, v.end());
        std::sort(s.begin(), s.end());
        return s.empty() ? 0.0 : s[s.size() / 2];
    };
    printf("inkernel: busy_med=%.2f us  gap_med=%.2f us  (atomic cross-check)\n", med(busy),
           med(gap));
    printf("e2e     : period=%.3f us/dispatch (hipEvent)  host_period=%.3f us/dispatch (chrono)  (best-of-5, K=%d) PERIOD\n",
           period_us, period_us_host, K);
    return 0;
}

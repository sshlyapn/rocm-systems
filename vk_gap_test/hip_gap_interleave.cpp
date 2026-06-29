// Interleave control for the dispatch-gap microbench: identical to
// hip_gap_graph_timed.cpp but the K-node chain alternates TWO distinct
// templated kernels (gapKernel<1>, gapKernel<3>). If the staircase persists
// with alternating kernels, the gap is a genuine CP completion-wait effect and
// not a same-kernel fast-path artifact.
//
// Usage: hip_gap_interleave [K] [target_us] [n]
// Build: hipcc -O2 --offload-arch=gfx1201 hip_gap_interleave.cpp -o hip_gap_interleave.x

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

// Two distinct instantiations are emitted (ADD=1 and ADD=3); both busy-wait the
// same deadline so their cost is identical. Name contains "gapKernel" for the
// rocprof kernel-trace filter.
template <int ADD>
__global__ void gapKernel(float* data, unsigned long long target_ticks, uint32_t n,
                          unsigned long long* tstart, unsigned long long* tend, uint32_t kidx) {
    unsigned long long t0 = gpu_realtime();
    if (threadIdx.x == 0) atomicMin(&tstart[kidx], t0);
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (gid < n) ? data[gid] : 0.0f;
    while (gpu_realtime() - t0 < target_ticks) {
    }
    if (gid < n) data[gid] = v + (float)ADD;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long t1 = gpu_realtime();
        atomicMax(&tend[kidx], t1);
    }
}

__global__ void calib_kernel(unsigned long long* out, int iters) {
    unsigned long long t0 = gpu_realtime();
    float v = 1.0f;
    for (int i = 0; i < iters; ++i) v = v * 1.0000001f + 1.0f;
    unsigned long long t1 = gpu_realtime();
    if (v < 0.0f) out[2] = (unsigned long long)v;
    out[0] = t0;
    out[1] = t1;
}

int main(int argc, char** argv) {
    int      K         = argc > 1 ? atoi(argv[1]) : 300;
    double   target_us = argc > 2 ? atof(argv[2]) : 20.0;
    uint32_t n         = argc > 3 ? (uint32_t)atoi(argv[3]) : 4096;

    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));

    unsigned long long* dcal;
    HIPCHECK(hipMalloc(&dcal, 3 * sizeof(unsigned long long)));
    hipEvent_t c0, c1;
    HIPCHECK(hipEventCreate(&c0));
    HIPCHECK(hipEventCreate(&c1));
    calib_kernel<<<1, 1>>>(dcal, 1 << 22);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipEventRecord(c0));
    calib_kernel<<<1, 1>>>(dcal, 1 << 24);
    HIPCHECK(hipEventRecord(c1));
    HIPCHECK(hipEventSynchronize(c1));
    float calib_ms = 0;
    HIPCHECK(hipEventElapsedTime(&calib_ms, c0, c1));
    unsigned long long hcal[3] = {0, 0, 0};
    HIPCHECK(hipMemcpy(hcal, dcal, 3 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    double ticks_per_us     = (double)(hcal[1] - hcal[0]) / ((double)calib_ms * 1000.0);
    unsigned long long ttgt = (unsigned long long)(target_us * ticks_per_us + 0.5);

    printf("device : %s  (gcnArch %s)\n", prop.name, prop.gcnArchName);
    printf("calib  : %.3f ticks/us (%.1f MHz)\n", ticks_per_us, ticks_per_us);
    printf("config : K=%d  target_us=%.1f (=%llu ticks)  n=%u  INTERLEAVED gapKernel<1>/<3>\n", K,
           target_us, ttgt, n);

    float* data;
    HIPCHECK(hipMalloc(&data, (size_t)std::max(n, 64u) * sizeof(float)));
    HIPCHECK(hipMemset(data, 0, (size_t)std::max(n, 64u) * sizeof(float)));
    unsigned long long *tstart, *tend;
    HIPCHECK(hipMalloc(&tstart, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMalloc(&tend, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMemset(tstart, 0xFF, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMemset(tend, 0x00, (size_t)K * sizeof(unsigned long long)));

    uint32_t blocks = (n + 255) / 256, threads = 256;
    void*    funcs[2] = {(void*)gapKernel<1>, (void*)gapKernel<3>};

    hipGraph_t graph;
    HIPCHECK(hipGraphCreate(&graph, 0));
    std::vector<hipGraphNode_t> nodes(K);
    for (int k = 0; k < K; k++) {
        uint32_t            kidx   = (uint32_t)k;
        void*               args[] = {&data, &ttgt, &n, &tstart, &tend, &kidx};
        hipKernelNodeParams p      = {};
        p.func                     = funcs[k & 1];
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
    HIPCHECK(hipGraphLaunch(exec, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipMemset(tstart, 0xFF, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMemset(tend, 0x00, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipGraphLaunch(exec, stream));
    HIPCHECK(hipStreamSynchronize(stream));

    std::vector<unsigned long long> hs(K), he(K);
    HIPCHECK(hipMemcpy(hs.data(), tstart, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(he.data(), tend, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));
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
        size_t              skip = std::min((size_t)50, v.size() / 10);
        std::vector<double> s(v.begin() + skip, v.end());
        std::sort(s.begin(), s.end());
        return s.empty() ? 0.0 : s[s.size() / 2];
    };
    printf("inkernel: busy_med=%.2f us  gap_med=%.2f us\n", med(busy), med(gap));
    return 0;
}

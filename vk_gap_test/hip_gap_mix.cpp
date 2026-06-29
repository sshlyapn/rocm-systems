// Short/long interleave: a linear-chain HIP graph whose nodes ALTERNATE between
// a short-spin kernel and a long-spin kernel. Even nodes = gap_kernel<1> with
// spin_short, odd nodes = gap_kernel<3> with spin_long. Measures profiler-free
// whole-graph e2e (best-of-7); pair with rocprofv3 --kernel-trace to read the
// two busy clusters. Same n / dispatch path as hip_gap_graph.x.
//
// Usage: hip_gap_mix.x <spin_short> <spin_long> [n] [K]
// Env  : HIP_PM4_GRAPH=1 selects the PM4 IB replay path.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIPCHECK(cmd)                                                                              \
    do {                                                                                           \
        hipError_t _e = (cmd);                                                                     \
        if (_e != hipSuccess) {                                                                    \
            fprintf(stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(_e), __FILE__,          \
                    __LINE__);                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

template <int ADD>
__global__ void gap_kernel(float* data, uint32_t spin, uint32_t n) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = data[gid];
        for (uint32_t i = 0; i < spin; ++i) v = v * 1.0000001f + (float)ADD;
        data[gid] = v;
    }
}

static const uint32_t kN    = 16384;
static const int      kReps = 7;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <spin_short> <spin_long> [n] [K]\n", argv[0]);
        return 1;
    }
    uint32_t spin_short = (uint32_t)atoi(argv[1]);
    uint32_t spin_long  = (uint32_t)atoi(argv[2]);
    uint32_t n          = argc >= 4 ? (uint32_t)atoi(argv[3]) : kN;
    int      K          = argc >= 5 ? atoi(argv[4]) : 800;

    const char* pm4env = getenv("HIP_PM4_GRAPH");
    const bool  pm4     = (pm4env != nullptr && atoi(pm4env) != 0);

    float* data = nullptr;
    HIPCHECK(hipMalloc(&data, (size_t)std::max(n, 64u) * sizeof(float)));
    HIPCHECK(hipMemset(data, 0, (size_t)std::max(n, 64u) * sizeof(float)));

    uint32_t blocks = (n + 255) / 256, threads = 256;
    void*    funcs[2] = {(void*)gap_kernel<1>, (void*)gap_kernel<3>};

    // Per-node spin lives in a stable vector so each node can point at its own arg.
    std::vector<uint32_t> spins(K);
    for (int k = 0; k < K; k++) spins[k] = (k & 1) ? spin_long : spin_short;

    hipGraph_t graph;
    HIPCHECK(hipGraphCreate(&graph, 0));
    std::vector<hipGraphNode_t> nodes(K);
    std::vector<void*>          argbufs(K * 3);
    for (int k = 0; k < K; k++) {
        void** args = &argbufs[k * 3];
        args[0]     = &data;
        args[1]     = &spins[k];
        args[2]     = &n;
        hipKernelNodeParams p = {};
        p.func                = funcs[k & 1];
        p.gridDim             = dim3(blocks, 1, 1);
        p.blockDim            = dim3(threads, 1, 1);
        p.sharedMemBytes      = 0;
        p.kernelParams        = args;
        p.extra               = nullptr;
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
    hipEvent_t e0, e1;
    HIPCHECK(hipEventCreate(&e0));
    HIPCHECK(hipEventCreate(&e1));

    HIPCHECK(hipGraphLaunch(exec, stream));
    HIPCHECK(hipStreamSynchronize(stream));

    double best = 1e30;
    for (int r = 0; r < kReps; r++) {
        HIPCHECK(hipEventRecord(e0, stream));
        HIPCHECK(hipGraphLaunch(exec, stream));
        HIPCHECK(hipEventRecord(e1, stream));
        HIPCHECK(hipStreamSynchronize(stream));
        float ms = 0;
        HIPCHECK(hipEventElapsedTime(&ms, e0, e1));
        best = std::min(best, (double)ms * 1000.0);
    }

    printf("MIX spin_short=%u spin_long=%u n=%u K=%d e2e_us=%.3f period_us=%.4f backend=%s\n",
           spin_short, spin_long, n, K, best, best / K, pm4 ? "PM4" : "AQL");

    HIPCHECK(hipEventDestroy(e0));
    HIPCHECK(hipEventDestroy(e1));
    HIPCHECK(hipStreamDestroy(stream));
    HIPCHECK(hipGraphExecDestroy(exec));
    HIPCHECK(hipGraphDestroy(graph));
    HIPCHECK(hipFree(data));
    return 0;
}

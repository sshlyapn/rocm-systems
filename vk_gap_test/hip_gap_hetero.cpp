// Heterogeneous interleave: a linear-chain HIP graph that cycles through FOUR
// DISTINCT kernels, each with a different block size AND a different kernel-arg
// signature (varying pointer count, including UNUSED pointers, plus extra scalar
// args). This is closer to a real workload where successive dispatches have
// different launch dims and kernarg layouts/sizes, so it stresses the AQL/PM4
// dispatch path's per-packet kernarg handling rather than one repeated kernel.
//
// All four do the same dependent-FMA spin loop on the SAME buffer `a` (chained
// by graph deps), with gid < n guards so the active thread count is identical
// across kernels -> similar execution time despite different block sizes.
//
// Usage: hip_gap_hetero.x <spin> [n] [K]
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

__device__ __forceinline__ void spin_fma(float* a, uint32_t gid, uint32_t spin, float add) {
    float v = a[gid];
    for (uint32_t i = 0; i < spin; ++i) v = v * 1.0000001f + add;
    a[gid] = v;
}

// k1: 1 pointer, 3 args total. block 256.
__global__ void k1(float* a, uint32_t spin, uint32_t n) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) spin_fma(a, gid, spin, 1.0f);
}

// k2: 3 pointers (b, c unused), 5 args. block 128.
__global__ void k2(float* a, float* b, float* c, uint32_t spin, uint32_t n) {
    (void)b;
    (void)c;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) spin_fma(a, gid, spin, 2.0f);
}

// k3: 5 pointers (b..e unused) + 2 unused int pads, 9 args. block 64.
__global__ void k3(float* a, float* b, float* c, float* d, float* e, uint32_t spin, uint32_t n,
                   int pad0, int pad1) {
    (void)b;
    (void)c;
    (void)d;
    (void)e;
    (void)pad0;
    (void)pad1;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) spin_fma(a, gid, spin, 3.0f);
}

// k4: 2 pointers (b unused) + 2 unused float scalars, 6 args. block 512.
__global__ void k4(float* a, float* b, uint32_t spin, uint32_t n, float s0, float s1) {
    (void)b;
    (void)s0;
    (void)s1;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) spin_fma(a, gid, spin, 4.0f);
}

static const uint32_t kN    = 16384;
static const int      kReps = 7;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <spin> [n] [K]\n", argv[0]);
        return 1;
    }
    uint32_t spin = (uint32_t)atoi(argv[1]);
    uint32_t n    = argc >= 3 ? (uint32_t)atoi(argv[2]) : kN;
    int      K    = argc >= 4 ? atoi(argv[3]) : 800;

    const char* pm4env = getenv("HIP_PM4_GRAPH");
    const bool  pm4     = (pm4env != nullptr && atoi(pm4env) != 0);

    // Five buffers; only `a` is used, b..e are the "unused pointer" args.
    float *a, *b, *c, *d, *e;
    size_t bytes = (size_t)std::max(n, 64u) * sizeof(float);
    HIPCHECK(hipMalloc(&a, bytes));
    HIPCHECK(hipMalloc(&b, bytes));
    HIPCHECK(hipMalloc(&c, bytes));
    HIPCHECK(hipMalloc(&d, bytes));
    HIPCHECK(hipMalloc(&e, bytes));
    HIPCHECK(hipMemset(a, 0, bytes));

    // Stable scalar arg storage (shared across all nodes; values identical).
    int   pad0 = 0, pad1 = 0;
    float s0 = 1.0f, s1 = 2.0f;

    // Per-kernel: function pointer, block size, and a stable kernarg array.
    const uint32_t blocks_for[4] = {256, 128, 64, 512};
    void*          func_for[4]   = {(void*)k1, (void*)k2, (void*)k3, (void*)k4};
    // Build the four arg arrays once (kept alive until instantiate).
    void* args1[] = {&a, &spin, &n};
    void* args2[] = {&a, &b, &c, &spin, &n};
    void* args3[] = {&a, &b, &c, &d, &e, &spin, &n, &pad0, &pad1};
    void* args4[] = {&a, &b, &spin, &n, &s0, &s1};
    void** args_for[4] = {args1, args2, args3, args4};

    hipGraph_t graph;
    HIPCHECK(hipGraphCreate(&graph, 0));
    std::vector<hipGraphNode_t> nodes(K);
    for (int k = 0; k < K; k++) {
        int      t      = k & 3;  // cycle k1,k2,k3,k4
        uint32_t block  = blocks_for[t];
        uint32_t blocks = (n + block - 1) / block;
        hipKernelNodeParams p = {};
        p.func                = func_for[t];
        p.gridDim             = dim3(blocks, 1, 1);
        p.blockDim            = dim3(block, 1, 1);
        p.sharedMemBytes      = 0;
        p.kernelParams        = args_for[t];
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

    printf("HETERO spin=%u n=%u K=%d e2e_us=%.3f period_us=%.4f backend=%s\n", spin, n, K, best,
           best / K, pm4 ? "PM4" : "AQL");

    HIPCHECK(hipEventDestroy(e0));
    HIPCHECK(hipEventDestroy(e1));
    HIPCHECK(hipStreamDestroy(stream));
    HIPCHECK(hipGraphExecDestroy(exec));
    HIPCHECK(hipGraphDestroy(graph));
    HIPCHECK(hipFree(a));
    HIPCHECK(hipFree(b));
    HIPCHECK(hipFree(c));
    HIPCHECK(hipFree(d));
    HIPCHECK(hipFree(e));
    return 0;
}

// Standalone AQL-vs-PM4 HIP-graph dispatch-gap benchmark. NO profiler, NO in-kernel
// timestamps -- the only metric is the whole-graph end-to-end wall time measured with
// one hipEvent pair around hipGraphLaunch (GPU timeline).
//
// A hipGraph of K kernel nodes in a linear dependency chain is the apples-to-apples
// analog of a recorded Vulkan command buffer: instantiated once, then launched with
// no per-dispatch CPU cost. The dispatch backend (AQL default vs PM4 IB replay) is
// chosen OUTSIDE this binary via the HIP_PM4_GRAPH env var -- the test does not touch
// it, it only runs the baked-in size sweep and reports e2e per case. Run it twice
// (HIP_PM4_GRAPH unset vs =1) and diff: busy cancels (identical kernels), so the e2e
// delta is purely the per-dispatch overhead PM4 removes. With spin=0 the kernels are
// tiny (<< gap), so e2e/K is itself ~ the per-dispatch cost and the run is gap-bound.
//
// Build: hipcc -O2 --offload-arch=gfx1201 hip_gap_graph.cpp -o hip_gap_graph.x
// Run:   ./hip_gap_graph.x                 (AQL, default)
//        HIP_PM4_GRAPH=1 ./hip_gap_graph.x  (PM4 replay)
// Pin clocks first for stable numbers (host: sudo gpu_pin_freq.sh pin 0 high).

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIPCHECK(x)                                                                                \
    do {                                                                                           \
        hipError_t _e = (x);                                                                       \
        if (_e != hipSuccess) {                                                                    \
            fprintf(stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(_e), __FILE__,          \
                    __LINE__);                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// Pure compute kernel: each thread spins to set the per-kernel duration. No timing
// instrumentation -- the host hipEvent measures the whole graph. Templated on the
// in-loop add scalar so two distinct instantiations (gap_kernel<1>, gap_kernel<3>)
// are emitted; the chain interleaves them so the runtime cannot collapse the graph
// into a single repeated-kernel fast path. Both have identical cost.
template <int ADD>
__global__ void gap_kernel(float* data, uint32_t spin, uint32_t n) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = data[gid];
        for (uint32_t i = 0; i < spin; ++i) v = v * 1.0000001f + (float)ADD;
        data[gid] = v;
    }
}

// One sweep point: kernel duration knob (spin), grid size (n), and dispatch count (K).
// K is per-case so long kernels keep the run short while short kernels get enough
// dispatches for a stable median.
struct Case {
    uint32_t spin;
    uint32_t n;
    int      K;
};
// n is ALIGNED to one value across every case so the only knob is spin (duration).
static const uint32_t kN = 16384;
static const Case kCases[] = {
    {0, kN, 2000},       // gap-dominated, tiny kernels
    {250, kN, 1500},     // granular launch-bound region
    {500, kN, 1500},
    {1000, kN, 1000},
    {1500, kN, 800},
    {2000, kN, 600},
    {3000, kN, 500},
    {5000, kN, 400},
    {7500, kN, 300},
    {10000, kN, 300},
    {15000, kN, 200},
    {20000, kN, 150},
    {50000, kN, 80},
    {100000, kN, 60},
};
static const int kNumCases = sizeof(kCases) / sizeof(kCases[0]);
static const int kReps = 7;  // best-of (min e2e) per point

// Build a linear-chain graph of K gap_kernel nodes and return the min e2e (us) over
// kReps launches, measured with a single hipEvent pair around hipGraphLaunch.
static double measure_e2e_us(const Case& c) {
    float* data = nullptr;
    HIPCHECK(hipMalloc(&data, (size_t)std::max(c.n, 64u) * sizeof(float)));
    HIPCHECK(hipMemset(data, 0, (size_t)std::max(c.n, 64u) * sizeof(float)));

    uint32_t blocks = (c.n + 255) / 256, threads = 256;
    uint32_t spin = c.spin, n = c.n;

    // Two distinct kernel functions, interleaved per node, to defeat any single-kernel
    // burn/collapse optimization in the dispatch path.
    void* funcs[2] = {(void*)gap_kernel<1>, (void*)gap_kernel<3>};

    hipGraph_t graph;
    HIPCHECK(hipGraphCreate(&graph, 0));
    std::vector<hipGraphNode_t> nodes(c.K);
    for (int k = 0; k < c.K; k++) {
        void*               args[] = {&data, &spin, &n};
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
    hipEvent_t e0, e1;
    HIPCHECK(hipEventCreate(&e0));
    HIPCHECK(hipEventCreate(&e1));

    HIPCHECK(hipGraphLaunch(exec, stream));  // warmup
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

    HIPCHECK(hipEventDestroy(e0));
    HIPCHECK(hipEventDestroy(e1));
    HIPCHECK(hipStreamDestroy(stream));
    HIPCHECK(hipGraphExecDestroy(exec));
    HIPCHECK(hipGraphDestroy(graph));
    HIPCHECK(hipFree(data));
    return best;
}

int main(int argc, char** argv) {
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));

    const char* pm4env = getenv("HIP_PM4_GRAPH");
    const bool  pm4 = (pm4env != nullptr && atoi(pm4env) != 0);

    // Single-case mode: "hip_gap_graph.x <spin> [n] [K]" runs ONE case and prints a
    // parseable line. Used by the gap experiment so each scenario is measured the same
    // way for rocprof-busy and profiler-free period (AQL and PM4). n defaults to kN so
    // it stays aligned with the baked sweep.
    if (argc >= 2) {
        Case c;
        c.spin = (uint32_t)atoi(argv[1]);
        c.n    = argc >= 3 ? (uint32_t)atoi(argv[2]) : kN;
        c.K    = argc >= 4 ? atoi(argv[3]) : 500;
        double e2e = measure_e2e_us(c);
        printf("SINGLE spin=%u n=%u K=%d e2e_us=%.3f period_us=%.4f backend=%s\n", c.spin, c.n,
               c.K, e2e, e2e / c.K, pm4 ? "PM4" : "AQL");
        return 0;
    }

    printf("HIP-graph dispatch e2e sweep (hipEvent only, best-of-%d)\n", kReps);
    printf("device : %s (%s)\n", prop.name, prop.gcnArchName);
    printf("backend: %s (HIP_PM4_GRAPH=%s)\n\n", pm4 ? "PM4" : "AQL",
           pm4env ? pm4env : "<unset>");

    printf("%-7s %-7s %-5s | %-12s | %-10s\n", "spin", "n", "K", "e2e_us", "e2e/K_us");
    printf("--------------------------+--------------+-----------\n");
    for (int i = 0; i < kNumCases; i++) {
        const Case& c   = kCases[i];
        double      e2e = measure_e2e_us(c);
        printf("%-7u %-7u %-5d | %-12.2f | %-10.4f\n", c.spin, c.n, c.K, e2e, e2e / c.K);
        fflush(stdout);
    }
    return 0;
}

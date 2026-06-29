/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

/**
 * This benchmark measures the per-dispatch overhead (inter-kernel gap) of launching
 * a serially dependent hipGraph kernel chain -- the apples-to-apples analog of a
 * recorded Vulkan command buffer (instantiated once, then replayed with no per-launch
 * CPU cost). The dispatch backend is selected ENTIRELY OUTSIDE this test via the
 * HIP_PM4_GRAPH env var (default AQL dispatch; HIP_PM4_GRAPH=1 enables PM4 IB replay);
 * the test only reads it to label output. Run the test twice (with and without
 * HIP_PM4_GRAPH=1) and diff the reported e2e: the per-kernel compute is identical, so
 * the delta is purely the per-dispatch overhead one backend removes.
 *
 * The only metric is the whole-graph end-to-end wall time from one hipEvent pair
 * around hipGraphLaunch (GPU timeline, no profiler, no in-kernel timestamps). With
 * spin=0 the kernels are tiny (<< gap), so e2e/K is itself ~ the per-dispatch cost.
 */

static constexpr int kReps = 5;  // best-of (min e2e) launches per sweep point

/**
 * Pure compute kernel; each thread spins to set the per-kernel duration. Templated on
 * the in-loop add scalar so two distinct instantiations (<1>, <3>) are emitted and the
 * chain can interleave them, preventing any single-repeated-kernel fast path. Both have
 * identical cost.
 */
template <int ADD>
static __global__ void gapKernel(float* data, uint32_t spin, uint32_t n) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    float v = data[gid];
    for (uint32_t i = 0; i < spin; ++i) v = v * 1.0000001f + static_cast<float>(ADD);
    data[gid] = v;
  }
}

/**
 * One sweep point: spin (kernel duration knob), n (grid size), K (dispatch count).
 * K is per-case so long kernels keep the run short while short kernels still get
 * enough dispatches for a stable median.
 */
struct GapCase {
  uint32_t spin;
  uint32_t n;
  int K;
};

/**
 * Build a linear-chain graph of K interleaved gapKernel nodes and return the minimum
 * end-to-end time (us) over kReps launches, measured with one hipEvent pair. Each node
 * depends on the previous one and they all read/write the same buffer, so this is a
 * strictly serial chain -- the analog of calling K kernels one-by-one on a single
 * stream with the same memory.
 */
static double MeasureChainE2eUs(const GapCase& c) {
  float* data = nullptr;
  HIP_CHECK(hipMalloc(&data, static_cast<size_t>(std::max(c.n, 64u)) * sizeof(float)));
  HIP_CHECK(hipMemset(data, 0, static_cast<size_t>(std::max(c.n, 64u)) * sizeof(float)));

  uint32_t blocks = (c.n + 255) / 256, threads = 256;
  uint32_t spin = c.spin, n = c.n;

  // Two distinct kernel functions interleaved per node to defeat any single-kernel
  // collapse optimization in the dispatch path. Set HIP_GAP_NOINTERLEAVE=1 to use a
  // single repeated kernel instead (experiment: isolate the cost of switching code
  // objects every dispatch).
  const char* nointl = std::getenv("HIP_GAP_NOINTERLEAVE");
  const bool noInterleave = (nointl != nullptr) && (std::atoi(nointl) != 0);
  void* funcs[2] = {reinterpret_cast<void*>(gapKernel<1>),
                    reinterpret_cast<void*>(noInterleave ? gapKernel<1> : gapKernel<3>)};

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  std::vector<hipGraphNode_t> nodes(c.K);
  for (int k = 0; k < c.K; k++) {
    void* args[] = {&data, &spin, &n};
    hipKernelNodeParams p = {};
    p.func = funcs[k & 1];
    p.gridDim = dim3(blocks, 1, 1);
    p.blockDim = dim3(threads, 1, 1);
    p.sharedMemBytes = 0;
    p.kernelParams = args;
    p.extra = nullptr;
    hipGraphNode_t deps[1];
    size_t ndeps = 0;
    if (k > 0) {
      deps[0] = nodes[k - 1];
      ndeps = 1;
    }
    HIP_CHECK(hipGraphAddKernelNode(&nodes[k], graph, ndeps ? deps : nullptr, ndeps, &p));
  }

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));

  // Warm up call
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  double best = 1e30;
  for (int r = 0; r < kReps; r++) {
    HIP_CHECK(hipEventRecord(e0, stream));
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipEventRecord(e1, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    best = std::min(best, static_cast<double>(ms) * 1000.0);
  }

  HIP_CHECK(hipEventDestroy(e0));
  HIP_CHECK(hipEventDestroy(e1));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(data));
  return best;
}

/**
 * @addtogroup hipGraphLaunch hipGraphLaunch
 * @{
 * @ingroup PerformanceTestGraph
 * `hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);`
 * - Launches an executable graph in the specified stream.
 */

/**
 * Test Description
 * ------------------------
 * - This test case, tests the following scenario :
 * - 1) Build a serially dependent chain of K kernel nodes (linear RAW chain),
 * -    interleaving two distinct kernel functions so no single-kernel fast path
 * -    applies.
 * - 2) Instantiate once and launch repeatedly, timing the whole graph with a single
 * -    hipEvent pair (best of kReps).
 * - 3) Sweep the kernel duration (spin) and grid size, reporting end-to-end time and
 * -    per-dispatch time (e2e/K).
 * - 4) The dispatch backend (AQL vs PM4 IB replay) is chosen via HIP_PM4_GRAPH; run
 * -    twice and diff e2e to isolate the per-dispatch overhead.
 *
 * Test source
 * ------------------------
 * - performance/scenarios/graph/hipPerfGraphLinearChainGap.cc
 *
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Performance_GraphLinearChainDispatchGap) {
  static const GapCase kCases[] = {
      {0, 4096, 2000},      // gap-dominated, tiny kernels
      {0, 16384, 2000},     // gap-dominated, larger grid
      {1000, 16384, 1000},  // ~10 us kernels
      {2000, 16384, 600},   // ~30 us kernels
      {5000, 16384, 400},   // ~70 us kernels
      {10000, 16384, 300},  // ~100 us kernels
      {20000, 16384, 150},  // ~280 us kernels
      {50000, 16384, 80},   // ~700 us kernels
      {100000, 16384, 60},  // ~1 ms kernels
  };

  const char* pm4env = std::getenv("HIP_PM4_GRAPH");
  const bool pm4 = (pm4env != nullptr && std::atoi(pm4env) != 0);

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::cout << "HIP-graph linear-chain dispatch-gap sweep (hipEvent, best-of-" << kReps << ")"
            << std::endl;
  std::cout << "device : " << prop.name << " (" << prop.gcnArchName << ")" << std::endl;
  std::cout << "backend: " << (pm4 ? "PM4" : "AQL")
            << " (HIP_PM4_GRAPH=" << (pm4env ? pm4env : "<unset>") << ")" << std::endl;
  std::cout << "spin     n        K      e2e_us         e2e/K_us" << std::endl;

  for (const auto& c : kCases) {
    double e2e = MeasureChainE2eUs(c);
    REQUIRE(e2e > 0.0);
    std::cout << std::left << std::setw(9) << c.spin << std::setw(9) << c.n << std::setw(7) << c.K
              << std::setw(15) << e2e << (e2e / c.K) << std::endl;
  }
}

/**
 * End doxygen group GraphTest.
 * @}
 */

// Strict RAW-coherence stress test for the PM4 graph replay path.
//
// Builds a HIP graph of K kernel nodes, each an IN-PLACE increment of the SAME
// device buffer: d[idx] += 1. Consecutive nodes therefore form a strict
// read-after-write chain on every element -- node j must observe ALL of node
// j-1's writes. If the per-edge fence is too weak (consumer launches before the
// producer's writes are coherent), some elements miss increments and the final
// value drifts below K, often NON-deterministically across replays.
//
// After R replays of the K-node graph the expected value of every element is
// exactly R*K. We compare against that and report any mismatch. This is the
// same failure mode that exposed the unsafe PWS fence, so it is the right gate
// for validating the AGENT-scope interior-edge reduction.
//
// Usage: ./pm4_coherence_test [K] [N] [R]
//   K = nodes per graph (default 256), N = elements (default 65536),
//   R = graph replays (default 200).
// Plain ASCII only.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x)                                                            \
  do {                                                                          \
    hipError_t e = (x);                                                         \
    if (e != hipSuccess) {                                                      \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),          \
              __FILE__, __LINE__);                                              \
      exit(2);                                                                  \
    }                                                                           \
  } while (0)

__global__ void incKernel(int* d, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Plain RAW: read, add, write back the same location. Coherence with the
    // previous node's write to d[idx] is mandatory for a correct result.
    d[idx] = d[idx] + 1;
  }
}

int main(int argc, char** argv) {
  int K = argc > 1 ? atoi(argv[1]) : 256;
  int N = argc > 2 ? atoi(argv[2]) : 65536;
  int R = argc > 3 ? atoi(argv[3]) : 200;

  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, N * sizeof(int)));
  HIP_CHECK(hipMemset(d, 0, N * sizeof(int)));

  const int block = 256;
  const int grid = (N + block - 1) / block;

  // Build the K-node chain as a HIP graph via stream capture so every node is a
  // real dependent edge in the captured dispatch chain (what PM4 replays).
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipGraph_t graph;
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  for (int j = 0; j < K; ++j) {
    incKernel<<<grid, block, 0, stream>>>(d, N);
  }
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  std::vector<int> h(N);
  long long badReplays = 0, totalMismatch = 0;
  int firstBadVal = -1, firstBadExp = -1;

  for (int r = 0; r < R; ++r) {
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(h.data(), d, N * sizeof(int), hipMemcpyDeviceToHost));
    int expected = (r + 1) * K;
    long long mism = 0;
    for (int i = 0; i < N; ++i) {
      if (h[i] != expected) {
        ++mism;
        if (firstBadVal < 0) { firstBadVal = h[i]; firstBadExp = expected; }
      }
    }
    if (mism) { ++badReplays; totalMismatch += mism; }
  }

  printf("K=%d N=%d R=%d : bad_replays=%lld total_mismatch=%lld",
         K, N, R, badReplays, totalMismatch);
  if (badReplays) printf(" first_bad_val=%d expected=%d", firstBadVal, firstBadExp);
  printf("  -> %s\n", badReplays == 0 ? "PASS" : "FAIL");

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d));
  return badReplays == 0 ? 0 : 1;
}

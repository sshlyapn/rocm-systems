// hip_keycache_mutate.cpp
//
// Correctness gate for the DEFAULT-ON PM4 IB key cache (recorded packet set
// version). The cache reuses the last compiled PM4 IB without rehashing as long
// as the version is unchanged, and the version is bumped on EVERY recorded-packet
// mutation. This test mutates an already-instantiated, already-replayed graph in
// the two ways that change the dispatched packet set and proves the next replay
// reflects the change (i.e. the stale cached IB is NOT reused):
//
//   1. hipGraphExecKernelNodeSetParams  -- in-place kernel param update
//   2. hipGraphNodeSetEnabled           -- disable / re-enable a node
//
// Run the SAME sequence against the AQL baseline (HIP_PM4_GRAPH unset) and the
// PM4 path (HIP_PM4_GRAPH=1, key cache default-on); every checkpoint must match.
// A stale-IB bug would show up as a PM4 value that lags the requested mutation.
// Plain ASCII only.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BS 64
#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP err %s @%d: %s\n",#x,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)

// x[i] = v  (a pure value-set so the result == the most recent param)
__global__ void setk(float* x, float v, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) x[i] = v;
}

static float launch_and_read(hipGraphExec_t exec, hipStream_t s, float* dx, int M){
    HC(hipGraphLaunch(exec, s));
    HC(hipStreamSynchronize(s));
    float h = -123.0f;
    HC(hipMemcpy(&h, dx, sizeof(float), hipMemcpyDeviceToHost));
    return h;
}

int main(int argc, char** argv){
    int M = argc>1 ? atoi(argv[1]) : 256;
    size_t bytes = (size_t)M*4;
    float* dx; HC(hipMalloc(&dx, bytes));
    dim3 grid((M+BS-1)/BS), block(BS);
    hipStream_t s; HC(hipStreamCreate(&s));

    // Build a 1-node graph explicitly so we hold the node handle for mutation.
    hipGraph_t graph; HC(hipGraphCreate(&graph, 0));
    float v = 1.0f;
    void* kargs[] = { (void*)&dx, (void*)&v, (void*)&M };
    hipKernelNodeParams np{};
    np.func = (void*)setk;
    np.gridDim = grid; np.blockDim = block;
    np.sharedMemBytes = 0; np.kernelParams = kargs; np.extra = nullptr;
    hipGraphNode_t node;
    HC(hipGraphAddKernelNode(&node, graph, nullptr, 0, &np));

    hipGraphExec_t exec; HC(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    // Warm the cache with the original params (v=1.0): arms the key-cache fast path.
    for(int i=0;i<5;++i) (void)launch_and_read(exec, s, dx, M);
    float c0 = launch_and_read(exec, s, dx, M);

    // (1) In-place param update v=1.0 -> v=7.0. Must invalidate the cached IB.
    v = 7.0f;
    HC(hipGraphExecKernelNodeSetParams(exec, node, &np));
    float c1 = launch_and_read(exec, s, dx, M);

    // Replay again (no mutation): cache may legitimately reuse the v=7.0 IB.
    float c2 = launch_and_read(exec, s, dx, M);

    // (2) Another param update v=7.0 -> v=42.0.
    v = 42.0f;
    HC(hipGraphExecKernelNodeSetParams(exec, node, &np));
    float c3 = launch_and_read(exec, s, dx, M);

    // (3) Disable the only node -> dispatched set becomes empty; dx keeps its
    //     last value (42). We prove disable took effect by first zeroing dx.
    HC(hipMemset(dx, 0, bytes));            // dx = 0
    HC(hipGraphNodeSetEnabled(exec, node, 0));
    float c4 = launch_and_read(exec, s, dx, M);   // node disabled -> stays 0

    // (4) Re-enable -> v=42 write happens again.
    HC(hipGraphNodeSetEnabled(exec, node, 1));
    float c5 = launch_and_read(exec, s, dx, M);    // 42 again

    const char* mode = getenv("HIP_PM4_GRAPH") ? "PM4" : "AQL";
    printf("%s  c0=%.1f c1=%.1f c2=%.1f c3=%.1f c4=%.1f c5=%.1f\n",
           mode, c0, c1, c2, c3, c4, c5);
    // Expected: 1.0 7.0 7.0 42.0 0.0 42.0
    bool ok = (c0==1.0f && c1==7.0f && c2==7.0f && c3==42.0f && c4==0.0f && c5==42.0f);
    printf("%s  %s\n", mode, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

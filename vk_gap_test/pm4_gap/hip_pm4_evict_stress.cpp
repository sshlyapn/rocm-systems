// hip_pm4_evict_stress.cpp
//
// Stress gate for the PM4 IB cache VRAM bound. A graph mutation that changes a
// packet field covered by the content hash (grid/block dims, kernel object,
// kernarg address) produces a NEW compiled IB while the pre-mutation IB becomes
// unreferenced. This test mutates an instantiated graph K times, each time to a
// STRICTLY NEW grid size, so each launch builds a distinct IB. Without the bound
// the executable memory pool would grow by one IB per iteration; with the bound
// (kPm4MaxCachedIbs) the dead IBs are evicted and the pool stays flat.
//
// We do NOT measure VRAM here (noisy); instead run with AMD_LOG_LEVEL=3 and grep
// "[pm4-evict]" -- the printed "cache size now N" must never exceed the cap. This
// test's own job is to prove eviction does not corrupt replay: every launch must
// still produce the value requested by the most recent mutation. Plain ASCII only.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BS 64
#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP err %s @%d: %s\n",#x,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)

__global__ void setk(float* x, float v, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) x[i] = v;
}

int main(int argc, char** argv){
    int M = argc>1 ? atoi(argv[1]) : 256;
    int K = argc>2 ? atoi(argv[2]) : 2000;   // distinct grid sizes -> distinct IBs
    size_t bytes = (size_t)M*4;
    float* dx; HC(hipMalloc(&dx, bytes));
    dim3 block(BS);
    hipStream_t s; HC(hipStreamCreate(&s));

    hipGraph_t graph; HC(hipGraphCreate(&graph, 0));
    float v = 1.0f;
    void* kargs[] = { (void*)&dx, (void*)&v, (void*)&M };
    hipKernelNodeParams np{};
    np.func = (void*)setk;
    np.gridDim = dim3(1); np.blockDim = block;
    np.sharedMemBytes = 0; np.kernelParams = kargs; np.extra = nullptr;
    hipGraphNode_t node;
    HC(hipGraphAddKernelNode(&node, graph, nullptr, 0, &np));
    hipGraphExec_t exec; HC(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    int fails = 0;
    for(int i=1;i<=K;++i){
        // Strictly-new grid size each iteration -> a brand new compiled IB.
        np.gridDim = dim3((unsigned)(i % 4096) + 1);
        v = (float)i;
        HC(hipGraphExecKernelNodeSetParams(exec, node, &np));
        HC(hipGraphLaunch(exec, s));
        HC(hipStreamSynchronize(s));
        float h = -1.0f;
        HC(hipMemcpy(&h, dx, sizeof(float), hipMemcpyDeviceToHost));  // x[0], block 0 always runs
        if(h != (float)i){ if(fails<5) printf("  MISMATCH i=%d got=%.1f\n", i, h); ++fails; }
    }

    const char* mode = getenv("HIP_PM4_GRAPH") ? "PM4" : "AQL";
    printf("%s  K=%d distinct-IB mutations  %s\n", mode, K, fails==0 ? "PASS (all replays correct)" :
           "FAIL");
    return fails==0 ? 0 : 1;
}

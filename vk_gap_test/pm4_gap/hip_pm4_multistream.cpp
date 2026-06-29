// hip_pm4_multistream.cpp
//
// Correctness gate for the GraphExec-owned, device-scoped SHARED IB (#5). A
// queue-independent graph (no scratch, no queue_ptr) compiles to ONE device-scoped
// PM4 IB built once at instantiate; every stream that replays the graph submits
// that single copy from its own queue (no per-stream specialize/upload). This test
// instantiates one exec and replays it from several DISTINCT streams, proving the
// shared IB is correctly submittable from each stream's vdev:
//
//   - sequential: zero the buffer on stream i, replay on stream i, sync, verify.
//   - interleaved/concurrent: launch on all streams without syncing between, then
//     sync all -- the last writer wins but every launch must have run the kernel.
//
// Run against the AQL baseline (HIP_PM4_GRAPH unset) and the PM4 path
// (HIP_PM4_GRAPH=1, shared IB default-on); every checkpoint must match. Set
// HIP_PM4_GRAPH_SHARED_IB=0 to A/B against the per-stream-IB path. Plain ASCII only.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BS 64
#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP err %s @%d: %s\n",#x,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)

__global__ void setk(float* x, float v, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) x[i] = v;
}

int main(int argc, char** argv){
    int M = argc>1 ? atoi(argv[1]) : 4096;
    int NS = argc>2 ? atoi(argv[2]) : 4;       // number of streams sharing the IB
    size_t bytes = (size_t)M*4;
    float* dx; HC(hipMalloc(&dx, bytes));
    dim3 grid((M+BS-1)/BS), block(BS);

    std::vector<hipStream_t> streams(NS);
    for(int i=0;i<NS;++i) HC(hipStreamCreate(&streams[i]));

    // One graph (writes v=42 to dx). Instantiate ONCE -> one shared IB at instantiate.
    hipGraph_t graph; HC(hipGraphCreate(&graph, 0));
    float v = 42.0f;
    void* kargs[] = { (void*)&dx, (void*)&v, (void*)&M };
    hipKernelNodeParams np{};
    np.func = (void*)setk;
    np.gridDim = grid; np.blockDim = block;
    np.sharedMemBytes = 0; np.kernelParams = kargs; np.extra = nullptr;
    hipGraphNode_t node;
    HC(hipGraphAddKernelNode(&node, graph, nullptr, 0, &np));
    hipGraphExec_t exec; HC(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    const char* mode = getenv("HIP_PM4_GRAPH") ? "PM4" : "AQL";
    bool ok = true;

    // (1) Sequential per-stream: prove EACH stream's submit of the shared IB runs.
    for(int rep=0; rep<3; ++rep){
        for(int i=0;i<NS;++i){
            HC(hipMemset(dx, 0, bytes));                    // dx = 0 (device-wide barrier)
            HC(hipDeviceSynchronize());
            HC(hipGraphLaunch(exec, streams[i]));           // shared IB -> dx = 42
            HC(hipDeviceSynchronize());
            float h0=-1.f, hL=-1.f;
            HC(hipMemcpy(&h0, dx, sizeof(float), hipMemcpyDeviceToHost));
            HC(hipMemcpy(&hL, dx+(M-1), sizeof(float), hipMemcpyDeviceToHost));
            if(h0!=42.0f || hL!=42.0f){
                printf("%s  FAIL seq rep=%d stream=%d  dx[0]=%.1f dx[M-1]=%.1f\n",
                       mode, rep, i, h0, hL);
                ok = false;
            }
        }
    }

    // (2) Interleaved: launch on all streams, then sync all. Every launch ran the
    //     shared IB; dx ends at 42 regardless of ordering.
    HC(hipMemset(dx, 0, bytes));
    for(int i=0;i<NS;++i) HC(hipGraphLaunch(exec, streams[i]));
    for(int i=0;i<NS;++i) HC(hipStreamSynchronize(streams[i]));
    {
        float h=-1.f; HC(hipMemcpy(&h, dx, sizeof(float), hipMemcpyDeviceToHost));
        if(h!=42.0f){ printf("%s  FAIL interleaved dx[0]=%.1f\n", mode, h); ok=false; }
    }

    printf("%s  streams=%d M=%d  shared_ib=%s  %s\n", mode, NS, M,
           getenv("HIP_PM4_GRAPH_SHARED_IB") ? getenv("HIP_PM4_GRAPH_SHARED_IB") : "1(default)",
           ok ? "PASS" : "FAIL");

    for(int i=0;i<NS;++i) HC(hipStreamDestroy(streams[i]));
    HC(hipGraphExecDestroy(exec));
    HC(hipGraphDestroy(graph));
    HC(hipFree(dx));
    return ok ? 0 : 1;
}

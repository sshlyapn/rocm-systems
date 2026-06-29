// hip_gpu_gap.cpp
//
// GPU-SIDE companion to hip_replay_timing.cpp. Where hip_replay_timing measures
// the HOST enqueue cost of one hipGraphLaunch, this binary is meant to be run
// UNDER rocprofv3 --kernel-trace so the profiler records the GPU begin/end
// timestamp of every dispatched kernel. From those timestamps a parser computes
// the on-device inter-kernel gap (next.begin - cur.end) and per-kernel duration.
//
// It captures an N-long RAW chain of addk kernels into one hipGraph, then replays
// that graph K times BACK TO BACK with no per-iteration hipStreamSynchronize, so
// the command processor sees a continuous stream of dispatches. PM4 (one IB for
// the whole chain) vs AQL (N packets) should differ in the gap the CP leaves
// between consecutive kernels. The same binary is run against every userspace
// stack purely by swapping LD_LIBRARY_PATH (and HIP_PM4_GRAPH). Plain ASCII only.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BS 64
#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP err %s @%d: %s\n",#x,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)

__global__ void addk(float* x, const float* w, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) x[i] += w[i];
}

int main(int argc,char** argv){
    int N = argc>1 ? atoi(argv[1]) : 64;     // chain length (dispatch packets)
    int M = argc>2 ? atoi(argv[2]) : 256;    // elems (keep tiny so GPU never lags)
    int K = argc>3 ? atoi(argv[3]) : 50;     // back-to-back graph replays under the profiler

    size_t bytes = (size_t)M*4;
    float *x,*w; HC(hipMalloc(&x,bytes)); HC(hipMalloc(&w,bytes));
    std::vector<float> hx(M,0.1f), hw(M,1.0f);
    HC(hipMemcpy(x,hx.data(),bytes,hipMemcpyHostToDevice));
    HC(hipMemcpy(w,hw.data(),bytes,hipMemcpyHostToDevice));

    dim3 grid((M+BS-1)/BS), block(BS);
    hipStream_t stream; HC(hipStreamCreate(&stream));

    hipGraph_t graph; hipGraphExec_t exec;
    HC(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for(int it=0; it<N; ++it)
        hipLaunchKernelGGL(addk, grid, block, 0, stream, x, w, M);
    HC(hipStreamEndCapture(stream,&graph));
    HC(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));

    // Warm up: build/cache the PM4 IB and let the kernarg pool settle so the
    // profiled region is steady state, not first-touch.
    for(int i=0;i<5;++i){ HC(hipGraphLaunch(exec,stream)); }
    HC(hipStreamSynchronize(stream));

    // Profiled region: K replays back to back, single drain at the end. No host
    // sync between replays -> the CP processes a continuous dispatch stream.
    for(int r=0;r<K;++r){ HC(hipGraphLaunch(exec,stream)); }
    HC(hipStreamSynchronize(stream));

    fprintf(stderr,"done N=%d M=%d K=%d total_kernels=%d\n", N, M, K, N*K);
    return 0;
}

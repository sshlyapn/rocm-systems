// hip_pm4_prewarm_probe.cpp
//
// Hypothesis: the ~4 ms PM4 "first-ever replay" floor is a ONE-TIME executable
// memory-pool warm-up (the first HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG allocation
// in the process), NOT a per-graph cost. If so, the cold cost of the SECOND
// distinct graph built in the same process should be MUCH smaller than the first.
//
// Build two DISTINCT graphs, time each one's first-ever hipGraphLaunch in
// isolation. cold1 >> cold2  => pool warm-up (fixable with a one-time prewarm).
// cold1 ~ cold2              => genuine per-graph build cost. Plain ASCII only.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define BS 64
#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP err %s @%d: %s\n",#x,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static double now_us(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1e6 + ts.tv_nsec*1e-3; }

__global__ void addk(float* x, const float* w, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) x[i] += w[i];
}

// Build an N-long addk chain, instantiate, time the first launch, return cold us.
static double build_time_cold(int N, int M, hipStream_t stream){
    size_t bytes=(size_t)M*4; float *x,*w;
    HC(hipMalloc(&x,bytes)); HC(hipMalloc(&w,bytes));
    HC(hipMemset(x,0,bytes)); HC(hipMemset(w,0,bytes));
    dim3 grid((M+BS-1)/BS), block(BS);
    hipGraph_t g; hipGraphExec_t ex;
    HC(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for(int i=0;i<N;++i) hipLaunchKernelGGL(addk, grid, block, 0, stream, x, w, M);
    HC(hipStreamEndCapture(stream,&g));
    HC(hipGraphInstantiate(&ex,g,nullptr,nullptr,0));
    double t0=now_us(); HC(hipGraphLaunch(ex,stream)); double t1=now_us();
    HC(hipStreamSynchronize(stream));
    return t1-t0;
}

int main(int argc,char** argv){
    int M = argc>1 ? atoi(argv[1]) : 256;
    hipStream_t s; HC(hipStreamCreate(&s));
    double cold1 = build_time_cold(64,  M, s);   // first distinct graph in process
    double cold2 = build_time_cold(64,  M, s);   // second, SAME shape, NEW buffers/graph
    double cold3 = build_time_cold(128, M, s);   // third, different shape
    const char* mode = getenv("HIP_PM4_GRAPH") ? "PM4" : "AQL";
    printf("%s  cold1(first graph)=%.1f us  cold2(2nd graph)=%.1f us  cold3(3rd, N=128)=%.1f us\n",
           mode, cold1, cold2, cold3);
    return 0;
}

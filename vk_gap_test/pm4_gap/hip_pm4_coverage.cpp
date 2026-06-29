// hip_pm4_coverage.cpp
//
// Coverage test for the CLR HIP_PM4_GRAPH replay path (RDNA3 gfx11 / RDNA4
// gfx12). Each scenario builds a captured hipGraph that stresses one of the
// e2e coverage gaps and prints a labeled checksum of the result buffer. The
// harness (run_pm4_coverage.sh) runs the binary twice -- baseline (env unset)
// and HIP_PM4_GRAPH=1 -- and diffs the checksums line by line; every scenario
// must be bit-exact. Scratch is additionally gated by HIP_PM4_GRAPH_SCRATCH=1.
//
// Scenarios:
//   multi      : multiple distinct kernels in one graph (G2 SGPR layout)
//   nonmult    : grid not a multiple of the block size (G8 partial workgroup)
//   coherence  : H2D copy immediately before the graph (G6 leading acquire)
//   recreate   : destroy a graph then build a DIFFERENT one (G7 stale-IB key)
//   wave64     : a wave64 kernel mixed with wave32 kernels (G3 CS_W32_EN)
//   barriernode: graph with an event/barrier node (G5 non-dispatch fallback)
//   scratch    : a kernel that spills to scratch (G1, opt-in)
//
// Plain ASCII only in comments.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#define HC(call) do { hipError_t _e=(call); if(_e!=hipSuccess){ \
  fprintf(stderr,"HIP error %s at %s:%d: %s\n",#call,__FILE__,__LINE__,hipGetErrorString(_e)); \
  exit(1);} } while(0)

#define BS 256

__global__ void k_copy (float* y, const float* x, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]; }
__global__ void k_scale(float* y, const float* w, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]*=w[i]; }
__global__ void k_blend(float* x, const float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=0.5f*x[i]+0.5f*y[i]; }
__global__ void k_add  (float* y, const float* w, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]+=w[i]; }

// Uses static __shared__ LDS (block reduction broadcast). Real decode kernels
// (rmsnorm/GEMM) use LDS, so compute_pgm_rsrc2.LDS_SIZE must be preserved.
__global__ void k_lds(float* y, const float* x, int n){
    __shared__ float s[BS];
    int t=threadIdx.x; int i=blockIdx.x*blockDim.x+t;
    s[t] = (i<n)? x[i] : 0.0f;
    __syncthreads();
    for(int o=BS/2;o>0;o>>=1){ if(t<o) s[t]+=s[t+o]; __syncthreads(); }
    float mean = s[0]*(1.0f/BS);
    if(i<n) y[i] = x[i] + 0.01f*mean;
}

// Dynamically-indexed local array large enough to force a scratch allocation
// (private_segment_fixed_size > 0), exercising the G1 scratch path.
__global__ void k_scratch(float* y, const float* x, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    volatile float buf[96];
    for(int k=0;k<96;++k) buf[k]=x[(i+k*13)%n];
    float s=0.0f;
    for(int k=0;k<96;++k) s+=buf[(k*37+i)%96];
    if(i<n) y[i]=s*(1.0f/96.0f);
}

// Defined in wave64_kernel.hip (compiled with -mwavefrontsize64).
void launch_w64_madd(float* y, const float* w, int n, hipStream_t s);

struct Buffers { float *x,*y,*w; int M; };

static double checksum(const Buffers& b){
    std::vector<float> out(b.M);
    HC(hipMemcpy(out.data(),b.x,b.M*4,hipMemcpyDeviceToHost));
    double s=0; for(int i=0;i<b.M;++i) s+=out[i]; return s;
}
static void reset(const Buffers& b, const std::vector<float>& xs,
                  const std::vector<float>& ys, const std::vector<float>& ws){
    HC(hipMemcpy(b.x,xs.data(),b.M*4,hipMemcpyHostToDevice));
    HC(hipMemcpy(b.y,ys.data(),b.M*4,hipMemcpyHostToDevice));
    HC(hipMemcpy(b.w,ws.data(),b.M*4,hipMemcpyHostToDevice));
}

// Capture a body lambda into a graph, instantiate, warm up, run once on fresh
// inputs, return the checksum. Destroys the graph before returning.
template <typename Body>
static double run_graph(const Buffers& b, hipStream_t s, const std::vector<float>& xs,
                        const std::vector<float>& ys, const std::vector<float>& ws, Body body){
    hipGraph_t graph; hipGraphExec_t exec;
    HC(hipStreamBeginCapture(s,hipStreamCaptureModeGlobal));
    body(s);
    HC(hipStreamEndCapture(s,&graph));
    HC(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int w=0;w<10;++w){ HC(hipGraphLaunch(exec,s)); }
    HC(hipStreamSynchronize(s));
    reset(b,xs,ys,ws);                       // H2D right before launch (G6)
    HC(hipGraphLaunch(exec,s)); HC(hipStreamSynchronize(s));
    double ck=checksum(b);
    HC(hipGraphExecDestroy(exec)); HC(hipGraphDestroy(graph));
    return ck;
}

int main(int argc,char** argv){
    int M = argc>1?atoi(argv[1]):16384;
    int dev=0; HC(hipSetDevice(dev));
    hipDeviceProp_t prop; HC(hipGetDeviceProperties(&prop,dev));
    printf("# GPU=%s gcn=%s M=%d HIP_PM4_GRAPH=%s HIP_PM4_GRAPH_SCRATCH=%s\n",
           prop.name, prop.gcnArchName, M,
           getenv("HIP_PM4_GRAPH")?getenv("HIP_PM4_GRAPH"):"(unset)",
           getenv("HIP_PM4_GRAPH_SCRATCH")?getenv("HIP_PM4_GRAPH_SCRATCH"):"(unset)");

    Buffers b; b.M=M;
    HC(hipMalloc(&b.x,M*4)); HC(hipMalloc(&b.y,M*4)); HC(hipMalloc(&b.w,M*4));
    std::vector<float> xs(M),ys(M),ws(M);
    for(int i=0;i<M;++i){ xs[i]=(i%7)*0.01f+0.1f; ws[i]=1.0f+((i%5)*0.001f); ys[i]=(i%9)*0.01f+0.05f; }
    int blocks=(M+BS-1)/BS;
    hipStream_t s; HC(hipStreamCreate(&s));

    // multi: distinct kernels, all kernarg-only ABI.
    {
        double ck=run_graph(b,s,xs,ys,ws,[&](hipStream_t st){
            for(int l=0;l<6;++l){
                hipLaunchKernelGGL(k_copy ,dim3(blocks),dim3(BS),0,st,b.y,b.x,b.M);
                hipLaunchKernelGGL(k_scale,dim3(blocks),dim3(BS),0,st,b.y,b.w,b.M);
                hipLaunchKernelGGL(k_blend,dim3(blocks),dim3(BS),0,st,b.x,b.y,b.M);
                hipLaunchKernelGGL(k_add  ,dim3(blocks),dim3(BS),0,st,b.y,b.w,b.M);
            }
        });
        printf("multi       checksum=%.6f\n",ck);
    }

    // nonmult: grid not a multiple of the block (M2 % BS != 0) -> partial last WG.
    {
        int M2=M-37; if(M2<1) M2=M;            // not a multiple of 256
        int bl2=(M2+BS-1)/BS;
        double ck=run_graph(b,s,xs,ys,ws,[&](hipStream_t st){
            for(int l=0;l<4;++l){
                hipLaunchKernelGGL(k_copy ,dim3(bl2),dim3(BS),0,st,b.y,b.x,M2);
                hipLaunchKernelGGL(k_blend,dim3(bl2),dim3(BS),0,st,b.x,b.y,M2);
            }
        });
        printf("nonmult     checksum=%.6f\n",ck);
    }

    // coherence: handled by run_graph's reset() (H2D) right before the launch.
    {
        double ck=run_graph(b,s,xs,ys,ws,[&](hipStream_t st){
            hipLaunchKernelGGL(k_copy ,dim3(blocks),dim3(BS),0,st,b.y,b.x,b.M);
            hipLaunchKernelGGL(k_blend,dim3(blocks),dim3(BS),0,st,b.x,b.y,b.M);
        });
        printf("coherence   checksum=%.6f\n",ck);
    }

    // recreate: build + destroy a throwaway graph, then build a DIFFERENT graph.
    // A stale content-keyed IB from the first must not be replayed for the second.
    {
        run_graph(b,s,xs,ys,ws,[&](hipStream_t st){
            hipLaunchKernelGGL(k_scale,dim3(blocks),dim3(BS),0,st,b.y,b.w,b.M);
        });
        double ck=run_graph(b,s,xs,ys,ws,[&](hipStream_t st){
            hipLaunchKernelGGL(k_copy ,dim3(blocks),dim3(BS),0,st,b.y,b.x,b.M);
            hipLaunchKernelGGL(k_add  ,dim3(blocks),dim3(BS),0,st,b.x,b.w,b.M);
        });
        printf("recreate    checksum=%.6f\n",ck);
    }

    // lds: kernels that use static __shared__ memory (rsrc2.LDS_SIZE preserved).
    {
        double ck=run_graph(b,s,xs,ys,ws,[&](hipStream_t st){
            for(int l=0;l<4;++l){
                hipLaunchKernelGGL(k_lds  ,dim3(blocks),dim3(BS),0,st,b.y,b.x,b.M);
                hipLaunchKernelGGL(k_lds  ,dim3(blocks),dim3(BS),0,st,b.x,b.y,b.M);
            }
        });
        printf("lds         checksum=%.6f\n",ck);
    }

    // wave64: mix a wave64 kernel between wave32 kernels (CS_W32_EN per kernel).
    {
        double ck=run_graph(b,s,xs,ys,ws,[&](hipStream_t st){
            hipLaunchKernelGGL(k_copy ,dim3(blocks),dim3(BS),0,st,b.y,b.x,b.M);
            launch_w64_madd(b.y,b.w,b.M,st);
            hipLaunchKernelGGL(k_blend,dim3(blocks),dim3(BS),0,st,b.x,b.y,b.M);
        });
        printf("wave64      checksum=%.6f\n",ck);
    }

    // barriernode: a graph with an explicit event record/wait -> non-dispatch
    // packet, must fall back to AQL and stay correct.
    {
        hipEvent_t ev; HC(hipEventCreateWithFlags(&ev,hipEventDisableTiming));
        hipGraph_t graph; hipGraphExec_t exec;
        HC(hipStreamBeginCapture(s,hipStreamCaptureModeGlobal));
        hipLaunchKernelGGL(k_copy ,dim3(blocks),dim3(BS),0,s,b.y,b.x,b.M);
        HC(hipEventRecord(ev,s));
        HC(hipStreamWaitEvent(s,ev,0));
        hipLaunchKernelGGL(k_blend,dim3(blocks),dim3(BS),0,s,b.x,b.y,b.M);
        HC(hipStreamEndCapture(s,&graph));
        HC(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for(int w=0;w<10;++w){ HC(hipGraphLaunch(exec,s)); } HC(hipStreamSynchronize(s));
        reset(b,xs,ys,ws);
        HC(hipGraphLaunch(exec,s)); HC(hipStreamSynchronize(s));
        printf("barriernode checksum=%.6f\n",checksum(b));
        HC(hipGraphExecDestroy(exec)); HC(hipGraphDestroy(graph)); HC(hipEventDestroy(ev));
    }

    // varblk: consecutive kernels with DIFFERENT block sizes (64/128/256). The
    // COMPUTE_NUM_THREAD_X sub-range of kRegStartX changes each time, exercising
    // both the delta SKIP (same block repeated) and the partial re-emit (block
    // change). Also mixes same-kernel-back-to-back to exercise PGM/RSRC skip.
    {
        double ck=run_graph(b,s,xs,ys,ws,[&](hipStream_t st){
            const int bss[4]={256,64,256,128};
            for(int l=0;l<4;++l){
                int bs=bss[l]; int bl=(M+bs-1)/bs;
                hipLaunchKernelGGL(k_copy ,dim3(bl),dim3(bs),0,st,b.y,b.x,b.M);
                hipLaunchKernelGGL(k_copy ,dim3(bl),dim3(bs),0,st,b.x,b.y,b.M); // repeat: PGM/RSRC skip
                hipLaunchKernelGGL(k_blend,dim3(bl),dim3(bs),0,st,b.x,b.y,b.M);
            }
        });
        printf("varblk      checksum=%.6f\n",ck);
    }

    // scratch: spilling kernel (opt-in via HIP_PM4_GRAPH_SCRATCH). First launch
    // defers (AQL sizes queue scratch); subsequent launches use the PM4 IB.
    {
        double ck=run_graph(b,s,xs,ys,ws,[&](hipStream_t st){
            hipLaunchKernelGGL(k_copy   ,dim3(blocks),dim3(BS),0,st,b.y,b.x,b.M);
            hipLaunchKernelGGL(k_scratch,dim3(blocks),dim3(BS),0,st,b.x,b.y,b.M);
        });
        printf("scratch     checksum=%.6f\n",ck);
    }

    HC(hipStreamDestroy(s));
    HC(hipFree(b.x)); HC(hipFree(b.y)); HC(hipFree(b.w));
    return 0;
}

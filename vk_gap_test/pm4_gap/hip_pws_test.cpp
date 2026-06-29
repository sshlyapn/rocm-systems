// hip_pws_test.cpp
//
// Real-HIP-runtime test + benchmark for the experimental PWS inter-kernel fence
// (CLR patch, gated by HIP_PWS_FENCE=1, gfx11 only).
//
// Runs a serially-dependent multi-kernel chain (the decode-like pattern that the
// PM4 study targeted) two ways -- eager stream launches and a captured hipGraph --
// and reports:
//   * a checksum of the final buffer (for bit-exact correctness checking across
//     HIP_PWS_FENCE on/off runs), and
//   * steady-state timing (us per chain, us per dispatch).
//
// Usage:  hip_pws_test [M] [LAYERS] [ITERS] [mode]
//   mode = eager | graph | both   (default both)
// Compare runs:
//   ./hip_pws_test               # baseline (firmware scope fence)
//   HIP_PWS_FENCE=1 ./hip_pws_test
// The checksums must match; the PWS timing should be <= baseline on gfx11.
//
// Plain ASCII only in comments.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <ctime>

#define HC(call) do { hipError_t _e=(call); if(_e!=hipSuccess){ \
  fprintf(stderr,"HIP error %s at %s:%d: %s\n",#call,__FILE__,__LINE__,hipGetErrorString(_e)); \
  exit(1);} } while(0)

#define BS 256

__global__ void k_copy (float* y, const float* x, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]; }
__global__ void k_scale(float* y, const float* w, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]*=w[i]; }
__global__ void k_blend(float* x, const float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=0.5f*x[i]+0.5f*y[i]; }
__global__ void k_add  (float* y, const float* w, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]+=w[i]; }
// reverse-read: consumer reads a different lane than the producer wrote -> stale
// L0/L1 line shows up unless the inter-kernel fence really invalidates.
__global__ void k_revadd(float* x, const float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ int j=n-1-i; x[i]=0.5f*x[i]+0.5f*y[j]; } }

static double now_s(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

struct Buffers { float *x,*y,*w; int M; };

// one decode-like layer: 5 dependent dispatches (copy,scale,mid,add,mid)
static void chain(const Buffers& b, hipStream_t s, int layers, bool rev){
    int blocks=(b.M+BS-1)/BS;
    for(int l=0;l<layers;++l){
        hipLaunchKernelGGL(k_copy ,dim3(blocks),dim3(BS),0,s, b.y,b.x,b.M);
        hipLaunchKernelGGL(k_scale,dim3(blocks),dim3(BS),0,s, b.y,b.w,b.M);
        if(rev) hipLaunchKernelGGL(k_revadd,dim3(blocks),dim3(BS),0,s, b.x,b.y,b.M);
        else    hipLaunchKernelGGL(k_blend ,dim3(blocks),dim3(BS),0,s, b.x,b.y,b.M);
        hipLaunchKernelGGL(k_add  ,dim3(blocks),dim3(BS),0,s, b.y,b.w,b.M);
        if(rev) hipLaunchKernelGGL(k_revadd,dim3(blocks),dim3(BS),0,s, b.x,b.y,b.M);
        else    hipLaunchKernelGGL(k_blend ,dim3(blocks),dim3(BS),0,s, b.x,b.y,b.M);
    }
}

static void reset(const Buffers& b, const std::vector<float>& xs,
                  const std::vector<float>& ys, const std::vector<float>& ws){
    HC(hipMemcpy(b.x,xs.data(),b.M*4,hipMemcpyHostToDevice));
    HC(hipMemcpy(b.y,ys.data(),b.M*4,hipMemcpyHostToDevice));
    HC(hipMemcpy(b.w,ws.data(),b.M*4,hipMemcpyHostToDevice));
}

static double checksum(const Buffers& b){
    std::vector<float> out(b.M);
    HC(hipMemcpy(out.data(),b.x,b.M*4,hipMemcpyDeviceToHost));
    double s=0; for(int i=0;i<b.M;++i) s+=out[i]; return s;
}

int main(int argc,char** argv){
    int M     = argc>1?atoi(argv[1]):16384;
    int LAYERS= argc>2?atoi(argv[2]):8;
    int ITERS = argc>3?atoi(argv[3]):200;
    std::string mode = argc>4?argv[4]:"both";
    bool rev = getenv("REV")!=nullptr;
    int dispPerIter = LAYERS*5;

    int dev=0; HC(hipSetDevice(dev));
    hipDeviceProp_t prop; HC(hipGetDeviceProperties(&prop,dev));
    printf("GPU: %s  gcn=%s  M=%d LAYERS=%d ITERS=%d rev=%d  HIP_PWS_FENCE=%s\n",
           prop.name, prop.gcnArchName, M, LAYERS, ITERS, rev,
           getenv("HIP_PWS_FENCE")?getenv("HIP_PWS_FENCE"):"(unset)");

    Buffers b; b.M=M;
    HC(hipMalloc(&b.x,M*4)); HC(hipMalloc(&b.y,M*4)); HC(hipMalloc(&b.w,M*4));
    std::vector<float> xs(M),ys(M),ws(M);
    for(int i=0;i<M;++i){ xs[i]=(i%7)*0.01f+0.1f; ws[i]=1.0f+((i%5)*0.001f); ys[i]=(i%9)*0.01f+0.05f; }

    hipStream_t s; HC(hipStreamCreate(&s));

    // ---- correctness (one chain, fresh inputs) ----
    reset(b,xs,ys,ws);
    chain(b,s,LAYERS,rev);
    HC(hipStreamSynchronize(s));
    double ck = checksum(b);

    // ---- eager benchmark ----
    if(mode=="eager"||mode=="both"){
        for(int w=0;w<20;++w){ chain(b,s,LAYERS,rev); } HC(hipStreamSynchronize(s));
        double t0=now_s();
        for(int it=0;it<ITERS;++it){ chain(b,s,LAYERS,rev); }
        HC(hipStreamSynchronize(s));
        double us=(now_s()-t0)*1e6/ITERS;
        printf("  eager : checksum=%.6f  %.2f us/chain  %.3f us/dispatch\n", ck, us, us/dispPerIter);
    }

    // ---- hipGraph benchmark ----
    if(mode=="graph"||mode=="both"){
        hipGraph_t graph; hipGraphExec_t exec;
        HC(hipStreamBeginCapture(s,hipStreamCaptureModeGlobal));
        chain(b,s,LAYERS,rev);
        HC(hipStreamEndCapture(s,&graph));
        HC(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for(int w=0;w<20;++w){ HC(hipGraphLaunch(exec,s)); } HC(hipStreamSynchronize(s));
        // graph correctness (fresh inputs)
        reset(b,xs,ys,ws);
        HC(hipGraphLaunch(exec,s)); HC(hipStreamSynchronize(s));
        double ckg = checksum(b);
        double t0=now_s();
        for(int it=0;it<ITERS;++it){ HC(hipGraphLaunch(exec,s)); }
        HC(hipStreamSynchronize(s));
        double us=(now_s()-t0)*1e6/ITERS;
        printf("  graph : checksum=%.6f  %.2f us/chain  %.3f us/dispatch\n", ckg, us, us/dispPerIter);
        HC(hipGraphExecDestroy(exec)); HC(hipGraphDestroy(graph));
    }

    HC(hipStreamDestroy(s));
    HC(hipFree(b.x)); HC(hipFree(b.y)); HC(hipFree(b.w));
    return 0;
}

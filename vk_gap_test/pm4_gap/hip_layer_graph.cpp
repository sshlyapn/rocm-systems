// hip_layer_graph.cpp
//
// Fair, PRE-RECORDED HIP counterpart of vk_layer / pm4_layer. hip_layer.cpp
// issues the chain with a CPU launch loop, so at small M its wall time includes
// per-dispatch CPU launch overhead (hipModuleLaunchKernel cost * 3 * N) that the
// pre-recorded Vulkan command buffer and the pre-built PM4 stream do NOT pay.
// This version captures the SAME "elem" chain (x+=w; x*=w; x+=w) x N into a
// hipGraph and launches the whole graph ONCE, mirroring the single Vulkan submit
// and the single PM4 doorbell -- so the three numbers measure the same thing
// (GPU-side execution of one pre-recorded stream), not CPU launch rate.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <vector>

#define BS 256
#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP err %s @%d: %s\n",#x,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)

static double now_s(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }

static void launch(hipFunction_t f, dim3 grid, hipStream_t s, void* p0, void* p1, int n){
    char buf[24]; memcpy(buf,&p0,8); memcpy(buf+8,&p1,8); memcpy(buf+16,&n,4);
    size_t sz=20;
    void* cfg[]={ HIP_LAUNCH_PARAM_BUFFER_POINTER, buf,
                  HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };
    HC(hipModuleLaunchKernel(f, grid.x,grid.y,grid.z, BS,1,1, 0, s, nullptr, cfg));
}

int main(int argc,char** argv){
    int N=argc>1?atoi(argv[1]):1000;
    int M=argc>2?atoi(argv[2]):4096;
    const char* co=argc>3?argv[3]:"layer.co";

    hipModule_t mod; HC(hipModuleLoad(&mod,co));
    hipFunction_t fScale,fAdd,fRev,fCopy;
    HC(hipModuleGetFunction(&fScale,mod,"k_scale"));
    HC(hipModuleGetFunction(&fAdd,mod,"k_add"));
    HC(hipModuleGetFunction(&fRev,mod,"k_revadd"));
    HC(hipModuleGetFunction(&fCopy,mod,"k_copy"));
    hipFunction_t fBlend; HC(hipModuleGetFunction(&fBlend,mod,"k_blend"));

    const char* chainEnv=getenv("CHAIN");
    bool rev = chainEnv && !strcmp(chainEnv,"rev");
    bool mix = chainEnv && !strcmp(chainEnv,"mix");
    bool serial = chainEnv && !strcmp(chainEnv,"serial");
    int KLEN = getenv("KLEN") ? atoi(getenv("KLEN")) : 64;
    if(KLEN<2) KLEN=2; KLEN &= ~1;
    int LAYERS = getenv("LAYERS") ? atoi(getenv("LAYERS")) : 1;
    if(LAYERS<1) LAYERS=1;
    int kpi = rev ? KLEN : ((mix||serial) ? 5*LAYERS : 3);

    size_t bytes=(size_t)M*4;
    float *x,*w,*y; HC(hipMalloc(&x,bytes)); HC(hipMalloc(&w,bytes)); HC(hipMalloc(&y,bytes));
    std::vector<float> hx(M),hw(M),hy(M);
    for(int i=0;i<M;++i){ hx[i]=(i%7)*0.01f+0.1f; hw[i]=1.0f+((i%5)*0.001f); hy[i]=(i%9)*0.01f+0.05f; }
    auto initbufs=[&](){
        HC(hipMemcpy(x,hx.data(),bytes,hipMemcpyHostToDevice));
        HC(hipMemcpy(w,hw.data(),bytes,hipMemcpyHostToDevice));
        HC(hipMemcpy(y,hy.data(),bytes,hipMemcpyHostToDevice));
    };
    initbufs();

    dim3 gFull((M+BS-1)/BS);
    hipStream_t stream; HC(hipStreamCreate(&stream));

    // capture the whole N-iteration chain into one graph (pre-recorded, like the
    // Vulkan command buffer / PM4 stream). In-order stream => RAW dependency is
    // honored by the runtime's normal acquire/release (AQL), no CPU per launch.
    hipGraph_t graph; hipGraphExec_t exec;
    HC(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for(int it=0;it<N;++it){
        if(rev){
            for(int k=0;k<KLEN;++k){
                if(k & 1) launch(fRev, gFull, stream, x,y,M);   // x = 0.5x + 0.5*rev(y)
                else      launch(fRev, gFull, stream, y,x,M);   // y = 0.5y + 0.5*rev(x)
            }
        } else if(mix||serial){
            hipFunction_t fMid = serial ? fBlend : fRev;
            for(int l=0;l<LAYERS;++l){
                launch(fCopy, gFull, stream, y,x,M);   // y = x
                launch(fScale,gFull, stream, y,w,M);   // y *= w
                launch(fMid,  gFull, stream, x,y,M);   // x = 0.5x + 0.5*(rev(y) | y)
                launch(fAdd,  gFull, stream, y,w,M);   // y += w
                launch(fMid,  gFull, stream, x,y,M);   // x = 0.5x + 0.5*(rev(y) | y)
            }
        } else {
            launch(fAdd,  gFull, stream, x,w,M);   // x += w
            launch(fScale,gFull, stream, x,w,M);   // x *= w
            launch(fAdd,  gFull, stream, x,w,M);   // x += w
        }
    }
    HC(hipStreamEndCapture(stream,&graph));
    HC(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));

    // warmup launch, then reset buffers so the timed run starts from same state
    HC(hipGraphLaunch(exec,stream)); HC(hipStreamSynchronize(stream));
    initbufs(); HC(hipDeviceSynchronize());

    double t0=now_s();
    HC(hipGraphLaunch(exec,stream));
    HC(hipStreamSynchronize(stream));
    double t1=now_s();
    double us=(t1-t0)*1e6;

    HC(hipMemcpy(hx.data(),x,bytes,hipMemcpyDeviceToHost));
    double sum=0; for(int i=0;i<M;++i) sum+=hx[i];
    printf("result : checksum(x)=%.6f x[0]=%.6f x[%d]=%.6f\n",sum,hx[0],M-1,hx[M-1]);
    printf("         CPU wall %.2f us  PER-ITER(%d kern)=%.3f us  PER-DISPATCH=%.3f us  (hipGraph, pre-recorded)\n",
           us, kpi, us/N, us/(double)(N*kpi));
    return 0;
}

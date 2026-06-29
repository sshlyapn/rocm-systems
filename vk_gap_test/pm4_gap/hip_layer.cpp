// hip_layer.cpp
//
// HIP/AQL reference for the decode-layer chain: loads the SAME code object as
// pm4_layer (layer.co) and launches k_rms -> k_scale -> k_add N times on a
// single in-order stream, then measures aggregate per-iteration/per-dispatch
// period. Compare checksum + timing against pm4_layer to isolate the AQL
// per-dispatch overhead vs the PM4 front-end on identical kernels.

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

struct Args { void* p0; void* p1; int n; };  // matches (ptr,ptr,int); 20B used

static void launch(hipFunction_t f, dim3 grid, void* p0, void* p1, int n){
    char buf[24]; memcpy(buf,&p0,8); memcpy(buf+8,&p1,8); memcpy(buf+16,&n,4);
    size_t sz=20;
    void* cfg[]={ HIP_LAUNCH_PARAM_BUFFER_POINTER, buf,
                  HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };
    HC(hipModuleLaunchKernel(f, grid.x,grid.y,grid.z, BS,1,1, 0, 0, nullptr, cfg));
}

int main(int argc,char** argv){
    int N=argc>1?atoi(argv[1]):2000;
    int M=argc>2?atoi(argv[2]):4096;
    const char* co=argc>3?argv[3]:"layer.co";

    hipModule_t mod; HC(hipModuleLoad(&mod,co));
    hipFunction_t fRms,fScale,fAdd,fRev,fCopy;
    HC(hipModuleGetFunction(&fRms,mod,"k_rms"));
    HC(hipModuleGetFunction(&fScale,mod,"k_scale"));
    HC(hipModuleGetFunction(&fAdd,mod,"k_add"));
    HC(hipModuleGetFunction(&fRev,mod,"k_revadd"));
    HC(hipModuleGetFunction(&fCopy,mod,"k_copy"));
    hipFunction_t fBlend; HC(hipModuleGetFunction(&fBlend,mod,"k_blend"));

    const char* chainEnv=getenv("CHAIN");
    bool elem = chainEnv && !strcmp(chainEnv,"elem");
    bool rev  = chainEnv && !strcmp(chainEnv,"rev");
    bool mix  = chainEnv && !strcmp(chainEnv,"mix");
    bool serial = chainEnv && !strcmp(chainEnv,"serial");
    int KLEN = getenv("KLEN") ? atoi(getenv("KLEN")) : 64;
    if(KLEN<2) KLEN=2; KLEN &= ~1;
    int LAYERS = getenv("LAYERS") ? atoi(getenv("LAYERS")) : 1;
    if(LAYERS<1) LAYERS=1;
    int kpi = rev ? KLEN : ((mix||serial) ? 5*LAYERS : 3);

    size_t bytes=(size_t)M*4;
    float *x,*y,*w; HC(hipMalloc(&x,bytes)); HC(hipMalloc(&y,bytes)); HC(hipMalloc(&w,bytes));
    std::vector<float> hx(M),hw(M),hy(M);
    for(int i=0;i<M;++i){ hx[i]=(i%7)*0.01f+0.1f; hw[i]=1.0f+((i%5)*0.001f); hy[i]=(i%9)*0.01f+0.05f; }
    auto initbufs=[&](){
        HC(hipMemcpy(x,hx.data(),bytes,hipMemcpyHostToDevice));
        HC(hipMemcpy(w,hw.data(),bytes,hipMemcpyHostToDevice));
        HC(hipMemcpy(y,hy.data(),bytes,hipMemcpyHostToDevice));
    };
    initbufs();

    dim3 gFull((M+BS-1)/BS);
    auto run_iter=[&](){
        if(rev){
            for(int k=0;k<KLEN;++k){
                if(k & 1) launch(fRev, gFull, x,y,M);   // x = 0.5x + 0.5*rev(y)
                else      launch(fRev, gFull, y,x,M);   // y = 0.5y + 0.5*rev(x)
            }
        } else if(mix||serial){
            // decode-like: 5 DISTINCT kernels rebound each, repeated LAYERS times to
            // model a full token (~300 dispatches). Forward k_blend (serial) vs
            // reverse k_revadd (mix) is the only difference; isolates dispatch cost.
            hipFunction_t fMid = serial ? fBlend : fRev;
            for(int l=0;l<LAYERS;++l){
                launch(fCopy, gFull,   y,x,M);   // y = x
                launch(fScale,gFull,   y,w,M);   // y *= w
                launch(fMid,  gFull,   x,y,M);   // x = 0.5x + 0.5*(rev(y) | y)
                launch(fAdd,  gFull,   y,w,M);   // y += w
                launch(fMid,  gFull,   x,y,M);   // x = 0.5x + 0.5*(rev(y) | y)
            }
        } else if(elem){
            launch(fAdd,  gFull, x,w,M);   // x += w
            launch(fScale,gFull, x,w,M);   // x *= w
            launch(fAdd,  gFull, x,w,M);   // x += w
        } else {
            launch(fRms,  dim3(1), x,y,M);
            launch(fScale,gFull,   y,w,M);
            launch(fAdd,  gFull,   x,y,M);
        }
    };

    // warmup (also resets buffers so the timed run starts from the same state)
    run_iter(); HC(hipDeviceSynchronize());
    initbufs();
    HC(hipDeviceSynchronize());

    double t0=now_s();
    for(int it=0;it<N;++it) run_iter();
    HC(hipDeviceSynchronize());
    double t1=now_s();
    double us=(t1-t0)*1e6;

    HC(hipMemcpy(hx.data(),x,bytes,hipMemcpyDeviceToHost));
    double sum=0; for(int i=0;i<M;++i) sum+=hx[i];
    printf("result : checksum(x)=%.6f x[0]=%.6f x[%d]=%.6f\n",sum,hx[0],M-1,hx[M-1]);
    printf("         total %.2f us  PER-ITER(%d kern)=%.3f us  PER-DISPATCH=%.3f us\n",
           us, kpi, us/N, us/(double)(N*kpi));
    return 0;
}

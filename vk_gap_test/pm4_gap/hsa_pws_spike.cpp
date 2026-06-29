// hsa_pws_spike.cpp
//
// FEASIBILITY SPIKE for the CLR PWS-fence work.
//
// Question this answers: can we inject an INLINE vendor PM4-IB AQL packet
// (AMD_AQL_FORMAT_PM4_IB) carrying our PWS deferred-wait fence
// (CS_PARTIAL_FLUSH + RELEASE_MEM(PWS) + ACQUIRE_MEM(PWS)) into a NORMAL HSA
// compute (AQL) queue, between dependent kernel dispatches, with
// completion_signal = 0 (no per-fence blocking signal), and have the firmware:
//   (a) execute the IB (run the fence),
//   (b) advance the read index / not hang,
//   (c) produce a bit-exact result vs the standard AQL packet-scope fence.
//
// This is exactly the mechanism rocclr would use in rocvirtual.cpp. It is built
// on the public HSA runtime (NOT hsakmt), so it exercises the same firmware AQL
// packet processor path the real HIP runtime uses.
//
// Strategy: run the SAME dependent kernel chain (from layer_gfx1100.co) three
// ways on a real AQL queue and compare final checksums:
//   FENCE=scope : dispatch packets carry AGENT acquire/release scope (firmware
//                 fences). This is the reference == what HIP does today.
//   FENCE=pws   : dispatch packets carry NONE scope, and an inline vendor
//                 PM4-IB (PWS fence) is injected after each dispatch.
//   FENCE=none  : NONE scope, no injection (barrier bit only, no cache flush)
//                 -- diagnostic: should diverge on the reverse-read chain.
// pws == scope (bit-exact) AND no hang  => the injection mechanism is sound.
//
// Plain ASCII only in comments.

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <sched.h>
#include <vector>
#include <string>

#define BS 256
#define CK(expr) do { hsa_status_t _s=(expr); if(_s!=HSA_STATUS_SUCCESS){ \
    const char* m=nullptr; hsa_status_string(_s,&m); \
    fprintf(stderr,"FAIL %s = 0x%x %s (line %d)\n",#expr,_s,m?m:"",__LINE__); exit(1);} } while(0)

static double now_s(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }

// ----------------------------------------------------------------------------
// Agent / pool discovery
// ----------------------------------------------------------------------------
struct Agents { hsa_agent_t gpu{}; hsa_agent_t cpu{}; bool haveGpu=false, haveCpu=false; };

static hsa_status_t find_agents(hsa_agent_t a, void* data){
    auto* ag=(Agents*)data; hsa_device_type_t t;
    CK(hsa_agent_get_info(a,HSA_AGENT_INFO_DEVICE,&t));
    if(t==HSA_DEVICE_TYPE_GPU && !ag->haveGpu){ ag->gpu=a; ag->haveGpu=true; }
    if(t==HSA_DEVICE_TYPE_CPU && !ag->haveCpu){ ag->cpu=a; ag->haveCpu=true; }
    return HSA_STATUS_SUCCESS;
}

struct PoolPick { hsa_amd_memory_pool_t pool{}; bool found=false; bool wantFine; };
static hsa_status_t find_pool(hsa_amd_memory_pool_t p, void* data){
    auto* pk=(PoolPick*)data; if(pk->found) return HSA_STATUS_SUCCESS;
    hsa_amd_segment_t seg; CK(hsa_amd_memory_pool_get_info(p,HSA_AMD_MEMORY_POOL_INFO_SEGMENT,&seg));
    if(seg!=HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t fl; CK(hsa_amd_memory_pool_get_info(p,HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,&fl));
    bool fine = fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
    bool coarse = fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    bool allowed=false; CK(hsa_amd_memory_pool_get_info(p,HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,&allowed));
    if(!allowed) return HSA_STATUS_SUCCESS;
    if(pk->wantFine ? fine : coarse){ pk->pool=p; pk->found=true; }
    return HSA_STATUS_SUCCESS;
}

// kernarg region (fine-grained, kernarg flag) via the region API
struct RegionPick { hsa_region_t region{}; bool found=false; };
static hsa_status_t find_kernarg(hsa_region_t r, void* data){
    auto* rp=(RegionPick*)data; if(rp->found) return HSA_STATUS_SUCCESS;
    hsa_region_segment_t seg; CK(hsa_region_get_info(r,HSA_REGION_INFO_SEGMENT,&seg));
    if(seg!=HSA_REGION_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t fl; CK(hsa_region_get_info(r,HSA_REGION_INFO_GLOBAL_FLAGS,&fl));
    if(fl & HSA_REGION_GLOBAL_FLAG_KERNARG){ rp->region=r; rp->found=true; }
    return HSA_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// Kernel symbol lookup
// ----------------------------------------------------------------------------
struct Kern { uint64_t object=0; uint32_t karg=0, group=0, priv=0; };
static Kern get_kernel(hsa_executable_t exe, hsa_agent_t gpu, const char* name){
    hsa_executable_symbol_t sym; 
    hsa_status_t s = hsa_executable_get_symbol_by_name(exe, name, &gpu, &sym);
    if(s!=HSA_STATUS_SUCCESS){
        // try with a trailing nothing / kd variants
        std::string kd = std::string(name)+".kd";
        s = hsa_executable_get_symbol_by_name(exe, kd.c_str(), &gpu, &sym);
    }
    if(s!=HSA_STATUS_SUCCESS){ fprintf(stderr,"symbol %s not found (0x%x)\n",name,s); exit(1); }
    Kern k;
    CK(hsa_executable_symbol_get_info(sym,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,&k.object));
    CK(hsa_executable_symbol_get_info(sym,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,&k.karg));
    CK(hsa_executable_symbol_get_info(sym,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,&k.group));
    CK(hsa_executable_symbol_get_info(sym,HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,&k.priv));
    return k;
}

// ----------------------------------------------------------------------------
// PM4 IB: CS_PARTIAL_FLUSH + RELEASE_MEM(PWS,GCR=0x380) + ACQUIRE_MEM(PWS wait).
// Dwords lifted verbatim from pm4_layer.cpp (Emit::partial_flush /
// release_mem_pws / acquire_pws) for GCR_AGENT_LIKE = GL1_INV|GLV_INV|GLK_INV.
// Header = type3(<<30) | (count=dw-2)<<16 | opcode<<8 | shaderType(1)<<1.
// ----------------------------------------------------------------------------
static const uint32_t PWS_IB[] = {
    // CS_PARTIAL_FLUSH (EVENT_WRITE, 2 dw): opcode 0x46
    0xC0004602u, 0x00000407u,                 // event_index=4, event_type=CS_PARTIAL_FLUSH(7)
    // RELEASE_MEM (8 dw): opcode 0x49, EVENT=BOTTOM_OF_PIPE_TS, GCR=0x380, PWS_ENABLE
    0xC0064902u, 0xC000C528u, 0,0,0,0,0,0,
    // ACQUIRE_MEM PWS wait (8 dw): opcode 0x58, PWS_STAGE_SEL=CP_ME, PWS_ENA, GCR_CNTL=0
    0xC0065802u, 0x00022800u, 0xFFFFFFFFu, 0x01FFFFFFu, 0, 0, 0x80000000u, 0,
};
static const uint32_t PWS_IB_DW = sizeof(PWS_IB)/4;

// ----------------------------------------------------------------------------
// Queue submission helpers (single doorbell per packet -- correctness-first)
// ----------------------------------------------------------------------------
static hsa_queue_t* g_q=nullptr;

static void submit_packet(const void* pkt64, uint32_t dword0){
    uint64_t idx = hsa_queue_add_write_index_screlease(g_q,1);
    while(idx - hsa_queue_load_read_index_scacquire(g_q) >= g_q->size) { sched_yield(); }
    uint32_t slot = (uint32_t)(idx % g_q->size);
    uint32_t* dst = (uint32_t*)((uintptr_t)g_q->base_address + slot*0x40);
    const uint32_t* src = (const uint32_t*)pkt64;
    // body first (dwords 1..15), then dword0 with release so CP sees a valid slot
    for(int i=1;i<16;i++) dst[i]=src[i];
    __atomic_store_n(&dst[0], dword0, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(g_q->doorbell_signal, idx);
}

// AQL kernel dispatch packet (64 bytes)
struct AqlDispatch {
    uint16_t header; uint16_t setup;
    uint16_t wg_x, wg_y, wg_z; uint16_t reserved0;
    uint32_t grid_x, grid_y, grid_z;
    uint32_t priv_seg, group_seg;
    uint64_t kernel_object;
    uint64_t kernarg_address;
    uint64_t reserved2;
    hsa_signal_t completion_signal;
};

static void dispatch(const Kern& k, void* kernarg, uint32_t threads,
                     int acqScope, int relScope, hsa_signal_t comp){
    AqlDispatch p; memset(&p,0,sizeof(p));
    p.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    p.wg_x = BS; p.wg_y=1; p.wg_z=1;
    p.grid_x = threads; p.grid_y=1; p.grid_z=1;
    p.priv_seg = k.priv; p.group_seg = k.group;
    p.kernel_object = k.object;
    p.kernarg_address = (uint64_t)kernarg;
    p.completion_signal = comp;
    uint16_t hdr = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE)
                 | (1u << HSA_PACKET_HEADER_BARRIER)
                 | ((uint16_t)acqScope << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
                 | ((uint16_t)relScope << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    p.header = hdr;
    uint32_t dword0 = (uint32_t)hdr | ((uint32_t)p.setup << 16);
    submit_packet(&p, dword0);
}

// vendor AMD_AQL_FORMAT_PM4_IB packet pointing to our PWS IB (inline, no signal)
struct AqlPm4Ib {
    uint16_t header; uint16_t ven_hdr;
    uint32_t ib_jump_cmd[4];
    uint32_t dw_cnt_remain;
    uint32_t reserved[8];
    hsa_signal_t completion_signal;
};
static void inject_pws(const uint32_t* ib, uint32_t ibDw, hsa_signal_t comp){
    constexpr uint32_t AMD_AQL_FORMAT_PM4_IB = 0x1;
    // INDIRECT_BUFFER PM4 (opcode 0x3F, 4 dw): base>>2 lo, base>>32 hi, size|valid.
    // shaderType=0 to match ROCr ExecutePM4 (PM4_HDR uses shaderType only on gfx7).
    uint32_t ibjump[4] = {
        (3u<<30) | ((4u-2u)<<16) | (0x3Fu<<8),
        (uint32_t)(((uintptr_t)ib >> 2) & 0x3FFFFFFFu) << 2,  // IB_BASE_LO[31:2]
        (uint32_t)((uintptr_t)ib >> 32) & 0xFFFFu,            // IB_BASE_HI
        (ibDw & 0xFFFFFu) | (1u<<23),                         // IB_SIZE | IB_VALID
    };
    AqlPm4Ib p; memset(&p,0,sizeof(p));
    uint16_t hdr = (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE);
    p.header = hdr; p.ven_hdr = AMD_AQL_FORMAT_PM4_IB;
    p.ib_jump_cmd[0]=ibjump[0]; p.ib_jump_cmd[1]=ibjump[1];
    p.ib_jump_cmd[2]=ibjump[2]; p.ib_jump_cmd[3]=ibjump[3];
    p.dw_cnt_remain = 0xA;
    p.completion_signal = comp;   // {0} == no blocking signal (the unknown under test)
    uint32_t dword0 = (uint32_t)hdr | ((uint32_t)p.ven_hdr << 16);
    if(getenv("DUMP")){
        const uint32_t* d=(const uint32_t*)&p;
        fprintf(stderr,"  vendor pkt dwords: %08x %08x %08x %08x %08x  dw_cnt=%08x  sig=%llx\n",
                dword0,d[1],d[2],d[3],d[4],d[5],(unsigned long long)comp.handle);
    }
    submit_packet(&p, dword0);
}

// ----------------------------------------------------------------------------

int main(int argc,char** argv){
    int M      = argc>1?atoi(argv[1]):16384;
    int LAYERS = argc>2?atoi(argv[2]):8;
    const char* co = argc>3?argv[3]:"layer_gfx1100.co";
    setvbuf(stdout,nullptr,_IONBF,0);
    const char* fenceEnv = getenv("FENCE") ? getenv("FENCE") : "pws";
    bool useSignalOnIb = getenv("IB_SIGNAL")!=nullptr;  // diagnostic: give the IB packet a real signal
    bool revChain = getenv("REV")!=nullptr;             // include reverse-read kernel (coherence stress)

    CK(hsa_init());
    Agents ag; CK(hsa_iterate_agents(find_agents,&ag));
    if(!ag.haveGpu||!ag.haveCpu){ fprintf(stderr,"need GPU+CPU agent\n"); return 1; }
    { char nm[64]={0}; hsa_agent_get_info(ag.gpu,HSA_AGENT_INFO_NAME,nm);
      printf("GPU agent: %s  M=%d LAYERS=%d FENCE=%s rev=%d ib_signal=%d\n",
             nm,M,LAYERS,fenceEnv,revChain,useSignalOnIb); }

    PoolPick devPick{}; devPick.wantFine=false;
    CK(hsa_amd_agent_iterate_memory_pools(ag.gpu,find_pool,&devPick));
    PoolPick hostPick{}; hostPick.wantFine=true;
    CK(hsa_amd_agent_iterate_memory_pools(ag.cpu,find_pool,&hostPick));
    if(!devPick.found||!hostPick.found){ fprintf(stderr,"pool discovery failed\n"); return 1; }
    RegionPick kr{}; CK(hsa_agent_iterate_regions(ag.gpu,find_kernarg,&kr));
    if(!kr.found){ fprintf(stderr,"no kernarg region\n"); return 1; }

    // load code object
    FILE* fp=fopen(co,"rb"); if(!fp){ fprintf(stderr,"open %s\n",co); return 1; }
    fseek(fp,0,SEEK_END); long flen=ftell(fp); fseek(fp,0,SEEK_SET);
    std::vector<uint8_t> buf(flen); if(fread(buf.data(),1,flen,fp)!=(size_t)flen){return 1;} fclose(fp);
    hsa_code_object_reader_t reader;
    CK(hsa_code_object_reader_create_from_memory(buf.data(),flen,&reader));
    hsa_executable_t exe;
    CK(hsa_executable_create_alt(HSA_PROFILE_FULL,HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,"",&exe));
    CK(hsa_executable_load_agent_code_object(exe,ag.gpu,reader,"",nullptr));
    CK(hsa_executable_freeze(exe,""));

    Kern kCopy=get_kernel(exe,ag.gpu,"k_copy");
    Kern kScale=get_kernel(exe,ag.gpu,"k_scale");
    Kern kAdd=get_kernel(exe,ag.gpu,"k_add");
    Kern kBlend=get_kernel(exe,ag.gpu,"k_blend");
    Kern kRev=get_kernel(exe,ag.gpu,"k_revadd");

    size_t bytes=(size_t)M*4;
    // device buffers (coarse-grained, == hipMalloc, L2-cached)
    float *x,*y,*w;
    CK(hsa_amd_memory_pool_allocate(devPick.pool,bytes,0,(void**)&x));
    CK(hsa_amd_memory_pool_allocate(devPick.pool,bytes,0,(void**)&y));
    CK(hsa_amd_memory_pool_allocate(devPick.pool,bytes,0,(void**)&w));
    CK(hsa_amd_agents_allow_access(1,&ag.gpu,nullptr,x));
    CK(hsa_amd_agents_allow_access(1,&ag.gpu,nullptr,y));
    CK(hsa_amd_agents_allow_access(1,&ag.gpu,nullptr,w));
    // host staging (fine-grained, CPU+GPU)
    float *xs,*ws,*ys,*outs;
    CK(hsa_amd_memory_pool_allocate(hostPick.pool,bytes,0,(void**)&xs));
    CK(hsa_amd_memory_pool_allocate(hostPick.pool,bytes,0,(void**)&ws));
    CK(hsa_amd_memory_pool_allocate(hostPick.pool,bytes,0,(void**)&ys));
    CK(hsa_amd_memory_pool_allocate(hostPick.pool,bytes,0,(void**)&outs));
    hsa_agent_t bothA[2]={ag.gpu,ag.cpu};
    CK(hsa_amd_agents_allow_access(2,bothA,nullptr,xs));
    CK(hsa_amd_agents_allow_access(2,bothA,nullptr,ws));
    CK(hsa_amd_agents_allow_access(2,bothA,nullptr,ys));
    CK(hsa_amd_agents_allow_access(2,bothA,nullptr,outs));
    for(int i=0;i<M;++i){ xs[i]=(i%7)*0.01f+0.1f; ws[i]=1.0f+((i%5)*0.001f);
                          ys[i]=(i%9)*0.01f+0.05f; outs[i]=-1.0f; }

    // PM4 IB buffer: the CP indirect-buffer fetcher needs EXECUTABLE GPU memory
    // (a normal pool IB faults the CP). The public HSA API exposes this via
    // HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG -- which is exactly what rocclr can use
    // (no hsakmt dependency). Allocate from the device pool with that flag.
    uint32_t* ib=nullptr;
    CK(hsa_amd_memory_pool_allocate(devPick.pool,0x1000,
            HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG,(void**)&ib));
    CK(hsa_amd_agents_allow_access(1,&ag.gpu,nullptr,ib));
    uint32_t* ibHost; CK(hsa_amd_memory_pool_allocate(hostPick.pool,0x1000,0,(void**)&ibHost));
    CK(hsa_amd_agents_allow_access(2,bothA,nullptr,ibHost));
    memset(ibHost,0,0x1000); memcpy(ibHost,PWS_IB,sizeof(PWS_IB));
    fprintf(stderr,"  IB: executable-pool VRAM@%p (CP fetch)\n",(void*)ib);

    // kernargs (kernarg region) -- layout [ptr0(8)][ptr1(8)][n(4)]
    auto mk_ka=[&](uint64_t p0,uint64_t p1,int n)->void*{
        void* k; CK(hsa_memory_allocate(kr.region,32,&k));
        uint32_t* u=(uint32_t*)k; u[0]=(uint32_t)p0; u[1]=(uint32_t)(p0>>32);
        u[2]=(uint32_t)p1; u[3]=(uint32_t)(p1>>32); u[4]=(uint32_t)n; return k; };
    void* kaCopy = mk_ka((uint64_t)y,(uint64_t)x,M);   // y = x
    void* kaScale= mk_ka((uint64_t)y,(uint64_t)w,M);   // y *= w
    void* kaMid  = mk_ka((uint64_t)x,(uint64_t)y,M);   // x = 0.5x + 0.5*(y | rev(y))
    void* kaAdd  = mk_ka((uint64_t)y,(uint64_t)w,M);   // y += w

    // queue + signals
    CK(hsa_queue_create(ag.gpu,4096,HSA_QUEUE_TYPE_SINGLE,nullptr,nullptr,
                        UINT32_MAX,UINT32_MAX,&g_q));
    hsa_signal_t done; CK(hsa_signal_create(1,0,nullptr,&done));
    hsa_signal_t copySig; CK(hsa_signal_create(1,0,nullptr,&copySig));
    hsa_signal_t ibSig{}; if(useSignalOnIb) CK(hsa_signal_create(1,0,nullptr,&ibSig));

    auto h2d=[&](void* dptr,void* hptr){
        hsa_signal_store_screlease(copySig,1);
        CK(hsa_amd_memory_async_copy(dptr,ag.gpu,hptr,ag.cpu,bytes,0,nullptr,copySig));
        while(hsa_signal_wait_scacquire(copySig,HSA_SIGNAL_CONDITION_LT,1,UINT64_MAX,HSA_WAIT_STATE_BLOCKED)>=1){}
    };

    int fmode = !strcmp(fenceEnv,"scope")?0 : (!strcmp(fenceEnv,"none")?2:1); // 0 scope,1 pws,2 none
    int acq = (fmode==0)?HSA_FENCE_SCOPE_AGENT:HSA_FENCE_SCOPE_NONE;
    int rel = (fmode==0)?HSA_FENCE_SCOPE_AGENT:HSA_FENCE_SCOPE_NONE;

    const Kern& kMid = revChain ? kRev : kBlend;

    auto fence=[&](){
        if(fmode==1) inject_pws(ib,PWS_IB_DW, useSignalOnIb?ibSig:hsa_signal_t{0});
    };

    // stage the PM4 IB into the executable device buffer (CP fetches from VRAM)
    {
        hsa_signal_store_screlease(copySig,1);
        CK(hsa_amd_memory_async_copy(ib,ag.gpu,ibHost,ag.cpu,0x1000,0,nullptr,copySig));
        while(hsa_signal_wait_scacquire(copySig,HSA_SIGNAL_CONDITION_LT,1,UINT64_MAX,HSA_WAIT_STATE_BLOCKED)>=1){}
    }
    // reset device buffers from staging
    h2d(x,xs); h2d(w,ws); h2d(y,ys);

    double t0=now_s();
    int totalDisp = LAYERS*5;
    int disp=0;
    for(int l=0;l<LAYERS;++l){
        auto last=[&](int extra){ return (disp+extra)==totalDisp-1; };
        // y = x
        dispatch(kCopy ,kaCopy ,M,acq,rel, last(0)?done:hsa_signal_t{0}); fence(); disp++;
        // y *= w
        dispatch(kScale,kaScale,M,acq,rel, last(0)?done:hsa_signal_t{0}); fence(); disp++;
        // x = 0.5x + 0.5*(y|rev(y))
        dispatch(kMid  ,kaMid  ,M,acq,rel, last(0)?done:hsa_signal_t{0}); fence(); disp++;
        // y += w
        dispatch(kAdd  ,kaAdd  ,M,acq,rel, last(0)?done:hsa_signal_t{0}); fence(); disp++;
        // x = 0.5x + 0.5*(y|rev(y))
        dispatch(kMid  ,kaMid  ,M,acq,rel, last(0)?done:hsa_signal_t{0}); fence(); disp++;
    }
    // wait for the chain (the last dispatch carries 'done')
    const uint64_t TIMEOUT_NS = 10ull*1000*1000*1000;
    if(hsa_signal_wait_scacquire(done,HSA_SIGNAL_CONDITION_LT,1,TIMEOUT_NS,HSA_WAIT_STATE_BLOCKED)>=1){
        fprintf(stderr,"TIMEOUT/HANG waiting on chain completion (FENCE=%s)\n",fenceEnv);
        return 2;
    }
    double t1=now_s();

    // D2H x -> outs, with an AGENT acquire so the copy sees fresh x
    h2d(outs,outs); // no-op warm; ensure outs accessible
    hsa_signal_store_screlease(copySig,1);
    CK(hsa_amd_memory_async_copy(outs,ag.cpu,x,ag.gpu,bytes,0,nullptr,copySig));
    while(hsa_signal_wait_scacquire(copySig,HSA_SIGNAL_CONDITION_LT,1,UINT64_MAX,HSA_WAIT_STATE_BLOCKED)>=1){}

    double sum=0; for(int i=0;i<M;++i) sum+=outs[i];
    double us=(t1-t0)*1e6;
    printf("FENCE=%-5s  checksum(x)=%.6f  x[0]=%.6f x[%d]=%.6f  | %.1f us total, %.3f us/dispatch\n",
           fenceEnv, sum, outs[0], M-1, outs[M-1], us, us/(double)(LAYERS*5));

    hsa_queue_destroy(g_q);
    hsa_shut_down();
    return 0;
}

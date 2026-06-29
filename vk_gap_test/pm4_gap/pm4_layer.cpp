// pm4_layer.cpp
//
// "Decode-layer" emulation through a single raw KFD PM4 compute queue.
// Chains THREE real, distinct, dependent hipcc kernels (k_rms -> k_scale ->
// k_add, from layer_kernels.hip) N times, with a RADV-like fence
// (CS_PARTIAL_FLUSH + ACQUIRE_MEM full-L2 flush) between every dispatch, then
// measures the aggregate per-iteration / per-dispatch period. Compare against
// hip_layer (same kernels via HIP/AQL) to isolate the engine overhead delta on
// a realistic dependent chain (multiple KDs, LDS/group_segment, barriers).
//
// Reuses the ABI replication validated in pm4_real.cpp: copy rsrc1/rsrc2
// verbatim from each kernel's 64-byte descriptor, place the kernarg pointer in
// s[0:1] (only ENABLE_SGPR_KERNARG_SEGMENT_PTR set), no scratch.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <elf.h>
#include <vector>
#include <string>

#include "hsakmt/hsakmt.h"
#include "kfd_pm4_opcodes.h"
#include "pm4_pkt_struct_common.h"
#include "pm4_pkt_struct_ci.h"
#include "pm4_pkt_struct_ai.h"
#include "pm4_pkt_struct_nv.h"
#include "asic_reg/gfx_7_2_d.h"
#include "asic_reg/gfx_7_2_sh_mask.h"
#include "asic_reg/gfx_7_2_enum.h"

#define BS 256
#define CHECK(expr) do { HSAKMT_STATUS _s=(expr); if(_s!=HSAKMT_STATUS_SUCCESS){ \
    fprintf(stderr,"FAIL %s = %d (line %d)\n",#expr,_s,__LINE__); exit(1);} } while(0)

// GCR_CNTL bit layout for ACQUIRE_MEM, matching RADV/PAL (mesa pkt3.json GCR_CNTL):
//   GLI_INV[0:1] GL1_RANGE[2:3] GLM_WB[4] GLM_INV[5] GLK_WB[6] GLK_INV[7]
//   GLV_INV[8] GL1_INV[9] GL2_US[10] GL2_RANGE[11:12] GL2_DISCARD[13]
//   GL2_INV[14] GL2_WB[15] SEQ[16:17] RANGE_IS_PA[18]
enum { GCR_GLI_INV=1u<<0, GCR_GLM_WB=1u<<4, GCR_GLM_INV=1u<<5, GCR_GLK_WB=1u<<6,
       GCR_GLK_INV=1u<<7, GCR_GLV_INV=1u<<8, GCR_GL1_INV=1u<<9, GCR_GL2_DISCARD=1u<<13,
       GCR_GL2_INV=1u<<14, GCR_GL2_WB=1u<<15 };
// Full system-scope flush (what RADV emits on a VkMemoryBarrier): writeback +
// invalidate ALL caches incl. L2. Overkill for device-local dependencies.
static const uint32_t GCR_RADV_LIKE = GCR_GL2_WB|GCR_GL2_INV|GCR_GLM_WB|GCR_GLM_INV|
                                      GCR_GL1_INV|GCR_GLV_INV|GCR_GLK_INV|GCR_GLI_INV;
// AGENT-scope equivalent (what ROCr/HIP uses between dependent kernels on one
// device): L2 is the device coherence point and L0/L1 are write-through, so we
// only need to INVALIDATE the per-WGP L0 vector / L1 caches so the next kernel
// reads fresh from L2. No L2 writeback/invalidate.
// Default INCLUDES GLK_INV (scalar K-cache invalidate, bit 7) for safety: this is
// byte-identical to what RADV emits for a compute->compute shader-read barrier
// (VK_ACCESS_SHADER_READ -> INV_VCACHE|INV_SCACHE = GL1_INV|GLV_INV|GLK_INV).
// RADV must set GLK_INV because its ACO compiler reads SSBOs via SMEM (scalar)
// loads, so produced data can land in the K-cache. A general PM4 consumer may do
// the same, so we keep it on by default.
// OPTIMIZATION KNOB (GCR_RAW=0x300): dropping GLK_INV is bit-exact AND 2.5-5x
// cheaper per dispatch, but ONLY for kernels (like these hipcc/clang ones) that
// read buffer DATA via VECTOR loads (global_load) and whose kernargs are
// immutable -- verified in the ISA. Do NOT drop GLK_INV for a kernel that reads
// another kernel's output via scalar/uniform loads; it would read stale data.
static const uint32_t GCR_AGENT_LIKE = GCR_GL1_INV|GCR_GLV_INV|GCR_GLK_INV;

struct kernel_descriptor_t {
    uint32_t group_segment_fixed_size, private_segment_fixed_size, kernarg_size;
    uint8_t  reserved0[4];
    int64_t  kernel_code_entry_byte_offset;
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3, compute_pgm_rsrc1, compute_pgm_rsrc2;
    uint16_t kernel_code_properties, kernarg_preload;
    uint8_t  reserved2[4];
};
enum { KCP_PRIVATE_SEGMENT_BUFFER=1u<<0, KCP_DISPATCH_PTR=1u<<1, KCP_QUEUE_PTR=1u<<2,
       KCP_KERNARG_SEGMENT_PTR=1u<<3, KCP_FLAT_SCRATCH_INIT=1u<<5 };

static double now_s(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }
static void set_hdr(PM4_TYPE_3_HEADER&h, it_opcode_type op, unsigned dw){
    h.u32All=0; h.type=PM4_TYPE_3; h.shaderType=1; h.opcode=op; h.count=dw-2; }
static uint64_t next_pow2(uint64_t v){ uint64_t p=1; while(p<v) p<<=1; return p; }

struct Emit {
    uint32_t *base; uint32_t idx;
    explicit Emit(uint32_t* b):base(b),idx(0){}
    void set_sh_reg(unsigned reg, const uint32_t* v, unsigned n){
        unsigned dw=2+n; PM4_TYPE_3_HEADER h; set_hdr(h,IT_SET_SH_REG,dw);
        base[idx]=h.u32All; base[idx+1]=reg-PERSISTENT_SPACE_START;
        memcpy(&base[idx+2],v,n*4); idx+=dw; }
    void dispatch_direct(uint32_t x,uint32_t y,uint32_t z,uint32_t init){
        PM4DISPATCH_DIRECT p; memset(&p,0,sizeof(p));
        set_hdr(p.header,IT_DISPATCH_DIRECT,sizeof(p)/4);
        p.dim_x=x; p.dim_y=y; p.dim_z=z; p.dispatch_initiator=init;
        memcpy(&base[idx],&p,sizeof(p)); idx+=sizeof(p)/4; }
    void partial_flush(){
        PM4EVENT_WRITE p; memset(&p,0,sizeof(p)); set_hdr(p.header,IT_EVENT_WRITE,2);
        p.bitfields2.event_index=event_index_event_write_CS_VS_PS_PARTIAL_FLUSH_4;
        p.bitfields2.event_type=CS_PARTIAL_FLUSH;
        base[idx]=p.ordinal1; base[idx+1]=p.ordinal2; idx+=2; }
    void acquire_mem(uint32_t gcr){
        // Byte-for-byte match to RADV's gfx10/11 compute ACQUIRE_MEM (mesa
        // radv_cs.c gfx10_cs_emit_cache_flush): CP_COHER_CNTL=0, full coherency
        // range (SIZE=0xffffffff, SIZE_HI=0xffffff -- range is ignored when
        // GL1/GL2_RANGE=ALL but RADV programs the full extent), BASE=0,
        // POLL_INTERVAL=0x0A, then GCR_CNTL. Earlier this used SIZE_HI=0 and
        // POLL_INTERVAL=4, which did not match the driver.
        PM4ACQUIRE_MEM_NV p; memset(&p,0,sizeof(p));
        set_hdr(p.header,IT_ACQUIRE_MEM,sizeof(p)/4);
        p.reserved=0;                 // CP_COHER_CNTL
        p.coher_size=0xFFFFFFFF;      // CP_COHER_SIZE
        p.ordinal4=0x00FFFFFF;        // CP_COHER_SIZE_HI (RADV: 0xffffff)
        p.coher_base_lo=0;            // CP_COHER_BASE
        p.ordinal6=0;                 // CP_COHER_BASE_HI
        p.bitfields5.poll_interval=0x0A;
        p.bitfields6.gcr_cntl=gcr;
        memcpy(&base[idx],&p,sizeof(p)); idx+=sizeof(p)/4; }
    void write_data(uint64_t dst,uint32_t val){
        PM4WRITE_DATA_CI p; memset(&p,0,sizeof(p));
        unsigned dw=(offsetof(PM4WRITE_DATA_CI,data)/4)+1; set_hdr(p.header,IT_WRITE_DATA,dw);
        p.bitfields2.dst_sel=dst_sel_mec_write_data_MEMORY_5;
        p.bitfields2.addr_incr=addr_incr_mec_write_data_INCREMENT_ADDR_0;
        p.bitfields2.wr_confirm=wr_confirm_mec_write_data_WAIT_FOR_CONFIRMATION_1;
        p.bitfields2.atc=atc_write_data_NOT_USE_ATC_0;
        p.bitfields2.cache_policy=cache_policy_mec_write_data_BYPASS_2;
        p.dst_addr_lo=(uint32_t)dst; p.dst_address_hi=(uint32_t)(dst>>32);
        memcpy(&base[idx],&p,dw*4); base[idx+dw-1]=val; idx+=dw; }
    // GFX11 PWS (Pixel/Pre-shader Wait Sync) deferred-wait fence, replicating
    // RADV's overlap mechanism (mesa radv_cs.c gfx11 path + radv_queue.c):
    // a RELEASE_MEM EOP carries the GCR cache flush AND bumps the PWS counter
    // (issued async, the CP does not block on it), then a PWS ACQUIRE_MEM waits
    // on that counter. This DECOUPLES flush-issue from flush-wait so the flush can
    // overlap with the next dispatch instead of fully stalling the queue.
    void release_mem_pws(uint32_t gcrAcq){
        // translate ACQUIRE GCR_CNTL bit positions (S_586) to RELEASE_MEM_OP_gfx11 (S_490)
        uint32_t op = 40u | (5u<<8);          // EVENT_TYPE=BOTTOM_OF_PIPE_TS, EVENT_INDEX=5
        if(gcrAcq & (1u<<4)) op |= (1u<<12);  // GLM_WB
        if(gcrAcq & (1u<<5)) op |= (1u<<13);  // GLM_INV
        if(gcrAcq & (1u<<8)) op |= (1u<<14);  // GLV_INV
        if(gcrAcq & (1u<<9)) op |= (1u<<15);  // GL1_INV
        if(gcrAcq & (1u<<14))op |= (1u<<20);  // GL2_INV
        if(gcrAcq & (1u<<15))op |= (1u<<21);  // GL2_WB
        if(gcrAcq & (1u<<6)) op |= (1u<<24);  // GLK_WB
        if(gcrAcq & (1u<<7)) op |= (1u<<30);  // GLK_INV
        op |= ((gcrAcq>>16)&3u)<<22;          // SEQ
        op |= (1u<<31);                       // PWS_ENABLE
        PM4_TYPE_3_HEADER h; set_hdr(h,IT_RELEASE_MEM,8);
        base[idx++]=h.u32All; base[idx++]=op;
        base[idx++]=0; base[idx++]=0; base[idx++]=0; base[idx++]=0; base[idx++]=0; base[idx++]=0;
    }
    void acquire_pws(){
        PM4_TYPE_3_HEADER h; set_hdr(h,IT_ACQUIRE_MEM,8);
        base[idx++]=h.u32All;
        base[idx++]=(5u<<11)|(0u<<14)|(1u<<17)|(0u<<18); // PWS_STAGE_SEL=CP_ME, COUNTER_SEL=TS, ENA2=1, COUNT=0
        base[idx++]=0xFFFFFFFF;  // GCR_SIZE
        base[idx++]=0x01FFFFFF;  // GCR_SIZE_HI
        base[idx++]=0;           // GCR_BASE_LO
        base[idx++]=0;           // GCR_BASE_HI
        base[idx++]=(1u<<31);    // PWS_ENA
        base[idx++]=0;           // GCR_CNTL (flush already carried by RELEASE_MEM)
    }
};

static void* alloc_gpu(uint32_t node,uint64_t size,bool exec,bool uncached,bool coarse=false,bool hostAccess=true){
    HsaMemFlags f; f.Value=0; f.ui32.PageSize=HSA_PAGE_SIZE_4KB; f.ui32.HostAccess=hostAccess?1:0;
    f.ui32.NonPaged=coarse?1:0; f.ui32.CoarseGrain=coarse?1:0; f.ui32.NoNUMABind=1;
    f.ui32.Uncached=uncached?1:0; f.ui32.ExecuteAccess=exec?1:0;
    void* p=nullptr; CHECK(hsaKmtAllocMemory(node,size,f,&p));
    CHECK(hsaKmtMapMemoryToGPU(p,size,nullptr)); return p; }

struct KInfo { std::string name; uint64_t kdVA; kernel_descriptor_t kd; };

static uint64_t g_base = 0;
static std::vector<KInfo> g_kernels;

static void load_code_object(uint32_t node,const char* path){
    FILE* fp=fopen(path,"rb"); if(!fp){fprintf(stderr,"open %s\n",path);exit(1);}
    fseek(fp,0,SEEK_END); long flen=ftell(fp); fseek(fp,0,SEEK_SET);
    std::vector<uint8_t> file(flen);
    if(fread(file.data(),1,flen,fp)!=(size_t)flen){fprintf(stderr,"short read\n");exit(1);}
    fclose(fp);
    auto* eh=(Elf64_Ehdr*)file.data();
    auto* ph=(Elf64_Phdr*)(file.data()+eh->e_phoff);
    uint64_t maxEnd=0;
    for(int i=0;i<eh->e_phnum;++i) if(ph[i].p_type==PT_LOAD)
        maxEnd=std::max(maxEnd,ph[i].p_vaddr+ph[i].p_memsz);
    uint64_t bytes=(maxEnd+0xFFF)&~0xFFFULL;
    uint8_t* buf=(uint8_t*)alloc_gpu(node,bytes,true,false);
    memset(buf,0,bytes);
    for(int i=0;i<eh->e_phnum;++i) if(ph[i].p_type==PT_LOAD)
        memcpy(buf+ph[i].p_vaddr,file.data()+ph[i].p_offset,ph[i].p_filesz);
    g_base=(uint64_t)buf;
    auto* sh=(Elf64_Shdr*)(file.data()+eh->e_shoff);
    for(int s=0;s<eh->e_shnum;++s){
        if(sh[s].sh_type!=SHT_DYNSYM && sh[s].sh_type!=SHT_SYMTAB) continue;
        auto* sym=(Elf64_Sym*)(file.data()+sh[s].sh_offset);
        int nsym=sh[s].sh_size/sizeof(Elf64_Sym);
        const char* str=(const char*)(file.data()+sh[sh[s].sh_link].sh_offset);
        for(int k=0;k<nsym;++k){
            const char* nm=str+sym[k].st_name; size_t L=strlen(nm);
            if(L>=3 && strcmp(nm+L-3,".kd")==0){
                std::string base(nm,L-3);
                bool seen=false; for(auto& ki:g_kernels) if(ki.name==base) seen=true;
                if(seen) continue;
                KInfo ki; ki.name=base; ki.kdVA=(uint64_t)buf+sym[k].st_value;
                memcpy(&ki.kd,(void*)ki.kdVA,sizeof(kernel_descriptor_t));
                g_kernels.push_back(ki);
            }
        }
    }
}
static const KInfo& kern(const char* name){
    for(auto& ki:g_kernels) if(ki.name==name) return ki;
    fprintf(stderr,"kernel %s not found\n",name); exit(1);
}

static const uint32_t DISPATCH_INIT = 0x00000021 | 0x8000; // CS_EN|USE_THREAD_DIMS|CS_W32

// Emit per-kernel register setup (PGM/RSRC/dims/USER_DATA), then a dispatch.
// DISPATCH_INIT has USE_THREAD_DIMS set, so dim_x is the TOTAL number of
// work-items (threads); the CP derives groups = ceil(threads / NUM_THREAD_X).
// kernarg pointer goes in s[0:1].
static void emit_dispatch(Emit& e, const KInfo& ki, uint64_t kernargVA, uint32_t numThreadsX){
    const kernel_descriptor_t& kd=ki.kd;
    const uint32_t dims[8]={0,0,0, BS,1,1, 0,0};
    e.set_sh_reg(mmCOMPUTE_START_X,dims,8);
    uint64_t entry=(ki.kdVA+kd.kernel_code_entry_byte_offset)>>8;
    const uint32_t pgm[2]={(uint32_t)entry,(uint32_t)(entry>>32)};
    e.set_sh_reg(mmCOMPUTE_PGM_LO,pgm,2);
    // The KD's rsrc2 does NOT carry static LDS size (the AQL CP fills it from the
    // dispatch packet's group_segment_size). Replicate: OR LDS_SIZE into rsrc2.
    // gfx11 LDS_SIZE = rsrc2[23:15], 9 bits, 64KB max -> 128-byte granule.
    uint32_t rsrc2=kd.compute_pgm_rsrc2;
    uint32_t ldsUnits=(kd.group_segment_fixed_size + 127) / 128;
    rsrc2 |= (ldsUnits << COMPUTE_PGM_RSRC2__LDS_SIZE__SHIFT) & COMPUTE_PGM_RSRC2__LDS_SIZE_MASK;
    const uint32_t rsrc[2]={kd.compute_pgm_rsrc1,rsrc2};
    e.set_sh_reg(mmCOMPUTE_PGM_RSRC1,rsrc,2);
    const uint32_t reslim[1]={0}; e.set_sh_reg(mmCOMPUTE_RESOURCE_LIMITS,reslim,1);
    const uint32_t tmpring[1]={0}; e.set_sh_reg(mmCOMPUTE_TMPRING_SIZE,tmpring,1);
    uint32_t udata[2]={(uint32_t)kernargVA,(uint32_t)(kernargVA>>32)};
    e.set_sh_reg(mmCOMPUTE_USER_DATA_0,udata,2);
    e.dispatch_direct(numThreadsX,1,1,DISPATCH_INIT);
}

int main(int argc,char** argv){
    int N=argc>1?atoi(argv[1]):2000;
    int M=argc>2?atoi(argv[2]):4096;
    const char* co=argc>3?argv[3]:"layer_gfx1100.co";

    CHECK(hsaKmtOpenKFD());
    HsaSystemProperties sys; CHECK(hsaKmtAcquireSystemProperties(&sys));
    int forced=getenv("KFD_NODE")?atoi(getenv("KFD_NODE")):-1;
    uint32_t node=(uint32_t)-1;
    for(uint32_t n=0;n<sys.NumNodes;++n){ HsaNodeProperties np;
        if(hsaKmtGetNodeProperties(n,&np)!=HSAKMT_STATUS_SUCCESS) continue;
        if(np.NumFComputeCores==0) continue;
        if(forced>=0){ if((int)n==forced) node=n; }
        else if((np.EngineId.ui32.Major==11||np.EngineId.ui32.Major==12) && node==(uint32_t)-1) node=n; }
    if(node==(uint32_t)-1){fprintf(stderr,"no gfx11/gfx12 node\n");return 1;}
    printf("using KFD node %u, M=%d N=%d\n",node,M,N);

    load_code_object(node,co);
    for(auto& ki:g_kernels){
        if(ki.kd.private_segment_fixed_size!=0 ||
           (ki.kd.kernel_code_properties&(KCP_PRIVATE_SEGMENT_BUFFER|KCP_FLAT_SCRATCH_INIT))){
            fprintf(stderr,"%s needs scratch (unsupported)\n",ki.name.c_str()); return 1; }
        printf("  %-8s group=%u kernarg=%u rsrc1=0x%08x rsrc2=0x%08x props=0x%04x\n",
               ki.name.c_str(),ki.kd.group_segment_fixed_size,ki.kd.kernarg_size,
               ki.kd.compute_pgm_rsrc1,ki.kd.compute_pgm_rsrc2,ki.kd.kernel_code_properties);
    }

    // DATA BUFFER MEMORY TYPE.
    //
    // DEFAULT (no env): COARSE-grain device VRAM, which faithfully replicates
    // hipMalloc -- the compute buffers (x,y,w) are PURE device VRAM (NonPaged +
    // CoarseGrain, NO HostAccess), exactly what the ROCm runtime hands back from
    // hipMalloc, and they are fully L2-cached. The CPU never touches them directly
    // (coarse-grain device memory is not CPU-coherent); host I/O goes through
    // host-accessible fine-grain staging buffers moved by on-GPU copy kernels --
    // the literal mechanism of hipMemcpy H2D/D2H. This is the REPRESENTATIVE path
    // and matches what HIP (hipMalloc) and Vulkan (DEVICE_LOCAL) use, so PM4/HIP/VK
    // are compared on equal memory.
    //
    // This is now the default ON PURPOSE: the old raw-UNCACHED default bypassed L2
    // and made large-M dependent chains up to ~20x slower, which repeatedly caused
    // confusing, non-comparable results. Opt out explicitly only for diagnostics:
    //   UNCACHED=1 -> raw L2-bypass device VRAM (slow; was the old default)
    //   CACHED=1   -> host-accessible fine-grain cacheable VRAM (MTYPE_CC)
    // DBG_ONLY (single-kernel isolation) needs CPU-writable buffers, so it forces
    // the host-accessible fine-grain path automatically.
    bool wantUncached = getenv("UNCACHED") != nullptr;
    bool wantCached   = getenv("CACHED")   != nullptr;
    bool coarse = !(wantUncached || wantCached || getenv("DBG_ONLY"));
    bool uncached = coarse ? false : (wantCached ? false : true);
    printf("buffers: %s\n", coarse ? "COARSE-grain device VRAM, NO HostAccess (== hipMalloc, L2-cached) [DEFAULT]; host I/O via GPU copy (== hipMemcpy)"
                                    : (uncached ? "UNCACHED (bypass L2) [diagnostic]" : "fine-grain CACHED (MTYPE_CC)"));

    // chain mode: elem = 3-kernel elementwise; rev = KLEN-kernel reverse-read
    // stress chain (ping-pong x<->y, every dispatch reads the just-written buffer
    // MIRRORED, forcing cross-WGP reads of recently written lines); default = rms.
    const char* chainEnv=getenv("CHAIN");
    bool elem = chainEnv && !strcmp(chainEnv,"elem");
    bool rev  = chainEnv && !strcmp(chainEnv,"rev");
    bool mix  = chainEnv && !strcmp(chainEnv,"mix");
    bool serial = chainEnv && !strcmp(chainEnv,"serial");
    int KLEN = getenv("KLEN") ? atoi(getenv("KLEN")) : 64;
    if(KLEN<2) KLEN=2; KLEN &= ~1;   // keep even so the chain ends writing x
    // LAYERS: repeat the 5-kernel mix block per iteration to model a full token
    // (e.g. ~10 kernels * 32 layers). LAYERS=64 -> 320 dispatches/token.
    int LAYERS = getenv("LAYERS") ? atoi(getenv("LAYERS")) : 1;
    if(LAYERS<1) LAYERS=1;
    if(rev) printf("chain  : rev (%d k_revadd kernels/iter, ping-pong x<->y, reverse reads)\n",KLEN);
    if(mix) printf("chain  : mix (%d dispatches/iter = 5 DISTINCT kernels x %d layers, rebind every dispatch, REVERSE-read blend -- cache-coherence stress)\n",5*LAYERS,LAYERS);
    if(serial) printf("chain  : serial (%d dispatches/iter = 5 DISTINCT kernels x %d layers, rebind every dispatch, FORWARD blend -- realistic decode write->read->modify->write)\n",5*LAYERS,LAYERS);

    size_t bytes=(size_t)M*4;
    size_t abytes=(bytes+0xFFF)&~0xFFFUL;
    // Compute buffers: coarse => pure device (hostAccess=false); else fine-grain.
    // COARSE_HA=1 keeps HostAccess=1 on the coarse buffers (diagnostic).
    bool devHA = coarse ? (getenv("COARSE_HA")!=nullptr) : true;
    float* x=(float*)alloc_gpu(node,abytes,false,uncached,coarse,devHA);
    float* y=(float*)alloc_gpu(node,abytes,false,uncached,coarse,devHA);
    float* w=(float*)alloc_gpu(node,abytes,false,uncached,coarse,devHA);
    // Staging buffers (coarse only): host-accessible fine-grain, used as the
    // hipMemcpy source/dest. xs/ws/ys hold the inputs, outs receives the result.
    float *xs=nullptr,*ws=nullptr,*outs=nullptr,*ys=nullptr;
    if(coarse){
        xs=(float*)alloc_gpu(node,abytes,false,true);
        ws=(float*)alloc_gpu(node,abytes,false,true);
        ys=(float*)alloc_gpu(node,abytes,false,true);
        outs=(float*)alloc_gpu(node,abytes,false,true);
        for(int i=0;i<M;++i){ xs[i]=(i%7)*0.01f+0.1f; ws[i]=1.0f+((i%5)*0.001f);
                              ys[i]=(i%9)*0.01f+0.05f; outs[i]=-1.0f; }
    } else {
        for(int i=0;i<M;++i){ x[i]=(i%7)*0.01f+0.1f; w[i]=1.0f+((i%5)*0.001f);
                              if(rev) y[i]=(i%9)*0.01f+0.05f; }
    }

    // kernarg segments (16-byte aligned). layout = [ptr0(8)][ptr1(8)][n(4)]
    uint32_t* ka=(uint32_t*)alloc_gpu(node,0x1000,false,false);
    auto setka=[&](int slot,uint64_t p0,uint64_t p1,int n){
        uint32_t* k=ka+slot*8; // 32 bytes per slot
        k[0]=(uint32_t)p0; k[1]=(uint32_t)(p0>>32);
        k[2]=(uint32_t)p1; k[3]=(uint32_t)(p1>>32); k[4]=(uint32_t)n; };
    setka(0,(uint64_t)x,(uint64_t)y,M);   // k_rms(x,y,n)
    setka(1,(uint64_t)y,(uint64_t)w,M);   // k_scale(y,w,n)
    setka(2,(uint64_t)x,(uint64_t)y,M);   // k_add(x,y,n) AND k_revadd(x,y): x mixes reverse(y)
    setka(3,(uint64_t)x,(uint64_t)w,M);   // elem: k_add/k_scale(x,w,n)
    setka(7,(uint64_t)y,(uint64_t)x,M);   // k_revadd(y,x): y mixes reverse(x)
    // coarse staging copies (== hipMemcpy): H2D x<-xs, w<-ws, y<-ys; D2H outs<-x
    if(coarse){
        setka(4,(uint64_t)x,(uint64_t)xs,M);    // k_copy(x, xs)   H2D
        setka(5,(uint64_t)w,(uint64_t)ws,M);    // k_copy(w, ws)   H2D
        setka(6,(uint64_t)outs,(uint64_t)x,M);  // k_copy(outs, x) D2H
        setka(8,(uint64_t)y,(uint64_t)ys,M);    // k_copy(y, ys)   H2D (rev only)
    }
    uint64_t kaRms=(uint64_t)(ka+0*8), kaScale=(uint64_t)(ka+1*8), kaAdd=(uint64_t)(ka+2*8);
    uint64_t kaXW=(uint64_t)(ka+3*8), kaYrevX=(uint64_t)(ka+7*8);
    uint64_t kaH2Dx=(uint64_t)(ka+4*8), kaH2Dw=(uint64_t)(ka+5*8), kaD2H=(uint64_t)(ka+6*8);
    uint64_t kaH2Dy=(uint64_t)(ka+8*8);

    uint32_t* sentinel=(uint32_t*)alloc_gpu(node,0x1000,false,true); sentinel[0]=0;

    uint64_t perIter = rev ? (uint64_t)KLEN : ((mix||serial) ? (uint64_t)5*LAYERS : 3);
    uint64_t needDwords=64 + (uint64_t)N*perIter*112 + 16 + (coarse?512:0);  // ~112 dw/dispatch incl fence (PWS pair is larger) (+coarse staging copies)
    uint64_t ringBytes=next_pow2(needDwords*4); if(ringBytes<0x10000) ringBytes=0x10000;
    void* ring=alloc_gpu(node,ringBytes,true,true);

    HsaQueueResource res; memset(&res,0,sizeof(res));
    CHECK(hsaKmtCreateQueue(node,HSA_QUEUE_COMPUTE,100,HSA_QUEUE_PRIORITY_NORMAL,
                            ring,ringBytes,nullptr,&res));

    // CUMASK=1: explicitly enable ALL compute units for this queue. A raw KFD
    // queue may default to a restricted CU set; ROCr/RADV program a full mask.
    // Test whether the large-M slowdown is the queue's effective CU count.
    if(getenv("CUMASK")){
        HsaNodeProperties np; memset(&np,0,sizeof(np)); hsaKmtGetNodeProperties(node,&np);
        uint32_t simdPerCU = np.NumSIMDPerCU ? np.NumSIMDPerCU : 2;
        uint32_t numCU = np.NumFComputeCores ? np.NumFComputeCores/simdPerCU : 96;
        if(!numCU) numCU=96;
        // CUMASK=N enables only the first N CUs (N<=0 or >numCU means all). Used to
        // test whether the GLK_INV cost scales with the number of active WGPs.
        int reqCU = atoi(getenv("CUMASK"));
        uint32_t enCU = (reqCU>0 && (uint32_t)reqCU<numCU) ? (uint32_t)reqCU : numCU;
        uint32_t words=(numCU+31)/32; std::vector<uint32_t> mask(words,0u);
        for(uint32_t c=0;c<enCU;++c) mask[c>>5] |= (1u<<(c&31));
        HSAKMT_STATUS s=hsaKmtSetQueueCUMask(res.QueueId,numCU,mask.data());
        printf("CUMASK: enabled %u of %u CUs (%u words) -> status %d\n",enCU,numCU,words,(int)s);
    }

    // Fence scope: 0 = RADV-like full L2 flush, 1 = AGENT-like L0/L1 invalidate.
    int gcrMode = getenv("GCR_MODE") ? atoi(getenv("GCR_MODE")) : 1;
    uint32_t GCR = gcrMode ? GCR_AGENT_LIKE : GCR_RADV_LIKE;
    if(getenv("GCR0")) GCR = 0;   // diagnostic: ACQUIRE_MEM packet+wait with NO cache op
    if(getenv("GCR_RAW")) GCR = (uint32_t)strtoul(getenv("GCR_RAW"),nullptr,0);  // diagnostic override
    printf("fence GCR=0x%x (%s)\n", GCR, gcrMode ? "AGENT-like (no L2 flush)" : "RADV-like (full L2)");

    // IB=1: emit the entire stream into a separate executable indirect buffer and
    // launch it from the ring with a SINGLE INDIRECT_BUFFER packet (one CP jump for
    // the whole chain). This isolates the IB-jump cost and mirrors how the HIP
    // runtime would replay a captured graph as one PM4 IB via a vendor packet.
    bool useIB = getenv("IB") != nullptr;
    void* ibBuf = useIB ? alloc_gpu(node,ringBytes,true,true) : nullptr;
    Emit e((uint32_t*)(useIB ? ibBuf : ring));

    // SEMASK=1: enable ALL CUs on ALL shader engines for this queue. gfx11
    // (Navi31) has 6 SEs; COMPUTE_STATIC_THREAD_MGMT_SE0..SE5 are the per-SE CU
    // enable masks (SA0_CU_EN[15:0] | SA1_CU_EN[31:16]). If the queue defaults to
    // only SE0 enabled, every workgroup serializes on one of six engines -- the
    // suspected cause of the linear-in-WG / ~6x large-M slowdown. Offsets follow
    // the same gfx_7_2 SH-register scheme the working code already uses
    // (START_X=0x2e04); SE4/SE5 are not in the CIK header so use literals
    // matching the gfx11 relative positions (+0x27/+0x28 from START_X).
    if(getenv("SEMASK")){
        const uint32_t all=0xFFFFFFFFu;
        const uint32_t se[6]={ mmCOMPUTE_STATIC_THREAD_MGMT_SE0, mmCOMPUTE_STATIC_THREAD_MGMT_SE1,
                               mmCOMPUTE_STATIC_THREAD_MGMT_SE2, mmCOMPUTE_STATIC_THREAD_MGMT_SE3,
                               0x2e2bu /*SE4*/, 0x2e2cu /*SE5*/ };
        for(int i=0;i<6;i++) e.set_sh_reg(se[i],&all,1);
        printf("SEMASK: wrote COMPUTE_STATIC_THREAD_MGMT_SE0..5 = 0xFFFFFFFF (all CUs, 6 SEs)\n");
    }

    const KInfo& kRms=kern("k_rms"); const KInfo& kScale=kern("k_scale"); const KInfo& kAdd=kern("k_add");
    uint32_t thrFull=M;        // total work-items for elementwise kernels
    uint32_t thrRms=BS;        // k_rms is one workgroup of BS threads

    // coarse-grain (== hipMalloc) H2D: GPU-copy the staged inputs into pure device
    // VRAM before the chain runs, then fence so the chain reads the fresh data.
    // This is the on-GPU equivalent of hipMemcpy(HostToDevice).
    if(coarse){
        const KInfo& kCopy=kern("k_copy");
        emit_dispatch(e,kCopy,kaH2Dx,thrFull);
        emit_dispatch(e,kCopy,kaH2Dw,thrFull);
        if(rev) emit_dispatch(e,kCopy,kaH2Dy,thrFull);  // y<-ys for the rev ping-pong chain
        // AGENT-scope only: keep the copied x/w resident (dirty) in L2 so the chain
        // reads them. A GL2 writeback+invalidate here would DISCARD the dirty lines
        // (the device coherence point is L2), losing the data.
        e.partial_flush(); e.acquire_mem(GCR_AGENT_LIKE);
    }

    const char* only=getenv("DBG_ONLY");
    if(only){
        // Isolation: run ONE kernel once. For scale, prefill y=1 so touched
        // indices become w[i] (else stay 1).
        for(int i=0;i<M;++i){ y[i]=1.0f; }
        if(!strcmp(only,"scale")){ emit_dispatch(e,kScale,kaScale,thrFull); }
        else if(!strcmp(only,"rms")){ emit_dispatch(e,kRms,kaRms,thrRms); }
        else if(!strcmp(only,"add")){ emit_dispatch(e,kAdd,kaAdd,thrFull); }
        e.partial_flush(); e.acquire_mem(GCR);
    } else {
    // FENCE: 0=none (race, throughput floor), 1=partial-flush only, 2=pf+acquire
    int fmode = getenv("FENCE") ? atoi(getenv("FENCE")) : 2;
    // PWS overlapped fence is the DEFAULT (correct + flat + faster). PWS=0 falls
    // back to the blocking partial_flush+acquire_mem path. The wave-drain
    // (CS_PARTIAL_FLUSH) is REQUIRED before the PWS pair for RAW correctness: the
    // RELEASE_MEM EOP alone does not order the consumer against the producer's
    // outstanding stores on a raw MEC queue (verified: dropping it gives
    // non-deterministic ~1%-wrong checksums). With the drain, PWS is bit-exact on
    // both the serial and reverse-read mix chains and no longer exposes the
    // per-CU GLK invalidate cost (it overlaps the flush, like RADV).
    int usePWS = getenv("PWS") ? atoi(getenv("PWS")) : 1;
    auto fence=[&](){
        if(usePWS){ e.partial_flush(); e.release_mem_pws(GCR); e.acquire_pws(); return; }
        if(fmode>=1) e.partial_flush(); if(fmode>=2) e.acquire_mem(GCR); };
    const KInfo& kRev=kern("k_revadd");
    const KInfo& kCopy=kern("k_copy");
    const KInfo& kBlend=kern("k_blend");
    for(int it=0;it<N;++it){
        if(rev){
            // KLEN-kernel reverse-read chain, ping-pong x<->y. Each dispatch reads
            // the buffer the previous one just wrote, MIRRORED (n-1-i), so threads
            // read lines written by other workgroups -> stresses L0/L1 invalidate.
            for(int k=0;k<KLEN;++k){
                if(k & 1) emit_dispatch(e,kRev,kaAdd,thrFull);    // x = 0.5x + 0.5*rev(y)
                else      emit_dispatch(e,kRev,kaYrevX,thrFull);  // y = 0.5y + 0.5*rev(x)
                fence();
            }
        } else if(mix||serial){
            // Decode-like chain: 5 DISTINCT SIMPLE kernels, a different pipeline
            // bound every dispatch (the realistic case -- real decode never repeats
            // one shader; LAYERS scales this to ~300 dispatches/token). All compile
            // to near-identical ISA across hipcc and RADV, so the comparison
            // isolates framework DISPATCH overhead, not kernel codegen. Contractive
            // (x updated only by the 0.5-averaging blend) so it stays bounded;
            // bit-exact across PM4/HIP/Vulkan. The blend kernel is the only
            // difference between the two chains:
            //   mix    -> k_revadd (reverse read src[n-1-i]: cache-coherence stress)
            //   serial -> k_blend  (forward read src[i]: realistic decode pattern)
            const KInfo& kMid = serial ? kBlend : kRev;
            for(int l=0;l<LAYERS;++l){
                emit_dispatch(e,kCopy, kaYrevX,thrFull); fence();  // y = x
                emit_dispatch(e,kScale,kaScale,thrFull); fence();  // y *= w
                emit_dispatch(e,kMid,  kaAdd,  thrFull); fence();  // x = 0.5x + 0.5*(rev(y) | y)
                emit_dispatch(e,kAdd,  kaScale,thrFull); fence();  // y += w
                emit_dispatch(e,kMid,  kaAdd,  thrFull); fence();  // x = 0.5x + 0.5*(rev(y) | y)
            }
        } else if(elem){
            // all-parallel elementwise dependent chain on x (full grid each)
            emit_dispatch(e,kAdd,kaXW,thrFull);   fence();  // x += w
            emit_dispatch(e,kScale,kaXW,thrFull); fence();  // x *= w
            emit_dispatch(e,kAdd,kaXW,thrFull);   fence();  // x += w
        } else {
            emit_dispatch(e,kRms,kaRms,thrRms);      fence();
            emit_dispatch(e,kScale,kaScale,thrFull); fence();
            emit_dispatch(e,kAdd,kaAdd,thrFull);     fence();
        }
    }
    }
    // coarse-grain (== hipMalloc) D2H: GPU-copy device x (resident in L2) into the
    // host-accessible UNCACHED staging buffer outs (== hipMemcpy(DeviceToHost)).
    // AGENT-scope fence only (invalidate L0/L1 so the copy reads fresh x from L2);
    // outs is uncached so its write bypasses L2 straight to HBM for a coherent CPU
    // readback -- no L2 writeback/invalidate of the coarse buffer is needed.
    if(coarse){
        const KInfo& kCopy=kern("k_copy");
        e.partial_flush(); e.acquire_mem(GCR_AGENT_LIKE);
        emit_dispatch(e,kCopy,kaD2H,thrFull);
    }
    e.partial_flush();
    e.acquire_mem(GCR_RADV_LIKE);  // final full-L2 writeback so cached results land in HBM for CPU verify
    e.write_data((uint64_t)sentinel,0xC0FFEE);
    uint32_t total=e.idx;
    if((uint64_t)total*4 > ringBytes){ fprintf(stderr,"ring overflow %u dw\n",total); return 1; }
    uint32_t ringTotal=total;
    if(useIB){
        uint64_t ibVA=(uint64_t)ibBuf;
        uint32_t* r=(uint32_t*)ring;
        r[0]=(3u<<30)|((4u-2u)<<16)|(0x3Fu<<8);   // IT_INDIRECT_BUFFER (MEC), 4 dwords
        r[1]=(uint32_t)(ibVA & 0xFFFFFFFCu);       // IB_BASE_LO (4-aligned)
        r[2]=(uint32_t)((ibVA>>32)&0xFFFFu);       // IB_BASE_HI
        r[3]=(total & 0xFFFFFu) | (1u<<23);        // IB_SIZE (dwords) | IB_VALID
        ringTotal=4;
    }
    printf("stream=%u dwords (IB=%d) ring=%lluKB\n",total,(int)useIB,(unsigned long long)(ringBytes>>10));

    asm volatile("":::"memory");
    double t0=now_s();
    *res.Queue_write_ptr_aql=ringTotal;
    asm volatile("":::"memory");
    *res.Queue_DoorBell_aql=ringTotal;
    const double timeout=120.0;
    while(sentinel[0]!=0xC0FFEE){
        if(now_s()-t0>timeout){ fprintf(stderr,"TIMEOUT rptr=%u/%u\n",*res.Queue_read_ptr,total);
            hsaKmtDestroyQueue(res.QueueId); return 2; } }
    double t1=now_s();
    double us=(t1-t0)*1e6;

    if(getenv("DBG") && !coarse){  // coarse x/y are device-only (no CPU access)
        printf("DBG y[0]=%.5f y[1]=%.5f y[255]=%.5f y[256]=%.5f y[4095]=%.5f\n",
               y[0],y[1],y[255],y[256],y[M-1]);
        printf("DBG x[0]=%.5f x[1]=%.5f x[255]=%.5f x[256]=%.5f x[4095]=%.5f\n",
               x[0],x[1],x[255],x[256],x[M-1]);
    }
    if(getenv("DBG") && coarse){
        printf("DBG coarse xs[0]=%.5f ws[0]=%.5f outs[0]=%.5f outs[1]=%.5f\n",
               xs[0],ws[0],outs[0],outs[1]);
        if(devHA) printf("DBG coarse(HA) x[0]=%.5f x[1]=%.5f w[0]=%.5f\n",x[0],x[1],w[0]);
    }
    const float* rd = coarse ? outs : x;  // coarse: read the host-staged D2H copy
    double sum=0; for(int i=0;i<M;++i) sum+=rd[i];
    printf("result : checksum(x)=%.6f x[0]=%.6f x[%d]=%.6f\n",sum,rd[0],M-1,rd[M-1]);
    printf("         total %.2f us  PER-ITER(%d kern)=%.3f us  PER-DISPATCH=%.3f us\n",
           us, (int)perIter, us/N, us/(double)(N*perIter));
    hsaKmtDestroyQueue(res.QueueId); hsaKmtCloseKFD(); return 0;
}

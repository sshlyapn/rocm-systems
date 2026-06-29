// pm4_real.cpp
//
// Like pm4_gap.cpp, but dispatches a REAL hipcc-compiled kernel through a raw
// KFD PM4 compute queue, replicating ROCr's AQL-to-hardware translation:
//   - load the ELF code object PT_LOAD segments into one GPU exec buffer
//     (preserving relative vaddrs so KD->entry offset stays valid)
//   - read the 64-byte kernel_descriptor_t (KD)
//   - COMPUTE_PGM_LO/HI = (kdVA + kernel_code_entry_byte_offset) >> 8
//   - COMPUTE_PGM_RSRC1/RSRC2 copied VERBATIM from the KD (exactly what the CP
//     loads), so we never hand-decode them
//   - place USER_DATA SGPRs in kernel_code_properties enable-bit order. For the
//     test kernel only ENABLE_SGPR_KERNARG_SEGMENT_PTR is set, so kernarg goes
//     in s[0:1] (USER_DATA_0/1).
//   - kernarg segment holds the explicit args (here: one int* counter)
//
// Validates: the chain serializes (counter == K) AND measures per-dispatch
// period, to confirm a real ROCm kernel can be PM4-dispatched at the RADV-like
// gap before transplanting this logic into the HIP/CLR runtime.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <elf.h>
#include <vector>

#include "hsakmt/hsakmt.h"

#include "kfd_pm4_opcodes.h"
#include "pm4_pkt_struct_common.h"
#include "pm4_pkt_struct_ci.h"
#include "pm4_pkt_struct_ai.h"
#include "pm4_pkt_struct_nv.h"
#include "asic_reg/gfx_7_2_d.h"
#include "asic_reg/gfx_7_2_sh_mask.h"
#include "asic_reg/gfx_7_2_enum.h"

#define CHECK(expr) do { \
    HSAKMT_STATUS _s = (expr); \
    if (_s != HSAKMT_STATUS_SUCCESS) { \
        fprintf(stderr, "FAIL %s = %d (line %d)\n", #expr, _s, __LINE__); \
        exit(1); \
    } \
} while (0)

enum {
    GCR_GLI_INV = 1u << 0, GCR_GLM_WB = 1u << 4, GCR_GLM_INV = 1u << 5,
    GCR_GLK_INV = 1u << 7, GCR_GLV_INV = 1u << 8, GCR_GL1_INV = 1u << 9,
    GCR_GL2_INV = 1u << 13, GCR_GL2_WB = 1u << 14,
};
static const uint32_t GCR_RADV_LIKE =
    GCR_GL2_WB | GCR_GL2_INV | GCR_GLM_WB | GCR_GLM_INV |
    GCR_GL1_INV | GCR_GLV_INV | GCR_GLK_INV | GCR_GLI_INV;

// 64-byte AMDHSA kernel descriptor (COV5). Offsets per LLVM amdhsa.
struct kernel_descriptor_t {
    uint32_t group_segment_fixed_size;     // 0
    uint32_t private_segment_fixed_size;   // 4
    uint32_t kernarg_size;                 // 8
    uint8_t  reserved0[4];                 // 12
    int64_t  kernel_code_entry_byte_offset;// 16
    uint8_t  reserved1[20];                // 24
    uint32_t compute_pgm_rsrc3;            // 44
    uint32_t compute_pgm_rsrc1;            // 48
    uint32_t compute_pgm_rsrc2;            // 52
    uint16_t kernel_code_properties;       // 56
    uint16_t kernarg_preload;              // 58
    uint8_t  reserved2[4];                 // 60
};
enum {
    KCP_PRIVATE_SEGMENT_BUFFER = 1u << 0,
    KCP_DISPATCH_PTR           = 1u << 1,
    KCP_QUEUE_PTR              = 1u << 2,
    KCP_KERNARG_SEGMENT_PTR    = 1u << 3,
    KCP_DISPATCH_ID            = 1u << 4,
    KCP_FLAT_SCRATCH_INIT      = 1u << 5,
    KCP_PRIVATE_SEGMENT_SIZE   = 1u << 6,
    KCP_WAVEFRONT_SIZE32       = 1u << 10,
};

static double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void set_hdr(PM4_TYPE_3_HEADER &h, it_opcode_type op, unsigned dwords) {
    h.u32All = 0; h.type = PM4_TYPE_3; h.shaderType = 1; h.opcode = op;
    h.count = dwords - 2;
}

struct Emit {
    uint32_t *base; uint32_t idx;
    explicit Emit(uint32_t *b) : base(b), idx(0) {}
    void set_sh_reg(unsigned baseReg, const uint32_t *vals, unsigned n) {
        unsigned dwords = 2 + n;
        PM4_TYPE_3_HEADER h; set_hdr(h, IT_SET_SH_REG, dwords);
        base[idx + 0] = h.u32All;
        base[idx + 1] = baseReg - PERSISTENT_SPACE_START;
        memcpy(&base[idx + 2], vals, n * sizeof(uint32_t));
        idx += dwords;
    }
    void dispatch_direct(uint32_t x, uint32_t y, uint32_t z, uint32_t init) {
        PM4DISPATCH_DIRECT p; memset(&p, 0, sizeof(p));
        set_hdr(p.header, IT_DISPATCH_DIRECT, sizeof(p) / 4);
        p.dim_x = x; p.dim_y = y; p.dim_z = z; p.dispatch_initiator = init;
        memcpy(&base[idx], &p, sizeof(p)); idx += sizeof(p) / 4;
    }
    void partial_flush() {
        PM4EVENT_WRITE p; memset(&p, 0, sizeof(p));
        set_hdr(p.header, IT_EVENT_WRITE, 2);
        p.bitfields2.event_index = event_index_event_write_CS_VS_PS_PARTIAL_FLUSH_4;
        p.bitfields2.event_type  = CS_PARTIAL_FLUSH;
        base[idx + 0] = p.ordinal1; base[idx + 1] = p.ordinal2; idx += 2;
    }
    void acquire_mem(uint32_t gcr) {
        PM4ACQUIRE_MEM_NV p; memset(&p, 0, sizeof(p));
        set_hdr(p.header, IT_ACQUIRE_MEM, sizeof(p) / 4);
        p.coher_size = 0xFFFFFFFF; p.bitfields3.coher_size_hi = 0;
        p.coher_base_lo = 0; p.bitfields4.coher_base_hi = 0;
        p.bitfields5.poll_interval = 4; p.bitfields6.gcr_cntl = gcr;
        memcpy(&base[idx], &p, sizeof(p)); idx += sizeof(p) / 4;
    }
    void write_data(uint64_t dst, uint32_t value) {
        PM4WRITE_DATA_CI p; memset(&p, 0, sizeof(p));
        unsigned dwords = (offsetof(PM4WRITE_DATA_CI, data) / 4) + 1;
        set_hdr(p.header, IT_WRITE_DATA, dwords);
        p.bitfields2.dst_sel      = dst_sel_mec_write_data_MEMORY_5;
        p.bitfields2.addr_incr    = addr_incr_mec_write_data_INCREMENT_ADDR_0;
        p.bitfields2.wr_confirm   = wr_confirm_mec_write_data_WAIT_FOR_CONFIRMATION_1;
        p.bitfields2.atc          = atc_write_data_NOT_USE_ATC_0;
        p.bitfields2.cache_policy = cache_policy_mec_write_data_BYPASS_2;
        p.dst_addr_lo    = (uint32_t)dst;
        p.dst_address_hi = (uint32_t)(dst >> 32);
        memcpy(&base[idx], &p, dwords * 4);
        base[idx + dwords - 1] = value; idx += dwords;
    }
};

static void *alloc_gpu(uint32_t node, uint64_t size, bool exec, bool uncached) {
    HsaMemFlags f; f.Value = 0;
    f.ui32.PageSize = HSA_PAGE_SIZE_4KB; f.ui32.HostAccess = 1;
    f.ui32.NonPaged = 0; f.ui32.CoarseGrain = 0; f.ui32.NoNUMABind = 1;
    f.ui32.Uncached = uncached ? 1 : 0; f.ui32.ExecuteAccess = exec ? 1 : 0;
    void *p = nullptr;
    CHECK(hsaKmtAllocMemory(node, size, f, &p));
    CHECK(hsaKmtMapMemoryToGPU(p, size, nullptr));
    return p;
}
static uint64_t next_pow2(uint64_t v){ uint64_t p=1; while(p<v) p<<=1; return p; }

// Load ELF code object PT_LOAD segments into one GPU exec buffer at their
// virtual addresses; return base VA and locate the "*.kd" symbol vaddr.
struct Loaded { uint64_t baseVA; uint64_t kdVA; const kernel_descriptor_t *kd; };
static Loaded load_code_object(uint32_t node, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(fp, 0, SEEK_END); long flen = ftell(fp); fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> file(flen);
    if (fread(file.data(), 1, flen, fp) != (size_t)flen) { fprintf(stderr,"short read\n"); exit(1);} 
    fclose(fp);

    auto *eh = reinterpret_cast<Elf64_Ehdr*>(file.data());
    if (memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0) { fprintf(stderr,"not ELF\n"); exit(1);} 

    // span of PT_LOAD by vaddr
    uint64_t maxEnd = 0;
    auto *ph = reinterpret_cast<Elf64_Phdr*>(file.data() + eh->e_phoff);
    for (int i = 0; i < eh->e_phnum; ++i)
        if (ph[i].p_type == PT_LOAD)
            maxEnd = std::max(maxEnd, ph[i].p_vaddr + ph[i].p_memsz);

    uint64_t bufBytes = (maxEnd + 0xFFF) & ~0xFFFULL;
    uint8_t *buf = (uint8_t *)alloc_gpu(node, bufBytes, /*exec*/true, /*uncached*/false);
    memset(buf, 0, bufBytes);
    for (int i = 0; i < eh->e_phnum; ++i)
        if (ph[i].p_type == PT_LOAD)
            memcpy(buf + ph[i].p_vaddr, file.data() + ph[i].p_offset, ph[i].p_filesz);

    // find "*.kd" symbol in .dynsym or .symtab
    auto *sh = reinterpret_cast<Elf64_Shdr*>(file.data() + eh->e_shoff);
    const char *shstr = (const char *)(file.data() + sh[eh->e_shstrndx].sh_offset);
    uint64_t kdVaddr = (uint64_t)-1;
    for (int s = 0; s < eh->e_shnum; ++s) {
        if (sh[s].sh_type != SHT_DYNSYM && sh[s].sh_type != SHT_SYMTAB) continue;
        auto *sym = reinterpret_cast<Elf64_Sym*>(file.data() + sh[s].sh_offset);
        int nsym = sh[s].sh_size / sizeof(Elf64_Sym);
        const char *str = (const char *)(file.data() + sh[sh[s].sh_link].sh_offset);
        for (int k = 0; k < nsym; ++k) {
            const char *nm = str + sym[k].st_name;
            size_t L = strlen(nm);
            if (L >= 3 && strcmp(nm + L - 3, ".kd") == 0) {
                kdVaddr = sym[k].st_value;
                printf("found KD symbol '%s' vaddr=0x%lx\n", nm, (unsigned long)kdVaddr);
            }
        }
    }
    (void)shstr;
    if (kdVaddr == (uint64_t)-1) { fprintf(stderr, "no .kd symbol found\n"); exit(1); }

    Loaded ld;
    ld.baseVA = (uint64_t)buf;
    ld.kdVA   = (uint64_t)buf + kdVaddr;
    ld.kd     = reinterpret_cast<const kernel_descriptor_t*>(buf + kdVaddr);
    return ld;
}

int main(int argc, char **argv) {
    int K = argc > 1 ? atoi(argv[1]) : 2000;
    int fence = argc > 2 ? atoi(argv[2]) : 2;   // 0 none,1 pf,2 pf+acquire
    const char *coPath = argc > 3 ? argv[3] : "k_gfx1100.co";
    const char *fenceName[] = {"none", "partial-flush", "pf+ACQUIRE_MEM(RADV-like)"};

    CHECK(hsaKmtOpenKFD());
    HsaSystemProperties sys; CHECK(hsaKmtAcquireSystemProperties(&sys));
    int forced = getenv("KFD_NODE") ? atoi(getenv("KFD_NODE")) : -1;
    uint32_t node = (uint32_t)-1;
    for (uint32_t n = 0; n < sys.NumNodes; ++n) {
        HsaNodeProperties np;
        if (hsaKmtGetNodeProperties(n, &np) != HSAKMT_STATUS_SUCCESS) continue;
        if (np.NumFComputeCores == 0) continue;
        if (forced >= 0) { if ((int)n == forced) node = n; }
        else if (np.EngineId.ui32.Major == 11 && node == (uint32_t)-1) node = n;
    }
    if (node == (uint32_t)-1) { fprintf(stderr, "no gfx11 node\n"); return 1; }
    printf("using KFD node %u\n", node);

    Loaded ld = load_code_object(node, coPath);
    const kernel_descriptor_t *kd = ld.kd;
    printf("KD: group=%u private=%u kernarg=%u entry_off=0x%lx rsrc1=0x%08x rsrc2=0x%08x rsrc3=0x%08x props=0x%04x\n",
           kd->group_segment_fixed_size, kd->private_segment_fixed_size, kd->kernarg_size,
           (unsigned long)kd->kernel_code_entry_byte_offset, kd->compute_pgm_rsrc1,
           kd->compute_pgm_rsrc2, kd->compute_pgm_rsrc3, kd->kernel_code_properties);

    if (kd->private_segment_fixed_size != 0 ||
        (kd->kernel_code_properties & (KCP_PRIVATE_SEGMENT_BUFFER | KCP_FLAT_SCRATCH_INIT))) {
        fprintf(stderr, "kernel needs scratch; not handled in this minimal test\n");
        return 1;
    }

    // data counter (uncached for coherent CPU verify) + kernarg + sentinel
    uint32_t *data     = (uint32_t *)alloc_gpu(node, 0x1000, false, true);
    uint64_t *kernarg  = (uint64_t *)alloc_gpu(node, 0x1000, false, false);
    uint32_t *sentinel = (uint32_t *)alloc_gpu(node, 0x1000, false, true);
    data[0] = 0; sentinel[0] = 0;
    kernarg[0] = (uint64_t)data;   // explicit arg 0: int* counter

    unsigned perIter = (fence == 0) ? 5 : (fence == 1) ? 7 : 15;
    uint64_t needDwords = 64 + (uint64_t)K * perIter + 16;
    uint64_t ringBytes = next_pow2(needDwords * 4);
    if (ringBytes < 0x10000) ringBytes = 0x10000;
    void *ring = alloc_gpu(node, ringBytes, true, true);

    HsaQueueResource res; memset(&res, 0, sizeof(res));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL,
                            ring, ringBytes, nullptr, &res));

    Emit e((uint32_t *)ring);

    const uint32_t dims[8] = { 0,0,0, 1,1,1, 0,0 };
    e.set_sh_reg(mmCOMPUTE_START_X, dims, 8);

    const uint64_t entryVA = ld.kdVA + kd->kernel_code_entry_byte_offset;
    const uint64_t shifted = entryVA >> 8;
    const uint32_t pgm[2] = { (uint32_t)shifted, (uint32_t)(shifted >> 32) };
    e.set_sh_reg(mmCOMPUTE_PGM_LO, pgm, 2);

    // copy rsrc1/rsrc2 verbatim from the KD (exactly what the CP loads)
    const uint32_t rsrc[2] = { kd->compute_pgm_rsrc1, kd->compute_pgm_rsrc2 };
    e.set_sh_reg(mmCOMPUTE_PGM_RSRC1, rsrc, 2);

    const uint32_t reslim[1] = { 0 };  e.set_sh_reg(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1);
    const uint32_t tmpring[1] = { 0 }; e.set_sh_reg(mmCOMPUTE_TMPRING_SIZE, tmpring, 1);

    // USER_DATA in enable-bit order. Only KERNARG_SEGMENT_PTR set -> s[0:1].
    uint32_t udata[16] = {0};
    unsigned u = 0;
    const uint16_t p = kd->kernel_code_properties;
    if (p & KCP_PRIVATE_SEGMENT_BUFFER) u += 4;            // not reached (guarded)
    if (p & KCP_DISPATCH_PTR)           u += 2;            // (not set for test kernel)
    if (p & KCP_QUEUE_PTR)              u += 2;
    if (p & KCP_KERNARG_SEGMENT_PTR) {
        udata[u + 0] = (uint32_t)(uint64_t)kernarg;
        udata[u + 1] = (uint32_t)((uint64_t)kernarg >> 32);
        u += 2;
    }
    e.set_sh_reg(mmCOMPUTE_USER_DATA_0, udata, 16);

    const uint32_t DISPATCH_INIT = 0x00000021 | 0x8000;  // CS_EN|USE_THREAD_DIMS|CS_W32_EN

    for (int k = 0; k < K; ++k) {
        e.dispatch_direct(1, 1, 1, DISPATCH_INIT);
        if (fence >= 1) e.partial_flush();
        if (fence >= 2) e.acquire_mem(GCR_RADV_LIKE);
    }
    e.partial_flush();
    e.write_data((uint64_t)sentinel, 0xC0FFEE);

    uint32_t total = e.idx;
    printf("config : K=%d fence=%s ring=%lluKB stream=%u dwords\n",
           K, fenceName[fence], (unsigned long long)(ringBytes >> 10), total);

    asm volatile("" ::: "memory");
    double t0 = now_s();
    *res.Queue_write_ptr_aql = total;
    asm volatile("" ::: "memory");
    *res.Queue_DoorBell_aql = total;

    const double timeout = 20.0;
    while (sentinel[0] != 0xC0FFEE) {
        if (now_s() - t0 > timeout) {
            fprintf(stderr, "TIMEOUT (sentinel=%x rptr=%u/%u data=%u)\n",
                    sentinel[0], *res.Queue_read_ptr, total, data[0]);
            hsaKmtDestroyQueue(res.QueueId); return 2;
        }
    }
    double t1 = now_s();
    double us = (t1 - t0) * 1e6;
    printf("result : data=%u (expect %d)  %s\n", data[0], K,
           data[0] == (uint32_t)K ? "OK chain serialized" : "MISMATCH");
    printf("         total %.2f us   PER-DISPATCH PERIOD = %.3f us\n", us, us / K);

    hsaKmtDestroyQueue(res.QueueId);
    hsaKmtCloseKFD();
    return 0;
}

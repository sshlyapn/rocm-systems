// pm4_gap.cpp
//
// Standalone inter-kernel gap microbenchmark that submits compute dispatches
// to a RDNA3 (gfx1100 / W7900) GPU through a RAW KFD PM4 compute queue
// (HSA_QUEUE_COMPUTE), bypassing the HSA/AQL soft-queue that HIP/ROCr use.
//
// Purpose: test whether driving the SAME silicon through the PM4 front-end
// (DISPATCH_DIRECT + CS_PARTIAL_FLUSH + ACQUIRE_MEM, exactly like RADV) gives
// the low per-dispatch period seen on Vulkan, versus the ~3 us-higher period
// of HIP's per-packet AQL AGENT acquire/release path.
//
// It builds ONE linear PM4 stream in the queue ring:
//   [one-time COMPUTE_* register setup]
//   K x [ DISPATCH_DIRECT
//         (fence mode) EVENT_WRITE(CS_PARTIAL_FLUSH)   // drain writer waves
//         (fence mode) ACQUIRE_MEM(gcr)                // flush/inval caches ]
//   EVENT_WRITE(CS_PARTIAL_FLUSH)                       // drain last dispatch
//   WRITE_DATA(sentinel)                                // CPU-pollable done flag
// then rings the doorbell ONCE and busy-polls the sentinel. period = wall/K.
//
// Build: see build.sh. Run as a user with /dev/kfd access.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <string>
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

// ---- ACQUIRE_MEM GCR_CNTL bit layout (gfx10/11), from RADV S_586_* ----
//  bit0:GLI_INV bit4:GLM_WB bit5:GLM_INV bit7:GLK_INV bit8:GLV_INV
//  bit9:GL1_INV bit13:GL2_INV bit14:GL2_WB
enum {
    GCR_GLI_INV = 1u << 0,
    GCR_GLM_WB  = 1u << 4,
    GCR_GLM_INV = 1u << 5,
    GCR_GLK_INV = 1u << 7,
    GCR_GLV_INV = 1u << 8,
    GCR_GL1_INV = 1u << 9,
    GCR_GL2_INV = 1u << 13,
    GCR_GL2_WB  = 1u << 14,
};
// Match RADV's compute SHADER_WRITE->SHADER_READ barrier exactly:
// full L2 writeback+invalidate + GLM + L1/L0(vector,scalar) + icache invalidate.
static const uint32_t GCR_RADV_LIKE =
    GCR_GL2_WB | GCR_GL2_INV | GCR_GLM_WB | GCR_GLM_INV |
    GCR_GL1_INV | GCR_GLV_INV | GCR_GLK_INV | GCR_GLI_INV;

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void set_hdr(PM4_TYPE_3_HEADER &h, it_opcode_type op, unsigned dwords) {
    h.u32All = 0;
    h.type = PM4_TYPE_3;
    h.shaderType = 1;          // compute
    h.opcode = op;
    h.count = dwords - 2;      // body dwords minus 1 (header excluded)
}

// Cursor-based PM4 emitters writing into the queue ring.
struct Emit {
    uint32_t *base;
    uint32_t  idx;   // dword write index
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

    // CS_PARTIAL_FLUSH: drain compute waves. Body is header + 1 dword.
    void partial_flush() {
        PM4EVENT_WRITE p; memset(&p, 0, sizeof(p));
        set_hdr(p.header, IT_EVENT_WRITE, 2);
        p.bitfields2.event_index = event_index_event_write_CS_VS_PS_PARTIAL_FLUSH_4;
        p.bitfields2.event_type  = CS_PARTIAL_FLUSH;
        base[idx + 0] = p.ordinal1;
        base[idx + 1] = p.ordinal2;
        idx += 2;
    }

    void acquire_mem(uint32_t gcr) {
        PM4ACQUIRE_MEM_NV p; memset(&p, 0, sizeof(p));
        set_hdr(p.header, IT_ACQUIRE_MEM, sizeof(p) / 4);
        p.coher_size = 0xFFFFFFFF;
        p.bitfields3.coher_size_hi = 0;
        p.coher_base_lo = 0;
        p.bitfields4.coher_base_hi = 0;
        p.bitfields5.poll_interval = 4;
        p.bitfields6.gcr_cntl = gcr;
        memcpy(&base[idx], &p, sizeof(p)); idx += sizeof(p) / 4;
    }

    // WRITE_DATA to memory with write-confirm; CPU can poll the value.
    void write_data(uint64_t dst, uint32_t value) {
        PM4WRITE_DATA_CI p; memset(&p, 0, sizeof(p));
        unsigned dwords = (offsetof(PM4WRITE_DATA_CI, data) / 4) + 1;
        set_hdr(p.header, IT_WRITE_DATA, dwords);
        p.bitfields2.dst_sel      = dst_sel_mec_write_data_MEMORY_5;
        p.bitfields2.addr_incr    = addr_incr_mec_write_data_INCREMENT_ADDR_0;
        p.bitfields2.wr_confirm   = wr_confirm_mec_write_data_WAIT_FOR_CONFIRMATION_1;
        p.bitfields2.atc          = atc_write_data_NOT_USE_ATC_0;   // dGPU
        p.bitfields2.cache_policy = cache_policy_mec_write_data_BYPASS_2;
        p.dst_addr_lo    = (uint32_t)dst;
        p.dst_address_hi = (uint32_t)(dst >> 32);
        memcpy(&base[idx], &p, dwords * 4);
        base[idx + dwords - 1] = value;
        idx += dwords;
    }
};

static void *alloc_gpu(uint32_t node, uint64_t size, bool exec, bool uncached) {
    HsaMemFlags f; f.Value = 0;
    f.ui32.PageSize     = HSA_PAGE_SIZE_4KB;
    f.ui32.HostAccess   = 1;
    f.ui32.NonPaged     = 0;
    f.ui32.CoarseGrain  = 0;
    f.ui32.NoNUMABind   = 1;
    f.ui32.Uncached     = uncached ? 1 : 0;
    f.ui32.ExecuteAccess = exec ? 1 : 0;
    void *p = nullptr;
    CHECK(hsaKmtAllocMemory(node, size, f, &p));
    CHECK(hsaKmtMapMemoryToGPU(p, size, nullptr));
    return p;
}

static uint64_t next_pow2(uint64_t v) {
    uint64_t p = 1; while (p < v) p <<= 1; return p;
}

int main(int argc, char **argv) {
    int      K        = argc > 1 ? atoi(argv[1]) : 2000;
    // fence: 0=none (dispatch only), 1=partial-flush only, 2=pf+acquire (RADV-like)
    int      fence    = argc > 2 ? atoi(argv[2]) : 2;
    int      priv     = argc > 3 ? atoi(argv[3]) : 0;
    const char *shaderPath = argc > 4 ? argv[4] : "gap_kernel.bin";
    const char *fenceName[] = {"none(dispatch-only)", "partial-flush-only", "pf+ACQUIRE_MEM(RADV-like)"};

    CHECK(hsaKmtOpenKFD());

    HsaSystemProperties sys;
    CHECK(hsaKmtAcquireSystemProperties(&sys));

    // Select the gfx1100 GPU node (or KFD_NODE override).
    int forced = getenv("KFD_NODE") ? atoi(getenv("KFD_NODE")) : -1;
    uint32_t node = (uint32_t)-1;
    for (uint32_t n = 0; n < sys.NumNodes; ++n) {
        HsaNodeProperties np;
        if (hsaKmtGetNodeProperties(n, &np) != HSAKMT_STATUS_SUCCESS) continue;
        if (np.NumFComputeCores == 0) continue;
        printf("node %u: gfx%u%u%u  FComputeCores=%u\n",
               n, np.EngineId.ui32.Major, np.EngineId.ui32.Minor,
               np.EngineId.ui32.Stepping, np.NumFComputeCores);
        if (forced >= 0) { if ((int)n == forced) node = n; }
        else if (np.EngineId.ui32.Major == 11 && node == (uint32_t)-1) node = n;
    }
    if (node == (uint32_t)-1) { fprintf(stderr, "no gfx11 GPU node found\n"); return 1; }
    printf("using KFD node %u\n", node);

    // Load the pre-assembled shader .text into an exec buffer.
    FILE *fp = fopen(shaderPath, "rb");
    if (!fp) { fprintf(stderr, "cannot open shader %s\n", shaderPath); return 1; }
    fseek(fp, 0, SEEK_END); long isaLen = ftell(fp); fseek(fp, 0, SEEK_SET);
    void *isa = alloc_gpu(node, (isaLen + 0xFFF) & ~0xFFFUL, /*exec*/true, /*uncached*/false);
    if (fread(isa, 1, isaLen, fp) != (size_t)isaLen) { fprintf(stderr, "short read\n"); return 1; }
    fclose(fp);

    // Data counter (uncached so CPU verify is coherent) and CPU-polled sentinel.
    uint32_t *data     = (uint32_t *)alloc_gpu(node, 0x1000, false, true);
    uint32_t *sentinel = (uint32_t *)alloc_gpu(node, 0x1000, false, true);
    data[0] = 0;
    sentinel[0] = 0;

    // Queue ring: must hold the whole linear stream without wraparound.
    unsigned perIter = (fence == 0) ? 5 : (fence == 1) ? 7 : 15;
    uint64_t needDwords = 64 /*reg setup*/ + (uint64_t)K * perIter + 16 /*tail*/;
    uint64_t ringBytes = next_pow2(needDwords * 4);
    if (ringBytes < 0x10000) ringBytes = 0x10000;
    void *ring = alloc_gpu(node, ringBytes, /*exec*/true, /*uncached*/true);

    HsaQueueResource res; memset(&res, 0, sizeof(res));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL,
                            ring, ringBytes, nullptr, &res));

    // ---- build the PM4 stream ----
    Emit e((uint32_t *)ring);

    const uint64_t isaVA = (uint64_t)isa;
    const uint64_t dataVA = (uint64_t)data;

    const uint32_t dims[8] = { 0,0,0, 1,1,1, 0,0 };   // 1x1x1 threads per group
    e.set_sh_reg(mmCOMPUTE_START_X, dims, 8);

    const uint64_t shifted = isaVA >> 8;
    const uint32_t pgm[6] = {
        (uint32_t)shifted, (uint32_t)(shifted >> 32), 0, 0, 0, 0
    };
    e.set_sh_reg(mmCOMPUTE_PGM_LO, pgm, 6);

    uint32_t rsrc1 =
        (0xc0 << COMPUTE_PGM_RSRC1__FLOAT_MODE__SHIFT) |
        ((uint32_t)(priv ? 1 : 0) << COMPUTE_PGM_RSRC1__PRIV__SHIFT) |
        (0x2 << COMPUTE_PGM_RSRC1__SGPRS__SHIFT) |
        (0x4 << COMPUTE_PGM_RSRC1__VGPRS__SHIFT);
    uint32_t rsrc2 =
        (4 << COMPUTE_PGM_RSRC2__USER_SGPR__SHIFT) |
        (1 << COMPUTE_PGM_RSRC2__TRAP_PRESENT__SHIFT) |
        (1 << COMPUTE_PGM_RSRC2__TGID_X_EN__SHIFT) |
        (1 << COMPUTE_PGM_RSRC2__TIDIG_COMP_CNT__SHIFT) |
        (1 << COMPUTE_PGM_RSRC2__EXCP_EN_MSB__SHIFT);
    const uint32_t rsrc[2] = { rsrc1, rsrc2 };
    e.set_sh_reg(mmCOMPUTE_PGM_RSRC1, rsrc, 2);

    const uint32_t reslim[1] = { 0 };
    e.set_sh_reg(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1);
    const uint32_t tmpring[1] = { 0 };
    e.set_sh_reg(mmCOMPUTE_TMPRING_SIZE, tmpring, 1);
    const uint32_t restart[4] = { 0,0,0,0 };
    e.set_sh_reg(mmCOMPUTE_RESTART_X, restart, 4);

    uint32_t udata[16] = {0};
    udata[0] = (uint32_t)dataVA;
    udata[1] = (uint32_t)(dataVA >> 32);
    udata[2] = (uint32_t)dataVA;
    udata[3] = (uint32_t)(dataVA >> 32);
    e.set_sh_reg(mmCOMPUTE_USER_DATA_0, udata, 16);

    const uint32_t DISPATCH_INIT = 0x00000021 | 0x8000;   // CS_EN|USE_THREAD_DIMS|CS_W32_EN (dGPU)

    for (int k = 0; k < K; ++k) {
        e.dispatch_direct(1, 1, 1, DISPATCH_INIT);
        if (fence >= 1) e.partial_flush();
        if (fence >= 2) e.acquire_mem(GCR_RADV_LIKE);
    }
    // Drain the last dispatch, then write the sentinel the CPU polls.
    e.partial_flush();
    e.write_data((uint64_t)sentinel, 0xC0FFEE);

    uint32_t total = e.idx;
    printf("config : K=%d  fence=%s  priv=%d  ring=%lluKB  stream=%u dwords\n",
           K, fenceName[fence], priv, (unsigned long long)(ringBytes >> 10), total);

    // ---- submit (GFX11 = 64-bit doorbell) and time ----
    asm volatile("" ::: "memory");
    double t0 = now_s();
    *res.Queue_write_ptr_aql = total;
    asm volatile("" ::: "memory");
    *res.Queue_DoorBell_aql = total;

    // Busy-poll the sentinel.
    const double timeout = 20.0;
    while (sentinel[0] != 0xC0FFEE) {
        if (now_s() - t0 > timeout) { fprintf(stderr, "TIMEOUT (sentinel=%x rptr=%u/%u data=%u)\n",
                                              sentinel[0], *res.Queue_read_ptr, total, data[0]); 
                                      hsaKmtDestroyQueue(res.QueueId); return 2; }
    }
    double t1 = now_s();

    double elapsed_us = (t1 - t0) * 1e6;
    printf("result : data=%u (expect %d)  %s\n", data[0], K,
           data[0] == (uint32_t)K ? "OK chain serialized" : "MISMATCH");
    printf("         total %.2f us   PER-DISPATCH PERIOD = %.3f us\n",
           elapsed_us, elapsed_us / K);

    hsaKmtDestroyQueue(res.QueueId);
    hsaKmtCloseKFD();
    return 0;
}

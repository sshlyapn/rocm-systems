// layer_kernels.hip
//
// Three real, distinct, dependent kernels approximating a transformer-decoder
// layer's elementwise/normalization ops, used to compare PM4 vs HIP/AQL
// inter-kernel overhead on a realistic dependent chain.
//
// Deliberately use ONLY hardware-provided threadIdx/blockIdx with a COMPILE-TIME
// block size BS, and no blockDim/gridDim/dynamic-LDS/printf, so the COV5 kernarg
// segment contains ONLY the explicit arguments (no hidden implicit-args buffer).
// This keeps the standalone PM4 dispatcher's kernarg setup trivial. k_rms uses
// static __shared__ LDS + __syncthreads to exercise group_segment + barriers.
#include <hip/hip_runtime.h>

#define BS 256

// LDS-reduction kernel (one workgroup, __shared__ + __syncthreads tree reduce).
// Kept in the set specifically to exercise LDS + barriers in the cross-framework
// bit-exact verification. NOTE: this is intentionally NOT a real RMSNorm -- the
// original used rsqrtf()/division, which are toolchain-AMBIGUOUS (HIP's correctly
// rounded estimate vs GLSL's <=2.5-ULP v_rcp/v_rsq), so HIP and RADV would not
// agree to the last bit. To stay bit-exact across HIP/PM4/Vulkan we use ONLY:
//   - a PURE-ADD partial sum + tree reduce (no multiply -> no FMA-contraction),
//   - a single multiply by a COMPILE-TIME exact power-of-two constant (2^-20),
//   - the already-verified 0.5*a + 0.5*b blend shape (same as k_revadd).
// Result: y = 0.5*x + 0.5*(sum(x) * 2^-20). Bounded and order-deterministic.
extern "C" __global__ void k_rms(const float* x, float* y, int n) {
    __shared__ float lds[BS];
    int t = threadIdx.x;
    float s = 0.0f;
    for (int i = t; i < n; i += BS) s += x[i];
    lds[t] = s;
    __syncthreads();
    for (int o = BS / 2; o > 0; o >>= 1) {
        if (t < o) lds[t] += lds[t + o];
        __syncthreads();
    }
    float mean = lds[0] * (1.0f / 1048576.0f);  // 2^-20: exact in fp32, no runtime div
    for (int i = t; i < n; i += BS) y[i] = 0.5f * x[i] + 0.5f * mean;
}

// Elementwise scale by a per-element weight: y *= w.
extern "C" __global__ void k_scale(float* y, const float* w, int n) {
    int i = blockIdx.x * BS + threadIdx.x;
    if (i < n) y[i] = y[i] * w[i];
}

// Residual add back into x: x += y.
extern "C" __global__ void k_add(float* x, const float* y, int n) {
    int i = blockIdx.x * BS + threadIdx.x;
    if (i < n) x[i] = x[i] + y[i];
}

// Plain copy: dst = src. Used to stage host<->device coarse-grain VRAM (the GPU
// equivalent of hipMemcpy), so PM4 can run on pure device memory without the
// CPU directly touching non-coherent coarse-grain buffers.
extern "C" __global__ void k_copy(float* dst, const float* src, int n) {
    int i = blockIdx.x * BS + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// FORWARD blend: dst[i] = 0.5*dst[i] + 0.5*src[i]. The forward (coalesced) twin of
// k_revadd: same contractive 0.5/0.5 blend shape (so the chain stays bounded and
// bit-exact), same cross-buffer read dependency, but a NORMAL forward read of
// src[i] instead of the mirrored src[n-1-i] scatter. Used by CHAIN=serial to model
// a realistic decode chain write->read->modify->write with DISTINCT kernels bound
// per dispatch, without the reverse-read cache-coherence stress of k_revadd.
extern "C" __global__ void k_blend(float* dst, const float* src, int n) {
    int i = blockIdx.x * BS + threadIdx.x;
    if (i < n) dst[i] = 0.5f * dst[i] + 0.5f * src[i];
}

// REVERSE-READ mix: dst[i] = 0.5*dst[i] + 0.5*src[n-1-i]. Thread i (in workgroup
// i/BS) reads src at the MIRRORED index n-1-i, which in the previous dispatch was
// written by a DIFFERENT workgroup (hence a different WGP / different L0,L1
// vector cache). This is the cache-coherence stress case: if the inter-dispatch
// fence does not INVALIDATE the per-WGP L0/L1 (e.g. a drain-only barrier), the
// reader can hit a stale cache line and produce a wrong result. With a correct
// acquire (GLV_INV|GL1_INV, what FENCE=2 / RADV / AQL emit) it reads fresh from
// L2 and is bit-exact. The averaging form keeps values bounded so the checksum
// stays finite over thousands of iterations. dst and src must be DISTINCT
// buffers (no in-place hazard).
extern "C" __global__ void k_revadd(float* dst, const float* src, int n) {
    int i = blockIdx.x * BS + threadIdx.x;
    if (i < n) dst[i] = 0.5f * dst[i] + 0.5f * src[n - 1 - i];
}

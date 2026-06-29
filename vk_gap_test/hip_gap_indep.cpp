// Does a SINGLE HIP stream serialize INDEPENDENT kernels?
//
// Kernel k writes buffer region (k % buffers). With buffers=1 every kernel hits
// the same buffer -> RAW hazard -> the memory-dependency tracker selects the
// "sync" dispatch header (barrier bit + AGENT fence). With buffers>=2 consecutive
// kernels touch different regions -> no hazard -> the "nosync" header (NO barrier
// bit) is used automatically (rocvirtual.cpp:764-767). If independent kernels then
// overlap (concurrency > 1, period drops), the barrier bit / dependency is the
// serializer; if they still serialize, single-queue CP ordering is fundamental.
//
// Usage: hip_gap_indep [K] [spin] [n] [buffers]
// Build: hipcc -O2 --offload-arch=gfx1100 hip_gap_indep.cpp -o hip_gap_indep

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIPCHECK(x) do { hipError_t _e=(x); if(_e!=hipSuccess){ \
  fprintf(stderr,"HIP error '%s' at %s:%d\n",hipGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

__device__ __forceinline__ unsigned long long gpu_realtime() {
    return (unsigned long long)__builtin_readsteadycounter();
}

__global__ void gap_kernel(float * data, uint32_t spin, uint32_t n,
                           unsigned long long * tstart, unsigned long long * tend, uint32_t kidx) {
    if (threadIdx.x == 0) atomicMin(&tstart[kidx], gpu_realtime());
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = data[gid];
        for (uint32_t i = 0; i < spin; ++i) v = v * 1.0000001f + 1.0f;
        data[gid] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(&tend[kidx], gpu_realtime());
}

int main(int argc, char ** argv) {
    int      K       = argc > 1 ? atoi(argv[1]) : 4000;
    uint32_t spin    = argc > 2 ? (uint32_t)atoi(argv[2]) : 0;
    uint32_t n       = argc > 3 ? (uint32_t)atoi(argv[3]) : 4096;
    int      buffers = argc > 4 ? atoi(argv[4]) : 1;

    hipDeviceProp_t prop; HIPCHECK(hipGetDeviceProperties(&prop, 0));
    printf("device : %s (%s)  K=%d spin=%u n=%u buffers=%d (1 stream)\n", prop.name, prop.gcnArchName, K, spin, n, buffers);

    size_t region = std::max(n, 64u);
    float * data; HIPCHECK(hipMalloc(&data, region * (size_t)buffers * sizeof(float)));
    HIPCHECK(hipMemset(data, 0, region * (size_t)buffers * sizeof(float)));
    unsigned long long *tstart, *tend;
    HIPCHECK(hipMalloc(&tstart, (size_t)K * sizeof(unsigned long long)));
    HIPCHECK(hipMalloc(&tend, (size_t)K * sizeof(unsigned long long)));

    hipStream_t stream; HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    uint32_t blocks = (n + 255) / 256, threads = 256;
    auto reset_ts = [&]() {
        HIPCHECK(hipMemset(tstart, 0xFF, (size_t)K * sizeof(unsigned long long)));
        HIPCHECK(hipMemset(tend, 0x00, (size_t)K * sizeof(unsigned long long)));
    };
    auto launch_all = [&]() {
        for (int k = 0; k < K; k++)
            gap_kernel<<<blocks, threads, 0, stream>>>(data + (size_t)(k % buffers) * region, spin, n, tstart, tend, (uint32_t)k);
        HIPCHECK(hipStreamSynchronize(stream));
    };
    reset_ts(); launch_all();              // warmup
    reset_ts();
    hipEvent_t e0, e1; HIPCHECK(hipEventCreate(&e0)); HIPCHECK(hipEventCreate(&e1));
    HIPCHECK(hipEventRecord(e0, stream)); launch_all(); HIPCHECK(hipEventRecord(e1, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    float wall_ms = 0; HIPCHECK(hipEventElapsedTime(&wall_ms, e0, e1));

    std::vector<unsigned long long> hs(K), he(K);
    HIPCHECK(hipMemcpy(hs.data(), tstart, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(he.data(), tend, (size_t)K * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    unsigned long long tmin = ~0ULL, tmax = 0; double busy = 0;
    for (int k = 0; k < K; k++) { tmin = std::min(tmin, hs[k]); tmax = std::max(tmax, he[k]); busy += (double)(he[k] - hs[k]); }
    unsigned long long span = tmax - tmin;
    double ns = (double)wall_ms * 1e6 / (double)span;
    printf("  calib %.4f ns/tick | period(span/K) = %.3f us | mean busy = %.3f us | concurrency = %.2fx\n",
           ns, (double)span * ns / 1000.0 / K, busy * ns / 1000.0 / K, busy / (double)span);
    return 0;
}

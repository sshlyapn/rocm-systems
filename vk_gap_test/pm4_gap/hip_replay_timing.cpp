// hip_replay_timing.cpp
//
// Isolated HOST (CPU) cost of issuing one hipGraph replay -- AQL graph path vs
// the PM4-IB path (HIP_PM4_GRAPH=1). This measures ONLY the host-side enqueue
// latency of hipGraphLaunch, not GPU execution:
//
//   for r in R:
//     hipStreamSynchronize(stream);   // drain the queue OUTSIDE the timer
//     t0 = now; hipGraphLaunch(exec); t1 = now;   // pure host enqueue cost
//     acc += t1 - t0;
//   report acc / R
//
// With the queue empty at each iteration the launch never blocks on queue
// space, so the number is the runtime's host work to submit the replay.
//
// HYPOTHESIS: the AQL path is O(N) -- it copies N 64-byte AQL packets into the
// ring -- while the PM4 path is O(1) -- one vendor PM4-IB packet + one doorbell
// regardless of N. Sweeping N (chain length) should show AQL rising ~linearly
// and PM4 staying flat. Tiny grids (small M) keep the GPU from ever falling
// behind. Plain ASCII only.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <vector>

#define BS 64
#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP err %s @%d: %s\n",#x,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)

static double now_us(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1e6 + ts.tv_nsec*1e-3; }

// Trivial dependent kernel: x += w. A chain of these on the same buffer is a
// pure RAW chain of N kernel-dispatch packets (no memcpy/host nodes), so the
// captured graph is all-dispatch and the PM4 path engages fully.
__global__ void addk(float* x, const float* w, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) x[i] += w[i];
}

int main(int argc,char** argv){
    int N = argc>1 ? atoi(argv[1]) : 64;     // chain length (dispatch packets)
    int M = argc>2 ? atoi(argv[2]) : 256;    // elems (keep tiny so GPU never lags)
    int R = argc>3 ? atoi(argv[3]) : 4000;   // timed repeats

    size_t bytes = (size_t)M*4;
    float *x,*w; HC(hipMalloc(&x,bytes)); HC(hipMalloc(&w,bytes));
    std::vector<float> hx(M,0.1f), hw(M,1.0f);
    HC(hipMemcpy(x,hx.data(),bytes,hipMemcpyHostToDevice));
    HC(hipMemcpy(w,hw.data(),bytes,hipMemcpyHostToDevice));

    dim3 grid((M+BS-1)/BS), block(BS);
    hipStream_t stream; HC(hipStreamCreate(&stream));

    // Capture an N-long addk chain into one graph.
    hipGraph_t graph; hipGraphExec_t exec;
    HC(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for(int it=0; it<N; ++it)
        hipLaunchKernelGGL(addk, grid, block, 0, stream, x, w, M);
    HC(hipStreamEndCapture(stream,&graph));

    // One-time instantiate cost (graph -> exec; captures AQL packets for the
    // PM4 path, allocates kernarg pool). Host-only.
    double ti0 = now_us();
    HC(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    double ti1 = now_us();
    double t_instantiate = ti1 - ti0;

    // FIRST-EVER replay: this host call is where the PM4 path compiles the whole
    // chain into one PM4 IB (buildPm4GraphIb), allocates the executable IB from a
    // device memory pool, and DMA-uploads it -- a synchronous one-time cost that
    // must finish before the submit doorbell. Time the host call in isolation;
    // every later launch reuses this IB. The AQL path has no IB build, so its
    // first launch ~ its steady launch.
    double tc0 = now_us();
    HC(hipGraphLaunch(exec,stream));
    double tc1 = now_us();
    double t_cold = tc1 - tc0;
    HC(hipStreamSynchronize(stream));
    double tc2 = now_us();
    double t_cold_e2e = tc2 - tc0;   // first launch host call + GPU completion (wall)

    // Second launch (warm, IB already built/cached) for an apples-to-apples delta.
    double tw0 = now_us();
    HC(hipGraphLaunch(exec,stream));
    double tw1 = now_us();
    double t_warm2 = tw1 - tw0;
    HC(hipStreamSynchronize(stream));

    // Remaining warmup.
    for(int i=0;i<3;++i){ HC(hipGraphLaunch(exec,stream)); }
    HC(hipStreamSynchronize(stream));

    // Timed: pure host enqueue cost with an empty queue each iteration.
    double acc = 0.0, mn = 1e30, mx = 0.0;
    for(int r=0; r<R; ++r){
        HC(hipStreamSynchronize(stream));      // drain OUTSIDE the timer
        double t0 = now_us();
        HC(hipGraphLaunch(exec,stream));
        double t1 = now_us();
        double dt = t1 - t0;
        acc += dt; if(dt<mn) mn=dt; if(dt>mx) mx=dt;
    }
    HC(hipStreamSynchronize(stream));

    double avg = acc / R;
    printf("N=%-4d M=%-6d R=%d  instantiate=%.3f us  FIRST(cold,build+upload)=%.3f us  FIRST_e2e(launch+sync)=%.3f us  "
           "2nd(warm)=%.3f us || steady host hipGraphLaunch: avg=%.3f us  min=%.3f us  max=%.3f us  per-dispatch=%.4f us\n",
           N, M, R, t_instantiate, t_cold, t_cold_e2e, t_warm2, avg, mn, mx, avg/(double)N);
    return 0;
}

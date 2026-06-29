// A real hipcc-compiled kernel to validate PM4 dispatch against the full ROCm
// kernel ABI (KERNARG_SEGMENT_PTR + dispatch_ptr + hidden args), as opposed to
// the hand-rolled kfdtest USER_DATA ABI. One thread increments *counter.
#include <hip/hip_runtime.h>

extern "C" __global__ void inc_kernel(int* counter) {
    *counter = *counter + 1;
}

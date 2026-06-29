/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/devhostcall.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rockernel.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/roccounters.hpp"
#include "platform/activity.hpp"
#include "platform/kernel.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/command_utils.hpp"
#include "platform/memory.hpp"
#include "platform/sampler.hpp"
#include "utils/debug.hpp"
#include "os/os.hpp"

#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <cinttypes>

#if defined(__linux__)
#include <dlfcn.h>
#endif

#if defined(__AVX__)
#if defined(__MINGW64__)
#include <intrin.h>
#else
#include <immintrin.h>
#endif
#endif

/**
 * HSA image object size in bytes (see HSA spec)
 */
#define HSA_IMAGE_OBJECT_SIZE 48

/**
 * HSA image object alignment in bytes (see HSA spec)
 */
#define HSA_IMAGE_OBJECT_ALIGNMENT 16

/**
 * HSA sampler object size in bytes (see HSA spec)
 */
#define HSA_SAMPLER_OBJECT_SIZE 32

/**
 * HSA sampler object alignment in bytes (see HSA spec)
 */
#define HSA_SAMPLER_OBJECT_ALIGNMENT 16

namespace amd::roc {
// (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) invalidates I, K and L1
// (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE) invalidates L1, L2 and flushes
// L2

static constexpr uint16_t kInvalidAql = (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE);

static constexpr uint16_t kBarrierPacketHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kNopPacketHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierPacketAcquireHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierPacketReleaseHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierVendorPacketHeader =
    (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierVendorPacketNopScopeHeader =
    (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr hsa_barrier_and_packet_t kBarrierAcquirePacket = {
    kBarrierPacketAcquireHeader, 0, 0, {{0}}, 0, {0}};

static constexpr hsa_barrier_and_packet_t kBarrierReleasePacket = {
    kBarrierPacketReleaseHeader, 0, 0, {{0}}, 0, {0}};

double Timestamp::ticksToTime_ = 0;

static unsigned extractAqlBits(unsigned v, unsigned pos, unsigned width) {
  return (v >> pos) & ((1 << width) - 1);
};

static inline void logAqlDispatchPacket(const hsa_queue_t* queue, uint16_t header,
                                        const hsa_kernel_dispatch_packet_t* pkt,
                                        uint64_t rptr, uint64_t wptr,
                                        const char* prefix = "") {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL,
          "SWq=0x%zx, HWq=0x%zx, id=%d,%s Dispatch Header = "
          "0x%x (type=%d, barrier=%d, acquire=%d, release=%d), "
          "setup=%d, grid=[%u, %u, %u], workgroup=[%u, %u, %u], "
          "private_seg_size=%u, group_seg_size=%u, kernel_obj=0x%zx, "
          "kernarg_address=0x%zx, completion_signal=0x%zx, correlation_id=%zu, "
          "rptr=%lu, wptr=%lu",
          queue, queue->base_address, queue->id, prefix, header,
          extractAqlBits(header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE),
          extractAqlBits(header, HSA_PACKET_HEADER_BARRIER, HSA_PACKET_HEADER_WIDTH_BARRIER),
          extractAqlBits(header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
          extractAqlBits(header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE),
          pkt->setup, pkt->grid_size_x, pkt->grid_size_y, pkt->grid_size_z,
          pkt->workgroup_size_x, pkt->workgroup_size_y, pkt->workgroup_size_z,
          pkt->private_segment_size, pkt->group_segment_size, pkt->kernel_object,
          pkt->kernarg_address, pkt->completion_signal.handle, pkt->reserved2,
          rptr, wptr);
}

static inline void logAqlDispatchPacketExtended(
    const hsa_queue_t* queue, uint16_t header, const hsa_amd_ext_kernel_dispatch_packet_t* pkt,
    uint64_t rptr, uint64_t wptr, const char* prefix = "") {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL,
          "SWq=0x%zx, HWq=0x%zx, id=%d,%s Dispatch Header = "
          "0x%x (type=%d, barrier=%d, acquire=%d, release=%d), "
          "setup=%d, cluster_count=[%u, %u, %u], cluster_size=[%u, %u, %u], "
          "workgroup=[%u, %u, %u], private_seg_size=%u, group_seg_size=%u, kernel_obj=0x%zx, "
          "kernarg_address=0x%zx, dep_signal=0x%zx, completion_signal=0x%zx, "
          "rptr=%lu, wptr=%lu",
          queue, queue->base_address, queue->id, prefix, header,
          extractAqlBits(header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE),
          extractAqlBits(header, HSA_PACKET_HEADER_BARRIER, HSA_PACKET_HEADER_WIDTH_BARRIER),
          extractAqlBits(header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
          extractAqlBits(header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE),
          pkt->setup, pkt->cluster_count_x, pkt->cluster_count_y, pkt->cluster_count_z,
          pkt->cluster_size_x, pkt->cluster_size_y, pkt->cluster_size_z, pkt->workgroup_size_x,
          pkt->workgroup_size_y, pkt->workgroup_size_z, pkt->private_segment_size,
          pkt->group_segment_size, pkt->kernel_object, pkt->kernarg_address,
          pkt->dep_signal.handle, pkt->completion_signal.handle, rptr, wptr);
}

static inline void logAqlBarrierPacket(const hsa_queue_t* queue, uint16_t header,
                                       const hsa_barrier_and_packet_t* pkt,
                                       uint64_t rptr, uint64_t wptr,
                                       const char* prefix = "") {
  uint16_t pktType = extractAqlBits(header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
  const char* typeStr = (pktType == HSA_PACKET_TYPE_BARRIER_OR) ? "Barrier-OR" : "Barrier-AND";
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL,
          "SWq=0x%zx, HWq=0x%zx, id=%d,%s %s Header = "
          "0x%x (type=%d, barrier=%d, acquire=%d, release=%d), "
          "dep_signal=[0x%zx, 0x%zx, 0x%zx, 0x%zx, 0x%zx], "
          "completion_signal=0x%zx, rptr=%lu, wptr=%lu",
          queue, queue->base_address, queue->id, prefix, typeStr, header,
          pktType,
          extractAqlBits(header, HSA_PACKET_HEADER_BARRIER, HSA_PACKET_HEADER_WIDTH_BARRIER),
          extractAqlBits(header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
          extractAqlBits(header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE),
          pkt->dep_signal[0].handle, pkt->dep_signal[1].handle,
          pkt->dep_signal[2].handle, pkt->dep_signal[3].handle,
          pkt->dep_signal[4].handle,
          pkt->completion_signal.handle, rptr, wptr);
}

static inline void logAqlBarrierValuePacket(const hsa_queue_t* queue, uint16_t header,
                                            const hsa_amd_barrier_value_packet_t* pkt,
                                            uint64_t rptr, uint64_t wptr,
                                            const char* prefix = "") {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL,
          "SWq=0x%zx, HWq=0x%zx, id=%d,%s BarrierValue Header = 0x%x AmdFormat = 0x%x "
          "(type=%d, barrier=%d, acquire=%d, release=%d), "
          "signal=0x%zx, value=0x%llx, mask=0x%llx, cond=%s, "
          "completion_signal=0x%zx, rptr=%lu, wptr=%lu",
          queue, queue->base_address, queue->id, prefix,
          header, HSA_AMD_PACKET_TYPE_BARRIER_VALUE,
          extractAqlBits(header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE),
          extractAqlBits(header, HSA_PACKET_HEADER_BARRIER, HSA_PACKET_HEADER_WIDTH_BARRIER),
          extractAqlBits(header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
          extractAqlBits(header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE),
          pkt->signal.handle, pkt->value, pkt->mask,
          pkt->cond == 0 ? "EQ" : pkt->cond == 1 ? "NE" : pkt->cond == 2 ? "LT" : "GTE",
          pkt->completion_signal.handle, rptr, wptr);
}

// ================================================================================================
void ProfilingSignal::CacheTimingData(hsa_agent_t gpu_device) {
  // Lock needed as async handler thread can also touch this structure
  std::scoped_lock lock(lock_);

  // Return if timing is already cached
  if (cached_timing_.valid_) {
    return;
  }

  // Wait for this signal to complete if not already done
  if (Hsa::signal_load_relaxed(signal_) > 0) {
    WaitForSignal(signal_);
  }

  // Extract timing and cache it
  if (IsSdmaEngine(engine_)) {
    hsa_amd_profiling_async_copy_time_t time = {};
    Hsa::profiling_get_async_copy_time(signal_, &time);
    cached_timing_.start_ = time.start;
    cached_timing_.end_ = time.end;
  } else {
    hsa_amd_profiling_dispatch_time_t time = {};
    Hsa::profiling_get_dispatch_time(gpu_device, signal_, &time);
    cached_timing_.start_ = time.start;
    cached_timing_.end_ = time.end;
  }
  cached_timing_.valid_ = true;
}

// ================================================================================================
// Process GPU timing for signals
// If single_signal is nullptr, processes all signals and clears the list
// If single_signal is provided, processes only that signal with merge enabled
void Timestamp::checkGpuTime(ProfilingSignal* single_signal) {
  std::scoped_lock s(lock_);

  // For single signal mode, validate it exists in the list
  if (single_signal != nullptr) {
    auto it = std::find(signals_.begin(), signals_.end(), single_signal);
    if (it == signals_.end()) {
      return;
    }
  }

  if (HwProfiling()) {
    uint64_t start = std::numeric_limits<uint64_t>::max();
    uint64_t end = 0;
    uint64_t sdmaStart = std::numeric_limits<uint64_t>::max();
    uint64_t sdmaEnd = 0;

    // Process either single signal or all signals
    auto process_signal = [&](ProfilingSignal* sig) {
      // Skip signals already processed
      if (sig->flags_.done_) {
        return;
      }

      // Ignore the wait if runtime processes API callback, because the signal value is bigger
      // than expected and the value reset will occur after API callback is done
      if (GetCallbackSignal().handle == 0 || GetBlocking() == false) {
        ExtractSignalTiming(sig, start, end, sdmaStart, sdmaEnd);
      }

      if (IsLogEnabled(amd::LOG_INFO, amd::LOG_TS) &&
          (command().GetBatchHead() == nullptr || command().profilingInfo().marker_ts_ ||
           command().type() == CL_COMMAND_TASK)) {
        uint64_t sig_start, sig_end;
        sig->GetCachedTiming(sig_start, sig_end);
        amd_signal_t* amdSignal = reinterpret_cast<amd_signal_t*>(sig->signal_.handle);

        ClPrint(amd::LOG_INFO, amd::LOG_TS,
                "Signal = (0x%lx), Translated start/end = %ld / %ld, Elapsed = %ld ns, "
                "ticks start/end = %ld / %ld, Ticks elapsed = %ld, Engine = %u",
                sig->signal_.handle, sig_start, sig_end, sig_end - sig_start,
                amdSignal->start_ts, amdSignal->end_ts, amdSignal->end_ts - amdSignal->start_ts,
                sig->engine_);
      }
    };

    if (single_signal != nullptr) {
      process_signal(single_signal);
      // Remove the signal from the list after extracting its timing.
      // This prevents a stale entry when ActiveSignal() recycles and re-adds
      // the same ProfilingSignal — without this, the recycled signal appears
      // twice in signals_ and the first (stale) entry re-reads wrong HW timestamps.
      auto it = std::find(signals_.begin(), signals_.end(), single_signal);
      if (it != signals_.end()) {
        signals_.erase(it);
      }
    } else {
      for (auto it : signals_) {
        process_signal(it);
      }
      signals_.clear();
    }

    // Aggregate timing based on command type
    if (end != 0 || sdmaEnd != 0) {
      uint64_t final_start, final_end;
      const auto cmd_type = command().type();

      if (cmd_type == CL_COMMAND_COPY_BUFFER || cmd_type == CL_COMMAND_READ_BUFFER ||
          cmd_type == CL_COMMAND_WRITE_BUFFER || cmd_type == CL_COMMAND_COPY_BUFFER_RECT ||
          cmd_type == CL_COMMAND_READ_BUFFER_RECT || cmd_type == CL_COMMAND_WRITE_BUFFER_RECT) {
        // Copy/Read/Write — prefer SDMA timing, fall back to compute
        final_start = ((sdmaEnd != 0) ? sdmaStart : start) * ticksToTime_;
        final_end = ((sdmaEnd != 0) ? sdmaEnd : end) * ticksToTime_;
      } else {
        // Batch copy and all other commands — min/max across all engines
        final_start = std::min(sdmaEnd != 0 ? sdmaStart : start,
                               end != 0 ? start : sdmaStart) * ticksToTime_;
        final_end = std::max(sdmaEnd, end) * ticksToTime_;
      }

      if (!accum_ena_) {
        start_ = final_start;
        accum_ena_ = true;
      } else {
        start_ = std::min(start_, final_start);
      }
      end_ = std::max(end_, final_end);
    }
  }
}

// ================================================================================================
// Extract timing from a single signal and update accumulators
void Timestamp::ExtractSignalTiming(ProfilingSignal* signal,
                                    uint64_t& start, uint64_t& end,
                                    uint64_t& sdmaStart, uint64_t& sdmaEnd) {
  // Ensure timing data is cached
  if (!signal->IsTimingCached()) {
    signal->CacheTimingData(gpu()->gpu_device());
  }

  // Get cached timing
  uint64_t sig_start, sig_end;
  signal->GetCachedTiming(sig_start, sig_end);

  // Lock signal for accessing engine_ and flags_
  std::scoped_lock sig_lock(signal->LockSignalOps());

  // Guard against invalid timestamps from GPU (e.g. HW didn't write start_ts/end_ts).
  // TranslateTime returns 0 for these cases. Skip accumulation to avoid corrupting timing.
  if (sig_start == 0 || sig_end == 0 || sig_end < sig_start) {
    ClPrint(amd::LOG_WARNING, amd::LOG_TS,
            "Invalid signal timing: start=%lu, end=%lu, signal=0x%lx",
            sig_start, sig_end, signal->signal_.handle);
    signal->flags_.done_ = true;
    return;
  }

  // Update appropriate accumulators based on engine type
  if (IsSdmaEngine(signal->engine_)) {
    sdmaStart = std::min(sig_start, sdmaStart);
    sdmaEnd = std::max(sig_end, sdmaEnd);
  } else {
    start = std::min(sig_start, start);
    end = std::max(sig_end, end);
  }

  // Handle AccumulateCommand timestamps (convert ticks to system time)
  if ((command().type() == CL_COMMAND_TASK) && (signal->flags_.isPacketDispatch_ == true)) {
    static_cast<amd::AccumulateCommand&>(command()).addTimestamps(
        static_cast<uint64_t>(sig_start * ticksToTime_),
        static_cast<uint64_t>(sig_end * ticksToTime_));
  }

  signal->flags_.done_ = true;
}

// ================================================================================================
bool HsaAmdSignalHandler(hsa_signal_value_t value, void* arg) {
  Timestamp* ts = reinterpret_cast<Timestamp*>(arg);

  VirtualGPU* const gpu = ts->gpu();

  ClPrint(amd::LOG_INFO, amd::LOG_SIG, "Handler: value(%d), timestamp(%p), handle(0x%lx)",
          static_cast<uint32_t>(value), arg,
          ts->HwProfiling() ? ts->Signals()[0]->signal_.handle : 0);

  // Save callback signal
  hsa_signal_t callback_signal = ts->GetCallbackSignal();

  bool isBlocking = ts->GetBlocking();

  // Update the batch, since signal is complete
  gpu->updateCommandsState(ts->command().GetBatchHead());

  // Opportunistically try to release the HW queue if it's now idle
  // This helps reclaim queues in async workloads without explicit sync
  gpu->ReleaseHwQueue();

  // Reset API callback signal. It will release AQL queue and start commands processing
  if (callback_signal.handle != 0 && isBlocking) {
    Hsa::signal_subtract_relaxed(callback_signal, 1);
  }

  // Return false, so the callback will not be called again for this signal
  gpu->QueuedAsyncHandlers()--;
  gpu->release();
  return false;
}

// ================================================================================================
bool VirtualGPU::MemoryDependency::create(size_t numMemObj) {
  if (numMemObj > 0) {
    // Allocate the array of memory objects for dependency tracking
    memObjectsInQueue_ = new MemoryState[numMemObj];
    if (nullptr == memObjectsInQueue_) {
      return false;
    }
    memset(memObjectsInQueue_, 0, sizeof(MemoryState) * numMemObj);
    maxMemObjectsInQueue_ = numMemObj;
  }

  return true;
}

// ================================================================================================
void VirtualGPU::MemoryDependency::validate(VirtualGPU& gpu, const Memory* memory, bool readOnly) {
  bool flushL1Cache = false;

  if (maxMemObjectsInQueue_ == 0) {
    // Sync AQL packets
    gpu.setAqlHeader(gpu.dispatchPacketHeader_);
    return;
  }

  uint64_t curStart = reinterpret_cast<uint64_t>(memory->getDeviceMemory());
  uint64_t curEnd = curStart + memory->size();

  // Loop through all memory objects in the queue and find dependency
  // @note don't include objects from the current kernel
  for (size_t j = 0; j < endMemObjectsInQueue_; ++j) {
    // Check if the queue already contains this mem object and
    // GPU operations aren't readonly
    uint64_t busyStart = memObjectsInQueue_[j].start_;
    uint64_t busyEnd = memObjectsInQueue_[j].end_;

    // Check if the start inside the busy region
    if ((((curStart >= busyStart) && (curStart < busyEnd)) ||
         // Check if the end inside the busy region
         ((curEnd > busyStart) && (curEnd <= busyEnd)) ||
         // Check if the start/end cover the busy region
         ((curStart <= busyStart) && (curEnd >= busyEnd))) &&
        // If the buys region was written or the current one is for write
        (!memObjectsInQueue_[j].readOnly_ || !readOnly)) {
      flushL1Cache = true;
      break;
    }
  }

  // Did we reach the limit?
  if (maxMemObjectsInQueue_ <= numMemObjectsInQueue_) {
    flushL1Cache = true;
  }

  if (flushL1Cache) {
    // Sync AQL packets
    gpu.setAqlHeader(gpu.dispatchPacketHeader_);

    // Clear memory dependency state
    const static bool All = true;
    clear(!All);
  }

  // Insert current memory object into the queue always,
  // since runtime calls flush before kernel execution and it has to keep
  // current kernel in tracking
  memObjectsInQueue_[numMemObjectsInQueue_].start_ = curStart;
  memObjectsInQueue_[numMemObjectsInQueue_].end_ = curEnd;
  memObjectsInQueue_[numMemObjectsInQueue_].readOnly_ = readOnly;
  numMemObjectsInQueue_++;
}

// ================================================================================================
void VirtualGPU::MemoryDependency::clear(bool all) {
  if (numMemObjectsInQueue_ > 0) {
    if (all) {
      endMemObjectsInQueue_ = numMemObjectsInQueue_;
    }

    // If the current launch didn't start from the beginning, then move the data
    if (0 != endMemObjectsInQueue_) {
      // Preserve all objects from the current kernel
      size_t i, j;
      for (i = 0, j = endMemObjectsInQueue_; j < numMemObjectsInQueue_; i++, j++) {
        memObjectsInQueue_[i].start_ = memObjectsInQueue_[j].start_;
        memObjectsInQueue_[i].end_ = memObjectsInQueue_[j].end_;
        memObjectsInQueue_[i].readOnly_ = memObjectsInQueue_[j].readOnly_;
      }
    } else if (numMemObjectsInQueue_ >= maxMemObjectsInQueue_) {
      // note: The array growth shouldn't occur under the normal conditions,
      // but in a case when SVM path sends the amount of SVM ptrs over
      // the max size of kernel arguments
      MemoryState* ptr = new MemoryState[maxMemObjectsInQueue_ << 1];
      if (nullptr == ptr) {
        numMemObjectsInQueue_ = 0;
        return;
      }
      maxMemObjectsInQueue_ <<= 1;
      memcpy(ptr, memObjectsInQueue_, sizeof(MemoryState) * numMemObjectsInQueue_);
      delete[] memObjectsInQueue_;
      memObjectsInQueue_ = ptr;
    }

    numMemObjectsInQueue_ -= endMemObjectsInQueue_;
    endMemObjectsInQueue_ = 0;
  }
}

// ================================================================================================
VirtualGPU::HwQueueTracker::~HwQueueTracker() {
  for (auto& signal : signal_list_) {
    CpuWaitForSignal(signal);
    signal->release();
  }
  // Destroy all extra signals. Note: these signals must be idle already
  while (signal_pool_.size() != 0) {
    signal_pool_.top()->release();
    signal_pool_.pop();
  }
  while (signal_pool_irq_.size() != 0) {
    signal_pool_irq_.top()->release();
    signal_pool_irq_.pop();
  }
}

// ================================================================================================
bool VirtualGPU::HwQueueTracker::CreateSignal(ProfilingSignal* signal, bool interrupt) const {
  const Settings& settings = gpu_.dev().settings();
  // Use interrupts when active wait is disabled to avoid extra polling on the CPU.
  interrupt |= !gpu_.dev().ActiveWait();
  // Check if the interrupt was requested for the signal
  if (interrupt && settings.system_scope_signal_) {
    if (HSA_STATUS_SUCCESS != Hsa::signal_create(0, 0, nullptr, &signal->signal_)) {
      return false;
    }
  } else {
    if (HSA_STATUS_SUCCESS !=
        Hsa::signal_create(0, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal->signal_)) {
      return false;
    }
  }
  signal->flags_.interrupt_ = interrupt;
  return true;
}

// ================================================================================================
bool VirtualGPU::HwQueueTracker::Create() {
  const uint kSignalListSize = ROC_SIGNAL_POOL_SIZE;
  signal_list_.resize(kSignalListSize);
  for (uint i = 0; i < kSignalListSize; ++i) {
    std::unique_ptr<ProfilingSignal> signal(new ProfilingSignal());
    if ((signal == nullptr) || !CreateSignal(signal.get())) {
      return false;
    }
    signal_list_[i] = signal.release();
  }
  // Add extra signals with the interrupts for the callbacks
  if (gpu_.dev().ActiveWait()) {
    for (uint32_t i = 0; i < 5; ++i) {
      std::unique_ptr<ProfilingSignal> signal(new ProfilingSignal());
      constexpr bool kEnableInterrupt = true;
      if ((signal == nullptr) || !CreateSignal(signal.get(), kEnableInterrupt)) {
        return false;
      }
      signal_pool_irq_.push(signal.release());
    }
  }
  return true;
}

// ================================================================================================
hsa_signal_t VirtualGPU::HwQueueTracker::ActiveSignal(hsa_signal_value_t init_val, Timestamp* ts,
                                                      bool attach_signal) {
  amd::Command* cmd = gpu_.command();
  // If no signal is needed, decrement the refcount and clear the hw_event of current command
  if (!attach_signal) {
    if (nullptr != cmd) {
      if (cmd->HwEvent() != nullptr) {
        reinterpret_cast<ProfilingSignal*>(cmd->HwEvent())->release();
      }
      cmd->SetHwEvent(nullptr);
    }
    return hsa_signal_t{0};
  }

  bool new_signal = false;

  // Peep signal +2 ahead to see if its done
  auto temp_id = (current_id_ + 2) % signal_list_.size();

  // If GPU is still busy with processing, then add more signals to avoid more frequent stalls
  if (Hsa::signal_load_relaxed(signal_list_[temp_id]->signal_) > 0) {
    std::unique_ptr<ProfilingSignal> signal(new ProfilingSignal());
    if ((signal != nullptr) && CreateSignal(signal.get())) {
      // Find valid new index
      ++current_id_ %= signal_list_.size();
      // Insert the new signal into the current slot and ignore any wait
      signal_list_.insert(signal_list_.begin() + current_id_, signal.release());
      new_signal = true;
    }
  }

  // If it's the new signal, then the wait can be avoided.
  // That will allow to grow the list of signals without stalls
  if (!new_signal) {
    // Find valid index
    ++current_id_ %= signal_list_.size();
    // Make sure the previous operation on the current signal is done
    WaitCurrent();

    // Have to wait the next signal in the queue to avoid a race condition between
    // a GPU waiter(which may be not triggered yet) and CPU signal reset below
    WaitNext();
  }

  if (signal_list_[current_id_]->referenceCount() > 1) {
    // The signal was assigned to the global marker's event, hence runtime can't reuse it
    // and needs a new signal
    std::unique_ptr<ProfilingSignal> signal(new ProfilingSignal());

    // Ensure that signals of the same type are created with the same interrupt flag,
    // as the tracking list depends on this for reuse.
    if ((signal != nullptr) && CreateSignal(signal.get(), signal_list_[current_id_]->flags_.interrupt_)) {
      signal_list_[current_id_]->release();
      signal_list_[current_id_] = signal.release();
    } else {
      assert(!"ProfilingSignal reallocation failed! Marker has a conflict with signal reuse!");
    }
  }

  bool enqueHandler = false;
  if (ts != nullptr) {
    enqueHandler =
        (ts->command().Callback() != nullptr || ts->command().GetBatchHead() != nullptr) &&
        !ts->command().CpuWaitRequested();
  }
  bool use_irq = enqueHandler || (IS_WINDOWS && gpu_.ForceIrq());
  // Check if the signal doesn't match the requested one.
  // Note: runtime needs the interrupts for the callbacks and the marker events, but it can reuse
  // the non-interrupt signals for the regular dispatches
  if ((signal_list_[current_id_]->flags_.interrupt_ != use_irq) && gpu_.dev().ActiveWait()) {
    // Use different stacks if an interrupt is required or not.
    // @note: if runtime needs an interrupt, then the tracking list replaces the original signal
    // with the interrupt signal and saves the signal without interrupt, or vise versa
    auto& pool_get = (use_irq) ? signal_pool_irq_ : signal_pool_;
    auto& pool_save = (use_irq) ? signal_pool_ : signal_pool_irq_;

    // Check if a free signal in the pop stack isn't available
    if (pool_get.empty()) {
      std::unique_ptr<ProfilingSignal> signal(new ProfilingSignal());
      if ((signal != nullptr) && CreateSignal(signal.get(), use_irq)) {
        pool_get.push(signal.release());
      }
    }
    // Make sure a free signal exists and replace it in the current slot
    if (!pool_get.empty()) {
      pool_save.push(signal_list_[current_id_]);
      signal_list_[current_id_] = pool_get.top();
      pool_get.pop();
    }
  }
  ProfilingSignal* prof_signal = signal_list_[current_id_];
  // Reset the signal and return
  Hsa::signal_silent_store_relaxed(prof_signal->signal_, init_val);
  prof_signal->flags_.done_ = false;
  prof_signal->engine_ = engine_;
  prof_signal->flags_.isPacketDispatch_ = false;
  prof_signal->ResetCachedTiming();

  if (nullptr != cmd) {
    // Release any existing HwEvent before setting new one for the same command
    if (cmd->HwEvent() != nullptr) {
      reinterpret_cast<ProfilingSignal*>(cmd->HwEvent())->release();
    }
    cmd->SetHwEvent(prof_signal);
    prof_signal->retain();
  }

  if (ts != nullptr) {
    // Save HSA signal earlier to make sure the possible callback will have a valid
    // value for processing
    ts->retain();
    prof_signal->ts_ = ts;
    ts->AddProfilingSignal(prof_signal);
    // If an enqueue handler is requested, set up marker/callback handling that updates the batch
    // upon HSA signal completion
    if (enqueHandler) {
      uint32_t init_value = kInitSignalValueOne;
      // If API callback is enabled, then use a blocking signal for AQL queue.
      // HSA signal will be acquired in SW and released after HSA signal callback
      if (ts->command().Callback() != nullptr) {
        bool blocking = ts->command().Callback()->blocking_;
        ts->SetCallbackSignal(prof_signal->signal_, blocking);
        // Blocks AQL queue from further processing
        if (blocking) {
          Hsa::signal_add_relaxed(prof_signal->signal_, 1);
          init_value += 1;
        }
      }
      gpu_.QueuedAsyncHandlers()++;
      ts->gpu()->retain();
      hsa_status_t result = Hsa::signal_async_handler(
          prof_signal->signal_, HSA_SIGNAL_CONDITION_LT, init_value, &HsaAmdSignalHandler, ts);
      if (HSA_STATUS_SUCCESS != result) {
        gpu_.QueuedAsyncHandlers()--;
        ts->gpu()->release();
        LogError("hsa_amd_signal_async_handler() failed to set the handler!");
      } else {
        ClPrint(amd::LOG_INFO, amd::LOG_SIG,
                "Set Handler: handle(0x%lx), timestamp(%p), blocking CB=%d",
                prof_signal->signal_.handle, prof_signal,
                ts->command().Callback() != nullptr && ts->GetBlocking());
      }
    }
  }
  return prof_signal->signal_;
}

// ================================================================================================
std::vector<hsa_signal_t>& VirtualGPU::HwQueueTracker::WaitingSignal(HwQueueEngine engine) {
  bool explicit_wait = false;
  // Reset all current waiting signals
  waiting_signals_.clear();

  // Does runtime switch the active engine?
  if (engine != engine_) {
    // Yes, return the signal from the previous operation for a wait
    engine_ = engine;
    explicit_wait = true;
  } else {
    // Unknown engine in use, hence return a wait signal always
    if (engine == HwQueueEngine::Unknown) {
      explicit_wait = true;
    } else {
      // Check if skip wait optimization is enabled. It will try to predict the same engine in ROCr
      // and ignore the signal wait, relying on in-order engine execution
      const Settings& settings = gpu_.dev().settings();
      if (engine != HwQueueEngine::Compute) {
        explicit_wait = true;
      }
    }
  }

  // Check if a wait is required
  if (explicit_wait) {
    bool skip_internal_signal = false;

    for (uint32_t i = 0; i < external_signals_.size(); ++i) {
      // If external signal matches internal one, then skip it
      if (external_signals_[i]->signal_.handle == signal_list_[current_id_]->signal_.handle) {
        skip_internal_signal = true;
      }
    }
    // Add the oldest signal into the tracking for a wait
    if (!skip_internal_signal) {
      external_signals_.push_back(signal_list_[current_id_]);
    }
  }

  // Validate all signals for the wait and skip already completed
  for (uint32_t i = 0; i < external_signals_.size(); ++i) {
    // Early signal status check
    if (Hsa::signal_load_relaxed(external_signals_[i]->signal_) > 0) {
      const Settings& settings = gpu_.dev().settings();
      if (settings.cpu_wait_for_signal_) {
        // Wait on CPU for completion if requested
        CpuWaitForSignal(external_signals_[i]);
      } else {
        // Add HSA signal for tracking on GPU
        waiting_signals_.push_back(external_signals_[i]->signal_);
      }
    }
  }
  external_signals_.clear();

  // Append raw signals added via AddDynamicQueueWait (e.g. IPC dep signals)
  for (auto& s : dynamic_queue_waits_) {
    waiting_signals_.push_back(s);
  }
  dynamic_queue_waits_.clear();

  // Return the array of waiting HSA signals
  return waiting_signals_;
}

// ================================================================================================
bool VirtualGPU::HwQueueTracker::CpuWaitForSignal(ProfilingSignal* signal) {
  // Wait for the current signal to complete
  if (Hsa::signal_load_relaxed(signal->signal_) > 0) {
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "Host wait on completion_signal=0x%zx",
            signal->signal_.handle);
    if (!WaitForSignal(signal->signal_, gpu_.ActiveWait())) {
      LogPrintfError("Failed signal [0x%lx] wait", signal->signal_);
      return false;
    }
  }

  // Process this signal's timing before signal reuse
  // This copies timing to the Timestamp
  if (signal->ts_ != nullptr) {
    signal->ts_->checkGpuTime(signal);
    signal->ts_->release();
    signal->ts_ = nullptr;
  } else {
    // No timestamp - just mark signal as done
    std::scoped_lock lock(signal->LockSignalOps());
    signal->flags_.done_ = true;
  }

  return true;
}

// ================================================================================================
bool VirtualGPU::HwQueueTracker::WaitCurrent() {
  ProfilingSignal* signal = signal_list_[current_id_];
  return CpuWaitForSignal(signal);
}

// ================================================================================================
void VirtualGPU::HwQueueTracker::WaitNext() {
  size_t next = (current_id_ + 1) % signal_list_.size();
  ProfilingSignal* signal = signal_list_[next];
  // Only wait, there is no need to save timestamp for the next signal
  // It will be saved when the signal is actually used
  WaitForSignal(signal->signal_, gpu_.ActiveWait());
}

// ================================================================================================
void VirtualGPU::HwQueueTracker::ResetCurrentSignal() {
  // Reset the signal and return
  Hsa::signal_silent_store_relaxed(signal_list_[current_id_]->signal_, 0);
  // Fallback to the previous signal
  current_id_ = (current_id_ == 0) ? (signal_list_.size() - 1) : (current_id_ - 1);
}

// ================================================================================================
bool VirtualGPU::processMemObjects(const amd::Kernel& kernel, const_address params,
                                   size_t& ldsAddress, bool cooperativeGroups,
                                   bool& imageBufferWrtBack,
                                   std::vector<device::Memory*>& wrtBackImageBuffer) {
  Kernel& hsaKernel =
      const_cast<Kernel&>(static_cast<const Kernel&>(*(kernel.getDeviceKernel(dev()))));
  const amd::KernelSignature& signature = kernel.signature();
  const amd::KernelParameters& kernelParams = kernel.parameters();

  if (!cooperativeGroups && memoryDependency().maxMemObjectsInQueue() != 0) {
    // AQL packets
    setAqlHeader(dispatchPacketHeaderNoSync_);
  }

  amd::Memory* const* memories =
      reinterpret_cast<amd::Memory* const*>(params + kernelParams.memoryObjOffset());

  // HIP shouldn't use cache coherency layer at any time
  if (!amd::IS_HIP) {
    // Process cache coherency first, since the extra transfers may affect
    // other mem dependency tracking logic: TS and signalWrite()
    for (uint i = 0; i < signature.numMemories(); ++i) {
      amd::Memory* mem = memories[i];
      if (mem != nullptr) {
        roc::Memory* gpuMem = dev().getGpuMemory(mem);
        // Don't sync for internal objects, since they are not shared between devices
        if (gpuMem->owner()->getVirtualDevice() == nullptr) {
          // Synchronize data with other memory instances if necessary
          gpuMem->syncCacheFromHost(*this);
        }
      }
    }
  }

  // Mark the tracker with a new kernel, so it can avoid checks of the aliased objects
  memoryDependency().newKernel();

  bool deviceSupportFGS = 0 != dev().isFineGrainedSystem(true);
  bool supportFineGrainedSystem = deviceSupportFGS;
  FGSStatus status = kernelParams.getSvmSystemPointersSupport();
  switch (status) {
    case FGS_YES:
      if (!deviceSupportFGS) {
        return false;
      }
      supportFineGrainedSystem = true;
      break;
    case FGS_NO:
      supportFineGrainedSystem = false;
      break;
    case FGS_DEFAULT:
    default:
      break;
  }

  size_t count = kernelParams.getNumberOfSvmPtr();
  size_t execInfoOffset = kernelParams.getTotalSize();
  bool sync = true;

  amd::Memory* memory = nullptr;
  // get svm non arugment information
  void* const* svmPtrArray = reinterpret_cast<void* const*>(params + execInfoOffset);
  for (size_t i = 0; i < count; i++) {
    memory = amd::MemObjMap::FindMemObj(svmPtrArray[i]);
    if (nullptr == memory) {
      if (!supportFineGrainedSystem) {
        return false;
      } else if (sync) {
        // Sync AQL packets
        setAqlHeader(dispatchPacketHeader_);
        // Clear memory dependency state
        const static bool All = true;
        memoryDependency().clear(!All);
        continue;
      }
    } else {
      Memory* rocMemory = static_cast<Memory*>(memory->getDeviceMemory(dev()));
      if (nullptr != rocMemory) {
        // Synchronize data with other memory instances if necessary
        rocMemory->syncCacheFromHost(*this);

        const static bool IsReadOnly = false;
        // Validate SVM passed in the non argument list
        memoryDependency().validate(*this, rocMemory, IsReadOnly);
      } else {
        return false;
      }
    }
  }

  // Check all parameters for the current kernel
  for (size_t i = 0; i < signature.numParameters(); ++i) {
    const amd::KernelParameterDescriptor& desc = signature.at(i);
    Memory* gpuMem = nullptr;
    amd::Memory* mem = nullptr;

    // Find if current argument is a buffer
    if (desc.type_ == T_POINTER) {
      if (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_LOCAL) {
        // Align the LDS on the alignment requirement of type pointed to
        ldsAddress = amd::alignUp(ldsAddress, desc.info_.arrayIndex_);
        if (desc.size_ == 8) {
          // Save the original LDS size
          uint64_t ldsSize = *reinterpret_cast<const uint64_t*>(params + desc.offset_);
          // Patch the LDS address in the original arguments with an LDS address(offset)
          WriteAqlArgAt(const_cast<address>(params), ldsAddress, desc.size_, desc.offset_);
          // Add the original size
          ldsAddress += ldsSize;
        } else {
          // Save the original LDS size
          uint32_t ldsSize = *reinterpret_cast<const uint32_t*>(params + desc.offset_);
          // Patch the LDS address in the original arguments with an LDS address(offset)
          uint32_t ldsAddr = ldsAddress;
          WriteAqlArgAt(const_cast<address>(params), ldsAddr, desc.size_, desc.offset_);
          // Add the original size
          ldsAddress += ldsSize;
        }
      } else {
        uint32_t index = desc.info_.arrayIndex_;
        mem = memories[index];
        const void* globalAddress = *reinterpret_cast<const void* const*>(params + desc.offset_);
        if (mem == nullptr) {
          ClPrint(amd::LOG_DEBUG, amd::LOG_KERN, "Arg%d: %s %s = ptr:%p ", i, desc.typeName_.c_str(),
                  desc.name_.c_str(), globalAddress);
          //! This condition is for SVM fine-grain
          if (dev().isFineGrainedSystem(true)) {
            // Sync AQL packets
            setAqlHeader(dispatchPacketHeader_);
            // Clear memory dependency state
            const static bool All = true;
            memoryDependency().clear(!All);
          }
        } else {
          gpuMem = static_cast<Memory*>(mem->getDeviceMemory(dev()));

          const void* globalAddress = *reinterpret_cast<const void* const*>(params + desc.offset_);
          ClPrint(amd::LOG_DEBUG, amd::LOG_KERN, "Arg%d: %s %s = ptr:%p obj:[%p-%p]", i,
                  desc.typeName_.c_str(), desc.name_.c_str(), globalAddress,
                  gpuMem->getDeviceMemory(),
                  reinterpret_cast<address>(gpuMem->getDeviceMemory()) + mem->getSize());

          // Validate memory for a dependency in the queue
          memoryDependency().validate(*this, gpuMem, (desc.info_.readOnly_ == 1));

          assert((desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_GLOBAL ||
                  desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_CONSTANT) &&
                 "Unsupported address qualifier");

          const bool readOnly = (desc.typeQualifier_ == CL_KERNEL_ARG_TYPE_CONST) ||
                                ((mem->getMemFlags() & CL_MEM_READ_ONLY) != 0);

          if (!readOnly) {
            mem->signalWrite(&dev());
          }

          if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) {
            Image* image = static_cast<Image*>(mem->getDeviceMemory(dev()));

            const uint64_t image_srd = image->getHsaImageObject().handle;
            assert(amd::isMultipleOf(image_srd, sizeof(image_srd)));
            WriteAqlArgAt(const_cast<address>(params), image_srd, sizeof(image_srd), desc.offset_);

            // Check if synchronization has to be performed
            if (image->CopyImageBuffer() != nullptr) {
              Memory* devBuf = dev().getGpuMemory(mem->parent());
              amd::Coord3D offs(0);
              Image* devCpImg = static_cast<Image*>(dev().getGpuMemory(image->CopyImageBuffer()));
              amd::Image* img = mem->asImage();

              // Copy memory from the original image buffer into the backing store image
              bool result =
                  blitMgr().copyBufferToImage(*devBuf, *devCpImg, offs, offs, img->getRegion(),
                                              true, img->getRowPitch(), img->getSlicePitch());
              // Make sure the copy operation is done
              setAqlHeader(dispatchPacketHeader_);
              // Use backing store SRD as the replacment
              const uint64_t srd = devCpImg->getHsaImageObject().handle;
              WriteAqlArgAt(const_cast<address>(params), srd, sizeof(srd), desc.offset_);

              // If it's not a read only resource, then runtime has to write back
              if (!desc.info_.readOnly_) {
                wrtBackImageBuffer.push_back(mem->getDeviceMemory(dev()));
                imageBufferWrtBack = true;
              }
            }
          }
        }
      }
    } else if (desc.type_ == T_QUEUE) {
      uint32_t index = desc.info_.arrayIndex_;
      const amd::DeviceQueue* queue =
          reinterpret_cast<amd::DeviceQueue* const*>(params + kernelParams.queueObjOffset())[index];

      if (!createVirtualQueue(queue->size()) || !createSchedulerParam()) {
        return false;
      }
      uint64_t vqVA = getVQVirtualAddress();
      WriteAqlArgAt(const_cast<address>(params), vqVA, sizeof(vqVA), desc.offset_);
    } else if (desc.type_ == T_VOID) {
      const_address srcArgPtr = params + desc.offset_;
      if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ReferenceObject) {
        void* mem = allocKernArg(desc.size_, 128);
        memcpy(mem, srcArgPtr, desc.size_);
        const auto it = hsaKernel.patch().find(desc.offset_);
        WriteAqlArgAt(const_cast<address>(params), mem, sizeof(void*), it->second);
      }

      if (IsLogEnabled(amd::LOG_INFO, amd::LOG_KERN)) {
        if (desc.size_ > 8) {
          std::string bytes = "0x";
          constexpr size_t kMaxBytes = 64;
          for (size_t j = 0; j < std::min(desc.size_, kMaxBytes); j++) {
            char byteStr[4];
            snprintf(byteStr, sizeof(byteStr), "%02x ",
                     reinterpret_cast<const uint8_t*>(srcArgPtr)[j]);
            bytes += byteStr;
          }
          if (desc.size_ > kMaxBytes) {
            bytes += "...";
          }
          ClPrint(amd::LOG_DEBUG, amd::LOG_KERN, "Arg%d: %s %s = %s (size:0x%x)", i,
                  desc.typeName_.c_str(), desc.name_.c_str(), bytes.c_str(), desc.size_);
        } else {
          ClPrint(amd::LOG_DEBUG, amd::LOG_KERN, "Arg%d: %s %s = val:0x%lx (size:0x%x)", i,
                  desc.typeName_.c_str(), desc.name_.c_str(),
                  (desc.size_ == 1)   ? *reinterpret_cast<const uint8_t*>(srcArgPtr)
                  : (desc.size_ == 2) ? *reinterpret_cast<const uint16_t*>(srcArgPtr)
                  : (desc.size_ == 4) ? *reinterpret_cast<const uint32_t*>(srcArgPtr)
                  : (desc.size_ == 8) ? *reinterpret_cast<const uint64_t*>(srcArgPtr)
                                      : 0LL,
                  desc.size_);
        }
      }
    } else if (desc.type_ == T_SAMPLER) {
      uint32_t index = desc.info_.arrayIndex_;
      const amd::Sampler* sampler =
          reinterpret_cast<amd::Sampler* const*>(params + kernelParams.samplerObjOffset())[index];

      device::Sampler* devSampler = sampler->getDeviceSampler(dev());

      uint64_t sampler_srd = devSampler->hwSrd();
      WriteAqlArgAt(const_cast<address>(params), sampler_srd, sizeof(sampler_srd), desc.offset_);
    }
  }

  if (hsaKernel.program()->hasGlobalStores()) {
    // Sync AQL packets
    setAqlHeader(dispatchPacketHeader_);
    // Clear memory dependency state
    const static bool All = true;
    memoryDependency().clear(!All);
  }

  return true;
}

// ================================================================================================
void VirtualGPU::SetGpuQueue(hsa_queue_t* queue, void* metadata_ring_buffer) {
  gpu_queue_ = queue;
  metadata_preloader_.SetQueueBase(metadata_ring_buffer,
                                   roc_device_.MetadataVersionHeader());
}

// ================================================================================================
void VirtualGPU::AcquireQueueWithPreference() {
  std::scoped_lock lock(execution());
  if (!dedicated_queue_ && gpu_queue_ == nullptr && last_hwq_ != nullptr) {
    void* md_rb = nullptr;
    SetGpuQueue(roc_device_.AcquireActiveQueue(priority_, last_hwq_, nullptr, &md_rb), md_rb);
    last_hwq_ = nullptr;
  }
}

// ================================================================================================
bool VirtualGPU::ReacquireQueueExcluding(const std::unordered_set<uint64_t>& excluded_ids) {
  std::scoped_lock lock(execution());
  if (gpu_queue_ != nullptr) {
    // Detach from the current queue: decrements refCount in the pool but never
    // destroys the queue.
    // Unlike ReleaseActiveQueue, this is unconditional — we are switching queues,
    // not conditionally reclaiming under pressure.
    roc_device_.releaseQueue(gpu_queue_, std::vector<uint32_t>{}, false, true);
    gpu_queue_ = nullptr;
  }
  void* md_rb = nullptr;
  SetGpuQueue(roc_device_.AcquireActiveQueue(priority_, nullptr, &excluded_ids, &md_rb), md_rb);
  return gpu_queue_ != nullptr;
}

// ================================================================================================
uint64_t VirtualGPU::getQueueID() {
  std::scoped_lock lock(execution());
  // Dedicated queues keep their HW queue, never acquire from pool
  if (!dedicated_queue_ && gpu_queue_ == nullptr) {
    void* md_rb = nullptr;
    SetGpuQueue(roc_device_.AcquireActiveQueue(priority_, nullptr, nullptr, &md_rb), md_rb);
  }
  return gpu_queue_->id;
}

// ================================================================================================
static inline void packet_store_release(uint32_t* packet, uint16_t header, uint16_t rest) {
#if IS_WINDOWS
  std::atomic_ref<uint32_t> atomic_header(*packet);
  atomic_header.store(header | (rest << 16), std::memory_order_release);
#else
  __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
#endif
}

// ================================================================================================
std::string VirtualGPU::AnalyzeAqlQueue() const {
  std::string kernelName = "<not identified>";
  const uint32_t queueSize = gpu_queue_->size;
  const uint32_t queueMask = queueSize - 1;
  uint64_t index = Hsa::queue_load_write_index_relaxed(gpu_queue_);
  uint64_t read = Hsa::queue_load_read_index_relaxed(gpu_queue_);

  if (index > read) {
    int valid_packet_idx = 0;
    constexpr int kAqlSearchWindow = 32;
    while (valid_packet_idx < kAqlSearchWindow) {
      auto aql_loc = &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
          gpu_queue_->base_address))[(read + valid_packet_idx) & queueMask];
      if (extractAqlBits((*aql_loc).header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE) ==
          HSA_PACKET_TYPE_INVALID) {
        valid_packet_idx++;
      } else {
        break;
      }
    }
    if (valid_packet_idx == kAqlSearchWindow) {
      fprintf(stderr, "VGPU(%p) Queue(%p). Couldn't find the hang AQL packet!\n", this, gpu_queue_);
      return kernelName;
    }
    auto aql_loc = &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
        gpu_queue_->base_address))[(read + valid_packet_idx) & queueMask];
    auto packet = *aql_loc;
    auto header = packet.header;
    auto pkt_type = extractAqlBits(header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);

    auto printKernelName = [&](uint64_t kernel_object) {
      auto it = dev().KernelMap().find(kernel_object);
      if (it != dev().KernelMap().end()) {
        kernelName = it->second.getDemangledName();
      } else {
        fprintf(stderr, "VGPU(%p) Queue(%p). Couldn't find kernel\n", this, gpu_queue_);
      }
    };

    auto printHeader = [&](const char* label) {
      fprintf(stderr, "VGPU=%p SWq=%p, HWq=%p, id=%" PRIu64 "\n\t%s Header ="
             "0x%x (type=%d, barrier=%d, acquire=%d, release=%d), ",
             this, gpu_queue_, gpu_queue_->base_address, gpu_queue_->id, label, header,
             extractAqlBits(header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE),
             extractAqlBits(header, HSA_PACKET_HEADER_BARRIER, HSA_PACKET_HEADER_WIDTH_BARRIER),
             extractAqlBits(header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                            HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
             extractAqlBits(header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                            HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE));
    };

    if (pkt_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC) {
      auto* vendor_hdr = reinterpret_cast<const hsa_amd_vendor_packet_header_t*>(aql_loc);
      if (vendor_hdr->AmdFormat == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH) {
        auto* ext = reinterpret_cast<const hsa_amd_ext_kernel_dispatch_packet_t*>(aql_loc);
        printKernelName(ext->kernel_object);
        printHeader("Ext Dispatch");
        fprintf(stderr,
               "amd_format=%d, setup=%d\n\tcluster_count=[%u, %u, %u], "
               "cluster_size=[%u, %u, %u], workgroup=[%u, %u, %u]\n\t"
               "private_seg_size=%u, group_seg_size=%u\n\t"
               "kernel_obj=0x%" PRIx64 ", kernarg_address=0x%p\n\t"
               "dep_signal=0x%" PRIx64 ", completion_signal=0x%" PRIx64
               "\n\trptr=%" PRIu64 ", wptr=%" PRIu64 "\n",
               (int)ext->amd_format, ext->setup,
               ext->cluster_count_x, ext->cluster_count_y, ext->cluster_count_z,
               ext->cluster_size_x, ext->cluster_size_y, ext->cluster_size_z,
               ext->workgroup_size_x, ext->workgroup_size_y, ext->workgroup_size_z,
               ext->private_segment_size, ext->group_segment_size,
               ext->kernel_object, ext->kernarg_address,
               ext->dep_signal.handle, ext->completion_signal.handle, read, index);
      } else {
        fprintf(stderr, "VGPU(%p) Queue(%p) rptr=%" PRIu64 ", wptr=%" PRIu64
               ". Vendor packet (amd_format=%d)\n",
               this, gpu_queue_, read, index, (int)vendor_hdr->AmdFormat);
      }
    } else if (pkt_type == HSA_PACKET_TYPE_KERNEL_DISPATCH ||
               (index == read && packet.kernel_object != 0)) {
      printKernelName(packet.kernel_object);
      printHeader("Dispatch");
      fprintf(stderr,
             "setup=%d\n\tgrid=[%u, %u, %u], workgroup=[%u, %u, %u]\n\t"
             "private_seg_size=%u, group_seg_size=%u\n\t"
             "kernel_obj=0x%" PRIx64 ", kernarg_address=0x%p\n\t"
             "completion_signal=0x%" PRIx64 ", correlation_id=%" PRIu64
             "\n\trptr=%" PRIu64 ", wptr=%" PRIu64 "\n",
             packet.setup, packet.grid_size_x, packet.grid_size_y, packet.grid_size_z,
             packet.workgroup_size_x, packet.workgroup_size_y, packet.workgroup_size_z,
             packet.private_segment_size, packet.group_segment_size,
             packet.kernel_object, packet.kernarg_address,
             packet.completion_signal.handle, packet.reserved2, read, index);
    } else {
      fprintf(stderr, "VGPU(%p) Queue(%p) rptr=%" PRIu64 ", wptr=%" PRIu64
             ". A barrier packet in the queue!\n",
             this, gpu_queue_, read, index);
    }
  } else {
    fprintf(stderr, "VGPU(%p) Queue(%p) is idle\n", this, gpu_queue_);
  }
  return kernelName;
}

// ================================================================================================
template <typename AqlPacket>
bool VirtualGPU::dispatchGenericAqlPacket(AqlPacket* packet, uint16_t header, uint16_t rest,
                                          bool blocking, bool attach_signal, bool cluster_launch) {
  const uint32_t queueSize = gpu_queue_->size;
  const uint32_t queueMask = queueSize - 1;
  const uint32_t sw_queue_size = queueMask;

  // Check for queue full and wait if needed.
  uint64_t index = Hsa::queue_add_write_index_screlease(gpu_queue_, 1);
  setFenceDirty(true);

  if (addSystemScope_) {
    header &= ~(HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE |
                HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE |
               HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    addSystemScope_ = false;
  }

  auto expected_fence_state = extractAqlBits(header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                                             HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);

  // Reset fence_dirty_ flag if we submit a packet with system scopes
  if (expected_fence_state == amd::Device::kCacheStateSystem) {
    setFenceDirty(false);
  }

  // Dirty optimization to save on consequent dispatch packets which have requested flushes
  if (fence_state_ == amd::Device::kCacheStateSystem &&
      expected_fence_state == amd::Device::kCacheStateSystem) {
    header = dispatchPacketHeader_;
    setFenceDirty(true);
  }

  fence_state_ = static_cast<Device::CacheState>(expected_fence_state);

  bool attachSignal = timestamp_ != nullptr || attach_signal;
  // Get active signal for current dispatch if profiling is necessary
  packet->completion_signal =
      Barriers().ActiveSignal(kInitSignalValueOne, timestamp_, attachSignal);

  if (std::is_same<decltype(packet), hsa_kernel_dispatch_packet_t*>::value &&
      timestamp_ != nullptr) {
    // If profiling is enabled, store the correlation ID in the dispatch packet. The profiler can
    // retrieve this correlation ID to attribute waves to specific dispatch locations.
    if (amd::activity_prof::IsEnabled(OP_ID_DISPATCH) && !dev().settings().ext_dispatch_packet_) {
      auto dispatchPacket = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet);
      dispatchPacket->reserved2 = timestamp_->command().profilingInfo().correlation_id_;
    }

    ProfilingSignal* current_signal = Barriers().GetLastSignal();
    current_signal->flags_.isPacketDispatch_ = true;
  }

  // Make sure the slot is free for usage
  while ((index - Hsa::queue_load_read_index_scacquire(gpu_queue_)) >= sw_queue_size) {
    amd::Os::yield();
  }

  // Add blocking command if the original value of read index was behind of the queue size.
  // Note: direct dispatch relies on the slot stall above to keep the forward progress
  // of the app if a dispatched kernel requires some CPU input for completion
  if (blocking) {
    if (packet->completion_signal.handle == 0) {
      packet->completion_signal = Barriers().ActiveSignal();
    }
    blocking = true;
  }

  TrackQueueProgress(*packet, index);

  AqlPacket* aql_loc = &((AqlPacket*)(gpu_queue_->base_address))[index & queueMask];
  *aql_loc = *packet;

  metadata_preloader_.Set(packet, header, index & queueMask);
  if (header != 0) {
    packet_store_release(reinterpret_cast<uint32_t*>(aql_loc), header, rest);
  }
  const auto virtual_pipe_prefix = IsLogEnabled(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL)
                                     ? [this]() -> const char* {
                                         if (!roc_device_.settings().queue_pipe_dist_) return "";
                                         static thread_local char buf[32];
                                         snprintf(buf, sizeof(buf), " virtual_pipe_id=%zu,",
                                                  gpu_queue_->id % roc_device_.NumHwPipes());
                                         return buf;
                                       }()
                                     : "";
  if (dev().settings().ext_dispatch_packet_) {
    logAqlDispatchPacketExtended(
        gpu_queue_, header, reinterpret_cast<hsa_amd_ext_kernel_dispatch_packet_t*>(packet),
        Hsa::queue_load_read_index_scacquire(gpu_queue_), index, virtual_pipe_prefix);
  } else {
    logAqlDispatchPacket(gpu_queue_, header,
                         reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet),
                         Hsa::queue_load_read_index_scacquire(gpu_queue_), index,
                         virtual_pipe_prefix);
  }
  // Optimization for native AQL path in Windows has problems with PM4 emulation,
  // skipping the doorbell will not wake up the AQL worker thread
  uint32_t skip_limit = DEBUG_CLR_DOORBELL_SKIP;
  bool ring_for_non_profiler_signal = attach_signal && (packet->completion_signal.handle != 0);
  bool ring_doorbell = IS_LINUX || dev().IsPm4Emulation() || blocking ||
                       (skippedDispatches_ >= skip_limit) || ring_for_non_profiler_signal;
  if (ring_doorbell) {
    Hsa::signal_store_screlease(gpu_queue_->doorbell_signal, index);
    skippedDispatches_ = 0;
  } else {
    ++skippedDispatches_;
  }

  // Mark the flag indicating if a dispatch is outstanding.
  // We are not waiting after every dispatch.
  hasPendingDispatch_ = true;

  // Wait on signal ?
  if (blocking) {
    LogInfo("Runtime reached the AQL queue limit. SW is much ahead of HW. Blocking AQL queue!");
    if (!Barriers().WaitCurrent()) {
      LogPrintfError("Failed blocking queue wait with signal [0x%lx]",
                     packet->completion_signal.handle);
      return false;
    }
  }

  return true;
}

// ================================================================================================
void VirtualGPU::dispatchBlockingWait(hsa_kernel_dispatch_packet_t* packet) {
  auto wait_signals = Barriers().WaitingSignal();
  if (dev().settings().ext_dispatch_packet_ && wait_signals.size() == 1 && packet != nullptr) {
      // The Ext Dispatch Packet supports only one dependent signal
      auto ext_packet = reinterpret_cast<hsa_amd_ext_kernel_dispatch_packet_t*>(packet);
      ext_packet->dep_signal = wait_signals[0];
  } else {
    // AQL dispatch doesn't support dependent signals and extra barrier packet must be generated
    for (uint32_t i = 0; i < wait_signals.size(); ++i) {
      uint32_t j = i % 5;
      barrier_packet_.dep_signal[j] = wait_signals[i];
      constexpr bool kSkipSignal = true;
      // If runtime reached the packet limit or the count limit, then flush the barrier
      if ((j == 4) || ((i + 1) == wait_signals.size())) {
        dispatchBarrierPacket(kNopPacketHeader, kSkipSignal);
      }
    }
  }
}

// ================================================================================================
bool VirtualGPU::dispatchAqlPacket(hsa_kernel_dispatch_packet_t* packet, uint16_t header,
                                   uint16_t rest, bool blocking, bool capturing,
                                   const uint8_t* aqlPacket, bool attach_signal) {
  if (capturing == true) {
    if (dev().settings().ext_dispatch_packet_) {
      // For ext dispatch packets the first 32 bits are {header(16), amd_format(8), setup(8)}.
      // rest already encodes (amd_format | (setup << 8)).
      auto* ext_packet = reinterpret_cast<hsa_amd_ext_kernel_dispatch_packet_t*>(packet);
      ext_packet->header = header;
      ext_packet->amd_format =
          static_cast<hsa_amd_packet_type8_t>(rest & 0xFF);
      ext_packet->setup = static_cast<uint8_t>((rest >> 8) & 0xFF);
    } else {
      packet->header = header;
      packet->setup = rest;
    }
    std::memcpy(const_cast<uint8_t*>(aqlPacket), packet, sizeof(hsa_kernel_dispatch_packet_t));
    return true;
  } else {
    dispatchBlockingWait(packet);
    return dispatchGenericAqlPacket(packet, header, rest, blocking, attach_signal);
  }
}
// ================================================================================================
bool VirtualGPU::dispatchAqlPacket(hsa_barrier_and_packet_t* packet, uint16_t header, uint16_t rest,
                                   bool blocking, bool attach_signal) {
  return dispatchGenericAqlPacket(packet, header, rest, blocking, attach_signal);
}

// ================================================================================================
// Unified flat-buffer graph dispatch.  Reserves all N queue slots with a single wptr bump,
// then processes them in kPeriod-sized chunks: yield-until-free → memcpy → per-packet fixups
// (profiling signals, correlation IDs, kernel-name printing) → valid-header writes → doorbell.
// For graphs that fit in the queue the yield never fires.  For oversized graphs the GPU drains
// earlier chunks while the CPU copies later ones, avoiding a deadlock.  Packet-0's header is
// committed last in the first chunk per the AQL protocol.
//
// recordedPacketVersion / pm4Template carry the optional PM4-IB graph-replay fast
// path (HIP_PM4_GRAPH): when active the whole captured dispatch chain replays as one
// vendor PM4 IB instead of N AQL packets, falling through to the flat AQL path below
// when PM4 replay is off or the graph is not PM4-replayable.
bool VirtualGPU::dispatchAqlPacketBatchFlat(const std::vector<uint8_t>& flatPacketData,
                                            const std::vector<uint32_t>& validFullHeaders,
                                            amd::AccumulateCommand* vcmd, bool attach_signal,
                                            const std::vector<const std::string*>* kernelNames,
                                            bool pre_patched, bool blocking,
                                            uint64_t recordedPacketVersion,
                                            const void* pm4Template) {
  if (vcmd == nullptr || flatPacketData.empty() || validFullHeaders.empty()) {
    return false;
  }
  const size_t numPackets = validFullHeaders.size();
  if (flatPacketData.size() != numPackets * sizeof(hsa_kernel_dispatch_packet_t)) {
    return false;
  }

  // EXPERIMENTAL: replay an all-dispatch captured graph as one PM4 IB (single CP
  // jump, lean raw-PM4 dispatches + in-place PWS fences). kernelNames != nullptr
  // marks the graph-replay path. Falls back to the flat AQL replay below if any
  // packet is unsupported (non-dispatch, scratch, or non-trivial kernarg ABI).
  //
  // The flat buffer stores 64-byte packet bodies with the 2-byte AQL header
  // invalidated (the real header is in validFullHeaders) and the completion signal
  // zeroed. The queue-independent PM4 encoder/replayer needs valid dispatch packets,
  // so restore the headers into a local copy and pass an array of packet pointers.
  //
  // HIP_PM4_GRAPH_BUILD_AFTER: on the FIRST replay (cache miss) do NOT build the IB
  // before submit (allowBuild=false) -- replay via AQL below and build right after
  // the submit (prebuildPm4Graph), so the build overlaps the GPU executing the
  // graph just submitted, and the AQL submit sizes the scratch the build needs.
  const auto* tmpl = static_cast<const Pm4GraphTemplate*>(pm4Template);
  const bool pm4GraphReplay = pm4GraphActive() && kernelNames != nullptr;
  const bool pm4BuildAfter = pm4GraphReplay && pm4GraphBuildAfterEnabled();
  if (pm4GraphReplay) {
    constexpr size_t kAqlBytes = sizeof(hsa_kernel_dispatch_packet_t);
    std::vector<uint8_t> hdrRestored(flatPacketData);
    std::vector<void*> pktPtrs(numPackets);
    for (size_t i = 0; i < numPackets; ++i) {
      const uint16_t hdr = static_cast<uint16_t>(validFullHeaders[i]);
      memcpy(hdrRestored.data() + i * kAqlBytes, &hdr, sizeof(hdr));
      pktPtrs[i] = hdrRestored.data() + i * kAqlBytes;
    }
    // Hand the per-packet kernel names to the (instrumented) replay so the per-kernel
    // timestamp log can print the shader name; aligned 1:1 with the dispatch packets.
    pm4LaunchKernelNames_ = kernelNames;
    if (tryReplayPm4Graph(pktPtrs.data(), numPackets, blocking, attach_signal,
                          recordedPacketVersion, /*allowBuild=*/!pm4BuildAfter, tmpl)) {
      return true;
    }
  }

  std::scoped_lock lock(execution());
  profilingBegin(*vcmd);
  dispatchBlockingWait(nullptr);

  if (kernelNames != nullptr) {
    vcmd->setKernelNamesRef(kernelNames);
  }

  const uint32_t queueSize = gpu_queue_->size;
  const uint32_t queueMask = queueSize - 1;
  const uint32_t sw_queue_size = queueMask;
  static constexpr size_t kPacketSize = sizeof(hsa_kernel_dispatch_packet_t);

  // Unpack first/last headers; apply system-scope once for the whole batch.
  // validFullHeaders stores the AQL full_header dword: low 16 = header, high 16 = setup.
  uint16_t firstHeader = static_cast<uint16_t>(validFullHeaders[0]);
  uint16_t firstSetup  = static_cast<uint16_t>(validFullHeaders[0] >> 16);
  uint16_t lastHeader  = static_cast<uint16_t>(validFullHeaders[numPackets - 1]);
  if (addSystemScope_) {
    firstHeader &= ~(HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    firstHeader |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    lastHeader &= ~(HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    lastHeader |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    addSystemScope_ = false;
  }

  uint8_t* queueBase = static_cast<uint8_t*>(gpu_queue_->base_address);

  // Reserve ALL slots with a single wptr bump, then submit in kPeriod-sized chunks.
  // Per-chunk: yield if the queue is full (handles graphs larger than the queue), then
  // memcpy + per-packet fixups + headers + doorbell.  For graphs that fit in the queue
  // the yield never fires.
  uint64_t startIndex = Hsa::queue_add_write_index_screlease(gpu_queue_, numPackets);
  setFenceDirty(true);

  // Update cached fence state from the last packet's release scope.
  // Clear fence dirty if the last packet has system-scope release, matching
  // the single-dispatch path in dispatchGenericAqlPacket (set dirty on reserve,
  // then conditionally clear if system scope).
  auto expected_fence_state =
      extractAqlBits(lastHeader, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                     HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);
  if (expected_fence_state == amd::Device::kCacheStateSystem) {
    setFenceDirty(false);
  }
  fence_state_ = static_cast<Device::CacheState>(expected_fence_state);

  const size_t kPeriod = DEBUG_HIP_GRAPH_BATCH_SIZE;
  auto* first_loc = reinterpret_cast<uint32_t*>(
      queueBase + (startIndex & queueMask) * kPacketSize);

  for (size_t chunkStart = 0; chunkStart < numPackets; ) {
    const size_t chunkEnd  = std::min(chunkStart + kPeriod, numPackets);
    const size_t thisChunk = chunkEnd - chunkStart;
    const bool isFirstChunk = (chunkStart == 0);
    const bool isLastChunk  = (chunkEnd == numPackets);

    // Yield until this chunk's physical slots are free.
    while (((startIndex + chunkEnd - 1) - Hsa::queue_load_read_index_scacquire(gpu_queue_)) >=
           sw_queue_size) {
      amd::Os::yield();
    }

    // Copy this chunk's packet bodies to the queue. Handles RB wrap-around.
    const size_t chunkSlot = (startIndex + chunkStart) & queueMask;
    const uint8_t* srcData = flatPacketData.data() + chunkStart * kPacketSize;
    if (chunkSlot + thisChunk <= queueSize) {
      memcpy(queueBase + chunkSlot * kPacketSize, srcData, thisChunk * kPacketSize);
    } else {
      const size_t firstCount = queueSize - chunkSlot;
      memcpy(queueBase + chunkSlot * kPacketSize, srcData, firstCount * kPacketSize);
      memcpy(queueBase, srcData + firstCount * kPacketSize,
             (thisChunk - firstCount) * kPacketSize);
    }

    // Attach signal to the last packet when requested (before per-packet logging).
    auto* lastSlotPtr = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
        queueBase + ((startIndex + chunkEnd - 1) & queueMask) * kPacketSize);
    if (isLastChunk && (attach_signal || blocking) && timestamp_ == nullptr) {
      lastSlotPtr->completion_signal = Barriers().ActiveSignal();
    }

    // Per-packet fixups: profiling signals, kernel-name printing, and inline barrier logging.
    if (timestamp_ != nullptr || IsLogEnabled(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN2) ||
        IsLogEnabled(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL)) {
      for (size_t i = chunkStart; i < chunkEnd; ++i) {
        const uint64_t slotIdx = (startIndex + i) & queueMask;
        auto* slot = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
            queueBase + slotIdx * kPacketSize);
        const uint16_t hdr = static_cast<uint16_t>(validFullHeaders[i]);
        const uint8_t pktType =
            extractAqlBits(hdr, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
        if (timestamp_ != nullptr) {
          // When pre_patched, skip any slot whose completion_signal was already
          // written by ApplyHwEventPatches (non-zero means pre-patched).
          bool has_prepatched_signal = pre_patched && (slot->completion_signal.handle != 0);
          if (!has_prepatched_signal) {
            slot->completion_signal =
                Barriers().ActiveSignal(kInitSignalValueOne, timestamp_, true);
            if (pktType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
              if (amd::activity_prof::IsEnabled(OP_ID_DISPATCH)) {
                slot->reserved2 = timestamp_->command().profilingInfo().correlation_id_;
              }
              Barriers().GetLastSignal()->flags_.isPacketDispatch_ = true;
            }
          } else if (has_prepatched_signal &&
                     pktType == HSA_PACKET_TYPE_KERNEL_DISPATCH &&
                     amd::activity_prof::IsEnabled(OP_ID_DISPATCH)) {
            slot->reserved2 = timestamp_->command().profilingInfo().correlation_id_;
          }
        }
        if ((IsLogEnabled(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN2) ||
             IsLogEnabled(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL)) &&
            kernelNames != nullptr && i < kernelNames->size() &&
            pktType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
          ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN2, "Graph ShaderName : %s, device id : %u",
                  (*kernelNames)[i]->c_str(), dev().index());
          ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL,
                  "SWq=0x%zx, HWq=0x%zx, id=%d, Dispatch Header = "
                  "0x%x (type=%d, barrier=%d, acquire=%d, release=%d), "
                  "setup=%d, grid=[%u, %u, %u], workgroup=[%u, %u, %u], "
                  "private_seg_size=%u, group_seg_size=%u, kernel_obj=0x%zx, "
                  "kernarg_address=0x%zx, completion_signal=0x%zx, correlation_id=%zu, "
                  "rptr=%u, wptr=%u",
                  gpu_queue_, gpu_queue_->base_address, gpu_queue_->id, hdr, pktType,
                  extractAqlBits(hdr, HSA_PACKET_HEADER_BARRIER,
                                 HSA_PACKET_HEADER_WIDTH_BARRIER),
                  extractAqlBits(hdr, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                                 HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
                  extractAqlBits(hdr, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                                 HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE),
                  slot->setup,
                  slot->grid_size_x, slot->grid_size_y, slot->grid_size_z,
                  slot->workgroup_size_x, slot->workgroup_size_y, slot->workgroup_size_z,
                  slot->private_segment_size, slot->group_segment_size,
                  slot->kernel_object, slot->kernarg_address,
                  slot->completion_signal, slot->reserved2,
                  Hsa::queue_load_read_index_scacquire(gpu_queue_), slotIdx);
        } else if ((IsLogEnabled(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN2) ||
                    IsLogEnabled(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL)) &&
                   (pktType == HSA_PACKET_TYPE_BARRIER_AND ||
                    pktType == HSA_PACKET_TYPE_BARRIER_OR)) {
          // Inline barriers placed in the batch by BuildSyncPlan never go
          // through dispatchBarrierPacket, so log them here. Classify by
          // patched fields: dep_signal set -> cross-dep barrier, else
          // completion_signal set -> per-segment completion barrier.
          const auto* bpkt = reinterpret_cast<const hsa_barrier_and_packet_t*>(slot);
          bool has_dep = false;
          for (int k = 0; k < 5 && !has_dep; ++k) {
            if (bpkt->dep_signal[k].handle != 0) has_dep = true;
          }
          const char* tag = has_dep
              ? " [Graph cross dep barrier]"
              : (bpkt->completion_signal.handle != 0 ? " [Graph completion barrier]"
                                                     : " [Graph batch barrier]");
          ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN2,
                  "Graph ShaderName :%s, device id : %u", tag, dev().index());
          logAqlBarrierPacket(gpu_queue_, hdr, bpkt,
                              Hsa::queue_load_read_index_scacquire(gpu_queue_), slotIdx, tag);
        }
      }
    }

    // Write valid headers and ring the doorbell for this chunk.
    // Hold global packet-0's header back in the first chunk so the GPU sees a
    // fully-committed batch before starting (AQL protocol).  Subsequent chunks
    // write in forward order — packet 0 is already committed.
    for (size_t i = (isFirstChunk ? 1 : chunkStart); i < chunkEnd; ++i) {
      const uint64_t idx = startIndex + i;
      auto* aql_loc =
          reinterpret_cast<uint32_t*>(queueBase + (idx & queueMask) * kPacketSize);
      const uint32_t dword = validFullHeaders[i];
      const uint16_t hdr = (i == numPackets - 1) ? lastHeader : static_cast<uint16_t>(dword);
      packet_store_release(aql_loc, hdr, static_cast<uint16_t>(dword >> 16));
    }
    if (isFirstChunk) {
      packet_store_release(first_loc, firstHeader, firstSetup);
    }
    Hsa::signal_store_screlease(gpu_queue_->doorbell_signal, startIndex + chunkEnd - 1);

    chunkStart = chunkEnd;
  }

  hasPendingDispatch_ = true;

  auto* finalLastSlot = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
      queueBase + ((startIndex + numPackets - 1) & queueMask) * kPacketSize);

  // Skip the pending dispatch only when both conditions are met: a completion
  // signal tracks the last packet and the fence is already clean (system scope).
  if (finalLastSlot->completion_signal.handle != 0 && !isFenceDirty()) {
    hasPendingDispatch_ = false;
  }

  TrackQueueProgress(*finalLastSlot, startIndex + numPackets - 1, pre_patched);

  if (blocking) {
    LogInfo("Running serialized as blocking is requested");
    if (!Barriers().WaitCurrent()) {
      LogPrintfError("Failed blocking queue wait with signal [0x%lx]",
                     finalLastSlot->completion_signal.handle);
      profilingEnd();
      return false;
    }
  }

  // HIP_PM4_GRAPH_BUILD_AFTER: the graph just replayed via the flat AQL path above.
  // Build the PM4 IB now -- overlapped with the GPU executing that replay, and with
  // scratch already sized by the submit -- so the NEXT replay uses the single-IB fast
  // path. Reconstruct the header-restored packet pointers (same bridge as the replay
  // attempt at the top of this function).
  if (pm4BuildAfter) {
    constexpr size_t kAqlBytes = sizeof(hsa_kernel_dispatch_packet_t);
    std::vector<uint8_t> hdrRestored(flatPacketData);
    std::vector<void*> pktPtrs(numPackets);
    for (size_t i = 0; i < numPackets; ++i) {
      const uint16_t hdr = static_cast<uint16_t>(validFullHeaders[i]);
      memcpy(hdrRestored.data() + i * kAqlBytes, &hdr, sizeof(hdr));
      pktPtrs[i] = hdrRestored.data() + i * kAqlBytes;
    }
    prebuildPm4Graph(pktPtrs.data(), numPackets, recordedPacketVersion, tmpl);
  }

  profilingEnd();
  return true;
}

// ================================================================================================
bool VirtualGPU::dispatchCounterAqlPacket(hsa_ext_amd_aql_pm4_packet_t* packet,
                                          const uint32_t gfxVersion, bool blocking,
                                          const hsa_ven_amd_aqlprofile_1_00_pfn_t* extApi) {
  // PM4 IB packet submission is different between GFX8 and GFX9:
  //  In GFX8 the PM4 IB packet blob is writing directly to AQL queue
  //  In GFX9 the PM4 IB is submitting by AQL Vendor Specific packet and
  switch (gfxVersion) {
    case PerfCounter::ROC_GFX9:
    case PerfCounter::ROC_GFX10: {
      packet->header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
      return dispatchGenericAqlPacket(packet, 0, 0, blocking);
    } break;
  }

  return false;
}

// ================================================================================================
void VirtualGPU::WaitCompleteSignal(hsa_signal_t signal) {
  // Add the signal as a dynamic dependency so WaitingSignal() includes it
  // alongside any pending external signals in the barrier's dep_signal list.
  Barriers().AddDynamicQueueWait(signal);
  dispatchBarrierPacket(kBarrierPacketHeader, false);
}

// ================================================================================================
void VirtualGPU::dispatchBarrierPacket(uint16_t packetHeader, bool skipSignal,
                                       hsa_signal_t signal) {
  const uint32_t queueSize = gpu_queue_->size;
  const uint32_t queueMask = queueSize - 1;

  if (!skipSignal) {
    // Make sure the wait is issued before queue index reservation
    auto wait_signals = Barriers().WaitingSignal();
    for (uint32_t i = 0; i < wait_signals.size(); ++i) {
      uint32_t j = i % 5;
      barrier_packet_.dep_signal[j] = wait_signals[i];
      constexpr bool kSkipSignal = true;
      // If runtime reached the packet limit and signals left, then flush the barrier
      if ((j == 4) && ((i + 1) < wait_signals.size())) {
        dispatchBarrierPacket(kNopPacketHeader, kSkipSignal);
      }
    }
  }

  uint64_t index = Hsa::queue_add_write_index_screlease(gpu_queue_, 1);
  uint64_t read = Hsa::queue_load_read_index_relaxed(gpu_queue_);

  setFenceDirty(true);
  auto cache_state = extractAqlBits(packetHeader, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                                    HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);
  if (!skipSignal && (signal.handle == 0)) {
    // Get active signal for current dispatch if profiling is necessary
    barrier_packet_.completion_signal = Barriers().ActiveSignal(kInitSignalValueOne, timestamp_);
  } else {
    // Attach external signal to the packet
    barrier_packet_.completion_signal = signal;
  }

  TrackQueueProgress(barrier_packet_, index);

  // Reset fence_dirty_ flag if we submit a barrier with system scopes
  if (cache_state == amd::Device::kCacheStateSystem) {
    setFenceDirty(false);
  }

  while ((index - Hsa::queue_load_read_index_scacquire(gpu_queue_)) >= queueMask);
  hsa_barrier_and_packet_t* aql_loc =
      &(reinterpret_cast<hsa_barrier_and_packet_t*>(gpu_queue_->base_address))[index & queueMask];
  *aql_loc = barrier_packet_;
  metadata_preloader_.Set(&barrier_packet_, packetHeader, index & queueMask);
  packet_store_release(reinterpret_cast<uint32_t*>(aql_loc), packetHeader, 0);

  Hsa::signal_store_screlease(gpu_queue_->doorbell_signal, index);
  logAqlBarrierPacket(gpu_queue_, packetHeader, &barrier_packet_, read, index,
                      IsLogEnabled(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL)
                        ? [this]() -> const char* {
                            if (!roc_device_.settings().queue_pipe_dist_) return "";
                            static thread_local char buf[32];
                            snprintf(buf, sizeof(buf), " virtual_pipe_id=%zu,",
                                     gpu_queue_->id % roc_device_.NumHwPipes());
                            return buf;
                          }()
                        : "");

  // Clear dependent signals for the next packet
  barrier_packet_.dep_signal[0] = hsa_signal_t{};
  barrier_packet_.dep_signal[1] = hsa_signal_t{};
  barrier_packet_.dep_signal[2] = hsa_signal_t{};
  barrier_packet_.dep_signal[3] = hsa_signal_t{};
  barrier_packet_.dep_signal[4] = hsa_signal_t{};
}

// ================================================================================================
// EXPERIMENTAL PWS inter-kernel fence (HIP_PWS_FENCE=1, gfx11 only).
//
// Replicates RADV's GFX11 overlap mechanism on the HIP AQL queue: instead of the
// firmware doing a blocking ACQUIRE_MEM from the dispatch packet's acquire/release
// scope, we strip that scope (see submitKernelInternal) and inject an inline
// vendor PM4-IB packet whose IB is CS_PARTIAL_FLUSH + RELEASE_MEM(PWS) +
// ACQUIRE_MEM(PWS). RELEASE_MEM issues the GCR cache flush async and bumps the PWS
// counter; the PWS ACQUIRE_MEM waits the counter without stalling the CP inline,
// so the flush overlaps the next dispatch. Validated standalone in
// vk_gap_test/pm4_gap/hsa_pws_spike.cpp (bit-exact vs the scope fence, no hang).
// Plain ASCII only.
bool VirtualGPU::pwsFenceActive() {
  if (pwsFenceState_ < 0) {
    pwsFenceState_ = (getenv("HIP_PWS_FENCE") != nullptr) ? 1 : 0;
  }
  return pwsFenceState_ == 1;
}

bool VirtualGPU::ensurePwsIb() {
  if (pwsIbBuf_ != nullptr) {
    return true;
  }
  // PM4 IB dwords lifted verbatim from pm4_layer.cpp (GCR_AGENT_LIKE = 0x380:
  // GL1_INV | GLV_INV | GLK_INV). shaderType=1 on the inner packets (MEC).
  // HIP_PWS_NODRAIN=1 drops the leading CS_PARTIAL_FLUSH: in the AQL path the
  // next dispatch's BARRIER bit already drains the prior dispatch, so the
  // explicit wave drain may be redundant (must be verified bit-exact).
  static const uint32_t kPwsIbFull[] = {
      0xC0004602u, 0x00000407u,                                  // CS_PARTIAL_FLUSH
      0xC0064902u, 0xC000C528u, 0, 0, 0, 0, 0, 0,                // RELEASE_MEM(PWS, GCR=0x380)
      0xC0065802u, 0x00022800u, 0xFFFFFFFFu, 0x01FFFFFFu, 0, 0, 0x80000000u, 0,  // ACQUIRE_MEM(PWS)
  };
  const bool noDrain = getenv("HIP_PWS_NODRAIN") != nullptr;
  const uint32_t* kPwsIb = noDrain ? (kPwsIbFull + 2) : kPwsIbFull;  // skip CS_PARTIAL_FLUSH
  pwsIbDw_ = noDrain ? 16u : 18u;
  // The CP indirect-buffer fetcher needs EXECUTABLE GPU memory.
  void* dev = nullptr;
  if (Hsa::memory_pool_allocate(roc_device_.getGpuvmSegment(), 0x1000,
                                HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, &dev) != HSA_STATUS_SUCCESS) {
    LogError("PWS: executable IB allocation failed");
    return false;
  }
  hsa_agent_t gpu = roc_device_.getBackendDevice();
  Hsa::agents_allow_access(1, &gpu, nullptr, dev);
  // Executable device memory is not CPU-writable, so stage via a host buffer.
  void* host = nullptr;
  if (Hsa::memory_pool_allocate(roc_device_.getCpuFineGrainPool(), 0x1000, 0, &host) !=
      HSA_STATUS_SUCCESS) {
    Hsa::memory_pool_free(dev);
    LogError("PWS: host staging allocation failed");
    return false;
  }
  hsa_agent_t agents[2] = {gpu, roc_device_.getCpuAgent()};
  Hsa::agents_allow_access(2, agents, nullptr, host);
  memset(host, 0, 0x1000);
  memcpy(host, kPwsIb, pwsIbDw_ * sizeof(uint32_t));
  hsa_signal_t s;
  Hsa::signal_create(1, 0, nullptr, &s);
  Hsa::memory_async_copy(dev, gpu, host, roc_device_.getCpuAgent(), 0x1000, 0, nullptr, s);
  while (Hsa::signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                    HSA_WAIT_STATE_BLOCKED) >= 1) {}
  Hsa::signal_destroy(s);
  Hsa::memory_pool_free(host);
  pwsIbBuf_ = dev;
  ClPrint(amd::LOG_INFO, amd::LOG_AQL, "PWS: inter-kernel fence IB ready @%p", pwsIbBuf_);
  return true;
}

// Vendor AMD_AQL_FORMAT_PM4_IB packet (same 64-byte layout ROCr
// AqlQueue::ExecutePM4 uses). completion_signal = 0 so the CP does not block on
// it inline -- the PWS RELEASE/ACQUIRE inside the IB provide the ordering, and
// the cache flush overlaps the next dispatch.
struct PwsVendorPkt {
  uint16_t header;
  uint16_t ven_hdr;
  uint32_t ib_jump_cmd[4];
  uint32_t dw_cnt_remain;
  uint32_t reserved[8];
  hsa_signal_t completion_signal;
};
static void buildPwsVendorPkt(PwsVendorPkt* pkt, void* ibBuf, uint32_t kPwsIbDw) {
  const uint64_t base = reinterpret_cast<uint64_t>(ibBuf);
  memset(pkt, 0, sizeof(*pkt));
  pkt->header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;  // = 0
  pkt->ven_hdr = 0x1;                                                       // AMD_AQL_FORMAT_PM4_IB
  pkt->ib_jump_cmd[0] = (3u << 30) | ((4u - 2u) << 16) | (0x3Fu << 8);     // INDIRECT_BUFFER, shaderType 0
  pkt->ib_jump_cmd[1] = static_cast<uint32_t>(((base >> 2) & 0x3FFFFFFFull) << 2);
  pkt->ib_jump_cmd[2] = static_cast<uint32_t>((base >> 32) & 0xFFFFull);
  pkt->ib_jump_cmd[3] = (kPwsIbDw & 0xFFFFFu) | (1u << 23);                // IB_SIZE | IB_VALID
  pkt->dw_cnt_remain = 0xA;
  pkt->completion_signal = hsa_signal_t{0};
}

// Eager path: append the PWS vendor packet directly to the live HW queue.
void VirtualGPU::injectPwsFence() {
  if (!ensurePwsIb()) {
    return;
  }
  PwsVendorPkt pkt;
  buildPwsVendorPkt(&pkt, pwsIbBuf_, pwsIbDw_);

  const uint32_t queueMask = gpu_queue_->size - 1;
  uint64_t index = Hsa::queue_add_write_index_screlease(gpu_queue_, 1);
  while ((index - Hsa::queue_load_read_index_scacquire(gpu_queue_)) >= queueMask) {
    amd::Os::yield();
  }
  PwsVendorPkt* slot =
      &(reinterpret_cast<PwsVendorPkt*>(gpu_queue_->base_address))[index & queueMask];
  // Write the body first, then the header dword with release so the CP does not
  // read the slot until it is fully written.
  memcpy(reinterpret_cast<uint8_t*>(slot) + sizeof(uint32_t),
         reinterpret_cast<uint8_t*>(&pkt) + sizeof(uint32_t), sizeof(pkt) - sizeof(uint32_t));
  packet_store_release(reinterpret_cast<uint32_t*>(slot), pkt.header, pkt.ven_hdr);
  Hsa::signal_store_screlease(gpu_queue_->doorbell_signal, index);
  hasPendingDispatch_ = true;
}

// Graph-capture path: write the PWS vendor packet into a recorded 64-byte slot.
// CaptureAndFormPacket pushes one kernel-name per gpuPackets_ entry, so the
// extra recorded packet stays aligned with the batch's kernelNames vector.
void VirtualGPU::capturePwsFence(uint8_t* dst) {
  if (dst == nullptr || !ensurePwsIb()) {
    return;
  }
  PwsVendorPkt pkt;
  buildPwsVendorPkt(&pkt, pwsIbBuf_, pwsIbDw_);
  memcpy(dst, &pkt, sizeof(pkt));
}

// ================================================================================================
// PM4 graph replay (HIP_PM4_GRAPH=1): compile a captured all-dispatch hipGraph into
// one PM4 indirect buffer and replay it via a single vendor PM4-IB packet.

namespace {
// AMD kernel descriptor (first 64 bytes), AMDHSA ABI. Mirrors pm4_layer.cpp.
struct AmdKernelDescriptor {
  uint32_t group_segment_fixed_size, private_segment_fixed_size, kernarg_size;
  uint8_t  reserved0[4];
  int64_t  kernel_code_entry_byte_offset;
  uint8_t  reserved1[20];
  uint32_t compute_pgm_rsrc3, compute_pgm_rsrc1, compute_pgm_rsrc2;
  uint16_t kernel_code_properties, kernarg_preload;
  uint8_t  reserved2[4];
};
// kernel_code_properties enable bits (AMDHSA), in user-SGPR layout order.
enum {
  KCP_PRIVATE_SEGMENT_BUFFER = 1u << 0,   // 4 SGPRs (scratch V#)
  KCP_DISPATCH_PTR           = 1u << 1,   // 2 SGPRs
  KCP_QUEUE_PTR              = 1u << 2,   // 2 SGPRs
  KCP_KERNARG_SEGMENT_PTR    = 1u << 3,   // 2 SGPRs
  KCP_DISPATCH_ID            = 1u << 4,   // 2 SGPRs
  KCP_FLAT_SCRATCH_INIT      = 1u << 5,   // 2 SGPRs
  KCP_PRIVATE_SEGMENT_SIZE   = 1u << 6,   // 1 SGPR
  KCP_WAVEFRONT_SIZE32       = 1u << 10,  // 1 = wave32, 0 = wave64
};
// SH compute register addresses (dword). Identical across gfx11 and gfx12: the
// COMPUTE_* block is at the same offsets in every RDNA shadow-reg table (mesa
// ac_shadowed_regs.c R_00B810/30/40/48/54/60/900). SET_SH ordinal = reg - base.
enum {
  kShBase        = 0x2c00,   // PERSISTENT_SPACE_START
  kRegStartX     = 0x2e04,   // COMPUTE_START_X..NUM_THREAD_Z span (8 regs)
  kRegPgmLo      = 0x2e0c,   // COMPUTE_PGM_LO/HI
  kRegScratchLo  = 0x2e10,   // COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI (R_00B840/44)
  kRegRsrc1      = 0x2e12,   // COMPUTE_PGM_RSRC1/2
  kRegResLim     = 0x2e15,   // COMPUTE_RESOURCE_LIMITS
  kRegTmpring    = 0x2e18,   // COMPUTE_TMPRING_SIZE
  kRegUserD0     = 0x2e40,   // COMPUTE_USER_DATA_0
};
// PM4 type-3 opcodes.
enum { kOpSetShReg = 0x76, kOpDispatchDirect = 0x15, kOpEventWrite = 0x46,
       kOpReleaseMem = 0x49, kOpAcquireMem = 0x58 };
// ACQUIRE_MEM GCR_CNTL bits: GLI_INV(0) GLM_WB(4) GLM_INV(5) GLK_INV(7) GLV_INV(8)
// GL1_INV(9) GL2_INV(14) GL2_WB(15). AGENT-like (no L2 flush) vs full L2.
constexpr uint32_t kGcrAgent  = (1u<<7)|(1u<<8)|(1u<<9);                       // 0x380
constexpr uint32_t kGcrGlmBits = (1u<<4)|(1u<<5);                              // metadata cache
constexpr uint32_t kGcrFullGfx11 = (1u<<15)|(1u<<14)|(1u<<9)|(1u<<8)|(1u<<7)|(1u<<0) | kGcrGlmBits; // 0xC3B1
// gfx12 has no metadata (GLM) cache: radv drops GLM_WB/GLM_INV from the L2 flush
// (radv_cs.c gfx10_cs_emit_cache_flush, "gfx_level < GFX12 ? GLM... : 0").
constexpr uint32_t kGcrFullGfx12 = kGcrFullGfx11 & ~kGcrGlmBits;              // 0xC381
// DISPATCH_INITIATOR: CS_EN(0) | USE_THREAD_DIMS(5). CS_W32_EN(15) added per wave.
constexpr uint32_t kDispatchBase  = 0x21u;       // CS_EN | USE_THREAD_DIMS
constexpr uint32_t kDispatchW32   = 0x8000u;     // CS_W32_EN (wave32 only)

// Per-arch PM4 emission parameters. gfx11 (RDNA3) and gfx12 (RDNA4) share the
// PWS RELEASE_MEM/ACQUIRE_MEM encoding and the COMPUTE_* register map; only the
// full-L2 GCR mask differs (no GLM cache on gfx12).
struct Pm4Arch {
  uint32_t gcrFull;   // GCR for the trailing/leading full-L2 acquire
  bool     isGfx12;
};

inline uint32_t pm4Hdr(uint32_t opcode, uint32_t dw) {
  // type3 (3<<30), count=(dw-2)<<16, opcode<<8, shaderType=1 (bit1).
  return 0xC0000000u | ((dw - 2u) << 16) | (opcode << 8) | 0x2u;
}

// Tri-state parse of a PM4 boolean env var that RESPECTS the value, so that
// HIP_PM4_GRAPH=0 means OFF (not "the variable is set, therefore on"). This
// avoids the foot-gun where =0 and =1 behaved identically. Semantics:
//   unset                         -> def (documented default for this knob)
//   "0","false","off","no","" etc -> false (case-insensitive)
//   any other non-empty value     -> true  (e.g. "1","yes","on","true")
static bool pm4EnvBool(const char* name, bool def) {
  const char* v = getenv(name);
  if (v == nullptr) return def;          // unset: take the knob default
  const char c = v[0];
  if (c == '\0') return false;           // empty string: treat as off
  if (c == '0' || c == 'f' || c == 'F' || c == 'n' || c == 'N') return false;  // 0/false/no
  if ((c == 'o' || c == 'O') && (v[1] == 'f' || v[1] == 'F')) return false;    // off (not "on")
  return true;                           // 1/yes/on/true/...
}
}  // namespace

bool VirtualGPU::pm4GraphActive() {
  if (pm4GraphState_ < 0) {
    pm4GraphState_ = pm4EnvBool("HIP_PM4_GRAPH", false) ? 1 : 0;
  }
  return pm4GraphState_ == 1;
}

// Scratch-kernel support is opt-in: replicating the CP scratch setup (TMPRING,
// scratch base, scratch V#) is correct only after ROCr has sized the queue
// scratch, and a mistake faults the GPU. Default off keeps scratch kernels on
// the safe AQL fallback.
bool VirtualGPU::pm4GraphScratchEnabled() {
  if (pm4GraphScratchState_ < 0) {
    pm4GraphScratchState_ = pm4EnvBool("HIP_PM4_GRAPH_SCRATCH", false) ? 1 : 0;
  }
  return pm4GraphScratchState_ == 1;
}

// Register delta-encoding: emit a SET_SH_REG only for the minimal contiguous
// sub-range whose value changed vs the previous dispatch in the same IB. SH
// registers are sticky and the drain/PWS packets do not touch them, so skipping
// an unchanged write leaves the latched state at the consuming DISPATCH
// bit-identical to the always-emit path. Opt-in; default off keeps the IB
// byte-identical to the baseline emitter. See HIP_VS_VULKAN.md Appendix D-impl.
bool VirtualGPU::pm4GraphDeltaEnabled() {
  if (pm4GraphDeltaState_ < 0) {
    pm4GraphDeltaState_ = pm4EnvBool("HIP_PM4_GRAPH_DELTA", false) ? 1 : 0;
  }
  return pm4GraphDeltaState_ == 1;
}

// IB-reorder: emit kernel N+1's register programming BETWEEN N's RELEASE_MEM and
// ACQUIRE_MEM, so the CP programs the next kernel while N's waves drain and the
// AGENT cache flush is in flight (CP otherwise idle). The drain stays first and
// the acquire stays before the consuming DISPATCH, so coherence is unchanged.
//
// WARNING: this lever does NOT improve performance on measured RDNA3 (gfx1100)
// hardware -- it is a validated NULL RESULT, kept only for experimentation. The
// CP micro-engine (ME) is a single in-order processor that handles BOTH the
// SET_SH_REG writes AND the ACQUIRE_MEM wait, so moving the register burst ahead
// of the acquire merely reorders serial ME work; there is no second engine to run
// it concurrently with the PWS wait. Across kernel sizes the inter-kernel gap was
// unchanged within noise (HIP_VS_VULKAN.md Appendix D-impl D.13). Use
// HIP_PM4_GRAPH_DELTA instead, which REDUCES the ME work and is a real win. This
// path is correct (bit-exact) and harmless, but pointless on this hardware.
bool VirtualGPU::pm4GraphReorderEnabled() {
  if (pm4GraphReorderState_ < 0) {
    pm4GraphReorderState_ = pm4EnvBool("HIP_PM4_GRAPH_REORDER", false) ? 1 : 0;
    if (pm4GraphReorderState_ == 1) {
      ClPrint(amd::LOG_WARNING, amd::LOG_AQL,
              "HIP_PM4_GRAPH_REORDER is enabled but is a NO-OP for performance on "
              "RDNA3 (CP micro-engine is in-order: register programming cannot "
              "overlap the PWS wait). It is bit-exact and harmless. Use "
              "HIP_PM4_GRAPH_DELTA instead, which actually reduces the gap.");
    }
  }
  return pm4GraphReorderState_ == 1;
}

// Last-lookup IB cache (HIP_PM4_GRAPH_KEYCACHE): the per-launch pm4GraphKey hash
// is O(N) over every packet, and in the steady-state decode loop the SAME packet
// array is replayed every token (nothing the hash covers changes -- only kernarg
// CONTENTS change, which the hash ignores). So the hash returns the same key every
// launch: pure overhead that also sits BEFORE the submit doorbell, delaying GPU
// start. This fast path keys the cached IB by the graph-supplied recorded packet
// set version (see GraphExec::RecordedPacketVersion) and, on a version match,
// reuses the cached IB pointer without rehashing -> O(1) host issue, GPU starts
// sooner. The version is bumped on EVERY recorded-packet mutation site (initial
// capture, per-node param update, node enable/disable -- all routed through
// CaptureAndFormPacketsForGraph / UpdateAQLPacket / UpdatePacketBatchesForNodeEnable
// Disable), so invalidation is RELIABLE and EXACT, not a heuristic: there is no
// interior-node blind spot. Default ON because it is correct for any caller that
// supplies a version; set HIP_PM4_GRAPH_KEYCACHE=0 to force the always-rehash path.
// version 0 (non-graph caller) always takes the slow rehash path regardless.
bool VirtualGPU::pm4GraphKeyCacheEnabled() {
  if (pm4GraphKeyCacheState_ < 0) {
    pm4GraphKeyCacheState_ = pm4EnvBool("HIP_PM4_GRAPH_KEYCACHE", true) ? 1 : 0;
  }
  return pm4GraphKeyCacheState_ == 1;
}

// Executable IB arena prewarm (HIP_PM4_GRAPH_PREWARM): reserve one device-local
// executable buffer once (warming the executable memory pool off the launch
// critical path) and sub-allocate IBs from it. Opt-in for now; measured before
// considering default-on. See HIP_VS_VULKAN.md Appendix D.
bool VirtualGPU::pm4GraphPrewarmEnabled() {
  if (pm4GraphPrewarmState_ < 0) {
    pm4GraphPrewarmState_ = pm4EnvBool("HIP_PM4_GRAPH_PREWARM", false) ? 1 : 0;
  }
  return pm4GraphPrewarmState_ == 1;
}

// Build-after (HIP_PM4_GRAPH_BUILD_AFTER): on the FIRST replay of a graph (cache
// miss), do NOT build the PM4 IB synchronously before submit. Instead replay this
// launch via the AQL fallback (GPU starts in ~us, no build-induced idle) and build
// the PM4 IB right after the submit, so the build (and, with PREWARM off, the
// one-time pool warm-up) overlaps the GPU executing the graph just submitted. The
// AQL submit also sizes the queue scratch, so the deferred build is never
// scratch-deferred. The next replay reuses the cached IB. Opt-in.
bool VirtualGPU::pm4GraphBuildAfterEnabled() {
  if (pm4GraphBuildAfterState_ < 0) {
    pm4GraphBuildAfterState_ = pm4EnvBool("HIP_PM4_GRAPH_BUILD_AFTER", false) ? 1 : 0;
  }
  return pm4GraphBuildAfterState_ == 1;
}

// GraphExec-owned shared IB: a queue-INDEPENDENT graph (no scratch, no queue_ptr)
// compiles to one device-scoped IB built once at instantiate and referenced by every
// stream that replays it (no per-stream specialize/upload, one VRAM copy). Default
// on; HIP_PM4_GRAPH_SHARED_IB=0 disables (each stream specializes its own IB).
bool VirtualGPU::pm4GraphSharedIbEnabled() {
  if (pm4GraphSharedIbState_ < 0) {
    pm4GraphSharedIbState_ = pm4EnvBool("HIP_PM4_GRAPH_SHARED_IB", true) ? 1 : 0;
  }
  return pm4GraphSharedIbState_ == 1;
}

// Per-edge fence scope inheritance: derive each kernel->kernel edge GCR mask from
// the scope the runtime actually recorded in the captured packet headers instead
// of forcing a blanket AGENT mask. For the fence after dispatch i we take
// max(release_scope(i), acquire_scope(i+1)) and map NONE -> no fence (independent
// kernels may overlap), AGENT -> kGcrAgent (invalidate GLK/GLV/GL1, no L2 flush),
// SYSTEM -> full-L2. The graph boundaries (leading acquire, trailing release)
// ALWAYS stay full-L2: the boundary SYSTEM requirement is injected at submit time
// by the runtime's addSystemScope_ / fence_dirty_ state machine and is NOT carried
// in the recorded packet, so the safe superset is to flush L2 at the ends.
// DEFAULT ON (respect the runtime's per-packet scope decision); set
// HIP_PM4_GRAPH_NO_INHERIT_SCOPE to fall back to the fixed blanket-AGENT policy.
bool VirtualGPU::pm4GraphInheritScopeEnabled() {
  if (pm4GraphInheritScopeState_ < 0) {
    pm4GraphInheritScopeState_ = pm4EnvBool("HIP_PM4_GRAPH_NO_INHERIT_SCOPE", false) ? 0 : 1;
  }
  return pm4GraphInheritScopeState_ == 1;
}

// Per-kernel GPU-clock profiling for the PM4 replay path. PM4 IB submission is
// invisible to a tool (one vendor INDIRECT_BUFFER packet hides the N dispatches), so
// this path bakes a RELEASE_MEM(BOTTOM_OF_PIPE_TS, send-GPU-clock) per kernel boundary
// into the IB, then reads the GPU-domain ticks back on the host and ClPrints per-kernel
// timing. There is no dedicated env var: it is gated on the SAME HIP logging channel
// the per-kernel output uses (LOG_INFO + LOG_AQL, i.e. AMD_LOG_LEVEL/AMD_LOG_MASK),
// mirroring the existing AQL / "Graph ShaderName" kernel logs -- so it turns on exactly
// when AQL-packet logging is requested and is otherwise fully dormant. IsLogEnabled
// reads the process-global log level/mask (set at startup), so the capture-time encode
// and the submit-time readback agree. While a profiler is armed the runtime falls back
// to AQL anyway (see tryReplayPm4Graph), so the two never fight over the same launch.
bool VirtualGPU::pm4GraphProfileEnabled() {
  return IsLogEnabled(amd::LOG_INFO, amd::LOG_AQL);
}

// True when a kernel-dispatch profiler is intercepting the queue, in which case the
// PM4 IB replay (one opaque vendor packet that hides the N dispatches) must give way
// to AQL so every kernel stays visible. There are two INDEPENDENT profiling planes,
// and neither alone is sufficient:
//
//  1. HIP-ops / roctracer plane (amd::activity_prof): a callback registered via
//     hipRegisterTracerCallback. This is the path roctracer and the legacy HIP
//     profiler use; it can be armed/disarmed at RUNTIME, so it is checked live.
//
//  2. rocprofiler-sdk plane (rocprofv3 --kernel-trace, etc): intercepts dispatches
//     below CLR, at the HSA api-table level via rocprofiler-register (see ROCr
//     runtime.cpp). It does NOT register the HIP-ops callback, so IsEnabled() is
//     false under it -- that is exactly why a kernel-trace run was capturing PM4 as
//     zero kernels. The runtime contract is that every rocprofiler-sdk TOOL exports
//     a strong rocprofiler_configure entry point (the core lib only carries a weak
//     undefined ref), so dlsym(RTLD_DEFAULT) resolving it means a tool is loaded in
//     this process. This is tool-agnostic (works for rocprofv3, HSA_TOOLS_LIB, a
//     direct LD_PRELOAD, ...) and far more robust than sniffing tool-specific env
//     vars. A tool loads at process init, so the lookup is done once and cached.
bool VirtualGPU::pm4TracingArmed() {
  if (amd::activity_prof::IsEnabled(OP_ID_DISPATCH)) {
    return true;
  }
#if defined(__linux__)
  if (pm4SdkProfilerState_ < 0) {
    pm4SdkProfilerState_ = (dlsym(RTLD_DEFAULT, "rocprofiler_configure") != nullptr) ? 1 : 0;
  }
  return pm4SdkProfilerState_ == 1;
#else
  return false;
#endif
}

// Reserve the executable IB arena once. The single allocation pays the one-time
// executable-pool first-touch cost here (at init / first PM4 use) instead of on
// the first replay. Idempotent.
void VirtualGPU::ensurePm4Arena() {
  if (pm4Arena_ != nullptr || !pm4GraphActive() || !pm4GraphPrewarmEnabled()) {
    return;
  }
  void* dev = nullptr;
  if (Hsa::memory_pool_allocate(roc_device_.getGpuvmSegment(), kPm4ArenaBytes,
                                HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, &dev) != HSA_STATUS_SUCCESS) {
    ClPrint(amd::LOG_WARNING, amd::LOG_CODE,
            "[pm4-arena] reserve of %zu bytes failed; falling back to per-IB pool alloc",
            kPm4ArenaBytes);
    return;
  }
  hsa_agent_t gpu = roc_device_.getBackendDevice();
  Hsa::agents_allow_access(1, &gpu, nullptr, dev);
  pm4Arena_ = dev;
  pm4ArenaBytes_ = kPm4ArenaBytes;
  pm4ArenaFree_.clear();
  pm4ArenaFree_.emplace_back(0, kPm4ArenaBytes);  // one big free span
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[pm4-arena] reserved %zu bytes at %p",
          kPm4ArenaBytes, pm4Arena_);

  // Warm the FULL upload path once, not just the executable pool: the one-time
  // first-touch cost is dominated by the CPU fine-grain staging pool's first
  // allocation and the first SDMA copy, which the arena reservation alone does
  // not exercise. Do one throwaway allocExecIbFromData (arena slice + host
  // staging + async copy + wait) and return the slice, so the first REAL build
  // pays only the (now-warm) per-build cost.
  uint32_t warmData[64] = {0};
  void* warm = allocExecIbFromData(warmData, 64);
  if (warm != nullptr) {
    Pm4GraphIb tmp;
    tmp.ib = warm;
    tmp.dw = 64;
    tmp.status = kPm4Ready;
    freePm4GraphIb(tmp);
  }
}

// First-fit allocation from the arena free list. bytes must be page-rounded by
// the caller. Returns nullptr if no span fits (caller falls back to a pool alloc).
void* VirtualGPU::pm4ArenaAlloc(size_t bytes) {
  if (pm4Arena_ == nullptr || bytes == 0) {
    return nullptr;
  }
  for (size_t i = 0; i < pm4ArenaFree_.size(); ++i) {
    if (pm4ArenaFree_[i].second >= bytes) {
      size_t off = pm4ArenaFree_[i].first;
      if (pm4ArenaFree_[i].second == bytes) {
        pm4ArenaFree_.erase(pm4ArenaFree_.begin() + i);
      } else {
        pm4ArenaFree_[i].first += bytes;
        pm4ArenaFree_[i].second -= bytes;
      }
      return reinterpret_cast<uint8_t*>(pm4Arena_) + off;
    }
  }
  return nullptr;  // fragmented / too big -> caller uses a direct pool alloc
}

bool VirtualGPU::pm4ArenaOwns(const void* p) const {
  if (pm4Arena_ == nullptr || p == nullptr) {
    return false;
  }
  auto base = reinterpret_cast<uintptr_t>(pm4Arena_);
  auto q = reinterpret_cast<uintptr_t>(p);
  return q >= base && q < base + pm4ArenaBytes_;
}

// Return a slice to the free list and coalesce with adjacent free spans.
void VirtualGPU::pm4ArenaFreeBytes(void* p, size_t bytes) {
  size_t off = reinterpret_cast<uint8_t*>(p) - reinterpret_cast<uint8_t*>(pm4Arena_);
  // Insert sorted by offset.
  size_t i = 0;
  while (i < pm4ArenaFree_.size() && pm4ArenaFree_[i].first < off) {
    ++i;
  }
  pm4ArenaFree_.insert(pm4ArenaFree_.begin() + i, std::make_pair(off, bytes));
  // Coalesce right then left.
  if (i + 1 < pm4ArenaFree_.size() &&
      pm4ArenaFree_[i].first + pm4ArenaFree_[i].second == pm4ArenaFree_[i + 1].first) {
    pm4ArenaFree_[i].second += pm4ArenaFree_[i + 1].second;
    pm4ArenaFree_.erase(pm4ArenaFree_.begin() + i + 1);
  }
  if (i > 0 && pm4ArenaFree_[i - 1].first + pm4ArenaFree_[i - 1].second == pm4ArenaFree_[i].first) {
    pm4ArenaFree_[i - 1].second += pm4ArenaFree_[i].second;
    pm4ArenaFree_.erase(pm4ArenaFree_.begin() + i);
  }
}

// Free an IB's storage, routing to the arena free list or the device pool.
void VirtualGPU::freePm4GraphIb(Pm4GraphIb& g) {
  if (g.ib == nullptr) {
    return;
  }
  // Per-kernel profiling buffer is always per-IB owned,
  // even when the IB itself is a non-owning shared reference, so free it first.
  if (g.tsBuf != nullptr) {
    Hsa::memory_pool_free(g.tsBuf);
    g.tsBuf = nullptr;
    g.tsCount = 0;
  }
  // Non-owning entry: only a reference to a GraphExec-owned shared IB. Drop the
  // reference; the GraphExec frees the storage in freePm4GraphTemplate.
  if (!g.owned) {
    g.ib = nullptr;
    return;
  }
  if (pm4ArenaOwns(g.ib)) {
    size_t bytes = (static_cast<size_t>(g.dw) * 4 + 0xFFF) & ~static_cast<size_t>(0xFFF);
    pm4ArenaFreeBytes(g.ib, bytes);
  } else {
    Hsa::memory_pool_free(g.ib);
  }
  g.ib = nullptr;
}

// Content hash over the fields that define the compiled IB, so a destroyed and
// reallocated graph that happens to reuse the same host packet address does not
// replay a stale IB (the old packet[0]-pointer key could alias). FNV-1a.
uint64_t VirtualGPU::pm4GraphKey(void* const* packets, size_t numPackets) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint64_t v) {
    for (int b = 0; b < 8; ++b) {
      h ^= (v & 0xff);
      h *= 1099511628211ull;
      v >>= 8;
    }
  };
  mix(numPackets);
  for (size_t i = 0; i < numPackets; ++i) {
    auto* p = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packets[i]);
    mix(p->header | (static_cast<uint64_t>(p->setup) << 16));
    mix(p->kernel_object);
    mix(reinterpret_cast<uint64_t>(p->kernarg_address));
    mix(static_cast<uint64_t>(p->grid_size_x) | (static_cast<uint64_t>(p->grid_size_y) << 32));
    mix(static_cast<uint64_t>(p->grid_size_z) | (static_cast<uint64_t>(p->workgroup_size_x) << 32));
    mix(static_cast<uint64_t>(p->workgroup_size_y) | (static_cast<uint64_t>(p->workgroup_size_z) << 32));
  }
  return h;
}

// Structural hash: like pm4GraphKey but EXCLUDING the mutable scalars (kernarg,
// grid, workgroup). Two packet sets with the same skeleton differ only by fields
// the in-place fast path can patch (kernarg/grid/workgroup), so a skeleton match
// means the resident IB can be patched rather than rebuilt. (group/private segment
// are not hashed here, matching pm4GraphKey, so a segment-only change is outside
// the supported scalar-mutation set -- same blind spot as the content key.)
uint64_t VirtualGPU::pm4GraphSkeletonKey(void* const* packets, size_t numPackets) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint64_t v) {
    for (int b = 0; b < 8; ++b) {
      h ^= (v & 0xff);
      h *= 1099511628211ull;
      v >>= 8;
    }
  };
  mix(numPackets);
  for (size_t i = 0; i < numPackets; ++i) {
    auto* p = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packets[i]);
    mix(p->header | (static_cast<uint64_t>(p->setup) << 16));
    mix(p->kernel_object);
  }
  return h;
}

void VirtualGPU::evictPm4GraphsIfNeeded(const Pm4GraphIb* protect) {
  // Free oldest-inserted entries until under the cap. Skip the armed entry (its
  // pointer is cached for the fast path) and the entry just built this launch.
  // The guard counter bounds the loop: each iteration either erases one entry
  // (size shrinks) or re-queues a protected key (no progress on size but finite).
  size_t guard = pm4GraphKeyOrder_.size();
  while (pm4Graphs_.size() > kPm4MaxCachedIbs && guard-- > 0) {
    uint64_t k = pm4GraphKeyOrder_.front();
    pm4GraphKeyOrder_.pop_front();
    auto it = pm4Graphs_.find(k);
    if (it == pm4Graphs_.end()) {
      continue;  // stale key (entry already erased) -- drop it
    }
    Pm4GraphIb* e = &it->second;
    if (e == protect || e == pm4IbCacheEntry_) {
      pm4GraphKeyOrder_.push_back(k);  // never free a live/armed IB; try the next
      continue;
    }
    freePm4GraphIb(*e);
    pm4Graphs_.erase(it);
    ClPrint(amd::LOG_INFO, amd::LOG_CODE,
            "[pm4-evict] freed stale IB; cache size now %zu (cap %zu)",
            pm4Graphs_.size(), kPm4MaxCachedIbs);
  }
}

void* VirtualGPU::allocExecIbFromData(const uint32_t* data, uint32_t dw, bool deviceScoped) {
  size_t bytes = (static_cast<size_t>(dw) * 4 + 0xFFF) & ~static_cast<size_t>(0xFFF);
  hsa_agent_t gpu = roc_device_.getBackendDevice();
  // Prefer a slice from the pre-reserved arena (already GPU-accessible); fall back
  // to a direct executable-pool allocation if the arena is off / full / too small.
  // deviceScoped skips the per-vdev arena so the IB outlives any single stream.
  void* dev = deviceScoped ? nullptr : pm4ArenaAlloc(bytes);
  if (dev == nullptr) {
    if (Hsa::memory_pool_allocate(roc_device_.getGpuvmSegment(), bytes,
                                  HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, &dev) != HSA_STATUS_SUCCESS) {
      return nullptr;
    }
    Hsa::agents_allow_access(1, &gpu, nullptr, dev);
  }
  void* host = nullptr;
  if (Hsa::memory_pool_allocate(roc_device_.getCpuFineGrainPool(), bytes, 0, &host) !=
      HSA_STATUS_SUCCESS) {
    if (pm4ArenaOwns(dev)) {
      pm4ArenaFreeBytes(dev, bytes);
    } else {
      Hsa::memory_pool_free(dev);
    }
    return nullptr;
  }
  hsa_agent_t agents[2] = {gpu, roc_device_.getCpuAgent()};
  Hsa::agents_allow_access(2, agents, nullptr, host);
  memcpy(host, data, static_cast<size_t>(dw) * 4);
  hsa_signal_t s;
  Hsa::signal_create(1, 0, nullptr, &s);
  Hsa::memory_async_copy(dev, gpu, host, roc_device_.getCpuAgent(), static_cast<size_t>(dw) * 4, 0,
                         nullptr, s);
  while (Hsa::signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                    HSA_WAIT_STATE_BLOCKED) >= 1) {}
  Hsa::signal_destroy(s);
  Hsa::memory_pool_free(host);
  return dev;
}

// Queue-runtime sentinel: every placeholder dword is encoded as this value so that
// delta-encoding makes the SAME emit/skip decisions it would with the real (queue)
// value -- e.g. a scratch kernel's TMPRING (sentinel) differs from a non-scratch
// kernel's 0, so the non-scratch kernel still re-emits 0. specializeFromTemplate()
// overwrites the sentinel with the launch stream's value. Plain ASCII only.
static constexpr uint32_t kPm4PlaceholderSentinel = 0xFFFFFFFFu;

// CPU-only, queue-INDEPENDENT encode of a captured graph into a PM4 template:
// queue-runtime fields (scratch base/size/V#, amd_queue_t pointer) are emitted as
// sentinels and their dword offsets recorded in out.patches. Runs at capture/
// instantiate on any vdev of the graph's device; specializeFromTemplate() later
// patches the placeholders and uploads. (Was the front half of buildPm4GraphIb.)
void VirtualGPU::encodePm4GraphTemplate(void* const* packets, size_t numPackets,
                                        Pm4GraphTemplate& out) {
  out.status = kPm4UnsupportedPermanent;  // pessimistic; promoted on success
  out.numPackets = numPackets;

  // Select arch (gfx11 RDNA3 vs gfx12 RDNA4). Anything else is unsupported.
  const auto& isa = roc_device_.isa();
  Pm4Arch arch;
  if (isa.versionMajor() == 11) {
    arch = Pm4Arch{kGcrFullGfx11, false};
  } else if (isa.versionMajor() == 12) {
    arch = Pm4Arch{kGcrFullGfx12, true};
  } else {
    return;  // only RDNA3/RDNA4 carry this PWS encoding
  }

  // Per-kernel GPU-clock profiling, gated on the LOG_INFO+LOG_AQL logging channel.
  // Decided at encode time (process-stable log level), so a graph captured with AQL
  // logging on carries timestamp packets and one captured without it does not. Only
  // the default blocking-AGENT path emits them; the legacy PWS paths are untouched.
  const bool instrument = pm4GraphProfileEnabled();
  out.instrumented = instrument;
  out.tsAddrOff.clear();

  std::vector<uint32_t>& ib = out.dwords;
  std::vector<Pm4Placeholder>& patches = out.patches;
  ib.reserve(numPackets * 64 + 24);

  auto setsh = [&](uint32_t reg, const uint32_t* v, uint32_t cnt,
                   const uint8_t* kinds = nullptr) {
    ib.push_back(pm4Hdr(kOpSetShReg, 2 + cnt));
    ib.push_back(reg - kShBase);
    for (uint32_t i = 0; i < cnt; ++i) {
      if (kinds && kinds[i] != kPhNone) {
        patches.push_back({static_cast<uint32_t>(ib.size()), kinds[i]});
      }
      ib.push_back(v[i]);
    }
  };
  auto partialFlush = [&]() {
    ib.push_back(pm4Hdr(kOpEventWrite, 2));
    ib.push_back(0x00000407u);  // EVENT_INDEX=4 (CS_VS_PS_PARTIAL_FLUSH), CS_PARTIAL_FLUSH
  };
  auto releaseMemPws = [&](uint32_t gcr) {
    uint32_t op = 40u | (5u << 8);          // EVENT_TYPE=BOTTOM_OF_PIPE_TS, EVENT_INDEX=5
    if (gcr & (1u << 7)) op |= (1u << 30);  // GLK_INV
    if (gcr & (1u << 8)) op |= (1u << 14);  // GLV_INV
    if (gcr & (1u << 9)) op |= (1u << 15);  // GL1_INV
    op |= (1u << 31);                       // PWS_ENABLE
    ib.push_back(pm4Hdr(kOpReleaseMem, 8));
    ib.push_back(op);
    for (int i = 0; i < 6; ++i) ib.push_back(0);
  };
  auto acquirePws = [&]() {
    ib.push_back(pm4Hdr(kOpAcquireMem, 8));
    ib.push_back((5u << 11) | (1u << 17));  // PWS_STAGE_SEL=CP_ME, PWS_ENA2=1, COUNT=0
    ib.push_back(0xFFFFFFFFu);
    ib.push_back(0x01FFFFFFu);
    ib.push_back(0);
    ib.push_back(0);
    ib.push_back(1u << 31);                 // PWS_ENA
    ib.push_back(0);                        // GCR_CNTL (flush carried by RELEASE_MEM)
  };
  auto acquireFull = [&](uint32_t gcr) {
    ib.push_back(pm4Hdr(kOpAcquireMem, 8));
    ib.push_back(0);            // CP_COHER_CNTL
    ib.push_back(0xFFFFFFFFu);  // CP_COHER_SIZE
    ib.push_back(0x00FFFFFFu);  // CP_COHER_SIZE_HI
    ib.push_back(0);            // CP_COHER_BASE
    ib.push_back(0);            // CP_COHER_BASE_HI
    ib.push_back(0x0Au);        // POLL_INTERVAL
    ib.push_back(gcr);          // GCR_CNTL
  };
  // RELEASE_MEM that writes the 64-bit GPU clock counter to a TS slot (no cache op,
  // no PWS). The address is a placeholder (0) patched in specializeFromTemplate; its
  // ADDRESS_LO dword offset is recorded so the slot index == position in tsAddrOff.
  // EVENT_TYPE=BOTTOM_OF_PIPE_TS fires after the preceding kernel's waves drain, so
  // consecutive slots bracket per-kernel wall time on the GPU timeline.
  auto emitTs = [&]() {
    if (!instrument) return;
    ib.push_back(pm4Hdr(kOpReleaseMem, 8));
    ib.push_back(40u | (5u << 8));            // EVENT_TYPE=BOTTOM_OF_PIPE_TS, EVENT_INDEX=end_of_pipe(5)
    // DST_SEL=memory_controller(0) so the clock write bypasses L2 straight to memory
    // (host fine-grain visible without a cache flush); DATA_SEL=send_gpu_clock(3).
    ib.push_back(3u << 29);
    out.tsAddrOff.push_back(static_cast<uint32_t>(ib.size()));  // ADDRESS_LO offset
    ib.push_back(0);                          // ADDRESS_LO (patched to TS slot)
    ib.push_back(0);                          // ADDRESS_HI (patched)
    ib.push_back(0);                          // DATA_LO  (ignored for DATA_SEL=3)
    ib.push_back(0);                          // DATA_HI
    ib.push_back(0);                          // INT_CTXID
  };

  // Register delta-encoding (HIP_PM4_GRAPH_DELTA). SH registers are sticky and the
  // drain/PWS packets do not touch them, so a SET_SH_REG whose value already
  // matches what is latched can be skipped: the state at the consuming DISPATCH
  // is identical to the always-emit path. We keep a per-group shadow of the last
  // value emitted in THIS IB and emit only the minimal contiguous changed
  // sub-range. The shadow starts invalid, so dispatch 0 emits every group (state
  // is re-established on every replay regardless of pre-IB register contents).
  // 'sig' guards layout-sensitive groups (USER_DATA): a different enabled-SGPR
  // set means the same offset holds a different register, so we force a full
  // re-emit. Plain ASCII only.
  // In-place mutation (HIP_PM4_GRAPH_INPLACE) forces delta-encoding OFF: every
  // dispatch must emit its OWN copy of the mutable register groups so a scalar
  // mutation has a dedicated dword slot to patch. With delta on, a dispatch whose
  // value matched the previous one would share that latched slot, and patching it
  // would corrupt the other dispatch. mutFields are recorded only in this mode.
  const bool inplace = pm4GraphInplaceEnabled();
  const bool delta = pm4GraphDeltaEnabled() && !inplace;
  const bool reorder = pm4GraphReorderEnabled();
  std::vector<Pm4MutField>& muts = out.mutFields;

  // Per-edge fence mechanism. DEFAULT (and the only correct choice) is a BLOCKING
  // AGENT acquire: CS_PARTIAL_FLUSH (drain the producer's waves) + ACQUIRE_MEM
  // with GCR_CNTL = kGcrAgent (invalidate GLK/GLV/GL1; NO L2 flush -- L2 is the
  // device coherence point and only the token boundary needs a full-L2 writeback).
  // This mirrors the runtime's AGENT dispatch scope and is sufficient for
  // device-local kernel->kernel RAW: L0/L1 are write-through to L2, the drain puts
  // the producer's writes in L2, and the consumer's L0/L1/K invalidate makes it
  // miss to L2. It is correct, deterministic, and ~2x cheaper per dispatch than the
  // PWS path (one ACQUIRE_MEM vs a RELEASE_MEM+ACQUIRE_MEM counter round-trip).
  //
  // HIP_PM4_GRAPH_PWS (opt-in, LEGACY, UNSAFE): the old partially-waited-sync
  // deferred fence (RELEASE_MEM arms a counter, ACQUIRE_MEM waits) that tried to
  // overlap the flush with the next dispatch. It is BUGGY: the consumer DISPATCH
  // can begin before the AGENT invalidate is globally complete, so a chain of many
  // tiny dependent kernels (e.g. gpt-oss MoE, 267 dispatches/token) reads stale
  // data NON-deterministically and diverges from the AQL baseline. Larger dense
  // kernels (Llama-8B) happened to hide the race behind launch latency. Kept only
  // for A/B; do NOT use in production. (See HIP_VS_VULKAN.md D.13/D.14.)
  const bool usePws = pm4EnvBool("HIP_PM4_GRAPH_PWS", false);

  // DIAGNOSTIC (HIP_PM4_GRAPH_FULLFENCE / HIP_PM4_GRAPH_EDGE_GCR): override the
  // blocking per-edge GCR mask. FULLFENCE uses the full-L2 mask; EDGE_GCR=<hexmask>
  // uses an explicit GCR_CNTL so we can bisect which cache bit a graph needs.
  // GCR bits: GLI_INV(0) GLM_WB(4) GLM_INV(5) GLK_INV(7) GLV_INV(8) GL1_INV(9)
  //           GL2_INV(14) GL2_WB(15). kGcrAgent=0x380, full gfx11=0xC3B1.
  const char* edgeGcrEnv = getenv("HIP_PM4_GRAPH_EDGE_GCR");
  const bool fullFence = pm4EnvBool("HIP_PM4_GRAPH_FULLFENCE", false) || (edgeGcrEnv != nullptr);
  const uint32_t edgeGcr =
      edgeGcrEnv ? static_cast<uint32_t>(strtoul(edgeGcrEnv, nullptr, 0)) : arch.gcrFull;
  const uint32_t edgeMask = fullFence ? edgeGcr : kGcrAgent;

  // Per-edge mask derived from the recorded packet scope (default; disabled by
  // HIP_PM4_GRAPH_NO_INHERIT_SCOPE, see pm4GraphInheritScopeEnabled). The
  // diagnostic fullFence override takes precedence. Boundaries stay full-L2.
  const bool inheritScope = pm4GraphInheritScopeEnabled() && !fullFence;
  // INTRA-GRAPH SCOPE REDUCTION (default ON; opt-out HIP_PM4_GRAPH_EDGE_FULL_L2=1).
  // On gfx12 the runtime stamps EVERY kernel dispatch packet with acquire=SYSTEM
  // (see dispatchPacketHeader_ / sysAcquireAgentReleaseHBits), a blanket-conservative
  // default that is NOT derived from any real data-dependency analysis. With
  // inherit-scope on, edgeMaskFor would therefore promote ~every interior edge to a
  // full-L2 flush (gcrFull), even for a pure device-local kernel->kernel RAW chain.
  // That full-L2 work is REDUNDANT in the interior of a graph: GL2 is the single
  // device coherence point shared by all CUs. The producer drain (CS_PARTIAL_FLUSH)
  // retires its waves and its write-through L0/L1 land in GL2; an AGENT acquire on the
  // consumer (invalidate per-CU GL0/GL1/GLK, NO L2 op) then misses to GL2 and reads
  // fresh data. A SYSTEM acquire's extra GL2 invalidate/writeback only buys HOST/peer
  // visibility, which mid-graph kernels never need: the graph's LEADING acquireFull
  // (pre-kernel-0) already pulls system state into coherence, and the TRAILING
  // acquireFull writes results back to system at the end. So we cap every INTERIOR
  // edge at AGENT and keep full-L2 strictly at the leading/trailing boundaries. This
  // matches this path's own stated intent ("AGENT ... the only correct choice") and
  // is correctness-preserving for device-local RAW; HIP_PM4_GRAPH_EDGE_FULL_L2
  // restores the old promote-to-SYSTEM behavior for paranoid A/B. Plain ASCII only.
  const bool edgeFullL2 = pm4EnvBool("HIP_PM4_GRAPH_EDGE_FULL_L2", false);
  // Map an HSA fence scope (NONE<AGENT<SYSTEM) to a per-edge GCR mask. Returns 0
  // for NONE to signal "emit no fence" (independent kernels may overlap).
  auto scopeToEdgeMask = [&](uint32_t scope) -> uint32_t {
    if (scope == HSA_FENCE_SCOPE_NONE) return 0u;  // independent -> no fence (overlap)
    // Interior RAW edge: AGENT is sufficient (see note above). Only the opt-out
    // keeps the hardware-conservative SYSTEM promotion.
    if (edgeFullL2 && scope == HSA_FENCE_SCOPE_SYSTEM) return arch.gcrFull;
    return kGcrAgent;
  };
  // Fence after dispatch i = max(this kernel's release, next kernel's acquire).
  auto edgeMaskFor = [&](size_t i) -> uint32_t {
    if (!inheritScope) return edgeMask;
    auto* p = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packets[i]);
    uint32_t rel = extractAqlBits(p->header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                                  HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);
    uint32_t acq = HSA_FENCE_SCOPE_NONE;
    if (i + 1 < numPackets) {
      auto* pn = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packets[i + 1]);
      acq = extractAqlBits(pn->header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                           HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE);
    }
    return scopeToEdgeMask(std::max(rel, acq));
  };
  struct ShadowReg { bool valid = false; uint32_t cnt = 0; uint32_t sig = 0; uint32_t v[8] = {0}; };
  ShadowReg shStartX, shPgm, shScratch, shRsrc, shResLim, shTmpring, shUserD;
  // Returns the ib dword offset of value[0] when the FULL range was emitted (delta
  // off / first emit / layout change), or kNoEmit when nothing/only a sub-range was
  // emitted. The in-place recorder only consumes the full-emit case (delta is forced
  // off in that mode), so value[j] lives at (returned offset + j).
  constexpr uint32_t kNoEmit = 0xFFFFFFFFu;
  auto setshDelta = [&](ShadowReg& sh, uint32_t reg, const uint32_t* v, uint32_t cnt,
                        uint32_t sig, const uint8_t* kinds = nullptr) -> uint32_t {
    if (!delta || !sh.valid || sh.cnt != cnt || sh.sig != sig) {
      const uint32_t valueStart = static_cast<uint32_t>(ib.size()) + 2;  // after hdr + reg
      setsh(reg, v, cnt, kinds);
      sh.valid = true; sh.cnt = cnt; sh.sig = sig;
      for (uint32_t i = 0; i < cnt; ++i) sh.v[i] = v[i];
      return valueStart;
    }
    uint32_t lo = 0;
    while (lo < cnt && v[lo] == sh.v[lo]) ++lo;
    if (lo == cnt) return kNoEmit;  // nothing changed -> emit nothing
    uint32_t hi = cnt;
    while (hi > lo && v[hi - 1] == sh.v[hi - 1]) --hi;
    setsh(reg + lo, v + lo, hi - lo, kinds ? kinds + lo : nullptr);
    for (uint32_t i = lo; i < hi; ++i) sh.v[i] = v[i];
    return kNoEmit;
  };

  // Emit the SET_SH_REG programming for one dispatch (delta-aware). Returns
  // kPm4Ready, or a fallback status (kPm4UnsupportedPermanent / kPm4DeferredScratch)
  // BEFORE emitting anything for this dispatch, so the caller can bail to AQL.
  auto emitRegs = [&](hsa_kernel_dispatch_packet_t* p, size_t pkt) -> int {
    uint8_t type = extractAqlBits(p->header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
    if (type != HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      return kPm4UnsupportedPermanent;  // non-dispatch packet -> AQL fallback
    }
    if (p->kernel_object == 0) return kPm4UnsupportedPermanent;
    auto* kd = reinterpret_cast<const AmdKernelDescriptor*>(p->kernel_object);
    const uint16_t props = kd->kernel_code_properties;

    // Kernarg preload (gfx11/gfx12) changes the entry/SGPR contract; not handled.
    if (kd->kernarg_preload != 0) return kPm4UnsupportedPermanent;

    // G1: scratch. Wire from the queue scratch state when ready, else defer
    // (replay via AQL once so ROCr sizes scratch, then rebuild next launch).
    const bool needsScratch = (kd->private_segment_fixed_size != 0) ||
                              (p->private_segment_size != 0) ||
                              (props & (KCP_PRIVATE_SEGMENT_BUFFER | KCP_FLAT_SCRATCH_INIT));
    if (needsScratch) {
      if (!pm4GraphScratchEnabled()) return kPm4UnsupportedPermanent;   // opt-in only
      if (props & KCP_FLAT_SCRATCH_INIT) return kPm4UnsupportedPermanent;  // not replicated
      // Scratch base/size/V# are queue-runtime values: emit them as placeholders
      // and let specializeFromTemplate() patch them (deferring if not yet sized).
      out.needsScratch = true;
    }

    // Only user SGPRs we can source correctly: private_segment_buffer (scratch
    // V# from the queue), queue_ptr (the amd_queue_t), kernarg_segment_ptr.
    // dispatch_ptr / dispatch_id / private_segment_size are not replicated.
    constexpr uint16_t kSupportedUserSgpr =
        KCP_PRIVATE_SEGMENT_BUFFER | KCP_QUEUE_PTR | KCP_KERNARG_SEGMENT_PTR;
    constexpr uint16_t kUserSgprMask =
        KCP_PRIVATE_SEGMENT_BUFFER | KCP_DISPATCH_PTR | KCP_QUEUE_PTR | KCP_KERNARG_SEGMENT_PTR |
        KCP_DISPATCH_ID | KCP_FLAT_SCRATCH_INIT | KCP_PRIVATE_SEGMENT_SIZE;
    if ((props & kUserSgprMask) & ~kSupportedUserSgpr) return kPm4UnsupportedPermanent;

    const uint32_t dims[8] = {0, 0, 0, p->workgroup_size_x, p->workgroup_size_y,
                              p->workgroup_size_z, 0, 0};
    const uint32_t dimsOff = setshDelta(shStartX, kRegStartX, dims, 8, 0);
    if (inplace && dimsOff != kNoEmit) {
      // dims = {0,0,0, wgX, wgY, wgZ, 0,0} -> workgroup dims at value indices 3,4,5.
      muts.push_back({dimsOff + 3, kMutWgX, static_cast<uint16_t>(pkt)});
      muts.push_back({dimsOff + 4, kMutWgY, static_cast<uint16_t>(pkt)});
      muts.push_back({dimsOff + 5, kMutWgZ, static_cast<uint16_t>(pkt)});
    }
    uint64_t entry = (p->kernel_object + kd->kernel_code_entry_byte_offset) >> 8;
    const uint32_t pgm[2] = {static_cast<uint32_t>(entry), static_cast<uint32_t>(entry >> 32)};
    setshDelta(shPgm, kRegPgmLo, pgm, 2, 0);

    // Scratch base (queue runtime) -> placeholder, patched at specialize time.
    if (needsScratch) {
      const uint32_t scratch[2] = {kPm4PlaceholderSentinel, kPm4PlaceholderSentinel};
      const uint8_t scratchKinds[2] = {kPhScratchBaseLo, kPhScratchBaseHi};
      setshDelta(shScratch, kRegScratchLo, scratch, 2, 0, scratchKinds);
    }

    // RSRC1 verbatim from the KD (exactly what the CP loads). RSRC2 too, EXCEPT
    // LDS_SIZE (bits[23:15]): COV5 leaves it 0 and the runtime/CP programs it from
    // the dispatch packet's group_segment_size. The LDS_SIZE field is encoded in
    // the gfx11 LDS encode granule = 128 dwords = 512 bytes (mesa ac_gpu_info.c
    // lds_encode_granularity). The original /128 (128-byte) recompute over-
    // allocated LDS 4x, which exceeds the per-workgroup limit on large-LDS GEMM
    // kernels and hangs the queue (the 0-LDS microbench never exercised it).
    constexpr uint32_t kLdsEncodeGranuleBytes = 512;
    constexpr uint32_t kLdsSizeMask = 0x00FF8000u;  // COMPUTE_PGM_RSRC2[23:15]
    const uint32_t groupSeg = std::max<uint32_t>(kd->group_segment_fixed_size,
                                                 p->group_segment_size);
    const uint32_t ldsUnits = std::min<uint32_t>(
        (groupSeg + kLdsEncodeGranuleBytes - 1) / kLdsEncodeGranuleBytes, 0x1FFu);
    uint32_t rsrc2 = (kd->compute_pgm_rsrc2 & ~kLdsSizeMask) | ((ldsUnits << 15) & kLdsSizeMask);
    const uint32_t rsrc[2] = {kd->compute_pgm_rsrc1, rsrc2};
    setshDelta(shRsrc, kRegRsrc1, rsrc, 2, 0);
    const uint32_t zero = 0;
    setshDelta(shResLim, kRegResLim, &zero, 1, 0);
    // TMPRING: queue scratch size for scratch kernels (placeholder), real 0 for
    // non-scratch kernels. The sentinel keeps delta distinguishing the two so a
    // non-scratch kernel following a scratch one still re-emits 0.
    if (needsScratch) {
      const uint32_t tmpring = kPm4PlaceholderSentinel;
      const uint8_t tmpringKind = kPhTmpring;
      setshDelta(shTmpring, kRegTmpring, &tmpring, 1, 0, &tmpringKind);
    } else {
      const uint32_t tmpring = 0u;
      setshDelta(shTmpring, kRegTmpring, &tmpring, 1, 0);
    }

    // G2: user SGPRs in kernel_code_properties enable-bit order. The enabled set
    // is the layout signature; a change forces a full re-emit (same offset would
    // otherwise alias a different register). Within a stable layout only the
    // changed sub-range (typically the kernarg pointer) is re-sent.
    uint32_t udata[8] = {0};
    uint8_t ukinds[8];
    for (int i = 0; i < 8; ++i) ukinds[i] = kPhNone;
    uint32_t u = 0;
    if (props & KCP_PRIVATE_SEGMENT_BUFFER) {
      // scratch V# (queue runtime) -> placeholders.
      for (int w = 0; w < 4; ++w) {
        udata[u + w] = kPm4PlaceholderSentinel;
        ukinds[u + w] = static_cast<uint8_t>(kPhScratchVdesc0 + w);
      }
      u += 4;
    }
    if (props & KCP_QUEUE_PTR) {
      // amd_queue_t pointer (per launch stream) -> placeholder.
      udata[u + 0] = kPm4PlaceholderSentinel;
      udata[u + 1] = kPm4PlaceholderSentinel;
      ukinds[u + 0] = kPhQueuePtrLo;
      ukinds[u + 1] = kPhQueuePtrHi;
      u += 2;
    }
    uint32_t kaIdx = 0xFFFFFFFFu;  // udata index of the kernarg pointer, if present
    if (props & KCP_KERNARG_SEGMENT_PTR) {
      // kernarg base is graph-fixed (allocated at instantiate) -> bake it in.
      const uint64_t ka = reinterpret_cast<uint64_t>(p->kernarg_address);
      kaIdx = u;
      udata[u + 0] = static_cast<uint32_t>(ka);
      udata[u + 1] = static_cast<uint32_t>(ka >> 32);
      u += 2;
    }
    if (u > 0) {
      const uint32_t userOff = setshDelta(shUserD, kRegUserD0, udata, u,
                                          props & kSupportedUserSgpr, ukinds);
      if (inplace && userOff != kNoEmit && kaIdx != 0xFFFFFFFFu) {
        muts.push_back({userOff + kaIdx + 0, kMutKernargLo, static_cast<uint16_t>(pkt)});
        muts.push_back({userOff + kaIdx + 1, kMutKernargHi, static_cast<uint16_t>(pkt)});
      }
    }
    return kPm4Ready;
  };

  // Emit the DISPATCH_DIRECT packet that launches the kernel. Kept separate from
  // emitRegs so the reorder path can place the register programming BEFORE the
  // acquire while the launch stays AFTER it (coherence preserved).
  auto emitDispatchPacket = [&](hsa_kernel_dispatch_packet_t* p, size_t pkt) {
    auto* kd = reinterpret_cast<const AmdKernelDescriptor*>(p->kernel_object);
    const bool wave32 = (kd->kernel_code_properties & KCP_WAVEFRONT_SIZE32) != 0;
    const uint32_t dispatchInit = kDispatchBase | (wave32 ? kDispatchW32 : 0u);
    // DISPATCH_DIRECT: USE_THREAD_DIMS -> dim_x is total work-items (AQL grid_size).
    ib.push_back(pm4Hdr(kOpDispatchDirect, 5));
    if (inplace) {
      muts.push_back({static_cast<uint32_t>(ib.size()) + 0, kMutGridX, static_cast<uint16_t>(pkt)});
      muts.push_back({static_cast<uint32_t>(ib.size()) + 1, kMutGridY, static_cast<uint16_t>(pkt)});
      muts.push_back({static_cast<uint32_t>(ib.size()) + 2, kMutGridZ, static_cast<uint16_t>(pkt)});
    }
    ib.push_back(p->grid_size_x);
    ib.push_back(p->grid_size_y ? p->grid_size_y : 1);
    ib.push_back(p->grid_size_z ? p->grid_size_z : 1);
    ib.push_back(dispatchInit);
  };

  // G6: leading full acquire so the first kernel sees writes from prior ops
  // (H2D copy, a previous graph) -- the normal AQL batch carries this on the
  // first packet's acquire scope; the self-contained microbench hid the need.
  acquireFull(arch.gcrFull);

  // Profiling: slot 0 = graph start (before kernel 0). Subsequent slots are the
  // per-edge kernel-end timestamps emitted in the default loop below.
  emitTs();

  // DIAGNOSTIC histogram of the per-edge fence mask chosen, printed once under
  // HIP_PM4_GRAPH_TIMING. Lets us see what scopes a real graph actually carries
  // (e.g. whether INHERIT_SCOPE ever differs from the blanket AGENT default).
  const bool scopeTiming = pm4EnvBool("HIP_PM4_GRAPH_TIMING", false);
  size_t edgeNone = 0, edgeAgent = 0, edgeSys = 0, edgeOther = 0;

  if (!usePws) {
    // DEFAULT: blocking AGENT fence per edge -- [regs i][dispatch i][drain][acquire].
    // Correct, deterministic, and faster than PWS. fullFence/EDGE_GCR override the
    // mask for diagnostics; otherwise it is kGcrAgent (no L2 flush per edge).
    for (size_t i = 0; i < numPackets; ++i) {
      auto* p = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packets[i]);
      int st = emitRegs(p, i);
      if (st != kPm4Ready) { out.status = static_cast<Pm4GraphStatus>(st); return; }
      emitDispatchPacket(p, i);
      const uint32_t mask = edgeMaskFor(i);  // edgeMask, or inherited per-edge scope
      if (scopeTiming) {
        if (mask == 0u) ++edgeNone;
        else if (mask == kGcrAgent) ++edgeAgent;
        else if (mask == arch.gcrFull) ++edgeSys;
        else ++edgeOther;
      }
      if (mask != 0u) {
        partialFlush();        // drain the producer's waves (writes reach L2)
        acquireFull(mask);     // blocking invalidate so the consumer sees them
      }
      // mask == 0 (inherited NONE scope): no dependency recorded -> emit no fence,
      // letting independent kernels overlap (matches the AQL NONE-scope behavior).
      // Profiling: slot i+1 = end of kernel i. When the edge had no fence the
      // BOTTOM_OF_PIPE_TS still drains kernel i's waves before firing.
      emitTs();
    }
    if (scopeTiming) {
      fprintf(stderr, "[PM4 scope] edges=%zu inherit=%d NONE=%zu AGENT=%zu SYSTEM=%zu other=%zu\n",
              numPackets, inheritScope ? 1 : 0, edgeNone, edgeAgent, edgeSys, edgeOther);
    }
  } else if (!reorder) {
    // LEGACY PWS (HIP_PM4_GRAPH_PWS): deferred-wait fence. UNSAFE (see above).
    for (size_t i = 0; i < numPackets; ++i) {
      auto* p = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packets[i]);
      int st = emitRegs(p, i);
      if (st != kPm4Ready) { out.status = static_cast<Pm4GraphStatus>(st); return; }
      emitDispatchPacket(p, i);
      partialFlush();
      releaseMemPws(kGcrAgent);
      acquirePws();
    }
  } else {
    // LEGACY PWS + reorder: program kernel i's registers BETWEEN i-1's release and
    // acquire. WARNING: reorder is a measured NO-OP on RDNA3 (the CP ME is in-order)
    // AND the PWS fence it builds on is UNSAFE. Experimentation only.
    auto* p0 = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packets[0]);
    int st = emitRegs(p0, 0);
    if (st != kPm4Ready) { out.status = static_cast<Pm4GraphStatus>(st); return; }
    emitDispatchPacket(p0, 0);
    for (size_t i = 1; i < numPackets; ++i) {
      partialFlush();              // drain kernel i-1 (its waves retire)
      releaseMemPws(kGcrAgent);    // async AGENT flush of kernel i-1 (PWS armed)
      auto* p = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packets[i]);
      st = emitRegs(p, i);         // program kernel i during the flush / drain
      if (st != kPm4Ready) { out.status = static_cast<Pm4GraphStatus>(st); return; }
      acquirePws();                // wait kernel i-1's flush counter
      emitDispatchPacket(p, i);    // launch kernel i (coherent + registers latched)
    }
  }
  // Final full-L2 writeback so the graph's results are system-visible (matches the
  // SYSTEM-scope release the normal AQL batch forces on its last packet).
  partialFlush();
  acquireFull(arch.gcrFull);

  out.status = kPm4Ready;  // fully encoded; queue specialization happens later
}

// Patch a capture-time template's queue-dependent placeholders from THIS vdev's
// queue, then alloc + DMA-upload the IB. No CPU re-encode. Returns kPm4DeferredScratch
// when the graph needs scratch the queue has not sized yet (replay via AQL once,
// which sizes it, then a later specialize succeeds).
VirtualGPU::Pm4GraphIb VirtualGPU::specializeFromTemplate(const Pm4GraphTemplate& t,
                                                          bool deviceScoped) {
  Pm4GraphIb out;
  out.status = kPm4UnsupportedPermanent;
  if (t.status != kPm4Ready) { out.status = t.status; return out; }

  const bool pm4Timing = pm4EnvBool("HIP_PM4_GRAPH_TIMING", false);
  const uint64_t tStart = pm4Timing ? amd::Os::timeNanos() : 0;
  auto* aq = reinterpret_cast<amd_queue_t*>(gpu_queue_);
  const uint64_t scratchBase = aq->scratch_backing_memory_location;
  const uint32_t tmpring = aq->compute_tmpring_size;
  if (t.needsScratch && !((tmpring != 0) && (scratchBase != 0))) {
    out.status = kPm4DeferredScratch;
    return out;
  }

  std::vector<uint32_t> ib = t.dwords;  // patch placeholders into a private copy
  const uint64_t sbase = scratchBase >> 8;
  const uint64_t qptr = reinterpret_cast<uint64_t>(gpu_queue_);
  for (const auto& ph : t.patches) {
    uint32_t v = 0;
    switch (ph.kind) {
      case kPhScratchBaseLo: v = static_cast<uint32_t>(sbase); break;
      case kPhScratchBaseHi: v = static_cast<uint32_t>(sbase >> 32); break;
      case kPhTmpring:       v = tmpring; break;
      case kPhScratchVdesc0:
      case kPhScratchVdesc1:
      case kPhScratchVdesc2:
      case kPhScratchVdesc3:
        v = aq->scratch_resource_descriptor[ph.kind - kPhScratchVdesc0];
        break;
      case kPhQueuePtrLo: v = static_cast<uint32_t>(qptr); break;
      case kPhQueuePtrHi: v = static_cast<uint32_t>(qptr >> 32); break;
      default: break;
    }
    ib[ph.offset] = v;
  }

  // Per-kernel profiling: allocate a CPU fine-grain buffer
  // the CP can write the GPU clock into and the host can read directly, then patch
  // each RELEASE_MEM's ADDRESS_LO/HI to its slot. Done BEFORE the upload below so
  // the patched addresses ride along in the uploaded IB. Skipped for device-scoped
  // (shared) IBs -- profiling forces the per-stream path, so this is never shared.
  if (t.instrumented && !t.tsAddrOff.empty() && !deviceScoped) {
    const uint32_t slots = static_cast<uint32_t>(t.tsAddrOff.size());
    const size_t bytes = (static_cast<size_t>(slots) * kPm4TsStride + 0xFFF) & ~size_t(0xFFF);
    void* ts = nullptr;
    if (Hsa::memory_pool_allocate(roc_device_.getCpuFineGrainPool(), bytes, 0, &ts) ==
        HSA_STATUS_SUCCESS) {
      hsa_agent_t agents[2] = {roc_device_.getBackendDevice(), roc_device_.getCpuAgent()};
      Hsa::agents_allow_access(2, agents, nullptr, ts);
      memset(ts, 0, bytes);
      const uint64_t base = reinterpret_cast<uint64_t>(ts);
      for (uint32_t k = 0; k < slots; ++k) {
        const uint64_t addr = base + static_cast<uint64_t>(k) * kPm4TsStride;
        ib[t.tsAddrOff[k] + 0] = static_cast<uint32_t>(addr);
        ib[t.tsAddrOff[k] + 1] = static_cast<uint32_t>(addr >> 32);
      }
      out.tsBuf = ts;
      out.tsCount = slots;
    } else {
      LogError("PM4 graph profile: timestamp buffer allocation failed");
    }
  }

  void* dev = allocExecIbFromData(ib.data(), static_cast<uint32_t>(ib.size()), deviceScoped);
  if (dev == nullptr) {
    if (out.tsBuf != nullptr) { Hsa::memory_pool_free(out.tsBuf); out.tsBuf = nullptr; }
    return out;
  }
  out.ib = dev;
  out.dw = static_cast<uint32_t>(ib.size());
  out.status = kPm4Ready;
  if (pm4Timing) {
    ClPrint(amd::LOG_INFO, amd::LOG_CODE,
            "[pm4-timing] specialize %zu dispatches -> %u dw: patch+alloc+upload=%.1f us (arena=%d)",
            t.numPackets, out.dw, (amd::Os::timeNanos() - tStart) / 1000.0,
            pm4ArenaOwns(dev) ? 1 : 0);
  }
  ClPrint(amd::LOG_INFO, amd::LOG_AQL,
          "PM4 graph: compiled %zu dispatches -> IB %u dwords @%p", t.numPackets, out.dw, out.ib);
  return out;
}

// Full build = CPU encode + specialize+upload on THIS vdev's queue. Used on the
// launch path when no capture-time template is available (e.g. post-mutation
// rebuild and HIP_PM4_GRAPH_BUILD_AFTER). With a template the cache miss path
// calls specializeFromTemplate() directly and skips the encode.
VirtualGPU::Pm4GraphIb VirtualGPU::buildPm4GraphIb(void* const* packets, size_t numPackets) {
  const bool pm4Timing = pm4EnvBool("HIP_PM4_GRAPH_TIMING", false);
  const uint64_t tEncodeStart = pm4Timing ? amd::Os::timeNanos() : 0;
  Pm4GraphTemplate t;
  encodePm4GraphTemplate(packets, numPackets, t);
  const uint64_t tEncodeEnd = pm4Timing ? amd::Os::timeNanos() : 0;
  Pm4GraphIb out = specializeFromTemplate(t);
  if (pm4Timing && out.status == kPm4Ready) {
    const uint64_t tAllocEnd = amd::Os::timeNanos();
    ClPrint(amd::LOG_INFO, amd::LOG_CODE,
            "[pm4-timing] build %zu dispatches -> %u dw: encode=%.1f us  alloc+upload=%.1f us "
            "(arena=%d)",
            numPackets, out.dw, (tEncodeEnd - tEncodeStart) / 1000.0,
            (tAllocEnd - tEncodeEnd) / 1000.0, pm4ArenaOwns(out.ib) ? 1 : 0);
  }
  return out;
}

// Capture-time entry (called from hipGraphInstantiate via the base virtual): encode
// the packets into a heap-owned template stamped with their content key. Returns
// nullptr (and allocates nothing) if the graph is not PM4-replayable, so the launch
// path stays on the existing build/AQL fallback. Queue-independent, so any vdev of
// the graph's device produces an identical template.
void* VirtualGPU::buildPm4GraphTemplate(void* const* packets, size_t numPackets) {
  if (!pm4GraphActive() || numPackets == 0) return nullptr;
  // A/B + safety toggle: HIP_PM4_GRAPH_TEMPLATE=0 disables capture-time encode so
  // the first replay pays the full build (the pre-template behavior). Default on.
  const char* te = getenv("HIP_PM4_GRAPH_TEMPLATE");
  if (te != nullptr && te[0] == '0') return nullptr;
  auto* t = new Pm4GraphTemplate();
  const bool pm4Timing = pm4EnvBool("HIP_PM4_GRAPH_TIMING", false);
  const uint64_t t0 = pm4Timing ? amd::Os::timeNanos() : 0;
  encodePm4GraphTemplate(packets, numPackets, *t);
  if (t->status != kPm4Ready) {
    delete t;
    return nullptr;
  }
  t->key = pm4GraphKey(packets, numPackets);
  t->skeletonKey = pm4GraphSkeletonKey(packets, numPackets);
  if (pm4Timing) {
    ClPrint(amd::LOG_INFO, amd::LOG_CODE,
            "[pm4-timing] capture-time encode %zu dispatches -> %zu dw: encode=%.1f us",
            numPackets, t->dwords.size(), (amd::Os::timeNanos() - t0) / 1000.0);
  }
  // GraphExec-owned shared IB: a queue-independent graph (no placeholders) gets ONE
  // device-scoped IB shared by every stream that replays it (no per-stream
  // specialize/upload). The build is DEFERRED to the first replay on the executing
  // vdev (see tryReplayPm4Graph): specializing+uploading here, on the null-stream
  // vdev at instantiate, produces an IB that is not guaranteed visible to a
  // different execution queue's CP fetch and replays as garbage. Just mark intent.
  if (t->shareable() && pm4GraphSharedIbEnabled()) {
    t->wantSharedIb = true;
  }
  return t;
}

void VirtualGPU::freePm4GraphTemplate(void* tmpl) {
  auto* t = static_cast<Pm4GraphTemplate*>(tmpl);
  if (t == nullptr) {
    return;
  }
  // The shared IB is device-scoped (allocated from the device pool, not the per-vdev
  // arena), so it is released directly here regardless of which vdev built it.
  if (t->sharedIb != nullptr) {
    Hsa::memory_pool_free(t->sharedIb);
    t->sharedIb = nullptr;
  }
  delete t;
}

bool VirtualGPU::tryReplayPm4Graph(void* const* packets, size_t numPackets, bool blocking,
                                   bool attach_signal, uint64_t recordedPacketVersion,
                                   bool allowBuild, const Pm4GraphTemplate* tmpl) {
  if (numPackets == 0) return false;

  // Profiler visibility fallback: rocprofiler intercepts AQL dispatch packets, but a
  // PM4 IB replay submits ONE vendor packet that hides the N dispatches, so a
  // profiled run would capture nothing. When dispatch tracing is armed -- whether
  // from process start or attached at runtime -- fall back to AQL so every kernel
  // stays profilable. The caller treats a false return as "run this launch on AQL",
  // so returning here is the entire fallback (it also covers the per-kernel profiling
  // case: if a real profiler is attached, defer to it instead of the HIP-log path).
  if (pm4TracingArmed()) {
    return false;
  }

  // Per-kernel GPU-clock profiling (LOG_INFO+LOG_AQL) instruments a per-stream IB
  // with timestamp packets and reads them back blocking. That is incompatible
  // with the cross-stream shared IB (one buffer, many concurrent writers) and the
  // double-buffered in-place resident, so force the per-stream specialize/keycache
  // path where each IB owns its own TS buffer.
  const bool profileTs = pm4GraphProfileEnabled();

  // In-place double-buffered fast path (HIP_PM4_GRAPH_INPLACE): for a graph mutated
  // only in scalar node params, patch + ping-pong a resident IB instead of building
  // a new one. Returns true if it submitted; false (not eligible) falls through to
  // the keycache / shared-IB / rebuild path below. Skipped in build-after mode
  // (allowBuild=false) where this launch intentionally goes AQL.
  if (allowBuild && !profileTs &&
      tryReplayPm4GraphInplace(packets, numPackets, blocking, attach_signal,
                               recordedPacketVersion, tmpl)) {
    return true;
  }

  // Fast path: same recorded packet set as the previous launch -> reuse the cached
  // IB pointer without recomputing the O(N) content hash (HIP_PM4_GRAPH_KEYCACHE).
  // Validated by the graph-supplied recorded packet set version: nonzero, unique
  // per GraphExec instantiation+batch, and bumped on ANY packet mutation (param
  // update, enable/disable, re-capture). This is a RELIABLE invalidation, not a
  // heuristic. version 0 (non-graph caller) always takes the slow path. Deferred-
  // scratch entries are never cached here (status flips), so the fast path only
  // serves kPm4Ready.
  const bool keyCache = pm4GraphKeyCacheEnabled();
  if (keyCache && recordedPacketVersion != 0) {
    if (pm4IbCacheValid_ && pm4IbCacheVersion_ == recordedPacketVersion &&
        pm4IbCacheEntry_ != nullptr && pm4IbCacheEntry_->status == kPm4Ready) {
      submitPm4Ib(*pm4IbCacheEntry_, blocking, attach_signal);
      return true;
    }
  }

  // GraphExec-owned shared IB fast path (#5): a queue-independent graph already has
  // its device-scoped IB built (at instantiate, shared across every stream). This
  // stream just submits it -- no per-stream specialize/upload, no entry in the
  // content-key cache (the shared IB's lifetime is the GraphExec, not this vdev, so
  // it is never inserted into pm4Graphs_ where it could outlive its owner). The key
  // match guards against a mutated/stale template; the O(N) hash is paid only when
  // the keycache misses (first launch + after a mutation), then armed for O(1) reuse.
  // Lazy build of the GraphExec-owned shared IB on THIS (executing) vdev. Deferred
  // from instantiate so the SDMA upload and the CP fetch share a queue ordering
  // edge; building it at instantiate on the null-stream vdev leaves the bytes
  // potentially invisible to this queue's CP and replays garbage. Built once,
  // guarded by sharedReady; concurrent first-replays on other streams that lose the
  // race simply fall through to their per-stream IB for that one launch.
  if (tmpl != nullptr && tmpl->wantSharedIb && !profileTs && pm4GraphSharedIbEnabled() &&
      !tmpl->sharedReady.load(std::memory_order_acquire) &&
      tmpl->key == pm4GraphKey(packets, numPackets)) {
    auto* mt = const_cast<Pm4GraphTemplate*>(tmpl);
    std::unique_lock<std::mutex> lk(mt->sharedMtx, std::try_to_lock);
    if (lk.owns_lock() && !mt->sharedReady.load(std::memory_order_relaxed)) {
      Pm4GraphIb shared = specializeFromTemplate(*tmpl, /*deviceScoped=*/true);
      if (shared.status == kPm4Ready) {
        mt->sharedIb = shared.ib;
        mt->sharedDw = shared.dw;
        ClPrint(amd::LOG_INFO, amd::LOG_AQL,
                "PM4 graph: shared IB %u dwords @%p (device-scoped, lazy on exec vdev, %zu dispatches)",
                mt->sharedDw, mt->sharedIb, numPackets);
      }
      mt->sharedReady.store(true, std::memory_order_release);
    }
  }
  if (tmpl != nullptr && tmpl->sharedIb != nullptr && !profileTs && pm4GraphSharedIbEnabled() &&
      tmpl->key == pm4GraphKey(packets, numPackets)) {
    pm4SharedRef_.ib = tmpl->sharedIb;
    pm4SharedRef_.dw = tmpl->sharedDw;
    pm4SharedRef_.status = kPm4Ready;
    pm4SharedRef_.owned = false;  // referenced only; the GraphExec frees the storage
    if (keyCache && recordedPacketVersion != 0) {
      pm4IbCacheValid_ = true;
      pm4IbCacheVersion_ = recordedPacketVersion;
      pm4IbCacheEntry_ = &pm4SharedRef_;
    }
    submitPm4Ib(pm4SharedRef_, blocking, attach_signal);
    return true;
  }

  Pm4GraphIb* g = findOrBuildPm4Graph(packets, numPackets, recordedPacketVersion, allowBuild, tmpl);
  if (g == nullptr) {
    return false;  // cache miss with allowBuild=false, or unsupported/deferred build
  }
  submitPm4Ib(*g, blocking, attach_signal);
  return true;
}

// ================================================================================================
VirtualGPU::Pm4GraphIb* VirtualGPU::findOrBuildPm4Graph(void* const* packets, size_t numPackets,
                                                        uint64_t recordedPacketVersion,
                                                        bool allowBuild,
                                                        const Pm4GraphTemplate* tmpl) {
  // With a capture-time template, a cache miss only needs specialize+upload (patch
  // the queue-dependent placeholders + DMA); the CPU encode already ran at
  // instantiate. Without one, fall back to a full build (encode+specialize). The
  // template must match these exact packets (same recorded set) -- enforced by the
  // hip layer keying it to the batch -- so it is ignored if its dword count is 0.
  const uint64_t key = pm4GraphKey(packets, numPackets);
  // Only use the template if it was encoded from THESE exact packets: a stale
  // template (graph mutated after instantiate -> different bytes) or a disabled-
  // node filtered subset has a different key and must fall back to a full build.
  // Per-stream path: specialize a per-stream IB from the template (queue-dependent
  // graphs: scratch / queue_ptr), or full-build when there is no template. The
  // GraphExec-owned SHARED IB (queue-independent graphs) is handled earlier in
  // tryReplayPm4Graph and never reaches here.
  const bool useTmpl = (tmpl != nullptr) && (tmpl->status == kPm4Ready) && (tmpl->key == key);
  auto it = pm4Graphs_.find(key);
  if (it == pm4Graphs_.end()) {
    if (!allowBuild) {
      return nullptr;  // build-after mode: this launch goes AQL; build happens post-submit
    }
    Pm4GraphIb built = useTmpl ? specializeFromTemplate(*tmpl) : buildPm4GraphIb(packets, numPackets);
    it = pm4Graphs_.emplace(key, built).first;
    pm4GraphKeyOrder_.push_back(key);
    // A mutated graph lands here every time its packet content changes, leaving
    // the previous content's IB unreferenced. Bound the VRAM held by dead IBs.
    evictPm4GraphsIfNeeded(&it->second);
  }
  Pm4GraphIb& g = it->second;
  // Deferred scratch: a prior launch fell back to AQL (which sizes the queue
  // scratch); rebuild now that scratch may be ready. Free the stale IB first so
  // the rebuild does not orphan it. specializeFromTemplate re-checks readiness.
  if (g.status == kPm4DeferredScratch && allowBuild) {
    freePm4GraphIb(g);
    g = useTmpl ? specializeFromTemplate(*tmpl) : buildPm4GraphIb(packets, numPackets);
  }
  if (g.status != kPm4Ready) {
    return nullptr;
  }
  // Arm the fast path for the next launch (ready IB + valid version).
  if (pm4GraphKeyCacheEnabled() && recordedPacketVersion != 0) {
    pm4IbCacheValid_ = true;
    pm4IbCacheVersion_ = recordedPacketVersion;
    pm4IbCacheEntry_ = &g;
  }
  return &g;
}

// ================================================================================================
void VirtualGPU::submitPm4Ib(const Pm4GraphIb& g, bool blocking, bool attach_signal,
                             hsa_signal_t* outSig) {
  // Per-kernel profiling needs the IB to retire before the host can read the GPU
  // clock slots, so force a blocking wait for instrumented IBs (HIP-log facility,
  // not the fast production path -- the extra sync is acceptable here).
  if (g.tsBuf != nullptr && g.tsCount > 0) {
    blocking = true;
  }
  // Clear the timestamp slots before the GPU writes them so a stale value from a
  // previous launch (the IB + its TS buffer are cached and reused) can never be
  // mistaken for this launch's data; the post-wait poll then keys off slot[n-1].
  if (g.tsBuf != nullptr && g.tsCount > 0) {
    memset(g.tsBuf, 0, static_cast<size_t>(g.tsCount) * kPm4TsStride);
  }
  bool attachSignal = timestamp_ != nullptr || attach_signal;
  hsa_signal_t sig = Barriers().ActiveSignal(kInitSignalValueOne, timestamp_, attachSignal);
  if (outSig != nullptr) *outSig = sig;

  PwsVendorPkt pkt;
  buildPwsVendorPkt(&pkt, g.ib, g.dw);
  pkt.completion_signal = sig;  // CP signals it after the whole IB completes

  const uint32_t queueMask = gpu_queue_->size - 1;
  uint64_t index = Hsa::queue_add_write_index_screlease(gpu_queue_, 1);
  while ((index - Hsa::queue_load_read_index_scacquire(gpu_queue_)) >= queueMask) {
    amd::Os::yield();
  }
  PwsVendorPkt* slot =
      &(reinterpret_cast<PwsVendorPkt*>(gpu_queue_->base_address))[index & queueMask];
  memcpy(reinterpret_cast<uint8_t*>(slot) + sizeof(uint32_t),
         reinterpret_cast<uint8_t*>(&pkt) + sizeof(uint32_t), sizeof(pkt) - sizeof(uint32_t));
  packet_store_release(reinterpret_cast<uint32_t*>(slot), pkt.header, pkt.ven_hdr);
  Hsa::signal_store_screlease(gpu_queue_->doorbell_signal, index);
  hasPendingDispatch_ = true;
  TrackQueueProgress(pkt, index);

  if (blocking) {
    Barriers().WaitCurrent();
  }

  if (g.tsBuf != nullptr && g.tsCount > 0) {
    reportPm4Timestamps(g);
  }
}

// Read the per-kernel GPU-clock slots written by the instrumented IB and ClPrint
// per-kernel timing. Slot 0 is the graph start, slot k (k>0) the end of kernel k-1,
// so the GPU-domain delta between consecutive slots is kernel (k-1)'s wall time.
// Ticks are GPU clock cycles; getGpuTicksToTime() scales them to nanoseconds.
void VirtualGPU::reportPm4Timestamps(const Pm4GraphIb& g) {
  if (g.tsBuf == nullptr || g.tsCount < 2) return;
  const auto* slots = reinterpret_cast<const volatile uint64_t*>(g.tsBuf);
  const uint32_t n = g.tsCount;

  // The RELEASE_MEM(memory_controller) clock writes are posted: they can still be
  // in flight when the queue completion signal we waited on fires. EOP events drain
  // in pipe order, so the LAST slot lands last -- spin briefly until it is non-zero
  // (cleared before submit), guaranteeing all earlier slots are visible too.
  for (int spins = 0; slots[n - 1] == 0 && spins < 100000; ++spins) {
    amd::Os::yield();
  }

  // data_sel=send_gpu_clock_counter ticks in the AGENT timestamp domain (e.g. gfx12
  // reports 100 MHz here while the SYSTEM frequency, which getGpuTicksToTime() uses,
  // is 1 GHz), so convert with the agent frequency, queried once and cached.
  if (pm4TsNsPerTick_ == 0.0) {
    uint64_t agentFreq = 0;
    if (Hsa::agent_get_info(roc_device_.getBackendDevice(),
                            (hsa_agent_info_t)HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY,
                            &agentFreq) == HSA_STATUS_SUCCESS && agentFreq != 0) {
      pm4TsNsPerTick_ = 1e9 / static_cast<double>(agentFreq);
    } else {
      pm4TsNsPerTick_ = Timestamp::getGpuTicksToTime();
    }
  }
  const double nsPerTick = pm4TsNsPerTick_;

  const double totalNs = static_cast<double>(slots[n - 1] - slots[0]) * nsPerTick;
  ClPrint(amd::LOG_INFO, amd::LOG_AQL,
          "HipGraph launch PM4: %u dispatches, total %.3f us (per-kernel GPU timestamps)",
          n - 1, totalNs / 1000.0);
  // Slot k (k>=1) ends kernel k-1, which is dispatch packet k-1, so the name comes
  // from the same index in the launch's per-packet kernel-name vector (when present).
  const auto* names = pm4LaunchKernelNames_;
  for (uint32_t k = 1; k < n; ++k) {
    const double ns = static_cast<double>(slots[k] - slots[k - 1]) * nsPerTick;
    const char* name = (names != nullptr && (k - 1) < names->size() &&
                        (*names)[k - 1] != nullptr) ? (*names)[k - 1]->c_str() : "?";
    ClPrint(amd::LOG_INFO, amd::LOG_AQL,
            "HipGraph launch PM4:   kernel %u %s: %.3f us", k - 1, name, ns / 1000.0);
  }
}

// ================================================================================================
// In-place double-buffered resident IB (HIP_PM4_GRAPH_INPLACE). For a graph mutated
// only in scalar node params, patch the changed dwords of a resident device IB and
// ping-pong between two slots instead of rebuilding a fresh IB + churning the LRU.

bool VirtualGPU::pm4GraphInplaceEnabled() {
  if (pm4GraphInplaceState_ < 0) {
    pm4GraphInplaceState_ = pm4EnvBool("HIP_PM4_GRAPH_INPLACE", false) ? 1 : 0;
  }
  return pm4GraphInplaceState_ == 1;
}

uint32_t VirtualGPU::mutFieldValue(const Pm4MutField& mf, void* const* packets) {
  auto* p = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packets[mf.pkt]);
  const uint64_t ka = reinterpret_cast<uint64_t>(p->kernarg_address);
  switch (mf.kind) {
    case kMutKernargLo: return static_cast<uint32_t>(ka);
    case kMutKernargHi: return static_cast<uint32_t>(ka >> 32);
    case kMutGridX:     return p->grid_size_x;
    case kMutGridY:     return p->grid_size_y ? p->grid_size_y : 1;  // matches emitDispatchPacket
    case kMutGridZ:     return p->grid_size_z ? p->grid_size_z : 1;
    case kMutWgX:       return p->workgroup_size_x;
    case kMutWgY:       return p->workgroup_size_y;
    case kMutWgZ:       return p->workgroup_size_z;
    default:            return 0;
  }
}

void VirtualGPU::freeInplaceResident() {
  for (int s = 0; s < 2; ++s) {
    if (pm4Inplace_.slot[s] != nullptr) {
      Hsa::memory_pool_free(pm4Inplace_.slot[s]);
    }
  }
  if (pm4Inplace_.stage != nullptr) {
    Hsa::memory_pool_free(pm4Inplace_.stage);
  }
  pm4Inplace_ = Pm4InplaceResident{};
}

bool VirtualGPU::buildInplaceResident(void* const* packets, size_t numPackets,
                                      const Pm4GraphTemplate* tmpl, uint64_t skeletonKey) {
  if (tmpl == nullptr || tmpl->status != kPm4Ready || tmpl->dwords.empty()) return false;
  // Specialize the queue-dependent placeholders for THIS vdev's queue (same as
  // specializeFromTemplate). If scratch is not sized yet, bail and let the normal
  // path run + size it; a later launch will build the resident.
  auto* aq = reinterpret_cast<amd_queue_t*>(gpu_queue_);
  const uint64_t scratchBase = aq->scratch_backing_memory_location;
  const uint32_t tmpringSz = aq->compute_tmpring_size;
  if (tmpl->needsScratch && !((tmpringSz != 0) && (scratchBase != 0))) return false;

  std::vector<uint32_t> hostd = tmpl->dwords;
  const uint64_t sbase = scratchBase >> 8;
  const uint64_t qptr = reinterpret_cast<uint64_t>(gpu_queue_);
  for (const auto& ph : tmpl->patches) {
    uint32_t v = 0;
    switch (ph.kind) {
      case kPhScratchBaseLo: v = static_cast<uint32_t>(sbase); break;
      case kPhScratchBaseHi: v = static_cast<uint32_t>(sbase >> 32); break;
      case kPhTmpring:       v = tmpringSz; break;
      case kPhScratchVdesc0:
      case kPhScratchVdesc1:
      case kPhScratchVdesc2:
      case kPhScratchVdesc3:
        v = aq->scratch_resource_descriptor[ph.kind - kPhScratchVdesc0];
        break;
      case kPhQueuePtrLo: v = static_cast<uint32_t>(qptr); break;
      case kPhQueuePtrHi: v = static_cast<uint32_t>(qptr >> 32); break;
      default: break;
    }
    hostd[ph.offset] = v;
  }
  // Bake the CURRENT packet values into the mutable slots so the first active slot
  // reflects this launch (the template's baked values are the instantiate-time set).
  uint32_t spanLo = 0xFFFFFFFFu, spanHi = 0;
  for (const auto& mf : tmpl->mutFields) {
    hostd[mf.offset] = mutFieldValue(mf, packets);
    if (mf.offset < spanLo) spanLo = mf.offset;
    if (mf.offset > spanHi) spanHi = mf.offset;
  }

  const uint32_t dw = static_cast<uint32_t>(hostd.size());
  const size_t bytes = (static_cast<size_t>(dw) * 4 + 0xFFF) & ~static_cast<size_t>(0xFFF);
  hsa_agent_t gpu = roc_device_.getBackendDevice();
  void* stage = nullptr;
  if (Hsa::memory_pool_allocate(roc_device_.getCpuFineGrainPool(), bytes, 0, &stage) !=
      HSA_STATUS_SUCCESS) {
    return false;
  }
  hsa_agent_t agents[2] = {gpu, roc_device_.getCpuAgent()};
  Hsa::agents_allow_access(2, agents, nullptr, stage);
  memcpy(stage, hostd.data(), static_cast<size_t>(dw) * 4);

  void* slot[2] = {nullptr, nullptr};
  for (int s = 0; s < 2; ++s) {
    if (Hsa::memory_pool_allocate(roc_device_.getGpuvmSegment(), bytes,
                                  HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, &slot[s]) !=
        HSA_STATUS_SUCCESS) {
      if (slot[0] != nullptr && s == 1) Hsa::memory_pool_free(slot[0]);
      Hsa::memory_pool_free(stage);
      return false;
    }
    Hsa::agents_allow_access(1, &gpu, nullptr, slot[s]);
    hsa_signal_t cs;
    Hsa::signal_create(1, 0, nullptr, &cs);
    Hsa::memory_async_copy(slot[s], gpu, stage, roc_device_.getCpuAgent(),
                           static_cast<size_t>(dw) * 4, 0, nullptr, cs);
    while (Hsa::signal_wait_scacquire(cs, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                      HSA_WAIT_STATE_BLOCKED) >= 1) {}
    Hsa::signal_destroy(cs);
  }

  freeInplaceResident();  // release any previous graph's resident
  pm4Inplace_.valid = true;
  pm4Inplace_.tmpl = tmpl;
  pm4Inplace_.skeletonKey = skeletonKey;
  pm4Inplace_.version = 0;  // set by caller after the first submit
  pm4Inplace_.slot[0] = slot[0];
  pm4Inplace_.slot[1] = slot[1];
  pm4Inplace_.dw = dw;
  pm4Inplace_.activeSlot = 0;
  pm4Inplace_.spanLo = (spanLo <= spanHi) ? spanLo : 0;
  pm4Inplace_.spanHi = (spanLo <= spanHi) ? spanHi : 0;
  pm4Inplace_.stage = stage;
  pm4Inplace_.mutFields = tmpl->mutFields;  // private copy; never deref tmpl again
  ClPrint(amd::LOG_INFO, amd::LOG_AQL,
          "PM4 graph: in-place resident %u dw, 2 slots, %zu mutable fields, span [%u,%u]",
          dw, pm4Inplace_.mutFields.size(), pm4Inplace_.spanLo, pm4Inplace_.spanHi);
  return true;
}

void VirtualGPU::patchInplaceSlot(uint32_t slot, void* const* packets, size_t numPackets) {
  (void)numPackets;
  uint32_t* stage = reinterpret_cast<uint32_t*>(pm4Inplace_.stage);
  // Write the COMPLETE current mutable set into the shared stage, then SDMA the
  // contiguous span covering all mutable fields into this slot. Writing the full
  // set (not just the deltas) makes the slot fully current regardless of which
  // values the OTHER slot last carried -- so a single shared stage is correct.
  for (const auto& mf : pm4Inplace_.mutFields) {
    stage[mf.offset] = mutFieldValue(mf, packets);
  }
  if (pm4Inplace_.mutFields.empty()) return;  // nothing to patch (skeleton-only IB)
  const uint32_t lo = pm4Inplace_.spanLo;
  const uint32_t hi = pm4Inplace_.spanHi;
  const size_t off = static_cast<size_t>(lo) * 4;
  const size_t len = static_cast<size_t>(hi - lo + 1) * 4;
  hsa_agent_t gpu = roc_device_.getBackendDevice();
  hsa_signal_t cs;
  Hsa::signal_create(1, 0, nullptr, &cs);
  Hsa::memory_async_copy(reinterpret_cast<uint8_t*>(pm4Inplace_.slot[slot]) + off, gpu,
                         reinterpret_cast<uint8_t*>(pm4Inplace_.stage) + off,
                         roc_device_.getCpuAgent(), len, 0, nullptr, cs);
  while (Hsa::signal_wait_scacquire(cs, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                    HSA_WAIT_STATE_BLOCKED) >= 1) {}
  Hsa::signal_destroy(cs);
}

bool VirtualGPU::tryReplayPm4GraphInplace(void* const* packets, size_t numPackets, bool blocking,
                                          bool attach_signal, uint64_t recordedPacketVersion,
                                          const Pm4GraphTemplate* tmpl) {
  if (!pm4GraphInplaceEnabled()) return false;
  if (tmpl == nullptr || tmpl->status != kPm4Ready) return false;
  if (recordedPacketVersion == 0) return false;  // no reliable invalidation signal
  if (pm4EnvBool("HIP_PM4_GRAPH_PWS", false)) return false;  // PWS fence path not supported

  const uint64_t skel = pm4GraphSkeletonKey(packets, numPackets);

  auto submitSlot = [&](uint32_t s) {
    Pm4GraphIb g;
    g.ib = pm4Inplace_.slot[s];
    g.dw = pm4Inplace_.dw;
    g.status = kPm4Ready;
    g.owned = false;
    hsa_signal_t sig{0};
    submitPm4Ib(g, blocking, attach_signal, &sig);
    pm4Inplace_.lastCompletion[s] = sig;
  };

  // Different graph / structural change -> (re)build the resident, or fall back.
  if (!pm4Inplace_.valid || pm4Inplace_.tmpl != tmpl || pm4Inplace_.skeletonKey != skel) {
    if (!buildInplaceResident(packets, numPackets, tmpl, skel)) return false;
    pm4Inplace_.version = recordedPacketVersion;
    submitSlot(pm4Inplace_.activeSlot);
    return true;
  }

  // Same structure, same version -> submit the active slot (O(1), like the keycache).
  if (pm4Inplace_.version == recordedPacketVersion) {
    submitSlot(pm4Inplace_.activeSlot);
    return true;
  }

  // Scalar mutation: patch the idle slot, swap, submit. Guard against patching a
  // slot whose previous submission is still in flight (usually already complete).
  const uint32_t idle = pm4Inplace_.activeSlot ^ 1u;
  if (pm4Inplace_.lastCompletion[idle].handle != 0) {
    while (Hsa::signal_wait_scacquire(pm4Inplace_.lastCompletion[idle], HSA_SIGNAL_CONDITION_LT, 1,
                                      UINT64_MAX, HSA_WAIT_STATE_BLOCKED) >= 1) {}
  }
  patchInplaceSlot(idle, packets, numPackets);
  pm4Inplace_.activeSlot = idle;
  pm4Inplace_.version = recordedPacketVersion;
  submitSlot(idle);
  return true;
}

// ================================================================================================
void VirtualGPU::prebuildPm4Graph(void* const* packets, size_t numPackets,
                                  uint64_t recordedPacketVersion, const Pm4GraphTemplate* tmpl) {
  // Build + insert + arm the IB without submitting. Called right after the AQL
  // fallback submit (HIP_PM4_GRAPH_BUILD_AFTER), which has already sized scratch,
  // so the build is never scratch-deferred. The next replay finds it armed. With a
  // capture-time template this is just specialize+upload (no CPU re-encode).
  (void)findOrBuildPm4Graph(packets, numPackets, recordedPacketVersion, /*allowBuild=*/true, tmpl);
}

// ================================================================================================
void VirtualGPU::dispatchBarrierValuePacket(uint16_t packetHeader, bool resolveDepSignal,
                                            hsa_signal_t signal, hsa_signal_value_t value,
                                            hsa_signal_value_t mask, hsa_signal_condition32_t cond,
                                            bool skipTs, hsa_signal_t completionSignal) {
  uint16_t rest = HSA_AMD_PACKET_TYPE_BARRIER_VALUE;
  const uint32_t queueSize = gpu_queue_->size;
  const uint32_t queueMask = queueSize - 1;

  barrier_value_packet_.signal = signal;
  barrier_value_packet_.value = value;
  barrier_value_packet_.mask = mask;
  barrier_value_packet_.cond = cond;

  // Dependent signal and external signal cant be true at the same time
  assert((resolveDepSignal && (signal.handle != 0)) == false);
  if (resolveDepSignal) {
    auto wait_signals = Barriers().WaitingSignal();
    if (wait_signals.size() > 0) {
      barrier_value_packet_.signal = wait_signals[0];
      barrier_value_packet_.value = kInitSignalValueOne;
      barrier_value_packet_.mask = std::numeric_limits<int64_t>::max();
      barrier_value_packet_.cond = HSA_SIGNAL_CONDITION_LT;
      for (uint32_t i = 1; i < wait_signals.size(); ++i) {
        uint32_t j = (i - 1) % 5;
        barrier_packet_.dep_signal[j] = wait_signals[i];
        constexpr bool kSkipSignal = true;
        // If runtime reached the packet limit or the count limit, then flush the barrier
        if ((j == 4) || ((i + 1) == wait_signals.size())) {
          dispatchBarrierPacket(kNopPacketHeader, kSkipSignal);
        }
      }
    }
  }

  setFenceDirty(true);
  auto cache_state = extractAqlBits(packetHeader, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                                    HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);

  if (completionSignal.handle == 0) {
    // Get active signal for current dispatch if profiling is necessary
    barrier_value_packet_.completion_signal =
        Barriers().ActiveSignal(kInitSignalValueOne, skipTs ? nullptr : timestamp_);
  } else {
    // Attach external signal to the packet
    barrier_value_packet_.completion_signal = completionSignal;
  }

  // Reset fence_dirty_ flag if we submit a barrier
  if (cache_state == amd::Device::kCacheStateSystem) {
    setFenceDirty(false);
  }

  uint64_t index = Hsa::queue_add_write_index_screlease(gpu_queue_, 1);
  uint64_t read = Hsa::queue_load_read_index_relaxed(gpu_queue_);

  TrackQueueProgress(barrier_value_packet_, index);

  while ((index - Hsa::queue_load_read_index_scacquire(gpu_queue_)) >= queueMask);
  hsa_amd_barrier_value_packet_t* aql_loc = &(reinterpret_cast<hsa_amd_barrier_value_packet_t*>(
      gpu_queue_->base_address))[index & queueMask];
  *aql_loc = barrier_value_packet_;
  metadata_preloader_.Set(&barrier_value_packet_, packetHeader, index & queueMask);
  packet_store_release(reinterpret_cast<uint32_t*>(aql_loc), packetHeader, rest);
  Hsa::signal_store_screlease(gpu_queue_->doorbell_signal, index);

  logAqlBarrierValuePacket(gpu_queue_, packetHeader, &barrier_value_packet_, read, index,
                           IsLogEnabled(amd::LOG_DETAIL_DEBUG, amd::LOG_AQL)
                             ? [this]() -> const char* {
                                 if (!roc_device_.settings().queue_pipe_dist_) return "";
                                 static thread_local char buf[32];
                                 snprintf(buf, sizeof(buf), " virtual_pipe_id=%zu,",
                                          gpu_queue_->id % roc_device_.NumHwPipes());
                                 return buf;
                               }()
                             : "");
  // Clear dependent signals for the next packet
  barrier_value_packet_.signal = hsa_signal_t{};
}

// ================================================================================================
void VirtualGPU::ResetQueueStates() {
  // Release all memory dependencies
  memoryDependency().clear();

  // Release the pool, since runtime just completed a barrier
  // @note: Runtime can reset kernel arg pool only if the barrier with L2 invalidation was issued
  resetKernArgPool();
}

// ================================================================================================
bool VirtualGPU::releaseGpuMemoryFence(bool skip_cpu_wait) {
  if (hasPendingDispatch_ || isFenceDirty() || !Barriers().IsExternalSignalListEmpty()) {
    // Dispatch barrier packet into the queue
    dispatchBarrierPacket(kBarrierPacketHeader);
    hasPendingDispatch_ = false;
    skippedDispatches_ = 0;
    retainExternalSignals_ = false;
  }

  // Check if runtime could skip CPU wait
  if (!skip_cpu_wait) {
    Barriers().WaitCurrent();

    ResetQueueStates();
  }
  return true;
}

// ================================================================================================
VirtualGPU::VirtualGPU(Device& device, bool profiling, bool cooperative,
                       const std::vector<uint32_t>& cuMask, amd::CommandQueue::Priority priority,
                       bool dedicated_queue)
    : device::VirtualDevice(device),
      state_(0),
      gpu_queue_(nullptr),
      roc_device_(device),
      virtualQueue_(nullptr),
      deviceQueueSize_(0),
      maskGroups_(0),
      schedulerThreads_(0),
      schedulerQueue_(nullptr),
      barriers_(*this),
      managed_buffer_(*this, kStagingPoolNumSignals * device.settings().stagedXferSize_, kStagingPoolNumSignals),
      managed_kernarg_buffer_(*this, device.settings().kernargPoolSize_, kKernArgPoolNumSignals),
      cuMask_(cuMask),
      priority_(priority),
      copy_command_type_(0),
      fence_state_(Device::CacheState::kCacheStateInvalid),
      fence_dirty_(false),
      dedicated_queue_(dedicated_queue),
      schedulerQueueThreadRunning_(false),
      hostcallBuffer_(nullptr) {
  index_ = device.numOfVgpus_++;
  gpu_device_ = device.getBackendDevice();
  printfdbg_ = nullptr;

  // Initialize the last signal and dispatch flags
  timestamp_ = nullptr;
  command_ = nullptr;
  hasPendingDispatch_ = false;
  skippedDispatches_ = 0;
  profiling_ = profiling;
  cooperative_ = cooperative;

  // Initialize barrier and barrier value packets
  barrier_packet_.header = kInvalidAql;
  barrier_value_packet_.header.header = kInvalidAql;

  constexpr uint16_t kernelDispatchHBits =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
  constexpr uint16_t barrierHBits = (1 << HSA_PACKET_HEADER_BARRIER);
  constexpr uint16_t agentScopeHBits =
      (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  constexpr uint16_t systemScopeHBits =
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  // acquire=SYSTEM, release=AGENT (needed for GFX12)
  constexpr uint16_t sysAcquireAgentReleaseHBits =
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  constexpr uint16_t vendorSpecificHBits = (HSA_PACKET_TYPE_VENDOR_SPECIFIC
                                            << HSA_PACKET_HEADER_TYPE);

  if (device.settings().fenceScopeAgent_) {
    const auto& isa = device.isa();
    const bool isGfx12 = (isa.versionMajor() == 12) && (isa.versionMinor() == 0) &&
                         (isa.versionStepping() == 0 || isa.versionStepping() == 1);

    dispatchPacketHeaderNoSync_ =
        ((device.settings().ext_dispatch_packet_ ? vendorSpecificHBits : kernelDispatchHBits) |
         (isGfx12 ? sysAcquireAgentReleaseHBits : agentScopeHBits));
    dispatchPacketHeader_ =
        ((device.settings().ext_dispatch_packet_ ? vendorSpecificHBits : kernelDispatchHBits) |
         barrierHBits | (isGfx12 ? sysAcquireAgentReleaseHBits : agentScopeHBits));
  } else {
    dispatchPacketHeaderNoSync_ =
        ((device.settings().ext_dispatch_packet_ ? vendorSpecificHBits : kernelDispatchHBits) |
         systemScopeHBits);
    dispatchPacketHeader_ =
        ((device.settings().ext_dispatch_packet_ ? vendorSpecificHBits : kernelDispatchHBits) |
         barrierHBits | systemScopeHBits);
  }

  aqlHeader_ = dispatchPacketHeader_;

  // ISOLATION TEST (inter-kernel gap study): env-gated overrides of the kernel
  // dispatch packet header, to separate per-dispatch cache-fence cost from the
  // command-processor AQL packet processing cost. Plain ASCII only.
  //   GAP_NOSCOPE=1   strip acquire/release fence scope (NONE) -- no per-dispatch
  //                   cache acquire/release at all.
  //   GAP_NOBARRIER=1 clear the AQL barrier bit -- CP need not wait for the prior
  //                   packet to complete before processing the next.
  {
    constexpr uint16_t kScopeMask =
        static_cast<uint16_t>(~((3u << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                                (3u << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE)));
    constexpr uint16_t kBarrierMask =
        static_cast<uint16_t>(~(1u << HSA_PACKET_HEADER_BARRIER));
    if (getenv("GAP_NOSCOPE") != nullptr) {
      dispatchPacketHeader_ &= kScopeMask;
      dispatchPacketHeaderNoSync_ &= kScopeMask;
    }
    if (getenv("GAP_NOBARRIER") != nullptr) {
      dispatchPacketHeader_ &= kBarrierMask;
    }
    aqlHeader_ = dispatchPacketHeader_;
  }

  // Note: Virtual GPU device creation must be a thread safe operation
  roc_device_.vgpus_.resize(roc_device_.numOfVgpus_);
  roc_device_.vgpus_[index()] = this;

  // HIP_PM4_GRAPH_PREWARM: reserve the executable IB arena now (at stream/vgpu
  // creation), so the one-time executable-pool first-touch cost is paid here --
  // off the first-replay critical path -- rather than during the first launch.
  ensurePm4Arena();
}

// ================================================================================================
VirtualGPU::~VirtualGPU() {
  // Release SDMA engine assignment for this VirtualGPU
  ReleaseSdmaEngines();
  // Free the executable PM4 IBs (PWS fence + compiled graphs). Arena-backed IBs
  // are returned to the arena free list; the arena itself is freed below.
  for (auto& kv : pm4Graphs_) {
    freePm4GraphIb(kv.second);
  }
  pm4Graphs_.clear();
  pm4GraphKeyOrder_.clear();
  freeInplaceResident();  // release the in-place double-buffered slots + staging
  if (pm4Arena_ != nullptr) {
    Hsa::memory_pool_free(pm4Arena_);
    pm4Arena_ = nullptr;
    pm4ArenaFree_.clear();
  }
  if (pwsIbBuf_ != nullptr) {
    Hsa::memory_pool_free(pwsIbBuf_);
    pwsIbBuf_ = nullptr;
  }

  delete blitMgr_;

  if (tracking_created_) {
    std::scoped_lock l(execution());
    // Dedicated queues keep their HW queue, never acquire from pool
    if (!dedicated_queue_ && gpu_queue_ == nullptr) {
      void* md_rb = nullptr;
      SetGpuQueue(roc_device_.AcquireActiveQueue(priority_, nullptr, nullptr, &md_rb), md_rb);
    }
    // Windows requires an interrupt in more cases than Linux for OS fence updates
    force_irq_ = IS_WINDOWS;
    // Force extra barrier to make sure OS gets an interrupt,
    // but avoid if the PM4 emulation, since PM4 path can deadlock during device destruction
    hasPendingDispatch_ |= IS_WINDOWS && !dev().IsPm4Emulation();
    // Release the resources of signal
    releaseGpuMemoryFence();
  }

  if (timestamp_ != nullptr) {
    timestamp_->release();
    timestamp_ = nullptr;
    LogError("There was a timestamp that was not used; deleting.");
  }

  delete printfdbg_;

  if (nullptr != schedulerQueue_) {
#if defined(_WIN32)
    // Stop the monitor thread before destroying the queue
    if (schedulerQueueThread_.joinable()) {
      schedulerQueueThreadRunning_.store(false, std::memory_order_release);
      scheduler_cv_.notify_one();
      schedulerQueueThread_.join();
    }
#endif  // _WIN32
    Hsa::queue_destroy(schedulerQueue_);
    schedulerQueue_ = nullptr;
  }

  if (nullptr != virtualQueue_) {
    virtualQueue_->release();
  }

  {
    // Lock the device to make the following thread safe
    std::scoped_lock lock(roc_device_.vgpusAccess());

    --roc_device_.numOfVgpus_;  // Virtual gpu unique index decrementing
    roc_device_.vgpus_.erase(roc_device_.vgpus_.begin() + index());
    for (uint idx = index(); idx < roc_device_.vgpus().size(); ++idx) {
      roc_device_.vgpus()[idx]->index_--;
    }
  }

  if (gpu_queue_ != nullptr) {
    roc_device_.releaseQueue(gpu_queue_, cuMask_, cooperative_);
  }

  if (hostcallBuffer_) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_QUEUE, "Deleting hostcall buffer %p", hostcallBuffer_);
    amd::disableHostcalls(hostcallBuffer_);
    roc_device_.hostFree(hostcallBuffer_, hostcallBufferSize_);
  }
}

// ================================================================================================
bool VirtualGPU::create() {
  // Pick a reasonable queue size
  uint32_t queue_size = ROC_AQL_QUEUE_SIZE;
  void* md_rb = nullptr;
  SetGpuQueue(roc_device_.acquireQueue(queue_size, cooperative_, cuMask_, priority_, false,
                                       dedicated_queue_, nullptr, nullptr, &md_rb), md_rb);
  if (!gpu_queue_) return false;

  if (dev().isa().versionMajor() == 12 && dev().isa().versionMinor() >= 5) {
    metadata_preloader_.SetLaunchDescriptorVersion(AMD_LAUNCH_DESCRIPTOR_VERSION_GFX1250);
  }

  if (!managed_kernarg_buffer_.Create(Device::MemorySegment::kKernArg)) {
    LogError("Couldn't allocate arguments/signals for the queue");
    return false;
  }

  device::BlitManager::Setup blitSetup;
  blitMgr_ = new KernelBlitManager(*this, blitSetup);
  if ((nullptr == blitMgr_) || !blitMgr_->create(roc_device_)) {
    LogError("Could not create BlitManager!");
    return false;
  }

  // Create a object of PrintfDbg
  printfdbg_ = new PrintfDbg(roc_device_);
  if (nullptr == printfdbg_) {
    LogError("Could not create printfDbg Object!");
    return false;
  }

  // Initialize timestamp conversion factor
  if (Timestamp::getGpuTicksToTime() == 0) {
    uint64_t frequency;
    Hsa::system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency);
    Timestamp::setGpuTicksToTime(1e9 / double(frequency));
  }

  if (!memoryDependency().create(GPU_NUM_MEM_DEPENDENCY)) {
    LogError("Could not create the array of memory objects!");
    return false;
  }

  // Allocate signal tracker for ROCr copy queue
  tracking_created_ = Barriers().Create();
  if (!tracking_created_) {
    LogError("Could not create signal for copy queue!");
    return false;
  }
  // Create managed buffer for staging copies
  if (!managed_buffer_.Create(Device::MemorySegment::kNoAtomics)) {
    LogError("Could not create managed buffer for this queue!");
    return false;
  }
  // Release HW queue until the first usage
  ReleaseHwQueue();
  return true;
}

// ================================================================================================
VirtualGPU::ManagedBuffer::~ManagedBuffer() {
  for (auto& it : pool_signal_) {
    if (it.handle != 0) {
      Hsa::signal_destroy(it);
    }
  }
  if (pool_base_ != nullptr) {
    gpu_.dev().hostFree(pool_base_, pool_size_);
  }
}

// ================================================================================================
bool VirtualGPU::ManagedBuffer::Create(Device::MemorySegment mem_segment) {
  pool_chunk_end_ = pool_size_ / num_chunk_signals_;
  active_chunk_ = 0;
  // Allocate memory for managed buffer
  if (mem_segment == Device::MemorySegment::kKernArg &&
      (gpu_.dev().settings().kernel_arg_impl_ != KernelArgImpl::HostKernelArgs) &&
      gpu_.dev().info().largeBar_) {
    amd::Device::AllocationFlags flags = {};
    flags.executable_ = true;
    pool_base_ = reinterpret_cast<address>(gpu_.dev().deviceLocalAlloc(pool_size_, flags, false));
    if (pool_base_ != nullptr) {
      // @note Workaround first access penalty.
      // KFD may update CPU page tables on the first CPU access
      *pool_base_ = 0;
    }
  } else {
    pool_base_ = reinterpret_cast<address>(gpu_.dev().hostAlloc(pool_size_, 0, mem_segment, nullptr, false));
  }
  if (pool_base_ == nullptr) {
    return false;
  }
  hsa_agent_t agent = gpu_.dev().getBackendDevice();
  for (auto& it : pool_signal_) {
    if (HSA_STATUS_SUCCESS != Hsa::signal_create(0, 1, &agent, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &it)) {
      return false;
    }
  }
  return true;
}

// ================================================================================================
address VirtualGPU::ManagedBuffer::Acquire(uint32_t size) {
  return Acquire(size, gpu_.dev().info().globalMemCacheLineSize_);
}

// ================================================================================================
address VirtualGPU::ManagedBuffer::Acquire(uint32_t size, uint32_t alignment) {
  assert(alignment != 0);
  address result = nullptr;
  result = amd::alignUp(pool_base_ + pool_cur_offset_, alignment);
  const size_t pool_new_usage = (result + size) - pool_base_;
  if (pool_new_usage <= pool_chunk_end_) {
    pool_cur_offset_ = pool_new_usage;
    return result;
  } else {
    // Reset the signal for the barrier packet
    Hsa::signal_silent_store_relaxed(pool_signal_[active_chunk_], kInitSignalValueOne);
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN, "Issue barrier to flush chunk %d",
            active_chunk_);
    // Currently don't skip wait signal check, because SDMA engine cna be used in staging copy
    constexpr bool kSkipSignal = false;
    // Dispatch a barrier packet into the queue
    gpu_.dispatchBarrierPacket(kBarrierPacketHeader, kSkipSignal, pool_signal_[active_chunk_]);
    // Get the next chunk
    active_chunk_ = ++active_chunk_ % num_chunk_signals_;
    // Make sure the new active chunk is free
    bool test = WaitForSignal(pool_signal_[active_chunk_], gpu_.ActiveWait());
    assert(test && "Runtime can't fail a wait for chunk!");
    // Make sure the current offset matches the new chunk to avoid possible overlaps
    // between chunks and issues during recycle
    pool_cur_offset_ = (active_chunk_ == 0) ? 0 : pool_chunk_end_;
    pool_chunk_end_ = pool_cur_offset_ + pool_size_ / num_chunk_signals_;
    result = amd::alignUp(pool_base_ + pool_cur_offset_, alignment);
    pool_cur_offset_ = (result + size) - pool_base_;
  }

  return result;
}

// ================================================================================================
void VirtualGPU::ManagedBuffer::ResetPool() {
  pool_cur_offset_ = 0;
  pool_chunk_end_ = pool_size_ / num_chunk_signals_;
  active_chunk_ = 0;
}

// ================================================================================================
void* VirtualGPU::allocKernArg(size_t size, size_t alignment) {
  return managed_kernarg_buffer_.Acquire(size, alignment);
}

// ================================================================================================
address VirtualGPU::allocKernelArguments(size_t size, size_t alignment) {
  if (ROC_SKIP_KERNEL_ARG_COPY) {
    // Make sure VirtualGPU has an exclusive access to the resources
    std::scoped_lock lock(execution());
    return reinterpret_cast<address>(allocKernArg(size, alignment));
  } else {
    return nullptr;
  }
}

// ================================================================================================
void VirtualGPU::ReleaseSdmaEngines() {
  if (!hasAssignedSdmaEngine()) {
    return;
  }
  // Release SDMA engine assignment when queue is idle
  // This allows the engine to be reassigned to other active streams
  dev().ReleaseSdmaEngine(this);
  ClearAssignedSdmaEngine();
}

// ================================================================================================
void VirtualGPU::ReleaseAllHwQueues() {
  if (roc_device_.settings().dynamic_queues_) {
    // Check if any priority level exceeds max_hw_queues_
    bool should_release = false;
    for (uint qIdx = 0; qIdx < Device::QueuePriority::Total; ++qIdx) {
      if (roc_device_.NumQueues(qIdx) > roc_device_.settings().max_hw_queues_) {
        should_release = true;
        break;
      }
    }
    if (should_release) {
      // Lock the device to make the following thread safe
      std::scoped_lock lock(roc_device_.vgpusAccess());
      for (uint idx = 0; idx < roc_device_.vgpus().size(); ++idx) {
        roc_device_.vgpus()[idx]->ReleaseHwQueue();
      }
    }
  }
}

// ================================================================================================
void VirtualGPU::ReleaseHwQueue() {
  // Dedicated queues and pinned graph queues keep their HW queue
  if (dedicated_queue_ || queue_pinned_) {
    return;
  }

  // Try to release queue to the pool of active queues.
  // Use tryLock() since this may be called from the HsaAmdSignalHandler
  // and blocking here could cause deadlock
  if (roc_device_.settings().dynamic_queues_ > 0 && !cooperative_ &&
      (cuMask_.size() == 0)) {
    // If tryLock fails, skip the release - the queue will be released
    // on next opportunity
    if (execution().try_lock()) {
      if (gpu_queue_ != nullptr) {
        if (IsQueueIdle()) {
          last_hwq_ = gpu_queue_;
          if (roc_device_.ReleaseActiveQueue(gpu_queue_, priority_)) {
            SetGpuQueue(nullptr);
          }
        }
      }
      execution().unlock();
    }
  }
}

// ================================================================================================
/* profilingBegin, when profiling is enabled, creates a timestamp to save in
 * virtualgpu's timestamp_, saves the pointer timestamp_ to the command's data
 * and then calls start() to get the current host timestamp.
 */
void VirtualGPU::profilingBegin(amd::Command& command, bool sdmaProfiling) {
  // Dedicated queues keep their HW queue, never acquire from pool
  if (!dedicated_queue_ && gpu_queue_ == nullptr) {
    void* md_rb = nullptr;
    SetGpuQueue(roc_device_.AcquireActiveQueue(priority_, nullptr, nullptr, &md_rb), md_rb);
  }
  // Track the current command
  command_ = &command;

  // Disable profiling when command is being captured to prevent memory leak from created timestamp_
  // which won't get freed, since the command is not being executed until graph launch
  if (!command.getPktCapturingState() && command.profilingInfo().enabled_) {
    if (timestamp_ != nullptr) {
      LogWarning(
          "Trying to create a second timestamp in VirtualGPU. \
                  This could have unintended consequences.");
      return;
    }

    // Without barrier profiling will wait for each individual signal
    timestamp_ = new Timestamp(this, command);
    command.data().emplace_back(timestamp_);
    timestamp_->start();
    // Enable SDMA profiling on the first access if profiling is set
    // Its not per command basis
    if (sdmaProfiling && !Barriers().GetSDMAProfiling()) {
      Barriers().SetSDMAProfiling(true);
    }
  }

  if (!retainExternalSignals_) {
    Barriers().ClearExternalSignals();
  }
  for (auto it = command.eventWaitList().begin(); it < command.eventWaitList().end(); ++it) {
    void* hw_event =
        ((*it)->NotifyEvent() != nullptr) ? (*it)->NotifyEvent()->HwEvent() : (*it)->HwEvent();
    if (hw_event != nullptr) {
      Barriers().AddExternalSignal(reinterpret_cast<ProfilingSignal*>(hw_event));
    } else if (static_cast<amd::Command*>(*it)->queue() != command.queue() &&
                ((*it)->status() != CL_COMPLETE)) {
      LogPrintfError("Waiting event(%p) doesn't have a HSA signal!\n", *it);
    } else {
      // Assume serialization on the same queue...
    }

    // Check if the waiting event's queue has a dirty fence and propagate it
    if (!isFenceDirty()) {
      amd::Command* wait_cmd = static_cast<amd::Command*>(*it);
      if (wait_cmd->queue() != nullptr && wait_cmd->queue() != command.queue()) {
        device::VirtualDevice* wait_vdev = wait_cmd->queue()->vdev();
        if (wait_vdev != nullptr && wait_vdev->isFenceDirty()) {
          setFenceDirty(true);
        }
      }
    }
  }
  for (auto it = command.getDepHwEvents().begin(); it < command.getDepHwEvents().end(); ++it) {
    ClPrint(amd::LOG_DEBUG, amd::LOG_SIG, "Adding dep hw event signal: 0x%lx",
            reinterpret_cast<ProfilingSignal*>(*it)->signal_.handle);
    Barriers().AddExternalSignal(reinterpret_cast<ProfilingSignal*>(*it));
  }
  command.clearDepHwEvents();
}

// ================================================================================================
/* profilingEnd, when profiling is enabled, checks to see if a signal was
 * created for whatever command we are running and calls end() to get the
 * current host timestamp if no signal is available.
 */
void VirtualGPU::profilingEnd(bool clearHwEvent) {
  if (!command_->getPktCapturingState() && command_->profilingInfo().enabled_) {
    if (timestamp_->HwProfiling() == false) {
      timestamp_->end();
    }
    timestamp_ = nullptr;
  }

  // Certain commands like map/unmap memory may not need hw_events as its not a
  // queue operation. In such cases clear already set events which may have been for sync
  // before some memory map/unmap operation
  if (clearHwEvent) {
    if (command_->HwEvent() != nullptr) {
      reinterpret_cast<ProfilingSignal*>(command_->HwEvent())->release();
      command_->SetHwEvent(nullptr);
    }
  }

  // Clear the command tracking
  command_ = nullptr;
}

// ================================================================================================
void VirtualGPU::updateCommandsState(amd::Command* list) const {
  Timestamp* ts = nullptr;

  amd::Command* current = list;
  amd::Command* next = nullptr;

  if (current == nullptr) {
    return;
  }

  uint64_t endTimeStamp = 0;
  uint64_t startTimeStamp = endTimeStamp;

  if (current->profilingInfo().enabled_) {
    // TODO: use GPU timestamp when available.
    endTimeStamp = amd::Os::timeNanos();
    startTimeStamp = endTimeStamp;

    // This block gets the first valid timestamp from the first command
    // that has one. This timestamp is used below to mark any command that
    // came before it to start and end with this first valid start time.
    current = list;
    while (current != nullptr) {
      if (!current->data().empty()) {
        ts = reinterpret_cast<Timestamp*>(current->data().back());
        ts->getTime(&startTimeStamp, &endTimeStamp);
        break;
      }
      current = current->getNext();
    }
  }

  // Iterate through the list of commands, and set timestamps as appropriate
  // Note, if a command does not have a timestamp, it does one of two things:
  // - if the command (without a timestamp), A, precedes another command, C,
  // that _does_ contain a valid timestamp, command A will set RUNNING and
  // COMPLETE with the RUNNING (start) timestamp from command C. This would
  // also be true for command B, which is between A and C. These timestamps
  // are actually retrieved in the block above (startTimeStamp, endTimeStamp).
  // - if the command (without a timestamp), C, follows another command, A,
  // that has a valid timestamp, command C will be set RUNNING and COMPLETE
  // with the COMPLETE (end) timestamp of the previous command, A. This is
  // also true for any command B, which falls between A and C.
  current = list;
  while (current != nullptr) {
    if (current->profilingInfo().enabled_) {
      if (!current->data().empty()) {
        for (auto i = 0; i < current->data().size(); i++) {
          // Since this is a valid command to get a timestamp, we use the
          // timestamp provided by the runtime (saved in the data())
          ts = reinterpret_cast<Timestamp*>(current->data()[i]);
          ts->getTime(&startTimeStamp, &endTimeStamp);
          ts->release();
        }
        current->data().clear();
      } else {
        // If we don't have a command that contains a valid timestamp,
        // we simply use the end timestamp of the previous command.
        // Note, if this is a command before the first valid timestamp,
        // this will be equal to the start timestamp of the first valid
        // timestamp at this point.
        startTimeStamp = endTimeStamp;
      }
    }

    if (current->status() == CL_SUBMITTED) {
      current->setStatus(CL_RUNNING, startTimeStamp);
      current->setStatus(CL_COMPLETE, endTimeStamp);
    } else if (current->status() != CL_COMPLETE) {
      LogPrintfError("Unexpected command status - %d.", current->status());
    }

    next = current->getNext();
    current->release();
    current = next;
  }
}

// ================================================================================================

void VirtualGPU::submitReadMemory(amd::ReadMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(cmd, true);

  size_t offset = 0;
  // Find if virtual address is a CL allocation
  device::Memory* hostMemory = dev().findMemoryFromVA(cmd.destination(), &offset);

  Memory* devMem = dev().getRocMemory(&cmd.source());
  // Synchronize data with other memory instances if necessary
  devMem->syncCacheFromHost(*this);

  void* dst = cmd.destination();
  amd::Coord3D size = cmd.size();

  //! @todo: add multi-devices synchronization when supported.

  cl_command_type type = cmd.type();
  bool result = false;
  bool imageBuffer = false;

  // Force buffer read for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_READ_IMAGE) && (cmd.source().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    type = CL_COMMAND_READ_BUFFER;
    imageBuffer = true;
  }

  switch (type) {
    case CL_COMMAND_READ_BUFFER: {
      amd::Coord3D origin(cmd.origin()[0]);
      if (imageBuffer) {
        size_t elemSize = cmd.source().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;
      }
      if (hostMemory != nullptr) {
        // Accelerated transfer without pinning
        amd::Coord3D dstOrigin(offset);
        result = blitMgr().copyBuffer(*devMem, *hostMemory, origin, dstOrigin, size,
                                      cmd.isEntireMemory(), cmd.copyMetadata());
      } else {
        result = blitMgr().readBuffer(*devMem, dst, origin, size, cmd.isEntireMemory(),
                                      cmd.copyMetadata());
      }
      break;
    }
    case CL_COMMAND_READ_BUFFER_RECT: {
      amd::BufferRect hostbufferRect;
      amd::Coord3D region(0);
      amd::Coord3D hostOrigin(cmd.hostRect().start_ + offset);
      hostbufferRect.create(hostOrigin.c, size.c, cmd.hostRect().rowPitch_,
                            cmd.hostRect().slicePitch_);
      if (hostMemory != nullptr) {
        result = blitMgr().copyBufferRect(*devMem, *hostMemory, cmd.bufRect(), hostbufferRect, size,
                                          cmd.isEntireMemory(), cmd.copyMetadata());
      } else {
        result = blitMgr().readBufferRect(*devMem, dst, cmd.bufRect(), cmd.hostRect(), size,
                                          cmd.isEntireMemory(), cmd.copyMetadata());
      }
      break;
    }
    case CL_COMMAND_READ_IMAGE: {
      if ((cmd.source().parent() != nullptr) &&
          (cmd.source().parent()->getType() == CL_MEM_OBJECT_BUFFER)) {
        Image* imageBuffer = static_cast<Image*>(devMem);
        // Check if synchronization has to be performed
        if (nullptr != imageBuffer->CopyImageBuffer()) {
          amd::Memory* memory = imageBuffer->CopyImageBuffer();
          devMem = dev().getGpuMemory(memory);
          Memory* buffer = dev().getGpuMemory(imageBuffer->owner()->parent());
          amd::Image* image = imageBuffer->owner()->asImage();
          amd::Coord3D offs(0);
          // Copy memory from the original image buffer into the backing store image
          result = blitMgr().copyBufferToImage(*buffer, *devMem, offs, offs, image->getRegion(),
                                               true, image->getRowPitch(), image->getSlicePitch());
        }
      }
      if (hostMemory != nullptr) {
        // Accelerated image to buffer transfer without pinning
        amd::Coord3D dstOrigin(offset);
        result = blitMgr().copyImageToBuffer(*devMem, *hostMemory, cmd.origin(), dstOrigin, size,
                                             cmd.isEntireMemory(), cmd.rowPitch(), cmd.slicePitch(),
                                             cmd.copyMetadata());
      } else {
        result = blitMgr().readImage(*devMem, dst, cmd.origin(), size, cmd.rowPitch(),
                                     cmd.slicePitch(), cmd.isEntireMemory(), cmd.copyMetadata());
      }
      break;
    }
    default:
      ShouldNotReachHere();
      break;
  }

  if (!result) {
    LogError("submitReadMemory failed!");
    cmd.setStatus(CL_OUT_OF_RESOURCES);
  }

  profilingEnd();
}

void VirtualGPU::submitWriteMemory(amd::WriteMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(cmd, true);

  size_t offset = 0;
  // Find if virtual address is a CL allocation
  device::Memory* hostMemory = dev().findMemoryFromVA(cmd.source(), &offset);

  Memory* devMem = dev().getRocMemory(&cmd.destination());

  // Synchronize memory from host if necessary
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = cmd.isEntireMemory();
  devMem->syncCacheFromHost(*this, syncFlags);

  const char* src = static_cast<const char*>(cmd.source());
  amd::Coord3D size = cmd.size();

  //! @todo add multi-devices synchronization when supported.

  cl_command_type type = cmd.type();
  bool result = false;
  bool imageBuffer = false;

  // Force buffer write for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_WRITE_IMAGE) &&
      (cmd.destination().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    type = CL_COMMAND_WRITE_BUFFER;
    imageBuffer = true;
  }

  switch (type) {
    case CL_COMMAND_WRITE_BUFFER: {
      amd::Coord3D origin(cmd.origin()[0]);
      if (imageBuffer) {
        size_t elemSize = cmd.destination().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;
      }
      if (hostMemory != nullptr) {
        // Accelerated transfer without pinning
        amd::Coord3D srcOrigin(offset);
        result = blitMgr().copyBuffer(*hostMemory, *devMem, srcOrigin, origin, size,
                                      cmd.isEntireMemory(), cmd.copyMetadata());
      } else {
        result = blitMgr().writeBuffer(src, *devMem, origin, size, cmd.isEntireMemory(),
                                       cmd.copyMetadata());
      }
      break;
    }
    case CL_COMMAND_WRITE_BUFFER_RECT: {
      amd::BufferRect hostbufferRect;
      amd::Coord3D region(0);
      amd::Coord3D hostOrigin(cmd.hostRect().start_ + offset);
      hostbufferRect.create(hostOrigin.c, size.c, cmd.hostRect().rowPitch_,
                            cmd.hostRect().slicePitch_);
      if (hostMemory != nullptr) {
        result = blitMgr().copyBufferRect(*hostMemory, *devMem, hostbufferRect, cmd.bufRect(), size,
                                          cmd.isEntireMemory(), cmd.copyMetadata());
      } else {
        result = blitMgr().writeBufferRect(src, *devMem, cmd.hostRect(), cmd.bufRect(), size,
                                           cmd.isEntireMemory(), cmd.copyMetadata());
      }
      break;
    }
    case CL_COMMAND_WRITE_IMAGE: {
      if (hostMemory != nullptr) {
        // Accelerated buffer to image transfer without pinning
        amd::Coord3D srcOrigin(offset);
        result = blitMgr().copyBufferToImage(*hostMemory, *devMem, srcOrigin, cmd.origin(), size,
                                             cmd.isEntireMemory(), cmd.rowPitch(), cmd.slicePitch(),
                                             cmd.copyMetadata());
      } else {
        result = blitMgr().writeImage(src, *devMem, cmd.origin(), size, cmd.rowPitch(),
                                      cmd.slicePitch(), cmd.isEntireMemory(), cmd.copyMetadata());
      }
      break;
    }
    default:
      ShouldNotReachHere();
      break;
  }

  if (!result) {
    LogError("submitWriteMemory failed!");
    cmd.setStatus(CL_OUT_OF_RESOURCES);
  } else {
    cmd.destination().signalWrite(&dev());
  }

  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  // in-order semantics: previous commands need to be done before we start
  releaseGpuMemoryFence();

  profilingBegin(cmd);
  const std::vector<void*>& svmPointers = cmd.svmPointers();
  if (cmd.pfnFreeFunc() == nullptr) {
    // pointers allocated using clSVMAlloc
    for (uint32_t i = 0; i < svmPointers.size(); i++) {
      amd::SvmBuffer::free(cmd.context(), svmPointers[i]);
    }
  } else {
    cmd.pfnFreeFunc()(as_cl(cmd.queue()->asCommandQueue()), svmPointers.size(),
                      (void**)(&(svmPointers[0])), cmd.userData());
  }
  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitSvmPrefetchAsync(amd::SvmPrefetchAsyncCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());
  profilingBegin(cmd);

  if (dev().info().hmmSupported_) {
    // Initialize signal for the barrier
    auto wait_events = Barriers().WaitingSignal(HwQueueEngine::Unknown);
    hsa_signal_t active = Barriers().ActiveSignal(kInitSignalValueOne, timestamp_);

    // Find the requested agent for the transfer
    hsa_agent_t agent =
        (cmd.cpu_access() || (dev().settings().hmmFlags_ & Settings::Hmm::EnableSystemMemory))
            ? dev().getCpuAgent(cmd.numa_id())
            : (static_cast<const roc::Device*>(cmd.device()))->getBackendDevice();

    // Initiate a prefetch command
    hsa_status_t status =
        Hsa::svm_prefetch_async(const_cast<void*>(cmd.dev_ptr()), cmd.count(), agent,
                                wait_events.size(), wait_events.data(), active);
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
            "HSA prefetch async dev_ptr=0x%zx, count=%d, wait_event=0x%zx, "
            "completion_signal=0x%zx",
            const_cast<void*>(cmd.dev_ptr()), cmd.count(),
            (wait_events.size() != 0) ? wait_events[0].handle : 0, active.handle);

    if ((status != HSA_STATUS_SUCCESS)) {
      Barriers().ResetCurrentSignal();
      LogError("hsa_amd_svm_prefetch_async failed");
      cmd.setStatus(CL_INVALID_OPERATION);
    }

    // Add system scope, since the prefetch scope is unclear
    addSystemScope();
  } else {
    LogWarning("hsa_amd_svm_prefetch_async is ignored, because no HMM support");
  }
  profilingEnd();
}

// ================================================================================================
void VirtualGPU::SubmitSvmPrefetchBatchAsync(amd::SvmPrefetchBatchAsyncCommand& command) {
  std::scoped_lock lock(execution());
  profilingBegin(command);

  auto wait_events = Barriers().WaitingSignal(HwQueueEngine::Unknown);
  hsa_signal_t active = Barriers().ActiveSignal(command.Count(), timestamp_);

  const bool enable_system_memory =
      (dev().settings().hmmFlags_ & Settings::Hmm::EnableSystemMemory) != 0;

  for (size_t idx = 0; idx < command.Count(); idx++) {
    void* dev_ptr = command.DevicePointers()[idx];
    size_t size = command.Sizes()[idx];
    const roc::Device* target_dev = static_cast<const roc::Device*>(command.TargetDevices()[idx]);
    bool cpu_access = target_dev == nullptr;

    hsa_agent_t agent = (cpu_access || enable_system_memory) ? dev().getCpuAgent(CpuDeviceId)
                                                             : target_dev->getBackendDevice();

    hsa_status_t status = Hsa::svm_prefetch_async(dev_ptr, size, agent, wait_events.size(),
                                                  wait_events.data(), active);
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
            "HSA prefetch batch async[%zu] dev_ptr=0x%zx, size=%zu, wait_event=0x%zx, "
            "completion_signal=0x%zx",
            idx, dev_ptr, size, wait_events.empty() ? 0 : wait_events[0].handle, active.handle);

    if (status != HSA_STATUS_SUCCESS) {
      Barriers().ResetCurrentSignal();
      LogError("HSA prefetch batch async failed in batch operation");
      command.setStatus(CL_INVALID_OPERATION);
      profilingEnd();
      return;
    }
  }

  addSystemScope();
  profilingEnd();
}

// ================================================================================================
bool VirtualGPU::copyMemory(cl_command_type type, amd::Memory& srcMem, amd::Memory& dstMem,
                            bool entire, const amd::Coord3D& srcOrigin,
                            const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                            const amd::BufferRect& srcRect, const amd::BufferRect& dstRect,
                            amd::CopyMetadata copyMetadata) {
  Memory* srcDevMem = dev().getRocMemory(&srcMem);
  Memory* dstDevMem = dev().getRocMemory(&dstMem);
  if (srcDevMem == nullptr || dstDevMem == nullptr) {
    LogError("submitCopyMemory failed!");
    return false;
  }
  // Synchronize source and destination memory
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = entire;
  dstDevMem->syncCacheFromHost(*this, syncFlags);
  srcDevMem->syncCacheFromHost(*this);

  bool result = false;
  amd::Memory* bufferFromImageSrc = nullptr;
  amd::Memory* bufferFromImageDst = nullptr;

  // Force buffer copy for IMAGE1D_BUFFER
  if (srcMem.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
    bufferFromImageSrc = createBufferFromImage(srcMem);
    if (nullptr == bufferFromImageSrc) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      srcDevMem = dev().getRocMemory(bufferFromImageSrc);
    }
  }
  // Force buffer write for IMAGE1D_BUFFER
  if (dstMem.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
    bufferFromImageDst = createBufferFromImage(dstMem);
    if (nullptr == bufferFromImageDst) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      dstDevMem = dev().getRocMemory(bufferFromImageDst);
    }
  }
  type = getCopyCommandType(type, srcMem.getType(), dstMem.getType());
  switch (type) {
    case CL_COMMAND_SVM_MEMCPY:
    case CL_COMMAND_COPY_BUFFER: {
      amd::Coord3D realSrcOrigin(srcOrigin[0]);
      amd::Coord3D realDstOrigin(dstOrigin[0]);
      amd::Coord3D realSize(size.c[0], size.c[1], size.c[2]);

      if (nullptr != bufferFromImageSrc) {
        const size_t elemSize = srcMem.asImage()->getImageFormat().getElementSize();
        realSrcOrigin.c[0] *= elemSize;
        if (nullptr != bufferFromImageDst) {
          realDstOrigin.c[0] *= elemSize;
        }
        realSize.c[0] *= elemSize;
      } else if (nullptr != bufferFromImageDst) {
        const size_t elemSize = dstMem.asImage()->getImageFormat().getElementSize();
        realDstOrigin.c[0] *= elemSize;
        realSize.c[0] *= elemSize;
      }

      result = blitMgr().copyBuffer(*srcDevMem, *dstDevMem, realSrcOrigin, realDstOrigin, realSize,
                                    entire, copyMetadata);
      break;
    }
    case CL_COMMAND_COPY_BUFFER_RECT: {
      result = blitMgr().copyBufferRect(*srcDevMem, *dstDevMem, srcRect, dstRect, size, entire,
                                        copyMetadata);
      break;
    }
    case CL_COMMAND_COPY_IMAGE: {
      result = blitMgr().copyImage(*srcDevMem, *dstDevMem, srcOrigin, dstOrigin, size, entire,
                                   copyMetadata);
      break;
    }
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER: {
      amd::Coord3D realDstOrigin(dstOrigin);
      if (nullptr != bufferFromImageDst) {
        const size_t elemSize = dstMem.asImage()->getImageFormat().getElementSize();
        realDstOrigin.c[0] *= elemSize;
      }
      result = blitMgr().copyImageToBuffer(*srcDevMem, *dstDevMem, srcOrigin, realDstOrigin, size,
                                   entire, dstRect.rowPitch_, dstRect.slicePitch_, copyMetadata);
      break;
    }
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE: {
      amd::Coord3D realSrcOrigin(srcOrigin);
      if (nullptr != bufferFromImageSrc) {
        const size_t elemSize = srcMem.asImage()->getImageFormat().getElementSize();
        realSrcOrigin.c[0] *= elemSize;
      }
      result = blitMgr().copyBufferToImage(*srcDevMem, *dstDevMem, realSrcOrigin, dstOrigin, size,
                                   entire, srcRect.rowPitch_, srcRect.slicePitch_, copyMetadata);
      break;
    }
    default:
      ShouldNotReachHere();
      break;
  }
  if (nullptr != bufferFromImageSrc) {
    bufferFromImageSrc->release();
  }
  if (nullptr != bufferFromImageDst) {
    bufferFromImageDst->release();
  }
  if (!result) {
    LogError("submitCopyMemory failed!");
    return false;
  }

  // Mark this as the most-recently written cache of the destination
  dstMem.signalWrite(&dev());
  return true;
}

// ================================================================================================
void VirtualGPU::submitCopyMemory(amd::CopyMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(cmd, true);

  cl_command_type type = cmd.type();
  bool entire = cmd.isEntireMemory();

  if (!copyMemory(type, cmd.source(), cmd.destination(), entire, cmd.srcOrigin(), cmd.dstOrigin(),
                  cmd.size(), cmd.srcRect(), cmd.dstRect(), cmd.copyMetadata())) {
    cmd.setStatus(CL_INVALID_OPERATION);
  }

  // Runtime may change the command type to report a more accurate info in ROC profiler
  if (copy_command_type_ != 0) {
    cmd.OverrrideCommandType(copy_command_type_);
    copy_command_type_ = 0;
  }
  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(cmd, true);
  // no op for FGS supported device
  if (!dev().isFineGrainedSystem(true)) {
    amd::Coord3D srcOrigin(0, 0, 0);
    amd::Coord3D dstOrigin(0, 0, 0);
    amd::Coord3D size(cmd.srcSize(), 1, 1);
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;

    bool result = false;
    amd::Memory* srcMem = amd::MemObjMap::FindMemObj(cmd.src());
    amd::Memory* dstMem = amd::MemObjMap::FindMemObj(cmd.dst());

    device::Memory::SyncFlags syncFlags;
    if (nullptr != srcMem) {
      srcOrigin.c[0] =
          static_cast<const_address>(cmd.src()) - static_cast<address>(srcMem->getSvmPtr());
      if (!(srcMem->validateRegion(srcOrigin, size))) {
        cmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    }
    if (nullptr != dstMem) {
      dstOrigin.c[0] =
          static_cast<const_address>(cmd.dst()) - static_cast<address>(dstMem->getSvmPtr());
      if (!(dstMem->validateRegion(dstOrigin, size))) {
        cmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    }

    if ((nullptr == srcMem && nullptr == dstMem) ||  // both not in svm space
        (nullptr != srcMem && dev().forceFineGrain(srcMem)) ||
        (nullptr != dstMem && dev().forceFineGrain(dstMem))) {
      // Wait on a kernel if one is outstanding
      releaseGpuMemoryFence();

      // If these are from different contexts, then one of them could be in the device memory
      // This is fine, since spec doesn't allow for copies with pointers from different contexts
      std::memcpy(cmd.dst(), cmd.src(), cmd.srcSize());
      result = true;
    } else if (nullptr == srcMem && nullptr != dstMem) {  // src not in svm space
      Memory* memory = dev().getRocMemory(dstMem);
      // Synchronize source and destination memory
      syncFlags.skipEntire_ = dstMem->isEntirelyCovered(dstOrigin, size);
      memory->syncCacheFromHost(*this, syncFlags);

      result = blitMgr().writeBuffer(cmd.src(), *memory, dstOrigin, size,
                                     dstMem->isEntirelyCovered(dstOrigin, size));
      // Mark this as the most-recently written cache of the destination
      dstMem->signalWrite(&dev());
    } else if (nullptr != srcMem && nullptr == dstMem) {  // dst not in svm space
      Memory* memory = dev().getRocMemory(srcMem);
      // Synchronize source and destination memory
      memory->syncCacheFromHost(*this);

      result = blitMgr().readBuffer(*memory, cmd.dst(), srcOrigin, size,
                                    srcMem->isEntirelyCovered(srcOrigin, size));
    } else if (nullptr != srcMem && nullptr != dstMem) {  // both in svm space
      bool entire =
          srcMem->isEntirelyCovered(srcOrigin, size) && dstMem->isEntirelyCovered(dstOrigin, size);
      result = copyMemory(cmd.type(), *srcMem, *dstMem, entire, srcOrigin, dstOrigin, size, srcRect,
                          dstRect);
    }

    if (!result) {
      cmd.setStatus(CL_INVALID_OPERATION);
    }
  } else {
    // Stall GPU for CPU access to memory
    releaseGpuMemoryFence();
    // direct memcpy for FGS enabled system
    amd::SvmBuffer::memFill(cmd.dst(), cmd.src(), cmd.srcSize(), 1);
  }
  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitCopyMemoryP2P(amd::CopyMemoryP2PCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(cmd, true);

  Memory* srcDevMem = static_cast<roc::Memory*>(
      cmd.source().getDeviceMemory(*cmd.source().getContext().devices()[0]));
  Memory* dstDevMem = static_cast<roc::Memory*>(
      cmd.destination().getDeviceMemory(*cmd.destination().getContext().devices()[0]));

  bool p2pAllowed = false;
  // Loop through all available P2P devices for the destination buffer
  for (auto agent : dstDevMem->dev().p2pAgents()) {
    // Find the device, which is matching the current
    if (agent.handle == dev().getBackendDevice().handle) {
      p2pAllowed = true;
      break;
    }

    for (auto agent : srcDevMem->dev().p2pAgents()) {
      if (agent.handle == dev().getBackendDevice().handle) {
        p2pAllowed = true;
        break;
      }
    }
  }

  // Synchronize source and destination memory
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = cmd.isEntireMemory();
  amd::Coord3D size = cmd.size();

  bool result = false;
  switch (cmd.type()) {
    case CL_COMMAND_COPY_BUFFER: {
      amd::Coord3D srcOrigin(cmd.srcOrigin()[0]);
      amd::Coord3D dstOrigin(cmd.dstOrigin()[0]);

      if (p2pAllowed) {
        result = blitMgr().copyBuffer(*srcDevMem, *dstDevMem, srcOrigin, dstOrigin, size,
                                      cmd.isEntireMemory());
      } else {
        // Sync the current queue, since P2P staging uses the device queues for transfer
        releaseGpuMemoryFence();

        std::scoped_lock lock(dev().P2PStageOps());
        Memory* dstStgMem = static_cast<Memory*>(
            dev().P2PStage()->getDeviceMemory(*cmd.source().getContext().devices()[0]));
        Memory* srcStgMem = static_cast<Memory*>(
            dev().P2PStage()->getDeviceMemory(*cmd.destination().getContext().devices()[0]));

        size_t copy_size = Device::kP2PStagingSize;
        size_t left_size = size[0];
        result = true;
        do {
          if (left_size <= copy_size) {
            copy_size = left_size;
          }
          left_size -= copy_size;
          amd::Coord3D stageOffset(0);
          amd::Coord3D cpSize(copy_size);

          // Perform 2 step transfer with staging buffer
          result &= srcDevMem->dev().xferMgr().copyBuffer(*srcDevMem, *dstStgMem, srcOrigin,
                                                          stageOffset, cpSize);
          srcOrigin.c[0] += copy_size;
          result &= dstDevMem->dev().xferMgr().copyBuffer(*srcStgMem, *dstDevMem, stageOffset,
                                                          dstOrigin, cpSize);
          dstOrigin.c[0] += copy_size;
        } while (left_size > 0);
      }
      break;
    }
    case CL_COMMAND_COPY_BUFFER_RECT: {
      if (p2pAllowed) {
        result = blitMgr().copyBufferRect(*srcDevMem, *dstDevMem, cmd.srcRect(), cmd.dstRect(),
                                          size, cmd.isEntireMemory(), cmd.copyMetadata());
      } else {
        // Sync the current queue, since P2P staging uses the device queues for transfer
        releaseGpuMemoryFence();

        std::scoped_lock lock(dev().P2PStageOps());
        Memory* dstStgMem = static_cast<Memory*>(
            dev().P2PStage()->getDeviceMemory(*cmd.source().getContext().devices()[0]));
        Memory* srcStgMem = static_cast<Memory*>(
            dev().P2PStage()->getDeviceMemory(*cmd.destination().getContext().devices()[0]));

        if ((cmd.srcRect().slicePitch_ * size[2]) <= Device::kP2PStagingSize) {
          result = true;
          // Perform 2 step transfer with staging buffer
          result &= srcDevMem->dev().xferMgr().copyBufferRect(*srcDevMem, *dstStgMem, cmd.srcRect(),
                                                              cmd.srcRect(), size, false,
                                                              cmd.copyMetadata());

          result &= dstDevMem->dev().xferMgr().copyBufferRect(*srcStgMem, *dstDevMem, cmd.srcRect(),
                                                              cmd.dstRect(), size, false,
                                                              cmd.copyMetadata());
        } else {
          size_t srcOffset;
          size_t dstOffset;
          result = true;

          for (size_t z = 0; z < size[2]; ++z) {
            for (size_t y = 0; y < size[1]; ++y) {
              srcOffset = cmd.srcRect().offset(0, y, z);
              dstOffset = cmd.dstRect().offset(0, y, z);

              amd::Coord3D srcOrigin(srcOffset);
              amd::Coord3D dstOrigin(dstOffset);
              size_t copy_size = Device::kP2PStagingSize;
              size_t left_size = size[0];
              amd::Coord3D stageOffset(0);
              do {
                if (left_size <= copy_size) {
                  copy_size = left_size;
                }
                left_size -= copy_size;

                // Perform 2 step transfer with staging buffer
                result &= srcDevMem->dev().xferMgr().copyBuffer(*srcDevMem, *dstStgMem, srcOrigin,
                                                                stageOffset, copy_size);

                result &= dstDevMem->dev().xferMgr().copyBuffer(*srcStgMem, *dstDevMem, stageOffset,
                                                                dstOrigin, copy_size);

                srcOrigin.c[0] += copy_size;
                dstOrigin.c[0] += copy_size;
              } while (left_size > 0);
            }
          }
        }
      }
      break;
    }
    case CL_COMMAND_COPY_IMAGE:
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      LogError("Unsupported P2P type!");
      break;
    default:
      ShouldNotReachHere();
      break;
  }

  if (!result) {
    LogError("submitCopyMemoryP2P failed!");
    cmd.setStatus(CL_OUT_OF_RESOURCES);
  }

  cmd.destination().signalWrite(&dstDevMem->dev());

  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitBatchCopyMemory(amd::BatchCopyMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(cmd, true);

  auto& copyOps = cmd.copyOps();
  if (copyOps.empty()) {
    profilingEnd();
    return;
  }

  bool result = true;

  // Sync caches for all ops
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = false;

  for (const auto& op : copyOps) {
    Memory* srcDevMem = dev().getRocMemory(op.srcMemory);
    Memory* dstDevMem = dev().getRocMemory(op.dstMemory);

    if (srcDevMem == nullptr || dstDevMem == nullptr) {
      LogError("submitBatchCopyMemory: Invalid memory objects!");
      cmd.setStatus(CL_INVALID_MEM_OBJECT);
      profilingEnd();
      return;
    }

    dstDevMem->syncCacheFromHost(*this, syncFlags);
    srcDevMem->syncCacheFromHost(*this);
  }

  // KernelBlitManager::copyBufferBatch handles the D2D/D2H/H2D/P2P split:
  // D2D copies use copyBuffer (kernel blit), D2H/H2D/P2P use DMA batch
  std::vector<amd::BatchCopyOp> batchOps(copyOps.begin(), copyOps.end());
  if (!blitMgr().copyBufferBatch(batchOps)) {
    LogError("submitBatchCopyMemory: Batch copy failed!");
    result = false;
  }

  // Synchronize the launch (compute) stream with SDMA engines.
  // WaitingSignal(Compute) inside dispatchBarrierPacket detects the engine switch
  // from SDMA→Compute, collects the SDMA completion signal as a barrier dependency,
  // and updates engine_ to Compute before ActiveSignal tags the profiling signal.
  if (result) {
    dispatchBarrierPacket(kNopPacketHeader);
  }

  if (!result) {
    LogError("submitBatchCopyMemory failed!");
    cmd.setStatus(CL_OUT_OF_RESOURCES);
  } else {
    // Mark all destinations as written
    for (const auto& op : copyOps) {
      op.dstMemory->signalWrite(&dev());
    }
  }

  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(cmd, true);

  // no op for FGS supported device
  if (!dev().isFineGrainedSystem(true) && !dev().forceFineGrain(cmd.getSvmMem())) {
    // Make sure we have memory for the command execution
    Memory* memory = dev().getRocMemory(cmd.getSvmMem());

    memory->saveMapInfo(cmd.svmPtr(), cmd.origin(), cmd.size(), cmd.mapFlags(),
                        cmd.isEntireMemory());

    if (memory->mapMemory() != nullptr) {
      if (cmd.mapFlags() & (CL_MAP_READ | CL_MAP_WRITE)) {
        Memory* hsaMapMemory = dev().getRocMemory(memory->mapMemory());

        if (!blitMgr().copyBuffer(*memory, *hsaMapMemory, cmd.origin(), cmd.origin(), cmd.size(),
                                  cmd.isEntireMemory())) {
          LogError("submitSVMMapMemory() - copy failed");
          cmd.setStatus(CL_MAP_FAILURE);
        }
        // Wait on a kernel if one is outstanding
        releaseGpuMemoryFence();
        const void* mappedPtr = hsaMapMemory->owner()->getHostMem();
        std::memcpy(cmd.svmPtr(), mappedPtr, cmd.size()[0]);
      }
    } else {
      LogError("Unhandled svm map!");
    }
  }

  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(cmd, true);

  // no op for FGS supported device
  if (!dev().isFineGrainedSystem(true) && !dev().forceFineGrain(cmd.getSvmMem())) {
    Memory* memory = dev().getRocMemory(cmd.getSvmMem());
    const device::Memory::WriteMapInfo* writeMapInfo = memory->writeMapInfo(cmd.svmPtr());

    if (memory->mapMemory() != nullptr) {
      if (writeMapInfo->isUnmapWrite()) {
        // Wait on a kernel if one is outstanding
        releaseGpuMemoryFence();
        amd::Coord3D srcOrigin(0, 0, 0);
        Memory* hsaMapMemory = dev().getRocMemory(memory->mapMemory());

        void* mappedPtr = hsaMapMemory->owner()->getHostMem();
        std::memcpy(mappedPtr, cmd.svmPtr(), writeMapInfo->region_[0]);
        // Target is a remote resource, so copy
        if (!blitMgr().copyBuffer(*hsaMapMemory, *memory, writeMapInfo->origin_,
                                  writeMapInfo->origin_, writeMapInfo->region_,
                                  writeMapInfo->isEntire())) {
          LogError("submitSvmUnmapMemory() - copy failed");
          cmd.setStatus(CL_OUT_OF_RESOURCES);
        }
      }
    } else {
      LogError("Unhandled svm map!");
    }

    memory->clearUnmapInfo(cmd.svmPtr());
  }

  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitMapMemory(amd::MapMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(cmd, true);

  //! @todo add multi-devices synchronization when supported.

  roc::Memory* devMemory =
      reinterpret_cast<roc::Memory*>(cmd.memory().getDeviceMemory(dev(), false));

  cl_command_type type = cmd.type();
  bool imageBuffer = (cmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER);
  if (imageBuffer) {
    type = CL_COMMAND_MAP_BUFFER;
  }

  // Save map requirement.
  cl_map_flags mapFlag = cmd.mapFlags();

  // Treat no map flag as read-write.
  if (mapFlag == 0) {
    mapFlag = CL_MAP_READ | CL_MAP_WRITE;
  }

  devMemory->saveMapInfo(cmd.mapPtr(), cmd.origin(), cmd.size(), mapFlag, cmd.isEntireMemory());

  // Sync to the map target.
  // If we have host memory, use it
  if ((devMemory->owner()->getHostMem() != nullptr) &&
      (devMemory->owner()->getSvmPtr() == nullptr)) {
    // Target is the backing store, so just ensure that owner is up-to-date
    devMemory->owner()->cacheWriteBack(this);

    if (devMemory->isHostMemDirectAccess()) {
      // Add memory to VA cache, so rutnime can detect direct access to VA
      dev().addVACache(devMemory);
    }
  } else if (devMemory->IsPersistentDirectMap()) {
    // Persistent memory - NOP map
  } else if (mapFlag & (CL_MAP_READ | CL_MAP_WRITE)) {
    bool result = false;
    roc::Memory* hsaMemory = static_cast<roc::Memory*>(devMemory);

    amd::Memory* mapMemory = hsaMemory->mapMemory();
    void* hostPtr =
        mapMemory == nullptr ? hsaMemory->owner()->getHostMem() : mapMemory->getHostMem();

    if (type == CL_COMMAND_MAP_BUFFER) {
      amd::Coord3D origin(cmd.origin()[0]);
      amd::Coord3D size(cmd.size()[0]);
      amd::Coord3D dstOrigin(cmd.origin()[0], 0, 0);
      if (imageBuffer) {
        size_t elemSize = cmd.memory().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;
      }

      if (mapMemory != nullptr) {
        roc::Memory* hsaMapMemory =
            static_cast<roc::Memory*>(mapMemory->getDeviceMemory(dev(), false));
        result = blitMgr().copyBuffer(*hsaMemory, *hsaMapMemory, origin, dstOrigin, size,
                                      cmd.isEntireMemory());
        void* svmPtr = devMemory->owner()->getSvmPtr();
        if ((svmPtr != nullptr) && (hostPtr != svmPtr)) {
          // Wait on a kernel if one is outstanding
          releaseGpuMemoryFence();
          std::memcpy(svmPtr, hostPtr, size[0]);
        }
      } else {
        result = blitMgr().readBuffer(*hsaMemory, static_cast<char*>(hostPtr) + origin[0], origin,
                                      size, cmd.isEntireMemory());
      }
    } else if (type == CL_COMMAND_MAP_IMAGE) {
      amd::Image* image = cmd.memory().asImage();
      if (mapMemory != nullptr) {
        roc::Memory* hsaMapMemory =
            static_cast<roc::Memory*>(mapMemory->getDeviceMemory(dev(), false));
        result =
            blitMgr().copyImageToBuffer(*hsaMemory, *hsaMapMemory, cmd.origin(),
                                        amd::Coord3D(0, 0, 0), cmd.size(), cmd.isEntireMemory());
      } else {
        result = blitMgr().readImage(*hsaMemory, hostPtr, amd::Coord3D(0), image->getRegion(),
                                     image->getRowPitch(), image->getSlicePitch(), true);
      }
    } else {
      ShouldNotReachHere();
    }

    if (!result) {
      LogError("submitMapMemory failed!");
      cmd.setStatus(CL_OUT_OF_RESOURCES);
    }
  }

  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitUnmapMemory(amd::UnmapMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  roc::Memory* devMemory = static_cast<roc::Memory*>(cmd.memory().getDeviceMemory(dev(), false));

  const device::Memory::WriteMapInfo* mapInfo = devMemory->writeMapInfo(cmd.mapPtr());
  if (nullptr == mapInfo) {
    LogError("Unmap without map call");
    return;
  }

  profilingBegin(cmd, true);

  // Force buffer write for IMAGE1D_BUFFER
  bool imageBuffer = (cmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER);

  // We used host memory
  if ((devMemory->owner()->getHostMem() != nullptr) &&
      (devMemory->owner()->getSvmPtr() == nullptr)) {
    if (mapInfo->isUnmapWrite()) {
      // Target is the backing store, so sync
      devMemory->owner()->signalWrite(nullptr);
      devMemory->syncCacheFromHost(*this);
    }
    if (devMemory->isHostMemDirectAccess()) {
      // Remove memory from VA cache
      dev().removeVACache(devMemory);
    }
  } else if (devMemory->IsPersistentDirectMap()) {
    // Persistent memory - NOP unmap
  } else if (mapInfo->isUnmapWrite()) {
    // Commit the changes made by the user.
    if (!devMemory->isHostMemDirectAccess()) {
      bool result = false;

      amd::Memory* mapMemory = devMemory->mapMemory();
      if (cmd.memory().asImage() && !imageBuffer) {
        amd::Image* image = cmd.memory().asImage();
        if (mapMemory != nullptr) {
          roc::Memory* hsaMapMemory =
              static_cast<roc::Memory*>(mapMemory->getDeviceMemory(dev(), false));
          result =
              blitMgr().copyBufferToImage(*hsaMapMemory, *devMemory, amd::Coord3D(0, 0, 0),
                                          mapInfo->origin_, mapInfo->region_, mapInfo->isEntire());
        } else {
          void* hostPtr = devMemory->owner()->getHostMem();

          result = blitMgr().writeImage(hostPtr, *devMemory, amd::Coord3D(0), image->getRegion(),
                                        image->getRowPitch(), image->getSlicePitch(), true);
        }
      } else {
        amd::Coord3D origin(mapInfo->origin_[0]);
        amd::Coord3D size(mapInfo->region_[0]);
        amd::Coord3D dstOrigin(mapInfo->origin_[0]);
        if (imageBuffer) {
          size_t elemSize = cmd.memory().asImage()->getImageFormat().getElementSize();
          origin.c[0] *= elemSize;
          size.c[0] *= elemSize;
        }
        if (mapMemory != nullptr) {
          roc::Memory* hsaMapMemory =
              static_cast<roc::Memory*>(mapMemory->getDeviceMemory(dev(), false));

          const void* svmPtr = devMemory->owner()->getSvmPtr();
          void* hostPtr = mapMemory->getHostMem();
          if ((svmPtr != nullptr) && (hostPtr != svmPtr)) {
            // Wait on a kernel if one is outstanding
            releaseGpuMemoryFence();
            std::memcpy(hostPtr, svmPtr, size[0]);
          }
          result = blitMgr().copyBuffer(*hsaMapMemory, *devMemory, origin, dstOrigin, size,
                                        mapInfo->isEntire());
        } else {
          result = blitMgr().writeBuffer(cmd.mapPtr(), *devMemory, origin, size);
        }
      }
      if (!result) {
        LogError("submitMapMemory failed!");
        cmd.setStatus(CL_OUT_OF_RESOURCES);
      }
    }

    cmd.memory().signalWrite(&dev());
  }

  devMemory->clearUnmapInfo(cmd.mapPtr());

  profilingEnd();
}

// ================================================================================================
bool VirtualGPU::fillMemory(cl_command_type type, amd::Memory* amdMemory, const void* pattern,
                            size_t patternSize, const amd::Coord3D& surface,
                            const amd::Coord3D& origin, const amd::Coord3D& size, bool forceBlit) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  Memory* memory = dev().getRocMemory(amdMemory);

  bool entire = amdMemory->isEntirelyCovered(origin, size);
  // Synchronize memory from host if necessary
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = entire;
  memory->syncCacheFromHost(*this, syncFlags);

  bool result = false;
  bool imageBuffer = false;
  float fillValue[4];

  // Force fill buffer for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_FILL_IMAGE) && (amdMemory->getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    type = CL_COMMAND_FILL_BUFFER;
    imageBuffer = true;
  }

  // Find the the right fill operation
  switch (type) {
    case CL_COMMAND_SVM_MEMFILL:
    case CL_COMMAND_FILL_BUFFER: {
      amd::Coord3D realSurf(surface[0], surface[1], surface[2]);
      amd::Coord3D realOrigin(origin[0], origin[1], origin[2]);
      amd::Coord3D realSize(size[0], size[1], size[2]);
      // Reprogram fill parameters if it's an IMAGE1D_BUFFER object
      if (imageBuffer) {
        size_t elemSize = amdMemory->asImage()->getImageFormat().getElementSize();
        realOrigin.c[0] *= elemSize;
        realSize.c[0] *= elemSize;
        memset(fillValue, 0, sizeof(fillValue));
        amdMemory->asImage()->getImageFormat().formatColor(pattern, fillValue);
        pattern = fillValue;
        patternSize = elemSize;
      }
      result = blitMgr().fillBuffer(*memory, pattern, patternSize, realSurf, realOrigin, realSize,
                                    entire, forceBlit);
      break;
    }
    case CL_COMMAND_FILL_IMAGE: {
      result = blitMgr().fillImage(*memory, pattern, origin, size, entire);
      break;
    }
    default:
      ShouldNotReachHere();
      break;
  }

  if (!result) {
    LogError("submitFillMemory failed!");
  }

  amdMemory->signalWrite(&dev());
  return true;
}

// ================================================================================================
void VirtualGPU::submitFillMemory(amd::FillMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(cmd);

  bool force_blit = false;
  if (amd::IS_HIP) {
    // Always use blit for memset for HIP.
    force_blit = true;
  }

  if (!fillMemory(cmd.type(), &cmd.memory(), cmd.pattern(), cmd.patternSize(), cmd.surface(),
                  cmd.origin(), cmd.size(), force_blit)) {
    cmd.setStatus(CL_INVALID_OPERATION);
  }
  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitStreamOperation(amd::StreamOperationCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());
  profilingBegin(cmd);

  const cl_command_type type = cmd.type();
  const uint64_t value = cmd.value();
  const uint64_t mask = cmd.mask();
  const unsigned int flags = cmd.flags();
  const size_t sizeBytes = cmd.sizeBytes();
  const size_t offset = cmd.offset();

  amd::Memory* amdMemory = &cmd.memory();
  Memory* memory = dev().getRocMemory(amdMemory);

  if (type == ROCCLR_COMMAND_STREAM_WAIT_VALUE) {
    if (GPU_STREAMOPS_CP_WAIT) {
      uint16_t header = kBarrierVendorPacketHeader;
      Buffer* buff = static_cast<Buffer*>(memory);
      hsa_signal_t signal = buff->getSignal();

      // mask is always applied on value at signal before performing
      // the comparision defiend by 'condition'
      switch (flags) {
        case ROCCLR_STREAM_WAIT_VALUE_GTE: {
          dispatchBarrierValuePacket(header, false, signal, value, mask, HSA_SIGNAL_CONDITION_GTE,
                                     true);
          break;
        }
        case ROCCLR_STREAM_WAIT_VALUE_EQ: {
          dispatchBarrierValuePacket(header, false, signal, value, mask, HSA_SIGNAL_CONDITION_EQ,
                                     true);
          break;
        }
        case ROCCLR_STREAM_WAIT_VALUE_AND: {
          dispatchBarrierValuePacket(header, false, signal, 0, (value & mask),
                                     HSA_SIGNAL_CONDITION_NE, true);
          break;
        }
        case ROCCLR_STREAM_WAIT_VALUE_NOR: {
          uint64_t norValue = ~value & mask;
          dispatchBarrierValuePacket(header, false, signal, norValue, norValue,
                                     HSA_SIGNAL_CONDITION_NE, true);
          break;
        }
        default:
          ShouldNotReachHere();
          break;
      }
    }
    // Use a blit kernel to perform the wait operation
    else {
      // Even though the blit kernel uses an atomic with system scope, we still need to add system scope on
      // the AQLPacket because atomics on kernels can bypass L2 cache on some hardware.
      addSystemScope();

      // mask is applied on value before performing
      // the comparision defined by 'condition'
      bool result = blitMgr().streamOpsWait(*memory, value, offset, sizeBytes, flags, mask);
      ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
              "Waiting for value: 0x%lx."
              " Flags: 0x%lx mask: 0x%lx",
              value, flags, mask);
      if (!result) {
        LogError("submitStreamOperation: Wait failed!");
      }
    }
  } else if (type == ROCCLR_COMMAND_STREAM_WRITE_VALUE) {
    // Even though the blit kernel uses system scope atomic, we still need to add system scope on
    // the AQLPacket because atomics on kernels can bypass L2 cache on some hardware.
    addSystemScope();

    bool result;
    switch (flags) {
      case ROCCLR_STREAM_WRITE_VALUE_DEFAULT: {
        result = blitMgr().streamOpsWrite(*memory, value, offset, sizeBytes);
        break;
      }
      case ROCCLR_STREAM_WRITE_VALUE_INCREMENT: {
        result = blitMgr().streamOpsIncrement(*memory, value, offset, sizeBytes);
        break;
      }
      case ROCCLR_STREAM_WRITE_VALUE_DECREMENT: {
        result = blitMgr().streamOpsDecrement(*memory, value, offset, sizeBytes);
        break;
      }
      default: {
        ShouldNotReachHere();
        break;
      }
    }

    if (!result) {
      LogError("submitStreamOperation: Write failed!");
    }
  } else {
    ShouldNotReachHere();
  }
  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitBatchMemoryOperation(amd::BatchMemoryOperationCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());
  profilingBegin(cmd);

  bool result = blitMgr().batchMemOps(cmd.getParamPtr(), cmd.paramSize(), cmd.count());
  if (!result) {
    LogError("submitBatchMemoryOperation failed!");
  }
  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitVirtualMap(amd::VirtualMapCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(vcmd);

  // Find the amd::Memory object for virtual ptr. vcmd.ptr() is vaddr.
  amd::Memory* vaddr_base_obj = amd::MemObjMap::FindVirtualMemObj(vcmd.ptr());
  if (vaddr_base_obj == nullptr || !(vaddr_base_obj->getMemFlags() & CL_MEM_VA_RANGE_AMD)) {
    profilingEnd();
    return;
  }

  // Get the amd::Memory object for the physical address
  amd::Memory* phys_mem_obj = vcmd.memory();
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;

  // If Physical address is not set, then it is map command. If set, it is unmap command.
  if (phys_mem_obj != nullptr) {
    constexpr bool kParent = false;
    amd::Memory* vaddr_sub_obj = phys_mem_obj->getContext().devices()[0]->CreateVirtualBuffer(
        phys_mem_obj->getContext(), const_cast<void*>(vcmd.ptr()), vcmd.size(),
        phys_mem_obj->getUserData().deviceId, phys_mem_obj->getUserData().locationType, kParent);
    // Map the physical to virtual address the hsa api
    hsa_amd_vmem_alloc_handle_t opaque_hsa_handle;
    opaque_hsa_handle.handle = phys_mem_obj->getUserData().hsa_handle;
    if ((hsa_status = Hsa::vmem_map(vaddr_sub_obj->getSvmPtr(), vcmd.size(),
                                       vaddr_sub_obj->getOffset(), opaque_hsa_handle, 0)) ==
        HSA_STATUS_SUCCESS) {
      assert(amd::MemObjMap::FindMemObj(vcmd.ptr()) == nullptr);
      amd::MemObjMap::AddMemObj(vcmd.ptr(), vaddr_sub_obj);
      vaddr_sub_obj->getUserData().phys_mem_obj = phys_mem_obj;
      phys_mem_obj->getUserData().vaddr_mem_obj = vaddr_sub_obj;
      if (phys_mem_obj->getMemFlags() & ROCCLR_MEM_INTERPROCESS) {
        vaddr_sub_obj->setVmmImported(true);
      }
    } else {
      LogError("HSA Command: hsa_amd_vmem_map failed!");
    }
  } else {
    dispatchBarrierPacket(kBarrierPacketHeader, false);
    Barriers().WaitCurrent();

    amd::Memory* vaddr_sub_obj = amd::MemObjMap::FindMemObj(vcmd.ptr());
    assert(vaddr_sub_obj != nullptr);

    // Unmap the object, since the physical addr is set.
    if ((hsa_status = Hsa::vmem_unmap(vaddr_sub_obj->getSvmPtr(), vcmd.size())) ==
        HSA_STATUS_SUCCESS) {
      // assert the va is mapped and needs to be removed
      vaddr_sub_obj->getContext().devices()[0]->DestroyVirtualBuffer(vaddr_sub_obj);
      amd::MemObjMap::RemoveMemObj(vcmd.ptr());
      if (vaddr_sub_obj->getUserData().phys_mem_obj != nullptr) {
        vaddr_sub_obj->getUserData().phys_mem_obj->getUserData().vaddr_mem_obj = nullptr;
        vaddr_sub_obj->getUserData().phys_mem_obj = nullptr;
      }
      // Release sub_obj now that HW unmap is complete.
      // ~Memory releases parent va_ via parent_->release().
      vaddr_sub_obj->release();
    } else {
      LogError("HSA Command: hsa_amd_vmem_unmap failed");
    }
  }

  // Since this is a memory operation, the HW event set for barrier packet
  // may not encapsulate what the command wants to do. Hence clear the hw_event
  constexpr bool kClearHwEvent = true;
  profilingEnd(kClearHwEvent);
}

// ================================================================================================
void VirtualGPU::submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(cmd);

  amd::Memory* dstMemory = amd::MemObjMap::FindMemObj(cmd.dst());

  if (!dev().isFineGrainedSystem(true) ||
      ((dstMemory != nullptr) && !dev().forceFineGrain(dstMemory))) {
    size_t patternSize = cmd.patternSize();
    size_t fillSize = patternSize * cmd.times();

    size_t offset = reinterpret_cast<uintptr_t>(cmd.dst()) -
                    reinterpret_cast<uintptr_t>(dstMemory->getSvmPtr());

    Memory* memory = dev().getRocMemory(dstMemory);

    amd::Coord3D origin(offset, 0, 0);
    amd::Coord3D size(fillSize, 1, 1);

    assert((dstMemory->validateRegion(origin, size)) && "The incorrect fill size!");

    if (!fillMemory(cmd.type(), dstMemory, cmd.pattern(), cmd.patternSize(), size, origin, size,
                    true)) {
      cmd.setStatus(CL_INVALID_OPERATION);
    }
  } else {
    // Stall GPU for CPU access to memory
    releaseGpuMemoryFence();
    // for FGS capable device, fill CPU memory directly
    amd::SvmBuffer::memFill(cmd.dst(), cmd.pattern(), cmd.patternSize(), cmd.times());
  }

  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitMigrateMemObjects(amd::MigrateMemObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(vcmd);

  for (auto itr : vcmd.memObjects()) {
    // Find device memory
    Memory* memory = dev().getRocMemory(&(*itr));

    if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_HOST) {
      if (!memory->isHostMemDirectAccess()) {
        // Make sure GPU finished operation before synchronization with the backing store
        releaseGpuMemoryFence();
      }
      memory->mgpuCacheWriteBack(*this);
    } else if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED) {
      // Synchronize memory from host if necessary.
      // The sync function will perform memory migration from
      // another device if necessary
      device::Memory::SyncFlags syncFlags;
      memory->syncCacheFromHost(*this, syncFlags);
    } else {
      LogWarning("Unknown operation for memory migration!");
    }
  }

  profilingEnd();
}

// ================================================================================================
bool VirtualGPU::createSchedulerParam() {
  if (nullptr != schedulerQueue_) {
    return true;
  }

  while (true) {
    // The queue is written by multiple threads of the scheduler kernel
    if (HSA_STATUS_SUCCESS !=
        Hsa::queue_create(gpu_device(), 2048, HSA_QUEUE_TYPE_MULTI, callbackQueue, &roc_device_,
                          std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max(),
                          &schedulerQueue_)) {
      break;
    }

#if defined(_WIN32)
  if (!isSchedulerQueueThreadRunning()) {
    std::call_once(scheduler_thread_init_, [this]() { startSchedulerQueueThread(); });
  }
#endif  // _WIN32
    return true;
  }

  if (nullptr != schedulerQueue_) {
    Hsa::queue_destroy(schedulerQueue_);
    schedulerQueue_ = nullptr;
  }

  return false;
}

void VirtualGPU::startSchedulerQueueThread() {
  schedulerQueueThreadRunning_.store(true, std::memory_order_release);

  schedulerQueueThread_ = std::thread([this]() {
    uint64_t updated_write_index = 0;
    while (isSchedulerQueueThreadRunning()) {

      // Wait until scheduler events are added or thread termination
      {
        std::unique_lock<std::mutex> lock(scheduler_mutex_);
        scheduler_cv_.wait(lock, [this]() {
          return !pendingSchedulerEvents_.empty() ||
                 !isSchedulerQueueThreadRunning();
        });
      }

      // Actively monitor the scheduler queue while any sync event is pending.
      bool has_active_events = true;
      while (has_active_events && isSchedulerQueueThreadRunning()) {
        uint64_t write_index = Hsa::queue_load_write_index_scacquire(schedulerQueue_);

        if (write_index > updated_write_index) {
          // New packets in the scheduler queue, ringing the doorbell
          Hsa::signal_store_screlease(schedulerQueue_->doorbell_signal, write_index - 1);
          updated_write_index = write_index;
        } else {
          // Yield briefly before re-checking.
          amd::Os::yield();
        }
        // Check all scheduler completion signals, remove the completed ones
        {
          std::lock_guard<std::mutex> lock(scheduler_mutex_);
          pendingSchedulerEvents_.erase(
            std::remove_if(pendingSchedulerEvents_.begin(),
                           pendingSchedulerEvents_.end(),
                           [](hsa_signal_t signal) {
                             return (Hsa::signal_load_relaxed(signal) == 0);
                           }),
                    pendingSchedulerEvents_.end());
          has_active_events = !pendingSchedulerEvents_.empty();
        }
      }
      // All scheduler completion signals completed, go back to wait
    }
  });
}

// ================================================================================================
uint64_t VirtualGPU::getVQVirtualAddress() {
  Memory* vqMem = dev().getRocMemory(virtualQueue_);
  return reinterpret_cast<uint64_t>(vqMem->getDeviceMemory());
}

// ================================================================================================
bool VirtualGPU::createVirtualQueue(uint deviceQueueSize) {
  uint MinDeviceQueueSize = 16 * 1024;
  deviceQueueSize = std::max(deviceQueueSize, MinDeviceQueueSize);

  maskGroups_ = deviceQueueSize / (512 * Ki);
  maskGroups_ = (maskGroups_ == 0) ? 1 : maskGroups_;

  // Align the queue size for the multiple dispatch scheduler.
  // Each thread works with 32 entries * maskGroups
  uint extra = deviceQueueSize % (sizeof(AmdAqlWrap) * DeviceQueueMaskSize * maskGroups_);
  if (extra != 0) {
    deviceQueueSize += (sizeof(AmdAqlWrap) * DeviceQueueMaskSize * maskGroups_) - extra;
  }

  if (deviceQueueSize_ == deviceQueueSize) {
    return true;
  } else {
    if (0 != deviceQueueSize_) {
      virtualQueue_->release();
      virtualQueue_ = nullptr;
      deviceQueueSize_ = 0;
      schedulerThreads_ = 0;
    }
  }

  uint numSlots = deviceQueueSize / sizeof(AmdAqlWrap);
  uint allocSize = deviceQueueSize;

  // Add the virtual queue header
  allocSize += sizeof(AmdVQueueHeader);
  allocSize = amd::alignUp(allocSize, sizeof(AmdAqlWrap));

  uint argOffs = allocSize;

  // Add the kernel arguments and wait events
  uint singleArgSize = amd::alignUp(
      dev().info().maxParameterSize_ + 64 + dev().settings().numWaitEvents_ * sizeof(uint64_t),
      sizeof(AmdAqlWrap));
  allocSize += singleArgSize * numSlots;

  uint eventsOffs = allocSize;
  // Add the device events
  allocSize += dev().settings().numDeviceEvents_ * sizeof(AmdEvent);

  uint eventMaskOffs = allocSize;
  // Add mask array for events
  allocSize += amd::alignUp(dev().settings().numDeviceEvents_, DeviceQueueMaskSize) / 8;

  uint slotMaskOffs = allocSize;
  // Add mask array for AmdAqlWrap slots
  allocSize += amd::alignUp(numSlots, DeviceQueueMaskSize) / 8;

  // Align size to 64 bytes for more efficient fill operation
  allocSize = amd::alignUp(allocSize, 8 * sizeof(uint64_t));

  // CL_MEM_ALLOC_HOST_PTR/CL_MEM_READ_WRITE
  virtualQueue_ = new (dev().context()) amd::Buffer(dev().context(), CL_MEM_READ_WRITE, allocSize);

  if ((nullptr != virtualQueue_) && !virtualQueue_->create(nullptr)) {
    virtualQueue_->release();
    return false;
  }

  Memory* vqMem = dev().getRocMemory(virtualQueue_);

  if (nullptr == vqMem) {
    return false;
  }

  uint64_t vqVA = reinterpret_cast<uint64_t>(vqMem->getDeviceMemory());

  // Use shadow to prepare the data structure in host.
  auto shadow = std::make_unique<uint8_t[]>(allocSize);

  std::memset(&shadow[0], 0, allocSize);

  AmdVQueueHeader* header = reinterpret_cast<AmdVQueueHeader*>(&shadow[0]);
  // Initialize the virtual queue header
  header->aql_slot_num = numSlots;
  header->event_slot_num = dev().settings().numDeviceEvents_;
  header->event_slot_mask = vqVA + eventMaskOffs;
  header->event_slots = vqVA + eventsOffs;
  header->aql_slot_mask = vqVA + slotMaskOffs;
  header->wait_size = dev().settings().numWaitEvents_;
  header->arg_size = dev().info().maxParameterSize_ + 64;
  header->mask_groups = maskGroups_;

  // Go over all slots and perform initialization
  size_t offset = sizeof(AmdVQueueHeader);
  for (uint i = 0; i < numSlots; ++i) {
    AmdAqlWrap* slot = reinterpret_cast<AmdAqlWrap*>(&shadow[0] + offset);
    uint64_t argStart = vqVA + argOffs + i * singleArgSize;

    slot->aql.kernarg_address = reinterpret_cast<void*>(argStart);
    slot->wait_list = argStart + dev().info().maxParameterSize_ + 64;

    offset += sizeof(AmdAqlWrap);
  }

  amd::Coord3D origin(0, 0, 0);
  amd::Coord3D region(allocSize, 1, 1);

  // copy the data structure from host to GPU
  if (!dev().xferMgr().writeBuffer(&shadow[0], *vqMem, origin, region)) {
    return false;
  }

  deviceQueueSize_ = deviceQueueSize;
  schedulerThreads_ = numSlots / (DeviceQueueMaskSize * maskGroups_);

  return true;
}

// ================================================================================================
#if IS_LINUX
__attribute__((optimize("unroll-all-loops"), always_inline)) static inline void nontemporalMemcpy(
    void* __restrict dst, const void* __restrict src, size_t size) {
#if defined(ATI_ARCH_X86)
#if defined(__AVX512F__)
  for (auto i = 0u; i != size / sizeof(__m512i); ++i) {
    _mm512_stream_si512(reinterpret_cast<__m512i* __restrict&>(dst)++,
                        *reinterpret_cast<const __m512i* __restrict&>(src)++);
  }
  size = size % sizeof(__m512i);
#endif

#if defined(__AVX__)
  for (auto i = 0u; i != size / sizeof(__m256i); ++i) {
    _mm256_stream_si256(reinterpret_cast<__m256i* __restrict&>(dst)++,
                        *reinterpret_cast<const __m256i* __restrict&>(src)++);
  }
  size = size % sizeof(__m256i);
#endif

  for (auto i = 0u; i != size / sizeof(__m128i); ++i) {
    _mm_stream_si128(reinterpret_cast<__m128i* __restrict&>(dst)++,
                     *(reinterpret_cast<const __m128i* __restrict&>(src)++));
  }
  size = size % sizeof(__m128i);

  for (auto i = 0u; i != size / sizeof(long long); ++i) {
    _mm_stream_si64(reinterpret_cast<long long* __restrict&>(dst)++,
                    *reinterpret_cast<const long long* __restrict&>(src)++);
  }
  size = size % sizeof(long long);

  for (auto i = 0u; i != size / sizeof(int); ++i) {
    _mm_stream_si32(reinterpret_cast<int* __restrict&>(dst)++,
                    *reinterpret_cast<const int* __restrict&>(src)++);
  }

  size = size % sizeof(int);
  // Copy remaining bytes for unaligned size
  std::memcpy(dst, src, size);

  // Add memory fence
  _mm_sfence();
#else
  std::memcpy(dst, src, size);
#endif
}
#else
static inline void nontemporalMemcpy(void* __restrict dst, const void* __restrict src,
                                     size_t size) {
  std::memcpy(dst, src, size);
}
#endif

void VirtualGPU::HiddenHeapInit() {
  // We don't really need its id, just want to ensure the queue is created.
  (void)getQueueID();
  const_cast<Device&>(dev()).HiddenHeapInit(*this);
}

// ================================================================================================
bool VirtualGPU::submitKernelInternal(const amd::NDRangeContainer& sizes, const amd::Kernel& kernel,
                                      const_address parameters, void* event_handle,
                                      uint32_t sharedMemBytes, amd::NDRangeKernelCommand* vcmd,
                                      hsa_kernel_dispatch_packet_t* aql_packet,
                                      bool attach_signal) {
  device::Kernel* devKernel = const_cast<device::Kernel*>(kernel.getDeviceKernel(dev()));
  Kernel& gpuKernel = static_cast<Kernel&>(*devKernel);
  size_t ldsUsage = gpuKernel.WorkgroupGroupSegmentByteSize();
  bool imageBufferWrtBack = false;                  // Image buffer write back is required
  std::vector<device::Memory*> wrtBackImageBuffer;  // Array of images for write back

  // Check memory dependency and SVM objects
  bool coopGroups = (vcmd != nullptr) ? vcmd->cooperativeGroups() : false;
  if (!processMemObjects(kernel, parameters, ldsUsage, coopGroups, imageBufferWrtBack,
                         wrtBackImageBuffer)) {
    LogError("Wrong memory objects!");
    return false;
  }

  // Init PrintfDbg object if printf is enabled.
  bool printfEnabled = (gpuKernel.printfInfo().size() > 0) ? true : false;
  if (!printfDbg()->init(printfEnabled)) {
    LogError("\nPrintfDbg object initialization failed!");
    return false;
  }

  const amd::KernelSignature& signature = kernel.signature();
  const amd::KernelParameters& kernelParams = kernel.parameters();

  bool isGraphCapture = command_ != nullptr && command_->getPktCapturingState();

  ClPrint(amd::LOG_INFO, amd::LOG_KERN2, "ShaderName : %s", gpuKernel.getDemangledName().c_str());

  amd::NDRange local_size(sizes.local());
  address hidden_arguments = const_cast<address>(parameters);
  // Calculate local size if it wasn't provided
  devKernel->FindLocalWorkSize(sizes.dimensions(), sizes.global(), local_size);

  uint16_t local[3] = {1, 1, 1};
  uint32_t global[3] = {1, 1, 1};
  for (uint i = 0; i < sizes.dimensions(); i++) {
    global[i] = static_cast<uint32_t>(sizes.global()[i]);
    local[i] = static_cast<uint16_t>(local_size[i]);
  }
  uint64_t spVA = 0;
  // Check if runtime has to setup hidden arguments
  for (uint32_t i = signature.numParameters(); i < signature.numParametersAll(); ++i) {
    const auto& it = signature.at(i);
    switch (it.info_.oclObject_) {
      case amd::KernelParameterDescriptor::HiddenNone:
        break;
      case amd::KernelParameterDescriptor::HiddenGlobalOffsetX: {
        WriteAqlArgAt(hidden_arguments, sizes.offset()[0], it.size_, it.offset_);
        break;
      }
      case amd::KernelParameterDescriptor::HiddenGlobalOffsetY: {
        if (sizes.dimensions() >= 2) {
          WriteAqlArgAt(hidden_arguments, sizes.offset()[1], it.size_, it.offset_);
        }
        break;
      }
      case amd::KernelParameterDescriptor::HiddenGlobalOffsetZ: {
        if (sizes.dimensions() >= 3) {
          WriteAqlArgAt(hidden_arguments, sizes.offset()[2], it.size_, it.offset_);
        }
        break;
      }
      case amd::KernelParameterDescriptor::HiddenPrintfBuffer: {
        uintptr_t bufferPtr = reinterpret_cast<uintptr_t>(printfDbg()->dbgBuffer());
        if (printfEnabled && bufferPtr) {
          WriteAqlArgAt(hidden_arguments, bufferPtr, it.size_, it.offset_);
        }
        break;
      }
      case amd::KernelParameterDescriptor::HiddenHostcallBuffer: {
        if (amd::IS_HIP) {
          if (dev().info().pcie_atomics_) {
            uintptr_t buffer = reinterpret_cast<uintptr_t>(getOrCreateHostcallBuffer());
            if (!buffer) {
              LogError("Kernel expects a hostcall buffer, but none found");
              return false;
            }
            WriteAqlArgAt(hidden_arguments, buffer, it.size_, it.offset_);
          } else {
            LogError("Pcie atomics not enabled, hostcall not supported");
            return false;
          }
        }
        break;
      }
      case amd::KernelParameterDescriptor::HiddenDefaultQueue: {
        uint64_t vqVA = 0;
        amd::DeviceQueue* defQueue = kernel.program().context().defDeviceQueue(dev());
        if (nullptr != defQueue && devKernel->dynamicParallelism()) {
          if (!createVirtualQueue(defQueue->size()) || !createSchedulerParam()) {
            return false;
          }
          vqVA = getVQVirtualAddress();
        }
        WriteAqlArgAt(hidden_arguments, vqVA, it.size_, it.offset_);
        break;
      }
      case amd::KernelParameterDescriptor::HiddenCompletionAction: {
        if (devKernel->dynamicParallelism()) {
          auto params = allocKernArg(sizeof(AmdAqlWrap), 64);
          AmdAqlWrap* wrap = reinterpret_cast<AmdAqlWrap*>(params);
          memset(wrap, 0, sizeof(AmdAqlWrap));
          wrap->state = AQL_WRAP_DONE;
          spVA = reinterpret_cast<uint64_t>(wrap);
        }
        WriteAqlArgAt(hidden_arguments, spVA, it.size_, it.offset_);
        break;
      }
      case amd::KernelParameterDescriptor::HiddenMultiGridSync: {
        bool multiGridSync = (vcmd != nullptr) ? vcmd->cooperativeMultiDeviceGroups() : false;
        bool singleGridSync = (vcmd != nullptr) ? vcmd->cooperativeGroups() : false;
        Device::MGSyncInfo* syncInfo = nullptr;
        if (multiGridSync) {
          // Find CPU pointer to the right sync info structure. It should be after MGSyncData
          syncInfo = reinterpret_cast<Device::MGSyncInfo*>(
              dev().MGSync() + Device::kMGInfoSizePerDevice * dev().index() +
              Device::kMGSyncDataSize);
          // Update sync data address. Use the offset adjustment to the right location
          syncInfo->mgs = reinterpret_cast<Device::MGSyncData*>(
              dev().MGSync() + Device::kMGInfoSizePerDevice * vcmd->firstDevice());
        } else if (singleGridSync) {
          syncInfo = reinterpret_cast<Device::MGSyncInfo*>(allocKernArg(Device::kSGInfoSize, 64));
          syncInfo->mgs = nullptr;
        }
        if (multiGridSync || singleGridSync) {
          // Update sync data address.
          syncInfo->sgs = {0};
          // Fill rest of sync info fields
          syncInfo->grid_id = vcmd->gridId();
          syncInfo->num_grids = vcmd->numGrids();
          syncInfo->prev_sum = vcmd->prevGridSum();
          syncInfo->all_sum = vcmd->allGridSum();
          syncInfo->num_wg = vcmd->numWorkgroups();
        }
        // Update GPU address for grid sync info. Use the offset adjustment for the right
        // location
        WriteAqlArgAt(hidden_arguments, reinterpret_cast<uint64_t>(syncInfo), it.size_, it.offset_);
        break;
      }
      case amd::KernelParameterDescriptor::HiddenHeap:
        // Allocate hidden heap for HIP applications only
        if ((amd::IS_HIP) && (dev().HeapBuffer() == nullptr)) {
          const_cast<Device&>(dev()).HiddenHeapAlloc(*this);
        }
        if (dev().HeapBuffer() != nullptr) {
          // Initialize hidden heap buffer
          if (!isGraphCapture) {
            const_cast<Device&>(dev()).HiddenHeapInit(*this);
            if (!heap_init_fence_emitted_) {
              addSystemScope();
              heap_init_fence_emitted_ = true;
            }
          }
          // Add heap pointer to the code
          size_t heap_ptr = static_cast<size_t>(dev().HeapBuffer()->virtualAddress());
          WriteAqlArgAt(hidden_arguments, heap_ptr, it.size_, it.offset_);
        }
        break;
      case amd::KernelParameterDescriptor::HiddenBlockCountX:
        WriteAqlArgAt(hidden_arguments, global[0] / local[0], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenBlockCountY:
        WriteAqlArgAt(hidden_arguments, global[1] / local[1], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenBlockCountZ:
        WriteAqlArgAt(hidden_arguments, global[2] / local[2], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenGroupSizeX:
        WriteAqlArgAt(hidden_arguments, local[0], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenGroupSizeY:
        WriteAqlArgAt(hidden_arguments, local[1], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenGroupSizeZ:
        WriteAqlArgAt(hidden_arguments, local[2], it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenRemainderX:
        WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(global[0] % local[0]), it.size_,
                      it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenRemainderY:
        if (sizes.dimensions() >= 2) {
          WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(global[1] % local[1]), it.size_,
                        it.offset_);
        }
        break;
      case amd::KernelParameterDescriptor::HiddenRemainderZ:
        if (sizes.dimensions() >= 3) {
          WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(global[2] % local[2]), it.size_,
                        it.offset_);
        }
        break;
      case amd::KernelParameterDescriptor::HiddenGridDims:
        WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(sizes.dimensions()), it.size_,
                      it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenPrivateBase:
        WriteAqlArgAt(hidden_arguments,
                      reinterpret_cast<amd_queue_t*>(gpu_queue_)->private_segment_aperture_base_hi,
                      it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenSharedBase:
        WriteAqlArgAt(hidden_arguments,
                      reinterpret_cast<amd_queue_t*>(gpu_queue_)->group_segment_aperture_base_hi,
                      it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenQueuePtr:
        WriteAqlArgAt(hidden_arguments, gpu_queue_, it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenDynamicLdsSize:
        WriteAqlArgAt(hidden_arguments, sharedMemBytes, it.size_, it.offset_);
        break;
    }
  }
  address argBuffer = hidden_arguments;
  size_t argSize = std::min(gpuKernel.KernargSegmentByteSize(), signature.paramsSize());

  // Find all parameters for the current kernel
  if (!kernel.parameters().deviceKernelArgs() || gpuKernel.isInternalKernel()) {
    // Allocate buffer to hold kernel arguments
    if (isGraphCapture) {
      argBuffer = command_->getGraphKernArg(gpuKernel.KernargSegmentByteSize(),
                                            gpuKernel.KernargSegmentAlignment(), dev().index());
      command_->SetKernelName(gpuKernel.getDemangledName());
    } else {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
              "Kernel name = %s, argSize = %zu, "
              "KernargSegmentByteSize = %lu "
              "KernargSegmentAlignment = %lu",
              gpuKernel.getDemangledName().c_str(), argSize,
              gpuKernel.KernargSegmentByteSize(), gpuKernel.KernargSegmentAlignment());
      argBuffer = reinterpret_cast<address>(
          allocKernArg(gpuKernel.KernargSegmentByteSize(), gpuKernel.KernargSegmentAlignment()));
    }

    nontemporalMemcpy(argBuffer, parameters, argSize);
    if (roc_device_.info().largeBar_ && !isGraphCapture) {
      const auto kernArgImpl = dev().settings().kernel_arg_impl_;
      if (kernArgImpl == KernelArgImpl::DeviceKernelArgsHDP) {
        *dev().info().hdpMemFlushCntl = 1u;
        auto kSentinel = *reinterpret_cast<volatile int*>(dev().info().hdpMemFlushCntl);
      } else if (kernArgImpl == KernelArgImpl::DeviceKernelArgsReadback && argSize != 0) {
#if defined(ATI_ARCH_X86)
        _mm_sfence();
#else
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
        *(argBuffer + argSize - 1) = *(parameters + argSize - 1);
#if defined(ATI_ARCH_X86)
        _mm_mfence();
#else
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
        auto kSentinel = *reinterpret_cast<volatile unsigned char*>(argBuffer + argSize - 1);
      }
    }
  }
  // Check for group memory overflow
  //! @todo Check should be in HSA - here we should have at most an assert
  assert(dev().info().localMemSizePerCU_ > 0);
  if (ldsUsage > dev().info().localMemSizePerCU_) {
    LogError("No local memory available\n");
    return false;
  }

      // Initialize the dispatch Packet
    static_assert(sizeof(hsa_kernel_dispatch_packet_t)
                  == sizeof(hsa_amd_ext_kernel_dispatch_packet_t));

    union {
      hsa_kernel_dispatch_packet_t kernelDispatch;
      hsa_amd_ext_kernel_dispatch_packet_t extKernelDispatch;
    } dispatchPacketUnion;

    auto& dispatchPacket = dispatchPacketUnion.kernelDispatch;
    memset(&dispatchPacket, 0, sizeof(dispatchPacket));

    uint32_t newGlobalSize[3] = {global[0], global[1], global[2]};

    dispatchPacket.header = kInvalidAql;
    dispatchPacket.kernel_object = gpuKernel.KernelCodeHandle();

    // dispatchPacket.header = aqlHeader_;
    // dispatchPacket.setup |= sizes.dimensions() << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

    bool extDispatchPacket =
        (sizes.dimensions() > 0 && sizes.cluster()[0] > 1) ||
        (sizes.dimensions() > 1 && sizes.cluster()[1] > 1) ||
        (sizes.dimensions() > 2 && sizes.cluster()[2] > 1) ||
        dev().settings().ext_dispatch_packet_;

    if (extDispatchPacket) {
      auto& dispatchPacketExt = dispatchPacketUnion.extKernelDispatch;

      dispatchPacketExt.cluster_size_x = sizes.dimensions() > 0 ? sizes.cluster()[0] : 1;
      dispatchPacketExt.cluster_size_y = sizes.dimensions() > 1 ? sizes.cluster()[1] : 1;
      dispatchPacketExt.cluster_size_z = sizes.dimensions() > 2 ? sizes.cluster()[2] : 1;

      // Already validated in HIP Launch Params that newGlobalSize is perfectly divisible by local
      // and it is divisible by cluster size.
      dispatchPacketExt.cluster_count_x = sizes.dimensions() > 0
                                          ? (newGlobalSize[0] / local[0] / sizes.cluster()[0]) : 1;
      dispatchPacketExt.cluster_count_y = sizes.dimensions() > 1
                                          ? (newGlobalSize[1] / local[1] / sizes.cluster()[1]) : 1;
      dispatchPacketExt.cluster_count_z = sizes.dimensions() > 2
                                          ? (newGlobalSize[2] / local[2] / sizes.cluster()[2]) : 1;

    } else {
      dispatchPacket.grid_size_x = sizes.dimensions() > 0 ? newGlobalSize[0] : 1;
      dispatchPacket.grid_size_y = sizes.dimensions() > 1 ? newGlobalSize[1] : 1;
      dispatchPacket.grid_size_z = sizes.dimensions() > 2 ? newGlobalSize[2] : 1;
    }

    if (dev().settings().groupMemCarveout_) {
      uint8_t percent = devKernel->workGroupInfo()->groupMemCarveout_
          ? devKernel->workGroupInfo()->groupMemCarveout_
          : dev().GetGroupMemCarveout();
      auto& dispatchPacketExt = dispatchPacketUnion.extKernelDispatch;
      // Encodings [1, 127] represent a range from 0% (no group memory) to 100% (maximum
      // group memory)
      if (dev().isa().versionMajor() == 12 && dev().isa().versionMinor() == 5) {
        dispatchPacketExt.perf_hint.group_mem_carveout = 127;
      } else {
        dispatchPacketExt.perf_hint.group_mem_carveout = (percent + 1) * 1.26F;
      }
    }

    dispatchPacket.workgroup_size_x = sizes.dimensions() > 0 ? local[0] : 1;
    dispatchPacket.workgroup_size_y = sizes.dimensions() > 1 ? local[1] : 1;
    dispatchPacket.workgroup_size_z = sizes.dimensions() > 2 ? local[2] : 1;

    dispatchPacket.kernarg_address = argBuffer;
    dispatchPacket.group_segment_size = ldsUsage + sharedMemBytes;
    dispatchPacket.private_segment_size = devKernel->workGroupInfo()->privateMemSize_;

    if ((devKernel->workGroupInfo()->usedStackSize_ & 0x1) == 0x1) {
      dispatchPacket.private_segment_size =
          std::max<uint64_t>(dev().StackSize(), dispatchPacket.private_segment_size);
      const size_t maxStackSize = dev().MaxStackSize();
      // we return an explicit error when we exceed the max stack size limit
      if (dispatchPacket.private_segment_size > maxStackSize) {
        LogPrintfError("Scratch size (%u) exceeds max allowed (%zu) for kernel : %s",
                       dispatchPacket.private_segment_size, maxStackSize,
                       gpuKernel.getDemangledName().c_str());
        return false;
      }
    }

    // Pass the header accordingly
    auto aqlHeaderWithOrder = aqlHeader_;

  if (vcmd != nullptr) {
    if (vcmd->getAnyOrderLaunchFlag()) {
      constexpr uint32_t kAqlHeaderMask = ~(1 << HSA_PACKET_HEADER_BARRIER);
      aqlHeaderWithOrder &= kAqlHeaderMask;
    }
    if (vcmd->getCommandEntryScope() == amd::Device::kCacheStateSystem) {
      addSystemScope_ = true;
    }
  }

  // EXPERIMENTAL PWS inter-kernel fence: strip the dispatch packet's acquire/
  // release cache scope (the appended PWS PM4-IB carries the cache flush instead).
  // Applies to both eager and graph-capture paths; skipped for system-scope
  // dispatches (host-visible boundary needs the real flush). The win is in the
  // graph path where the recorded stream replays under a single doorbell, so the
  // overlapped flush hides instead of paying an extra eager launch+doorbell.
  const bool pwsFence = pwsFenceActive() && !addSystemScope_;
  if (pwsFence) {
    constexpr uint16_t kScopeMask =
        static_cast<uint16_t>(~((3u << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                                (3u << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE)));
    aqlHeaderWithOrder &= kScopeMask;
  }

  // Copy scheduler's AQL packet for possible relaunch from the scheduler itself
    if (aql_packet != nullptr) {
      *aql_packet = dispatchPacket;
      if (extDispatchPacket) {
        aql_packet->header = (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) |
                              (1 << HSA_PACKET_HEADER_BARRIER) |
                              (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                              (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
        aql_packet->setup = static_cast<uint8_t>(sizes.dimensions()
                              << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
      } else {
        aql_packet->header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                              (1 << HSA_PACKET_HEADER_BARRIER) |
                              (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                              (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
        aql_packet->setup = static_cast<uint16_t>(sizes.dimensions()
                              << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
      }
    }

    uint16_t rest = 0;
    if (extDispatchPacket) {
      // When launching an AQL packet, the 32 bits has to be written atomically for CP to track,
      // on normal dispatch packet, first 32 bits are header & setup. In ext dispatch packet,
      // the first 32 bits are header, amd_format, setup. Update the "rest" of the 32 bits, so we
      // can commit it atomically in packet_store_release.
      rest = (HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH
              | ((sizes.dimensions() << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS) << 8));
    } else {
      rest = (sizes.dimensions() << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
    }

    metadata_preloader_.PrepareDispatch(gpuKernel.MetadataKernelDescriptor(),
                                        gpuKernel.MetadataPreloadLength(),
                                        gpuKernel.MetadataPreloadOffset());

    if (isGraphCapture) {
      // EXPERIMENTAL PWS inter-kernel fence: record the overlapped PWS cache fence
      // as an extra captured packet BEFORE this dispatch, so the recorded stream is
      // [..., PWS, dispatch] and the last graph packet stays a real dispatch (clean
      // completion-signal semantics in the single-doorbell batch replay). The
      // leading fence (before the very first kernel) is a harmless cache acquire.
      if (pwsFence) {
        capturePwsFence(const_cast<uint8_t*>(command_->getAqlPacket()));
      }
      // Dispatch the packet
      if (!dispatchAqlPacket(&dispatchPacket, aqlHeaderWithOrder, rest,
                             GPU_FLUSH_ON_EXECUTION, command_->getPktCapturingState(),
                             command_->getAqlPacket())) {
        return false;
      }
    } else {
      if (!dispatchAqlPacket(&dispatchPacket, aqlHeaderWithOrder, rest,
                             GPU_FLUSH_ON_EXECUTION, false, nullptr, attach_signal)) {
        return false;
      }
      // Append the overlapped PWS cache fence right after the dispatch.
      if (pwsFence) {
        injectPwsFence();
      }
    }

  // Output printf buffer
  if (!printfDbg()->output(*this, printfEnabled, gpuKernel.printfInfo())) {
    LogError("\nCould not print data from the printf buffer!");
    return false;
  }

  if (gpuKernel.dynamicParallelism()) {
    dispatchBarrierPacket(kBarrierPacketHeader, true);
    if (virtualQueue_ != nullptr) {
      static_cast<KernelBlitManager&>(blitMgr()).runScheduler(
          getVQVirtualAddress(), schedulerQueue_, schedulerThreads_, spVA);
    }
  }

  // Check if image buffer write back is required
  if (imageBufferWrtBack) {
    // Make sure the original kernel execution is done
    releaseGpuMemoryFence();
    for (const auto imageBuffer : wrtBackImageBuffer) {
      Memory* buffer = dev().getGpuMemory(imageBuffer->owner()->parent());
      amd::Image* image = imageBuffer->owner()->asImage();
      Image* devImage = static_cast<Image*>(dev().getGpuMemory(imageBuffer->owner()));
      Memory* cpyImage = dev().getGpuMemory(devImage->CopyImageBuffer());
      amd::Coord3D offs(0);
      // Copy memory from the the backing store image into original buffer
      bool result = blitMgr().copyImageToBuffer(*cpyImage, *buffer, offs, offs, image->getRegion(),
                                                true, image->getRowPitch(), image->getSlicePitch());
    }
  }
  return true;
}

/**
 * @brief Api to dispatch a kernel for execution. The implementation
 * parses the input object, an instance of virtual command to obtain
 * the parameters of global size, work group size, offsets of work
 * items, enable/disable profiling, etc.
 *
 * It also parses the kernel arguments buffer to inject into Hsa Runtime
 * the list of kernel parameters.
 */
// ================================================================================================
void VirtualGPU::submitKernel(amd::NDRangeKernelCommand& vcmd) {
  if (vcmd.cooperativeGroups()) {
    // Wait for the execution on the current queue, since the coop groups will use the device queue
    releaseGpuMemoryFence(kSkipCpuWait);

    // Get device queue for exclusive GPU access
    VirtualGPU* queue = dev().xferQueue();
    if (!queue) {
      LogError("Runtime failed to acquire a cooperative queue!");
      vcmd.setStatus(CL_INVALID_OPERATION);
      return;
    }

    // Lock the queue, using the blit manager lock
    std::scoped_lock k(*(queue->blitMgr().lockXfer()));

    queue->profilingBegin(vcmd);

    // Add a dependency into the device queue on the current queue
    queue->Barriers().AddExternalSignal(Barriers().GetLastSignal());

    if (dev().settings().gwsInitSupported_ == true) {
      uint32_t workgroups = vcmd.numWorkgroups();
      static_cast<KernelBlitManager&>(queue->blitMgr()).RunGwsInit(workgroups - 1);
    }

    // Sync AQL packets
    queue->setAqlHeader(dispatchPacketHeader_);

    // Submit kernel to HW
    if (!queue->submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(),
                                     static_cast<void*>(as_cl(&vcmd.event())),
                                     vcmd.sharedMemBytes(), &vcmd)) {
      LogError("AQL dispatch failed!");
      vcmd.setStatus(CL_INVALID_OPERATION);
    }
    // Wait for the execution on the device queue. Keep the current queue in-order
    queue->releaseGpuMemoryFence(kSkipCpuWait);

    // Add a dependency into the current queue on the coop queue
    Barriers().AddExternalSignal(queue->Barriers().GetLastSignal());
    hasPendingDispatch_ = true;
    retainExternalSignals_ = true;

    queue->profilingEnd();
  } else {
    // Make sure VirtualGPU has an exclusive access to the resources
    std::scoped_lock lock(execution());

    profilingBegin(vcmd);

    if (vcmd.dynDataPrefetchConfig().isEnabled()) {
      metadata_preloader_.SetDynDataPrefetchRegions(vcmd.dynDataPrefetchConfig());
    }

    // Submit kernel to HW
    if (!submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(),
                              static_cast<void*>(as_cl(&vcmd.event())), vcmd.sharedMemBytes(),
                              &vcmd)) {
      LogError("AQL dispatch failed!");
      vcmd.setStatus(CL_INVALID_OPERATION);
    }

    metadata_preloader_.ClearDynDataPrefetchConfig();

    profilingEnd();
  }
}

// ================================================================================================
void VirtualGPU::submitNativeFn(amd::NativeFnCommand& cmd) {}

// ================================================================================================
void VirtualGPU::submitMarker(amd::Marker& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());
  if (vcmd.CpuWaitRequested()) {
    force_irq_ = IS_WINDOWS;
    // It should be safe to call flush directly if there are not pending dispatches without
    // HSA signal callback
    if (!dedicated_queue_ && gpu_queue_ == nullptr) {
      void* md_rb = nullptr;
      SetGpuQueue(roc_device_.AcquireActiveQueue(priority_, nullptr, nullptr, &md_rb), md_rb);
    }
    flush(vcmd.GetBatchHead());
  } else {
    profilingBegin(vcmd);
    const Settings& settings = dev().settings();
    hsa_signal_t ipc_s{0};
    if (vcmd.ipcCompletionSignal() != nullptr) {
      ipc_s.handle = static_cast<uint64_t>(
          reinterpret_cast<uintptr_t>(vcmd.ipcCompletionSignal()->getHandle()));
    }

    if (vcmd.ipcDepSignal() != nullptr) {
      hsa_signal_t s;
      s.handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
          vcmd.ipcDepSignal()->getHandle()));
      WaitCompleteSignal(s);
    } else if (timestamp_ != nullptr || ipc_s.handle != 0) {
      // IPC event record: if ipc_s is non-zero, first dispatch a NOP barrier with
      // the IPC signal as completion_signal; it fires when prior work completes.
      if (ipc_s.handle != 0) {
        if (settings.barrier_value_packet_) {
          dispatchBarrierValuePacket(kBarrierVendorPacketNopScopeHeader, false, hsa_signal_t{0}, 0,
                                     0, HSA_SIGNAL_CONDITION_EQ, true, ipc_s);
        } else {
          dispatchBarrierPacket(kNopPacketHeader, true, ipc_s);
        }
      }

      int32_t releaseFlags = vcmd.getCommandEntryScope();
      if (releaseFlags == Device::CacheState::kCacheStateIgnore) {
        if (settings.barrier_value_packet_ && vcmd.profilingInfo().marker_ts_) {
          dispatchBarrierValuePacket(kBarrierVendorPacketNopScopeHeader, true);
        } else {
          dispatchBarrierPacket(kNopPacketHeader, false);
        }
      } else {
        // Submit a barrier with a cache flushes.
        force_irq_ = IS_WINDOWS;
        if (settings.barrier_value_packet_ && vcmd.profilingInfo().marker_ts_) {
          dispatchBarrierValuePacket(kBarrierVendorPacketHeader, true);
        } else {
          dispatchBarrierPacket(kBarrierPacketHeader, false);
        }
        hasPendingDispatch_ = false;
      }
    }
    profilingEnd();
  }
  force_irq_ = false;
}

// ================================================================================================
void VirtualGPU::submitAccumulate(amd::AccumulateCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());
  profilingBegin(vcmd);

  // Register pre-patched HW event signals with the Timestamp for profiling.
  // These signals were configured by ApplyHwEventPatches (isPacketDispatch_,
  // done_ flags set there) but bypass ActiveSignal, so they must be added
  // here so checkGpuTime → ExtractSignalTiming → addTimestamps picks them up.
  if (timestamp_ != nullptr) {
    for (const auto& [_, events] : vcmd.getHwEvents()) {
      for (void* hw_event : events) {
        auto* ps = reinterpret_cast<ProfilingSignal*>(hw_event);
        if (ps != nullptr) {
          timestamp_->AddProfilingSignal(ps);
        }
      }
    }
  }

  const Settings& settings = dev().settings();
  if (settings.barrier_value_packet_) {
    dispatchBarrierValuePacket(kBarrierVendorPacketNopScopeHeader, true);
  } else {
    dispatchBarrierPacket(kNopPacketHeader, false);
  }

  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitAcquireExtObjects(amd::AcquireExtObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  profilingBegin(vcmd);
  addSystemScope();
  profilingEnd();
}

// ================================================================================================
void VirtualGPU::submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());
  profilingBegin(vcmd);
  profilingEnd();
}

// ================================================================================================
void VirtualGPU::flush(amd::Command* list, bool wait) {
  // If barrier is requested, then wait for everything, otherwise
  // a per disaptch wait will occur later in updateCommandsState()
  releaseGpuMemoryFence();
  updateCommandsState(list);
}

// ================================================================================================
void VirtualGPU::addPinnedMem(amd::Memory* mem) {
  // Optimize pinning path for Linux only(KFD has special tracking for pinned memory) or OpenCL,
  // since OpenCL always waits for completion on CPU
  if ((command_ != nullptr) && (IS_LINUX || !amd::IS_HIP)) {
    command_->AddPinnedMemory(mem);
  } else {
    //! @note: ROCr backend doesn't have per resource busy tracking, hence runtime has to wait
    //!        unconditionally, before it can release pinned memory
    releaseGpuMemoryFence();
    mem->release();
  }
}

// ================================================================================================
void VirtualGPU::enableSyncBlit() const { blitMgr_->enableSynchronization(); }

// ================================================================================================
void VirtualGPU::submitPerfCounter(amd::PerfCounterCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  std::scoped_lock lock(execution());

  const amd::PerfCounterCommand::PerfCounterList counters = vcmd.getCounters();

  if (vcmd.getState() == amd::PerfCounterCommand::Begin) {
    // Create a profile for the profiling AQL packet
    PerfCounterProfile* profileRef = new PerfCounterProfile(roc_device_);
    if (profileRef == nullptr || !profileRef->Create()) {
      LogError("Failed to create performance counter profile");
      vcmd.setStatus(CL_INVALID_OPERATION);
      return;
    }

    // Make sure all performance counter objects to use the same profile
    PerfCounter* counter = nullptr;
    for (uint i = 0; i < vcmd.getNumCounters(); ++i) {
      amd::PerfCounter* amdCounter = static_cast<amd::PerfCounter*>(counters[i]);
      counter = static_cast<PerfCounter*>(amdCounter->getDeviceCounter());

      if (nullptr == counter) {
        amd::PerfCounter::Properties prop = amdCounter->properties();
        PerfCounter* rocCounter = new PerfCounter(roc_device_, prop[CL_PERFCOUNTER_GPU_BLOCK_INDEX],
                                                  prop[CL_PERFCOUNTER_GPU_COUNTER_INDEX],
                                                  prop[CL_PERFCOUNTER_GPU_EVENT_INDEX]);

        if (nullptr == rocCounter || rocCounter->gfxVersion() == PerfCounter::ROC_UNSUPPORTED) {
          LogError("Failed to create the performance counter");
          vcmd.setStatus(CL_INVALID_OPERATION);
          delete rocCounter;
          return;
        }

        amdCounter->setDeviceCounter(rocCounter);
        counter = rocCounter;
      }

      counter->setProfile(profileRef);
    }

    if (!profileRef->initialize()) {
      LogError("Failed to initialize performance counter");
      vcmd.setStatus(CL_INVALID_OPERATION);
    } else if (profileRef->createStartPacket() == nullptr) {
      LogError("Failed to create AQL packet for start profiling");
      vcmd.setStatus(CL_INVALID_OPERATION);
    } else {
      dispatchCounterAqlPacket(profileRef->prePacket(), counter->gfxVersion(), false,
                               profileRef->api());
    }

    profileRef->release();
  } else if (vcmd.getState() == amd::PerfCounterCommand::End) {
    // Since all performance counters should use the same profile, use the 1st
    // one to get the profile object
    amd::PerfCounter* amdCounter = static_cast<amd::PerfCounter*>(counters[0]);
    PerfCounter* counter = static_cast<PerfCounter*>(amdCounter->getDeviceCounter());
    if (counter == nullptr) {
      LogError("Invalid Performance Counter");
      vcmd.setStatus(CL_INVALID_OPERATION);
      return;
    }
    PerfCounterProfile* profileRef = counter->profileRef();

    // create the AQL packet for stop profiling
    if (profileRef->createStopPacket() == nullptr) {
      LogError("Failed to create AQL packet for stop profiling");
      vcmd.setStatus(CL_INVALID_OPERATION);
    }
    dispatchCounterAqlPacket(profileRef->postPacket(), counter->gfxVersion(), true,
                             profileRef->api());
  } else {
    LogError("Unsupported performance counter state");
    vcmd.setStatus(CL_INVALID_OPERATION);
  }
}

// ================================================================================================
void *VirtualGPU::getOrCreateHostcallBuffer() {
  if (hostcallBuffer_ != nullptr) {
    return hostcallBuffer_;
  }

  // The number of packets required in each buffer is at least equal to the
  // maximum number of waves supported by the device.
  auto wavesPerCu =
      dev().info().maxThreadsPerCU_ / dev().info().wavefrontWidth_;
  auto numPackets = dev().info().maxComputeUnits_ * wavesPerCu;

  auto size = amd::getHostcallBufferSize(numPackets);
  auto align = amd::getHostcallBufferAlignment();

  hostcallBuffer_ = dev().hostAlloc(
      size, align, Device::MemorySegment::kAtomics, nullptr, false);
  if (!hostcallBuffer_) {
    ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE, "Failed to create hostcall buffer");
    return nullptr;
  }
  hostcallBufferSize_ = size;

  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
          "Created hostcall buffer %p (numPackets == %d, size == %d, align == "
          "%d) for virtual "
          "queue %p\n",
          hostcallBuffer_, numPackets, size, align, this);

  if (!amd::enableHostcalls(dev(), hostcallBuffer_, numPackets)) {
    ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE,
            "Failed to register hostcall buffer %p with listener",
            hostcallBuffer_);
    dev().hostFree(hostcallBuffer_, hostcallBufferSize_);
    hostcallBuffer_ = nullptr;
    hostcallBufferSize_ = 0;
    return nullptr;
  }
  return hostcallBuffer_;
}

// ================================================================================================
static void convertDynDataPrefetchToHsa(const amd::DynDataPrefetchRegion* regions,
                                        uint8_t hints,
                                        amd_data_prefetch_t* hw,
                                        uint32_t numRegions) {
  for (uint32_t i = 0; i < numRegions && i < amd::kDynDataPrefetchMaxRegions; ++i) {
    const auto& r = regions[i];

    uintptr_t addr = reinterpret_cast<uintptr_t>(r.baseAddress);
    // addr_lo is VA[31:8] (256B aligned), addr_hi is VA[56:32]
    hw[i].addr_lo = static_cast<uint32_t>((addr >> 8) & 0xFFFFFFu);
    hw[i].addr_hi = static_cast<uint32_t>((addr >> 32) & 0x1FFFFFFu);
    // 256B is the hardware prefetch engine request unit
    hw[i].burst_size = static_cast<uint32_t>((r.burstSize / 256u) - 1u);
    hw[i].num_burst = r.numBursts - 1u;
    hw[i].stride = static_cast<uint32_t>(r.stride / 256u);
    hw[i].cooperative = 1;
    hw[i].temporal = hints & 0x3u;
    hw[i].scope = 2; // DEVICE
    hw[i].mode = 1; // ABSOLUTE_VA
  }
}

void VirtualGPU::MetaDataPreloader::SetPacket(
    hsa_kernel_dispatch_packet_t* aql,  uint16_t header,
    hsa_amd_metadata_kernel_dispatch_packet_t* metadata) {
  assert(pending_descriptor_ != nullptr);

  // Headers must remain HSA_PACKET_TYPE_INVALID until we publish valid metadata headers
  // at the end. Only clear fields that could otherwise contain stale required-zero data.
  std::memset(metadata->reserved0, 0, sizeof(metadata->reserved0));
  std::memset(&metadata->launch_descriptor, 0, sizeof(metadata->launch_descriptor));

  if (dyn_data_prefetch_enabled_ && dyn_data_prefetch_num_regions_ > 0) {
    metadata->launch_descriptor.version = launch_descriptor_version_;
    convertDynDataPrefetchToHsa(
        dyn_data_prefetch_regions_,
        dyn_data_prefetch_hints_,
        metadata->launch_descriptor.prefetch,
        dyn_data_prefetch_num_regions_);
  }

  // Write event_id from amd_signal_t directly (non-interrupt signals have event_id == 0).
  if (aql->completion_signal.handle) {
    auto* signal = reinterpret_cast<amd_signal_t*>(aql->completion_signal.handle);
    metadata->event_id = signal->event_id;
  } else {
    metadata->event_id = 0;
  }

  // Fill hsa_amd_metadata_kernel_dispatch_packet->kernel_descriptor fields.
  // The metadata packet kernel descriptor fields are a subset of
  // kernel_descriptor_t (Code Object V3 Kernel Descriptor) from the AQL packet, from bytes
  // llvm::amdhsa::KERNEL_CODE_ENTRY_BYTE_OFFSET_OFFSET(16) to sizeof(kernel_descriptor_t).
  // See include/llvm/Support/AMDHSAKernelDescriptor.h.
  std::memcpy(&metadata->kernel_descriptor, pending_descriptor_,
              sizeof(metadata->kernel_descriptor));

  // Fill hsa_amd_metadata_kernel_dispatch_packet->kernarg_preload_* fields
  constexpr uint16_t kPreload_limit =
      (sizeof(metadata->kernarg_preload_0_14) +
       sizeof(metadata->kernarg_preload_15_29) +
       sizeof(metadata->kernarg_preload_30_31)) / sizeof(uint32_t);

  uint16_t preload_length = pending_preload_length_;
  if (preload_length > kPreload_limit) {
    metadata->kernel_descriptor.kernarg_preload.length = kPreload_limit;
    preload_length = kPreload_limit;
  }
  ClPrint(amd::LOG_DEBUG, amd::LOG_AQL,
          "metadata prefetch: preload_length=%u, preload_offset=%u, preload_limit=%u",
          preload_length, pending_preload_offset_, kPreload_limit);

  if (preload_length > 0) {
    const uint8_t* kernargs = reinterpret_cast<const uint8_t*>(aql->kernarg_address);
    assert(kernargs);
    // Kernarg preload offset is in DWORDs
    const uint8_t* src = kernargs + pending_preload_offset_ * sizeof(uint32_t);
    uint16_t remain = preload_length;

    // Copy kernarg_preload 0-14
    uint16_t n = std::min<uint16_t>(
        remain, sizeof(metadata->kernarg_preload_0_14) / sizeof(uint32_t));
    if (n < (sizeof(metadata->kernarg_preload_0_14) / sizeof(uint32_t))) {
      std::memset(metadata->kernarg_preload_0_14, 0, sizeof(metadata->kernarg_preload_0_14));
    }
    std::memcpy(metadata->kernarg_preload_0_14, src, n * sizeof(uint32_t));
    remain -= n;

    // Copy kernarg_preload 15-29
    if (remain > 0) {
      src += n * sizeof(uint32_t);
      n = std::min<uint16_t>(
          remain, sizeof(metadata->kernarg_preload_15_29) / sizeof(uint32_t));
      if (n < (sizeof(metadata->kernarg_preload_15_29) / sizeof(uint32_t))) {
        std::memset(metadata->kernarg_preload_15_29, 0,
                    sizeof(metadata->kernarg_preload_15_29));
      }
      std::memcpy(metadata->kernarg_preload_15_29, src, n * sizeof(uint32_t));
      remain -= n;

      // Copy kernarg_preload 30-31
      if (remain > 0) {
        src += n * sizeof(uint32_t);
        n = std::min<uint16_t>(
            remain, sizeof(metadata->kernarg_preload_30_31) / sizeof(uint32_t));
        if (n < (sizeof(metadata->kernarg_preload_30_31) / sizeof(uint32_t))) {
          std::memset(metadata->kernarg_preload_30_31, 0,
                      sizeof(metadata->kernarg_preload_30_31));
        }
        std::memcpy(metadata->kernarg_preload_30_31, src, n * sizeof(uint32_t));
      }
    }
  }

  // Write headers last — arms the metadata packet for the CP.
  // Plain stores are sufficient here: the subsequent packet_store_release on the main
  // AQL dispatch header provides a release fence that orders all metadata writes (body
  // and headers) before the CP sees the valid dispatch packet.
  uint32_t metadata_header = GetType(header) | metadata_version_header_;
  metadata->header3 = metadata_header;
  metadata->header2 = metadata_header;
  metadata->header1 = metadata_header;
  metadata->header0 = metadata_header;
}
}  // End of roc namespace

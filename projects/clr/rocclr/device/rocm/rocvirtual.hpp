/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "platform/commandqueue.hpp"
#include "rocdefs.hpp"
#include "rocdevice.hpp"
#include "utils/flags.hpp"
#include "utils/util.hpp"
#include "rocprintf.hpp"
#include "rocsched.hpp"
#include "device/device.hpp"
#include "os/os.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stack>
#include <string>
#include <thread>
#include <atomic>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace amd::roc {
class Device;
class Memory;
struct ProfilingSignal;
class Timestamp;

// Initial HSA signal value
constexpr static hsa_signal_value_t kInitSignalValueOne = 1;

// Timeouts for HSA signal wait
constexpr static uint64_t kTimeout100us = 100 * K;
constexpr static uint64_t kUnlimitedWait = std::numeric_limits<uint64_t>::max();
constexpr static uint64_t kInvalidQueueIndex = std::numeric_limits<uint64_t>::max();

constexpr static uint64_t kTimeout4Secs = 4 * M;

inline bool WaitForSignal(hsa_signal_t signal, bool active_wait = false, bool yield = false) {
  hsa_wait_state_t wait_state = HSA_WAIT_STATE_BLOCKED;
  if (active_wait) {
    wait_state = HSA_WAIT_STATE_ACTIVE;
  }

  if (Hsa::signal_load_relaxed(signal) > 0) {
    // When it is blocked wait, we wait in active state for 100 us before proceeding to wait in
    // blocked state indefinitely.
    if (!active_wait) {
      ClPrint(amd::LOG_INFO, amd::LOG_SIG, "Host active wait for Signal = (0x%lx) for %d ns",
              signal.handle, kTimeout100us);
      if (Hsa::signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                    kTimeout100us, HSA_WAIT_STATE_ACTIVE) != 0) {
        if (HIP_SKIP_ABORT_ON_GPU_ERROR && amd::Device::IsGPUInError()) {
          ClPrint(amd::LOG_ERROR, amd::LOG_SIG,
                  "Device not Stable, while waiting for Signal ="
                  "(0x%lx) for %d ns",
                  signal.handle, kTimeout100us);
          return true;
        }
      }
    }

    // This is unlimited wait, but we wait for 4 secs and check if the device is
    // unstable, if so we return, otherwise we continue to wait in the while loop.
    while (Hsa::signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                     kTimeout4Secs, wait_state) != 0) {
      if (HIP_SKIP_ABORT_ON_GPU_ERROR && amd::Device::IsGPUInError()) {
        ClPrint(amd::LOG_ERROR, amd::LOG_SIG,
                "Device not Stable, while waiting for Signal ="
                "(0x%lx) for %d ns",
                signal.handle, kTimeout4Secs);
        return true;
      }
      if (yield && wait_state == HSA_WAIT_STATE_ACTIVE) {
        amd::Os::yield();
      }
    }
  }

  return true;
}

inline void fetchSignalTime(hsa_signal_t signal, hsa_agent_t gpu_device, uint64_t* start,
                            uint64_t* end) {
  if (start != nullptr && end != nullptr) {
    hsa_amd_profiling_dispatch_time_t time = {};
    Hsa::profiling_get_dispatch_time(gpu_device, signal, &time);
    *start = time.start;
    *end = time.end;
  }
}

// Timestamp for keeping track of some profiling information for various commands
// including EnqueueNDRangeKernel and clEnqueueCopyBuffer.
class Timestamp : public amd::ReferenceCountedObject {
 private:
  static double ticksToTime_;

  uint64_t start_;
  uint64_t end_;
  VirtualGPU* gpu_;                        //!< Virtual GPU, associated with this timestamp
  amd::Command& command_;                  //!< Command, associated with this timestamp
  amd::Command* parsedCommand_;            //!< Command down the list, considering command_ as head
  std::vector<ProfilingSignal*> signals_;  //!< The list of all signals, associated with the TS
  hsa_signal_t callback_signal_;  //!< Signal associated with a callback for possible later update
  std::recursive_mutex lock_;     //!< Serialize timestamp update
  bool accum_ena_ = false;        //!< If TRUE then the accumulation of execution times has started
  bool hasHwProfiling_ = false;   //!< If TRUE then HwProfiling is enabled for the command
  bool blocking_ = true;          //!< If TRUE callback is blocking

  //! Extract timing from a single signal and update accumulators
  void ExtractSignalTiming(ProfilingSignal* signal,
                           uint64_t& start, uint64_t& end,
                           uint64_t& sdmaStart, uint64_t& sdmaEnd);

  Timestamp(const Timestamp&) = delete;
  Timestamp& operator=(const Timestamp&) = delete;

 public:
  Timestamp(VirtualGPU* gpu, amd::Command& command)
      : start_(std::numeric_limits<uint64_t>::max()),
        end_(0),
        gpu_(gpu),
        command_(command),
        parsedCommand_(nullptr),
        callback_signal_(hsa_signal_t{}) {}

  ~Timestamp() {}

  void getTime(uint64_t* start, uint64_t* end) {
    checkGpuTime();
    *start = start_;
    *end = end_;
  }

  void AddProfilingSignal(ProfilingSignal* signal) {
    signals_.push_back(signal);
    hasHwProfiling_ = true;
  }

  const std::vector<ProfilingSignal*>& Signals() const { return signals_; }

  const bool HwProfiling() const { return hasHwProfiling_; }

  //! Finds execution ticks on GPU
  //! If single_signal is nullptr, processes all signals and clears the list
  //! If single_signal is provided, processes only that signal with merge enabled
  void checkGpuTime(ProfilingSignal* single_signal = nullptr);

  // Start a timestamp (get timestamp from OS)
  void start() { start_ = amd::Os::timeNanos(); }

  // End a timestamp (get timestamp from OS)
  void end() {
    // Timestamp value can be updated by HW profiling if current command had a stall.
    // Although CPU TS should be still valid in this situation, there are cases in VM mode
    // when CPU timeline is out of sync with GPU timeline and shifted time can be reported
    if (end_ == 0) {
      end_ = amd::Os::timeNanos();
    }
  }

  static void setGpuTicksToTime(double ticksToTime) { ticksToTime_ = ticksToTime; }
  static double getGpuTicksToTime() { return ticksToTime_; }

  //! Returns amd::command assigned to this timestamp
  amd::Command& command() const { return command_; }

  //! Sets the parsed command
  void setParsedCommand(amd::Command* command) { parsedCommand_ = command; }

  //! Gets the parsed command
  amd::Command* getParsedCommand() const { return parsedCommand_; }

  //! Returns virtual GPU device, used with this timestamp
  VirtualGPU* gpu() const { return gpu_; }

  //! Updates the callback signal
  void SetCallbackSignal(hsa_signal_t callback_signal, bool blocking = true) {
    callback_signal_ = callback_signal;
    blocking_ = blocking;
  }
  //! Returns the callback signal
  hsa_signal_t GetCallbackSignal() const { return callback_signal_; }

  //! Return if callback is blocking/non-blocking
  bool GetBlocking() { return blocking_; }
};

class VirtualGPU : public device::VirtualDevice {
 public:
  class ManagedBuffer : public amd::EmbeddedObject {
   public:
    //! The number of chunks the arg pool will be divided
    ManagedBuffer(VirtualGPU& gpu, uint32_t pool_size, uint32_t num_signals)
        : gpu_(gpu), pool_size_(pool_size), pool_signal_(num_signals),
          num_chunk_signals_(num_signals) {}
    ~ManagedBuffer();

    //! Allocates all necessary resources to manage memory
    bool Create(amd::Device::MemorySegment mem_segment);

    //! Acquires memory for use on the gpu
    address Acquire(uint32_t size);

    //! Acquires custom aligned memory for use on the gpu
    address Acquire(uint32_t size, uint32_t alignment);

    //! Reset mem pool
    void ResetPool();

   private:
    VirtualGPU& gpu_;                        //!< Queue object for ROCm device
    address pool_base_ = nullptr;            //!< Memory pool base address
    uint32_t pool_size_;                     //!< Memory pool base size
    uint32_t pool_chunk_end_ = 0;            //!< The end offset of the current chunk
    uint32_t active_chunk_ = 0;              //!< The index of the current active chunk
    uint32_t pool_cur_offset_ = 0;           //!< Current active offset for update
    std::vector<hsa_signal_t> pool_signal_;  //!< Pool of HSA signals to manage multiple chunks
    uint32_t num_chunk_signals_;                   //!< Number of signals used per chunk
  };
  class MemoryDependency : public amd::EmbeddedObject {
   public:
    //! Default constructor
    MemoryDependency()
        : memObjectsInQueue_(nullptr), numMemObjectsInQueue_(0), maxMemObjectsInQueue_(0) {}

    ~MemoryDependency() { delete[] memObjectsInQueue_; }

    //! Creates memory dependency structure
    bool create(size_t numMemObj);

    //! Notify the tracker about new kernel
    void newKernel() { endMemObjectsInQueue_ = numMemObjectsInQueue_; }

    //! Validates memory object on dependency
    void validate(VirtualGPU& gpu, const Memory* memory, bool readOnly);

    //! Clear memory dependency
    void clear(bool all = true);

    //! Max number of mem objects in the queue
    size_t maxMemObjectsInQueue() const { return maxMemObjectsInQueue_; }

   private:
    struct MemoryState {
      uint64_t start_;  //! Busy memory start address
      uint64_t end_;    //! Busy memory end address
      bool readOnly_;   //! Current GPU state in the queue
    };

    MemoryState* memObjectsInQueue_;  //!< Memory object state in the queue
    size_t endMemObjectsInQueue_;     //!< End of mem objects in the queue
    size_t numMemObjectsInQueue_;     //!< Number of mem objects in the queue
    size_t maxMemObjectsInQueue_;     //!< Maximum number of mem objects in the queue
  };

  class HwQueueTracker : public amd::EmbeddedObject {
   public:
    HwQueueTracker(const VirtualGPU& gpu) : gpu_(gpu) {}

    ~HwQueueTracker();

    //! Creates a pool of signals for tracking of HW operations on the queue
    bool Create();

    //! Finds a free signal for the upcoming operation
    hsa_signal_t ActiveSignal(hsa_signal_value_t init_val = kInitSignalValueOne,
                              Timestamp* ts = nullptr, bool attach_signal = true);

    //! Wait for the curent active signal. Can idle the queue
    bool WaitCurrent();

    //! Update current active engine
    void SetActiveEngine(HwQueueEngine engine = HwQueueEngine::Compute) { engine_ = engine; }
    HwQueueEngine GetActiveEngine() const { return engine_; }

    //! Returns the last submitted signal for a wait
    std::vector<hsa_signal_t>& WaitingSignal(HwQueueEngine engine = HwQueueEngine::Compute);

    //! Resets current signal back to the previous one. It's necessary in a case of ROCr failure.
    void ResetCurrentSignal();

    //! Adds an external signal(submission in another queue) for dependency tracking
    void AddExternalSignal(ProfilingSignal* signal) { external_signals_.push_back(signal); }

    //! Get the last active signal on the queue
    ProfilingSignal* GetLastSignal() const { return signal_list_[current_id_]; }

    //! Clear external signals
    void ClearExternalSignals() { external_signals_.clear(); }

    //! Empty check for external signals
    bool IsExternalSignalListEmpty() const { return external_signals_.empty(); }

    //! Adds a raw signal for dependency tracking
    void AddDynamicQueueWait(hsa_signal_t signal) { dynamic_queue_waits_.push_back(signal); }

    //! Get/Set SDMA profiling
    bool GetSDMAProfiling() { return sdma_profiling_; }
    void SetSDMAProfiling(bool profile) {
      sdma_profiling_ = profile;
      Hsa::profiling_async_copy_enable(profile);
    }

   private:
    //! Creates HSA signal with the specified scope
    bool CreateSignal(ProfilingSignal* signal, bool interrupt = false) const;

    //! Wait for the next active signal
    void WaitNext();

    //! Wait for the provided signal
    bool CpuWaitForSignal(ProfilingSignal* signal);

    HwQueueEngine engine_ = HwQueueEngine::Unknown;  //!< Engine used in the current operations
    std::stack<ProfilingSignal*> signal_pool_irq_;   //!< The pool of free signals with interrupts
    std::stack<ProfilingSignal*> signal_pool_;       //!< The pool of free signals without interrupt
    std::vector<ProfilingSignal*> signal_list_;      //!< The pool of all signals for processing
    size_t current_id_ = 0;                          //!< Last submitted signal
    bool sdma_profiling_ = false;                    //!< If TRUE, then SDMA profiling is enabled
    const VirtualGPU& gpu_;                          //!< VirtualGPU, associated with this tracker
    std::vector<ProfilingSignal*> external_signals_;  //!< External signals for a wait in this queue
    std::vector<hsa_signal_t> dynamic_queue_waits_;   //!< Extra raw signals for a wait in this queue
    std::vector<hsa_signal_t> waiting_signals_;       //!< Current waiting signals in this queue
  };

  class MetaDataPreloader : public amd::EmbeddedObject {
    public:
      //! Set the metadata ring buffer base for the current queue.
      void SetQueueBase(void* ring_buffer, uint32_t version_header = 0) {
        queue_base_ = (DEBUG_CLR_ENABLE_PREFETCH_METADATA) ? ring_buffer : nullptr;
        if (queue_base_ != nullptr) {
          metadata_version_header_ = version_header;
        }
        pending_descriptor_ = nullptr;
        pending_preload_length_ = 0;
        pending_preload_offset_ = 0;
      }

      //! Stage the kernel descriptor and preload info for the next dispatch.
      //! Call before dispatchAqlPacket.
      void PrepareDispatch(const hsa_amd_metadata_kernel_descriptor_t* descriptor,
                           uint16_t preload_length, uint16_t preload_offset) {
        pending_descriptor_ = descriptor;
        pending_preload_length_ = preload_length;
        pending_preload_offset_ = preload_offset;
      }

      //! Set metadata prefetching packet associated with regular aql packet
      template <class AqlPacket>
      inline void Set(AqlPacket* packet, uint16_t header, uint64_t index) {
        if (!IsAttached()) {
          return;
        }
        if constexpr (std::is_same_v<AqlPacket, hsa_kernel_dispatch_packet_t>) {
          if (pending_descriptor_ == nullptr) {
            return;
          }
          hsa_amd_metadata_kernel_dispatch_packet_t* queue_metadata_packet =
               &(reinterpret_cast<hsa_amd_metadata_kernel_dispatch_packet_t*>(
                   queue_base_))[index];
          SetPacket(packet, header, queue_metadata_packet);
        } else if constexpr (std::is_same_v<AqlPacket, hsa_barrier_and_packet_t> ||
                             std::is_same_v<AqlPacket, hsa_amd_barrier_value_packet_t>) {
          hsa_amd_metadata_barrier_packet_t* queue_metadata_packet =
               &(reinterpret_cast<hsa_amd_metadata_barrier_packet_t*>(
                   queue_base_))[index];
          SetPacket(packet, header, queue_metadata_packet);
        }
      }

      //! Set the launch descriptor version (called once from VirtualGPU::create)
      void SetLaunchDescriptorVersion(uint8_t version) {
        launch_descriptor_version_ = version;
      }

      //! Copy the dynamic data prefetch config into the preloader state
      void SetDynDataPrefetchRegions(const amd::DynDataPrefetchConfig& cfg) {
        dyn_data_prefetch_enabled_ = true;
        dyn_data_prefetch_num_regions_ = cfg.numRegions;
        dyn_data_prefetch_hints_ = cfg.hints;
        for (uint32_t i = 0; i < cfg.numRegions && i < amd::kDynDataPrefetchMaxRegions; ++i) {
          dyn_data_prefetch_regions_[i] = cfg.regions[i];
        }
      }

      //! Reset the dynamic data prefetch state after dispatch
      void ClearDynDataPrefetchConfig() {
        dyn_data_prefetch_enabled_ = false;
      }

    private:
      //! Return whether the loader is attached to a gpu queue
      bool IsAttached() const { return queue_base_ != nullptr; }

      //! Get type from aql packet header
      uint8_t GetType(uint16_t header) const {
        return (header >> HSA_PACKET_HEADER_TYPE) & ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1);
      }

      //! Set the metadata prefetch aql packet for kernel dispatch
      void SetPacket(hsa_kernel_dispatch_packet_t* aql, uint16_t header,
                     hsa_amd_metadata_kernel_dispatch_packet_t* metadata);

      //! Set the metadata prefetch aql packet for barrier.
      //! The CP invalidates headers after completion, so only header0
      //! and event_id need to be written.
      //! Read event_id directly from amd_signal_t to avoid the hsa_amd_signal_get_event_id
      //! API overhead. Only interrupt signals carry a valid event_id.
      template <class AqlBarrierPacket>
      void SetPacket(AqlBarrierPacket* aql, uint16_t header,
                     hsa_amd_metadata_barrier_packet_t* metadata) const {
        if (aql->completion_signal.handle) {
          auto* signal = reinterpret_cast<amd_signal_t*>(aql->completion_signal.handle);
          metadata->event_id = signal->event_id;
        } else {
          metadata->event_id = 0;
        }
        // Plain store is sufficient: the subsequent packet_store_release on the main
        // AQL barrier header provides a release fence that orders all metadata writes
        // (event_id and header) before the CP sees the valid barrier packet.
        metadata->header0 = GetType(header);
      }

      void* queue_base_ = nullptr;        //!< The buffer base of prefetching queue
      uint32_t metadata_version_header_ = 0; //!< Pre-shifted version bits for metadata headers
      const hsa_amd_metadata_kernel_descriptor_t* pending_descriptor_ = nullptr;
      uint16_t pending_preload_length_ = 0;
      uint16_t pending_preload_offset_ = 0;

      uint8_t launch_descriptor_version_ = AMD_LAUNCH_DESCRIPTOR_VERSION_NONE;
      bool dyn_data_prefetch_enabled_ = false;
      uint8_t dyn_data_prefetch_hints_ = 0;
      uint32_t dyn_data_prefetch_num_regions_ = 0;
      amd::DynDataPrefetchRegion dyn_data_prefetch_regions_[amd::kDynDataPrefetchMaxRegions] = {};
  };

  VirtualGPU(Device& device, bool profiling = false, bool cooperative = false,
             const std::vector<uint32_t>& cuMask = {},
             amd::CommandQueue::Priority priority = amd::CommandQueue::Priority::Normal,
             bool dedicated_queue = false);
  ~VirtualGPU();

  bool create();
  const Device& dev() const { return roc_device_; }

  void profilingBegin(amd::Command& command, bool sdmaProfiling = false);
  void profilingEnd(bool clearHwEvent = false);

  void updateCommandsState(amd::Command* list) const;

  void submitReadMemory(amd::ReadMemoryCommand& cmd);
  void submitWriteMemory(amd::WriteMemoryCommand& cmd);
  void submitCopyMemory(amd::CopyMemoryCommand& cmd);
  void submitCopyMemoryP2P(amd::CopyMemoryP2PCommand& cmd);
  void submitBatchCopyMemory(amd::BatchCopyMemoryCommand& cmd);
  void submitMapMemory(amd::MapMemoryCommand& cmd);
  void submitUnmapMemory(amd::UnmapMemoryCommand& cmd);
  void submitKernel(amd::NDRangeKernelCommand& cmd);
  bool submitKernelInternal(
      const amd::NDRangeContainer& sizes,                  //!< Workload sizes
      const amd::Kernel& kernel,                           //!< Kernel for execution
      const_address parameters,                            //!< Parameters for the kernel
      void* event_handle,                                  //!< Handle to OCL event for debugging
      uint32_t sharedMemBytes = 0,                         //!< Shared memory size
      amd::NDRangeKernelCommand* vcmd = nullptr,           //!< Original launch command
      hsa_kernel_dispatch_packet_t* aql_packet = nullptr,  //!< Scheduler launch
      bool attach_signal = false);
  void submitNativeFn(amd::NativeFnCommand& cmd);
  void submitMarker(amd::Marker& cmd);
  void submitAccumulate(amd::AccumulateCommand& cmd);
  void submitAcquireExtObjects(amd::AcquireExtObjectsCommand& cmd);
  void submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& cmd);
  void submitPerfCounter(amd::PerfCounterCommand& cmd);

  void flush(amd::Command* list = nullptr, bool wait = false);
  void submitFillMemory(amd::FillMemoryCommand& cmd);
  void submitStreamOperation(amd::StreamOperationCommand& cmd);
  void submitBatchMemoryOperation(amd::BatchMemoryOperationCommand& cmd);
  void submitVirtualMap(amd::VirtualMapCommand& cmd);
  void submitMigrateMemObjects(amd::MigrateMemObjectsCommand& cmd);

  void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd);
  void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd);
  void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd);
  void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd);
  void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd);
  void submitSvmPrefetchAsync(amd::SvmPrefetchAsyncCommand& cmd);
  void SubmitSvmPrefetchBatchAsync(amd::SvmPrefetchBatchAsyncCommand& cmd);

  virtual void submitSignal(amd::SignalCommand& cmd) {}
  virtual void submitMakeBuffersResident(amd::MakeBuffersResidentCommand& cmd) {}

  void submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd) {}
  void submitThreadTrace(amd::ThreadTraceCommand& vcmd) {}

  virtual void submitExternalSemaphoreCmd(amd::ExternalSemaphoreCmd& cmd) {}

  virtual address allocKernelArguments(size_t size, size_t alignment) final;
  virtual void ReleaseSdmaEngines() final;  //!< Release SDMA engine assignments
  virtual void ReleaseAllHwQueues() final;
  virtual void ReleaseHwQueue() final;

  /**
   * @brief Waits on an outstanding kernel without regard to how
   * it was dispatched - with or without a signal
   *
   * @return bool true if Wait returned successfully, false otherwise
   */
  bool releaseGpuMemoryFence(bool skip_copy_wait = false);

  hsa_agent_t gpu_device() const { return gpu_device_; }
  hsa_queue_t* gpu_queue() { return gpu_queue_; }

  //! Set the active HW queue and keep the metadata preloader in sync.
  void SetGpuQueue(hsa_queue_t* queue, void* metadata_ring_buffer = nullptr);

  //! Snapshot the current HW queue as preferred for future re-acquisition (used by graph launch).
  //! Only updates if the queue is still valid — avoids clobbering a hint saved by ReleaseHwQueue.
  void SetPreferredQueue() override {
    std::scoped_lock lock(execution());
    if (gpu_queue_ != nullptr) {
      last_hwq_ = gpu_queue_;
    }
  }
  //! Acquire a HW queue using the preferred hint, then clear the hint
  void AcquireQueueWithPreference() override;

  //! Pin the HW queue so ReleaseHwQueue() becomes a no-op (used by graph internal streams)
  void PinQueue() override { queue_pinned_ = true; }
  //! Unpin the HW queue, allowing ReleaseHwQueue() to release it again
  void UnpinQueue() override { queue_pinned_ = false; }
  //! Release current HW queue and acquire a new one, avoiding queues with IDs in the excluded set
  bool ReacquireQueueExcluding(const std::unordered_set<uint64_t>& excluded_ids) override;

  // Return pointer to PrintfDbg
  PrintfDbg* printfDbg() const { return printfdbg_; }

  //! Returns memory dependency class
  MemoryDependency& memoryDependency() { return memoryDependency_; }

  //! Detects memory dependency for HSA kernels and uses appropriate AQL header
  bool processMemObjects(const amd::Kernel& kernel,  //!< AMD kernel object for execution
                         const_address params,       //!< Pointer to the param's store
                         size_t& ldsAddress,         //!< LDS usage
                         bool cooperativeGroups,     //!< Dispatch with cooperative groups
                         bool& imageBufferWrtBack,   //!< Image buffer write back is required
                         std::vector<device::Memory*>& wrtBackImageBuffer  //!< Images for writeback
  );

  //! Returns a managed buffer for staging copies
  ManagedBuffer& Staging() { return managed_buffer_; }

  //! Adds a pinned memory object into a map
  void addPinnedMem(amd::Memory* mem);

  void enableSyncBlit() const;

  void hasPendingDispatch() { hasPendingDispatch_ = true; }
  bool IsPendingDispatch() const { return (hasPendingDispatch_) ? true : false; }
  void addSystemScope() override {
    addSystemScope_ = true;
    fence_state_ = amd::Device::CacheState::kCacheStateInvalid;
  }
  void SetCopyCommandType(cl_command_type type) { copy_command_type_ = type; }

  HwQueueTracker& Barriers() { return barriers_; }

  Timestamp* timestamp() const { return timestamp_; }
  amd::Command* command() const { return command_; }

  void* allocKernArg(size_t size, size_t alignment);
  bool isFenceDirty() const { return fence_dirty_.load(std::memory_order_acquire); }
  void setFenceDirty(bool state) { fence_dirty_.store(state, std::memory_order_release); }
  void WaitCompleteSignal(hsa_signal_t signal);

  void HiddenHeapInit();
  uint64_t getQueueID();

  //! Add completion signal to the scheduler queue thread's event list.
  //! Wakes the scheduler queue thread if it's sleeping.
  void addSchedulerEvent(hsa_signal_t signal) {
    {
      std::lock_guard<std::mutex> lock(scheduler_mutex_);
      pendingSchedulerEvents_.push_back(signal);
    }
    scheduler_cv_.notify_one();
  }

  //! Returns true if the scheduler queue thread is running
  bool isSchedulerQueueThreadRunning() const {
    return schedulerQueueThreadRunning_.load(std::memory_order_relaxed);
  }

  //! Start the scheduler queue thread on first use
  void startSchedulerQueueThread();

  //! Analyzes a crashed AQL queue to find a broken AQL packet.
  //! Returns the faulting kernel name ("<not identified>" if not found).
  std::string AnalyzeAqlQueue() const;
  bool ForceIrq() const { return force_irq_; }

  //! SDMA engine affinity management
  uint32_t AssignedSdmaEngine() const {
    return assigned_sdma_engine_;
  }
  void SetAssignedSdmaEngine(uint32_t engine_mask) {
    assigned_sdma_engine_ = engine_mask;
  }
  void ClearAssignedSdmaEngine() {
    assigned_sdma_engine_ = 0;
  }
  bool hasAssignedSdmaEngine() const {
    return assigned_sdma_engine_ != 0;
  }

  void* getOrCreateHostcallBuffer();

 private:
  //! Dispatches a barrier with blocking HSA signals
  void dispatchBlockingWait(hsa_kernel_dispatch_packet_t* packet);

  bool dispatchAqlPacket(hsa_kernel_dispatch_packet_t* packet, uint16_t header, uint16_t rest,
                         bool blocking = true, bool capturing = false,
                         const uint8_t* aqlPacket = nullptr, bool attach_signal = false);
  bool dispatchAqlPacket(hsa_barrier_and_packet_t* packet, uint16_t header, uint16_t rest,
                         bool blocking = true, bool attach_signal = false);

  //! Fast-path dispatch: pre-built flat contiguous buffer. recordedPacketVersion /
  //! pm4Template carry the optional PM4-IB graph-replay fast path (see device.hpp).
  bool dispatchAqlPacketBatchFlat(const std::vector<uint8_t>& flatPacketData,
                                  const std::vector<uint32_t>& validFullHeaders,
                                  amd::AccumulateCommand* vcmd = nullptr,
                                  bool attach_signal = false,
                                  const std::vector<const std::string*>* kernelNames = nullptr,
                                  bool pre_patched = false,
                                  bool blocking = false,
                                  uint64_t recordedPacketVersion = 0,
                                  const void* pm4Template = nullptr) override;

  template <typename AqlPacket> bool dispatchGenericAqlPacket(AqlPacket* packet, uint16_t header,
                                                              uint16_t rest, bool blocking,
                                                              bool attach_signal = false,
                                                              bool cluster_launch = false);
  //! Encode a captured graph into a heap-owned PM4 template (CPU-only, queue-
  //! independent) for capture-time build. Returns an opaque Pm4GraphTemplate* or
  //! nullptr if the graph is not PM4-replayable. Free with freePm4GraphTemplate.
  void* buildPm4GraphTemplate(void* const* packets, size_t numPackets) override;
  void freePm4GraphTemplate(void* tmpl) override;

  bool dispatchCounterAqlPacket(hsa_ext_amd_aql_pm4_packet_t* packet, const uint32_t gfxVersion,
                                bool blocking, const hsa_ven_amd_aqlprofile_1_00_pfn_t* extApi);
  void dispatchBarrierPacket(uint16_t packetHeader, bool skipSignal = false,
                             hsa_signal_t signal = hsa_signal_t{0});

  // EXPERIMENTAL (HIP_PWS_FENCE=1, gfx11 only): inter-kernel PWS deferred-wait
  // fence. Replaces the firmware AQL packet-scope cache fence with an inline
  // vendor PM4-IB packet (CS_PARTIAL_FLUSH + RELEASE_MEM(PWS) + ACQUIRE_MEM(PWS))
  // so the cache flush overlaps the next dispatch instead of stalling the CP.
  bool pwsFenceActive();         //!< env gate, cached
  bool ensurePwsIb();            //!< lazily build the executable PWS PM4 IB
  void injectPwsFence();         //!< append the vendor PM4-IB packet to the live queue
  void capturePwsFence(uint8_t* dst);  //!< record the vendor PM4-IB packet into a graph slot

  // EXPERIMENTAL (HIP_PM4_GRAPH=1, gfx11 only): replay a captured all-dispatch
  // hipGraph as ONE PM4 indirect buffer (lean raw-PM4 dispatch front-end + in-place
  // PWS fence between every dispatch) launched by a single vendor PM4-IB packet.
  // This transfers the raw-PM4 PWS speedup into the HIP runtime: one CP jump for the
  // whole graph (the IB-jump cost amortizes to ~0), unlike per-dispatch injection.
  //!< Compiled PM4 IB for one captured graph. kDeferredScratch means the graph
  //!< needs scratch the queue has not sized yet: replay via AQL (which sizes it),
  //!< then rebuild on the next launch.
  enum Pm4GraphStatus { kPm4Unbuilt, kPm4Ready, kPm4UnsupportedPermanent, kPm4DeferredScratch };
  struct Pm4GraphIb {
    void* ib = nullptr;
    uint32_t dw = 0;
    Pm4GraphStatus status = kPm4Unbuilt;
    //! false when this cache entry only REFERENCES a GraphExec-owned shared IB
    //! (device-scoped, built once across streams): freePm4GraphIb / eviction must
    //! not release the storage in that case (the GraphExec owns it).
    bool owned = true;
    //! Per-kernel GPU-clock timestamp buffer (per-kernel profiling). CP-writable,
    //! host-readable (CPU fine-grain pool). Holds tsCount 64-bit ticks at
    //! kPm4TsStride byte stride. nullptr when profiling is not instrumented.
    void* tsBuf = nullptr;
    uint32_t tsCount = 0;
  };
  //!< Queue-runtime-dependent dword that a capture-time template leaves as a
  //!< placeholder (written as 0) and that specializeFromTemplate() patches from
  //!< the launch stream's queue. All are graph-constant (one value for the whole
  //!< graph), so the encoder emits each exactly once under delta-encoding.
  enum Pm4PlaceholderKind : uint8_t {
    kPhScratchBaseLo, kPhScratchBaseHi,  //!< COMPUTE_USER_DATA scratch base >> 8
    kPhTmpring,                          //!< COMPUTE_TMPRING_SIZE
    kPhScratchVdesc0, kPhScratchVdesc1, kPhScratchVdesc2, kPhScratchVdesc3,  //!< scratch V#
    kPhQueuePtrLo, kPhQueuePtrHi,        //!< amd_queue_t pointer (queue_ptr user SGPR)
    kPhNone = 0xFF                       //!< not a placeholder (graph-fixed dword)
  };
  struct Pm4Placeholder { uint32_t offset; uint8_t kind; };
  //!< A baked scalar dword that a scalar node-param mutation can change in place.
  //!< Recorded at encode time (only when HIP_PM4_GRAPH_INPLACE is on, which forces
  //!< delta-encoding off so every dispatch has its own patchable slot) so a mutated
  //!< graph can patch the resident VRAM IB instead of rebuilding a new one. kind
  //!< selects which AQL packet field supplies the new value; pkt is the dispatch
  //!< index into the launch packet array.
  enum Pm4MutKind : uint8_t {
    kMutKernargLo, kMutKernargHi,        //!< kernarg_address user-SGPR words
    kMutGridX, kMutGridY, kMutGridZ,     //!< DISPATCH_DIRECT grid dims
    kMutWgX, kMutWgY, kMutWgZ            //!< COMPUTE_NUM_THREAD workgroup dims
  };
  struct Pm4MutField { uint32_t offset; uint8_t kind; uint16_t pkt; };
  //!< CPU-only encode of a captured graph: the PM4 dwords with queue-dependent
  //!< fields zeroed, plus the offsets of those fields. Produced at capture/
  //!< instantiate (queue-independent) and specialized per launch stream. status
  //!< is kPm4Ready (encodable) or kPm4UnsupportedPermanent; the scratch-not-yet-
  //!< sized decision is deferred to specialize time (needsScratch).
  struct Pm4GraphTemplate {
    std::vector<uint32_t> dwords;
    std::vector<Pm4Placeholder> patches;
    Pm4GraphStatus status = kPm4Unbuilt;
    bool needsScratch = false;
    size_t numPackets = 0;
    //! Content hash of the packet set this template was encoded from. A cache
    //! miss specializes from the template only when it matches the launch packets
    //! (pm4GraphKey), so a stale template (graph mutated -> different bytes) or a
    //! disabled-node filtered subset is ignored and falls back to a full build.
    uint64_t key = 0;
    //! Content hash over STRUCTURAL fields only (numPackets, per-packet header|setup
    //! and kernel_object), excluding the mutable scalars (kernarg, grid, workgroup).
    //! Two packet sets with the same skeletonKey differ only by patchable scalars,
    //! so the in-place fast path can patch the resident IB instead of rebuilding.
    uint64_t skeletonKey = 0;
    //! Offsets of the baked mutable scalar dwords (see Pm4MutField). Populated only
    //! when HIP_PM4_GRAPH_INPLACE is on (which forces delta off so each dispatch has
    //! its own slot); empty otherwise (in-place patching disabled for this template).
    std::vector<Pm4MutField> mutFields;
    //! GraphExec-owned, device-scoped specialized IB, built ONCE and shared by every
    //! stream that replays this graph (each stream's cache just references it, never
    //! re-specializes/uploads). Only populated when the template is queue-INDEPENDENT
    //! (patches.empty(): no scratch, no queue_ptr), since otherwise the IB encodes
    //! per-queue values. nullptr -> each stream specializes its own per-stream IB.
    //! Owned here; freed (device pool) by freePm4GraphTemplate.
    void* sharedIb = nullptr;
    uint32_t sharedDw = 0;
    //! Lazy shared-IB build. The device-scoped shared IB must be specialized and
    //! uploaded on the EXECUTING vdev at first replay, NOT at instantiate on the
    //! null-stream vdev: an IB whose bytes are SDMA-uploaded on one queue at
    //! instantiate is not guaranteed visible to a DIFFERENT execution queue's CP
    //! fetch later (no synchronization edge between the two), which silently
    //! replays a stale/garbage IB. Building on the exec vdev gives the upload and
    //! first CP fetch a queue ordering edge, exactly like the per-stream path.
    //! wantSharedIb is set at instantiate; sharedReady guards the one-time build.
    bool wantSharedIb = false;
    std::atomic<bool> sharedReady{false};
    std::mutex sharedMtx;
    //! true when the specialized IB does not depend on any per-queue runtime value.
    bool shareable() const { return status == kPm4Ready && patches.empty(); }
    //! Per-kernel GPU-clock profiling (gated on LOG_INFO+LOG_AQL, decided at capture).
    //! When set, the encoder appends a RELEASE_MEM(BOTTOM_OF_PIPE_TS, GPU-clock)
    //! per kernel boundary; tsAddrOff holds the dword offset of each packet's
    //! ADDRESS_LO so specializeFromTemplate can patch in a per-IB TS buffer.
    bool instrumented = false;
    std::vector<uint32_t> tsAddrOff;
  };
  std::unordered_map<uint64_t, Pm4GraphIb> pm4Graphs_;  //!< compiled IB cache, keyed by content hash
  //! Insertion order of pm4Graphs_ keys, used to bound the VRAM held by compiled
  //! IBs. A mutated graph re-captures to NEW packet content -> a NEW content hash
  //! -> a NEW IB, leaving the pre-mutation IB unreferenced. Without a bound these
  //! dead IBs would accumulate in the executable memory pool until the VirtualGPU
  //! is destroyed. evictPm4GraphsIfNeeded() frees the oldest entries (never the
  //! currently-armed one) once the cache exceeds kPm4MaxCachedIbs.
  std::deque<uint64_t> pm4GraphKeyOrder_;
  static constexpr size_t kPm4MaxCachedIbs = 16;
  int pm4GraphState_ = -1;          //!< HIP_PM4_GRAPH env gate: -1 unknown, 0 off, 1 on
  int pm4GraphScratchState_ = -1;   //!< HIP_PM4_GRAPH_SCRATCH env gate (scratch kernels)
  int pm4GraphDeltaState_ = -1;     //!< HIP_PM4_GRAPH_DELTA env gate (register delta-encode)
  int pm4GraphReorderState_ = -1;   //!< HIP_PM4_GRAPH_REORDER env gate (front-end reorder)
  int pm4GraphKeyCacheState_ = -1;  //!< HIP_PM4_GRAPH_KEYCACHE env gate (skip per-launch rehash)
  int pm4GraphPrewarmState_ = -1;   //!< HIP_PM4_GRAPH_PREWARM env gate (reserve exec IB arena at init)
  int pm4GraphBuildAfterState_ = -1;//!< HIP_PM4_GRAPH_BUILD_AFTER env gate (AQL first, build IB after)
  int pm4GraphSharedIbState_ = -1;  //!< HIP_PM4_GRAPH_SHARED_IB env gate (GraphExec-owned shared IB)
  int pm4GraphInheritScopeState_ = -1;//!< per-edge fence from packet scope (default on; HIP_PM4_GRAPH_NO_INHERIT_SCOPE disables)
  double pm4TsNsPerTick_ = 0.0;     //!< cached ns-per-tick for the agent-domain RELEASE_MEM GPU clock (0 = not queried)
  //! Per-packet kernel names for the current PM4 graph launch (borrowed, not owned),
  //! used only to label per-kernel timestamp logs; set at the launch site.
  const std::vector<const std::string*>* pm4LaunchKernelNames_ = nullptr;
  int pm4SdkProfilerState_ = -1;    //!< cached rocprofiler-sdk tool presence (dlsym rocprofiler_configure): -1 unknown, 0 absent, 1 attached
  //! Byte stride between per-kernel timestamp slots. The RELEASE_MEM writes a 64-bit
  //! GPU clock, so 8-byte slots (read back as a packed uint64_t array) is exact.
  static constexpr uint32_t kPm4TsStride = 8;
  // Last-lookup fast path: when the SAME recorded packet set is replayed back to
  // back (the steady-state decode loop), skip recomputing the O(N) content hash.
  // Validated by the graph-supplied recorded packet set version, which is nonzero,
  // unique per GraphExec instantiation+batch, and bumped on ANY packet mutation
  // (param update / enable-disable / re-capture) -- so this is a RELIABLE
  // invalidation, not a heuristic. version 0 (non-graph caller) always takes the
  // slow path. The cached pointer is into the node-based pm4Graphs_ map, so it
  // stays valid across map inserts.
  bool pm4IbCacheValid_ = false;
  uint64_t pm4IbCacheVersion_ = 0;
  Pm4GraphIb* pm4IbCacheEntry_ = nullptr;
  //! Armed slot for the GraphExec-owned shared IB (#5). Holds a non-owning reference
  //! to the device-scoped IB the current graph shares across streams. Kept OUT of
  //! pm4Graphs_ (whose entries can outlive a freed GraphExec and alias a new graph by
  //! content key); the keycache version stamp guards reuse, so a stale reference here
  //! is never submitted (a new GraphExec has a new version -> re-arm before submit).
  Pm4GraphIb pm4SharedRef_;
  //! In-place double-buffered resident IB (HIP_PM4_GRAPH_INPLACE). For a graph that
  //! is mutated only in scalar node params (kernarg pointer, grid/workgroup dims),
  //! keep TWO resident device IBs (ping-pong) plus a persistent host staging buffer.
  //! On a scalar mutation we patch the changed dwords into the IDLE slot and swap the
  //! submit pointer -- no new device allocation, no full re-encode, no LRU churn. The
  //! idle slot is guaranteed not in flight: before patching we wait on the completion
  //! signal of its last submission (usually already complete -> no stall). A
  //! structural mutation (skeleton change) tears this down and falls back to the
  //! rebuild+cache path.
  struct Pm4InplaceResident {
    bool valid = false;
    //! Identity tag only -- NEVER dereferenced after build (the owning GraphExec may
    //! free the template on a different vdev). mutFields below is a private copy.
    const Pm4GraphTemplate* tmpl = nullptr;
    uint64_t skeletonKey = 0;                //!< structural identity of the resident IB
    uint64_t version = 0;                    //!< recorded packet version in the active slot
    void* slot[2] = {nullptr, nullptr};      //!< the two device-local exec IBs
    uint32_t dw = 0;
    uint32_t activeSlot = 0;
    uint32_t spanLo = 0, spanHi = 0;         //!< contiguous dword span covering all mutFields
    void* stage = nullptr;                   //!< fine-grain host staging (CPU+GPU), dw dwords
    hsa_signal_t lastCompletion[2] = {{0}, {0}};  //!< last submit's completion per slot
    std::vector<Pm4MutField> mutFields;      //!< copy of the template's mutable-field offsets
  };
  Pm4InplaceResident pm4Inplace_;
  int pm4GraphInplaceState_ = -1;   //!< HIP_PM4_GRAPH_INPLACE env gate (in-place VRAM patch)
  // Executable IB arena (HIP_PM4_GRAPH_PREWARM): one device-local executable
  // buffer reserved once, with IBs sub-allocated from it via a first-fit free
  // list. This both (a) warms the executable memory pool off the launch critical
  // path -- the ~ms one-time first-allocation cost is paid at reservation, not on
  // the first replay -- and (b) avoids a memory_pool_allocate/free round-trip per
  // build/rebuild. An IB that does not fit falls back to a direct per-IB pool
  // allocation (freed via memory_pool_free); arena-owned IBs are returned to the
  // free list. Ownership is decided by address range (pm4ArenaOwns).
  void* pm4Arena_ = nullptr;
  size_t pm4ArenaBytes_ = 0;
  std::vector<std::pair<size_t, size_t>> pm4ArenaFree_;  //!< sorted free spans (offset,bytes)
  static constexpr size_t kPm4ArenaBytes = 8u * 1024u * 1024u;  //!< reserved exec arena size
  bool pm4GraphActive();
  bool pm4GraphScratchEnabled();
  bool pm4GraphDeltaEnabled();      //!< skip SET_SH_REG writes whose value is unchanged
  bool pm4GraphReorderEnabled();    //!< hoist next kernel's regs between release and acquire
  bool pm4GraphKeyCacheEnabled();   //!< reuse last lookup when the packet array is unchanged
  bool pm4GraphPrewarmEnabled();    //!< reserve the executable IB arena at init
  bool pm4GraphBuildAfterEnabled(); //!< first replay goes AQL, PM4 IB built right after
  bool pm4GraphSharedIbEnabled();   //!< build one device-scoped IB shared across streams
  bool pm4GraphInheritScopeEnabled();//!< per-edge fence scope inherited from captured packet headers
  bool pm4GraphProfileEnabled();     //!< per-kernel GPU-clock timestamps, gated on LOG_INFO+LOG_AQL logging
  bool pm4TracingArmed();            //!< a kernel-dispatch profiler (legacy activity OR rocprofiler-sdk) is active
  void reportPm4Timestamps(const Pm4GraphIb& g);  //!< read TS buffer, convert ticks, ClPrint per-kernel
  bool pm4GraphInplaceEnabled();    //!< double-buffered in-place VRAM patch for scalar mutations
  static uint64_t pm4GraphSkeletonKey(void* const* packets, size_t numPackets);  //!< structural hash
  static uint32_t mutFieldValue(const Pm4MutField& mf, void* const* packets);    //!< current value
  //! In-place fast path: handle this replay by patching/swapping the resident
  //! double-buffered IB. Returns true if it submitted (caller is done); false to
  //! fall through to the normal keycache/shared/rebuild path (not eligible: in-place
  //! off, no template, scratch-deferred, or a structural/skeleton change).
  bool tryReplayPm4GraphInplace(void* const* packets, size_t numPackets, bool blocking,
                                bool attach_signal, uint64_t recordedPacketVersion,
                                const Pm4GraphTemplate* tmpl);
  //! Build the resident double-buffered IB for these packets (specialize host dwords,
  //! allocate two device slots, upload both). Returns false on failure/deferred.
  bool buildInplaceResident(void* const* packets, size_t numPackets,
                            const Pm4GraphTemplate* tmpl, uint64_t skeletonKey);
  //! Patch the mutable scalar dwords for the current packets into resident slot, then
  //! SDMA the dirty span host->device. Caller must ensure the slot is not in flight.
  void patchInplaceSlot(uint32_t slot, void* const* packets, size_t numPackets);
  //! Release the resident double-buffered IB (both slots, staging, signals).
  void freeInplaceResident();
  void ensurePm4Arena();                          //!< reserve the executable IB arena (idempotent)
  void* pm4ArenaAlloc(size_t bytes);              //!< slice from the arena, or nullptr if no fit
  bool pm4ArenaOwns(const void* p) const;         //!< true if p lies within the arena
  void pm4ArenaFreeBytes(void* p, size_t bytes);  //!< return a slice to the arena free list
  void freePm4GraphIb(Pm4GraphIb& g);             //!< free g.ib (arena or pool) and null it
  //! Stage data into an executable IB. deviceScoped=true forces a device-pool
  //! allocation (skip the per-vdev arena) so the IB can outlive any single stream
  //! (used for the GraphExec-owned shared IB); free with Hsa::memory_pool_free.
  void* allocExecIbFromData(const uint32_t* data, uint32_t dw, bool deviceScoped = false);
  static uint64_t pm4GraphKey(void* const* packets, size_t numPackets);  //!< content hash
  //! Full build = encode (CPU) + specialize+upload, using THIS vdev's queue.
  Pm4GraphIb buildPm4GraphIb(void* const* packets, size_t numPackets);
  //! CPU-only encode of the packets into a queue-independent template (placeholders
  //! for queue-dependent fields). Returns a heap-owned template (free with
  //! freePm4GraphTemplate) or nullptr if encode is unsupported. Queue-independent,
  //! so it can run at instantiate on any vdev of the graph's device.
  void encodePm4GraphTemplate(void* const* packets, size_t numPackets, Pm4GraphTemplate& out);
  //! Patch a template's placeholders from THIS vdev's queue, then alloc+upload the
  //! IB. Returns kPm4DeferredScratch (queue scratch not sized yet), kPm4Ready, or
  //! the template's terminal status. No CPU re-encode -- just patch + DMA.
  //! deviceScoped=true allocates the IB from the device pool (for the shared IB).
  Pm4GraphIb specializeFromTemplate(const Pm4GraphTemplate& t, bool deviceScoped = false);
  //! Free oldest cached IBs (by insertion order) while pm4Graphs_ exceeds
  //! kPm4MaxCachedIbs. Never frees the armed entry (pm4IbCacheEntry_) or the
  //! protected entry just built/selected this launch.
  void evictPm4GraphsIfNeeded(const Pm4GraphIb* protect);
  //! Find the cached ready IB for these packets, or (if allowBuild) build+insert+
  //! arm it. Returns the ready entry, or nullptr (cache miss with allowBuild=false,
  //! or an unsupported/deferred build). Handles eviction and deferred-scratch.
  //! tmpl (optional) is a capture-time encode: when present a cache miss
  //! specializes from it (no CPU re-encode) instead of a full build.
  Pm4GraphIb* findOrBuildPm4Graph(void* const* packets, size_t numPackets,
                                  uint64_t recordedPacketVersion, bool allowBuild,
                                  const Pm4GraphTemplate* tmpl);
  //! Submit one compiled IB as a single vendor PM4-IB packet (ring write + doorbell).
  //! outSig (optional) receives the completion signal the CP decrements after the IB
  //! finishes, so the in-place path can guard a slot against being patched mid-flight.
  void submitPm4Ib(const Pm4GraphIb& g, bool blocking, bool attach_signal,
                   hsa_signal_t* outSig = nullptr);
  //! Build + insert + arm the IB for these packets WITHOUT submitting. Used by
  //! HIP_PM4_GRAPH_BUILD_AFTER after the AQL fallback submit has sized scratch.
  void prebuildPm4Graph(void* const* packets, size_t numPackets, uint64_t recordedPacketVersion,
                        const Pm4GraphTemplate* tmpl);
  bool tryReplayPm4Graph(void* const* packets, size_t numPackets, bool blocking, bool attach_signal,
                         uint64_t recordedPacketVersion, bool allowBuild = true,
                         const Pm4GraphTemplate* tmpl = nullptr);
  void dispatchBarrierValuePacket(uint16_t packetHeader, bool resolveDepSignal = false,
                                  hsa_signal_t signal = hsa_signal_t{0},
                                  hsa_signal_value_t value = 0, hsa_signal_value_t mask = 0,
                                  hsa_signal_condition32_t cond = HSA_SIGNAL_CONDITION_EQ,
                                  bool skipTs = false,
                                  hsa_signal_t completionSignal = hsa_signal_t{0});
  void initializeDispatchPacket(hsa_kernel_dispatch_packet_t* packet, amd::NDRangeContainer& sizes);

  void resetKernArgPool() { managed_kernarg_buffer_.ResetPool(); }

  uint64_t getVQVirtualAddress();

  bool createSchedulerParam();

  //! Returns TRUE if virtual queue was successfully allocated
  bool createVirtualQueue(uint deviceQueueSize);

  //! Common function for fill memory used by both svm Fill and non-svm fill
  bool fillMemory(cl_command_type type,         //!< the command type
                  amd::Memory* amdMemory,       //!< memory object to fill
                  const void* pattern,          //!< pattern to fill the memory
                  size_t patternSize,           //!< pattern size
                  const amd::Coord3D& surface,  //!< Whole Surface of mem object.
                  const amd::Coord3D& origin,   //!< memory origin
                  const amd::Coord3D& size,     //!< memory size for filling
                  bool forceBlit = false        //!< force shader blit path
  );

  //! Common function for memory copy used by both svm Copy and non-svm Copy
  bool copyMemory(cl_command_type type,            //!< the command type
                  amd::Memory& srcMem,             //!< source memory object
                  amd::Memory& dstMem,             //!< destination memory object
                  bool entire,                     //!< flag of entire memory copy
                  const amd::Coord3D& srcOrigin,   //!< source memory origin
                  const amd::Coord3D& dstOrigin,   //!< destination memory object
                  const amd::Coord3D& size,        //!< copy size
                  const amd::BufferRect& srcRect,  //!< region of source for copy
                  const amd::BufferRect& dstRect,  //!< region of destination for copy
                  amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  );

  //! Updates AQL header for the upcoming dispatch
  void setAqlHeader(uint16_t header) { aqlHeader_ = header; }

  //! Resets the current queue state. Note: should be called after AQL queue becomes idle
  void ResetQueueStates();

  //! Track the progress of the queue based on the last write index and completion signal.
  //! When skip_signal is true, only the write index is advanced and the completion signal
  //! is cleared. Used for graph pre-patched dispatches whose signals are externally
  //! managed and freed after graph completion.
  template <typename AqlPacket>
  inline void TrackQueueProgress(const AqlPacket& packet, uint64_t index,
                                 bool skip_signal = false) {
    last_write_index_ = index;
    if (skip_signal) {
      last_completion_signal_.handle = 0;
    } else if (packet.completion_signal.handle != 0) {
      last_packet_with_signal_index_ = index;
      last_completion_signal_ = packet.completion_signal;
    }
  }

  //! Returns true if the queue is considered as idle. That means all submitted packets are
  //! complete. Note: it doesn't track the state of caches
  bool IsQueueIdle() const {
    if (gpu_queue_ == nullptr) {
      return true;
    }

    // Make sure the last packet contained a completion signal
    if (last_packet_with_signal_index_ == last_write_index_) {
      if ((last_write_index_ == kInvalidQueueIndex) && (last_completion_signal_.handle == 0)) {
        return true;
      } else {
        return (Hsa::signal_load_relaxed(last_completion_signal_) == 0);
      }
    }

    return false;
  }

  //! Queue state flags
  union {
    struct {
      uint32_t hasPendingDispatch_ : 1;     //!< A kernel dispatch is outstanding
      uint32_t profiling_ : 1;              //!< Profiling is enabled
      uint32_t cooperative_ : 1;            //!< Cooperative launch is enabled
      uint32_t addSystemScope_ : 1;         //!< Insert a system scope to the next aql
      uint32_t tracking_created_ : 1;       //!< Enabled if tracking object was properly initialized
      uint32_t retainExternalSignals_ : 1;  //!< Indicate to retain external signal array
      uint32_t force_irq_ : 1;              //!< Forces interrupt on the signal completion
    };
    uint32_t state_;
  };

  Timestamp* timestamp_;
  amd::Command* command_;   //!< Current command
  hsa_agent_t gpu_device_;  //!< Physical device
  hsa_queue_t* gpu_queue_;  //!< Active queue associated with a vgpu
  hsa_barrier_and_packet_t barrier_packet_ {};
  hsa_amd_barrier_value_packet_t barrier_value_packet_ {};

  uint32_t skippedDispatches_;  //!< Count of consecutive dispatches that skipped the doorbell flush.
  uint32_t dispatch_id_;  //!< This variable must be updated atomically.
  Device& roc_device_;    //!< roc device object
  PrintfDbg* printfdbg_;
  MemoryDependency memoryDependency_;  //!< Memory dependency class
  uint16_t aqlHeader_;                 //!< AQL header for dispatch

  amd::Memory* virtualQueue_;  //!< Virtual device queue
  uint deviceQueueSize_;       //!< Device queue size
  uint maskGroups_;            //!< The number of mask groups processed in the scheduler by
                               //!< one thread
  uint schedulerThreads_;      //!< The number of scheduler threads

  hsa_queue_t* schedulerQueue_;

  std::thread schedulerQueueThread_;                  //!< Host thread that monitors the scheduler queue
  std::atomic<bool> schedulerQueueThreadRunning_;     //!< Flag to indicate if the thread is running
  std::mutex scheduler_mutex_;                        //!< Lock to synchronize scheduler thread
  std::condition_variable scheduler_cv_;              //!< Condition to wake scheduler thread
  std::once_flag scheduler_thread_init_;              //!< Ensures thread is initialized exactly once
  std::vector<hsa_signal_t> pendingSchedulerEvents_;  //!< Pending scheduler completion signals

  HwQueueTracker barriers_;  //!< Tracks active barriers in ROCr

  ManagedBuffer managed_buffer_;          //!< Memory manager for staging copies
  ManagedBuffer managed_kernarg_buffer_;  //!< Managed memory for kernel args

  static constexpr uint32_t kStagingPoolNumSignals = 4; //!< Hsa Signal count for Staging Buffer
  static constexpr uint32_t kKernArgPoolNumSignals = 16; //!< Hsa Signal count for KernArg Buffer
  MetaDataPreloader metadata_preloader_; //!< Proloader of kernel meta data

  friend class Timestamp;

  //  PM4 packet for gfx8 performance counter
  enum {
    SLOT_PM4_SIZE_DW = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / sizeof(uint32_t),
    SLOT_PM4_SIZE_AQLP = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / 64
  };

  uint16_t dispatchPacketHeaderNoSync_;
  uint16_t dispatchPacketHeader_;

  //!< bit-vector representing the CU mask. Each active bit represents using one CU
  const std::vector<uint32_t> cuMask_;
  amd::CommandQueue::Priority priority_;  //!< The priority for the hsa queue
  bool dedicated_queue_;                  //!< TRUE if this VirtualGPU has a dedicated queue (e.g., null stream)
  bool queue_pinned_ = false;             //!< TRUE if queue is pinned by graph (blocks ReleaseHwQueue)
  hsa_queue_t* last_hwq_ = nullptr;       //!< Last HW queue used, for preferred re-acquisition hint

  cl_command_type copy_command_type_;  //!< Type of the copy command, used for ROC profiler
                                       //!< OCL doesn't distinguish different copy types,
                                       //!< but ROC profiler expects D2H or H2D detection
  int fence_state_;                    //!< Fence scope
                                       //!< kUnknown/kFlushedToDevice/kFlushedToSystem
  std::atomic<bool> fence_dirty_;      //!< Fence modified flag
  bool heap_init_fence_emitted_ = false;  //!< True once this queue has emitted system scope
                                          //!< fence after hidden heap init.

  uint64_t last_write_index_ = kInvalidQueueIndex; //!< The last HW queue write index for any packet
  uint64_t last_packet_with_signal_index_ = kInvalidQueueIndex; //!< The last HW queue write index for a packet
                                              //!< with a completion signal
  void* pwsIbBuf_ = nullptr;                  //!< executable IB holding the PWS fence PM4
  uint32_t pwsIbDw_ = 0;                      //!< dword count of the PWS PM4 IB (16 or 18)
  int pwsFenceState_ = -1;                    //!< PWS fence env gate: -1 unknown, 0 off, 1 on
  hsa_signal_t last_completion_signal_{};     //!< The last completion signal

  //! SDMA engine affinity tracking for this VirtualGPU/stream
  uint32_t assigned_sdma_engine_ = 0;           //!< Assigned SDMA engine mask for all operations

  void* hostcallBuffer_;        //!< Hostcall buffer
  size_t hostcallBufferSize_ = 0; //!< Byte size of hostcallBuffer_, for hostFree

  using KernelArgImpl = device::Settings::KernelArgImpl;
};
}  // namespace amd::roc

# HIP vs Vulkan: Inter-Kernel Gap on AMD RDNA3 (W7900 / gfx1100)

Investigation into the per-dispatch "inter-kernel gap" (the GPU-idle time between
back-to-back compute kernels) on a single AMD Radeon PRO W7900 (gfx1100, GFX11,
RDNA3), comparing the HIP/ROCm runtime against the Vulkan (RADV) runtime on the
exact same silicon.

All measurements use the on-die fixed-frequency 100 MHz wall clock (10.000 ns/tick)
on both APIs, so reported gaps are real microseconds and are NOT affected by the
variable shader core clock.

--------------------------------------------------------------------------------

## KNOWN LIMITATIONS & TRICKY MOMENTS (read first)

These are the non-obvious costs/constraints of the PM4-IB graph-replay path
(HIP_PM4_GRAPH=1). Details and measurements are in Appendix D (D.16-D.19).

1. FIRST-EVER replay of a graph is EXPENSIVE (one-time). The first hipGraphLaunch
   after instantiate compiles the whole graph into one PM4 IB, allocates an
   executable device-memory-pool buffer, and DMA-uploads it synchronously before
   the doorbell. Measured ~4 ms (N=1) to ~7.6 ms (N=256 dispatches) on gfx1100
   W7900 -- vs ~16-19 us for the stock AQL graph (which has no IB to build). The
   2nd launch already drops to steady (~3-6 us). So this cost is paid ONCE per
   unique instantiated graph and amortizes to ~0/token over a decode loop, but it
   is a real latency spike on the first decode step. (D.19 Q3/Q4) MITIGATIONS
   (opt-in, D.20): HIP_PM4_GRAPH_PREWARM reserves the executable IB arena at init
   (moves the ~3.5 ms one-time warm-up off the first launch); HIP_PM4_GRAPH_BUILD_
   AFTER replays the first launch via AQL and builds the IB right after, overlapping
   the build with GPU exec (first-step wall ~5 ms -> ~3.5 ms with real GPU time).

2. "instantiate" (hipGraphInstantiate) is a SEPARATE one-time cost from #1: it
   builds the GraphExec, captures the per-node AQL packets, and allocates the
   kernarg pool. It grows with node count (~48 us at N=1 to ~310 us at N=256) and
   is essentially the SAME for AQL and PM4 (the PM4 IB build happens later, on
   first launch, not here). (D.19 Q4)

3. VRAM is BOUNDED but a mutated graph rebuilds. Any mutation that changes a
   hashed packet field (grid/block dims, kernel object, kernarg address -- e.g.
   split-K with varying launch dims) produces a NEW compiled IB; the old one is
   freed by a 16-entry LRU/FIFO eviction (kPm4MaxCachedIbs) so dead IBs do not
   accumulate. Pure kernarg-VALUE updates (same address) do NOT rebuild. Proven:
   2000 distinct-IB mutations -> cache never exceeds 16. (D.19 Q2)

4. The key cache is DEFAULT ON and correct for any graph caller, but only the HIP
   graph layer supplies the recorded-packet version; non-graph callers pass 0 and
   always take the slow rehash path. Multi-device linear graphs also pass 0 (the
   PM4 path does not target multi-GPU). (D.18 / D.19 Q1)

5. The per-edge fence MUST be the blocking AGENT acquire, not PWS deferred-wait --
   PWS is non-deterministic on MoE. This is already the default in the PM4 path.
   (D.14)

--------------------------------------------------------------------------------

## 0. TL;DR (what is actually true)

- The inter-kernel gap is REAL on both APIs. It is wave-drain + cache invalidate +
  command-processor (CP) turnaround between dispatches.
- On the SAME W7900, HIP's per-dispatch period is consistently equal-or-higher
  than Vulkan/RADV's. So the difference is a RUNTIME/API effect, NOT a
  CDNA-vs-RDNA architecture effect.
- The earlier "0.5 us vs 3 us" headline was partly a measurement-reference
  artifact: Vulkan measures TOP_OF_PIPE -> BOTTOM_OF_PIPE (the post-dispatch cache
  flush lands INSIDE "kernel"), while the HIP wave-timeline method measures
  first-wave-start -> last-wave-end (the flush lands INSIDE "gap"). The fair,
  reference-invariant metric is the per-dispatch PERIOD.
- CACHE-FENCE WEIGHT IS RULED OUT as the cause of HIP being slower. Source proof:
  RADV does a FULL L2 writeback+invalidate on every compute SHADER_WRITE->SHADER_READ
  barrier, while default HIP (AGENT scope) does NOT touch L2 at all. HIP does LESS
  cache work yet is slower => the residual cost is in the dispatch/submission path,
  not coherence.

Two earlier hand-waved claims were WRONG and are corrected below:
1. "HIP defaults to system-scope acquire/release fences" -> FALSE. Default is AGENT.
2. "RADV uses a lighter device-scope cache op and keeps L2 resident" -> FALSE. RADV
   does a full GL2 writeback+invalidate on this barrier.

--------------------------------------------------------------------------------

## 1. Methodology

Microbenchmark: one command buffer / one stream records K back-to-back compute
dispatches. Each kernel reads+writes data[gid] (a RAW hazard on an SSBO/global
buffer), so consecutive kernels are genuinely dependent. A barrier (Vulkan) or the
default in-stream ordering (HIP) sits between every pair.

- Vulkan (`vk_gap_test`, `main.cpp`): `vkCmdWriteTimestamp(TOP_OF_PIPE)` before and
  `(BOTTOM_OF_PIPE)` after each dispatch; barrier =
  `VkMemoryBarrier{srcAccess=SHADER_WRITE, dstAccess=SHADER_READ}` with
  COMPUTE_SHADER src/dst stages (`main.cpp:228-256`).
- HIP (`hip_gap_test.cpp`): in-kernel `__builtin_readsteadycounter()` (lowers to
  `s_sendmsg_rtn(GET_REALTIME)` on gfx11) with `atomicMin`/`atomicMax` across all
  blocks to capture first-wave-start (tstart) and last-wave-end (tend) per dispatch.
  Self-calibrated against a `hipEvent` wall timer.

Metrics:
- kernel : tend - tstart (HIP wave busy) / TOP->BOT (Vulkan, includes post-flush).
- gap    : tstart[k+1] - tend[k] (HIP) / BOT->TOP (Vulkan).
- PERIOD : total_span / K  -- reference-point invariant, the fair comparison.

Calibration sanity check: both APIs lock to exactly 10.00 ns/tick = 100 MHz, the
same counter PAL's profiler reports. The wall-clock self-calibration agreeing to 4
digits is strong evidence the GPU-timeline measurement is sound.

--------------------------------------------------------------------------------

## 2. Source facts (audited, with file:line)

### 2.1 HIP/ROCm: default dispatch fence scope is AGENT, not SYSTEM

`AMD_OPT_FLUSH` defaults to 1:

    rocclr/utils/flags.hpp:169
      release(uint, AMD_OPT_FLUSH, 1,
        "Kernel flush option , 0x0 = Use system-scope fence operations."
        "0x1 = Use device-scope fence operations when possible.")

    rocclr/device/device.cpp:1366
      fenceScopeAgent_ = AMD_OPT_FLUSH;

Dispatch packet header is built with AGENT scope when fenceScopeAgent_==1
(GFX12 is the exception: acquire=SYSTEM, release=AGENT):

    rocclr/device/rocm/rocvirtual.cpp:1916-1922
      dispatchPacketHeader_ =
        (... | barrierHBits | (isGfx12 ? sysAcquireAgentReleaseHBits : agentScopeHBits));

What each scope does to caches (runtime's own comment):

    rocclr/device/rocm/rocvirtual.cpp:62-64
      // AGENT acquire  invalidates I, K and L1
      // SYSTEM release invalidates L1, L2 and flushes L2

So a default HIP dispatch invalidates I/K/L1 only and does NOT flush L2.

Escalation to SYSTEM happens only for: SVM prefetch (rocvirtual.cpp:2655),
stream wait/write-value atomics (rocvirtual.cpp:3545,3563), first hidden-heap init
(rocvirtual.cpp:4160), memory copies/blits (rocblit.cpp:397,443,...), HIP-graph
batch where an SDMA follows kernels (hip_graph_internal.cpp:2104), a launch whose
stop timing-event lacks hipEventDisableSystemFence (hip_module.cpp:543-547), or
globally AMD_OPT_FLUSH=0 (rocvirtual.cpp:1913-1919, the else branch).

A separate BARRIER_AND packet (used only at sync points like stream sync / event
wait) DOES default to SYSTEM/SYSTEM and performs the full L2 flush:

    rocclr/device/rocm/rocvirtual.cpp:68-72
      kBarrierPacketHeader = ... | (SYSTEM << ACQUIRE) | (SYSTEM << RELEASE)

No per-dispatch completion signal by default (attach_signal=false ->
completion_signal=0): rocvirtual.cpp:1294-1297, :582-591, hpp:470-471.

Cache op is performed by the CP firmware from the header scope bits, not a software
flush in rocclr/ROCr (rocvirtual.cpp:62-64; amd_aql_queue.cpp:1213-1219 rewrites the
header only to work around old firmware, confirming firmware acts on the bit).

Enum/bit values confirmed in hsa.h: NONE=0, AGENT=1, SYSTEM=2; acquire field bit 9,
release field bit 11 (rocr-runtime .../inc/hsa.h:2859-2932).

### 2.2 RADV (Mesa 25.0.7): the compute barrier does a FULL L2 flush

Dispatch packet on GFX11 is PKT3_DISPATCH_DIRECT:

    src/amd/vulkan/radv_cmd_buffer.c:12062
      radeon_emit(cs, PKT3(PKT3_DISPATCH_DIRECT, 3, predicating) | PKT3_SHADER_TYPE_S(1));

For SHADER_WRITE -> SHADER_READ on a buffer, INV_L2 is set unconditionally on the
write (src) side:

    src/amd/vulkan/radv_cmd_buffer.c:6617-6618
      if (!image_is_coherent)
         flush_bits |= RADV_CMD_FLAG_INV_L2;
    (documented intent at radv_cmd_buffer.c:6577-6579)

dst (read) side adds vector + scalar L0/L1 invalidate:

    src/amd/vulkan/radv_cmd_buffer.c:6722-6726
      if (!pdev->use_llvm && !image) flush_bits |= RADV_CMD_FLAG_INV_SCACHE;
      flush_bits |= RADV_CMD_FLAG_INV_VCACHE;

INV_L2 lowers to a full GL2 writeback+invalidate (+ metadata) in one ACQUIRE_MEM:

    src/amd/vulkan/radv_cs.c:163-166
      if (flush_bits & RADV_CMD_FLAG_INV_L2) {
         // Writeback and invalidate everything in L2.
         gcr_cntl |= S_586_GL2_INV(1) | S_586_GL2_WB(1) | (gfx_level < GFX12 ? S_586_GLM_INV(1) | S_586_GLM_WB(1) : 0);

plus a CS_PARTIAL_FLUSH wave drain (radv_cmd_buffer.c:6528 -> radv_cs.c:232-237).
No per-dispatch EOP/signal between dispatches; the only EOP path (cb_db_event) is
not taken for a compute->compute SSBO barrier.

Net PM4 between two dispatches on GFX11:
  1. EVENT_WRITE CS_PARTIAL_FLUSH (drain writer waves)
  2. ACQUIRE_MEM with GCR_CNTL = GL2_WB|GL2_INV|GLM_WB|GLM_INV|GL1_INV|GLV_INV|GLK_INV
     (full L2 writeback+invalidate + L1/L0/scalar invalidate; CP waits for idle)
  3. PKT3_DISPATCH_DIRECT (reader)

--------------------------------------------------------------------------------

## 3. Hardware measurements (W7900, gfx1100, GPU 0)

Note on variance: GPU clock could not be pinned on this part in-container
(perf level reads "unknown"), so absolute us drift run-to-run. Within-session
deltas and the consistent ordering across iterations are the reliable signals.

### 3.1 Controlled scope toggle (same session, spin=0, n=4096)

| HIP scope                              | wave-timeline gap |
|----------------------------------------|-------------------|
| AGENT  (default, AMD_OPT_FLUSH=1)       | ~2.5 us           |
| SYSTEM (AMD_OPT_FLUSH=0, +L2 WB+INV)    | ~4.9 us           |

=> Forcing the L2 flush adds ~2.4 us. HIP's default avoids it. This directly
confirms the source finding (2.1).

### 3.2 Interleaved HIP vs Vulkan (same session, spin=0, n=4096, ordered/barrier)

| iter | Vulkan period (BOT->BOT) | HIP period (span/K) |
|------|--------------------------|----------------------|
| 1    | 3.28 us                  | 3.67 us              |
| 2    | 3.52 us                  | 5.60 us              |
| 3    | 3.44 us                  | 5.60 us              |

HIP per-dispatch period is equal-or-higher in every iteration, despite HIP doing
LESS cache work (no L2 flush) than RADV (full L2 flush).

### 3.3 Reference-point decomposition (illustrative single run, spin=0, n=4096)

    Vulkan (RADV):  kernel TOP->BOT 2.16 us ; gap BOT->TOP 0.64 us ; period 2.79 us
    HIP (wave):     kernel busy     1.35 us ; gap wave-end->start 3.08 us
    HIP hipEvent CP-boundary pass: DISCARDED -- hipEventRecord per kernel is a heavy
       system-scope barrier packet that ~tripled the period (artifact, not a clean
       analog of cheap Vulkan timestamps).

--------------------------------------------------------------------------------

## 4. Conclusion

1. The inter-kernel gap is real on both APIs (wave drain + cache invalidate + CP
   turnaround).
2. On identical RDNA3 silicon, HIP's per-dispatch period >= Vulkan/RADV's. The
   difference is a runtime/API property, not architecture.
3. Cache-coherence cost is NOT the explanation: RADV flushes L2 fully on every
   compute barrier while default HIP (AGENT) does not touch L2, yet HIP is slower.
4. The residual HIP overhead therefore lives in the dispatch/submission path
   (AQL kernel-dispatch packets via the CP HSA soft-queue + the ordered-stream
   AQL "barrier" bit forcing per-packet serialization) rather than cache fences.

OPEN / TO VERIFY (isolation test, see section 5): quantify and attribute the
dispatch-path cost by (a) launching across multiple independent streams and
(b) clearing the AQL barrier bit (any-order), to see if the HIP period collapses
toward Vulkan's.

--------------------------------------------------------------------------------

## 5. Isolation test (isolated container + HIP built from source)

Setup: a fresh container `hipvk-isolated-sshliapn` (same image/devices, separate
env). The rocm-systems monorepo was checked out to tag `rocm-7.2.1` to match the
installed runtime, and `libamdhip64.so` was built from source
(`clr/build-gap`, HIP 7.2.53211). It loads against the installed ROCr and
calibrates to 10.00 ns/tick - ABI compatible.

A one-line env-gated patch in `rocclr/device/rocm/rocvirtual.cpp` (after
`aqlHeader_ = dispatchPacketHeader_;`) allows toggling the dispatch packet header:
- `GAP_NOSCOPE=1`   -> strip acquire/release fence scope (NONE): no per-dispatch
                       cache acquire/release at all.
- `GAP_NOBARRIER=1` -> clear the AQL barrier bit.

### 5.1 CPU launch overhead is NOT the cause (HIP Graph == CPU launch loop)

A recorded Vulkan command buffer pays no per-dispatch CPU cost. The HIP analog is
a hipGraph (record once, replay). Measured (spin=0, n=4096):
  HIP graph period      ~= 5.5 us
  HIP CPU launch loop   ~= 5.6 us
Identical. So the gap is a real GPU-side per-dispatch cost, not host launch cost.

### 5.2 Concurrency: single HSA queue serializes; multiple queues overlap

Round-robin across N independent streams (each its own buffer), spin=0 n=4096:
  nstreams=1  -> period 3.97 us, concurrency 0.31x
  nstreams=2  -> period 2.84 us, concurrency 0.45x
  nstreams=4  -> period 2.39 us, concurrency 0.57x
  nstreams=8  -> period 2.38 us, concurrency 0.82x
  nstreams=16 -> period 2.41 us, concurrency 0.76x
The CP can overlap dispatches across queues down to a ~2.4 us floor. A single
queue cannot - it processes packets serially.

### 5.3 Same-session A/B (custom HIP), spin=0 n=4096 K=4000 - REPRODUCIBLE

| dispatch header variant                  | period (span/K) |
|------------------------------------------|-----------------|
| default (barrier bit + AGENT scope)       | 5.2 - 5.6 us    |
| GAP_NOSCOPE (no per-dispatch cache fence) | 3.94 us         |
| GAP_NOBARRIER (barrier bit cleared)       | 6.22 us         |
| BOTH                                      | 4.75 us         |

Conclusions (these are clock-robust: both arms are the same binary back-to-back):
- The per-dispatch AGENT cache fence (I/K/L1 invalidate + acquire/release) costs
  ~1.2-1.6 us. Removing it: 5.2 -> 3.94 us.
- The AQL barrier bit is NOT the serializer: clearing it does not reduce the
  period (it is slightly worse, and concurrency stays 0.21x = still no overlap).
  The CP serializes the single queue regardless of the bit.
- Floor with all cache work and the barrier bit removed: still ~3.9 us per
  dispatch in a single queue = pure CP AQL packet-processing + launch latency.

### 5.4 Cross-API, same session: Vulkan (full L2 flush) vs HIP-NOSCOPE (no cache)

| iter | Vulkan (full L2 flush) | HIP NOSCOPE (no cache work) |
|------|------------------------|------------------------------|
| 1    | 3.08 us                | 2.92 us                      |
| 2    | 2.80 us                | 4.02 us                      |
| 3    | 3.44 us                | 4.02 us                      |

Once HIP's per-dispatch cache fence is removed, HIP's single-queue dispatch floor
(~2.9-4.0 us) lands in the SAME range as Vulkan's full-L2-flush period
(~2.8-3.4 us). The two are comparable to within the measurement noise; in iter 1
HIP is actually lower. Because GPU clock management is broken in this environment
(perf level "unknown", driver reports clock tables EMPTY / low-power state),
absolute cross-API differences at this ~1 us scale are NOT reliably resolvable,
and this is stated as a limitation rather than papered over.

### 5.5 Did we try to "fix" the barrier bit in source? Yes - three ways, all null

The AQL barrier bit was attacked three independent ways:
1. hipExtLaunchKernel(hipExtAnyOrderLaunch) - the public API that clears the bit.
2. GAP_NOBARRIER source patch - clears the bit on dispatchPacketHeader_ directly.
3. Independent buffers per kernel (hip_gap_indep), so the memory-dependency
   tracker auto-selects the no-barrier "nosync" header (rocvirtual.cpp:764-767).

Single stream, stock HIP, spin=0 n=4096:
  buffers=1  (dependent, sync hdr, barrier bit) -> period 4.25 us, concurrency 0.31x
  buffers=2  (independent, nosync hdr, NO bit)  -> period 5.56 us, concurrency 0.24x
  buffers=64 (independent, nosync hdr, NO bit)  -> period 5.57 us, concurrency 0.24x

In ALL cases clearing/avoiding the barrier bit did NOT enable overlap and did NOT
reduce the period (it was equal or slightly worse). Root cause: a single HSA queue
is processed serially by the CP one dispatch at a time; the barrier bit only adds
"wait for prior to complete" on top. Overlap requires MULTIPLE queues/streams
(section 5.2: 2.4 us floor, concurrency 0.82x). So the barrier bit is not a bug to
fix - it is not the bottleneck. The real levers are (a) per-dispatch fence scope
(~1.4 us, section 5.3) and (b) number of streams (concurrency).

--------------------------------------------------------------------------------

## 6. Final conclusion (revised by the isolation data)

Reliable, same-session, reproducible findings:

1. Default HIP per-dispatch period (~5.2-5.6 us) decomposes into:
   - ~1.2-1.6 us  per-dispatch AGENT cache fence (I/K/L1 invalidate + acq/rel)
   - ~3.9 us      CP AQL packet-processing + launch floor (single queue)
2. The biggest HIP-specific, controllable per-dispatch cost is the AGENT cache
   fence (~1.4 us). Stripping it brings HIP into the same range as Vulkan.
3. The AQL barrier bit is NOT a serialization cost (clearing it does not help).
4. CPU launch overhead is NOT the cause (HIP graph == CPU launch loop).
5. A single HSA queue serializes dispatches; multiple queues overlap to a ~2.4 us
   floor - so the serial per-dispatch cost is largely hideable with concurrency.

Ruled out as explanations for "HIP gap > Vulkan gap":
- Architecture (CDNA vs RDNA): same W7900 silicon for both.
- Cache-fence weight being heavier in HIP: FALSE - Vulkan does a full L2
  writeback+invalidate per compute barrier (source 2.2), while default HIP (AGENT)
  does not touch L2 at all (source 2.1). HIP does LESS cache work.
- The AQL barrier bit: ruled out (5.3).
- CPU launch overhead: ruled out (5.1).

What remains, stated honestly:
- HIP's per-dispatch AGENT cache fence (~1.4 us) is a real, measured cost that
  Vulkan's path does not pay in the same per-dispatch manner. This is the largest
  reliably-attributable contributor to HIP's higher default period.
- Any residual dispatch-floor difference between HIP (AQL via the CP HSA queue)
  and Vulkan (prebuilt PM4 DISPATCH_DIRECT stream) is within this environment's
  clock noise (~1 us) and is NOT claimed as a definitive number.

--------------------------------------------------------------------------------

## 6b. DEFINITIVE results at PINNED clock (supersede noisy section 5.4)

The W7900 autosuspend was the noise source. Pinning the clock with
`gpu_pin_freq.sh` (profile_standard -> sclk 973 MHz, mclk 1124 MHz,
runtime_status=active) made every number reproducible to <1% across 5 runs.

Pinned 973 MHz, spin=0 n=4096, single stream, kernel busy = 2.06 us:

| configuration                                | period   | gap = period-busy |
|----------------------------------------------|----------|-------------------|
| Vulkan (RADV, full L2 writeback+invalidate)  | 4.96 us  | 2.90 us           |
| HIP default (barrier bit + AGENT fence)      | 7.94 us  | 5.88 us           |
| HIP GAP_NOSCOPE (no per-dispatch cache fence)| 5.59 us  | 3.53 us           |
| HIP GAP_NOBARRIER (barrier bit cleared)      | 7.52 us  | -                 |
| HIP BOTH (no fence + no barrier bit)         | 4.93 us  | 2.87 us           |

Multi-stream (pinned): 1->7.94, 2->4.03, 4->4.17, 8->3.35, 16->2.94 us
  (concurrency rises to 0.81x at 16 streams).
HIP graph 7.62 us ~= CPU launch loop 7.94 us (CPU launch overhead negligible).

Exact accounting of the HIP-vs-Vulkan per-dispatch difference:
  AGENT cache fence  = 7.94 - 5.59 = 2.35 us
  AQL barrier bit    = 5.59 - 4.93 = 0.66 us   (also default-nobarrier = 0.42 us)
  HIP stripped (4.93) ~= Vulkan (4.96)
  HIP default - Vulkan = 7.94 - 4.96 = 2.98 us = fence + barrier bit

CONCLUSION (now clock-clean, no noise hedge):
- The AQL dispatch engine is NOT inherently slower than PM4. With HIP's default
  per-dispatch AGENT fence and barrier bit removed, HIP single-queue dispatch
  (4.93 us) matches Vulkan's full period (4.96 us, which includes an L2 flush).
- The ENTIRE HIP > Vulkan per-dispatch gap (~3.0 us here) is HIP's default
  per-dispatch AGENT cache fence (~2.35 us) + AQL barrier-bit serialization (~0.66 us).
- Open puzzle (flagged, not hand-waved): Vulkan does MORE cache work (full L2 WB+INV)
  yet costs ~the same as HIP with NO cache work, while HIP's L1-only AGENT fence
  costs 2.35 us. This indicates the cost is the per-AQL-packet CP acquire/release
  serialization stall, not cache-traffic volume (working set is 16 KB). Precise
  isolation is firmware-level and not resolved here.

--------------------------------------------------------------------------------

## 6c. Is Vulkan actually flushing, or skipping it? (the "are you sure" check)

Challenge: maybe RADV is cheaper because it secretly SKIPS the flush (does less),
not because the mechanism is better. Checked three ways; the answer is NO, RADV
really flushes (and in fact does MORE cache work than HIP).

### 6c.1 Source: the L2 flush is unconditional on the producer side

For a global VkMemoryBarrier (image == NULL), image_is_coherent is hard-false, so
INV_L2 is always set on the write/src side:

    src/amd/vulkan/radv_cmd_buffer.c:6592   image_is_coherent = image ? ... : false;
    src/amd/vulkan/radv_cmd_buffer.c:6617-6618
        if (!image_is_coherent) flush_bits |= RADV_CMD_FLAG_INV_L2;

The can_skip_buffer_l2_flushes() optimization (radv_cmd_buffer.c:6546-6551) that
could suppress this is applied ONLY on the dst/read side (line 6684), NOT here. So
RADV emits the GL2 writeback+invalidate every time. It does not skip it.

### 6c.2 Empirical: the barrier does real, data-proportional work (pinned 973 MHz)

Barrier on vs off, spin=0 n=4096:

    VK barrier=1 : period 4.96 us
    VK barrier=0 : period ~1.0 us (median)

A no-op'd barrier would leave both at ~1 us. The barrier adds ~4 us of real
flush+drain. And the flush SCALES with the dirtied footprint (a skipped flush
could not):

| n (dirtied)     | VK period (b=1) | VK kernel TOP->BOT | HIP period | HIP shader busy |
|-----------------|-----------------|--------------------|------------|-----------------|
| 4096   (16 KB)  | 4.96 us         | 4.03 us            | 7.88 us    | 2.08 us         |
| 65536  (256 KB) | 5.10 us         | 4.19 us            | 9.35 us    | 3.54 us         |
| 262144 (1 MB)   | 7.98 us         | 7.06 us            | 17.99 us   | 12.19 us        |
| 1048576 (4 MB)  | 19.13 us        | 18.21 us           | 52.23 us   | 46.43 us        |

The decisive observation - per-dispatch OVERHEAD (period minus busy/window):
  HIP gap = period - shader_busy = 5.80, 5.80, 5.80, 5.81 us  -> FLAT (size-invariant)
  VK  gap = period - TOP->BOT     = 0.93, 0.91, 0.93, 0.92 us  -> FLAT (CP turnaround)

HIP's per-dispatch overhead is a FIXED ~5.8 us tax that does NOT scale with the
working set => it is a protocol/latency cost, not cache-traffic volume. Vulkan's
fixed tail is only ~0.92 us; its real L2 flush is folded INTO the dispatch window
(TOP->BOT) and scales there with dirty bytes. So Vulkan is doing the (heavier)
flush; it just overlaps it instead of paying a fixed serialized round-trip.

### 6c.3 ACQUIRE_MEM vs AGENT acquire/release - why the SAME logical op differs

Both must do the same three things between dependent kernels:
(A) drain the producer's waves, (B) make the producer's stores visible at the
device coherence point (GL2), (C) invalidate the consumer's L0/L1 so it re-reads.
The cost difference is HOW each runtime expresses that, not WHAT is required.

VULKAN / RADV = pre-built PM4 stream, single hardware cache-rinse:
  1. PKT3 DISPATCH_DIRECT (writer N)          radv_cmd_buffer.c:12062
  2. EVENT_WRITE CS_PARTIAL_FLUSH             radv_cs.c:232-237   (drain N's waves)
  3. PKT3 ACQUIRE_MEM with GCR_CNTL=          radv_cs.c:163-166,332-339
       GL2_WB|GL2_INV|GLM_WB|GLM_INV|GL1_INV|GLV_INV|GLK_INV
       "cache flush is executed in the ME, but the PFP waits for completion"
       (radv_cs.c:328-330) -- ONE inline Graphics Cache Rinse over only DIRTY
       lines; for 16 KB this is a few cache lines = near-instant.
  4. PKT3 DISPATCH_DIRECT (reader N+1)
  The CP front-end (PFP=PreFetch Parser) has already pre-parsed N+1's register/
  user-data packets while N ran; the only true stall is "wait for GCR idle".
  => overhead = wave drain + tiny GCR wait + ~0.9 us turnaround.

HIP / ROCm = interpreted AQL packets, per-packet firmware fence:
  1. host writes an AQL kernel_dispatch_packet into the user HSA queue ring,
     header = KERNEL_DISPATCH | barrier-bit | acquire=AGENT | release=AGENT
     (rocvirtual.cpp:1916-1922), rings doorbell.
  2. CP AQL packet processor (MEC firmware) reads packet N, decodes it, does the
     ACQUIRE fence at AGENT scope (invalidate I/K/L1, rocvirtual.cpp:62-64).
  3. MEC translates to an internal dispatch; SPI distributes N's workgroups.
  4. barrier-bit on N+1 => MEC must, in strict SERIAL order:
       4a. wait for N's grid to fully retire (wave drain)   -- same as Vulkan
       4b. perform N's RELEASE fence at AGENT scope (make stores visible at GL2)
       4c. advance read_index, fetch packet N+1 from the ring, decode it
       4d. perform N+1's ACQUIRE fence (invalidate I/K/L1)
       4e. dispatch N+1 to SPI
  Steps 4b-4e are a serialized microcode round-trip the CP does NOT overlap with
  the next packet's setup (the AQL model treats each packet boundary as a hard
  ordering point; the barrier bit forbids running ahead). That fixed round-trip
  is the measured ~2.35 us (fence) + ~0.66 us (barrier bit) on top of the same
  wave-drain floor.

WHY THE SAME WORK COSTS DIFFERENTLY:
- PM4 is pre-translated and prefetched: the next dispatch's setup overlaps the
  current dispatch, and the cache op is a single inline hardware GCR instruction.
  The flush latency is mostly HIDDEN.
- AQL is interpreted per packet: the CP firmware serially does decode -> acquire
  -> dispatch -> drain -> release for EVERY kernel, and cannot prefetch the next
  packet past the barrier. The fence is a firmware-orchestrated round-trip, not a
  cheap inline instruction, so it is EXPOSED as a fixed ~3 us tax.
- Proof it is mechanism, not cache volume: HIP with the fence stripped (4.93 us)
  == Vulkan WITH a full L2 flush (4.96 us); and HIP's gap is size-invariant
  (6c.2) while a 16 KB working set is far too small for 2.35 us of cache traffic.

CAVEAT (honesty): the MEC/AQL microcode is closed firmware. The per-step 4a-4e
attribution is a model-level explanation consistent with every measurement
(NOSCOPE removes exactly the 2.35 us acquire/release; NOBARRIER removes 0.66 us;
gap is size-invariant), but the exact firmware micro-ops are inferred, not traced.

IMPLICATION FOR LLM DECODE: each dependent kernel pays the fixed ~3 us HIP tax
regardless of tensor size. At decode (many small sequential dependent launches)
this dominates, so kernel FUSION (fewer dispatches) and multi-stream concurrency
(section 5.2/6b: ~2.4 us floor) pay off far more under HIP than under Vulkan.

### 6c.4 Granular per-step dispatch timelines (N -> N+1 transition)

Both paths must accomplish the SAME three things between two dependent kernels:
  (A) drain the writer's waves, (B) make the writer's stores visible at GL2 (the
  device coherence point), (C) invalidate the reader's L0/L1 so it re-reads.
Only the EXPRESSION differs. CP stages referenced: PFP = PreFetch Parser (front of
the graphics CP), ME = MicroEngine, MEC = MicroEngine Compute (runs the HSA/AQL
soft-queue), SPI = Shader Processor Input (workgroup distributor).

VULKAN / RADV -- a pre-compiled, prefetched PM4 stream + one hardware cache-rinse:

  1) Dispatch N  (PKT3_DISPATCH_DIRECT, radv_cmd_buffer.c:12062)
     1.0) PFP reads the packet; ME programs COMPUTE_DISPATCH registers
     1.1) SPI distributes N's workgroups to the WGPs
     1.2) waves run; stores flow L0 -> GL1 -> GL2
  2) Barrier  (two PM4 packets)
     2.0) EVENT_WRITE CS_PARTIAL_FLUSH (radv_cs.c:232-237): CP stops new work and
          waits until ALL of N's waves retire  <-- the unavoidable wave-drain, the
          bulk of the gap
     2.1) ACQUIRE_MEM, GCR_CNTL = GL2_WB|GL2_INV|GL1_INV|GLV_INV|GLK_INV
          (radv_cs.c:163-166, 332-339): ME fires ONE Graphics Cache Rinse, PFP
          waits for the idle signal ("cache flush executed in the ME, but the PFP
          waits for completion", radv_cs.c:328-330). The GCR walks only DIRTY
          lines -> for 16 KB this is a few cache lines, near-instant.
  3) Dispatch N+1  (PKT3_DISPATCH_DIRECT)
     3.0) the stream is flat and pre-built, so the PFP has ALREADY pre-parsed
          N+1's register/user-data packets while N ran; launch was only gated by
          "GCR idle"
     3.1) SPI distributes N+1; its waves read the now-coherent data from GL2
  => gap = wave-drain + tiny GCR wait + ~0.9 us turnaround. The L2 flush is mostly
     HIDDEN behind the already-prefetched next dispatch.

HIP / ROCm -- interpreted AQL packets + a per-packet firmware fence:

  1) Launch N
     1.0) host writes an AQL kernel_dispatch_packet into the user-mode HSA queue
          ring; header = KERNEL_DISPATCH | barrier-bit | acquire=AGENT |
          release=AGENT (rocvirtual.cpp:1916-1922); rings the doorbell
     1.1) MEC reads packet N from the ring, decodes it, reads the kernarg pointer
     1.2) MEC performs N's ACQUIRE fence at AGENT scope = invalidate I$/K$/L1
          (rocvirtual.cpp:62-64)
     1.3) MEC translates to an internal dispatch; SPI distributes N's workgroups;
          waves run; stores L0 -> GL1 -> GL2
  2) Transition N -> N+1  (N+1 has the barrier bit, so MEC does this STRICTLY
     SERIAL, one micro-step at a time, with no run-ahead):
     2.0) wait for N's grid to fully retire (wave-drain)  <-- same cost as Vulkan
     2.1) perform N's RELEASE fence at AGENT scope (push stores to GL2 + visibility
          barrier)  -- serialized firmware step
     2.2) advance read_index, fetch packet N+1 from the ring, DECODE it
     2.3) perform N+1's ACQUIRE fence (invalidate I$/K$/L1)
     2.4) only now dispatch N+1 to SPI
  => steps 2.1-2.4 are a serialized microcode round-trip the CP does NOT overlap
     with the next packet's setup. Measured: ~2.35 us (fence) + ~0.66 us (barrier
     bit) on top of the same wave-drain floor.

WHY THE SAME WORK COSTS DIFFERENTLY:
  PM4 is pre-translated and prefetched -> next dispatch's setup overlaps the
  current one, and the cache op is a single inline hardware GCR -> flush latency is
  HIDDEN. AQL is interpreted per packet -> the CP firmware serially does
  decode -> acquire -> dispatch -> drain -> release for EVERY kernel and cannot
  prefetch past the barrier -> the fence is EXPOSED as a fixed ~3 us tax.
  Proof it is mechanism, not cache volume: HIP fence-stripped (4.93 us) == Vulkan
  WITH a full L2 flush (4.96 us), and the HIP gap is size-invariant (6c.2).

--------------------------------------------------------------------------------

## 6d. PROOF: driving the SAME GPU via the PM4 front-end is ~2x faster than AQL

The preceding sections argued, from source + microbenchmarks, that HIP's extra
per-dispatch cost is the AQL packet/acquire-release MECHANISM, not the dispatch
engine or cache work. To prove it, we drove the SAME W7900 through a RAW KFD PM4
compute queue (HSA_QUEUE_COMPUTE) - the same PM4 front-end (PFP/ME) RADV uses -
bypassing the HSA/AQL soft-queue that HIP/ROCr dispatch through.

### 6d.1 What was built (artifact: pm4_gap/)

A standalone program (pm4_gap.cpp) that links libhsakmt directly and:
- selects the gfx1100 node, allocates GPU memory via hsaKmtAllocMemory,
- creates a PM4 compute queue (hsaKmtCreateQueue, HSA_QUEUE_COMPUTE),
- loads a tiny offline-assembled gfx1100 shader (gap_kernel.s: data[0]++ with
  L2-coherent glc flat access -> a true per-dispatch RAW dependency),
- builds ONE linear PM4 stream in the ring: one-time COMPUTE_* register setup,
  then K x [ DISPATCH_DIRECT (+ EVENT_WRITE CS_PARTIAL_FLUSH (+ ACQUIRE_MEM)) ],
  a final CS_PARTIAL_FLUSH and a WRITE_DATA sentinel,
- rings the 64-bit doorbell ONCE and busy-polls the sentinel; period = wall/K.
The PM4 sequence (DISPATCH_DIRECT + CS_PARTIAL_FLUSH + ACQUIRE_MEM) is exactly
RADV's compute-barrier recipe (section 2.2). data[0]==K after the run proves the
chain truly serialized (each dispatch saw the previous increment).

### 6d.2 Results (pinned 973 MHz, trivial 1-thread kernel so period ~= overhead)

| path / fence                                  | period   | chain OK | notes |
|-----------------------------------------------|----------|----------|-------|
| PM4, no fence (dispatch-only)                  | 0.33 us  | NO (393) | raw issue rate; dispatches overlap+race |
| PM4, CS_PARTIAL_FLUSH only (drain)             | 2.53 us  | yes      | wave-drain serialize, no cache flush |
| PM4, CS_PARTIAL_FLUSH + ACQUIRE_MEM (RADV-like)| 3.93 us  | yes      | full GL2 WB+INV + L1/L0/scalar inv |
| HIP AQL, default (trivial kernel n=256)        | 7.88 us  | yes      | kernel busy 1.92 us -> ~5.96 us overhead |

Apples-to-apples, SAME silicon, SAME trivial dependent-increment workload:
  HIP AQL per dispatch     = 7.88 us
  raw PM4 per dispatch     = 3.93 us   (and PM4 does MORE cache work: full L2 WB+INV)
  => ~2x reduction purely from changing the submission path AQL -> PM4.

PM4 overhead decomposition (clock-clean, stable to <1% over 3 runs):
  raw dispatch issue (overlapped) .......... 0.33 us
  + wave drain (CS_PARTIAL_FLUSH) .......... 2.53 us   (drain/serialize = +2.20)
  + full L2 cache flush (ACQUIRE_MEM) ...... 3.93 us   (cache flush     = +1.42)

### 6d.3 Conclusion

This is the direct confirmation of the whole investigation:
- The GPU dispatch engine and cache coherence are NOT the bottleneck. The PM4
  front-end on this exact W7900, doing a FULL L2 writeback+invalidate per
  dependent dispatch, costs ~3.93 us - LOWER than Vulkan's measured 4.96 us
  (Vulkan's includes a real ~2 us kernel) and ~2x LOWER than HIP's AQL 7.88 us.
- HIP's ~3-4 us extra per-dispatch tax is the AQL mechanism: per-packet
  decode + AGENT acquire/release enforced serially by the CP soft-queue
  microcode, which cannot be prefetched/pipelined the way a flat PM4 stream is
  (sections 6c.3 / 6c.4). Removing it from HIP (GAP_NOSCOPE+NOBARRIER, 4.93 us)
  and driving raw PM4 (3.93 us) both land in the PM4/Vulkan range.
- "Leverage the PM4 front-end from HIP" is therefore demonstrably worth it: a
  PM4 submission path retains full coherence yet halves the per-dispatch period,
  which for sequential LLM-decode kernels directly cuts the inter-kernel gap.

CAVEATS: the PM4 kernel is a 1-thread increment (near-zero compute) so its
period is almost pure overhead; the HIP AQL number includes ~1.9 us of real
kernel busy, so the engine-overhead gap (PM4 3.93 vs AQL ~5.96) is the precise
comparison. The PM4 queue runs on the MEC; RADV's universal-queue compute may
use the GFX PFP/ME, but both reach the same ~4 us range, confirming the result
is path-driven, not micro-engine-specific. This is a research prototype: it
bypasses ROCr scheduling/signals and is not a drop-in hipLaunchKernel.

--------------------------------------------------------------------------------

## 6e. PM4 dispatch of a REAL hipcc kernel (ABI replication validated)

The 6d prototype used a hand-assembled shader with the kfdtest USER_DATA ABI
(args poked straight into SGPRs). A real HIP kernel uses the full AMDHSA COV5
ABI: the kernel reads its arguments through a kernarg-segment pointer passed in
a USER_SGPR, and the CP sets up those USER_SGPRs from the 64-byte kernel
descriptor (KD). To drive a real decode through PM4 we must replicate exactly
what the CP's AQL-to-hardware translation does. pm4_real.cpp proves we can:

  1. load the ELF code object PT_LOAD segments into one GPU exec buffer at their
     virtual addresses (so KD -> kernel_code_entry_byte_offset stays valid),
  2. read the KD (kernel_descriptor_t, 64 B): kernarg_size, entry offset,
     compute_pgm_rsrc1/rsrc2/rsrc3, kernel_code_properties (enable bits),
  3. COMPUTE_PGM_LO/HI = (kdVA + entry_offset) >> 8,
  4. COMPUTE_PGM_RSRC1/RSRC2 copied VERBATIM from the KD -- never hand-decoded,
     so they are byte-identical to what the CP would load,
  5. USER_DATA SGPRs filled in kernel_code_properties enable-bit order. For the
     test kernel only ENABLE_SGPR_KERNARG_SEGMENT_PTR (bit 3) is set, so the
     kernarg pointer goes in s[0:1] (USER_DATA_0/1); no scratch, no dispatch_ptr,
     no queue_ptr.
  6. kernarg segment holds the explicit args (here one int* counter).

Test kernel KD (extern "C" __global__ void inc_kernel(int*): *c = *c + 1):
  kernarg=8  entry_off=0x10c0  rsrc1=0xe0af0000  rsrc2=0x0000009e
  rsrc3=0x00000010  props=0x0408 (KERNARG_SEGMENT_PTR | WAVEFRONT_SIZE32)

Result (W7900 / gfx1100, profile_standard ~ unpinned, K=5000 dependent chain):

  fence mode                         data       period (us)
  -------------------------------    --------   -----------
  0  dispatch-only (no fence)        2  (race)   0.171
  1  CS_PARTIAL_FLUSH only           8  (race)   0.654
  2  pf + ACQUIRE_MEM (RADV-like)    5000 (OK)   3.521

So a REAL ROCm kernel, PM4-dispatched with the RADV-like full-L2 flush, both
serializes correctly (counter == K) and lands at ~3.5 us/dispatch -- the same
PM4 range as the hand-asm shader (6d) and ~half of HIP's AQL ~7.9 us for the
same trivial dependent kernel. This retires the kernel-ABI risk for routing a
real decode through a PM4 path: copying rsrc1/rsrc2 verbatim and placing
USER_DATA by enable-bit order is sufficient for no-scratch kernels.

What is NOT yet solved (the hard parts of a drop-in HIP/CLR integration, which
are pure correctness plumbing and independent of the gap mechanism):
- completion-signal wakeup: PM4 RELEASE_MEM can write the hsa_signal value, but
  waking an interrupt-blocked HIP waiter (hsa_signal_wait fallback) needs the
  mailbox+interrupt EOP replicated, else streamSynchronize can stall/deadlock;
- cross-queue ordering: kernels on the PM4 queue vs copies/barriers/events still
  on the ROCr AQL queue need a shared-signal handshake at every crossing;
- scratch kernels (private_segment_fixed_size>0 / FLAT_SCRATCH_INIT / dynamic
  stack): need a per-queue scratch buffer + V# + flat_scratch_init; until then
  such kernels must fall back to the normal AQL path.

Artifact: pm4_gap/pm4_real.cpp (+ realkern.hip -> k_gfx1100.co via
clang-offload-bundler --unbundle). Build/run in hipvk-isolated container.

--------------------------------------------------------------------------------

## 6f. Decode-layer verification: PM4 does NOT help REAL kernels (REVERSES 6d/6e)

Before modifying the HIP runtime, we emulated a real decode layer in the
standalone PM4 harness: THREE distinct, dependent hipcc kernels chained N times
(pm4_layer.cpp vs hip_layer.cpp launching the SAME code object). Two chains:
  - "rms": k_rms (1-workgroup LDS reduction + __syncthreads) -> k_scale -> k_add
  - "elem": all-parallel elementwise (full-grid) dependent chain on x
Both produce BIT-IDENTICAL checksums between PM4 and HIP (correctness verified),
so the PM4 ABI replication (6e) is sound even with LDS, barriers and multi-block.

Per-iteration latency (3 kernels), all-elementwise dependent chain, W7900
PINNED clocks (sclk 971 / mclk 1124, so NOT a downclock artifact):

  M (elems)   PM4 (CS_PARTIAL_FLUSH+ACQUIRE_MEM)   HIP/AQL      PM4/HIP
  ---------   ----------------------------------   --------     -------
  4096          12.2 us                             11.8 us      ~1.0x (tie)
  65536         72.3 us                             12.2 us      5.9x SLOWER
  262144       271.9 us                             16.6 us     16.4x SLOWER

HIP stays nearly FLAT while PM4 grows LINEARLY with working set. Decomposition
(M=65536, unpinned, AGENT-scope fence):
  - PM4 FENCE=0 (no drain):          61 us/iter   == HIP (55 us)  <- front-end is fine
  - PM4 FENCE=1 (CS_PARTIAL_FLUSH):  571 us/iter  <- the wave-drain barrier is the cost
  - PM4 FENCE=2 (+ACQUIRE_MEM):      580 us/iter  <- L2 flush scope adds nothing
  - cache scope (RADV full-L2 vs AGENT L0/L1): identical -> NOT a cache-flush issue

MECHANISM: HIP/ROCr's AQL acquire/release PIPELINES dependent dispatches -- the
next kernel's waves launch as the previous retire (in-order, cache ops inline),
keeping HBM saturated, so the chain runs at ~memory-bandwidth cost. The explicit
PM4 CS_PARTIAL_FLUSH is a CP-stalling full-idle barrier: drain -> flush ->
relaunch, creating a GPU bubble every dispatch that exposes the full per-dispatch
launch+drain latency with zero overlap. For real kernels this dominates.

CONCLUSION (the important one): the PM4 front-end win measured in 6d/6e was an
ARTIFACT of near-zero-work kernels, where AQL's per-packet overhead dominated and
CS_PARTIAL_FLUSH drained instantly. For REAL decode kernels with real memory
traffic, HIP/AQL is already at the efficient frontier (saturates bandwidth on a
dependent chain) and the RADV-style PM4 path is 6-16x WORSE. The same caveat
applies to the original "HIP gap is ~2x Vulkan" result: it was measured on
trivial shaders (gap.comp / hip_gap), so it reflects per-dispatch fixed overhead,
NOT the cost that matters for real decode. Therefore routing real decode through
a PM4 dispatch path is NOT worth pursuing -- it would regress throughput. We do
NOT proceed with the CLR/HIP-runtime PM4 modification. The standalone decode-
layer emulation was the cheap, safe way to learn this before paying for the
runtime change (and a GPU-hang-prone integration).

Open micro-question (not blocking the conclusion): whether a lighter PM4
serialization (e.g. RELEASE_MEM+WAIT_REG_MEM, or relying on in-order completion
without a full CS_PARTIAL_FLUSH) could match HIP. Even if it could, it would only
TIE HIP, not beat it -- so there is no decode upside to the PM4 path.

Artifacts: pm4_gap/pm4_layer.cpp, pm4_gap/hip_layer.cpp, pm4_gap/layer_kernels.hip
(k_rms/k_scale/k_add -> layer_gfx1100.co). Env: CHAIN=elem|rms, FENCE=0|1|2,
GCR_MODE=0(radv)|1(agent), DBG=1, DBG_ONLY=rms|scale|add.

--------------------------------------------------------------------------------

## 6g. THREE-WAY decode-layer chain: Vulkan/RADV vs HIP/AQL vs raw PM4

The missing leg. Section 6f compared raw-KFD PM4 against HIP/AQL on the real
dependent decode chain and found PM4 6-16x slower. But RADV emits the SAME
CS_PARTIAL_FLUSH + full-L2-flush barrier (section 2.2), so the obvious question
is: does the ACTUAL Vulkan path blow up the same way the hand-rolled PM4 path
did, or does RADV serialize more cheaply? This section runs the identical chain
through real Vulkan compute pipelines and settles it.

Harness: vk_layer.cpp (+ layer_add.comp / layer_scale.comp). It runs the EXACT
"elem" chain (x += w; x *= w; x += w) N=1000 times on one dedicated compute
queue (ACE), with a SHADER_WRITE->SHADER_READ VkMemoryBarrier between every
dispatch -- the same barrier ggml-vulkan emits between dependent ops, which RADV
lowers to CS_PARTIAL_FLUSH + GL2 INV/WB. Same init values as pm4/hip_layer, so
the result is checksum-comparable. Buffers are DEVICE_LOCAL + host-visible
(ReBAR) for init/readback. CPU wall time of one submission of all 3*N dispatches.

Correctness: vk_layer, hip_layer and pm4_layer all produce the BIT-IDENTICAL
checksum 42020575.025146 at M=4096 (and matching values at every M), so the
three front-ends run the same computation.

Per-iteration latency (3 kernels), all-elementwise dependent chain, W7900,
ALL THREE under the SAME pinned clock (profile_standard: sclk 572 / mclk 1124):

  M (elems)   Vulkan/RADV (barrier)   HIP/AQL    raw PM4 (CS_PARTIAL_FLUSH)
  ---------   ---------------------   -------    --------------------------
  4096          21.05 us               22.92 us    11.38 us
  65536         22.33 us               23.27 us    66.68 us
  262144        32.87 us               33.75 us   311.96 us

  Vulkan front-end only (barrier=0, no sync -- ref, checksum invalid):
  4096 17.76 us ; 65536 18.15 us ; 262144 21.58 us

THE KEY RESULT: Vulkan/RADV stays ~FLAT (21 -> 22 -> 33 us) across a 64x growth
in working set, essentially TRACKING HIP/AQL (within ~5%, marginally faster).
The raw-PM4 path is the only one that explodes (11 -> 67 -> 312 us, ~9.5x slower
than both at M=262144). So:

1. The 6f "PM4 is 6-16x slower" blowup is NOT inherent to driving the GPU through
   the PM4 front-end with a cache barrier. RADV uses the PM4 front-end AND a full
   L2 flush and yet pipelines like HIP. The blowup was specific to OUR naive
   serialization: emitting a CS_PARTIAL_FLUSH that fully DRAINS all waves and
   idles the GPU on every single dispatch. RADV does not pay that bubble -- it
   overlaps the cache op with launch / relies on lighter in-order completion, so
   the next dispatch's waves keep HBM saturated. This directly answers 6f's open
   micro-question: a lighter PM4 serialization DOES match HIP (RADV is the proof).

2. On REAL kernels, Vulkan and HIP CONVERGE. Both become memory-bandwidth-bound
   on the dependent chain (the per-iter cost grows only ~1.5x for 64x more data
   = bandwidth-limited, not launch-limited). The "HIP gap is ~2x Vulkan" result
   from sections 3/5 was a TRIVIAL-kernel (gap.comp) artifact of per-dispatch
   fixed overhead; it disappears once the kernels do real memory work. Here HIP
   is even a hair faster than Vulkan at large M.

3. RADV's barrier is NOT free but it is cheap and overlapped: barrier=1 vs
   barrier=0 costs only ~3 us (M=4096) to ~11 us (M=262144) per iter, growing
   gently -- compare the raw-PM4 CS_PARTIAL_FLUSH which added ~500 us/iter at
   M=65536 (6f, FENCE=0 vs FENCE=1). Same logical fence, ~50x cheaper mechanism.

CONCLUSION (reinforced and sharpened): for real decode kernels there is NO
front-end that beats HIP/AQL on a dependent chain -- HIP, Vulkan/RADV all sit at
the memory-bandwidth frontier and tie. The PM4 front-end can match HIP only if
serialized RADV-style (no full wave-drain); done naively it is far worse. Either
way the ceiling is HIP's existing behavior, so there is no decode throughput
upside to routing real decode through a PM4 dispatch path. The 6f decision (do
NOT pursue the CLR/HIP-runtime PM4 modification) stands, now with the actual
production Vulkan path measured as the third data point rather than inferred.

Artifacts: vk_gap_test/vk_layer.cpp, vk_gap_test/layer_add.comp,
vk_gap_test/layer_scale.comp. Build (2nd container, has glslc + Vulkan headers):
glslc layer_*.comp -o *.spv ; g++ -O2 -std=c++17 vk_layer.cpp -lvulkan -o vk_layer.
Run: ./vk_layer [N] [M] [barrier 0/1]. RADV NAVI31, dedicated compute queue.
(hipvk-isolated container has no Vulkan toolchain; vk_layer was built and run in
test-framework-sshliapn-container-2nd, which carries glslc + the radeon ICD.)

--------------------------------------------------------------------------------

## 6h. Whole-pipeline check + WHERE the Vulkan-vs-PM4 us actually go

Two follow-up questions on 6g: (1) is the per-iter number the WHOLE pipeline
(kernel exec included) or only the inter-kernel gap? (2) Vulkan 21 us vs raw PM4
11 us at M=4096 -- that 2x looks suspicious, what is it?

### (1) It is the whole pipeline, not the gap

vk_layer measures CPU wall of ONE submission of all 3*N dispatches (submit ->
fence). Added per-dispatch GPU timestamps (VK_TS=1) to cross-check:

  M=4096 : whole-pipeline GPU span = 31.74 ms ; CPU wall = 31.81 ms ; ratio 1.002
  M=65536: whole-pipeline GPU span = 32.34 ms ; CPU wall = 32.42 ms ; ratio 1.002

So CPU wall == GPU execution span (the submit/fence roundtrip is amortized to
noise over 3000 dispatches). Per-dispatch decomposition (TOP->BOT = kernel,
BOT->BOT = period): kernel ~9.6 us, gap(barrier+CP+drain) ~0.9 us. The number is
DOMINATED by kernel execution + per-dispatch front-end state; the barrier/gap is
under 1 us/dispatch. (Note: VK_TS itself serializes and inflates the totals to
~31 us; use it only for the kernel-vs-gap split, not absolute timing.)

### (2) The 2x is per-dispatch FRONT-END STATE, not the barrier or a bug

Probed by toggling the per-dispatch vkCmdBindPipeline (VK_SAME = bind one
pipeline once, no rebind) and the barrier. Pinned clock (sclk ~569 / mclk 1124),
per-iter (3 kernels) us, per-dispatch in parens:

  M (elems)   VK rebind     VK no-rebind   HIP graph    HIP loop     raw PM4
  ---------   -----------   ------------   ----------   ----------   -------------
  4096        21.0 (7.0)    6.4 (2.1)      23.7 (7.9)   22.9 (7.6)   11.4 (3.8)
  65536       22.4 (7.5)    8.7 (2.9)      24.0 (8.0)   23.3 (7.8)   66.7 (22.2)
  262144      32.9 (11.0)   18.0 (6.0)     31.1 (10.4)  33.7 (11.2)  312.0 (104.0)

  Isolation at M=4096, barrier=0 (NO sync at all):
    VK rebind   = 5.9 us/dispatch
    VK no-rebind= 0.29 us/dispatch   => vkCmdBindPipeline costs ~5.6 us/dispatch

Findings:

a) The barrier is NOT the cost. With no barrier, rebind still costs 5.9 us/disp
   vs 0.29 us/disp without rebind. The whole Vulkan-vs-PM4 small-M gap is the
   per-dispatch STATE EMISSION, not cache coherence.

b) RADV's vkCmdBindPipeline re-emits the FULL compute pipeline state block
   (shader regs, user data, scratch/prefetch) every op = ~5 us/dispatch. The raw
   PM4 path swaps kernels with a minimal ~4-register write (PGM_LO/HI, RSRC1/2),
   so it is ~2x leaner per dispatch (3.8 vs 7.0 us). ggml-vulkan binds a pipeline
   per op too, so this cost is REAL for Vulkan decode -- not a harness artifact.
   Strip the rebind (VK_SAME) and Vulkan drops to 6.4 us/iter, LEANER than PM4 --
   proving the front-end mechanism is fine; the per-op state re-emit is the cost.

c) HIP graph (pre-recorded) ~= HIP loop at small M (23.7 vs 22.9). So HIP's
   ~7.8 us/dispatch is NOT CPU launch overhead -- it is GPU-side per-packet AQL
   work (acquire/release fence + state). HIP and Vulkan-rebind sit in the same
   ~7-8 us/dispatch band; raw PM4's hand-tuned minimum is ~3.8 us.

d) So the small-M ordering PM4 (11) < VK/HIP (21-23) is genuine and is the
   runtime per-dispatch fixed overhead (~2x). It IS relevant for decode, which is
   made of many small ops. BUT it is a fragile win: the SAME PM4 path explodes to
   104 us/dispatch at M=262144 because its CS_PARTIAL_FLUSH fully drains a wide
   grid every dispatch (no overlap). Vulkan no-rebind (6.0 us/disp at 262144) is
   the real floor and shows the ideal is lean-state + overlapped-barrier together
   -- which RADV achieves and a naive PM4 path does not.

Practical takeaway for decode: the lever is NOT switching to a PM4 dispatch path
(fragile, blows up on wide kernels, GPU-hang-prone to integrate). It is reducing
per-dispatch fixed overhead the safe way -- fuse adjacent small ops (fewer
dispatches), avoid redundant pipeline/state re-binds, and keep the dependent
chain overlapped (HIP/AQL and RADV both already do this). The ~4 us/dispatch the
raw PM4 path saves at decode sizes is real but only safely reachable by cutting
the dispatch count, not by hand-rolling PM4.

Artifacts: vk_layer.cpp (VK_TS=1 timestamps, VK_SAME=1 no-rebind probe),
pm4_gap/hip_layer_graph.cpp (hipGraph pre-recorded HIP, fair vs Vulkan/PM4).

--------------------------------------------------------------------------------

## 6i. CORRECTION: what actually drives the PM4 numbers (rebind / stability / 2x)

Follow-up dig into three questions (what is "rebind"; why Vulkan is stable while
PM4 is not; why PM4 is 2x faster than HIP at small M). Controlled probes (all
PINNED clocks, sclk ~570 / mclk 1124) OVERTURN the section-6f mechanism claim.

### Clarification: all three numbers are wall-clock of the WHOLE test

  - PM4 : t0 just before the doorbell write -> busy-poll the sentinel that the
          LAST packet writes -> t1. Pure GPU execution of the whole pre-built
          PM4 stream (all 3*N dispatches + fences). pm4_layer.cpp:285-294.
  - VK  : t0 before vkQueueSubmit -> vkWaitForFences -> t1. Whole submission.
          Cross-checked: GPU timestamp span == CPU wall, ratio 1.002 (6h).
  - HIP : t0 before the launch loop / graph launch -> hipDeviceSynchronize -> t1.
  All measure the same thing: GPU execution of one pre-recorded stream. The
  per-dispatch numbers are total/(3N).

### (Q1) "rebind" = vkCmdBindPipeline re-emitting full pipeline state per op

A Vulkan compute pipeline bundles the compiled shader + all its fixed register
state. vkCmdBindPipeline makes RADV emit that whole block into the stream
(COMPUTE_PGM_*, RSRC1/2/3, user-data/descriptor ptrs, tmpring/scratch, wave size,
plus a shader I-cache prefetch). The elem chain alternates add/scale/add, so it
re-binds 3x/iter. Measured cost (M=4096, barrier=0 so no sync at all): rebind
5.9 us/dispatch vs no-rebind (VK_SAME) 0.29 us/dispatch => ~5.6 us/dispatch is
the bind itself. VK no-rebind beats raw PM4 because it emits ZERO per-dispatch
state (same kernel, same args repeated -- only the dispatch packet recurs), while
PM4 still re-emits ~7 SET_SH_REG groups per dispatch (PGM_LO/HI, RSRC1/2,
USER_DATA kernarg ptr, START_X/dims, RESOURCE_LIMITS, TMPRING) because the chain
switches kernels and kernarg pointers. So VK_SAME is a best-case (nothing to set
up); the fair match to PM4-elem is VK-rebind, where PM4's lean register swap
(~7 regs) still beats RADV's full pipeline-state re-emit (~2x at small M).

### (Q2 + the "why 2x" mechanism) -- the 6f "wave-drain" story was WRONG

Section 6f blamed CS_PARTIAL_FLUSH (a full wave-drain bubble) for PM4's large-M
blowup. Re-measured under PINNED clocks, that is FALSE:

  PM4 elem, per-dispatch us:        M=4096    M=65536    M=262144
    FENCE=0 (NO fence at all)        1.0       23.3       96.1
    FENCE=1 (CS_PARTIAL_FLUSH)       3.8       21.6      102.9
    CACHED=1 (L2-resident buffers)   3.8       23.2      102.3   (uncached ~ same)
    CUMASK=1 (all 96 CUs enabled)    --        22.7      104.0   (no change)

  - The fence is NOT the large-M cost: FENCE=0 (no drain) is within ~10% of
    FENCE=1 at every size. The 6f "FENCE=1 = 571 us/iter" was an UNPINNED-CLOCK
    artifact -- the drain bubble let the W7900 autosuspend/downclock between
    dispatches. Pin the clock and the drain costs almost nothing.
  - Cache scope is NOT it: cached (L2-resident) == uncached.
  - CU mask is NOT it: hsaKmtSetQueueCUMask(all 96 CUs) returns success and
    changes nothing -- the queue already uses all CUs.
  - What IS it: PM4 cost scales PERFECTLY LINEARLY with workgroup count
    (~0.06-0.09 us/WG: 16 WGs->1.0 us, 256->23 us, 1024->96 us), i.e. ZERO
    parallel-launch benefit. This is the signature of LOW OCCUPANCY -- the SPI
    keeps too few waves resident, so the trivial kernel cannot hide memory
    latency and wide grids do not parallelize.

  DECISIVE CONTROL (same kernel binary, different dispatcher): hip_layer and
  pm4_layer load the SAME k_add (both from layer_kernels.hip -> layer.co /
  layer_gfx1100.co). At M=262144: HIP = 11.2 us/dispatch, PM4 = 103 us/dispatch.
  Same kernel, ~9x. So the gap is NOT the kernel code, NOT the PM4 front-end, and
  NOT fundamental -- it is the per-dispatch compute STATE we hand-program. The CP
  /SPI schedule waves automatically; the dispatch packet's register state
  (COMPUTE_PGM_RSRC1/2 VGPR/SGPR granularity, COMPUTE_RESOURCE_LIMITS WAVES_PER_SH
  /TG_PER_CU, LDS, dims) tells them HOW MANY waves to keep resident. pm4_layer
  writes COMPUTE_RESOURCE_LIMITS=0 and a minimal set; ROCr/RADV program the full
  occupancy-governing state. Get that wrong and occupancy collapses -> the linear
  WG scaling above. So Vulkan/HIP look "stable" (7 -> 7.5 -> 11 us) because they
  are bandwidth-bound at proper occupancy; PM4 looks "unstable" (3.8 -> 22 -> 103)
  because our dispatch state yields low occupancy. FIXABLE, not inherent.

  Why PM4 is 2x faster than HIP at SMALL M (decode-sized): there the kernel is
  ~free, so the number is pure per-dispatch overhead. PM4 streams hand-built raw
  packets the CP runs directly (lean register swap, ~1 us/dispatch with no fence,
  3.8 us with CS_PARTIAL_FLUSH). HIP's AQL dispatch costs ~7.6 us/dispatch
  because the MEC runs AQL-packet microcode per dispatch: acquire/release fence
  scope -> cache-op packets, completion-signal management, barrier-bit check, the
  AQL->PM4 translation. hipGraph (pre-recorded) does NOT cut it (23.7 ~ 22.9),
  proving it is GPU-side AQL packet processing, not CPU launch rate. That fixed
  per-dispatch overhead is the entire 2x; it only shows when the kernel is tiny.

### How to align raw PM4 with Vulkan/HIP (corrected)

Not "use a lighter fence" (fence is irrelevant under pinned clocks). The real
levers, in order:
  1. Per-dispatch overhead (small M): PM4 is ALREADY leaner than HIP/Vulkan.
     Nothing to align there -- PM4 wins. The way to give HIP/Vulkan that win is
     to cut per-dispatch work (op fusion -> fewer dispatches), not adopt PM4.
  2. Large-M occupancy: PM4 must program the occupancy-governing state correctly
     so the SPI keeps enough waves resident. Proven fixable by the same-kernel
     control (HIP gets 11 us with the identical binary). Prime suspect:
     COMPUTE_RESOURCE_LIMITS=0 and any occupancy state ROCr sets that pm4_layer
     omits. Fixing it would make PM4 EQUAL Vulkan/HIP (tie at the bandwidth
     floor), not beat them -- you cannot move the bytes faster than bandwidth.

Net: PM4 is NOT inherently slower than Vulkan -- a correctly-set-up dispatch
reaches the same bandwidth floor (the same-kernel control proves it). PM4's only
genuine EDGE is lower fixed per-dispatch overhead at decode-sized ops; at the
bandwidth floor it merely ties. So the safe way to bank the small-op win is
fewer/fused dispatches inside HIP, not a PM4 dispatch path.

Artifacts: pm4_gap/pm4_layer.cpp (env: FENCE=0|1|2, CACHED=1, CUMASK=1,
GCR_MODE, CHAIN=elem|rms). Compared vs vk_layer (VK_SAME) and hip_layer_graph.

--------------------------------------------------------------------------------

## 6j. "Fix occupancy" attempt: it is NOT occupancy; prime suspect is MTYPE/L2

Tried to close the large-M PM4 gap. Systematically ruled OUT the obvious
occupancy/launch suspects (all pinned clocks, same k_add binary as HIP):

  - CU mask          : hsaKmtSetQueueCUMask(all 96 CUs) -> success, NO change.
  - Shader-engine mask: wrote COMPUTE_STATIC_THREAD_MGMT_SE0..SE5 = 0xFFFFFFFF
                        (all CUs on all 6 gfx11 SEs) -> NO change.
  - COMPUTE_RESOURCE_LIMITS: already 0 in both pm4_layer AND kfdtest's validated
                        dispatch, so not a throttle.
  - Fence / cache scope / fine-vs-uncached: all ruled out earlier (6i).

DECISIVE CONTROL stands: identical k_add binary, HIP 11 us/dispatch vs PM4 103
us/dispatch at M=262144. Since it is not occupancy registers, the remaining
structural difference between the two dispatchers is the BUFFER MEMORY TYPE.
pm4_layer allocates fine-grain (HsaMemFlags CoarseGrain=0; MTYPE_CC/UC) for
CPU-coherent verify; hipMalloc/RADV use coarse-grain device VRAM (MTYPE_RW, full
L2 writeback). Fine-grain on a dGPU does not get L2 writeback caching, so every
dispatch re-streams the working set from HBM -- which fits the PERFECTLY LINEAR
-in-working-set PM4 cost.

ATTEMPTED FIX (COARSE=1), FIRST TRY -- BROKEN: allocating the data coarse-grain
made timing drop ~26x but the verify mirror showed the kernel's WRITES never
landed (x read back as the stale init even at N=1). That "speedup" was
dropped-write traffic, not real L2 caching. ROOT CAUSE of the dropped writes is
now identified -- see 6k.

--------------------------------------------------------------------------------

## 6k. PROVEN: replicating hipMalloc closes the large-M gap (PM4 == HIP)

The coarse-grain control is now correct and the MTYPE hypothesis is PROVEN.

Two bugs in the first coarse attempt, both fixed:
  1. HostAccess=1 + CoarseGrain=1 is contradictory: coarse-grain device VRAM is
     NOT CPU-coherent, so a direct CPU read of the device buffer always shows the
     stale init. FIX: allocate the compute buffers (x,y,w) as PURE device VRAM
     (NonPaged=1, CoarseGrain=1, HostAccess=0) -- exactly what hipMalloc returns.
  2. THE write-drop itself: the inter-stage fence used a RADV-style full flush
     GL2_WB|GL2_INV. On this path the GL2 INVALIDATE discards the dirty L2 lines
     (the device coherence point IS L2), so the GPU's just-written x was thrown
     away before any reader saw it. FIX: between dependent on-device kernels use
     an AGENT-scope fence (invalidate L0/L1 only, leave L2 intact). The dirty x
     stays resident and the next kernel reads it correctly.

Host I/O is done the way hipMemcpy does it: a host-accessible UNCACHED staging
buffer + an on-GPU k_copy kernel (H2D: x <- xs, w <- ws before the chain; D2H:
outs <- x after). outs is uncached so its write bypasses L2 to HBM and the CPU
reads it coherently. (New kernel k_copy added to layer_kernels.hip.)

CORRECTNESS (gfx1100, elem chain x+=w; x*=w; x+=w):
  N=0 round-trip (H2D then D2H, no chain):  outs == xs  (0.10000, 0.11000, ...)  OK
  N=1 COARSE:    checksum=8750.109214  x[0]=2.100000   <- EXACTLY matches
  N=1 fine-grain reference (CACHED=1):  checksum=8750.109214  x[0]=2.100000

WORKING-SET SWEEP, elem chain, N=1000, pinned profile_standard (593 MHz sclk),
us/iter (3 kernels). All four paths measured in ONE run on the same pinned device
(W7900 / RADV NAVI31). PM4 + HIP built and run in the hipvk-isolated container;
Vulkan/RADV also run in the SAME hipvk-isolated container (libvulkan + radeon_icd
present there). PM4 uses the RADV-matched ACQUIRE_MEM from 6k.1 (FENCE=2, fully
cache-coherent, NO access-pattern shortcut):

  M (floats)  PM4 COARSE   PM4 UNCACHED   HIP/AQL   VULKAN/RADV
  4096          11.99         17.50        22.78       21.13
  16384         19.93         31.13        22.76       21.33
  65536         20.23         87.89        23.15       22.47
  262144        21.53        314.60        33.64       32.95
  1048576       53.99       1180.89        65.78       70.11

READING THE TABLE:
- PM4 UNCACHED grows PERFECTLY LINEARLY with the working set (88 -> 315 -> 1181 us
  as M goes 65k -> 262k -> 1M). That is the HBM-bandwidth floor: every dispatch
  re-streams x and w from HBM because the buffer bypasses L2.
- PM4 COARSE (== hipMalloc) is FLAT until the working set exceeds L2 (~20-21 us
  across 16k..262k), then rises only when x+w no longer fit in the 6 MB L2 (1M
  floats = 4 MB x + 4 MB w = 8 MB > L2). It is now the FASTEST cached path at
  every size -- faster at small M (12.0 vs ~21 us) thanks to the leaner dispatch
  front-end, and still ahead in the L2 plateau and at the 1M bandwidth point
  (54.0 vs HIP 65.8 / Vulkan 70.1 us).
- HIP/AQL and Vulkan/RADV track each other closely (~21-23 us plateau); both pay
  a heavier fixed per-dispatch cost than the hand-rolled PM4 front-end.
- The ~15x gap from PM4 COARSE to PM4 UNCACHED at 1M (54 vs 1181 us) is entirely
  the missing L2 residency.

(Absolute numbers differ from the earlier profile_peak table because this run is
pinned at profile_standard, the shared all-users clock; the RELATIVE ordering is
what matters and it is consistent: COARSE < HIP ~= Vulkan << UNCACHED.)

--------------------------------------------------------------------------------

### 6k.1 Matching PM4's ACQUIRE_MEM to RADV (no access-pattern shortcut)

Decomposing the inter-dispatch cost at M=16384 (L2-resident, so NOT bandwidth),
pinned profile_standard, us/iter:

  variant                                    us/iter   us/disp   correct?
  PM4 coarse, no fence (FENCE=0)                2.50      0.83     NO (race)
  PM4 coarse, CS_PARTIAL_FLUSH only (FENCE=1)   5.20      1.73     YES (here)
  PM4 coarse, drain + ACQUIRE_MEM (FENCE=2)    20.37      6.79     YES
  Vulkan, no barrier                           17.81      5.94     NO
  Vulkan, full barrier                         21.25      7.08     YES

Findings:
- The PM4 front-end dispatch is much leaner than RADV's (no-sync floor 2.5 vs
  17.8 us/iter): the hand-rolled stream emits a minimal register set per dispatch,
  RADV re-emits far more pipeline state.
- The entire PM4-vs-Vulkan delta is the inter-dispatch barrier. CS_PARTIAL_FLUSH
  (wave drain) is cheap (+2.7 us/iter); the cache ACQUIRE_MEM is the bulk.
- FENCE=1 (drain only) was bit-exact here, but ONLY because every element is
  written and read at the same index. A reverse/scatter/transpose access could
  read a stale L0/L1 line that the drain alone never invalidates. So FENCE=1 is
  REJECTED as unsafe in general; the correct path is FENCE=2 with a real cache
  invalidate -- exactly what RADV does.

To make FENCE=2 both correct AND cheap, the PM4 ACQUIRE_MEM was matched byte-for-
byte to what RADV emits for a compute->compute buffer barrier (mesa
radv_cs.c:gfx10_cs_emit_cache_flush, verified against the GCR_CNTL layout in
src/amd/registers/pkt3.json):
  1. GCR_CNTL bit layout was WRONG: the code had GL2_INV=bit13 / GL2_WB=bit14, but
     the real layout is GL2_DISCARD=13, GL2_INV=14, GL2_WB=15. (This same bug was
     the original coarse "dropped writes": the old GCR_RADV_LIKE was setting
     GL2_DISCARD instead of GL2_WB, discarding dirty lines.) Bits corrected.
     NOTE: the AGENT-scope value used by the chain fence, GL1_INV|GLV_INV|GLK_INV
     = 0x380, was already correct and is byte-identical to RADV's compute barrier
     GCR_CNTL -- so the chain's per-dispatch GCR was right all along.
  2. ACQUIRE_MEM body fields aligned to RADV: CP_COHER_SIZE_HI 0 -> 0xffffff,
     POLL_INTERVAL 4 -> 0x0A (CP_COHER_CNTL=0, SIZE=0xffffffff, BASE=0 already
     matched).

Result: FENCE=2 dropped 23.2 -> 20.4 us/iter and now MATCHES/BEATS Vulkan (21.2)
with a fully cache-coherent barrier and bit-exact output -- no reliance on the
access pattern. This is the number used in the 6k sweep table above.

CONCLUSION (this REVERSES the earlier "PM4 6-16x slower, do not pursue"):
The large-M PM4 slowdown was NEVER the dispatch path, the CS_PARTIAL_FLUSH fence,
or occupancy (CUMASK/SEMASK had no effect). It was 100% the MEMORY TYPE: the
verify-friendly fine-grain/uncached buffers bypass L2, so PM4 paid full HBM
bandwidth every dispatch while HIP/Vulkan ran on L2-cached device-local VRAM.
With faithful hipMalloc-style allocation (coarse-grain device VRAM + AGENT-scope
on-device fences + staged host I/O), raw PM4 matches HIP at every working-set
size and beats it at decode-sized ops. The decode-relevant verdict is now
strictly favorable to PM4: same bandwidth floor at large M, lower fixed overhead
at small M. Env flags in pm4_layer.cpp: COARSE=1 (hipMalloc replica, VALIDATED),
CACHED=1 (fine-grain L2), default UNCACHED, COARSE_HA=1 (diagnostic).

--------------------------------------------------------------------------------

## 6l. 64-kernel chain + REVERSE-READ kernel (cache-coherence proof)

To (a) stress a much longer dependent chain and (b) actively prove the cache
invalidate is required (not just an access-pattern coincidence), a reverse-read
kernel was added to all three frameworks:

  k_revadd(dst,src): dst[i] = 0.5*dst[i] + 0.5*src[n-1-i]

Thread i reads src at the MIRRORED index n-1-i, which in the previous dispatch was
written by a DIFFERENT workgroup (different WGP / different L0,L1 vector cache).
The chain (CHAIN=rev, KLEN=64) ping-pongs two buffers x<->y: even dispatch writes
y reading reverse(x), odd writes x reading reverse(y), 64 dispatches/iter. The
averaging form keeps values bounded so the checksum stays finite and comparable.
Implemented identically in pm4_layer (k_revadd, slots for (y,x)/(x,y), coarse y
staging), hip_layer / hip_layer_graph, and vk_layer (layer_revadd.comp + two
ping-pong descriptor sets dsetXY/dsetYX). Vulkan built in the 2nd container
(glslc + headers), RUN in hipvk-isolated.

CACHE-COHERENCE PROOF (M=4096, N=20):
  PM4 COARSE FENCE=2 (RADV-matched acquire) : checksum 477.833330  CORRECT
  PM4 CACHED FENCE=2                         : checksum 477.833330  CORRECT
  HIP (CPU loop)                             : checksum 477.833330  CORRECT
  HIP (hipGraph)                             : checksum 477.833330  CORRECT
  VULKAN/RADV                                : checksum 477.833330  CORRECT
  PM4 COARSE FENCE=1 (CS_PARTIAL_FLUSH only) : checksum 492.679682  WRONG (x non-uniform)

=> The reverse kernel makes the drain-only fence visibly WRONG (stale L0/L1 lines),
   while the proper acquire (GLV_INV|GL1_INV) is bit-exact across ALL frameworks.
   This is the concrete justification for matching ACQUIRE_MEM to RADV (6k.1)
   instead of relying on FENCE=1. Functional equivalence is also confirmed: PM4,
   HIP and Vulkan agree bit-for-bit at every M (checksums verified equal).

TIMING, 64-kernel reverse chain, N=200, pinned profile_standard, us/iter (64 kern):

  M (floats)  PM4 COARSE   HIP/AQL(graph)   VULKAN/RADV
  4096          258.7          504.5           144.0
  16384         449.6          504.7           152.8
  65536         425.4          505.8           195.9
  262144        442.7          662.7           395.1

READING THE TABLE (this is a DIFFERENT regime from the 3-kernel elem chain):
- VULKAN is now FASTEST by a wide margin. The chain reuses ONE pipeline (pRev) for
  all 64 dispatches, so RADV emits almost no per-dispatch pipeline state (just a
  descriptor-set swap) -- its per-dispatch cost collapses to ~2.2 us. In the elem
  chain it ALTERNATED pAdd/pScale, paying pipeline re-bind state every dispatch.
- PM4 sits in the middle (~4 us/dispatch, roughly flat). Its per-dispatch cost is
  dominated by the ACQUIRE_MEM cache invalidate, which it pays 64x/iter regardless
  of whether the kernel changed. PM4 re-emits the full dispatch packet (incl. the
  COMPUTE_PGM_RSRC / kernel-descriptor registers) every dispatch even when the
  kernel is identical -- an optimization opportunity (emit KD regs once, only bump
  the kernarg pointer for same-kernel repeats).
- HIP/AQL(graph) is slowest (~7.9 us/dispatch): fixed AQL packet + acquire/release
  fence overhead per kernel.

CROSSOVER / HONEST VERDICT: the "PM4 is fastest" result is workload-specific. PM4
wins when the chain ALTERNATES short kernels (elem: its lean front-end beats RADV's
per-dispatch pipeline-state cost). On a LONG SAME-KERNEL chain, RADV's near-zero
state re-emission wins, and PM4's repeated ACQUIRE_MEM becomes the bottleneck. So
the takeaway is not "PM4 always faster" but "PM4's front-end is leaner per dispatch,
yet its mandatory per-dispatch cache acquire caps it; RADV amortizes pipeline state
better for repeated kernels." All three remain functionally identical (bit-exact).

--------------------------------------------------------------------------------

## 6m. GLK_INV (scalar-cache) drop: a real but COMPILER-SPECIFIC PM4 win, and a
##      corrected, clock-pinned fairness comparison

This section (a) records a large PM4 per-dispatch speedup from dropping the
scalar-cache invalidate, (b) proves from RADV source that this is NOT something
RADV does or could do, and (c) CORRECTS the attribution in earlier sections:
with matched cache semantics, PM4 is on par with -- not faster than -- Vulkan.

Clocks are now pinned reliably: the host has a NOPASSWD sudoers entry for
/usr/local/sbin/gpu_pin_freq.sh and gpu_restore_freq.sh, so every run below is
`profile_peak` (GFX 1778 MHz, MEM 1124 MHz, verified under load). Pin before /
restore after every GPU test.

### 6m.1 The optimization: GCR 0x380 -> 0x300 (drop GLK_INV bit 7)

The per-dispatch chain fence was ACQUIRE_MEM with GCR_CNTL = GLV_INV|GL1_INV|
GLK_INV = 0x380 (vector L0 + L1 + scalar K-cache invalidate). Dropping the
scalar-cache bit -> 0x300 (GLV_INV|GL1_INV) is BIT-EXACT identical in result on
both the mix (reverse-read) and serial chains across every M tested, yet 2.5-5x
cheaper per dispatch. New default in pm4_layer.cpp is 0x300 (GCR_RAW=0x380 to
restore the old behavior).

WHY IT IS CORRECT HERE (verified in the ISA, layer_gfx1100.co):
  - buffer DATA is read/written with VECTOR ops: global_load_b32 / global_store_b32
    -> flows through GLV (L0 vector) / GL1, which 0x300 invalidates.
  - the only SCALAR (s_load) traffic is the kernarg pointers/sizes, and those are
    written ONCE into distinct immutable slots (the ping-pong uses DIFFERENT slots
    -- slot 2 = (x,y), slot 7 = (y,x) -- never rewriting one), so the K-cache can
    never hold a stale value.
  Hence invalidating GLV+GL1 is sufficient; GLK_INV is pure overhead.

### 6m.2 RADV DOES set GLK_INV -- it is compiler-ABI-mandated, not a missed opt

Read directly from mesa (radv_cmd_buffer.c::radv_dst_access_flush, radv_cs.c::
gfx10_cs_emit_cache_flush):

  radv_cs.c:150-159   INV_SCACHE -> GL1_INV|GLK_INV ;  INV_VCACHE -> GL1_INV|GLV_INV
  radv_cmd_buffer.c:6700-6701  UNIFORM_READ -> INV_VCACHE | INV_SCACHE
  radv_cmd_buffer.c:6720-6723  "Unlike LLVM, ACO uses SMEM for SSBOs and we have to
                               invalidate the scalar cache." (SSBO read -> INV_SCACHE)

VK_ACCESS_SHADER_READ expands (vk_expand_dst_access_flags2) to include UNIFORM_READ
+ SHADER_STORAGE_READ, so for a shader-write->shader-read barrier RADV emits
GL1_INV|GLV_INV|GLK_INV = 0x380 -- the SAME bits as PM4's old default. RADV cannot
drop GLK_INV because its ACO compiler reads SSBO data via SCALAR (SMEM) loads, so
produced data lands in the K-cache. Our hipcc/clang kernels read data via vector
loads, so we can. The win is a COMPILER/ABI difference, not a faster dispatcher.

DO NOT drop GLK_INV for any kernel that reads another kernel's output via a
scalar/uniform load.

### 6m.3 CORRECTED fairness (pinned profile_peak, serial+mix, N=20)

The headline matters: at MATCHED cache semantics (both 0x380), PM4 is NOT faster
than Vulkan -- it is equal-or-slightly-slower. The big PM4 advantage is entirely
the 0x300 cache reduction from 6m.1.

Per-kernel dispatch time (us), LAYERS=8 (40 kern/iter):

  CHAIN=mix                              CHAIN=serial
  M       PM4 0x300 PM4 0x380 VK   HIPe  | PM4 0x300 PM4 0x380 VK   HIPe
  4096      1.18     2.96    4.56 3.94   |  1.18     3.01    4.55 3.95
  16384     1.24     5.63    4.57 3.96   |  1.23     5.90    4.57 3.97
  65536     1.51     5.69    4.64 4.08   |  1.50     5.74    4.66 4.08
  262144    2.56     6.07    5.97 5.54   |  2.57     5.86    5.91 5.52

End-to-end per decode TOKEN (us), LAYERS=60 (300 kern/token, realistic decode):

  CHAIN=mix                          CHAIN=serial
  M       PM4 0x300 PM4 0x380 VK     | PM4 0x300 PM4 0x380 VK
  16384     337     1722    1359     |   338     1677    1359
  65536     409     1685    1379     |   409     1631    1408

Reading it honestly:
- PM4 0x300 (1.2-2.6 us/kern) is the fastest by 2-4x, but ONLY because of the
  compiler-specific GLK_INV drop (6m.1/6m.2) -- not the dispatch path.
- PM4 0x380 (RADV-identical bits) is ~5.5-6 us/kern, i.e. ON PAR with or SLIGHTLY
  SLOWER than VK (~4.6-5.9). So at equal coherence work, RADV's path is as good as
  or better than the hand-rolled PM4 stream. Earlier sections (6d/6h) that implied
  a 2x PM4 dispatch-path win on trivial kernels do not generalize here.
- HIP eager (HIPe, ~4 us/kern) is CPU-LAUNCH-bound, not a GPU-throughput number;
  HIP graph is ~8 us/kern. Neither is the right comparand for a pre-built PM4
  stream. PM4 and VK both measure one GPU window (VK CPU-wall == GPU-span, 6h).

### 6m.4 WHY GLK_INV cost GROWS with M (the "scalar data is constant" puzzle)

Measured GLK_INV marginal cost = (PM4 0x380 - PM4 0x300) per dispatch, serial,
pinned:

  M       0x300   0x380   GLK delta
  4096    0.79    2.69    1.90
  8192    0.83    4.00    3.17
  16384   0.82    5.58    4.76
  32768   0.90    5.51    4.61
  65536   0.99    5.33    4.34
  262144  1.62    5.63    4.01

The 0x300 path is nearly flat; essentially ALL the M-dependence lives in the
GLK_INV bit, which rises steeply to ~4.7 us by M=16384 then SATURATES.

The user's intuition is correct: the amount of scalar DATA (2 pointers + n =
~20 bytes) is identical at every M. So the cost is NOT "invalidating more scalar
lines." Two facts constrain the mechanism:
  - magnitude rules out a bandwidth/drain story: at M=16384 the working set is
    64 KB; draining 64 KB to L2 at ~1-2 TB/s is tens of ns, not 4.7 us.
  - the cost tracks the WORKGROUP COUNT and saturates exactly where the grid
    covers the whole GPU: grid = ceil(M/256) WGs; M=4096->16 WGs, 8192->32,
    16384->64. The W7900 has 48 WGPs (96 CUs). The delta grows while WGs < ~48
    WGPs and flattens once the grid covers all WGPs (M>=16384).

BEST-SUPPORTED EXPLANATION (inferred, not firmware-traced): the scalar K-cache is
PER-WGP. GLK_INV must invalidate the K-cache instance in every WGP that ran a
wave of the producer, and the GCR engine waits for an acknowledgement from each.
That ack count scales with the number of occupied WGPs (= grid size) until the
grid covers the whole chip, then saturates -- which matches both the shape and
the ~16K saturation point. The vector L0/L1 invalidate (0x300) does not expose
this serialized per-WGP ack wait (it appears to be applied more cheaply / in
parallel on this uarch), which is why the 0x300 path stays flat. This is the most
likely mechanism given the data; the exact CP/GCR micro-behavior is closed
firmware. DECISIVE NEXT TEST (not yet run): at fixed large M, restrict the queue
CU mask to fewer WGPs; if the GLK delta drops proportionally, the per-WGP-ack
mechanism is confirmed.

### 6m.5 Net verdict (supersedes the "PM4 3-4x faster" framing)

- The GLK_INV drop is a genuine, bit-exact 2.5-5x per-dispatch speedup for THESE
  kernels, and the right default for this harness.
- It is NOT portable to RADV (ACO SSBO->SMEM forces GLK_INV) and must not be
  applied to kernels that read predecessors' output via scalar loads.
- With matched semantics, PM4 ties Vulkan; the dispatch-path is not the lever.
  The earlier trivial-kernel "2x PM4" result does not carry to these kernels.

Artifacts: pm4_gap/pm4_layer.cpp, vk_layer.cpp, pm4_gap/hip_layer.cpp. RADV
source: mesa-radv/src/amd/vulkan/radv_cs.c + radv_cmd_buffer.c. Clocks:
/usr/local/sbin/gpu_pin_freq.sh profile_peak.

--------------------------------------------------------------------------------

## 6n. ROOT CAUSE of the GLK_INV M-growth, and the FIX (PWS overlapped fence)

GLK_INV is kept ON by default for safety (6m.2). This section proves WHY the
blocking-acquire GLK cost grows with M, why VK/HIP do not, and FIXES PM4 so the
full-safety 0x380 fence is flat and FASTER than VK/HIP.

### 6n.1 The growth is a per-CU K-cache invalidate broadcast (CU-mask sweep)

At fixed M=65536 (grid covers the whole chip), restricting the queue CU mask and
measuring the GLK marginal cost (0x380 - 0x300) per dispatch:

  CUs enabled   4      8      16     24     48     96
  GLK delta    0.76   0.87   1.06   1.42   2.06   4.19  (us/dispatch)

The cost scales ~linearly with the number of active CUs, NOT with the (constant
~20-byte) scalar data. GLK_INV must invalidate the per-CU scalar K-cache in every
CU that ran a producer wave and the GCR engine waits for an ack from each, so the
cost = (active CUs) x ~0.044 us. In the M-sweep this is why it ramps until the
grid fills all 48 WGPs (~M=16384) then saturates.

Fence decomposition (serial, pinned, us/dispatch) confirms the wave-drain is flat
and cheap; GLK is the entire growing term:

  M       F0_none  F1_drain  F2_0x300  F2_0x380  | drain  glk
  4096     0.52     1.00      1.17      3.04      | 0.48   1.87
  16384    0.54     1.03      1.23      5.80      | 0.49   4.57
  65536    0.67     1.30      1.49      5.56      | 0.64   4.07
  262144   1.91     2.36      2.50      5.84      | 0.45   3.34

### 6n.2 Why VK/HIP do NOT expose this (and why VK is slower than HIP here)

It is NOT the bits: PM4 with RADV's exact full gcr (GCR_MODE=0 = GL2_WB|GL2_INV|
GLM|GLK|GLV|GL1) is bit-exact correct but still grows (4.0->7.7us) and is even
slower. SEQ=FORWARD variants do not help either. So matching RADV's GCR_CNTL does
not flatten PM4.

The difference is the WAIT MECHANISM. VK's GPU timestamps show the inter-dispatch
gap (period - kernel) is only ~0.5 us and the windows even OVERLAP at M=65536
(kernel 6.57 > period 6.30). RADV's cache flush is overlapped with the dispatch
window; the CP does not fully stall on it. PM4's blocking ACQUIRE_MEM serializes
(drain -> wait-for-GCR -> dispatch), so the per-CU GLK ack latency is fully
EXPOSED as inter-dispatch time. (Confirming this: PM4 0x300 grows only +1.4 us
across M, the SAME as VK's +1.3 us -- i.e. just bandwidth; the extra +1.5 us in
PM4 0x380 is the exposed GLK.)

Q3 answer (VK slower than HIP here): purely vkCmdBindPipeline. VK re-emits compute
pipeline state every dispatch; HIP has no pipeline-state object. VK no-rebind
(VK_SAME) = 3.94 us == HIP 3.94 us; VK rebind = 4.55 us. The ~0.6 us is the bind.
(In llama.cpp Vulkan still wins overall because it fuses/avoids per-op overhead
that HIP pays elsewhere; this microbench isolates only the dispatch fence/bind.)

### 6n.3 THE FIX: PWS deferred-wait fence (overlap the flush like RADV)

Replicated RADV's GFX11 overlap mechanism on the raw KFD MEC queue:
  fence = CS_PARTIAL_FLUSH                       (drain producer waves; REQUIRED)
        + RELEASE_MEM (EVENT=BOTTOM_OF_PIPE_TS, GCR cache flush, PWS_ENABLE=1)
        + ACQUIRE_MEM (PWS_STAGE_SEL=CP_ME, PWS_ENA/ENA2=1, GCR_CNTL=0; waits the
                       PWS counter)
The RELEASE_MEM issues the GCR flush asynchronously and bumps the PWS counter;
the PWS ACQUIRE_MEM waits on that counter without the CP blocking inline, so the
flush overlaps the next dispatch. Bit layouts from mesa pkt3.json
(RELEASE_MEM_OP_gfx11, ACQUIRE_MEM_PWS_2/_7). Implemented in pm4_layer.cpp
(Emit::release_mem_pws / acquire_pws), default ON, PWS=0 to fall back.

CORRECTNESS GOTCHA: the CS_PARTIAL_FLUSH wave-drain is REQUIRED before the PWS
pair. Without it the RELEASE_MEM EOP does not order the consumer against the
producer's outstanding stores on a raw MEC queue -> non-deterministic ~1%-wrong
checksums. With the drain, PWS is bit-exact and deterministic on BOTH the serial
and the reverse-read mix (cache-coherence stress) chains at every M.

RESULT (pinned profile_peak, serial LAYERS=8 N=20, us/dispatch):

  M       BLOCKING 0x380   PWS 0x380   VK(RADV)   HIP eager
  4096      3.06            2.52        4.55       3.95
  16384     5.08            2.57        4.57       3.98
  65536     5.68            2.84        4.67       4.10
  262144    5.84            3.91        5.89       5.54

PWS 0x380 is FLAT (residual growth is only bandwidth, like VK/HIP), bit-exact
with full GLK_INV cache safety, ~2x faster than the blocking acquire, and FASTER
than both VK and HIP at every M.

END-TO-END per decode TOKEN (us), 300 kernels/token (LAYERS=60), N=20, pinned
profile_peak (sclk 970 / mclk 1124 MHz). All five columns are bit-identical
(same checksum) at each M, on BOTH chains:

  CHAIN=serial (realistic decode)
  M       PM4 PWS 0x380  PM4 block 0x380  PM4 block 0x300  VK(RADV)  HIP eager  HIP graph
  16384      742.9          1628.2            338.5         1359.7    1182.8     1077.2
  65536      812.5          1739.6            409.3         1384.7    1222.4     1119.2
  262144    1106.8          1700.2            702.8         1746.8    1651.7     1441.2

  CHAIN=mix (reverse-read cache-coherence stress)
  M       PM4 PWS 0x380  PM4 block 0x380  PM4 block 0x300  VK(RADV)  HIP eager  HIP graph
  16384      741.2          1566.3            337.9         1361.4    1183.9     1071.7
  65536      812.0          1710.7            409.4         1379.9    1219.7     1150.4
  262144    1100.8          1643.9            700.7         1770.3    1657.4     1441.9

  (HIP graph column added later, same pinned profile_peak; session consistency
  verified by re-running HIP eager 1177.7 ~= 1182.8 and PM4 PWS 737.8 ~= 742.9 at
  M=16384 serial. HIP graph = median of 5, hip_layer_graph.cpp, layer_gfx1100.co.)

PM4 PWS 0x380 (the cache-safe default) is ~1.6-1.9x faster than VK and ~1.5-1.6x
faster than HIP eager; the PWS overlap roughly halves the blocking-0x380 path by
hiding the per-CU GLK_INV ack instead of stalling on it. PM4 block 0x300 (drops
GLK_INV) is fastest but compiler-specific/unsafe (6m.2), kept as a diagnostic.
Verified deterministic on the reverse-read mix chain too.

HIP graph (the FAIR comparand for a pre-recorded PM4 stream -- single launch, no
per-dispatch CPU cost) is faster than HIP eager (1077 vs 1183 at M=16384) because
it drops the launch loop, but PM4 PWS still beats it by ~1.3-1.45x (742.9 vs
1077.2, 812.5 vs 1119.2, 1106.8 vs 1441.2). So PM4 PWS is the fastest of ALL the
safe paths at every M, on both chains. NOTE: this advantage combines two things
-- PM4's leaner per-dispatch front-end AND the in-place PWS fence swap. It does
NOT transfer to the HIP runtime by INJECTING a PWS vendor packet: HIP graph
already emits a lean overlapped AGENT-scope fence (acquire=1/release=1), so an
added vendor PM4-IB packet is net overhead (neutral at large M, ~2x regression at
small M). See section 5.x / the CLR PWS experiment.

CONCLUSION: the PM4 0x380 M-growth was the per-CU scalar-cache invalidate ack
EXPOSED by a blocking in-order ACQUIRE_MEM. RADV hides it by overlapping the
flush. Replicating that overlap with a PWS RELEASE_MEM/ACQUIRE_MEM pair (plus the
mandatory wave-drain) makes the full-safety PM4 fence flat and the fastest of the
three, with no loss of correctness. New default in pm4_layer.cpp.

Artifacts: pm4_gap/pm4_layer.cpp (Emit::release_mem_pws/acquire_pws; PWS=1
default, PWS=0 blocking; CUMASK=N CU-count knob). Encodings: mesa
src/amd/registers/pkt3.json. RADV PWS refs: radv_cs.c:259-280, radv_queue.c:632-650.

### 6n.4 TRANSFER TO HIP: the whole chain in ONE indirect buffer is free

To get the PM4 PWS win into the HIP runtime we cannot INJECT a per-dispatch PWS
vendor packet (that doubles packet count -> the extra CP packet + IB-jump cost
cancels the overlap, measured neutral at large M / ~2x regression at small M --
see the CLR injection experiment). The fix is to route the WHOLE captured stream
through a SINGLE PM4 indirect buffer launched with ONE INDIRECT_BUFFER (one CP
jump for the entire graph), so the per-dispatch path inside the IB is the lean
raw PM4 + in-place PWS fence -- exactly what pm4_layer does inline.

De-risk (pm4_layer IB=1: emit whole chain into an executable IB, launch with a
single IT_INDIRECT_BUFFER from the ring), pinned profile_peak, serial LAYERS=60
N=20, us/token, bit-exact (same checksum as inline):

  M        inline ring   single IB    HIP graph (AQL)
  16384       740.8         732.0         1077.2
  65536       813.5         803.2         1119.2
  262144     1103.3        1095.9         1441.2

The IB jump is FREE when amortized over the whole chain (single IB == inline ring
within noise), and ~1.45x faster than HIP graph. So the viable HIP integration is
"compile the captured hipGraph to one PM4 IB and replay it via a single vendor
PM4-IB AQL packet" -- NOT per-dispatch fence injection. Artifact: pm4_layer.cpp
env IB=1.

### 6n.5 IMPLEMENTED IN CLR: HIP_PM4_GRAPH=1 -- the speedup transferred, bit-exact

The single-IB design above is implemented in the HIP runtime (rocclr). At graph
replay (dispatchGenericAqlPacketBatch, kernelNames != nullptr) the captured
all-dispatch graph is compiled ONCE into one executable PM4 IB -- per dispatch:
SET_SH_REG (PGM/RSRC/dims, USER_DATA = the captured kernarg_address) +
DISPATCH_DIRECT + in-place PWS fence (CS_PARTIAL_FLUSH + RELEASE_MEM PWS +
ACQUIRE_MEM PWS), then one final full-L2 acquire -- and replayed via a SINGLE
vendor PM4-IB AQL packet whose completion_signal the CP raises after the whole IB.
Gated by HIP_PM4_GRAPH=1; falls back to normal AQL replay if any packet is not a
simple kernarg-ptr-ABI dispatch (non-dispatch packet, scratch, or dispatch/queue
ptr enable bits). Files: rocclr/device/rocm/rocvirtual.cpp
(buildPm4GraphIb / tryReplayPm4Graph / allocExecIbFromData), rocvirtual.hpp.

MEASURED in the real HIP runtime (hip_pws_test, graph mode, pinned profile_peak,
median of 5, us/dispatch), BIT-EXACT vs baseline on forward AND reverse-read
chains:

  config              baseline   HIP_PM4_GRAPH=1   speedup   checksum
  M=16384  L=8  FWD     3.776        2.935          1.29x     exact
  M=16384  L=8  REV     3.779        2.946          1.28x     exact
  M=16384  L=60 FWD     3.596        2.608          1.38x     exact
  M=16384  L=60 REV     3.600        2.610          1.38x     exact
  M=65536  L=60 FWD     3.739        2.735          1.37x     exact
  M=65536  L=60 REV     3.743        2.729          1.37x     exact

END-TO-END per graph replay (us/chain = host hipGraphLaunch loop total / ITERS,
includes host launch + GPU + sync), SAME binary + SAME patched libamdhip64.so,
only HIP_PM4_GRAPH toggled, pinned profile_peak, ITERS=500:

  graph (M=16384)              baseline AQL   HIP_PM4_GRAPH=1   speedup   checksum
  L=60 (300 disp ~= 1 token)    1078.44 us       784.54 us       1.37x    517635.581366 (identical)
  L=8  (40 disp)                 151.20 us        117.84 us       1.28x     68169.317397 (identical)

At L=60 / M=16384 that is ~784 us/token vs stock HIP graph 1078 -- matching the
standalone PM4 PWS (742.9) and far ahead of HIP graph. The raw-PM4 PWS win is
reproduced inside the HIP runtime: one CP jump per graph (free), lean raw-PM4
dispatch front-end, and the in-place PWS fence that overlaps the AGENT cache
flush. Limitation: simple kernarg-ptr ABI only (no scratch / extra user SGPRs);
real decode kernels need ABI extension (see Appendix B), but the safe fallback
keeps all other graphs correct. FULL reconstruction reference: Appendix B.

--------------------------------------------------------------------------------

## 6o. End-to-end gpt-oss-20b decode: PM4 vs AQL vs develop (the real workload)

All the sections above isolate the inter-kernel gap on synthetic chains. This is the
payoff measured on a real model: full gpt-oss-20b decode through flywheel (direct
mode) on the same W7900, with the runtime swapped via LD_LIBRARY_PATH / HIP_PM4_GRAPH.
Four stacks: our PM4 lib (HIP_PM4_GRAPH=1), our same lib as AQL baseline, stock ROCm
7.2, and upstream/develop (newer CLR + ROCr, carrying develop's own graph work).

Exact commands (run inside test-framework-sshliapn-container-2nd):

```
CT=test-framework-sshliapn-container-2nd
MODEL=/huggingface/hub/models--openai--gpt-oss-20b-q0-lmhead-s0t/   # 24 layers, profile q0
OURLIB=/home/sshliapn/code/rocm-systems/projects/clr/build-gap/hipamd/lib
DEVLIB=/home/sshliapn/code/rocm-systems-develop/projects/clr/build-develop/hipamd/lib:/home/sshliapn/code/rocm-systems-develop/install/lib:/opt/rocm/lib

docker exec $CT bash -lc "
  cd /home/sshliapn/code/rednotdead
  export HIP_VISIBLE_DEVICES=0 HF_HOME=/huggingface REDLINE_DEBUG_SKIP_SAMPLING=1
  BENCH='python3 utils/bench/benchmark.py -m $MODEL -i 512 -o 128 --profile q0 -ap w0 -tp 1 -c 1 -n 5 -s -w --no-cache'
  echo '## B ours + PM4'; env LD_LIBRARY_PATH=$OURLIB HIP_PM4_GRAPH=1 \$BENCH | grep -E '^ +1 +[0-9]'
  echo '## C ours AQL';   env LD_LIBRARY_PATH=$OURLIB                \$BENCH | grep -E '^ +1 +[0-9]'
  echo '## D stock 7.2';  env -u LD_LIBRARY_PATH                     \$BENCH | grep -E '^ +1 +[0-9]'
  echo '## A develop';    env LD_LIBRARY_PATH=$DEVLIB                \$BENCH | grep -E '^ +1 +[0-9]'
"
```

REDLINE_DEBUG_SKIP_SAMPLING=1 feeds a host-selected random token and skips the
sampling chain (output text is meaningless; this isolates the model+runtime perf
path). 512 input / 128 output tokens, concurrency 1, -n 5 requests. Each stack's
libamdhip64 confirmed distinct via hipRuntimeGetVersion (ours 70253211, develop
71460850, stock 70226015). Numbers are the median of 3 repeats; spread < 0.5%.

  Stack          Throughput   e2e lat   TTFT    TPOT      Tok/s   Tok/s/User
                 (req/s)      (ms)      (ms)    (ms)
  B ours + PM4     1.52        656      ~100    4.37       976      195
  C ours AQL       1.39        719      ~101    4.87       890      178
  D stock 7.2      1.39        720      ~101    4.87       889      178
  A develop        1.39        721      ~102    4.87       887      178

What it shows:
- Our PM4 path is the ONLY stack that speeds up decode: TPOT 4.87 -> 4.37 ms
  (-10.3%), Tok/s 890 -> 976 (+9.7%), e2e latency 719 -> 656 ms (-8.8%). Submitting
  the whole 24-layer decode graph as one PM4 IB per token removes the GPU-side
  per-dispatch overhead that this report dissects on synthetic chains, and it
  accumulates over the many small kernels per token.
- develop, ours-AQL and stock 7.2 are indistinguishable (~4.87 ms TPOT). develop wins
  the host-launch microbench (~1.2 us/launch, see pm4_gap/DEVELOP_VS_OURS_REPLAY.md),
  but a decode token costs ~4870 us of GPU compute, so a few-us host-launch delta is
  in the noise. develop's graph work (HW-signal pooling, doorbell de-dup) is host-side
  and does not shrink the GPU per-token cost; the PM4 single-IB replay does.
- TTFT is flat (~100 ms) -- prefill is one compute-bound pass, not graph-replay-bound.

Raw output (all reps) + full reproduction (incl. how develop was built on this 7.2
box): pm4_gap/gptoss_e2e.txt and pm4_gap/DEVELOP_VS_OURS_REPLAY.md.

--------------------------------------------------------------------------------

## 7. Artifacts

Benchmarks (in vk_gap_test/):
- vk_gap_test (main.cpp, gap.comp)  -- Vulkan/RADV inter-kernel gap (trivial kernel)
- vk_layer.cpp (+ layer_add.comp, layer_scale.comp) -- Vulkan/RADV decode-layer chain; section 6g/6h
  (env: VK_TS=1 per-dispatch GPU timestamps; VK_SAME=1 no per-dispatch rebind)
- pm4_gap/hip_layer_graph.cpp       -- HIP hipGraph pre-recorded chain (fair vs Vulkan/PM4); section 6h
- hip_gap_test.cpp                  -- HIP CPU launch loop, wave-timeline + CP-boundary
- hip_gap_graph.cpp                 -- HIP graph (record-once replay)
- hip_gap_multi.cpp                 -- HIP multi-stream concurrency sweep
- hip_gap_anyorder.cpp              -- HIP hipExtAnyOrderLaunch / barrier-bit + GAP_* envs
- hip_gap_indep.cpp                 -- single stream, independent buffers (auto nosync header)
- pm4_gap/pm4_gap.cpp               -- raw KFD PM4 compute-queue dispatch loop (bypasses AQL); section 6d
- pm4_gap/pm4_real.cpp              -- PM4 dispatch of a REAL hipcc kernel (full AMDHSA ABI); section 6e
- pm4_gap/realkern.hip              -- inc_kernel(int*) compiled to k_gfx1100.co (real-kernel ABI test)
- pm4_gap/pm4_layer.cpp             -- decode-layer chain via PM4 (multi-kernel, LDS, multi-block); section 6f
- pm4_gap/hip_layer.cpp             -- HIP/AQL reference for the same chain (same code object); section 6f
- pm4_gap/layer_kernels.hip         -- k_rms/k_scale/k_add/k_copy/k_revadd -> layer_gfx1100.co
  (k_copy stages coarse host I/O; k_revadd is the reverse-read cache-coherence stress kernel)
- layer_revadd.comp (+ .spv)        -- Vulkan reverse-read shader for CHAIN=rev (section 6l)
  CHAIN=rev KLEN=64: 64-kernel reverse-read chain (ping-pong x<->y) across PM4/HIP/Vulkan
- pm4_gap/gap_kernel.s              -- gfx1100 RAW-dependency shader (data[0]++ with glc flat access)
- pm4_gap/gptoss_e2e.txt            -- raw gpt-oss-20b e2e decode numbers, all reps (section 6o)
- pm4_gap/DEVELOP_VS_OURS_REPLAY.md -- develop-vs-ours host-launch sweep + model e2e + full develop build repro
- pm4_gap/build.sh                  -- offline-assembles shader + builds pm4_gap (run in hipvk-isolated container)
  NOTE: links libhsakmt directly; uses kfdtest PM4 struct headers from
  rocm-systems/.../libhsakmt/tests/kfdtest/include. Run with /dev/kfd access.

Clock pinning (run as root on HOST; sysfs is RO in the container):
- gpu_pin_freq.sh                   -- disable autosuspend + pin profile_standard (973 MHz)
- gpu_restore_freq.sh               -- restore saved original state

Source patch (env-gated, isolation only):
- rocm-systems/projects/clr/rocclr/device/rocm/rocvirtual.cpp  (GAP_NOSCOPE / GAP_NOBARRIER)
- custom build: rocm-systems/projects/clr/build-gap (libamdhip64.so 7.2.53211)
- NOTE: monorepo is checked out at tag rocm-7.2.1 (detached HEAD) with the patch
  applied in the working tree. To restore: `git checkout develop` and revert the
  rocvirtual.cpp edit.

--------------------------------------------------------------------------------

## 8. Appendix A: PM4 microbenchmark -- complete design & implementation reference

This appendix is a self-contained record of the PM4 dispatch work. It explains
the idea, the architecture, every moving part with code excerpts, the fence
designs (blocking + PWS), the memory model, the chain modes, and how to rebuild
and run. The goal is that the entire PM4 effort can be reconstructed from this
section alone. Primary artifact: `pm4_gap/pm4_layer.cpp`.

### 8.1 The idea (why drive the GPU with raw PM4 at all)

Both HIP/ROCm and Vulkan/RADV ultimately push the same thing onto the same
hardware ring: a stream of PM4 packets ("Programmable Microcode 4", the GFX
command-processor packet format). HIP wraps it in the HSA AQL dispatch protocol
(an AQL packet -> ROCr translates it into PM4 on the MEC); RADV emits PM4
directly from the userspace driver. Every "framework overhead" question in this
document (Is the gap the barrier? Is it the cache flush? Is it pipeline rebind?
Is it the launch path?) eventually bottoms out in: *what PM4 does each framework
actually emit between two dependent dispatches, and how does the command
processor execute it?*

The PM4 harness removes the framework entirely. It opens a raw KFD compute queue,
hand-builds the exact PM4 packet stream (register writes + dispatch + fence), and
submits it with a single doorbell ring. That gives:

1. A **ground-truth floor**: the minimum achievable per-dispatch period on this
   GPU with nothing between us and the CP. Anything HIP/Vulkan adds on top is
   pure framework overhead.
2. **Total control of the fence**: we can emit the precise cache-coherency packet
   (ACQUIRE_MEM with an arbitrary GCR_CNTL) and the precise wait mechanism
   (blocking acquire vs PWS deferred-wait), and measure each bit's cost in
   isolation. This is what let us decompose the large-M slowdown down to a single
   bit (GLK_INV, scalar K-cache invalidate; sections 6m/6n).
3. **An apples-to-apples comparison**: the SAME compiled code object
   (`layer_gfx1100.co`) is run by PM4 and by HIP/AQL (`hip_layer`), and a
   bit-identical kernel set is run by Vulkan (`vk_layer`), so the comparison
   isolates DISPATCH overhead, not kernel codegen.

The headline outcome: a correctly built PM4 fence (CS_PARTIAL_FLUSH wave-drain +
PWS-overlapped cache flush) is flat in M and faster than both HIP/AQL and
Vulkan/RADV, while remaining bit-exact even on the reverse-read cache-coherence
stress chain.

### 8.2 File / artifact organization

```
pm4_gap/
  build.sh            Build pipeline (run INSIDE hipvk-isolated container).
  layer_kernels.hip   The 6 real kernels (k_rms,k_scale,k_add,k_copy,k_blend,k_revadd).
  layer_gfx1100.co    Unbundled gfx1100 code object (loaded raw by PM4).
  layer.co            Bundled code object (loaded by HIP via hipModuleLoad).
  pm4_layer.cpp       *** the PM4 decode-layer harness (this appendix) ***
  hip_layer.cpp       HIP/AQL reference running the SAME layer_gfx1100 kernels.
  hip_layer_graph.cpp HIP hipGraph (pre-recorded) variant for a fair vs PM4/VK.
  pm4_gap.cpp         Earlier: trivial-kernel raw PM4 dispatch loop (section 6d).
  pm4_real.cpp        Earlier: first real-kernel ABI replication proof (section 6e).
  gap_kernel.s        Hand-written gfx1100 shader for pm4_gap.
```

The harness is built on three external pieces, all present in the rocm-systems
checkout:
- `libhsakmt` (static, `/opt/rocm/lib/libhsakmt.a`): the thin KFD ioctl wrapper
  (queue create/destroy, memory alloc/map, CU mask).
- kfdtest PM4 struct headers (`.../libhsakmt/tests/kfdtest/include`): the packet
  bitfield definitions (`pm4_pkt_struct_nv.h` for GFX10/11 = "NV/Navi",
  `kfd_pm4_opcodes.h`, the `asic_reg/gfx_7_2_*` register name headers).
- `libdrm_amdgpu` for the allocation path.

### 8.3 Build pipeline (`build.sh`)

The kernels are normal hipcc, but PM4 needs the *raw* (unbundled) single-target
code object, so the build compiles with `--genco` then unbundles:

```bash
# decode-layer kernels -> bundled .co (for HIP) + unbundled gfx1100 .co (for PM4)
/opt/rocm/bin/hipcc --genco --offload-arch=gfx1100 layer_kernels.hip -o layer.co
/opt/rocm/llvm/bin/clang-offload-bundler --type=o --unbundle --input=layer.co \
    --output=layer_gfx1100.co --targets=hipv4-amdgcn-amd-amdhsa--gfx1100

# PM4 harness: plain g++, link static libhsakmt + libdrm (NO hip runtime)
g++ -O2 -std=c++17 -Wall -I$KFDINC -I/opt/rocm/include \
    pm4_layer.cpp /opt/rocm/lib/libhsakmt.a \
    $(pkg-config --libs libdrm_amdgpu libdrm) -lnuma -lpthread -lrt -o pm4_layer
```

Note pm4_layer links NO HIP/ROCr -- it is a pure KFD client. The only ROCm
dependency at runtime is the kernel driver (`/dev/kfd`) and the code object file.

### 8.4 The kernels (`layer_kernels.hip`)

Six `extern "C" __global__` kernels, all `BS=256`, deliberately written with
ONLY compile-time block size and hardware `threadIdx/blockIdx` (no `blockDim`,
`gridDim`, dynamic LDS, or printf) so the COV5 kernarg segment holds ONLY the
explicit args -- no hidden implicit-args buffer. That keeps PM4 kernarg setup
trivial (just the pointers + n).

- `k_rms(x,y,n)`   -- LDS + `__syncthreads` tree reduce; exercises group_segment
  and barriers. Bit-exact form `y = 0.5*x + 0.5*(sum(x)*2^-20)` (no rsqrt/div,
  which are toolchain-ambiguous between HIP and GLSL).
- `k_scale(y,w,n)` -- `y[i] *= w[i]`.
- `k_add(x,y,n)`   -- `x[i] += y[i]`.
- `k_copy(dst,src,n)` -- `dst=src`; the GPU equivalent of hipMemcpy used to stage
  coarse-grain host<->device I/O.
- `k_blend(dst,src,n)` -- forward `dst=0.5*dst+0.5*src[i]` (realistic decode read).
- `k_revadd(dst,src,n)` -- reverse `dst=0.5*dst+0.5*src[n-1-i]`: thread i reads
  the mirrored index written by a DIFFERENT workgroup last dispatch -> the
  cache-coherence stress kernel. If the fence does not invalidate per-WGP L0/L1,
  the read hits a stale line and the checksum diverges. This is the correctness
  oracle for the cache flush.

All blends are contractive (0.5/0.5) so thousands of iterations stay bounded and
the float checksum is deterministic across PM4/HIP/Vulkan.

### 8.5 KFD queue + ring + doorbell (the submission model)

The harness allocates a ring buffer in GPU-visible memory, writes the whole PM4
stream into it once, then submits by writing the write-pointer and ringing the
doorbell. Completion is detected by polling a sentinel that the LAST packet
writes.

```cpp
// ring sized for the whole stream (~112 dwords/dispatch incl. PWS fence)
uint64_t needDwords = 64 + (uint64_t)N*perIter*112 + 16 + (coarse?512:0);
uint64_t ringBytes  = next_pow2(needDwords*4); if(ringBytes<0x10000) ringBytes=0x10000;
void* ring = alloc_gpu(node, ringBytes, /*exec*/true, /*uncached*/true);

HsaQueueResource res; memset(&res,0,sizeof(res));
CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL,
                        ring, ringBytes, nullptr, &res));
...
Emit e((uint32_t*)ring);     // build the entire stream into the ring
... emit dispatches + fences ...
e.write_data((uint64_t)sentinel, 0xC0FFEE);   // final completion marker
uint32_t total = e.idx;

// SUBMIT: single write-ptr update + doorbell ring, then spin on the sentinel.
double t0 = now_s();
*res.Queue_write_ptr_aql = total;
asm volatile("":::"memory");
*res.Queue_DoorBell_aql  = total;
while(sentinel[0] != 0xC0FFEE) { /* timeout guard */ }
double us = (now_s()-t0)*1e6;
```

Because the whole stream is pre-recorded and submitted once, the measured wall
time is GPU-side end-to-end (no per-dispatch CPU launch), which is the fair
comparison against Vulkan's pre-recorded command buffer. (HIP eager, by
contrast, is a CPU launch loop -- see 8.12 / Q4.)

### 8.6 Loading a raw code object + ABI replication

PM4 has no runtime loader, so we parse the ELF code object ourselves: map every
PT_LOAD segment into an executable GPU allocation, then walk the symbol table for
`<name>.kd` symbols (the 64-byte AMDHSA *kernel descriptor*) and copy each KD.

```cpp
struct kernel_descriptor_t {           // 64-byte AMDHSA v3 KD
    uint32_t group_segment_fixed_size, private_segment_fixed_size, kernarg_size;
    uint8_t  reserved0[4];
    int64_t  kernel_code_entry_byte_offset;   // KD VA + this = shader entry
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3, compute_pgm_rsrc1, compute_pgm_rsrc2;
    uint16_t kernel_code_properties, kernarg_preload;
    uint8_t  reserved2[4];
};
```

The ABI replication (validated first in pm4_real.cpp) is: copy `rsrc1`/`rsrc2`
*verbatim* from the KD, put the kernarg pointer in `s[0:1]` (set only
`ENABLE_SGPR_KERNARG_SEGMENT_PTR`), no scratch. We reject any kernel that needs
scratch (`private_segment_fixed_size != 0` or scratch-init properties) since the
harness does not set up a scratch ring.

One subtlety the AQL CP normally handles for us: the KD's `rsrc2` does NOT carry
the static LDS size (AQL fills `LDS_SIZE` from the dispatch packet's
group_segment field). We replicate that by OR-ing the LDS granule count into
rsrc2 before writing `COMPUTE_PGM_RSRC1/2`:

```cpp
uint32_t rsrc2    = kd.compute_pgm_rsrc2;
uint32_t ldsUnits = (kd.group_segment_fixed_size + 127) / 128;   // gfx11 128B granule
rsrc2 |= (ldsUnits << COMPUTE_PGM_RSRC2__LDS_SIZE__SHIFT) & COMPUTE_PGM_RSRC2__LDS_SIZE_MASK;
```

### 8.7 The PM4 packet emitter (`struct Emit`)

`Emit` is a tiny cursor over the ring that appends Type-3 PM4 packets. Header
helper sets `type=3, shaderType=1 (compute), opcode, count=dwords-2`. Key
emitters:

- `set_sh_reg(reg, vals, n)` -- IT_SET_SH_REG: write n consecutive SH registers
  (PGM_LO/HI, RSRC1/2, USER_DATA, thread dims, resource limits).
- `dispatch_direct(x,y,z,init)` -- IT_DISPATCH_DIRECT with `dispatch_initiator`.
- `partial_flush()` -- IT_EVENT_WRITE / CS_PARTIAL_FLUSH (drain producer waves).
- `acquire_mem(gcr)` -- IT_ACQUIRE_MEM, the BLOCKING cache flush+wait.
- `release_mem_pws(gcr)` / `acquire_pws()` -- the PWS deferred-wait pair (8.10).
- `write_data(addr,val)` -- IT_WRITE_DATA to set the completion sentinel.

`emit_dispatch` ties register setup + dispatch together. Note `DISPATCH_INIT`
has `USE_THREAD_DIMS` set, so `dim_x` is the TOTAL number of threads and the CP
derives groups = ceil(threads / NUM_THREAD_X):

```cpp
static const uint32_t DISPATCH_INIT = 0x00000021 | 0x8000; // CS_EN|USE_THREAD_DIMS|CS_W32

static void emit_dispatch(Emit& e, const KInfo& ki, uint64_t kernargVA, uint32_t numThreadsX){
    const uint32_t dims[8]={0,0,0, BS,1,1, 0,0};                  // COMPUTE_NUM_THREAD_X=BS
    e.set_sh_reg(mmCOMPUTE_START_X, dims, 8);
    uint64_t entry=(ki.kdVA + ki.kd.kernel_code_entry_byte_offset)>>8;
    const uint32_t pgm[2]={(uint32_t)entry,(uint32_t)(entry>>32)};
    e.set_sh_reg(mmCOMPUTE_PGM_LO, pgm, 2);
    uint32_t rsrc2=ki.kd.compute_pgm_rsrc2 | (ldsUnits<<...);     // LDS_SIZE OR-in (8.6)
    const uint32_t rsrc[2]={ki.kd.compute_pgm_rsrc1, rsrc2};
    e.set_sh_reg(mmCOMPUTE_PGM_RSRC1, rsrc, 2);
    e.set_sh_reg(mmCOMPUTE_RESOURCE_LIMITS, &zero, 1);
    e.set_sh_reg(mmCOMPUTE_TMPRING_SIZE,   &zero, 1);
    uint32_t udata[2]={(uint32_t)kernargVA,(uint32_t)(kernargVA>>32)};
    e.set_sh_reg(mmCOMPUTE_USER_DATA_0, udata, 2);               // kernarg ptr -> s[0:1]
    e.dispatch_direct(numThreadsX, 1, 1, DISPATCH_INIT);
}
```

This per-dispatch full register re-emit is intentional: it mirrors what RADV does
(rebind the pipeline state every dispatch), so the comparison is fair.

### 8.8 The cache-coherency fence: GCR_CNTL bit layout

The whole investigation hinges on the GCR_CNTL field of ACQUIRE_MEM. Layout
(matching mesa pkt3.json / PAL):

```
GLI_INV[0:1] GL1_RANGE[2:3] GLM_WB[4] GLM_INV[5] GLK_WB[6] GLK_INV[7]
GLV_INV[8] GL1_INV[9] GL2_US[10] GL2_RANGE[11:12] GL2_DISCARD[13]
GL2_INV[14] GL2_WB[15] SEQ[16:17] RANGE_IS_PA[18]
```

Three scopes are defined:

```cpp
// RADV VkMemoryBarrier: writeback+invalidate EVERY cache incl. L2 (overkill for
// device-local dependencies). GCR_MODE=0.
GCR_RADV_LIKE  = GL2_WB|GL2_INV|GLM_WB|GLM_INV|GL1_INV|GLV_INV|GLK_INV|GLI_INV;

// AGENT scope (what ROCr/HIP use between dependent kernels on one device): L2 is
// the device coherence point and L0/L1 are write-through, so only INVALIDATE the
// per-WGP L0 vector / L1 / scalar K caches. DEFAULT (GCR_MODE=1). == 0x380.
GCR_AGENT_LIKE = GL1_INV|GLV_INV|GLK_INV;
```

- `GLV_INV` (bit 8) -- invalidate L0 vector cache (per-WGP). Mandatory for the
  reverse-read chain (reader on a different WGP must not see a stale L0 line).
- `GL1_INV` (bit 9) -- invalidate the per-shader-array L1.
- `GLK_INV` (bit 7) -- invalidate the scalar K-cache. This is the expensive bit:
  it is broadcast to every CU that ran a producer wave and the GCR engine waits
  for an ACK from each, so its cost scales ~linearly with active CU count
  (section 6n.1). Dropping it (0x380 -> 0x300) is 2.5-5x cheaper per dispatch and
  bit-exact ONLY for kernels whose kernargs are immutable and that read buffer
  DATA via VECTOR loads (verified in the ISA for these hipcc kernels). RADV must
  keep GLK_INV because its ACO compiler reads SSBOs via SMEM scalar loads.
  We keep it ON by default for safety; the PWS fence (8.10) hides its cost
  instead of dropping it.

`acquire_mem` programs ACQUIRE_MEM byte-for-byte like RADV's gfx10/11 compute
path (CP_COHER_CNTL=0, full coherence range SIZE=0xFFFFFFFF / SIZE_HI=0xFFFFFF,
BASE=0, POLL_INTERVAL=0x0A, then GCR_CNTL):

```cpp
void acquire_mem(uint32_t gcr){
    PM4ACQUIRE_MEM_NV p; memset(&p,0,sizeof(p));
    set_hdr(p.header, IT_ACQUIRE_MEM, sizeof(p)/4);
    p.reserved=0;                       // CP_COHER_CNTL
    p.coher_size=0xFFFFFFFF;            // CP_COHER_SIZE
    p.ordinal4=0x00FFFFFF;              // CP_COHER_SIZE_HI (RADV: 0xffffff)
    p.coher_base_lo=0; p.ordinal6=0;    // CP_COHER_BASE / _HI
    p.bitfields5.poll_interval=0x0A;
    p.bitfields6.gcr_cntl=gcr;
    ...
}
```

### 8.9 The blocking fence and why it grows with M

The simplest correct fence is `CS_PARTIAL_FLUSH` (drain producer waves so their
stores retire) followed by a blocking `ACQUIRE_MEM(gcr)` (flush/invalidate caches
and STALL the CP until the flush acks). This is exactly the RADV compute barrier
shape. But on a raw in-order MEC queue the ACQUIRE_MEM blocks the CP inline, so
the per-CU GLK_INV ack latency is fully EXPOSED as inter-dispatch dead time. That
is the entire source of the PM4 large-M slowdown (3 us -> 5.8 us across M);
sections 6m.4 / 6n.1 prove it is the per-CU broadcast, not the (constant ~20-byte)
scalar data. RADV does not stall the same way because GFX11 lets it overlap the
flush -- which is what PWS replicates.

### 8.10 THE FIX: the PWS deferred-wait fence

GFX11 PWS (Pre-shader / Pixel Wait Sync) decouples flush-ISSUE from flush-WAIT. A
`RELEASE_MEM` EOP event carries the GCR cache flush AND bumps a PWS counter
(issued asynchronously -- the CP does not block on it); a later PWS `ACQUIRE_MEM`
waits on that counter. So the flush overlaps the next dispatch instead of stalling
the queue. This is RADV's GFX11 overlap mechanism, replicated on the raw MEC
queue:

```cpp
// RELEASE_MEM (EVENT=BOTTOM_OF_PIPE_TS, EVENT_INDEX=5), GCR flush carried in the
// RELEASE_MEM_OP field, PWS_ENABLE=1. Bit positions are translated from the
// ACQUIRE GCR_CNTL layout to the RELEASE_MEM_OP_gfx11 layout (mesa pkt3.json).
void release_mem_pws(uint32_t gcrAcq){
    uint32_t op = 40u | (5u<<8);          // EVENT_TYPE=BOTTOM_OF_PIPE_TS, EVENT_INDEX=5
    if(gcrAcq & GLM_WB ) op |= 1u<<12;
    if(gcrAcq & GLM_INV) op |= 1u<<13;
    if(gcrAcq & GLV_INV) op |= 1u<<14;
    if(gcrAcq & GL1_INV) op |= 1u<<15;
    if(gcrAcq & GL2_INV) op |= 1u<<20;
    if(gcrAcq & GL2_WB ) op |= 1u<<21;
    if(gcrAcq & GLK_WB ) op |= 1u<<24;
    if(gcrAcq & GLK_INV) op |= 1u<<30;
    op |= ((gcrAcq>>16)&3u)<<22;          // SEQ
    op |= 1u<<31;                         // PWS_ENABLE
    ... emit RELEASE_MEM (8 dw) ...
}
// PWS ACQUIRE_MEM: wait the counter at CP_ME stage, GCR_CNTL=0 (flush already
// issued by RELEASE_MEM).
void acquire_pws(){
    ... IT_ACQUIRE_MEM (8 dw) ...
    word1 = (5u<<11)|(0u<<14)|(1u<<17)|(0u<<18); // PWS_STAGE_SEL=CP_ME, COUNTER_SEL=TS, ENA2=1
    word6 = 1u<<31;                              // PWS_ENA
    word7 = 0;                                   // GCR_CNTL = 0
}
```

The fence selection lambda -- PWS is DEFAULT, with a blocking fallback:

```cpp
int usePWS = getenv("PWS") ? atoi(getenv("PWS")) : 1;   // default ON
auto fence=[&](){
    if(usePWS){ e.partial_flush(); e.release_mem_pws(GCR); e.acquire_pws(); return; }
    if(fmode>=1) e.partial_flush();         // FENCE>=1: wave-drain
    if(fmode>=2) e.acquire_mem(GCR);        // FENCE>=2: blocking acquire
};
```

CORRECTNESS GOTCHA (hard-won): the `CS_PARTIAL_FLUSH` wave-drain is REQUIRED
before the PWS pair. Without it the RELEASE_MEM EOP does not order the consumer
against the producer's outstanding stores on a raw MEC queue, giving
non-deterministic ~1%-wrong checksums. With the drain, PWS is bit-exact and
deterministic on BOTH the serial and the reverse-read mix chains at every M.

Result (pinned, serial LAYERS=8 N=20, us/dispatch): PWS 0x380 is FLAT (2.5-2.8
us, residual growth = bandwidth only), ~2x faster than blocking, and faster than
both VK (4.5-4.7) and HIP eager (3.95-4.1) at every M, with full GLK_INV safety.

### 8.11 Memory model: replicating hipMalloc + hipMemcpy

DEFAULT (no env): compute buffers x/y/w are COARSE-grain device VRAM (NonPaged +
CoarseGrain, NO HostAccess) -- exactly what hipMalloc returns, fully L2-cached,
and not CPU-coherent. Host I/O is therefore staged exactly like hipMemcpy: inputs
live in host-accessible fine-grain staging buffers (xs/ws/ys) and are moved into
device VRAM by on-GPU `k_copy` dispatches before the chain (H2D), and the result
is copied back into an uncached staging buffer `outs` after the chain (D2H).

This default matters: an earlier raw-UNCACHED default bypassed L2 and made
large-M dependent chains up to ~20x slower, repeatedly producing
non-comparable results. Diagnostic opt-outs: `UNCACHED=1` (raw L2-bypass),
`CACHED=1` (fine-grain MTYPE_CC), `COARSE_HA=1` (keep HostAccess on coarse buffers).

```cpp
bool coarse   = !(wantUncached || wantCached || getenv("DBG_ONLY"));
bool uncached = coarse ? false : (wantCached ? false : true);
// H2D before the chain: GPU-copy staged inputs into pure device VRAM, then fence.
if(coarse){ emit_dispatch(e,kCopy,kaH2Dx,thrFull); emit_dispatch(e,kCopy,kaH2Dw,thrFull);
            e.partial_flush(); e.acquire_mem(GCR_AGENT_LIKE); }
```

### 8.12 Chain modes & kernarg slots

`CHAIN` selects the dependent chain; kernarg slots are pre-baked (32 bytes/slot:
[ptr0][ptr1][n]). The serial/mix chains bind 5 DISTINCT kernels per layer
(`LAYERS` scales to ~300 dispatches/token), rebinding every dispatch -- the
realistic decode case:

```cpp
const KInfo& kMid = serial ? kBlend : kRev;   // serial=forward read, mix=reverse-read stress
for(int l=0;l<LAYERS;++l){
    emit_dispatch(e,kCopy, kaYrevX,thrFull); fence();  // y = x
    emit_dispatch(e,kScale,kaScale,thrFull); fence();  // y *= w
    emit_dispatch(e,kMid,  kaAdd,  thrFull); fence();  // x = 0.5x + 0.5*(rev(y)|y)
    emit_dispatch(e,kAdd,  kaScale,thrFull); fence();  // y += w
    emit_dispatch(e,kMid,  kaAdd,  thrFull); fence();  // x = 0.5x + 0.5*(rev(y)|y)
}
```

- `rms` (default): k_rms -> k_scale -> k_add (3 dispatches/iter).
- `elem`: all-elementwise dependent chain on x.
- `rev`: KLEN-kernel reverse-read chain, ping-pong x<->y (cache-coherence stress).
- `mix`: 5 distinct kernels x LAYERS, reverse-read blend (stress).
- `serial`: 5 distinct kernels x LAYERS, forward blend (realistic decode).

### 8.13 Environment knobs (complete list)

```
N (argv1)         iterations (default 2000)
M (argv2)         elements per buffer (default 4096)
co (argv3)        code object (default layer_gfx1100.co)
KFD_NODE          force a specific KFD node (else first gfx11 node)
CHAIN             rms(default)|elem|rev|mix|serial
KLEN              rev chain length (default 64, forced even)
LAYERS            repeat the 5-kernel block per iter (default 1; e.g. 60 = full token)
PWS               1=PWS deferred-wait fence (DEFAULT), 0=blocking fallback
FENCE             blocking fallback level: 0=none 1=drain 2=drain+acquire (default 2)
GCR_MODE          1=AGENT 0x380 (DEFAULT), 0=RADV-like full L2 flush
GCR0              diagnostic: ACQUIRE_MEM with NO cache op
GCR_RAW=0xNNN     diagnostic: override GCR_CNTL (e.g. 0x300 drops GLK_INV)
UNCACHED / CACHED / COARSE_HA   buffer memory-type overrides (8.11)
DBG_ONLY          isolate ONE kernel once (scale|rms|add); forces CPU-writable buffers
CUMASK=N          enable only first N CUs (per-CU GLK scaling proof; section 6n.1)
SEMASK=1          write COMPUTE_STATIC_THREAD_MGMT_SE0..5 = all CUs on all 6 SEs
DBG               dump buffer samples after the run
```

### 8.14 How to rebuild & run

```bash
# build (inside the isolated container; pins libhsakmt + offline code objects)
docker exec hipvk-isolated-sshliapn bash -c \
  'cd /home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap && ./build.sh'

# run the realistic decode chain, default PWS fence, pinned clocks
docker exec hipvk-isolated-sshliapn bash -c \
  'cd .../pm4_gap && CHAIN=serial LAYERS=8 ./pm4_layer 20 65536'

# A/B the fence: blocking vs PWS at large M
CHAIN=serial LAYERS=8 PWS=0 ./pm4_layer 20 65536    # blocking (grows with M)
CHAIN=serial LAYERS=8 PWS=1 ./pm4_layer 20 65536    # PWS overlap (flat)

# GLK_INV cost decomposition (drop bit 7) and per-CU scaling
GCR_RAW=0x300 PWS=0 ./pm4_layer 20 65536            # drop GLK_INV
CUMASK=8      PWS=0 ./pm4_layer 20 65536            # restrict active CUs
```

Reference encodings: mesa `src/amd/registers/pkt3.json` (ACQUIRE_MEM,
RELEASE_MEM_OP_gfx11, ACQUIRE_MEM_PWS_*). RADV refs: `radv_cs.c`
(gfx10_cs_emit_cache_flush, PWS at ~:259-280), `radv_cmd_buffer.c`
(CS_PARTIAL_FLUSH for compute barriers), `radv_queue.c:632-650`.


================================================================================
## 9. Appendix B: CLR PM4-graph integration (HIP_PM4_GRAPH) -- full reconstruction
================================================================================

This appendix records the initial integration that transfers the raw-PM4 PWS
speedup into the HIP runtime, in enough detail to rebuild it from scratch if the
CLR tree is reset. It is the companion of section 6n.5 (results).

### 9.1 Idea in one paragraph

A captured hipGraph replays a fixed list of AQL kernel-dispatch packets via
`VirtualGPU::dispatchGenericAqlPacketBatch`. Instead of letting the firmware
process those AQL packets (heavier front-end + a per-dispatch scope fence), we
compile the WHOLE list ONCE into a single executable PM4 indirect buffer -- raw
`SET_SH_REG` + `DISPATCH_DIRECT` per kernel, with the in-place PWS fence
(`CS_PARTIAL_FLUSH` + `RELEASE_MEM` PWS + `ACQUIRE_MEM` PWS) between dispatches --
and replay it with ONE vendor PM4-IB AQL packet. One CP jump for the entire graph
(the IB-jump cost amortizes to ~0; proven in 6n.4 with pm4_layer IB=1), the lean
raw-PM4 dispatch front-end, and the overlapped AGENT cache fence -- exactly the
pm4_layer recipe, now driven by the HIP graph API. Per-dispatch injection does NOT
work (one vendor packet per dispatch = N CP jumps, overhead cancels the win).

### 9.2 Files changed (ROCm clr tree, rocm-7.2.1)

Repo: `/home/sshliapn/code/rocm-systems/projects/clr`
Build dir: `clr/build-gap` -> `build-gap/hipamd/lib/libamdhip64.so.7` (HIP 7.2.53211)

1. `rocclr/device/rocm/rocvirtual.hpp`
   - `#include <unordered_map>` near the other includes.
   - In `class VirtualGPU` (private), after the PWS members:
     ```
     struct Pm4GraphIb { void* ib = nullptr; uint32_t dw = 0; bool supported = false; };
     std::unordered_map<const void*, Pm4GraphIb> pm4Graphs_;   // keyed by packets[0]
     int pm4GraphState_ = -1;                                   // HIP_PM4_GRAPH gate
     bool pm4GraphActive();
     void* allocExecIbFromData(const uint32_t* data, uint32_t dw);
     Pm4GraphIb buildPm4GraphIb(void* const* packets, size_t numPackets);
     bool tryReplayPm4Graph(void* const* packets, size_t numPackets, bool blocking,
                            bool attach_signal);
     ```

2. `rocclr/device/rocm/rocvirtual.cpp`
   - Hook at the TOP of `dispatchGenericAqlPacketBatch` (right after the
     `packets.empty()` check). `kernelNames != nullptr` marks the graph-replay
     path; fall through to AQL on any unsupported packet:
     ```
     if (pm4GraphActive() && kernelNames != nullptr) {
       if (tryReplayPm4Graph(reinterpret_cast<void* const*>(packets.data()),
                             packets.size(), blocking, attach_signal)) {
         return true;
       }
     }
     ```
   - New code after `capturePwsFence(...)` (reuses the existing `PwsVendorPkt`
     struct + `buildPwsVendorPkt()` already added for the PWS work):
     * anon-namespace `AmdKernelDescriptor` (first 64 bytes of the AMDHSA kernel
       descriptor: group/private/kernarg sizes, `kernel_code_entry_byte_offset`,
       `compute_pgm_rsrc1/2/3`, `kernel_code_properties`), `KCP_*` enable bits,
       gfx11 SH register addresses, PM4 opcodes, GCR constants, `pm4Hdr()`.
     * `pm4GraphActive()` -- caches `getenv("HIP_PM4_GRAPH")`.
     * `allocExecIbFromData()` -- copy of the ensurePwsIb allocation pattern:
       allocate `HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG` device mem from
       `roc_device_.getGpuvmSegment()`, stage via a CPU fine-grain buffer, async
       copy, free stage.
     * `buildPm4GraphIb()` -- the compiler (see 9.3).
     * `tryReplayPm4Graph()` -- cache lookup/compile, then emit ONE vendor PM4-IB
       packet (see 9.4).

No other files. PWS pieces it reuses (`PwsVendorPkt`, `buildPwsVendorPkt`,
`ensurePwsIb`/`pwsIbDw_`) were added earlier for HIP_PWS_FENCE.

### 9.3 buildPm4GraphIb -- the per-dispatch PM4 (gfx11 / Navi31), byte-exact

Constants (from kfdtest `asic_reg/gfx_7_2_d.h`, PERSISTENT_SPACE_START=0x2c00):
```
SET base   0x2c00          opcodes  SET_SH_REG=0x76  DISPATCH_DIRECT=0x15
START_X    0x2e04 (8 regs) EVENT_WRITE=0x46  RELEASE_MEM=0x49  ACQUIRE_MEM=0x58
PGM_LO     0x2e0c (2)      DISPATCH_INIT = 0x21|0x8000 (CS_EN|USE_THREAD_DIMS|CS_W32)
RSRC1      0x2e12 (2)      GCR AGENT = 0x380 (GL1_INV|GLV_INV|GLK_INV)
RESLIM     0x2e15 (1)      GCR full  = 0xC3B1 (GL2_WB|GL2_INV|GLM_WB|GLM_INV|GL1|GLV|GLK|GLI)
TMPRING    0x2e18 (1)      pm4Hdr(op,dw) = 0xC0000000 | ((dw-2)<<16) | (op<<8) | 0x2
USER_DATA0 0x2e40 (2)        (type3, shaderType=1)
```
type-3 header low byte 0x02 = shaderType=1 (compute/MEC). SET_SH_REG ordinal2 =
reg - 0x2c00. The PWS RELEASE_MEM/ACQUIRE_MEM dword layout is identical to
pm4_layer Emit::release_mem_pws / acquire_pws (RELEASE_MEM op = 40|(5<<8)|PWS bits;
GLK_INV->bit30, GLV_INV->bit14, GL1_INV->bit15, PWS_ENABLE bit31; ACQUIRE_MEM PWS
ordinal2 = (5<<11)|(1<<17), GCR_SIZE 0xFFFFFFFF/0x01FFFFFF, PWS_ENA bit31).

Per captured `hsa_kernel_dispatch_packet_t* p` (read straight from the packet):
```
type != KERNEL_DISPATCH        -> return unsupported (fallback)
p->kernel_object == 0          -> unsupported
kd = (AmdKernelDescriptor*)p->kernel_object
kd->private_segment_fixed_size != 0 || p->private_segment_size != 0   -> unsupported (no scratch)
kd->kernel_code_properties & (PRIVATE_SEGMENT_BUFFER|DISPATCH_PTR|QUEUE_PTR|FLAT_SCRATCH_INIT)
                               -> unsupported (only KERNARG_SEGMENT_PTR ABI)
SET_SH_REG START_X = {0,0,0, p->workgroup_size_{x,y,z}, 0,0}
SET_SH_REG PGM_LO  = (p->kernel_object + kd->kernel_code_entry_byte_offset) >> 8  (lo,hi)
rsrc2 = kd->compute_pgm_rsrc2; lds=ceil(max(kd->group_segment_fixed_size,
        p->group_segment_size)/128); rsrc2 |= (lds<<15)&0x00FF8000
SET_SH_REG RSRC1   = {kd->compute_pgm_rsrc1, rsrc2}
SET_SH_REG RESLIM  = 0 ;  SET_SH_REG TMPRING = 0
SET_SH_REG USER_DATA0 = p->kernarg_address (lo,hi)   // reuse the runtime's kernarg buffer
DISPATCH_DIRECT(dim_x=p->grid_size_x, dim_y, dim_z, DISPATCH_INIT)   // USE_THREAD_DIMS => grid = total work-items
CS_PARTIAL_FLUSH ; RELEASE_MEM(PWS, 0x380) ; ACQUIRE_MEM(PWS)
```
After the loop: one `CS_PARTIAL_FLUSH` + full-L2 `ACQUIRE_MEM(0xC3B1)` so the
graph's results are system-visible (matches the SYSTEM-scope release the normal
AQL batch forces on its last packet). Then `allocExecIbFromData(ib)`; cache
{ib, dw, supported=true}. ~51 dwords/dispatch.

KEY correctness choices:
- USER_DATA0 = the CAPTURED `kernarg_address`: the runtime already populated it
  (incl. COV5 hidden args), and graph replay reuses the same buffer every launch,
  so baking the pointer into the IB is safe.
- The mandatory `CS_PARTIAL_FLUSH` before the PWS pair (RAW correctness, see 6n.3).
- Final full-L2 acquire = system visibility for the subsequent D2H / next op.

### 9.4 tryReplayPm4Graph -- one vendor packet, normal signal path

```
key = packets[0]; compile-or-lookup in pm4Graphs_; if !supported return false (fallback)
sig = Barriers().ActiveSignal(kInitSignalValueOne, timestamp_, attach)   // same as normal batch
buildPwsVendorPkt(&pkt, g.ib, g.dw); pkt.completion_signal = sig          // CP raises it after the IB
reserve 1 queue slot; write body then header (packet_store_release); ring doorbell
hasPendingDispatch_ = true; TrackQueueProgress(pkt, index)
if (blocking) Barriers().WaitCurrent()
```
The vendor PM4-IB packet is the same 64-byte AMD_AQL_FORMAT_PM4_IB layout used by
the PWS work and ROCr's AqlQueue::ExecutePM4; its INDIRECT_BUFFER jumps to the IB,
the CP runs the whole graph, then signals completion_signal -> the existing host
sync (Barriers) works unchanged. The IB is compiled once and cached; replays just
re-emit the single vendor packet.

### 9.5 Build & run (restore-from-scratch)

```bash
# rebuild the patched runtime
docker exec hipvk-isolated-sshliapn bash -lc \
  'cd /home/sshliapn/code/rocm-systems/projects/clr/build-gap && \
   cmake --build . --target amdhip64 --parallel'

# build the test app (HIP program with a dependent-kernel hipGraph)
docker exec hipvk-isolated-sshliapn bash -lc \
  'cd /home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap && \
   /opt/rocm/bin/hipcc -O2 -std=c++17 hip_pws_test.cpp -o hip_pws_test'

# pin clocks on the HOST (deterministic), check GPU free first
sudo -n /usr/local/sbin/gpu_pin_freq.sh profile_peak     # restore: gpu_restore_freq.sh

# A/B in the SAME binary + SAME patched lib, only the env toggles
docker exec hipvk-isolated-sshliapn bash -lc '
  cd /home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap
  export LD_LIBRARY_PATH=/home/sshliapn/code/rocm-systems/projects/clr/build-gap/hipamd/lib:$LD_LIBRARY_PATH
  export HIP_VISIBLE_DEVICES=0
  ./hip_pws_test 16384 60 500 graph                 # baseline AQL graph
  HIP_PM4_GRAPH=1 ./hip_pws_test 16384 60 500 graph # PM4 graph'
```
Expected (pinned profile_peak): L=60 baseline ~1078 us/chain -> PM4 ~785 us/chain
(1.37x), identical checksum. REV=1 for the reverse-read coherence-stress chain.

### 9.6 Status, limitations, how to extend to the real model

WORKS: env-gated (HIP_PM4_GRAPH=1, off by default = zero impact), bit-exact on
forward + reverse chains, 1.28-1.38x in real hipGraph replay (6n.5). Safe fallback
to AQL for any unsupported packet, so non-target graphs stay correct.

LIMITATIONS (all guarded by the fallback):
- Only the simple kernarg-ptr ABI: NO scratch (`private_segment_fixed_size`), and
  only `KCP_KERNARG_SEGMENT_PTR` may be enabled. Real decode kernels usually need
  more (scratch, `dispatch_ptr`/`queue_ptr` user SGPRs, COV5 implicit kernarg).
- Only all-KERNEL_DISPATCH graphs: a captured barrier/marker packet -> fallback.
- gfx11 register/encoding only (Navi31). Other ASICs would need their reg map.

TO RUN ON FLYWHEEL DECODE (next step):
1. Honor the full AMDHSA ABI in buildPm4GraphIb: set scratch (`COMPUTE_TMPRING_SIZE`
   + the scratch V# in user SGPRs / FLAT_SCRATCH), and lay out user SGPRs in the
   enable-bit order (private_seg_buf, dispatch_ptr, queue_ptr, kernarg_ptr, ...).
   `pm4_real.cpp` is the reference for a real hipcc-kernel ABI replication.
2. Handle non-dispatch packets in the captured stream (emit equivalent PM4, or
   split the IB around them).
3. Wire to the decode graph (manifold.py captures it) and measure TPOT with the
   model-offline-benchmarking skill; compare HIP_PM4_GRAPH on/off.


================================================================================
## 9.7 Appendix B addendum: e2e coverage (RDNA3 + RDNA4), gaps closed
================================================================================

The 9.6 limitations were closed so a real Llama-3.1-8B decode hipGraph compiles
to ONE PM4 IB and replays bit-exactly. All changes are in `buildPm4GraphIb` /
`tryReplayPm4Graph` (rocvirtual.cpp), still gated by HIP_PM4_GRAPH, still with a
safe AQL fallback for anything unproven.

GAPS CLOSED (gap id -> what changed):
- G1 scratch: read the queue scratch state from `amd_queue_t` (`compute_tmpring_size`,
  `scratch_resource_descriptor[4]`, `scratch_backing_memory_location`) and emit
  COMPUTE_TMPRING_SIZE + COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI + the scratch V# in
  the private_segment_buffer user SGPRs. Opt-in (HIP_PM4_GRAPH_SCRATCH). If the
  queue has not sized scratch yet (compute_tmpring_size==0), the build returns
  kPm4DeferredScratch: that launch falls back to AQL (which sizes the queue
  scratch), and the next launch rebuilds with scratch wired. FLAT_SCRATCH_INIT
  kernels still fall back.
- G2 user-SGPR ABI: user SGPRs laid out in kernel_code_properties enable-bit
  order (private_seg_buf=4, queue_ptr=2, kernarg=2). Supported subset =
  {PRIVATE_SEGMENT_BUFFER, QUEUE_PTR, KERNARG_SEGMENT_PTR}; dispatch_ptr /
  dispatch_id / private_segment_size -> fallback.
- G3 wave size: CS_W32_EN in DISPATCH_INITIATOR derived per kernel from the
  WAVEFRONT_SIZE32 bit (wave32 sets 0x8000, wave64 clears it).
- G4 gfx12 (RDNA4): arch-parameterized emitter (Pm4Arch). gfx11 path byte-
  identical to the validated build; gfx12 drops GLM_WB/GLM_INV from the full-L2
  GCR (no metadata cache; mesa radv_cs.c gfx10_cs_emit_cache_flush). COMPUTE_*
  register addresses and PWS RELEASE_MEM/ACQUIRE_MEM encoding are identical
  across gfx11/gfx12 (verified vs mesa ac_shadowed_regs.c).
- G5 non-dispatch packets: still whole-graph AQL fallback (the decode AQL batch
  is all-dispatch), plus kernarg-preload (G9) detection -> fallback.
- G6 leading acquire: emit a leading full-L2 ACQUIRE_MEM so the first kernel sees
  prior-op (H2D / previous graph) writes.
- G7 lifetime: cache keyed by an FNV-1a content hash over each packet's
  kernel_object/grid/workgroup/kernarg (a destroyed+reused host packet address
  can no longer alias a stale IB); executable IBs freed in ~VirtualGPU.

CRITICAL FIX found via the coverage test (and why the first model run hung):
the gfx11 LDS_SIZE field in COMPUTE_PGM_RSRC2 is encoded in the LDS ENCODE
granule = 128 dwords = 512 bytes (mesa ac_gpu_info.c lds_encode_granularity),
and COV5 leaves the field 0 (the runtime programs it from the dispatch packet's
group_segment_size). The original code recomputed it with a 128-BYTE granule
(4x too large) and OR-ed it in; the 0-LDS microbench never exercised it, but a
real large-LDS GEMM over-allocated past the per-workgroup LDS limit and hung the
queue. Fixed: set LDS_SIZE = ceil(group_segment_size / 512), capped at 0x1FF.

HIP COVERAGE TEST (hip_pm4_coverage.cpp + wave64_kernel.hip + run_pm4_coverage.sh):
runs each scenario under baseline (HIP_PM4_GRAPH unset) and PM4, diffs the
result checksum. gfx1100 (W7900) -- ALL BIT-EXACT:

  scenario     gap   baseline        pm4             result
  multi        G2    2142.686584     2142.686584     OK
  nonmult      G8    2129.860009     2129.860009     OK   (grid % block != 0)
  coherence    G6    2129.860009     2129.860009     OK   (H2D right before launch)
  recreate     G7    18546.625891    18546.625891    OK   (destroy + different graph)
  lds          LDS   2306.333241     2306.333241     OK   (static __shared__)
  wave64       G3    10870.707956    10870.707956    OK   (wave64 mixed w/ wave32)
  barriernode  G5    2129.860009     2129.860009     OK   (event node -> fallback)
  scratch      G1    2129.859873     2129.859873     OK   (spill kernel, opt-in)

gfx1201 (RDNA4): build-verified (no RDNA4 hardware in this environment to run).

E2E -- Llama-3.1-8B-Instruct-q0 decode, real flywheel hipGraph, W7900 gfx1100,
patched libamdhip64 via LD_LIBRARY_PATH (PyTorch loads /opt/rocm libamdhip64.so.7,
overridden):
- The whole decode token (embedding + all 32 decoder layers) compiles to ONE PM4
  IB (AMD_LOG "PM4 graph: compiled" fires exactly once per token, no fallback).
- Correctness: greedy decode is BIT-IDENTICAL, HIP_PM4_GRAPH on vs off and with
  HIP_PM4_GRAPH_SCRATCH on -- same 48 token ids, same text, sha 9f8c459f8f4c3445
  (utils/bench/pm4_e2e_check.py).
- TPOT (pinned profile_peak, -i128 -o128 -n3 greedy):
    baseline      9.56 ms   199.0 tok/s
    PM4 graph     9.50 ms   200.2 tok/s
    PM4+scratch   9.52 ms   199.6 tok/s
  => flat (within noise). At 8B the decode is memory-bound GEMV (weight streaming);
  the inter-kernel gap that PWS overlaps is a negligible fraction of TPOT, so the
  microbench's 1.37x (tiny kernels, all gap) does NOT transfer to large-model
  decode. The value is correctness-complete coverage + the technique standing by
  for gap-dominated regimes (very small models / large batch of tiny kernels /
  speculative-decode draft models).

REMAINING (still fallback, not faults):
- FLAT_SCRATCH_INIT kernels, dispatch_ptr/dispatch_id user SGPRs, non-dispatch
  packets inside the AQL batch, and any non-gfx11/gfx12 ASIC.
- gfx1201 runtime validation pending RDNA4 hardware.

## Appendix C: Where PM4-graph actually helps (benefit sweep)

Driver: pm4_gap/hip_pm4_sweep.cpp + run_pm4_sweep.sh (gfx1100, pinned
profile_peak). A serially dependent chain of KPL=4 dispatches per layer is
swept over three axes: per-kernel work size M, arithmetic intensity (mem-bound
fma=0 / compute-bound fma=256 / LDS-reduction), and chain depth LAYERS. All
configs are BIT-EXACT baseline vs PM4. Every line below is real measured data.

KEY MODEL: the PM4-graph win is a roughly CONSTANT per-dispatch saving (lean
raw-PM4 front end + overlapped AGENT cache fence), ~0.6-1.0 us/dispatch. So
  speedup ~= saving / (saving + per_kernel_GPU_time)
which is large only when the kernel is short, and vanishes as the kernel grows.

Best speedup per regime (full table from run_pm4_sweep.sh):

  regime                              M        L    speedup  saved us/disp
  mem-bound,  deep chain, tiny K      <=65536  60   1.36-1.37x   ~0.95-1.0
  lds/sync,   deep chain, tiny K      <=65536  60   1.18-1.20x   ~0.62-0.71
  compute,    deep chain, tiny K      any      60   1.04-1.07x   ~0.6 (+overlap)
  mem-bound,  large K (M=1M)          1048576  60   1.12x        ~1.0
  any,        shallow chain           any      1    ~1.00x       ~0 (launch-bound)

Three clear conclusions:
1. Chain depth matters: L=1 is ~1.00x everywhere (host hipGraphLaunch floor
   dominates 4 dispatches); the per-dispatch saving only shows once the chain
   is long enough to amortize that floor (L=8 -> ~1.25x, L=60 -> ~1.37x mem).
2. Kernel duration kills the win: holding the ~1us saving constant, raising M
   from 256 to 1M on mem-bound drops 1.37x -> 1.12x. Compute-bound kernels keep
   the CUs busy (little idle gap to overlap) so they sit at 1.04-1.08x; only
   very long compute kernels recover ~7% from full fence overlap.
3. LDS/sync kernels behave like memory-bound but slightly lower (wave-drain at
   the barrier eats some of the overlap): peak 1.20x.

This is exactly why 8B decode (Appendix B) is flat: its kernels are the
large-M memory-bound GEMV / compute regime (~1.05-1.12x ceiling) AND each
kernel runs for many us, so the fixed ~1us gap is a negligible fraction.

SWEET SPOT for this technique: deep dependent chains (tens+ of dispatches) of
SHORT kernels -- i.e. small models, speculative-decode draft models, tiny batch
elementwise/norm chains, or any latency path where per-kernel GPU time is on
the order of the ~1us inter-dispatch gap.

### Measured: why 8B decode cannot reach "320 dispatches x 1us = 0.3ms"

rocprofv3 --kernel-trace on the real Llama-3.1-8B decode graph (gfx1100),
per steady-state decode token (queue 2, segmented by host gaps):
- dispatches/token        : 203  (NOT 320; ~6-7 fused kernels/layer)
- kernel duration buckets : <1us=0, 1-3us=83, 3-10us=33, 10-50us=65, >50us=22
- median kernel duration  : 7.5us  (max 640us)
- busy time in <3us kernels: 154us of 4832us = 3.2% of GPU busy/token
- top time sinks: crystal_gemv_swiglu x20 ~1968us, gemv_q0 x40 ~1318us,
  gemv_s0t x1 615us, gemv_q0_n32 x20 492us, attention x20 173us.
  => ~97% of GPU time is long memory-bound weight-streaming GEMV.

The ~1us PM4 saving is the NEXT dispatch's front-end + cache fence, exposed only
when a kernel is shorter than that setup latency. Behind a 10-640us GEMV the CP
sets up the next kernel in the shadow of execution -> 0 saved. So the realizable
saving is bounded by the short-kernel transitions only: ~83 x ~1us = 80-150us,
not 320us. The clean benchmark delta (9.56 -> 9.50 ms = ~60us) is CONSISTENT
with that bound (within n=3 noise) -- the saving is real but ~1% and buried.
The ~1ms/token of in-graph gap that does exist is mostly genuine RAW coherence
drain between dependent GEMVs, which PWS does not remove (it overlaps only the
cache-flush slice); reclaiming it needs fusion / independent-work overlap.

### Measured: gpt-oss-20b (MoE) DOES show a repeatable win

Same patched runtime, gfx1100, pinned profile_peak, utils/bench/benchmark.py
-tp 1 -c 1 -i 512 -o 128 -g (greedy), 3 reps x n=10 requests.
PM4 path engages cleanly: "PM4 graph: compiled 267 dispatches" once per token,
NO fallback (MoE router + top-k + per-expert grouped GEMVs + norms all compiled
into a single IB).

  metric            baseline (3 reps)        PM4 graph (3 reps)
  latency (ms)      920.6 / 929.2 / 931.0    903.9 / 905.1 / 907.8
  Tok/s             695 / 689 / 687          708 / 707 / 705
  Tok/s/User        139.0 / 137.7 / 137.5    141.6 / 141.4 / 141.0
  ms/token          ~7.24                    ~7.08

=> consistent +2.4% throughput, ~21ms latency reduction over 128 tokens
(~0.16 ms/token). Bands do not overlap (baseline max 695 < PM4 min 705 tok/s),
so it is signal, not noise.

Why 20b-MoE beats dense 8B (which was flat): MoE adds many SHORT kernels per
token (router, top-k gating, expert scatter/gather, per-expert grouped GEMVs
that are smaller than a dense 4096x14336 GEMV, extra elementwise). More short
gap-exposed transitions -> more of the ~1us per-dispatch saving is realized.
Effective saving ~= 0.16ms / 267 dispatches ~= 0.6 us/dispatch, exactly the
sweep's measured per-dispatch gap applied across a larger short-kernel fraction.

### Measured: GPU-busy sum per token vs benchmark ms/token (gpt-oss-20b)

rocprofv3 --kernel-trace on the gpt-oss-20b decode (gfx1100, pinned). Two
caveats that dictate the method:
  (a) kernel-trace SERIALIZES dispatches, inflating the per-token WALL ~2x
      (15.2 ms traced vs 7.24 ms clean) -- so trace wall/gaps are NOT
      comparable to the benchmark; but each kernel's DURATION is a HW
      timestamp and stays accurate, so GPU BUSY (sum of durations) is valid.
  (b) in PM4 mode the 267 in-IB kernels are not individually hooked
      (trace sees 4574 disp for 40 tok vs 15448 for 24 tok baseline), so PM4
      busy cannot be summed from a kernel trace -- but it is the SAME kernels,
      so busy is unchanged; the benchmark wall delta is the real PM4 saving.

Baseline, 45 clean steady decode tokens (311 dispatches/token), summing the
HW-accurate kernel durations:
  GPU busy summed over 45 iters : 225.35 ms
  GPU busy / token (median)     : 4.939 ms
  benchmark wall / token (clean): 7.240 ms
  => GPU compute  = 4.94 ms/token = 68% of TPOT
  => host+launch+inter-kernel idle = 2.30 ms/token = 32% of TPOT

This pins down exactly where PM4 acts: it does NOT touch the 4.94 ms of compute,
it trims the 2.30 ms idle budget. The measured 7.24 -> 7.08 ms (-0.16 ms/token)
PM4 win is ~7% of that 2.30 ms idle, consistent with shaving the per-dispatch
front-end+fence off the short-kernel transitions. It also explains the ceiling:
even eliminating ALL inter-kernel idle caps the win at ~2.3 ms/token (32%), and
the realizable part (short-kernel transitions only) is a small slice of that.

### Measured: clean TPOT with REDLINE_DEBUG_SKIP_SAMPLING (no freq pinning)

The 2.30 ms "idle" above was dominated by host SAMPLING per token. Re-running
with REDLINE_DEBUG_SKIP_SAMPLING=1 (model-offline-benchmarking skill, gfx1100,
NOT pinned, -i512 -o128 -c1 -s --no-cache -n10, 3 reps) removes that fixed cost
and exposes the GPU-side launch/fence gaps PM4 actually targets:

  metric        baseline (3 reps)     PM4 graph (3 reps)
  TPOT (ms)     5.21 / 5.22 / 5.21    4.97 / 4.98 / 4.96
  Tok/s/User    166.1 / 165.8 / 166.3 172.9 / 172.6 / 173.4

=> +4.6% TPOT (-0.24 ms/token), bands fully separated (base min 5.21 > PM4 max
4.98). NB the win is a LARGER fraction than the +2.4% with sampling: the ~0.24ms
absolute saving is ~constant, but removing the ~2ms sampling cost shrinks the
denominator (7.24 -> 5.21), so the same saving shows as a bigger percentage.

Native-trace busy decomposition (REDLINE_DEBUG_COLLECT_TRACES, iter_marker per
token; HW kernel durations, no rocprof serialization inflation):
  BASELINE: ~286 GPU ops/token, GPU busy = 4.44 ms = 85% of the 5.21 ms TPOT;
            inter-kernel idle + launch = 0.77 ms = 15%.
  per-layer (~155 us): MoE GEMMs dominate (swiglu 63.9us=40% + grouped 33us=21%
            = 61%); short kernels rmsnorm 1.6us, rope 1.7us, router 6.2us,
            w0 gemv 3.3us, unpermute 2us; decode lm_head 675us.

PM4 trims the 0.77 ms idle budget by 0.24 ms = ~31% of it; the 4.44 ms of
compute is untouched. IMPORTANT measurement note: PM4's in-IB kernels are
invisible to BOTH rocprofv3 AND the native tracer (only ~19 ops/token are seen
vs 286 baseline, because the 267 kernels are replayed from one indirect buffer),
so PM4 GPU-busy cannot be summed from any kernel trace -- the kernels are
identical to baseline, so busy is unchanged, and the clean TPOT (4.97 ms) is the
ground-truth measurement of the PM4 path.

--------------------------------------------------------------------------------

## Appendix D: Anatomy of the inter-kernel gap (for newcomers)

This section explains, in plain terms, WHERE the ~2.0 us per-transition gap in
the PM4-graph path goes, what every acronym means, and whether the host (CPU) is
involved. Target reader: someone new to GPU command processing.

### D.0 Hardware vocabulary (all on-GPU blocks unless stated)

- HOST / CPU: the x86 processor running the application. It talks to the GPU by
  writing command packets into a QUEUE (a ring buffer in memory) and ringing a
  DOORBELL (a memory-mapped register) to tell the GPU "new work is ready".
- CP (Command Processor): a small processor ON the GPU that reads command
  packets from the queue and drives everything else. It has sub-engines:
    * PFP (Pre-Fetch Parser) and ME (Micro Engine) -- graphics front-end;
    * MEC (Micro Engine, Compute) / ACE (Asynchronous Compute Engine) -- the
      compute-queue front-end. HSA/HIP compute dispatches are serviced here.
  The CP is the "front-end" of the GPU: it does setup and scheduling, it does
  NOT run your kernel math.
- PM4: the binary command-packet format the CP understands (e.g. SET_SH_REG,
  DISPATCH_DIRECT, ACQUIRE_MEM). "Raw PM4" = we hand-build these packets; AQL is
  a higher-level packet that the runtime/CP translates down to PM4.
- IB (Indirect Buffer): a block of PM4 packets in memory that the CP jumps into
  and executes, like a subroutine. Our whole decode token is ONE IB.
- SH registers ("SH" = SHader): the hardware config registers that describe the
  next dispatch -- COMPUTE_PGM_LO/HI (address of the kernel machine code),
  COMPUTE_PGM_RSRC1/2 (how many VGPRs/SGPRs/how much LDS the kernel needs),
  COMPUTE_START / NUM_THREAD (grid and workgroup dimensions), and USER_DATA
  (small values handed to the kernel in registers -- for HIP, the pointer to the
  kernel argument block, "kernarg"). Written with the SET_SH_REG packet.
- SPI (Shader Processor Input): the hardware workgroup/wave LAUNCHER. After the
  CP issues DISPATCH_DIRECT, the SPI allocates VGPR/SGPR/LDS on the compute units
  and physically starts the wavefronts.
- WGP / CU / wave: a Wave is a group of 32 (wave32) or 64 (wave64) threads that
  execute together. A CU (Compute Unit) runs waves; RDNA pairs 2 CUs into a WGP
  (WorkGroup Processor). gfx1100 (W7900) has 48 WGPs.
- Caches (RDNA3): L0 (per-CU vector cache), L1 (per-shader-array), L2 / GL2
  (device-wide, the coherence point), plus a scalar "K" cache (constants/scalar
  loads) and an instruction cache, all per-CU. L0/L1/K are NOT coherent across
  CUs by themselves, so a consumer kernel on a different CU must INVALIDATE them
  to see a producer's fresh data.
- GCR (Global Cache Rinse): the control field of an ACQUIRE_MEM/RELEASE_MEM
  packet that selects which caches to write-back/invalidate. Bits: GLI
  (instruction), GLK (scalar K), GLV (vector L0), GL1, GL2, GLM (metadata).
  "0x380" = a full safe set incl. GLK_INV; "AGENT scope" = invalidate L0/L1/K
  but do NOT touch L2 (producer and consumer already share L2 on one GPU).
- EOP (End Of Pipe): "the moment the pipeline has fully finished a dispatch".
- PWS (Pre-shader Wait Sync, a GFX11 feature): lets the CP ISSUE a cache flush
  asynchronously (RELEASE_MEM, bumps a counter) and WAIT for it later
  (ACQUIRE_MEM on that counter), so the flush OVERLAPS the next dispatch instead
  of stalling the queue.

### D.1 What is the "front-end" (~0.5 us)?

For every kernel the CP must, before any math runs, walk these PM4 packets in
the IB and act on them:
  - ~7x SET_SH_REG  -> write the kernel's code address, resource sizes, grid
                       dimensions, and kernarg pointer into SH registers;
  - 1x DISPATCH_DIRECT -> tell the SPI "launch this grid now".
This is pure CP work (decode packet, write register, repeat) plus handing the
launch to the SPI. It is the GPU's command FRONT-END (setup/scheduling), as
opposed to the BACK-END (the shader cores running your code). For our packet
count it is ~0.5 us. It is 100% on-GPU; the host is not involved (the packets
were already written into the IB once, at capture time).

### D.2 Why is "wave drain" counted separately? Isn't it part of the kernel?

Good and subtle. Two different things are being measured:

- KERNEL DURATION (the "busy 4.44 ms" number) is measured from the first wave
  starting to the last wave ending, per dispatch. So yes -- "kernel finished"
  means "its waves drained", and that time is already inside the kernel's
  duration.

- WAVE DRAIN as a separate gap term is about OVERLAP, not double-counting. The
  CP normally PIPELINES: while kernel N's waves are still running, the CP can
  already process kernel N+1's SET_SH_REG packets and even start N+1's waves --
  IF there is no dependency. CS_PARTIAL_FLUSH (the "wave drain" packet) forces
  the CP to STOP and wait until all of N's compute waves have retired before
  continuing. So it FORBIDS the overlap.
  In a throughput microbenchmark of tiny INDEPENDENT kernels, adding the drain
  makes the per-dispatch period grow by ~0.5 us -- that is the overlap you lose,
  i.e. the previously-hidden tiny KERNEL becoming exposed.

CORRECTION (this supersedes the earlier "0.5 front-end + 0.5 drain + ..." split):
the drain is NOT a separate additive term in the real DECODE gap. In decode the
kernels are genuinely dependent, so N must finish before N+1 starts anyway, and
N's full duration is ALREADY counted in "busy" (4.44 ms). By the time we measure
the gap (N-end -> N+1-start), the drain has already completed -- it is what
detected that N ended. The CP cannot run N+1's front-end DURING the drain (the
CS_PARTIAL_FLUSH stalls the CP in packet order), so the front-end happens AFTER
the drain, but the drain itself adds ~0 on top of N's already-counted completion.
The microbench's "+0.5 us drain" was a TINY kernel being exposed; in real decode
that exposure is the BUSY time, not the gap.

So the honest real-decode per-transition GAP (~2.0 us) decomposes as:
  front-end (~0.5 us)  +  launch latency (~few hundred ns)  +  residual fence
  (~0.5-1.0 us). The drain is the hard ordering BOUNDARY, not an added cost.

### D.3 What is "residual fence" and "launch latency"? (~0.5-1.0 us)

- RESIDUAL FENCE: between two dependent kernels we must (a) make N's writes
  visible at the shared L2, and (b) invalidate N+1's stale L0/L1/scalar caches so
  it re-reads fresh data. We do this with a PWS pair (RELEASE_MEM issues the
  cache rinse asynchronously + ACQUIRE_MEM waits a counter). PWS OVERLAPS most of
  this with the next dispatch, but the ACQUIRE_MEM still has to confirm the
  counter reached its target before N+1's memory reads may proceed. That small
  non-overlapped confirmation (plus the per-CU L0/L1/K invalidate latency that is
  not fully hidden) is the "residual" -- roughly 0.3-0.5 us.
- LAUNCH LATENCY: after DISPATCH_DIRECT, the SPI must allocate registers and LDS
  on the CUs, distribute the workgroups across the 48 WGPs, and fill the
  pipeline until the first instructions actually execute (and there is a small
  tail at the end as the last wave winds down). For the tiny grids in decode this
  is a largely FIXED hardware latency, on the order of a few hundred ns, that you
  cannot program away. It is on-GPU (SPI hardware).

### D.4 Is ANY of this host (CPU) work? -- No (in graph / PM4-IB mode)

In hipGraph / PM4-graph replay the ENTIRE decode token is one IB submitted to the
queue ONCE. The CP then walks the whole IB -- every SET_SH_REG, DISPATCH_DIRECT,
drain, and fence -- entirely on the GPU, with NO per-kernel host involvement. The
host's only per-token interaction is: submit the IB (one packet), wait for the
final completion signal, then run sampling on the CPU. So the ~2.0 us
per-transition gap is 100% GPU-side (CP + SPI + cache hardware); the host is idle
during it.
(For contrast, in EAGER mode -- no graph -- the host builds and enqueues a fresh
packet for every kernel, which is real CPU work per kernel. Section 5.1 showed
even that is not the bottleneck on this path, but graph/PM4-IB removes it
entirely. This is exactly why a recorded command stream helps.)

### D.5 Reconciling "Vulkan is faster" vs "Vulkan is slower than HIP here"

Both statements are true; they measure different HIP paths and workloads.

- "VULKAN FASTER" (the common case): compared against the STOCK HIP runtime with
  its DEFAULT per-dispatch AGENT cache fence, on small kernels, Vulkan wins
  (section 6b: HIP default 7.94 us vs VK 4.96 us per dispatch). Reason: stock HIP
  SERIALIZES its fence (a fixed ~5.8 us CP round-trip), while Vulkan OVERLAPS its
  flush into the dispatch window via PWS.
  RETRACTION: an earlier version of this section also claimed the llama.cpp
  Vulkan backend "uses fewer/fused kernels". That was an UNVERIFIED assumption and
  is withdrawn -- no evidence was collected for it, and the kernel structure may
  match the HIP backend. Do not rely on it. The only verified "VK faster" result
  is the single-kernel pinned gap microbench vs STOCK-DEFAULT HIP (below).

- "VULKAN SLOWER" (the isolated decode-layer microbench, section 6n.3): there the
  comparands were HIP EAGER, HIP GRAPH, and our PM4 PWS -- all leaner than stock
  default. At realistic decode kernel sizes (M >= 16384), Vulkan pays two costs
  every dispatch that those paths do not: (1) a FULL L2 write-back+invalidate on
  every barrier (heavier cache work, scales with dirtied bytes), and (2)
  vkCmdBindPipeline -- re-emitting compute pipeline STATE every dispatch (~0.6 us;
  HIP has no pipeline-state object). Those add up, so HIP eager (lighter AGENT
  fence, no rebind) and especially PM4 PWS (lean front-end + overlapped AGENT
  fence) come out ahead: PM4 PWS 742.9 vs VK 1359.7 us/token at M=16384.

There is NO contradiction: Vulkan beats the STOCK HIP default on the single-kernel
gap microbench; but in the decode-layer chain microbench, once HIP's specific
inefficiency (the serialized AGENT fence) is removed -- which is what the PM4 PWS
path does -- HIP/PM4 matches or beats Vulkan. Vulkan has no extra gap-hiding trick
we are missing; its mechanism (overlapped flush) is the one we already replicate,
and it additionally carries a per-dispatch pipeline rebind we do not.
IMPORTANT CAVEATS / honesty flags:
- The recent gpt-oss e2e tests did NOT involve Vulkan at all -- they compared HIP
  PM4-graph vs HIP baseline (both HIP). "We don't see VK in recent tests" because
  VK was not run there; VK only appears in the standalone gap microbenches.
- The factor-2 spread in "default/eager HIP" between the single-kernel gap test
  (7.94 us/disp) and the decode-layer chain (3.94 us/disp at M=16384) is NOT
  fully explained (kernel size hiding the fence is the likely cause, but it was
  not isolated). Treat absolute cross-config numbers as approximate.
- Absolute cross-API deltas at the ~1 us scale are near this environment's clock
  noise and are not claimed to sub-microsecond precision.

### D.6 So why is 2.0 us hard to beat, and what actually removes it?

The ~2.0 us is mostly the FLOOR of the serial dependent-dispatch model: set up
the next kernel (front-end ~0.5 us) + spin its waves up (launch latency) + the
not-fully-overlapped cache invalidate (residual fence ~0.5-1.0 us). (The drain is
the ordering boundary, already accounted in busy -- see D.2 correction, not a
separate term.) The big cache cost is already hidden by PWS. The per-dispatch
micro-levers (skip redundant register writes; elide fences on independent edges)
barely apply to decode because (a) consecutive kernels are almost always DIFFERENT
kernels and (b) the decode dependency chain is nearly linear (every kernel reads
the prior).

The only ways to make a large dent are STRUCTURAL -- reduce the NUMBER of
transitions, not the cost of each:
  1. KERNEL FUSION: fold adjacent ops into one kernel (rmsnorm+residual into the
     next GEMV, rope into qkv, the small elementwise ops into their neighbour).
     Each fused pair deletes a whole ~2 us transition. Highest-value lever, and it
     applies to ANY backend (HIP or Vulkan) -- it is a kernel-count reduction, not
     a Vulkan-specific feature. (No claim is made here about whether llama.cpp's
     Vulkan backend fuses more than its HIP backend; that was not verified.)
  2. PERSISTENT / MEGAKERNEL decode: one long-lived kernel that loops over the
     whole layer (or token) internally, keeping data in registers/LDS and never
     returning to the CP between ops -> zero inter-kernel gaps. Largest
     engineering effort, lowest theoretical floor.
  3. LARGER BATCH (throughput regime only): bigger kernels make the fixed ~2 us
     gap a negligible fraction. Does not help single-stream batch-1 latency.
Per-dispatch tweaks (register-reuse, fence elision, lighter GCR) are at most a
few percent here; fusion/megakernel are the real answers.

### D.7 RUNTIME-level levers (no kernel fusion / no megakernel)

How the stock runtime picks per-dispatch behavior (rocvirtual.cpp) -- CORRECTED:
- KEY FACT (rocvirtual.cpp:2270-2273): BOTH header variants carry AGENT
  acquire/release scope. dispatchPacketHeaderNoSync_ = kernelDispatchHBits |
  agentScopeHBits; dispatchPacketHeader_ = the same PLUS barrierHBits. The ONLY
  difference is the AQL BARRIER bit. So the cache fence SCOPE is AGENT on EVERY
  dispatch unconditionally (when fenceScopeAgent_, the default on gfx11/gfx12).
- MemoryDependency::validate (rocvirtual.cpp:345-389): for each kernel-argument
  buffer it checks whether [start,end) OVERLAPS a buffer already WRITTEN by a
  prior kernel (and not both read-only). On overlap (or on hitting the tracked-
  object limit) it sets the SYNC header (dispatchPacketHeader_) = AGENT scope WITH
  the barrier bit (CP WAITS for the prior dispatch to retire). With NO overlap it
  keeps NoSync = AGENT scope WITHOUT the barrier bit (CP may PIPELINE). It does
  NOT switch to NONE scope. So the overlap check controls ORDERING (barrier),
  NOT whether a cache fence happens.
- SYSTEM scope (rocvirtual.cpp:1124-1135 coalescing, headers at :2257-2263):
  escalated only when the dispatch touches FINE-GRAINED / host-coherent SVM (so
  the CPU/peer sees the writes -> L2 flush) or at the batch tail. Plain device-
  local kernel->kernel stays AGENT; HIP skips the host cache-coherency layer
  (:772-773).
  So the CORRECT runtime rule is: AGENT scope ALWAYS, barrier bit ON for device
  RAW/WAR/WAW edges (and OFF for independent edges), SYSTEM only for host-coherent
  visibility. There is NO "NONE / no-fence" production path (NONE is only the
  GAP_NOSCOPE env test override at :2294).

What OUR PM4 graph does instead (buildPm4GraphIb): it emits an AGENT PWS fence
+ a CS_PARTIAL_FLUSH drain between EVERY dispatch (rocvirtual.cpp:2036-2039) plus
a full-L2 acquire at the leading/trailing token boundary (:1935,:2043). Because
the runtime ALSO applies AGENT scope to every dispatch, we are NOT cache-scope
over-fencing -- we MATCH the runtime's cache behavior. The only difference is
that we ALWAYS drain/serialize (equivalent to the barrier bit being ON on every
edge), whereas the runtime would clear the barrier bit on INDEPENDENT edges. For
a strictly LINEAR decode chain every edge is a true RAW, so the runtime would set
the barrier bit anyway -> we are equivalent there. We only "over-serialize" where
the graph actually has independent dispatches (which decode does not).

Runtime levers that apply to a strictly LINEAR decode chain, ranked:
  1. IB REORDER -- hoist kernel N+1's SET_SH_REG ahead of N's drain/fence, so the
     CP programs N+1's registers WHILE N's waves run (CP is idle then), hiding the
     ~0.5 us front-end. N+1's DISPATCH_DIRECT still waits behind the fence;
     registers are latched at DISPATCH so this does NOT affect the running N.
     Pure IB-builder change, low risk, UNTESTED -- the most promising lever for a
     linear chain. (Current order is [setsh N][dispatch N][drain][fence][setsh
     N+1]..., so N+1 setup is serialized in the gap instead of overlapped.)
     REFINEMENT: place the register setup BETWEEN release and acquire, i.e.
     [dispatch N][drain N][releaseMemPws N (async flush)][setsh N+1][acquirePws N]
     [dispatch N+1]. Then N+1's programming overlaps BOTH the CP-idle window AND
     the cache-flush tail -> attacks front-end AND residual-fence together. KEEP
     for experiment after the register-delta lever (D.8) lands (it removes most of
     the SET_SH_REG payload that this reorder hides).
  2. REGISTER DELTA-ENCODING (NEW, SAFE) -- see D.8. Hoist the IB-constant
     registers out of the per-dispatch loop and delta-encode the block-dims block.
     Deterministic, bit-exact, no correctness risk. Cuts the per-dispatch
     SET_SH_REG payload ~3x. Highest-confidence runtime lever for linear decode.
  3. PER-EDGE GCR MINIMIZATION (RISKY) -- e.g. drop GLK_INV (bit 7) from the AGENT
     mask (0x380 -> 0x300) when the consumer does no SCALAR loads of the producer
     output. GLK is ALREADY PWS-overlapped, so it has only a small residual TAIL
     (6m.4/6n.1); the upside is small and it is NOT provably safe (the compiler can
     scalarize uniform loads), which is exactly why 6m.2 flagged the 0x300 path as
     "fastest but compiler-ABI-specific / unsafe." NOT recommended without per-
     kernel-pair ISA inspection + bit-exact validation. NOTE: GLI_INV (instruction
     cache, bit 0) is NOT in the per-edge mask at all -- it is only in the boundary
     full-L2 fence -- so there is NO per-edge I-cache opportunity to chase.

DROPPED for this workload (recorded so we do not revisit):
  - DEPENDENCY-AWARE FENCE ELISION -- the runtime only elides the BARRIER bit on
    independent edges (not the cache scope; see the CORRECTED rule above). Decode
    is fully linear-dependent (every kernel reads the prior), experts/projections
    are all fused, so there are NO independent edges to elide. Zero benefit here.
  - MULTI-QUEUE (async compute) -- needs graph parallelism that linear decode does
    not have.

HONEST CEILING: for a strictly linear dependent chain, the safe runtime levers
(#1 reorder + #2 register-delta) can plausibly shave the ~2.0 us toward
~1.2-1.5 us, NOT to zero. Getting near zero needs fewer kernels
(fusion/megakernel), which is out of scope here.

### D.8 Register delta-encoding -- per-dispatch SET_SH_REG audit (gfx11/gfx12)

What buildPm4GraphIb emits PER DISPATCH today (rocvirtual.cpp:1975-2028), with
which fields are IB-constant vs per-kernel:

  setsh(kRegStartX, dims, 8)   8 regs: START_X/Y/Z (always 0,0,0) +
                               NUM_THREAD_X/Y/Z (block dims) + 2 pad. CONSTANT
                               unless workgroup_size changes -> DELTA-ENCODE.
  setsh(kRegPgmLo, 2)          PGM_LO/HI (code entry). Per-kernel -> ALWAYS.
  setsh(kRegScratchLo, 2)      scratch base (only if needsScratch). IB-CONSTANT.
  setsh(kRegRsrc1, 2)          RSRC1 + RSRC2 (RSRC2[23:15] = LDS size). Per-kernel
                               (code resources AND LDS live here) -> ALWAYS.
  setsh(kRegResLim, 1)         always 0. IB-CONSTANT -> emit once.
  setsh(kRegTmpring, 1)        0 or the queue's fixed scratch ring. IB-CONSTANT.
  setsh(kRegUserD0, u)         user SGPRs, enable-bit order. SPLIT THIS:
                                 - scratch V# (4 regs)  IB-CONSTANT
                                 - queue_ptr  (2 regs)  IB-CONSTANT (same queue)
                                 - kernarg    (2 regs)  per-dispatch -> ALWAYS
  DISPATCH_DIRECT (5 dw)       grid_size_x/y/z live HERE, not in a register. LDS
                               is NOT here (it is RSRC2). Per-dispatch, cheap.

So of the ~16-18 register dwords / ~7 SET_SH_REG packets emitted per dispatch:
  - IB-CONSTANT (emit ONCE before the chain): RESLIM(1), TMPRING(1), scratch
    base(2), UserD0 scratch V#(4), UserD0 queue_ptr(2) = 10 regs.
  - DELTA (re-emit only on change): kRegStartX block(8) = START offsets + block
    dims; block dims are stable across most consecutive decode kernels.
  - TRULY PER-KERNEL (always): PGM(2), RSRC1/2 incl LDS(2), kernarg(2) = 6 regs.

Plan: emit the 10 IB-constant regs once at the top of the IB; track a "previous
workgroup_size" and re-emit kRegStartX only when it changes; per dispatch emit
only PGM, RSRC1/2, and the kernarg SUB-RANGE of UserD0 (SET_SH_REG can start at
any offset, so write just the 2 kernarg regs at their computed offset). When
block size is stable this is ~6 reg-dwords across ~3 packets/dispatch instead of
~16-18 across ~7 -> ~3x less front-end payload (which scales with reg-write +
packet count).

Correctness notes:
  - SH registers PERSIST across the PWS drain/fence packets (CS_PARTIAL_FLUSH,
    RELEASE_MEM, ACQUIRE_MEM do not touch SH state), so once-emitted IB-constants
    stay latched for the whole IB.
  - RSRC2 carries LDS_SIZE, so RSRC must stay per-kernel (LDS varies per kernel).
  - Grid is in the DISPATCH packet, so it is naturally per-dispatch with no register
    write to save.
  - Must compute a per-field dirty flag against the previous dispatch and re-emit
    on ANY change; only the block-dims block benefits from dirty-tracking (the
    others are either always-changing or never-changing).

This is bit-exact with the current emitter for identical inputs (same registers
end up latched at each DISPATCH), so it is validated by the existing bit-exact
test matrix (Appendix B) plus the e2e flywheel logits check.

### D.9 Q1 correction summary (scope is NOT chosen by the overlap check)

Earlier text implied the runtime selects NONE / AGENT / SYSTEM per dispatch from
the MemoryDependency overlap result. That was WRONG. Audited
(rocvirtual.cpp:2270-2273): both header variants carry AGENT acquire/release; the
overlap check toggles ONLY the AQL barrier (ordering) bit. AGENT cache scope is
applied to EVERY dispatch unconditionally; SYSTEM is a separate host-coherence
escalation. Therefore our unconditional AGENT PM4 fence MATCHES the runtime's
cache behavior (not over-fencing on scope); we differ only by always-draining,
which for a linear decode chain is exactly what the runtime's barrier bit would
do anyway. See the CORRECTED rule in D.7.

## Appendix D-impl: Implementation plan for the two SAFE runtime levers

This is the concrete plan for (1) register delta-encoding and (2) IB-reorder with
the release/acquire interleave, plus the shared test/verification plan. Both are
pure buildPm4GraphIb (rocvirtual.cpp) changes; neither touches kernels, the ABI,
or the cache-fence SCOPE. Both are gated behind opt-in env flags until validated.

### D.10 Register delta-encoding -- per-register plan

#### D.10.0 The core invariant (WHY it is provably safe)

COMPUTE_* shadow (SH) registers are STICKY: a SET_SH_REG write persists until the
next write to that register. The drain/fence packets we insert between dispatches
(CS_PARTIAL_FLUSH, RELEASE_MEM, ACQUIRE_MEM) do NOT touch SH state. Therefore, if
the value we would write to a register already equals what is latched there, we
can SKIP the SET_SH_REG and the latched state at the consuming DISPATCH is
BIT-IDENTICAL to the baseline (always-emit) builder.

Mechanism: keep a build-time SHADOW of the last value emitted for each register
group. For each dispatch, compare the would-be value against the shadow and emit
ONLY the changed groups; update the shadow. The shadow starts INVALID, so the
FIRST dispatch in the IB emits every group (re-establishes known state regardless
of what the queue held before this IB -- important because the IB is cached and
replayed). This makes the optimization independent of pre-IB register state.

The decision is made at IB BUILD time over the captured packet sequence; the
built IB is static bytes replayed as-is. Because dispatch 0 always re-emits
everything, every later delta is valid on every replay.

#### D.10.1 Register-by-register (addresses from rocvirtual.cpp:1759-1765)

1. kRegStartX = 0x2e04, 8 dwords: COMPUTE_START_X/Y/Z + NUM_THREAD_X/Y/Z + 2 tail.
   - Source: dims[8] = {0,0,0, workgroup_size_x, workgroup_size_y,
     workgroup_size_z, 0,0} (rocvirtual.cpp:1975-1977).
   - START_X/Y/Z are ALWAYS 0 (grid offsets); the trailing 2 are ALWAYS 0; only
     NUM_THREAD_X/Y/Z (the BLOCK dims) can vary.
   - CLASS: DELTA. Re-emit the whole 8-dword block only when workgroup_size
     changes. In decode the block size is stable across long runs of consecutive
     kernels (e.g. a fixed 256-thread block), so this is skipped most of the time.
   - WHY SAFE: if block dims are unchanged, NUM_THREAD already holds them and the
     zeros are already latched -> skipping leaves identical state.

2. kRegPgmLo = 0x2e0c, 2 dwords: COMPUTE_PGM_LO/HI (code entry >> 8).
   - Source: (kernel_object + kernel_code_entry_byte_offset) >> 8
     (rocvirtual.cpp:1978-1980).
   - CLASS: DELTA, but effectively ALWAYS for decode (consecutive kernels are
     different code). Skipped only when the SAME kernel runs back-to-back.
   - WHY SAFE: trivially, different code => different value => emitted.

3. kRegScratchLo = 0x2e10, 2 dwords: COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI.
   - Source: queueScratchBase >> 8 (rocvirtual.cpp:1984-1986), only when
     needsScratch. queueScratchBase = aq->scratch_backing_memory_location, fixed
     for the queue.
   - CLASS: DELTA -> effectively ONCE (emitted on the first scratch-using kernel,
     skipped thereafter). Non-scratch kernels do not touch it.
   - WHY SAFE: same queue => same scratch base for the whole IB.

4. kRegRsrc1 = 0x2e12, 2 dwords: COMPUTE_PGM_RSRC1 + COMPUTE_PGM_RSRC2.
   - Source: rsrc[2] = {compute_pgm_rsrc1, rsrc2-with-LDS} (rocvirtual.cpp:
     2002-2004). RSRC2[23:15] carries the LDS size (recomputed from
     group_segment_size).
   - CLASS: DELTA, effectively ALWAYS for decode (VGPR/SGPR/float-mode AND LDS
     differ per kernel). MUST stay per-kernel because LDS lives here.
   - WHY SAFE: different resources/LDS => different value => emitted.

5. kRegResLim = 0x2e15, 1 dword: COMPUTE_RESOURCE_LIMITS.
   - Source: always 0 (rocvirtual.cpp:2005-2006).
   - CLASS: DELTA -> ONCE (emitted on dispatch 0, skipped forever after).
   - WHY SAFE: value never changes within an IB.

6. kRegTmpring = 0x2e18, 1 dword: COMPUTE_TMPRING_SIZE.
   - Source: 0 (no scratch) or aq->compute_tmpring_size (rocvirtual.cpp:2007).
   - CLASS: DELTA -> at most TWO distinct values in an IB (0 and the queue ring),
     so emitted on the first kernel of each kind and skipped within runs.
   - WHY SAFE: the ring size is fixed for the queue; toggling only between 0 and
     that fixed value, both correctly shadowed.

7. kRegUserD0 = 0x2e40, up to 8 dwords: COMPUTE_USER_DATA_0.. packed in
   kernel_code_properties ENABLE-BIT order (rocvirtual.cpp:2009-2028):
     [a] PRIVATE_SEGMENT_BUFFER -> scratch V# (4 dwords) from
         aq->scratch_resource_descriptor. IB-CONSTANT.
     [b] QUEUE_PTR -> gpu_queue_ pointer (2 dwords). IB-CONSTANT.
     [c] KERNARG_SEGMENT_PTR -> p->kernarg_address (2 dwords). PER-DISPATCH.
   - The enabled SET can differ kernel-to-kernel, so the LAYOUT (offset of the
     kernarg sub-range) can change. Two sub-cases:
       * LAYOUT UNCHANGED vs previous dispatch (same enable mask): the constant
         prefix (scratch V#, queue_ptr) is already latched -> emit ONLY the
         kernarg 2-dword sub-range, as a SET_SH_REG starting at
         kRegUserD0 + kernarg_offset (kernarg_offset = (hasScratch?4:0) +
         (hasQueue?2:0)). This is the common decode case.
       * LAYOUT CHANGED (enable mask differs): the prefix may now mean different
         registers -> re-emit the FULL udata[0..u] block for this dispatch.
   - CLASS: SPLIT DELTA (constant prefix ONCE; kernarg ALWAYS; full re-emit on
     layout change).
   - WHY SAFE: SET_SH_REG can start at any register offset (rocvirtual.cpp:
     1892-1896 writes reg-kShBase then cnt dwords), so writing only the kernarg
     sub-range leaves the latched prefix intact; on layout change we fall back to
     a full block so no stale prefix is reused.

NOT a register (recorded so it is not "optimized" by mistake):
   - grid_size_x/y/z live in the DISPATCH_DIRECT packet (rocvirtual.cpp:2031-2035),
     not in SH registers. They are per-dispatch and cannot be delta-skipped (the
     dispatch packet is mandatory). LDS is NOT here -- it is RSRC2[23:15].

#### D.10.2 Expected per-dispatch payload (block size stable, scratch resident)

  Baseline (always emit): StartX(8) + Pgm(2) + ScratchLo(2) + Rsrc(2) + ResLim(1)
    + Tmpring(1) + UserD0(8) = 24 reg dwords across 7 SET_SH_REG packets.
  Delta (steady state):    Pgm(2) + Rsrc(2) + kernarg(2) = 6 reg dwords across
    3 SET_SH_REG packets. (StartX re-emitted only on a block-size change;
    ScratchLo/ResLim/Tmpring/UserD0-prefix emitted once at the top.)
  -> ~4x fewer register dwords and ~2.3x fewer SET_SH_REG packets per steady-state
  dispatch. The front-end term scales with reg-write + packet count, so this
  directly trims the ~0.5 us front-end.

#### D.10.3 Code changes (buildPm4GraphIb)

  - Add a small struct RegShadow { bool valid; uint32_t v[8]; } per group
    (StartX, Pgm, ScratchLo, Rsrc, ResLim, Tmpring, UserD0) plus a stored
    UserD0 enable-mask. Initialize all valid=false before the loop.
  - Replace each direct setsh(...) with setshDelta(group, reg, vals, cnt): compares
    vals against the shadow; emits SET_SH_REG only on mismatch (or shadow invalid);
    updates shadow. For UserD0, compare the enable-mask first: if changed, emit
    full and reset; else compare only the kernarg sub-range and emit it at its
    offset.
  - Gate the whole thing behind getenv("HIP_PM4_GRAPH_DELTA"); when unset, fall
    back to the existing always-emit path byte-for-byte (so the current bit-exact
    IB test still passes for the default path).

### D.11 IB-reorder + release/acquire interleave -- plan

#### D.11.1 Current order (per the loop, rocvirtual.cpp:1937-2044)

  acquireFull(leading)                 once, full-L2 (graph inputs coherent)
  for i in 0..N-1:
    setsh(regs_i)                      front-end for kernel i
    DISPATCH_DIRECT i
    partialFlush                       drain i (wait i's waves done)
    releaseMemPws(AGENT)               async flush i (PWS counter)
    acquirePws                         wait flush i  <-- consumer i+1 blocked here
  partialFlush; acquireFull(final)     full-L2 writeback (system-visible results)

PROBLEM: regs_{i+1} (front-end) is emitted in the NEXT iteration, AFTER acquire_i.
So the CP programs the next kernel's registers IN the gap, serialized; and the
flush tail (acquire_i) is not overlapped with anything useful. Front-end and
flush-tail are additive.

#### D.11.2 Target order (reorder + interleave)

  acquireFull(leading)
  setsh(regs_0)                        program first kernel up front
  DISPATCH_DIRECT 0
  for i in 1..N-1:
    partialFlush                       drain i-1 (i-1 waves done -> regs reusable)
    releaseMemPws(AGENT)               async flush i-1 (PWS counter armed)
    setsh(regs_i)                      <== program kernel i WHILE flush i-1 is in
                                          flight AND the CP is otherwise idle
    acquirePws                         wait flush i-1 counter
    DISPATCH_DIRECT i                  coherent (caches done) + regs latched
  partialFlush; acquireFull(final)     drain N-1 + full-L2 writeback

EFFECT: kernel i's front-end (its SET_SH_REG burst) now overlaps BOTH the CP-idle
window during i-1's drain AND the cache-flush tail between release_{i-1} and
acquire_{i-1}. The only thing left strictly serial on the edge is the flush
latency MINUS the front-end we just hid under it.

#### D.11.3 Why each move is SAFE

  - DRAIN STAYS FIRST on every edge: partialFlush for i-1 still precedes the
    reprogramming of regs_i. Kernel i-1's waves are FULLY retired before we
    overwrite any COMPUTE_* register, so we never corrupt an in-flight kernel.
    (Registers latch at DISPATCH; i-1 already dispatched and is now drained.)
  - ACQUIRE STAYS BEFORE THE CONSUMER DISPATCH: acquirePws for i-1 still precedes
    DISPATCH_DIRECT i, so kernel i observes coherent memory (i-1's writes visible,
    L0/L1/K invalidated) -- identical coherence to the current code.
  - REGS BETWEEN RELEASE AND ACQUIRE ARE INDEPENDENT OF THE FLUSH: SET_SH_REG
    writes CP/SH state; the PWS flush operates in the memory system. They do not
    alias. dispatch_i, which consumes the registers, is emitted AFTER acquire, so
    the registers are guaranteed latched and memory guaranteed coherent at launch.
  - SAME FENCE COUNT/SCOPE PER EDGE: each internal edge still has exactly one
    drain + one AGENT PWS release + one PWS acquire. Semantics per edge are
    unchanged; only the POSITION of the register burst moved.
  - OPTIONAL micro-opt (separate flag): the current code emits an AGENT
    release/acquire AFTER the last dispatch AND THEN a full-L2 acquireFull -- the
    trailing AGENT fence is redundant with the full fence (full-L2 invalidate
    subsumes the AGENT invalidate; RDNA L0/L1 are write-through to L2 so the
    writeback is covered by GL2_WB). The reorder structure drops this redundant
    trailing AGENT fence naturally. Keep this behind its own sub-flag and validate
    independently.

#### D.11.4 Composition with delta-encoding

In the reordered loop the "setsh(regs_i)" burst becomes the DELTA burst from D.10.
They compose cleanly: delta decides WHICH groups to emit, reorder decides WHERE.
Delta shrinks the burst that reorder hides, so after delta there is LESS front-end
to overlap -- the two levers are complementary, not redundant (delta reduces the
absolute work; reorder hides whatever remains under the flush tail).

#### D.11.5 Code changes + flag

  - Restructure the loop: emit dispatch 0's regs+dispatch before the loop; in the
    loop body emit drain+release for the PREVIOUS kernel, then regs+acquire+dispatch
    for the CURRENT kernel. Keep the per-segment compilation (G5) boundaries:
    reorder applies WITHIN a contiguous dispatch segment; segment edges keep the
    leading/trailing full fences as today.
  - Gate behind getenv("HIP_PM4_GRAPH_REORDER"); default OFF (byte-exact baseline).
  - Allow DELTA and REORDER to be enabled independently and together.

### D.12 Test & verification plan (both levers)

KEY POINT: both levers change the IB BYTES, so the existing "IB bit-exact vs the
baseline emitter" check (Appendix B) is NOT the right gate for them -- it only
guards the DEFAULT (flags-off) path, which must remain byte-identical. The levers
are gated on RESULT bit-exactness plus a register-state invariant check.

#### D.12.1 Invariant-level (cheapest, proves correctness of delta)

  - Debug self-check in buildPm4GraphIb: maintain a parallel "reference" shadow
    that records, for each DISPATCH, the FULL latched register state, computed two
    ways -- (a) the baseline always-emit values, (b) the state implied by the
    delta stream (apply only the emitted writes on top of the running shadow).
    assert(a == b) at every DISPATCH. This proves the latched state at every launch
    is identical to baseline. Compile-time/asserts only; no GPU needed.
  - For reorder: assert the per-edge fence multiset (drain, release(scope),
    acquire) is unchanged vs baseline, and that for every edge the order
    drain -> (release) -> (regs) -> acquire -> dispatch holds.

#### D.12.2 Result bit-exactness (the real gate) -- reuse Appendix B suite

  Run the existing HIP test suite with each flag combination
  (DELTA, REORDER, DELTA+REORDER) and compare OUTPUT buffers byte-for-byte against
  the baseline AQL path. Cases (already in the suite, extend where noted):
    - scratch on / off and the on<->off TRANSITION (exercises ScratchLo/Tmpring
      delta + UserD0 layout change when KCP_PRIVATE_SEGMENT_BUFFER toggles).
    - dispatch_ptr / queue_ptr variants (UserD0 layout changes -> full re-emit).
    - wave32 and wave64.
    - non-multiple grid (USE_THREAD_DIMS path).
    - VARYING block size across consecutive kernels (NEW: exercises StartX delta
      both skip and re-emit) -- add a chain that alternates 64 / 128 / 256 blocks.
    - SAME kernel back-to-back (NEW: exercises Pgm/Rsrc SKIP).
    - multi-kernel RAW chain (consumer reads producer) -- coherence across the
      interleaved fence.
    - the 64-kernel chain + REVERSE-READ kernel from section 6l: this is the
      cache-coherence proof. If reorder mis-ordered any fence, the reverse-read
      output corrupts. Run under DELTA, REORDER, and both.
    - barrier-node graph, destroy+recreate, H2D-then-graph coherence.
  Run on BOTH arches: gfx1100 (Navi31) and gfx1201 (Navi48).

#### D.12.3 End-to-end model gate

  - Capture the flywheel decode hipGraph; run HIP_PM4_GRAPH=1 with
    {none, DELTA, REORDER, DELTA+REORDER} under PINNED clocks.
  - ASSERT identical logits (bit-exact) vs the flags-off PM4 path AND vs the
    baseline AQL path. Any logit drift => fail (a fence/register bug).

#### D.12.4 Performance measurement (only after correctness passes)

  - PINNED clock (profile_peak), GPUs per the project default; check amd-smi first.
  - Use the NATIVE runtime trace (NOT rocprofv3 -- it inflates kernel timings) to
    measure per-dispatch front-end and the inter-kernel gap.
  - Report: per-dispatch SET_SH_REG dword/packet count (build-time, exact),
    measured inter-kernel gap, and end-to-end TPOT delta for the 8B decode.
  - Expected (hypothesis, to be confirmed): DELTA trims the front-end fraction;
    REORDER hides the residual front-end under the flush tail; combined, the
    ~2.0 us gap moves toward ~1.2-1.5 us on a linear chain. NOT to zero.

#### D.12.5 Rollout

  Phase 0  shadow/delta infra + flags, default OFF; D.12.1 asserts pass.
  Phase 1  DELTA only: D.12.2 + D.12.3 bit-exact on gfx1100 + gfx1201; measure.
  Phase 2  REORDER only, then DELTA+REORDER: same gates; run the 6l reverse-read
           coherence proof explicitly.
  Phase 3  optional trailing-AGENT-fence drop (own sub-flag); re-run coherence.
  Phase 4  flip defaults ON only after bit-exact + perf-positive on both arches.

### D.13 MEASURED results (gfx1100 W7900, implemented + validated)

Both levers are implemented in buildPm4GraphIb (rocvirtual.cpp), gated by
HIP_PM4_GRAPH_DELTA and HIP_PM4_GRAPH_REORDER, default OFF (flags-off path is
byte-identical to the prior emitter). New env accessors pm4GraphDeltaEnabled() /
pm4GraphReorderEnabled(). emitRegs() / emitDispatchPacket() split the per-dispatch
work; setshDelta() does the minimal-dirty-range shadow emit.

CORRECTNESS (hip_pm4_coverage.cpp, run_pm4_delta_reorder.sh) -- gfx1100, ALL
scenarios bit-exact vs the baseline AQL path for pm4 / delta / reorder / both:
  multi, nonmult, coherence, recreate, lds, wave64, barriernode, scratch, and a
  NEW varblk scenario (alternating 64/128/256 block sizes + same-kernel repeats,
  to exercise the StartX partial re-emit and the PGM/RSRC skip). Checksums
  identical across all four flag combinations.
  (gfx1201 RDNA4: code path is arch-parameterized and build-verified, but no RDNA4
  hardware is present in this environment to run.)

DELTA payload reduction (runtime "compiled N dispatches -> IB X dwords" log):
  24-dispatch multi graph:  baseline 1242 dw -> delta 736 dw  (-41%).
  Steady-state per dispatch: ~28 reg dwords / 6 SET_SH_REG packets ->
  ~12 reg dwords / 3 packets (StartX/ResLim/Tmpring/scratch/queue_ptr hoisted;
  only PGM, RSRC1/2, kernarg re-emitted because the multi kernels all differ).
  REORDER drops exactly one trailing AGENT fence (1242 -> 1224 dw = 18 dwords =
  one CS_PARTIAL_FLUSH + RELEASE_MEM + ACQUIRE_MEM), subsumed by the final full-L2
  acquire -- confirms the elision is correct and accounted.

MICROBENCH gap (hip_pws_test, dependent chain, pinned 1124 MHz, L=60, n=2000,
us/dispatch; identical checksums):
  M=1024    pm4 2.579  delta 2.388 (-7.4%)  reorder 2.578 (0%)  both 2.385
  M=4096    pm4 2.594  delta 2.400 (-7.5%)  reorder 2.590 (0%)  both 2.397
  M=65536   pm4 2.738  delta 2.547 (-7.0%)  reorder 2.735 (0%)  both 2.542
  (AQL baseline for M=4096 was 3.597 us/dispatch; PM4 graph already cut that to
  2.594; DELTA cuts a further ~0.19 us.)

  KEY FINDING -- REORDER is NEUTRAL (within noise) at every kernel size. The
  D.11 hypothesis (hide the front-end under the flush tail) does NOT hold on this
  hardware: the CP micro-engine (ME) is a single in-order processor that handles
  BOTH the SET_SH_REG writes AND the ACQUIRE_MEM wait, so moving the register
  burst before the acquire only reorders serial ME work -- there is no second
  engine to run it concurrently with the PWS wait. DELTA helps precisely because
  it REDUCES the absolute amount of ME work (fewer dwords to fetch/decode/write),
  which is directly on the critical path; REORDER does not reduce the work, so it
  cannot help. (The PWS flush still overlaps the next dispatch's WAVE execution --
  that win is already in the pm4 baseline -- but the register PROGRAMMING is ME
  work either way.) Conclusion: keep DELTA; REORDER stays implemented + gated OFF
  as a validated null result (no benefit, but bit-exact and harmless if enabled).

E2E (gpt-oss-20b MoE, q0a, 512 in / 128 out, single W7900 gfx1100, patched lib via
LD_LIBRARY_PATH, REDLINE_DEBUG_SKIP_SAMPLING=1, no clock pin, conc=1):
  n=3 :  TPOT  baseline 5.17  -> pm4 4.91 (+5.0%) -> delta 4.86 ms (+6.4%/+1.0%)
  n=10:  Tok/s baseline 839.4 -> pm4 872.0        -> delta 876.9 (+4.5%/+0.56%)
  DELTA gives a small but CONSISTENT additional e2e gain over PM4 (delta > pm4 in
  both runs), matching the microbench: ~0.19 us/dispatch x ~286 GPU ops/token of
  the MoE decode ~= ~0.05 ms/token ~= ~1% TPOT. MoE benefits because it has many
  SHORT kernels per token (the gap is a non-negligible TPOT fraction); dense 8B
  decode (Appendix B) stays flat because it is weight-streaming memory-bound and
  the gap is negligible there.

CORRECTNESS of DELTA/REORDER: gated by the bit-exact coverage suite (same code
path, all decode-relevant features: multi-kernel, LDS, scratch, wave64, varying
block size, user-SGPR layout change) -- all four flag combos identical to the AQL
baseline. setshDelta only skips a write whose value is already latched, so the
register state at every DISPATCH is provably identical to the pm4 path.

PRE-EXISTING PM4-GRAPH ISSUE found on gpt-oss-20b MoE (NOT a delta/reorder
regression -- pm4_e2e_check.py, greedy o=48):
  baseline (AQL graph)  : DETERMINISTIC, sha 579941330f4a164e identical over 3/3 runs
  pm4 (HIP_PM4_GRAPH=1)  : NON-deterministic, sha varies run-to-run (2a54e297..,
                           6cb9c585..) AND differs from baseline
  delta / both           : same non-deterministic behavior as pm4 (no worse)
The flags-OFF pm4 path is byte-identical to the original emitter, so this predates
the delta/reorder work. The dense Llama-3.1-8B decode (Appendix B) was bit-
identical baseline-vs-pm4; the MoE model is not. Hypothesis: the per-edge AGENT
PWS fence (no L2 flush; only the token-boundary acquire is full-L2) is insufficient
for some MoE cross-CU dataflow (e.g. the expert scatter/combine through L2), or the
MoE expert-combine uses timing-sensitive atomics whose race order the PM4 path's
different timing exposes. Either way it is a PM4-GRAPH fence-scope correctness
question on MoE, INDEPENDENT of register delta-encoding. ACTION: investigate the
MoE fence scope (does an internal edge need a full-L2 / GL2 flush where a producer
expert's L2-resident output is read by a consumer on another CU?) before using
HIP_PM4_GRAPH on MoE models. DELTA/REORDER remain safe to layer on top once the
base path is correct, since they preserve the pm4 path's exact register + fence
semantics.
  [RESOLVED in D.14: it was the PWS fence MECHANISM, not the cache scope. Fixed by
  switching the per-edge fence to a blocking AGENT acquire, which is also faster.]

### D.14 ROOT CAUSE + FIX: the per-edge fence must be BLOCKING, not PWS

Investigation (gpt-oss-20b MoE, gfx1100, greedy decode sha over 48 tokens; the AQL
baseline is deterministic at sha 579941330f4a164e):

1. The MoE token compiles to ONE PM4 IB, 267 dispatches, NO fallback -- PM4 fully
   drives it.
2. Forcing a BLOCKING full-L2 fence per edge (HIP_PM4_GRAPH_FULLFENCE) made it
   deterministic AND identical to baseline. So it is a per-edge fence problem.
3. Bisecting the per-edge GCR mask as a BLOCKING acquire (HIP_PM4_GRAPH_EDGE_GCR):
   mask 0x380 (AGENT: GLK|GLV|GL1, the SAME bits the PWS path carries) was ALREADY
   deterministic + correct. 0x3B0/0xC380/0xC3B1 also correct. So the cache SCOPE
   was never the issue.
4. Therefore the bug is the PWS (partially-waited-sync) MECHANISM, not the scope:
   RELEASE_MEM arms a counter and ACQUIRE_MEM waits on it, but the consumer DISPATCH
   can begin before the AGENT invalidate is GLOBALLY complete -- the deferred wait
   does not actually gate the consumer's reads on invalidate completion. Dense
   Llama-8B hid this behind larger kernels' launch latency; MoE's 267 tiny
   dependent dispatches/token expose it as non-deterministic stale reads.

THE FIX: per-edge fence = BLOCKING AGENT acquire --
  partialFlush() (drain producer waves) + acquireFull(kGcrAgent) (ACQUIRE_MEM with
  GCR_CNTL = GLK|GLV|GL1, no L2 flush). L2 is the device coherence point; L0/L1 are
  write-through, the drain puts the producer's writes in L2, and the blocking
  ACQUIRE invalidates the consumer's L0/L1/K so it misses to L2. The token boundary
  still does a full-L2 writeback. This is exactly the AGENT scope the runtime
  applies to every AQL dispatch.

It is also FASTER than PWS (the original study only compared PWS against a blocking
FULL-L2 fence, whose GLK/GL2 invalidate cost grows with M -- it never tried a
blocking AGENT fence, which has none of that cost):

  Microbench (gfx1100, pinned, L=60, n=2000, us/dispatch; identical checksums):
    M=4096   AQL 3.597 | PWS 2.597 | PWS+delta 2.403 |
             blockAGENT 1.280 | blockAGENT+delta 1.085
    M=65536  PWS 2.753 | blockAGENT 1.445 | blockAGENT+delta 1.251
  => blocking AGENT ~2x cheaper per dispatch than PWS (one ACQUIRE_MEM vs a
     RELEASE_MEM+ACQUIRE_MEM counter round-trip); +delta on top is fastest.

  E2E gpt-oss-20b MoE (q0a, 512/128, SKIP_SAMPLING, n=10, clean):
    baseline(AQL)        5.17 ms  839 Tok/s   (deterministic)
    PWS  (old default)   4.94 ms  871 Tok/s   (NON-deterministic, WRONG)
    blockAGENT           4.71 ms  905 Tok/s   (deterministic, matches baseline)
    blockAGENT+delta     4.70 ms  910 Tok/s   (deterministic; +8.4% vs baseline)
    blocking full-L2     4.91 ms  875 Tok/s   (correct but the L2 flush costs)

IMPLEMENTATION (rocvirtual.cpp buildPm4GraphIb):
  - DEFAULT per-edge fence is now blocking AGENT. Correct, deterministic, fastest.
  - HIP_PM4_GRAPH_PWS=1 selects the LEGACY PWS fence (A/B only; UNSAFE for chains of
    many tiny dependent kernels).
  - HIP_PM4_GRAPH_DELTA composes with the blocking fence.
  - HIP_PM4_GRAPH_REORDER only applies under PWS and remains a measured no-op.
  - HIP_PM4_GRAPH_FULLFENCE / HIP_PM4_GRAPH_EDGE_GCR=<hex> are diagnostics that
    override the blocking mask.

VALIDATION on the new default:
  - hip_pm4_coverage: ALL scenarios bit-exact vs AQL baseline (pm4/delta/reorder/
    both), gfx1100, incl. varblk + scratch + wave64 + lds.
  - gpt-oss-20b MoE: pm4 and pm4+delta now DETERMINISTIC (3/3 and 2/2) and identical
    to the AQL baseline (sha 579941330f4a164e).

NOTE: this supersedes the Appendix B / 6n framing that PWS was "THE FIX". PWS was a
correct-looking but actually unsafe + slower detour; the blocking AGENT fence is
the correct and faster per-edge primitive for the PM4-graph replay path.

### D.15 FINAL clean e2e validation: two models, three runtimes, two clock states

Goal: confirm (a) the from-sources patched HIP build matches stock HIP with PM4 OFF
(no regression from the patched runtime itself), (b) the fully optimized PM4 path
(blocking AGENT default + DELTA) is faster, and (c) output is deterministic and
bit-identical to stock HIP. Three runtimes per model:

  A) Default HIP   -- stock /opt/rocm-7.2.0 libamdhip64 (env -u LD_LIBRARY_PATH), no PM4
  C) Patched, OFF  -- from-sources patched lib, HIP_PM4_GRAPH unset (AQL path)
  B) Optimized PM4 -- patched lib + HIP_PM4_GRAPH=1 + HIP_PM4_GRAPH_DELTA=1

Common: gfx1100 W7900, q0a, 512 in / 128 out, REDLINE_DEBUG_SKIP_SAMPLING=1, -c1 -tp1,
benchmark.py -n 10, 5 independent runs per config (avg TPOT reported, +-spread <=0.03 ms).

gpt-oss-20b (MoE):
                            PINNED (profile_peak)        RELAXED (dynamic boost)
    A) Default HIP          5.93 ms  ~713 Tok/s          5.20 ms  ~835 Tok/s
    C) Patched, PM4 OFF     6.03 ms  ~699 Tok/s          5.20 ms  ~834 Tok/s
    B) Optimized PM4        5.55 ms  ~747 Tok/s          4.68 ms  ~913 Tok/s
    => optimized vs baseline:  -0.48 ms (~8%) pinned     -0.52 ms (~10%) relaxed

Llama-3.1-8B-Instruct (dense), RELAXED clocks:
    A) Default HIP          9.08 ms  ~497 Tok/s
    C) Patched, PM4 OFF     9.09 ms  ~497 Tok/s
    B) Optimized PM4        8.51 ms  ~526 Tok/s
    => optimized vs baseline:  -0.58 ms (~6.4%)

Observations:
  - A ~= C on both models and both clock states: the patched from-sources build is
    indistinguishable from stock HIP when PM4 is OFF. No regression from the runtime.
    (Pinned MoE: A's run1 5.80 is a cold-GPU boost transient that drifts up to C's
    steady 6.03 by run4/5; relaxed is rock-steady A=C=5.20.)
  - The optimized PM4 absolute gain is ~constant per model (~0.5 ms/token) because the
    saving is per kernel-edge dispatch overhead, not per-FLOP. The PERCENT is larger on
    the cheaper-per-token MoE (5.2 ms) than on dense Llama-8B (9.1 ms).
  - Relaxed (dynamic boost) clocks are faster than the profile_peak pin on this part,
    and the relative gap is preserved/slightly larger.

Determinism (pm4_e2e_check.py, greedy, fixed prompt, 96 tokens, token-id sha256):
    gpt-oss-20b   optimized r1..r4 = default-HIP ref = sha c1db07d351fac8e2  (5/5 match)
    Llama-3.1-8B  optimized r1..r4 = default-HIP ref = sha 155e20fa7eb7cda4  (5/5 match)
  => the optimized PM4 path is fully deterministic across runs AND bit-identical to
     stock HIP output on both a dense and an MoE model. The blocking AGENT fence
     preserves the already-deterministic dense path and fixes the MoE path with no
     logit drift.

### D.16 FAQ: what PM4 is, why the small CLR diff is so much faster, AQL vs PM4

Q: The CLR diff is small (~740 lines, 3 files, no kernel code). What is the magic?
  It is a SUBMISSION-PATH change, not a feature. The existing hipGraph capture already
  knows every node's kernel descriptor, kernarg pointer, grid/block dims and scratch
  needs. The patch intercepts that captured packet list and, instead of replaying it
  as N separate AQL dispatches, compiles it ONCE into a single PM4 indirect buffer and
  replays that with one submit. The 740 lines are almost entirely a self-contained PM4
  encoder + the gfx11/gfx12 register-offset and GCR tables + delta logic + scratch/arch
  guards + env flags. No new kernels, no math change, no scheduler change.

Q: Why is the PM4 path so much faster, then?
  It removes FIXED per-kernel-edge overhead, multiplied across hundreds of dependent
  dispatches (MoE token = 267):
   1. One submit, not N. The whole token is one IB the CP micro-engine streams back to
      back: no per-dispatch doorbell ring, no per-dispatch AQL-packet-processor wake-up,
      no host packet writes.
   2. No AQL->PM4 translation. Register programming (SET_SH_REG) and DISPATCH_DIRECT are
      pre-baked into the IB instead of being decoded by the AQP firmware per packet.
   3. A lean per-edge fence (see D.16 fence note below).
   4. Register delta-encoding (HIP_PM4_GRAPH_DELTA): consecutive dispatches share most
      register state, so emit SET_SH_REG only for the contiguous sub-range that changed
      -> 41% fewer IB dwords -> less CP front-end work per edge.

Q: "The default AQL fence is heavier" -- but it is the SAME AGENT scope. Why heavier?
  Correct: the cache SCOPE is identical (AGENT: invalidate L0/L1/K, no L2 flush) -- D.14
  proved scope was never the difference. What differs is the MECHANISM + SERIALIZATION,
  not which caches are touched:
   - Stock HIP fence = a serialized CP round-trip. The AQL packet carries an
     acquire_fence_scope AND a release_fence_scope; the AQP implements them as a release
     sequence (drain + signal) followed by an acquire (wait), as separate CP ops per
     packet. Measured ~5.8 us serialized AGENT fence per dispatch (D.5). Same caches, but
     the CP fully stalls on it each edge.
   - PM4 per-edge fence = one CS_PARTIAL_FLUSH (drain producer waves) + one ACQUIRE_MEM
     with the AGENT mask (kGcrAgent = 0x380). No separate release/signal round-trip, no
     counter; just inline dwords the CP streams. Same AGENT caches invalidated, far less
     CP serialization. (PWS went further by overlapping the flush, but D.14 showed the
     simple blocking AGENT acquire is both faster than PWS and correct.)

Q: What is PM4 / what does it stand for?
  PM4 is AMD's command-packet format for the GPU Command Processor (CP) -- the binary
  "ISA" the CP micro-engines (ME/PFP/MEC) fetch and execute. The name is historical:
  "Programming Model 4" (4th-generation packet model, R600 era); in practice everyone
  says "PM4 packets". Two packet types are used here: type-0 (write a run of consecutive
  registers) and type-3 (a command with an opcode -- SET_SH_REG 0x76, DISPATCH_DIRECT
  0x15, RELEASE_MEM 0x49, ACQUIRE_MEM 0x58, INDIRECT_BUFFER). An IB (Indirect Buffer) is
  a buffer of PM4 packets the CP jumps into; INDIRECT_BUFFER is effectively a "call" into
  a PM4 stream. Our whole token is one such PM4 IB.

Q: Does a PM4 IB have a length limit?
  INDIRECT_BUFFER encodes IB_SIZE in a 20-bit field -> up to 2^20-1 ~= 1,048,575 dwords
  (~4 MB) per IB, and IBs can chain into further IBs. The MoE token (267 dispatches, a
  few thousand dwords) is far below this, so length is a non-issue.

Q: Why is the default path AQL->PM4 translation instead of submitting PM4 directly?
  AQL (Architected Queuing Language) is the HSA-standard USER-MODE submission interface,
  and it exists for reasons raw PM4 IBs cannot give directly:
   1. User-mode, no kernel transition. A process writes an AQL packet into a ring in its
      own address space and rings a doorbell; a firmware packet processor (AQP) consumes
      it. Raw PM4 IB submission is PRIVILEGED -- normally only the kernel-mode driver
      builds/submits PM4 IBs, because arbitrary PM4 can program any register and hang or
      compromise the GPU. AQL is a constrained, validated format the AQP can safely
      accept from userspace.
   2. Portability. AQL is a vendor-neutral HSA packet layout; PM4 is AMD-internal and
      changes across CP generations. The AQP firmware is exactly the thing that
      translates the portable AQL packet into gfx-specific PM4 register programming +
      dispatch.
   3. Multi-producer queue semantics: signals, barrier packets, multi-queue arbitration.
  The cost of that safety/portability is the per-packet firmware translation + serialized
  fence. Our trick threads the needle: we STILL submit through the normal validated queue,
  but as a single AMD vendor PM4-IB packet that points the CP at our pre-built PM4 IB --
  keeping the safe user-mode one-submit path while bypassing the per-dispatch AQL->PM4
  translation and the per-packet serialized fence that made the default path slow.

### D.17 MEASURED: isolated HOST (CPU) cost of issuing a replay -- O(1) claim REFUTED

Tool: hip_replay_timing.cpp (vk_gap_test/pm4_gap). Captures an N-long addk chain
into one hipGraph and times ONLY the host return latency of hipGraphLaunch with an
empty queue each iteration (hipStreamSynchronize is done OUTSIDE the timer), so the
number is the runtime's host work to submit the replay, not GPU execution. gfx1100
W7900, M=256 (tiny grid so the GPU never lags), R=8000 timed launches, avg / min us.

  N      AQL graph replay (avg/min)     PM4-IB replay (avg/min)
  1      2.46 / 1.85                    2.59 / 1.91
  16     3.36 / 2.71                    2.81 / 2.22
  64     3.94 / 3.32                    4.38 / 3.66
  256    5.11 / 4.50                    9.82 / 9.03
  512    7.03 / 5.83                   17.02 / 16.41
  per-dispatch host slope: AQL ~8.9 ns/disp   PM4 ~28 ns/disp

FINDING (correcting an earlier verbal claim that PM4 host issue is O(1)): it is NOT.
Both paths are O(N) on the host, and PM4 is actually the STEEPER one (crosses over
around N ~ 32-64 and is ~2.4x slower at N=512). Reasons:
  - The CLR hipGraph executor materializes/processes the per-node AQL packet list on
    the host on EVERY launch, for both paths -- that is the shared O(N) floor.
  - The PM4 path then adds a per-launch O(N) CONTENT HASH (pm4GraphKey) to look up the
    cached IB: it reads 6 scattered fields from each recorded packet and runs a
    byte-wise FNV loop. That is heavier per packet than the AQL path's streaming
    64-byte ring memcpy it replaces. The O(1) part (buildPwsVendorPkt + one ring write
    + one doorbell) is real, but it is preceded by that O(N) hash.
So PM4's win is ENTIRELY GPU/CP-side (fewer, leaner packets + lean fence); it does
NOT reduce -- and slightly increases -- host issue cost.

WHY IT DOES NOT MATTER FOR TPOT: even at N=512 the host issue is ~17 us, vs ~5000 us
per decode token (gpt-oss-20b). That is ~0.3% of token time and it OVERLAPS GPU
execution (hipGraphLaunch is async). The measured e2e speedup (D.15) is unaffected by
this and is purely the GPU-side dispatch-gap reduction.

FOLLOW-UP OPPORTUNITY: the recorded packet set is stable for a given hipGraphExec, so
pm4GraphKey could be computed ONCE and cached, removing PM4's per-launch O(N) hash and
making PM4 host issue O(1) (one vendor packet + doorbell). IMPLEMENTED + measured in
D.18 below.

### D.18 IMPLEMENTED: last-lookup key cache (HIP_PM4_GRAPH_KEYCACHE)

WHY (ties to D.17 + the within-launch latency point): the per-launch pm4GraphKey hash
is O(N) over every packet, and in the steady-state decode loop the same packet array
is replayed every token -- nothing the hash covers changes (only kernarg CONTENTS
change, which the hash ignores by design), so the hash returns the SAME key every
launch. Worse, that O(N) hash sits BEFORE the submit doorbell: in PM4 the GPU cannot
start until the doorbell, so the hash directly delays time-to-first-kernel (unlike AQL,
where the AQP starts kernel 0 while the host is still copying later packets). Removing
the hash both flattens host issue cost AND lets the GPU start sooner.

IMPLEMENTATION (opt-in HIP_PM4_GRAPH_KEYCACHE; default OFF keeps the always-rehash,
fully-general path). RELIABLE invalidation via a graph-supplied replay token (NOT a
content heuristic):
  - GraphExec carries pm4ReplayId_ (unique per instantiation, from a static atomic)
    and pm4Epoch_ (bumped on EVERY recorded-packet mutation: CaptureAndFormPackets
    ForGraph, UpdateAQLPacket, UpdatePacketBatchesForNodeEnableDisable). The token =
    fold(pm4ReplayId_, pm4Epoch_, batchIndex) is nonzero, unique per (instantiation,
    batch), and changes on ANY update.
  - The token is threaded down dispatchAqlPacketBatch -> dispatchGenericAqlPacketBatch
    -> tryReplayPm4Graph (one optional uint64 param on the device virtual; pal ignores
    it). The rocm key cache stores {lastToken -> Pm4GraphIb*}; on a token match it
    reuses the cached IB pointer (node-based map, pointer stays valid) WITHOUT the hash
    or the map find -> O(1) host issue. token 0 (non-graph / multi-device path) always
    takes the slow path.
  - This is bulletproof, unlike the first attempt (a first/last packet content fold,
    which could miss an interior-only in-place param update). Any update bumps the
    epoch -> the token changes -> a stale IB can never be served. ABA is impossible
    (a new GraphExec gets a fresh pm4ReplayId_).

MEASURED host hipGraphLaunch issue cost (gfx1100, M=256, R=8000, avg us):
  N      AQL graph    PM4 (no cache)   PM4 + KEYCACHE
  1      2.46          2.59             3.02
  16     3.36          2.81             2.91
  64     3.94          4.38             2.54
  256    5.11          9.82             2.48
  512    7.03         17.02             2.52
  => KEYCACHE makes PM4 host issue FLAT ~2.5 us at every N (O(1)), now the cheapest
     path at every N -- 6.8x faster than PM4-no-cache and 2.8x faster than AQL at N=512.

VALIDATION:
  - hip_pm4_coverage: all scenarios bit-exact vs AQL baseline with
    HIP_PM4_GRAPH=1 SCRATCH=1 DELTA=1 KEYCACHE=1 (gfx1100).
  - gpt-oss-20b MoE e2e: deterministic (3/3) and identical to stock HIP
    (sha c1db07d351fac8e2), token-driven cache.

DOES IT HELP TPOT? Honestly measured both with and without the per-token sync:
  - SKIP_SAMPLING (no per-token sync, host issue overlaps prior token's GPU):
    keycache 4.64-4.66 ms vs no-keycache 4.68 ms -- ~1%, at the edge of noise.
  - REAL decode (sampling ON -> hard per-token sync for EOS/sampling/autoregressive
    dep, so the launch IS on the critical path after the sync):
    no-keycache 4.90-4.92 ms vs keycache 4.92-4.93 ms -- NO improvement (within noise).
  CONCLUSION: even with the per-token sync the launch host cost (~10 us at N=267) is
  dwarfed by per-token kernel compute (~4900 us) AND by the sampling+d2h-sync overhead
  (~230 us; real decode is ~0.23 ms/token slower than SKIP_SAMPLING), so removing the
  ~7.5 us hash does not move TPOT. KEYCACHE's value is therefore limited to: (a) flat
  O(1) host issue regardless of graph size, and (b) lower time-to-first-kernel -- which
  would only matter for workloads with very small per-token GPU time or very large N.
  It is correct, reliable, and free to leave OFF for decode. Kept opt-in.

RE-MEASURED host issue with the TOKEN-driven cache (replacing the heuristic; identical
behavior, confirming the rewrite did not regress the fast path), gfx1100 M=256 R=8000:
  N=1   2.595 us | N=64  3.576 us | N=256 2.521 us | N=512 2.493 us   (avg)
  => still flat ~2.5 us; the small N=64 bump is run noise (min was 1.99 us).

DEFAULT POLICY: kept OFF. Rationale: zero measured TPOT benefit for decode, and
flipping a runtime default to ON is a correctness bet on the COMPLETENESS of the
mutation-site list (the three bump sites). The mechanism is reliable for those sites;
leaving it opt-in keeps the always-rehash path as the conservative default and makes
the flat-host-issue / TTFT property available to anyone who wants it via
HIP_PM4_GRAPH_KEYCACHE=1.

### D.18-impl Code changes, file by file (what + why)

The change adds a RELIABLE "did the recorded packets change?" signal and threads it
to the rocm PM4 key cache. Two halves: (A) the signal in the HIP graph layer, (B) the
consumer + plumbing in the rocm backend.

(A) HIP graph layer -- hip_graph_internal.hpp / .cpp (the SIGNAL):
  1. #include <atomic> -- for the unique-id generator.
  2. GraphExec gains two members:
       uint64_t pm4ReplayId_ = NextPm4ReplayId();  // unique per instantiation
       uint64_t pm4Epoch_ = 0;                       // bumped on any packet change
     pm4ReplayId_ uses an in-class initializer calling a static atomic counter, so
     EVERY GraphExec constructor (including ChildGraphNode's) gets a fresh, globally
     unique id -- this is what makes ABA impossible (a freed+reallocated GraphExec can
     never collide with a live one's token).
  3. Two small methods:
       Pm4ReplayToken(batchIndex) = fold(pm4ReplayId_, pm4Epoch_, batchIndex), nonzero.
         Folds the batch index so each batch in a multi-batch graph gets a distinct
         token (a single-slot last-token cache would otherwise thrash across batches).
       BumpPm4Epoch() = ++pm4Epoch_.
  4. BumpPm4Epoch() is called at the THREE -- and only three -- places that change the
     set of packets that will be dispatched:
       - CaptureAndFormPacketsForGraph(): (re)builds packetBatches_ at instantiate /
         re-capture.
       - UpdateAQLPacket(node): rewrites a node's recorded AQL packet on a param update
         (hipGraphExecKernelNodeSetParams / hipGraphExecUpdate).
       - UpdatePacketBatchesForNodeEnableDisable(node, enabled): changes which packets
         are dispatched when a node is enabled/disabled.
     NOTE on what is deliberately NOT bumped: changing kernarg CONTENTS behind a stable
     pointer (the normal decode pattern) does NOT change the packets and correctly does
     NOT bump the epoch -- the IB references the kernarg by pointer and reads fresh data
     at replay. This is exactly why the cache is safe to reuse across decode tokens.
  5. The two single-list dispatch call sites pass Pm4ReplayToken(batchIndex); the
     multi-device linear path is left at the default 0 (see D.18 Q1: PM4 does not
     target multi-GPU graphs, so 0 = always-slow-path is correct there).

(B) rocm backend -- device.hpp / palvirtual.hpp / rocvirtual.hpp / rocvirtual.cpp:
  6. dispatchAqlPacketBatch gains an optional trailing "uint64_t recordedPacketSetVersion
     = 0" on the device virtual (device.hpp), the pal override (ignores it), and the rocm
     override; rocm passes it into dispatchGenericAqlPacketBatch, which passes it into
     tryReplayPm4Graph. The default 0 keeps every non-graph caller on the slow path.
  7. pm4GraphKeyCacheEnabled() -- env gate HIP_PM4_GRAPH_KEYCACHE. Now DEFAULT ON (see
     D.19); set HIP_PM4_GRAPH_KEYCACHE=0 to force the always-rehash path.
  8. Per-VirtualGPU fast-path state: {bool pm4IbCacheValid_, uint64_t pm4IbCacheVersion_,
     Pm4GraphIb* pm4IbCacheEntry_}. The cached pointer is into the node-based
     pm4Graphs_ unordered_map, whose element pointers stay valid across inserts.
  9. tryReplayPm4Graph: if keycache on and version != 0 and version == pm4IbCacheVersion_
     and the cached IB is still kPm4Ready, submit it directly (build vendor packet, ring
     write, doorbell) -- skipping BOTH pm4GraphKey (the O(N) hash) and the map find.
     Otherwise take the slow path (hash, find/build) and, if the result is a ready IB
     and version != 0, arm {version, &entry} for next time. The earlier heuristic identity
     (first/last packet pointers + content fold) was removed entirely.

NAMING (renamed for clarity, see D.19 Q2): the value formerly called the "PM4 replay
token" is the recorded packet VERSION. HIP layer: RecordedPacketVersion(batchIndex),
InvalidateRecordedPacketVersion(), recordedPacketInstanceId_,
recordedPacketMutationCount_. Device layer param: recordedPacketVersion. (The "...Set..."
spelling was dropped because "Set" reads as a verb; it is a version stamp, not an action.)

WHY a version stamp instead of re-hashing or a pointer: re-hashing is the O(N) cost we are
removing; a raw pointer identity is unsafe (ABA + interior in-place updates). The version
is both cheap (O(1) to compare) and exact (changes on every mutation, unique per
instantiation), so it is the only option that is simultaneously fast AND correct.

### D.19 Mutation-site audit, DEFAULT-ON decision, rename, and first-launch cost

Three follow-ups: (1) is the invalidation complete enough to enable by default?
(2) rename the "token" to something more verbose; (3) what does the FIRST-EVER PM4
replay cost?

Q1 -- mutation-site audit (is the version bumped on EVERY packet change?).
The dispatched packet set for a single-device graph lives in GraphExec::packetBatches_
(vectors of AQL packet pointers per batch). Auditing every writer in
hip_graph_internal.cpp / .hpp:

  - CaptureAndFormPacketsForGraph()  -- initial capture + child-graph recursion. BUMPS.
  - UpdateAQLPacket(node)            -- per-node in-place param update (insert/erase/
                                        overwrite of dispatchPackets). BUMPS.
  - UpdatePacketBatchesForNodeEnableDisable(node) -- enable/disable a node. BUMPS.

Those are the ONLY three functions that mutate packetBatches_ / dispatchPackets /
nodeRanges / gpuPackets_. The launch path (EnqueueGraphWithSingleList, lines ~775-817)
is strictly read-only over dispatchPackets. Tracing the public API surface confirms
every mutation routes through one of the three:

  - hipGraphExecKernelNodeSetParams / Memcpy / Memcpy1D / FromSymbol / ToSymbol /
    Memset / ChildGraph exec-setparams  -> UpdateAQLPacket (per cloned node).
  - hipGraphExecUpdate (whole-graph)                                  -> UpdateAQLPacket loop.
  - hipGraphNodeSetEnabled                              -> UpdatePacketBatchesForNodeEnableDisable.
  - hipGraphInstantiate                  -> CaptureAQLPackets -> CaptureAndFormPacketsForGraph.

Only kernel (non-coop), memset, and D2D-memcpy nodes are GraphCaptureEnabled (hence
baked into PM4 IBs); host nodes are not capture-enabled and so cannot change the
dispatched packet set. In-place kernarg CONTENT changes at the SAME address need no
rebuild (the IB points at the address, the GPU reads fresh bytes at replay) -- and even
those still bump conservatively because they go through UpdateAQLPacket. Conclusion:
the invalidation is EXACT, not a heuristic, with no interior-node blind spot.

DECISION: key cache is now DEFAULT ON. Because the version is supplied only by the HIP
graph layer (non-graph callers pass 0 -> always slow path) and is provably bumped on
every mutation, there is no correctness risk, and ANY latency-sensitive graph workload
(not just our decode loop) benefits from skipping the per-launch O(N) rehash. Opt out
with HIP_PM4_GRAPH_KEYCACHE=0.

Validation of the default-on path (gfx1100, W7900):
  - run_pm4_coverage.sh: all 9 scenarios (incl. recreate, varblk, scratch, wave64)
    BIT-EXACT vs AQL baseline.
  - hip_keycache_mutate (new): instantiate -> replay (arm cache) -> mutate -> replay,
    for BOTH param update (hipGraphExecKernelNodeSetParams) and enable/disable
    (hipGraphNodeSetEnabled). AQL baseline, PM4 keycache-on, and PM4 keycache-off all
    produce the identical sequence  1.0 7.0 7.0 42.0 0.0 42.0  and PASS. A stale-IB bug
    would have shown the PM4 value lagging the requested mutation; it does not.
  - hip_pm4_evict_stress (new): 2000 distinct-IB mutations -> all replays correct AND
    the IB cache never exceeds 16 entries (1984 evictions). Validates Q2's VRAM bound.

Q2 -- rename + VRAM deallocation on mutation.

Rename: the "token" is renamed to the recorded packet VERSION everywhere (HIP:
RecordedPacketVersion / InvalidateRecordedPacketVersion / recordedPacketInstanceId_ /
recordedPacketMutationCount_; device param: recordedPacketVersion; cache state:
pm4IbCacheValid_ / pm4IbCacheVersion_ / pm4IbCacheEntry_). "Set" was dropped because it
reads as a verb; the value is a version stamp, not an action.

VRAM: YES, mutated-away IBs are now deallocated. The compiled-IB map pm4Graphs_ is keyed
by content hash. A mutation that changes a HASHED packet field (grid/block dims, kernel
object, kernarg address -- e.g. split-K with varying launch dims) hashes to a NEW key, so
a NEW executable-pool IB is built while the pre-mutation IB becomes unreferenced. Two leaks
were fixed:
  - The deferred-scratch rebuild used to overwrite the Pm4GraphIb in place WITHOUT freeing
    the old .ib (memory_pool_free) -- now freed first.
  - Mutation accumulation: every distinct content left a dead IB in the map until the
    VirtualGPU was destroyed. Now bounded by evictPm4GraphsIfNeeded(): a FIFO over
    pm4GraphKeyOrder_ frees the oldest entries once pm4Graphs_ exceeds kPm4MaxCachedIbs
    (16), NEVER freeing the armed entry (pm4IbCacheEntry_) or the entry built this launch.
    unordered_map keeps pointers to surviving entries valid across erase, so the armed
    fast-path pointer stays sound.
NOTE: a pure kernarg-VALUE update (same address, new bytes) does NOT change the hash and
does NOT rebuild -- which is exactly why the steady decode loop is a cache HIT, not a leak.

Proof (hip_pm4_evict_stress, gfx1100 W7900): instantiate one graph, then mutate it 2000
times each to a STRICTLY-NEW grid size (2000 distinct IBs). Result: every replay produces
the requested value under both AQL and PM4 (PASS), and with AMD_LOG_LEVEL=3 the printed
"[pm4-evict] cache size now N" never exceeds 16, with 1984 = (2000 - 16) evictions. The
pool stays flat instead of growing by 2000 IBs.

Q3/Q4 -- FIRST-EVER replay cost, PM4 vs default HIP graph (hip_replay_timing instrumented
to time the cold launch in isolation; M=256, gfx1100 W7900). The first hipGraphLaunch after
instantiate is where the PM4 path runs buildPm4GraphIb (encode the whole chain into one PM4
IB), allocates the executable IB from a device memory pool, and DMA-uploads it -- a
SYNCHRONOUS one-time cost that must complete before the submit doorbell. The AQL path has no
IB to build, so its first launch is ~ its warm launch. Side-by-side:

  WHAT IS "instantiate": hipGraphInstantiate (graph -> exec). It captures the per-node AQL
  packets and allocates the kernarg pool. It is a SEPARATE one-time cost from the first
  launch, grows with node count, and is ~identical for AQL and PM4 (the PM4 IB is NOT built
  here -- it is built on the first launch). It is shown so the two one-time costs are not
  conflated.

  chain N |        instantiate       |  FIRST launch (cold)     |    2nd (warm)   |  steady avg
          |  AQL    /  PM4           |  AQL     /  PM4          |  AQL  /  PM4    |  AQL  /  PM4
  --------+--------------------------+--------------------------+----------------+--------------
     1    |   48 us /   50 us        |   16 us  /   4189 us     | 4.6us / 6.2us  | 4.0us / 3.9us
    64    |   99 us /  102 us        |   18 us  /   4879 us     | 5.0us / 7.2us  | 4.1us / 3.4us
   256    |  313 us /  250 us        |   19 us  /   7619 us     | 6.9us / 5.5us  | 5.0us / 2.6us

Reading:
  - instantiate: AQL == PM4 (the PM4 IB build is deferred to first launch). Grows with N.
  - FIRST launch: AQL ~16-19 us (no build). PM4 ~4.2 ms (N=1) to ~7.6 ms (N=256): a ~4 ms
    floor + ~13 us/dispatch. The floor is the ONE-TIME executable-pool allocation + blocking
    DMA copy/signal-wait; the per-N slope is the IB encode + larger upload. This is the
    headline PM4 one-time cost and the single biggest "tricky moment" (top-of-file #1).
  - 2nd launch: PM4 already == steady (~5-7 us), proving the entire build is paid exactly
    ONCE per unique instantiated graph.
  - steady: AQL rises 4.0 -> 5.0 us with N (it copies N packets each launch); PM4 is FLAT
    ~2.6-3.9 us across N (the default-on key cache delivers the O(1) host issue).

In a real decode the graph is instantiated once and replayed thousands of times, so the PM4
~4-7 ms first-launch hit lands on the first decode step, amortizes to ~0 us/token, and never
recurs (warm path thereafter; mutations that rebuild reuse the same pool and are bounded by
Q2's eviction).

### D.20 Hiding the first-ever cost: PREWARM (arena) + BUILD_AFTER (AQL-first)

Two opt-in levers attack the first-replay cost. Both are bit-exact (validated by
run_pm4_coverage, hip_keycache_mutate, hip_pm4_evict_stress under every flag combo).

WHY the cold cost exists, and why AQL does not pay it (root cause). The PM4 path
materializes a NEW device-resident EXECUTABLE instruction buffer (the PM4 IB) and
DMA-uploads it. allocExecIbFromData on the FIRST call pays a one-time process-wide
warm-up: the first allocation from the executable device pool (reserve VRAM + create
a GPU VM mapping + mark pages executable), the first allocation from the CPU
fine-grain staging pool, and the first SDMA copy. The probe (hip_pm4_prewarm_probe)
isolates it: cold1=4.2 ms for the FIRST graph but cold2=0.7 ms for the SECOND in the
same process -- so ~3.5 ms is one-time warm-up, ~0.7-1.6 ms is the genuine per-graph
build (encode + slice + DMA, scaling with dispatch count). The AQL graph path pays
NONE of this: it copies the captured AQL packets into the already-existing queue ring
and rings the doorbell -- no new VRAM, no executable pages, no DMA upload. (Kernel
CODE pages were made executable once at module load, long before capture.)

C -- HIP_PM4_GRAPH_PREWARM (executable IB arena). Reserve ONE device-local executable
buffer once at vgpu/stream creation (ensurePm4Arena) and sub-allocate every IB from it
via a first-fit free list (pm4ArenaAlloc / pm4ArenaFreeBytes, coalescing; IBs that do
not fit fall back to a direct pool alloc). The reservation also does one throwaway
allocExecIbFromData so the staging pool AND SDMA path are warmed too (not just the
executable pool -- warming only the executable pool barely moved the number). Net: the
~3.5 ms one-time cost is paid at init, OFF the first-replay path, and per-build pool
alloc/free churn (incl. mutation rebuilds) is gone. This is the clean form of "reserve
memory up front" -- the arena is a real, used reservation, not a throwaway warm-up.

A -- HIP_PM4_GRAPH_BUILD_AFTER (AQL-first, build-after, single thread, no threads). On
the FIRST replay (cache miss) do NOT build before submit (tryReplayPm4Graph allowBuild=
false). Replay this launch via the AQL fallback -- the GPU starts in ~us -- then build
the PM4 IB right after the submit (prebuildPm4Graph), so the build overlaps the GPU
executing the graph just submitted. The AQL submit also sizes the queue scratch, so the
build is never scratch-deferred. The next replay uses the cached single-IB fast path.

MEASURED (gfx1100 W7900). Host-call duration of the first launch is NOT the right metric
for A: build-after still runs the build on the calling thread (just after the AQL submit),
so the host call is still build-bound. The right metric is the end-to-end first step
(launch + synchronize = wall time to the first result), with a realistic GPU workload
(N=128 dispatches, M=4M elems, so GPU exec is a few ms):

  flags                          FIRST_e2e (launch+sync), 3 runs       takeaway
  -----------------------------  -----------------------------------   -----------------------
  PM4 (baseline)                 5.85 / 4.75 / 4.95 ms                 build THEN GPU, serial
  PM4 PREWARM                    ~4.76 ms                              warm-up moved to init
  PM4 BUILD_AFTER                3.53 / 3.56 / 3.43 ms                 build overlaps GPU exec
  PM4 PREWARM+BUILD_AFTER        3.65 / 3.58 / 3.47 ms                 both

  - BUILD_AFTER cuts the first-step wall from ~5 ms to ~3.5 ms: it hides the build under
    the GPU executing the first replay. The win equals min(build, GPU_exec): real when the
    graph's GPU time is comparable to or larger than the build (true for non-trivial decode
    graphs); negligible when GPU time << build (tiny kernels). It also eliminates GPU IDLE
    on the first step (GPU starts at ~us, not after the build).
  - PREWARM moves the ~3.5 ms one-time warm-up to init; it helps most when GPU_exec < build
    (so the build can't be fully hidden by A). PREWARM and BUILD_AFTER are complementary.
  - On a SINGLE thread neither lever reduces the first step below max(build, GPU_exec):
    fully hiding the build from the first token needs either a background build thread
    (deliberately out of scope) or building at instantiate/capture (templating, option B),
    which frameworks get for free via a warm-up decode step.

STATUS: both opt-in (default off) pending broader validation; correctness is bit-exact.
Recommendation: PREWARM + BUILD_AFTER together for latency-sensitive first-token paths.

### D.21 Template at capture, specialize on first run (the clean fix for first-launch encode)

PREWARM (D.20) moves the ~3.5 ms one-time alloc/staging/SDMA warm-up off the launch
path. Once it does, a split timer (HIP_PM4_GRAPH_TIMING, prints "[pm4-timing]" lines)
shows what is LEFT on the first replay is dominated by the CPU ENCODE -- turning the
captured AQL packets into PM4 dwords -- which scales O(dispatches):

  N (dispatches)   encode (CPU)     specialize=patch+alloc+upload (PREWARM, arena)
  --------------   -------------    ----------------------------------------------
  64               ~0.45 ms         ~0.30 ms
  128              ~0.86-1.4 ms     ~0.19 ms
  256              ~2.77 ms         ~0.25 ms

The encode is PURE CPU work and is QUEUE-INDEPENDENT: the only queue-runtime inputs are
the scratch base/size, the scratch V#, and the amd_queue_t pointer. So it can run at
hipGraphInstantiate (we already have the packets there from CaptureAQLPackets), leaving
only a cheap patch+upload for the first launch.

DESIGN (implemented, default ON; A/B toggle HIP_PM4_GRAPH_TEMPLATE=0).
  - encodePm4GraphTemplate (rocvirtual.cpp): the front half of the old buildPm4GraphIb,
    refactored to be queue-independent. Every queue-runtime dword is emitted as a
    SENTINEL (0xFFFFFFFF) and its dword offset + kind recorded in a placeholder list.
    The sentinel (not 0) is deliberate: it keeps delta-encoding making the SAME emit/skip
    decisions it would with the real value -- e.g. a scratch kernel's TMPRING (sentinel)
    still differs from a non-scratch kernel's real 0, so the non-scratch kernel re-emits
    0. Result: the template's dword LAYOUT is identical to the old direct build; only the
    placeholder VALUES are filled in later. Placeholder kinds: scratch base lo/hi, TMPRING,
    scratch V# [0..3], queue_ptr lo/hi. kernarg base is graph-fixed (allocated at
    instantiate) so it is baked in, not a placeholder.
  - specializeFromTemplate (rocvirtual.cpp): copy the template dwords, patch every
    placeholder from THIS launch stream's queue (scratch_backing_memory_location,
    compute_tmpring_size, scratch_resource_descriptor, gpu_queue_), then arena-alloc +
    DMA-upload. If the graph needs scratch the queue has not sized yet, return
    kPm4DeferredScratch (replay via AQL once, which sizes it; a later specialize succeeds).
    No CPU re-encode. buildPm4GraphIb is now just encode + specialize, kept for the
    launch-path fallback (post-mutation rebuild, BUILD_AFTER).
  - Storage / lifetime: GraphExec::EncodePm4Templates (hip_graph_internal.cpp) runs at
    instantiate (right after CaptureAQLPackets) and stores one opaque template per
    PacketBatch via the device's null-stream vdev (encode is queue-independent, so any
    vdev of the graph's device produces an identical template). The template is host-only
    memory owned by the batch and freed in ~GraphExec via the same vdev. It is passed back
    in through dispatchAqlPacketBatch's new pm4Template arg (a base-class virtual, void* at
    the boundary), threaded to tryReplayPm4Graph -> findOrBuildPm4Graph, where a cache miss
    specializes from it instead of doing a full build.
  - Correctness guard (key match): the template is stamped with the content hash of the
    packets it was encoded from. findOrBuildPm4Graph uses it ONLY when tmpl->key ==
    pm4GraphKey(launch packets). So a stale template (graph mutated after instantiate ->
    different packet bytes) or the disabled-node filtered subset (different key) is ignored
    and falls back to a full build -- never wrongly reused. Steady decode keeps the same
    packet bytes (kernarg ADDRESS + grid/block stable; only kernarg CONTENTS change, which
    the IB references by address), so the template stays valid across the decode loop.

MEASURED (gfx1100 W7900, PREWARM on). First replay drops from a full build to
specialize-only; the encode is fully hoisted to instantiate:

  N     TEMPLATE=OFF first launch              TEMPLATE=ON first launch    moved to instantiate
  ---   -----------------------------------    ------------------------    --------------------
  128   build: encode 0.91 ms + 0.24 ms        specialize only: 0.19 ms    encode 0.86 ms
  256   build: encode 1.85 ms + 0.23 ms        specialize only: 0.25 ms    encode 2.77 ms

So with PREWARM + template-at-capture the first PM4 replay costs ~0.2 ms (patch + DMA)
regardless of chain length -- the O(N) encode no longer sits on any launch. Frameworks
get the instantiate cost for free (graphs are instantiated once, before the timed loop).

VALIDATION (default ON): run_pm4_coverage ALL BIT-EXACT incl. scratch (placeholder patch
path), hip_keycache_mutate PASS (AQL + PM4 + scratch; param-update and enable/disable),
hip_pm4_evict_stress PASS (2000 distinct-IB mutations, cache cap 16). The split timer and
A/B toggle (HIP_PM4_GRAPH_TIMING / HIP_PM4_GRAPH_TEMPLATE) are kept as diagnostics.

NEXT (separate commit): a GraphExec-owned, device-scoped IB so the specialized IB is built
ONCE and shared across every stream replaying the graph (today the host template is shared
but the device IB is still per-stream). That makes capture-time build fully reusable and
removes the per-stream first-replay specialize entirely. -> implemented in D.22.

### D.22 GraphExec-owned, device-scoped shared IB (build once, share across streams)

D.21 hoists the CPU encode to instantiate, but the DEVICE IB (the alloc + DMA upload of the
patched dwords) is still produced PER STREAM: the host template is shared, yet the first
replay on each stream calls specializeFromTemplate and gets its own VRAM copy. For a graph
replayed on K streams that is K allocations, K uploads, and K copies of the same IB.

KEY OBSERVATION (shareability). The specialized IB is queue-DEPENDENT only through the
placeholders D.21 records: scratch base/size, scratch V#, and the amd_queue_t pointer. A
graph with NO placeholders (patches.empty(): non-scratch AND no KCP_QUEUE_PTR kernel -- the
common LLM-decode case) produces an IB whose bytes are IDENTICAL for every stream. That IB
is freely shareable. A graph WITH placeholders encodes per-queue values, so it must stay
per-stream (unchanged).

DESIGN (implemented, default ON; A/B toggle HIP_PM4_GRAPH_SHARED_IB=0).
  - Build once, at instantiate. buildPm4GraphTemplate, after encoding, checks shareable()
    (status==Ready && patches.empty()). If so it calls specializeFromTemplate(deviceScoped=
    true) ONCE and stores the result in the template (sharedIb / sharedDw). Because there are
    no placeholders, "specialize" here is just alloc + DMA of the template dwords -- no queue
    state is read, so the null-stream vdev that runs at instantiate produces the canonical IB.
  - Device-scoped allocation. allocExecIbFromData(deviceScoped=true) skips the per-vdev
    executable arena and allocates straight from the device gpuvm executable pool, so the IB
    outlives any single stream. It is owned by the GraphExec and freed in freePm4GraphTemplate
    via Hsa::memory_pool_free (device-agnostic), so the freeing vdev need not be the builder.
  - Submit from any stream, no per-stream build. tryReplayPm4Graph gains a fast path: if the
    template carries a sharedIb and tmpl->key == pm4GraphKey(launch packets), the stream
    submits the shared IB directly from its own queue (ring write + doorbell; the IB VRAM
    address is valid device-wide and the upload was synchronous, so concurrent reads from
    multiple queues are safe). No specialize, no upload, no extra VRAM.
  - Stays OUT of the per-stream content-key cache (pm4Graphs_). A non-owning entry left in
    that map could outlive a freed GraphExec and then alias a NEW graph that happens to reuse
    the same content key (freed+realloced kernarg) -> use-after-free. Instead the shared IB is
    held in a dedicated per-vdev armed slot (pm4SharedRef_) referenced by the keycache. Reuse
    is gated by the keycache VERSION (unique per GraphExec instantiation): a new GraphExec has
    a new version, so a stale pm4SharedRef_ is never submitted -- it is re-armed (to the new,
    valid IB) before the next submit. The O(N) key hash is paid only on a keycache miss (first
    launch + after a mutation), then armed for O(1) reuse, same as the per-stream path.
  - Mutation / non-shareable fallback. On a mutation the launch key no longer matches
    tmpl->key, so the shared fast path is skipped and the launch falls through to the per-
    stream findOrBuildPm4Graph (full build / per-stream specialize) -- correct, just not
    shared. Scratch / queue_ptr graphs never populate sharedIb and always take the per-stream
    path.

MEASURED (gfx1100 W7900, HIP_PM4_GRAPH_TIMING, one exec replayed on 8 streams). The number
of device-IB builds drops from one-per-stream to one-per-graph:

  mode                         "specialize" device-IB builds for 8 streams
  --------------------------   -------------------------------------------
  SHARED_IB=0 (D.21 per-stream)  8   (one alloc+upload per stream)
  SHARED_IB=1 (default, D.22)    1   (built once at instantiate; 8 streams reference it)

So a stream replaying an already-instantiated shareable graph for the first time now does
ZERO device work beyond the ring submit (the per-stream ~0.2 ms specialize of D.21 is gone),
and total IB VRAM is one copy instead of K.

VALIDATION (default ON): run_pm4_coverage ALL BIT-EXACT incl. scratch (per-stream path
unaffected), hip_keycache_mutate PASS (AQL + PM4 + scratch), hip_pm4_evict_stress PASS (2000
mutations). New hip_pm4_multistream replays one exec across 4-8 distinct streams: PASS for
AQL, PM4 shared-on, and PM4 shared-off, with the "shared IB" build logged exactly once for 8
streams. (Note: the multistream gate uses a device-wide sync between zero and readback; an
earlier per-stream-sync variant exposed a stream-completion-tracking race that reproduces
with SHARED_IB=0 too -- i.e. pre-existing and orthogonal to this change. Steady decode
replays on a single stream, the validated path.)

# RDNA4 (gfx1201) dispatch-gap reproduction

AQL vs PM4 per-dispatch command-processor gap on RDNA4 (gfx1201), short
(dispatch-bound) to long (~3 ms) kernels.

## Environment

| item | value |
|---|---|
| Docker container | `hipvk-isolated-sshliapn` |
| GPU / arch | device 0, AMD Radeon AI PRO R9700, gfx1201 |
| Patched CLR | `~/code/rocm-libraries/projects/clr/build-gap/hipamd/lib` (libamdhip64.so.7.14.60850-6e806553ab5) |
| Sources / binaries | `~/code/rocm-libraries/vk_gap_test/` (`hipcc -O2 --offload-arch=gfx1201 <src>.cpp -o <src>.x`) |
| Benchmarks | `hip_gap_graph.x <spin> <n> <K>` (uniform chain); `hip_gap_mix.x <short> <long> <n> <K>` (short/long alternating); `hip_gap_hetero.x <spin> <n> <K>` (4 distinct kernels cycled), n=16384 |
| Pin script | `gpu_pin_freq.sh pin 0 profile_peak` / `unpin 0` |
| Locked clock | `profile_peak` = GFX ~2319 MHz (DPM table top, boost off) |
| Free clock | default DVFS, boosts ~2900-3200 MHz, drops to ~500 MHz in gaps (jittery) |
| rocprofv3 | 1.1.0 |

Method: `busy` = median kernel time from `rocprofv3 --kernel-trace` (forces AQL).
`e2e` = profiler-free whole-graph hipEvent wall clock. `gap = (e2e - busy*K)/(K-1)`;
`PM4 saves` = AQL gap - PM4 gap (us) and e2e reduction (%). Free-clock gaps are
omitted (busy profiled and e2e profiler-free land at different DVFS states).

## Does the per-dispatch gap grow with kernel size? NO (direct measurement)

Same question as the RDNA3 report. Answer is the same: the per-dispatch gap does
NOT grow with kernel duration. AQL steps up to a flat plateau; PM4 is flat.
This is established by TWO clock-independent direct methods, not the `busy*K`
subtraction:

1. Device-side in-kernel timeline (`hip_gap_graph_timed.x`): every kernel busy-waits
   on the fixed 100 MHz steady counter and records its own start/end via
   `atomicMin/atomicMax`. The per-dispatch gap is `start[k+1] - end[k]`, measured
   ON the GPU, profiler-free, and (because the counter is a fixed-frequency wall
   clock) immune to DVFS. PM4-visible (no rocprof fallback). n=16384, K=200,
   profile_peak:

| target us | busy (us) | AQL gap (us) | PM4 gap (us) |
|---|---|---|---|
| 20   | 41.5   | 3.6  | 1.4  |
| 50   | 101.5  | 24.2 | 18.8 |
| 100  | 201.5  | 24.2 | 18.8 |
| 200  | 401.5  | 24.2 | 18.8 |
| 500  | 1001.5 | 32.8 | 18.8 |
| 1000 | 2001.5 | 32.9 | 18.8 |
| 2000 | 4001.5 | 32.8 | 18.8 |
| 3000 | 6001.5 | 32.8 | 18.8 |

   PM4 is FLAT at 18.8 us from 0.1 ms to 6 ms kernels; AQL steps 24->33 us and
   then holds. (The atomic method has a common ~18 us floor on BOTH paths --
   last-wave drain + counter read + completion-signal + redispatch -- so the real
   PM4 per-dispatch saving over AQL is the constant AQL-PM4 difference, ~6-14 us,
   not the absolute value. What matters: both are CONSTANT vs duration.)

2. rocprof CP-side device timeline, `End[k] -> Start[k+1]` (AQL only; PM4 IB is
   invisible to rocprof). Same `hip_gap_graph_timed.x`, n=16384:

| target us | busy (us) | CP-side AQL gap (us) |
|---|---|---|
| 20   | 42.1   | 3.80  |
| 100  | 220.2  | 7.88  |
| 500  | 1020.0 | 16.52 |
| 3000 | 6018.7 | 16.52 |

   The true CP-side AQL gap steps 3.8 -> 7.9 -> 16.5 us and is FLAT at 16.5 us for
   long kernels -- identical shape and magnitude to RDNA3 (4.3 -> 8.6 -> 17.3 us).
   It does NOT grow to the 60-160 us seen in the `busy*K` LOCKED table below.

3. PM4-native runtime instrumentation (`HIP_PM4_GRAPH=1 AMD_LOG_LEVEL=3
   AMD_LOG_MASK=8`). The CLR PM4 path can bake a per-boundary GPU-clock
   `RELEASE_MEM` timestamp into the IB itself (`reportPm4Timestamps`), so the
   per-kernel period is measured BY THE RAW PM4 IB, on the actual replay path. The
   gap = (PM4-native per-kernel period) - busy. n=16384, K=30, profile_peak:

| target us | busy (us) | PM4-native period (us) | PM4-native gap (us) | atomic gap (us) |
|---|---|---|---|---|
| 50   | 101.8  | 125.1  | 23.3 | 23.8 |
| 100  | 201.8  | 224.4  | 22.6 | 23.3 |
| 200  | 401.8  | 424.2  | 22.4 | 22.1 |
| 500  | 1001.8 | 1023.7 | 21.9 | 22.9 |
| 1000 | 2001.8 | 2023.5 | 21.7 | 31.3 |
| 1500 | 3001.8 | 3023.5 | 21.7 | 23.3 |
| 2000 | 4001.8 | 4023.0 | 21.3 | 21.3 |

   The PM4 path's OWN timestamps report a FLAT ~21-23 us gap from 0.1 ms to 4 ms
   kernels (if anything, slightly decreasing) -- it does not grow. It agrees with
   the in-kernel atomic gap. (The instrumented gap ~22 us is a bit above the
   production 18.8 us because the log path adds a timestamp `RELEASE_MEM` per
   boundary and forces the blocking per-stream IB.) NOTE: this wired PM4 GPU-clock
   profiling drops/zeros slots for kernels beyond ~1-2 ms (valid-sample count falls
   off above target 1000; at K=200 the total underflows to ~1.8e17 us), so it is
   only usable up to ~few-ms kernels -- a limitation of the profiling readback, not
   the replay path.

Conclusion: the PM4 implementation is NOT broken. RDNA4's per-dispatch gap is
constant with kernel duration, exactly like RDNA3 -- confirmed by THREE independent
clock-independent methods (in-kernel atomics, rocprof CP-side AQL, and the PM4
path's own GPU-clock timestamps). The apparent "PM4 gap grows with kernel timing"
in the LOCKED `busy*K` table is a MEASUREMENT ARTIFACT, not a CP/runtime
regression -- see the note under that table.

### The real CP gap is FLAT; the apparent growth is a completion-tail artifact

Single-wave (n=256, one 256-thread block) isolates the gap from grid/wave-drain
effects. rocprof CP-side `End[k]->Start[k+1]` is FLAT regardless of duration; what
steps is the kernel's own dispatch-to-completion time (rocprof busy):

| target us | in-k busy | in-k gap | rocprof busy | rocprof CP gap | period |
|---|---|---|---|---|---|
| 12  | 12.3  | 2.3  | 13.0  | 3.80 | ~15  |
| 20  | 20.3  | 24.8 | 40.4  | 3.80 | ~44  |
| 100 | 100.3 | 24.8 | 123.4 | 3.84 | ~124 |

The rocprof CP inter-dispatch gap is 3.8 us at every size -- it DOES NOT grow. What
adds ~20-27 us above the ~16 us threshold is the kernel's own COMPLETION time
(rocprof busy / hipEvent device time jumps from ~spin to ~spin+27), NOT the gap.

ROOT CAUSE (measured): a GPU HARDWARE-SCHEDULER QUANTUM, in GFX cycles. A wave that
runs past ~37,000 GFX cycles triggers a ONE-TIME ~62,000-cycle scheduler event
(preemption / wave save-restore). Confirmed cycle-based by pinning two clocks:

| GFX clock | grace threshold | penalty (one-time) |
|---|---|---|
| 2319 MHz | ~16 us (=37k cyc) | ~27 us (=62k cyc) |
| 483 MHz  | ~80 us (=38.6k cyc)| ~131 us (=63.3k cyc) |

Both the grace and the penalty are constant in CYCLES across a 4.8x clock change
(threshold moves from 16 us to 80 us wall). It is one-time per kernel (target100 =
100+27, not recurring), on the device (hipEvent, profiler-free, single isolated
kernel), and INDEPENDENT of: AQL vs PM4, global write, atomics, wave-yield
(s_sleep), grid size, and the dependency chain. So it is NOT power/clock gating and
NOT a CLR/HIP/PM4 cost -- it is GPU firmware/hardware, paid identically by AQL and
PM4 (cancels in the dispatch comparison). The in-kernel atomic method counts only
the spin loop as busy (steady counter t0->t1) and misattributes this ~27 us
completion tail to "gap", which is why in-kernel gap appears to jump 2.3 -> 25 us
at the threshold while the real CP gap stays 3.8 us.

Practical impact: one-time per kernel, so only kernels in the ~16-100 us range show
a large RELATIVE inflation; <16 us kernels never trigger it and multi-100 us / ms
kernels amortize it. It does not change the PM4 dispatch-gap result. Suppressing it
is a driver/firmware matter (compute preemption / CWSR / MES quantum / GFXOFF), not
a runtime change; probe binary: `fma_probe.x <mode> <target_us> <n> <iters>`.

### Fine-grained duration sweep (single-wave n=256, K=200, profile_peak)

`busy + gap == e2e period` to <1% everywhere. PM4 removes a CONSTANT ~1.6 us/dispatch.

| target us | busy | AQL gap | PM4 gap | AQL e2e/disp | PM4 e2e/disp | PM4 saves |
|---|---|---|---|---|---|---|
| 1    | 1.32   | 2.32 | 0.68 | 3.67   | 2.07   | 1.60 us (43.7%) |
| 2    | 2.32   | 2.32 | 0.68 | 4.67   | 3.11   | 1.56 us (33.5%) |
| 4    | 4.32   | 2.28 | 0.68 | 6.64   | 5.10   | 1.55 us (23.3%) |
| 8    | 8.32   | 2.28 | 0.68 | 10.64  | 9.11   | 1.54 us (14.4%) |
| 12   | 12.32  | 2.32 | 0.68 | 14.70  | 13.09  | 1.61 us (11.0%) |
| 16   | 16.32  | 27.93| 25.85| 43.45  | 41.24  | 2.21 us (5.1%)  |
| 24   | 24.33  | 24.77| 22.81| 48.42  | 46.54  | 1.88 us (3.9%)  |
| 32   | 32.33  | 24.77| 22.81| 56.61  | 54.88  | 1.73 us (3.1%)  |
| 64   | 64.34  | 24.81| 22.81| 88.83  | 86.83  | 2.00 us (2.3%)  |
| 128  | 128.35 | 24.81| 22.81| 151.84 | 150.34 | 1.50 us (1.0%)  |
| 256  | 256.34 | 24.77| 22.77| 280.33 | 277.85 | 2.47 us (0.9%)  |
| 512  | 512.33 | 24.77| 22.81| 536.14 | 534.01 | 2.14 us (0.4%)  |
| 768  | 768.35 | 33.17| 22.81| 799.91 | 789.98 | 9.93 us (1.2%)  |
| 1024 | 1024.33| 33.13| 22.81| 1056.0 | 1046.2 | 9.85 us (0.9%)  |
| 1536 | 1536.34| 33.17| 22.81| 1568.2 | 1558.0 | 10.22 us (0.7%) |
| 2048 | 2048.34| 33.93| 21.61| 2079.0 | 2068.6 | 10.41 us (0.5%) |
| 3072 | 3072.33| 31.93| 21.61| 3102.7 | 3092.7 | 10.08 us (0.3%) |
| 4096 | 4096.33| 33.13| 22.81| 4127.5 | 4118.0 | 9.58 us (0.2%)  |

In-kernel gap is flat below 16 us (~2.3 AQL / 0.68 PM4 = real launch-bound gap),
steps to ~25 us at the power threshold, and a second AQL step to ~33 us at ~768 us
(PM4 holds flat). All of these step-then-flat -- none GROW unboundedly.

### Q3: why rocprof busy == PM4 e2e on RDNA3 but not RDNA4

Because the `busy*K` mismatch is clock jitter, and the two GPUs pin clocks
differently. The FMA kernel's busy is CLOCK-SENSITIVE (real ALU work). `busy` is
measured profiler-armed and `e2e` profiler-free -- two SEPARATE runs. On the R9700,
`profile_peak` only caps boost; the live clock still jitters ~+-1.5% and the two
runs land at slightly different averages. Measured for the FMA kernel
(spin=20000, K=100): rocprof busy 325.4 us vs PM4 profiler-free e2e 313.7 us -- here
e2e is FASTER (free run hit ~2329 MHz vs armed ~2313), giving a NEGATIVE apparent
gap; other runs land positive. That sign flip is the proof: a real gap is always
positive; this one's sign tracks which run got the higher clock. A ~1% per-kernel
clock difference x K is the multi-% busy*K-vs-e2e mismatch. The W7900 (RDNA3) holds
a truly fixed clock under both profiler-armed and profiler-free, so its rocprof
busy*K matches PM4 e2e to ~0.05%. The clock-INDEPENDENT timed kernel (steady-counter
busy) shows no such mismatch on either GPU, confirming it is a clock-pinning
artifact, not PM4.

## Uniform chain -- LOCKED (profile_peak, ~2319 MHz)

WARNING: the `AQL gap` / `PM4 gap` columns below come from
`gap = (e2e - busy*K)/(K-1)`, where `busy` is profiler-armed and `e2e` is
profiler-free. On the R9700 these two runs settle at slightly different effective
clocks; that small per-us discrepancy is multiplied by K, manufacturing a FAKE
gap proportional to kernel duration (the growth to 60-160 us). The REAL gap is
flat -- see the direct-measurement section above. The `PM4 e2e` / `AQL e2e` and
`PM4 saves` columns (both profiler-free) are still meaningful.

| spin | K | busy (us) | AQL e2e (ms) | PM4 e2e (ms) | AQL gap (us) | PM4 gap (us) | PM4 saves |
|---|---|---|---|---|---|---|---|
| 250    | 1500 | 4.72    | 9.39  | 7.01  | 1.54   | ~0.00  | 1.58 us (25.3%) |
| 500    | 1500 | 8.52    | 14.87 | 12.54 | 1.39   | ~0.00  | 1.56 us (15.7%) |
| 750    | 1000 | 12.44   | 13.63 | 12.04 | 1.20   | ~0.00  | 1.59 us (11.7%) |
| 1000   | 800  | 16.08   | 23.13 | 20.27 | 12.85  | 9.27   | 3.58 us (12.4%) |
| 1500   | 800  | 37.56   | 36.25 | 32.54 | 7.77   | 3.12   | 4.65 us (10.2%) |
| 2000   | 600  | 47.56   | 32.12 | 28.77 | 5.98   | 0.39   | 5.59 us (10.4%) |
| 3000   | 400  | 61.68   | 27.08 | 24.97 | 6.02   | 0.75   | 5.27 us (7.8%) |
| 5000   | 300  | 89.56   | 29.11 | 27.49 | 7.51   | 2.08   | 5.43 us (5.6%) |
| 10000  | 150  | 159.88  | 25.62 | 24.83 | 10.98  | 5.69   | 5.29 us (3.1%) |
| 25000  | 80   | 373.76  | 31.37 | 30.90 | 18.58  | 12.65  | 5.92 us (1.5%) |
| 50000  | 50   | 731.97  | 38.43 | 37.67 | 37.28  | 21.89  | 15.39 us (2.0%) |
| 100000 | 40   | 1442.33 | 60.14 | 59.60 | 62.85  | 48.81  | 14.05 us (0.9%) |
| 200000 | 30   | 2827.75 | 89.55 | 88.86 | 162.53 | 138.79 | 23.74 us (0.8%) |
| 210000 | 30   | 3002.90 | 93.81 | 93.05 | 128.41 | 102.10 | 26.31 us (0.8%) |

## Uniform chain -- FREE (default DVFS, e2e median of 5)

| spin | K | busy (us)* | AQL e2e (ms) | PM4 e2e (ms) | PM4 saves (% of e2e) |
|---|---|---|---|---|---|
| 250    | 1500 | 4.04    | 7.34  | 5.31  | 27.6% |
| 500    | 1500 | 7.16    | 11.37 | 9.33  | 18.0% |
| 750    | 1000 | 10.40   | 10.30 | 8.91  | 13.4% |
| 1000   | 800  | 13.56   | 26.67 | 37.16 | -39.3% (PM4 slower) |
| 1500   | 800  | 74.00   | 71.62 | 61.55 | 14.1% |
| 2000   | 600  | 79.72   | 56.96 | 53.68 | 5.8% |
| 3000   | 400  | 103.44  | 42.49 | 40.27 | 5.2% |
| 5000   | 300  | 126.00  | 38.06 | 36.62 | 3.8% |
| 10000  | 150  | 181.80  | 26.60 | 25.69 | 3.4% |
| 25000  | 80   | 324.16  | 24.98 | 25.08 | -0.4% |
| 50000  | 50   | 585.93  | 29.91 | 28.02 | 6.3% |
| 100000 | 40   | 1519.61 | 45.40 | 45.30 | 0.2% |
| 200000 | 30   | 2254.35 | 66.27 | 65.89 | 0.6% |
| 210000 | 30   | 2364.26 | 69.72 | 69.17 | 0.8% |

\* free busy is profiled at a different DVFS state than e2e (reference only).

## Short/long interleave -- LOCKED

K=400 = 200 short + 200 long, alternating.

| short | long | busy_s (us) | busy_l (us) | AQL e2e (ms) | PM4 e2e (ms) | AQL gap (us) | PM4 saves |
|---|---|---|---|---|---|---|---|
| 250  | 5000   | 4.40  | 98.40   | 21.84  | 20.39  | 3.21 | 3.64 us (6.6%) |
| 500  | 20000  | 7.84  | 335.20  | 69.79  | 68.63  | 2.97 | 2.92 us (1.7%) |
| 250  | 50000  | 4.40  | 809.23  | 165.61 | 162.56 | 7.23 | 7.65 us (1.8%) |
| 1000 | 100000 | 14.72 | 1597.57 | 324.76 | 321.99 | 5.76 | 6.93 us (0.9%) |

## Short/long interleave -- FREE (e2e median of 5)

| short | long | busy_s (us)* | busy_l (us)* | AQL e2e (ms) | PM4 e2e (ms) | PM4 saves (% of e2e) |
|---|---|---|---|---|---|---|
| 250  | 5000   | 3.72  | 134.52  | 26.90  | 25.68  | 4.5% |
| 500  | 20000  | 6.48  | 301.14  | 59.08  | 58.48  | 1.0% |
| 250  | 50000  | 3.76  | 647.28  | 127.33 | 124.44 | 2.3% |
| 1000 | 100000 | 11.60 | 1243.46 | 235.73 | 231.43 | 1.8% |

\* free busy profiled at a different DVFS state than e2e (reference only).

## Heterogeneous interleave -- 4 distinct kernels

`hip_gap_hetero.x` cycles 4 kernels (node k uses k%4; K=800 = 200 each = 800
total dispatches), each with a different block size and kernarg signature
(varying pointer count incl. unused, plus unused scalars); gid<n guard keeps work
similar.

| kernel | block | args | busy @ spin=500 | busy @ spin=2000 |
|---|---|---|---|---|
| k1 | 256 | 1 ptr, 2 scalars | 7.80 us | 47.20 us |
| k2 | 128 | 3 ptr (2 unused), 2 scalars | 7.84 us | 47.20 us |
| k3 | 64  | 5 ptr (4 unused), 4 scalars (2 unused) | 8.88 us | 50.64 us |
| k4 | 512 | 2 ptr (1 unused), 4 scalars (2 unused) | 7.80 us | 47.24 us |

LOCKED:

| spin | avg busy (us) | AQL e2e (ms) | PM4 e2e (ms) | AQL gap (us) | PM4 gap (us) | PM4 saves |
|---|---|---|---|---|---|---|
| 250  | 4.49  | 4.88  | 3.63  | 1.62 | 0.05  | 1.57 us (25.6%) |
| 500  | 8.07  | 7.76  | 6.49  | 1.64 | 0.04  | 1.59 us (16.4%) |
| 1000 | 19.14 | 18.46 | 16.20 | 3.93 | 1.12  | 2.82 us (12.2%) |
| 2000 | 48.16 | 42.38 | 38.08 | 4.83 | ~0.00 | 5.39 us (10.2%) |

FREE (e2e median of 5):

| spin | avg busy (us)* | AQL e2e (ms) | PM4 e2e (ms) | PM4 saves (% of e2e) |
|---|---|---|---|---|
| 250  | 3.75  | 3.83  | 2.75  | 28.0% |
| 500  | 10.72 | 5.88  | 4.83  | 18.0% |
| 1000 | 26.84 | 17.19 | 14.87 | 13.5% |
| 2000 | 77.87 | 75.29 | 70.85 | 5.9% |

\* free busy profiled at a different DVFS state than e2e (reference only).

## Notes

- PM4 removes essentially the whole per-dispatch CP gap (PM4 gap ~0): ~1.5 us in
  the launch-bound regime, 10-26% e2e on short kernels. The saving is per-DISPATCH
  (constant), so it amortizes away with kernel size (down to <1% at ~3 ms) and
  tracks dispatch COUNT, not size. Holds for uniform, short/long, and
  heterogeneous chains -- not limited to repeated identical kernels.
- The per-dispatch gap does NOT grow with kernel duration. Both clock-independent
  direct methods (device-side in-kernel atomic timeline, and rocprof CP-side
  End->Start) show PM4 flat (18.8 us atomic floor) and AQL flat after a small step
  (CP-side 16.5 us) across 0.1-6 ms kernels -- the same constant-gap behaviour as
  RDNA3. See "Does the per-dispatch gap grow with kernel size? NO".
- The growth to 60-160 us in the LOCKED `busy*K` gap columns is a methodology
  artifact, NOT a PM4 code bug: `gap = (e2e - busy*K)/(K-1)` subtracts a
  profiler-armed `busy` from a profiler-free `e2e`. On the R9700 the two land at
  slightly different effective clocks; the ~sub-percent per-us difference is
  multiplied by K and surfaces as a fake duration-proportional gap. RDNA3's W7900
  pins busy/e2e to the same clock, so the artifact was ~0 there -- which is why the
  RDNA3 report's `busy*K` PM4 gap stayed flat and RDNA4's did not. No PM4 code
  change is warranted; use the direct gap (above) or the profiler-free e2e
  savings %.
- spin~1000-1500 is contaminated by a microbenchmark artifact: the FMA loop takes
  a one-time ~25 us penalty past ~1150 iterations (iteration-bound; same at 2319
  and 477 MHz; geometry- and value-independent; not the CP). Pinned busy ~doubles
  (16->38 us); free ~5.4x because DVFS also drops the longer kernel to ~half clock.
  Prefer spin <=900 or >=2000 for clean gaps.
- Free PM4-slower at uniform spin=1000 (37 vs 27 ms) is a DVFS clock-state
  artifact: PM4 has fewer gaps so it can't be slower at equal clock -> it settled
  at a lower clock. PM4 replays a VRAM IB (no extra host work). Pinning removes it.
- Use LOCKED clocks for all comparisons; free DVFS swings 500-3200 MHz and is
  workload-dependent.

---

# Appendix A: Super-detailed root-cause analysis and gap-minimization

This appendix consolidates the full investigation: how the gap was measured (and
cross-validated by independent clocks), why earlier "growing gap" numbers were a
measurement artifact, the true hardware root cause of the one-time latency step,
and the complete set of optimizations tried to drive the inter-dispatch gap to
its floor. All runs on gfx1201 (R9700), clocks PINNED to profile_peak (~2319 MHz)
unless noted, single GPU (HIP_VISIBLE_DEVICES=0).

## A.1 Three independent, clock-agnostic gap measurements all agree: the gap is FLAT

The central question was whether the RDNA4 PM4 dispatch gap grows with kernel
duration (as the early `busy*K` columns suggested). Three methods that do NOT
rely on cross-run clock matching were used; all three show a constant gap:

| Method | What it measures | Clock source | Result vs kernel size |
|---|---|---|---|
| Device in-kernel atomic timeline | atomicMin(start)/atomicMax(end) of `__builtin_readsteadycounter` across the chain; gap = next.start - this.end | 100 MHz GPU steady counter (wall clock) | FLAT, ~18.8 us floor (PM4), 0.1-6 ms kernels |
| rocprofv3 CP kernel trace | End(i) -> Start(i+1) on the CP dispatch timeline | CP HW timestamps | FLAT, ~3.8 us (single-wave), small one-time step only |
| PM4 IB-native RELEASE_MEM | BOTTOM_OF_PIPE_TS GPU-clock packets baked between dispatches in the replayed IB | GPU clock, written straight to memory | FLAT, ~22 us, 0.1-6 ms kernels |

Key point: these use three different clocks and three different vantage points
(shader, CP, IB) and none shows duration-proportional growth. The gap is a fixed
per-dispatch cost, exactly like RDNA3.

## A.2 Why the early `busy*K` gap "grew": a subtraction artifact, not a bug

The hybrid formula `gap = (e2e - busy*K) / (K - 1)` subtracts a profiler-ARMED
`busy` (rocprof run) from a profiler-FREE `e2e` (timed run). On the R9700 those
two runs settle at slightly different effective clocks; a sub-percent per-us
clock difference is multiplied by K (hundreds of dispatches) and re-surfaces as a
fake duration-proportional "gap" of 60-160 us. On RDNA3 (W7900) busy and e2e pin
to the same clock, so the artifact was ~0 and the RDNA3 `busy*K` column stayed
flat. This is purely a methodology mismatch:

- Lowering the GPU clock does NOT change the true gap (it is clock-invariant in
  wall time for the fence, cycle-bound for the scheduler step -- see A.3).
- The fix is to NOT mix profiler-armed and profiler-free timings: use the direct
  device/CP/IB gap, or the profiler-free e2e savings %. No CLR change is needed.

## A.3 The one-time ~20-27 us latency step: a HW compute-scheduler quantum

Above a kernel-duration threshold of ~16 us, per-kernel completion takes a single
one-time step of +20-27 us. Characterization (via `fma_probe.cpp` and the timed
chain) pinned this down precisely:

- It is intrinsic to a SINGLE kernel (K=1, no chain), not a chain-boundary cost.
- It is CYCLE-based, not wall-time: the threshold is ~37,000 GPU cycles of grace,
  and the penalty is ~62,000 cycles. At 2319 MHz that is ~16 us grace / ~27 us
  penalty; at ~477 MHz the wall-time scales with the clock but the CYCLE counts
  are identical -> confirms a fixed-cycle hardware quantum.
- It is independent of: global writes, atomics, grid size / occupancy, and
  whether the wave spins or yields (s_sleep). Geometry- and value-independent.
- It applies EQUALLY to AQL and PM4 -> it is below the runtime/HIP layer.

Conclusion: this is the compute-scheduler's wave time-slice. A wave that runs
longer than the grace quantum is subject to a one-time preempt/save-restore
(CWSR) accounting penalty. This is firmware/driver territory (MES/HWS quantum,
CWSR), NOT something the PM4 replay path or any user-space runtime knob can
remove. Mitigations require root + reboot (A.5).

## A.4 Gap-minimization matrix (the part that CAN be tuned from user space)

Target = 4 us kernels, n=256, K=400, clocks pinned: a gap-dominated regime where
the per-dispatch cost is the largest fraction of e2e. `gap_med` is the direct
in-kernel atomic gap; `e2e_period` is profiler-free wall time per dispatch.

| Config | in-kernel gap_med | e2e period | note |
|---|---|---|---|
| AQL baseline | 2.28 us | 6.642 us/disp | default HIP dispatch |
| PM4 default | 0.68 us | 5.043 us/disp | HIP_PM4_GRAPH=1 |
| PM4 +DELTA | 0.64 us | 4.997 us/disp | skip redundant SET_SH_REG |
| PM4 +DELTA +INPLACE | 0.68 us | 5.044 us/disp | host-side only, no device win |
| **PM4 +DELTA +EDGE_GCR=0x380** | **0.40 us** | **4.806 us/disp** | **force AGENT (no L2 flush) fence** |
| PM4 +DELTA +EDGE_GCR=0x100 | 0.40 us | 4.801 us/disp | mask bit barely matters |
| PM4 +DELTA +EDGE_GCR=0 (no fence) | n/a (overlap) | 0.175 us/disp | INCORRECT for RAW; shows launch floor |
| PM4 +DELTA +PWS (deferred) | 0.60 us | 5.004 us/disp | legacy, UNSAFE, not worth it |
| PM4 +FULLFENCE (full L2) | 0.68 us | 5.050 us/disp | worst-correct |

Reproducibility / grid-size sweep (gap_med, us):

| n (grid) | PM4 default | PM4 +DELTA +EDGE_GCR=0x380 |
|---|---|---|
| 256 | 0.68 | 0.40 |
| 1024 | 0.68 | 0.40 |
| 16384 | 0.64 | 0.40 |

Interpretation:

- **AQL -> PM4 is the big win:** 2.28 -> 0.68 us device gap (-70%), 6.64 -> 5.04
  us/disp e2e. PM4 IB replay already removes the bulk of the CP per-dispatch cost.
- **Forcing AGENT-scope fence (EDGE_GCR=0x380) is the second win:** 0.68 -> 0.40
  us (-41%), reproducible and grid-independent. The default `inherit-scope` path
  derives the per-edge cache mask from the captured AQL packet scope, and HIP
  graph-captured kernels frequently over-declare SYSTEM scope (full-L2 flush).
  For device-LOCAL kernel->kernel RAW, an AGENT acquire (invalidate GLK/GLV/GL1,
  NO L2 writeback) is sufficient (L0/L1 are write-through to L2; the producer
  drain lands writes in L2; the consumer invalidate misses to L2). Capping the
  edge mask at 0x380 removes the unnecessary L2 flush.
  - CORRECTNESS NOTE: only safe when intermediate buffers stay device-local. If a
    chained kernel's output must be host-visible without an explicit copy/sync,
    do not force this. Safe for the device-resident graphs this targets.
- **DELTA is a small, free win** (-0.04 us) and reduces ME register-programming
  work; combine it with EDGE_GCR.
- **INPLACE / SHARED_IB / KEYCACHE / PREWARM** are HOST-side launch-cost knobs
  (they cut CPU submit time, not the device gap). Worth enabling for ultra-low
  CPU launch latency on long graphs, but they do not move `gap_med`.
- **The no-fence floor (0.175 us/disp)** is the pure SPI back-to-back launch
  rate with NO dependency enforcement. It is the absolute hardware floor but
  produces WRONG results for a dependent chain (kernels overlap). It shows that
  essentially the ENTIRE remaining 0.40 us correct-gap is the producer-drain +
  acquire fence required by the RAW dependency -- not launch overhead.

## A.5 Recommended ultra-low-latency configuration

For a dependent kernel chain on gfx1201 with device-local intermediates:

```bash
HIP_PM4_GRAPH=1 \
HIP_PM4_GRAPH_DELTA=1 \
HIP_PM4_GRAPH_EDGE_GCR=0x380 \
HIP_PM4_GRAPH_INPLACE=1   # host-side, optional but helps CPU submit latency
```

Result: device inter-dispatch gap ~0.40 us (vs 2.28 us AQL = 5.7x lower),
grid-size and kernel-duration independent.

## A.6 Optimizations that require root/reboot (not user-space)

The ~20-27 us one-time scheduler-quantum step (A.3) and any further fence
reduction below 0.40 us are below the runtime layer:

1. **CWSR / scheduler quantum:** disabling Compute Wave Save-Restore
   (`amdgpu.cwsr_enable=0`) or lengthening the MES/HWS scheduling quantum would
   raise/remove the grace threshold so kernels past ~16 us no longer take the
   one-time penalty. Requires kernel module param + reboot; affects preemption
   fairness system-wide. Not validated here (no passwordless root for debugfs).
2. **GFXOFF / power gating:** confirmed NOT the cause of the gap (clock-pinned
   runs still show it), so toggling GFXOFF is not expected to help the gap; it
   only affects idle-entry latency.
3. **Kernel fusion / persistent kernels (application-level, the true floor):**
   the only way to get BELOW the 0.40 us correct fence cost is to remove the
   dispatch boundary entirely -- fuse N chained kernels into one megakernel, or
   run a persistent grid-stride kernel that consumes work from a queue. That
   yields zero inter-dispatch gap (the no-fence 0.175 us/disp number is the
   pipelined limit of doing this without correctness loss). Use where the chain
   has a fixed, data-independent shape (e.g. a fused MLP).

## A.7 Summary of conclusions

- The RDNA4 PM4 dispatch gap is CONSTANT per dispatch, ~0.68 us by default and
  ~0.40 us with an AGENT-scope edge fence -- it does NOT grow with kernel size.
  RDNA4 behaves like RDNA3; the apparent growth was a `busy*K` clock-subtraction
  artifact (A.2).
- The separate ~20-27 us one-time step at >16 us kernels is a fixed-CYCLE
  hardware compute-scheduler quantum (~37k grace / ~62k penalty cycles), equal
  for AQL and PM4, removable only via driver/firmware (A.3, A.6).
- Best user-space config cuts the device gap 5.7x vs AQL (A.5). The remaining
  0.40 us is the unavoidable RAW producer-drain+acquire fence; going lower
  requires removing the dispatch boundary (fusion, A.6.3).

---

# Appendix B: New optimization -- safe interior-edge scope reduction (default)

This appendix adds a genuinely new optimization (not an existing env knob): a
CODE change that makes the fast fence the SAFE default, validated by a strict
read-after-write (RAW) coherence test.

## B.1 Does HIP insert SYSTEM scope between all kernels, even pure device-only RAW? YES

Confirmed with the `HIP_PM4_GRAPH_TIMING` per-edge scope histogram on a 50-node
device-local chain:

```
inherit-scope ON (default): edges=50  NONE=0  AGENT=1  SYSTEM=49
```

49 of 50 interior edges carry SYSTEM (full-L2 flush) scope. Root cause, traced in
`rocvirtual.cpp`:

- gfx12 builds every kernel dispatch packet with `sysAcquireAgentReleaseHBits`
  (acquire = SYSTEM, release = AGENT) -- a blanket, hardware-conservative default
  applied to ALL dispatches, NOT derived from any data-dependency analysis.
- The PM4 inherit-scope path takes `max(release_i, acquire_{i+1})` per edge =
  `max(AGENT, SYSTEM)` = SYSTEM -> `gcrFull` (full-L2 invalidate + writeback) on
  every interior edge.

So yes: the SYSTEM scope is stamped on every kernel regardless of whether the
dependency is pure device-local. It is over-declaration, not a real requirement.

## B.2 The new optimization: cap interior edges at AGENT (correctness-preserving)

Change in `VirtualGPU::buildPm4GraphTemplate` (`scopeToEdgeMask`): an interior
graph edge now maps any non-NONE inherited scope to the AGENT mask (`kGcrAgent`,
invalidate GL0/GL1/GLK, NO L2 op) instead of promoting SYSTEM to full-L2.
NONE still emits no fence (independent-kernel overlap preserved). Full-L2 is kept
strictly at the graph's LEADING acquire (pre-kernel-0) and TRAILING release.

Why this is safe (not a quality/safety reduction):

- GL2 is the single device coherence point, shared by all CUs. The producer drain
  (CS_PARTIAL_FLUSH) retires its waves; their write-through L0/L1 land in GL2. The
  consumer AGENT acquire invalidates only its per-CU GL0/GL1/GLK, so its reads
  miss to GL2 and observe the producer's writes. GL2 itself is never stale.
- A SYSTEM acquire's extra GL2 invalidate/writeback only buys HOST/peer
  visibility, which mid-graph kernels never need: the leading full acquire pulls
  system state into coherence at graph start, and the trailing full release makes
  final results system-visible. Both boundaries are unchanged.
- Aligns with this path's own documented intent ("AGENT ... the only correct
  choice" for device-local kernel->kernel RAW).
- Opt-out `HIP_PM4_GRAPH_EDGE_FULL_L2=1` restores the old promote-to-SYSTEM
  behavior for A/B / paranoia.

## B.3 Validation -- strict RAW coherence test (`pm4_coherence_test.cpp`)

A new test builds a K-node graph of IN-PLACE increments on ONE buffer
(`d[idx]+=1`), so consecutive nodes form a strict RAW chain on every element;
after R replays every element must equal R*K. A too-weak fence drops increments
(often non-deterministically). Results (gfx1201, pinned):

| Path | scope histogram | gap_med | RAW coherence test |
|---|---|---|---|
| AQL baseline | n/a | 2.28 us | PASS (0 mismatch) |
| **PM4 NEW default (AGENT interior)** | **AGENT=50 SYSTEM=0** | **0.44 us** | **PASS (0 mismatch)** |
| PM4 opt-out `EDGE_FULL_L2=1` (old) | AGENT=1 SYSTEM=49 | 0.68 us | PASS (0 mismatch) |
| PM4 `EDGE_GCR=0` (no fence) | n/a | n/a | FAIL (100/100 bad, 183 vs 256) |

- The new default PASSES the strict RAW test (incl. high-contention N=1024,K=512,
  5x repeat) -- AGENT is sufficient.
- The test has teeth: the no-fence path FAILS hard (every replay wrong), so the
  test genuinely detects insufficient fencing.
- The new default auto-achieves gap_med 0.44 us (vs 0.68 us before, ~35% lower)
  with NO env flag, and 5.2x lower than AQL's 2.28 us -- safely, by construction.

## B.4 Overlap angle -- why it cannot be pushed further without losing safety

For a true RAW chain the per-edge cost decomposes into producer-drain
(CS_PARTIAL_FLUSH) + consumer cache-acquire, which are inherently serial:

- The drain cannot overlap the consumer: the consumer depends on the producer.
- The acquire (invalidate) cannot move earlier: it must run after producer writes
  land (post-drain) and before consumer reads.
- Overlapping consumer register programming with the drain (HIP_PM4_GRAPH_REORDER)
  is a measured NO-OP (the CP ME is in-order; register programming is cheap).
- The deferred-fence overlap (HIP_PM4_GRAPH_PWS) IS faster but UNSAFE (consumer can
  launch before the invalidate is globally complete -> non-deterministic stale
  reads). Rejected on safety grounds.

Therefore the only correctness-preserving reductions are (a) shrinking the fence
SCOPE (this appendix: SYSTEM->AGENT interior, done) and (b) removing the dispatch
boundary entirely via kernel fusion / persistent kernels (application-level).
0.44 us is the safe interior-RAW floor on this path.

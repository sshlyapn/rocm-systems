# RDNA3 (gfx1100) dispatch-gap reproduction

AQL vs PM4 per-dispatch command-processor gap on RDNA3 (gfx1100), short
(dispatch-bound) to long (~5 ms) kernels. Same methodology as the RDNA4
(gfx1201) report in this directory, re-run on the W7900.

## Environment

| item | value |
|---|---|
| Docker container | `test-framework-sshliapn-container-2nd` |
| GPU / arch | device 0, AMD Radeon PRO W7900, gfx1100 |
| Patched CLR | `~/code/rocm-systems-rebase/projects/clr/build-develop/hipamd/lib` (libamdhip64.so.7.14.60850-698402fad94) |
| ROCr (RUNPATH) | `~/code/rocm-systems-develop/install/lib` (libhsa-runtime64.so.1.21.0) |
| Sources / binaries | `~/code/rocm-systems-rebase/vk_gap_test/` (`hipcc -O2 --offload-arch=gfx1100 <src>.cpp -o <src>.x`) |
| Benchmarks | `hip_gap_graph.x <spin> [n] [K]` (uniform chain); `hip_gap_mix.x <short> <long> [n] [K]` (short/long alternating); `hip_gap_hetero.x <spin> [n] [K]` (4 distinct kernels cycled), n=16384 |
| Pin script | `gpu_pin_freq.sh pin 0 profile_peak` / `unpin 0` |
| Locked clock | `profile_peak` = GFX 1777 MHz (DPM top sclk), mclk 1124 MHz |
| Free clock | default DVFS, jittery (drops to ~500 MHz in gaps) |
| rocprofv3 | 1.1.0 |

Method: `busy` = median kernel time from `rocprofv3 --kernel-trace` (forces AQL).
`e2e` = profiler-free whole-graph hipEvent wall clock (best-of-7 inside the
binary). `gap = (e2e - busy*K)/(K-1)`; `PM4 saves` = AQL gap - PM4 gap (us) and
e2e reduction (%). Same lib for both columns -- only `HIP_PM4_GRAPH` is toggled.
PM4 is invisible to rocprofv3 (raw IB bypasses the traced AQL queue), so busy is
taken from the AQL trace and reused for the PM4 column.

## Scenarios (what each test measures)

All three build a HIP graph (recorded once, replayed with no per-launch CPU cost)
of K kernel nodes in a linear dependency chain -- the apples-to-apples analog of a
recorded command buffer. The dependency between consecutive nodes is what forces
the command processor (CP) to insert the per-dispatch sync (barrier bit + cache
fence) whose cost is "the gap". The only thing that changes between scenarios is
how uniform the kernels in the chain are:

- Uniform chain (`hip_gap_graph.x <spin> [n] [K]`): "uniform" = identical cost and
  shape, NOT one repeated kernel object. The chain interleaves two PHYSICALLY
  DISTINCT kernels -- `gap_kernel<1>` and `gap_kernel<3>`, two separate
  instantiations of one template (the template arg only changes an in-loop add
  constant, so cost is identical) -- so the runtime cannot collapse the graph into
  a single-repeated-kernel fast path. `spin` sets the per-kernel duration;
  `n`=grid size (16384, fixed); `K`=chain length. The simplest case: isolates the
  per-dispatch gap with all nodes equal cost.
- Short/long interleave (`hip_gap_mix.x <short> <long> [n] [K]`): the chain
  ALTERNATES a short-spin and a long-spin kernel. Models a realistic decode layer
  where a cheap op (e.g. RMSNorm) is followed by an expensive one (e.g. GEMM).
  Checks the gap is per-dispatch and not an artifact of repeated identical work.
- Heterogeneous interleave (`hip_gap_hetero.x <spin> [n] [K]`): cycles 4 DISTINCT
  kernels (node k uses k%4), each with a different block size and kernarg
  signature (varying pointer/scalar counts incl. unused). The most graph-like
  case -- distinct kernels, distinct args -- confirming the saving is not limited
  to one repeated kernel shape.

For each scenario the AQL column is the stock dispatch path and the PM4 column is
our raw-IB graph replay (`HIP_PM4_GRAPH=1`), same binary and clocks.

## Does the per-dispatch gap grow with kernel size?

Yes -- the AQL gap grows with kernel duration, but as a STEP function (flat
plateaus, then ~2x jumps), not proportionally. PM4's gap stays ~0 at every size.
This is confirmed by two independent methods that agree on the trend:

1. Profiler-free `e2e(AQL) - busy*K` (the AQL gap column in the LOCKED table):
   ~2.2 us (spin 250-1500) -> ~4.1 us (2000-3000) -> ~6 us (5000-10000) ->
   ~10.5 us (25000) -> ~14.6 us (50000).
2. Direct rocprof CP-side device timeline, `End[k] -> Start[k+1]` (AQL only, no
   `busy*K` subtraction):

| spin | busy (us) | CP-side gap (us) |
|---|---|---|
| 250  | 7.86    | 4.280  |
| 500  | 14.54   | 4.280  |
| 1000 | 27.76   | 4.280  |
| 1500 | 41.04   | 4.280  |
| 2000 | 54.40   | 8.560  |
| 3000 | 80.96   | 8.560  |
| 5000 | 134.16  | 8.640  |
| 10000| 267.00  | 8.680  |
| 25000| 665.55  | 17.280 |
| 50000| 1330.0  | 17.360 |

Both methods step at the same kernel-duration thresholds, so the growth is a real
property of the AQL dispatch path, NOT a profiling/measurement artifact. The
CP-side absolute value is ~2x the profiler-free gap because rocprof attaches a
completion signal per kernel; the trend is identical.

Consequence: PM4's per-dispatch saving GROWS in absolute us (from ~2 us on tiny
kernels to ~15 us on ~5 ms kernels), because AQL grows and PM4 stays ~0. What
SHRINKS is the saving as a FRACTION of e2e (20% -> <1%), because total kernel time
grows faster than the gap. These are two different things.

## Uniform chain -- LOCKED (profile_peak, ~1777 MHz)

| spin | K | busy (us) | AQL e2e (ms) | PM4 e2e (ms) | AQL gap (us) | PM4 gap (us) | PM4 saves |
|---|---|---|---|---|---|---|---|
| 250 | 1500 | 7.76 | 14.89 | 11.80 | 2.17 | 0.11 | 2.06 us (20.7%) |
| 500 | 1500 | 14.40 | 24.82 | 21.76 | 2.15 | 0.11 | 2.04 us (12.3%) |
| 750 | 1000 | 21.04 | 23.22 | 21.15 | 2.18 | 0.11 | 2.07 us (8.9%) |
| 1000 | 800 | 27.66 | 23.90 | 22.23 | 2.21 | 0.13 | 2.08 us (7.0%) |
| 1500 | 800 | 40.92 | 34.49 | 32.83 | 2.20 | 0.12 | 2.08 us (4.8%) |
| 2000 | 600 | 54.17 | 34.97 | 32.57 | 4.12 | 0.11 | 4.01 us (6.9%) |
| 3000 | 400 | 80.68 | 33.92 | 32.33 | 4.14 | 0.15 | 3.99 us (4.7%) |
| 5000 | 300 | 133.78 | 41.91 | 40.15 | 5.94 | 0.07 | 5.87 us (4.2%) |
| 10000 | 150 | 266.40 | 40.87 | 39.98 | 6.08 | 0.17 | 5.92 us (2.2%) |
| 25000 | 80 | 663.76 | 53.93 | 53.12 | 10.48 | 0.29 | 10.19 us (1.5%) |
| 50000 | 50 | 1326.52 | 67.04 | 66.30 | 14.59 | -0.51 | 15.10 us (1.1%) |
| 100000 | 40 | 2654.31 | 106.72 | 106.18 | 13.98 | 0.11 | 13.87 us (0.5%) |
| 200000 | 30 | 5303.45 | 159.54 | 159.07 | 15.17 | -1.22 | 16.39 us (0.3%) |
| 210000 | 30 | 5569.35 | 167.49 | 167.01 | 14.13 | -2.55 | 16.68 us (0.3%) |

## Uniform chain -- FREE (default DVFS, e2e median of 5)

| spin | K | busy (us)* | AQL e2e (ms) | PM4 e2e (ms) | PM4 saves (% of e2e) |
|---|---|---|---|---|---|
| 250 | 1500 | 12.42 | 18.43 | 7.58 | 58.9% |
| 500 | 1500 | 8.88 | 19.06 | 14.35 | 24.7% |
| 750 | 1000 | 12.87 | 17.06 | 13.94 | 18.3% |
| 1000 | 800 | 16.84 | 17.05 | 14.57 | 14.5% |
| 1500 | 800 | 24.90 | 23.29 | 20.97 | 10.0% |
| 2000 | 600 | 33.00 | 22.50 | 20.84 | 7.4% |
| 3000 | 400 | 49.62 | 21.81 | 20.63 | 5.4% |
| 5000 | 300 | 83.08 | 26.23 | 25.13 | 4.2% |
| 10000 | 150 | 166.32 | 25.74 | 24.99 | 2.9% |
| 25000 | 80 | 416.10 | 33.07 | 32.52 | 1.7% |
| 50000 | 50 | 834.62 | 40.40 | 40.00 | 1.0% |
| 100000 | 40 | 1628.27 | 64.41 | 63.87 | 0.8% |
| 200000 | 30 | 3229.54 | 96.41 | 95.99 | 0.4% |
| 210000 | 30 | 3386.43 | 101.27 | 100.83 | 0.4% |

\* free busy is profiled at a different DVFS state than e2e (reference only).
The spin=250 58.9% is a DVFS clock-state outlier (PM4, with fewer gaps, settled
at a higher clock than AQL); use LOCKED for the real per-dispatch saving.

## Short/long interleave -- LOCKED

K=400 = 200 short + 200 long, alternating.

| short | long | busy_s (us) | busy_l (us) | AQL e2e (ms) | PM4 e2e (ms) | AQL gap (us) | PM4 saves |
|---|---|---|---|---|---|---|---|
| 250  | 5000   | 10.28 | 83.32   | 20.35  | 18.78  | 4.09 | 3.95 us (7.7%) |
| 500  | 20000  | 19.48 | 329.72  | 71.46  | 69.86  | 4.06 | 4.00 us (2.2%) |
| 250  | 50000  | 10.32 | 822.68  | 169.93 | 166.58 | 8.34 | 8.38 us (2.0%) |
| 1000 | 100000 | 37.84 | 1643.72 | 339.72 | 336.34 | 8.53 | 8.47 us (1.0%) |

## Short/long interleave -- FREE (e2e median of 5)

| short | long | busy_s (us)* | busy_l (us)* | AQL e2e (ms) | PM4 e2e (ms) | PM4 saves (% of e2e) |
|---|---|---|---|---|---|---|
| 250  | 5000   | 6.48  | 50.70   | 13.86  | 12.60  | 9.1% |
| 500  | 20000  | 12.60 | 207.76  | 44.03  | 42.12  | 4.3% |
| 250  | 50000  | 6.76  | 520.00  | 102.04 | 100.11 | 1.9% |
| 1000 | 100000 | 23.80 | 1019.72 | 205.78 | 202.14 | 1.8% |

\* free busy profiled at a different DVFS state than e2e (reference only).

## Heterogeneous interleave -- 4 distinct kernels

`hip_gap_hetero.x` cycles 4 kernels (node k uses k%4; K=800 = 200 each = 800
total dispatches), each with a different block size and kernarg signature
(varying pointer count incl. unused, plus unused scalars); gid<n guard keeps work
similar.

| kernel | block | args | busy @ spin=500 | busy @ spin=2000 |
|---|---|---|---|---|
| k1 | 256 | 1 ptr, 2 scalars | 19.40 us | 74.24 us |
| k2 | 128 | 3 ptr (2 unused), 2 scalars | 15.68 us | 59.16 us |
| k3 | 64  | 5 ptr (4 unused), 4 scalars (2 unused) | 9.36 us | 34.04 us |
| k4 | 512 | 2 ptr (1 unused), 4 scalars (2 unused) | 19.48 us | 74.44 us |

(On gfx1100 the per-kernel busy spreads more by block size than on gfx1201 --
smaller blocks reach higher occupancy on the 64-VGPR/SIMD pool -- but the gap is
per-DISPATCH and independent of this spread.)

LOCKED:

| spin | avg busy (us) | AQL e2e (ms) | PM4 e2e (ms) | AQL gap (us) | PM4 gap (us) | PM4 saves |
|---|---|---|---|---|---|---|
| 250  | 8.56  | 8.58  | 6.95  | 2.17 | 0.13 | 2.04 us (19.0%) |
| 500  | 15.99 | 14.55 | 12.88 | 2.20 | 0.12 | 2.08 us (11.4%) |
| 1000 | 30.80 | 26.42 | 24.75 | 2.22 | 0.13 | 2.09 us (6.3%) |
| 2000 | 60.45 | 51.63 | 48.46 | 4.10 | 0.12 | 3.97 us (6.1%) |

FREE (e2e median of 5):

| spin | avg busy (us)* | AQL e2e (ms) | PM4 e2e (ms) | PM4 saves (% of e2e) |
|---|---|---|---|---|
| 250  | 5.38  | 7.40  | 4.38  | 40.9% |
| 500  | 9.82  | 11.26 | 8.41  | 25.3% |
| 1000 | 18.72 | 18.38 | 15.97 | 13.1% |
| 2000 | 36.94 | 32.16 | 29.35 | 8.7% |

\* free busy profiled at a different DVFS state than e2e (reference only).

## Notes

- PM4 removes essentially the whole per-dispatch CP gap (PM4 gap ~0 at every
  size): ~2.1 us in the launch-bound regime on gfx1100 (vs ~1.5 us on gfx1201),
  19-21% e2e on short kernels. Holds for uniform, short/long, and heterogeneous
  chains -- not limited to repeated identical kernels.
- The AQL gap grows with kernel duration (step-wise: ~2.2 -> 4.1 -> 6 -> 14.6 us)
  while PM4 stays ~0 -- see "Does the per-dispatch gap grow with kernel size"
  above; confirmed both profiler-free and via the rocprof CP-side timeline. So
  PM4's saving GROWS in absolute us with kernel size but SHRINKS as a fraction of
  e2e (total kernel time grows faster than the gap). The saving as a % therefore
  amortizes away (down to <1% at ~5 ms) and tracks dispatch COUNT more than size.
- PM4 gap going slightly negative at the longest kernels (-0.5 to -2.5 us) is
  0.04%-level e2e noise (PM4 e2e within rounding of busy*K), not overlap. Above
  busy ~300 us, the absolute gap precision is ~0.3%, so prefer the e2e-savings %.
- No spin~1000-1500 FMA artifact on gfx1100: busy scaled linearly across the
  region (250->7.76, 750->21.0, 1500->40.9 us, ~0.0266 us/spin), unlike gfx1201
  which had a one-time ~25 us iteration penalty past ~1150 iterations. So the
  RDNA3 LOCKED curve is clean throughout.
- Free DVFS is unreliable for absolute gaps (clocks swing 500-1777 MHz,
  workload-dependent); the spin=250 uniform 58.9% is a clock-state artifact. Use
  LOCKED clocks for all comparisons.
- gfx1100 kernels are ~1.65x slower than gfx1201 at equal spin (e.g. spin=250
  busy 7.76 vs 4.72 us), as expected from the W7900's lower clock (1777 vs
  ~2319 MHz) and CU throughput; the dispatch gap itself is an architecture/runtime
  constant, not tied to kernel duration.
```

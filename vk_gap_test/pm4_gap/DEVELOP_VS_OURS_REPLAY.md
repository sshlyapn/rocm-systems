# Graph-replay microbenchmark: upstream/develop vs our PM4 work

Host-side `hipGraphLaunch` cost for an N-long pure-dispatch (kernel-only) graph,
swept over chain length N. Tool: `hip_replay_timing.cpp` (drains the queue outside
the timer, so the number is the runtime's host work to submit one replay).
M=256 elems, R=4000 timed repeats, gfx1100 (Radeon PRO W7900), single GPU.

All builds and runs done inside the `test-framework-sshliapn-container-2nd`
container. The same client binary is run against four different userspace stacks
via `LD_LIBRARY_PATH`.

## Stacks compared

- A. **develop** -- upstream/develop CLR + a matching develop ROCr, both built from
  the `rocm-systems-develop` worktree (tip 698402fad94, 2026-06-12). Carries
  develop's graph work (HW-signal pooling across launches #6670, doorbell de-dup
  #6762, SDMA handoff + lastSlotPtr ordering restores). AQL graph replay.
- B. **ours + PM4** -- our branch's CLR (`build-gap`) on ROCm 7.2 ROCr, with
  `HIP_PM4_GRAPH=1` (template-at-capture + GraphExec-owned shared IB, both default on).
- C. **ours AQL** -- same lib as B, `HIP_PM4_GRAPH` unset (our branch's AQL baseline).
- D. **stock 7.2** -- installed `/opt/rocm` (ROCm 7.2) AQL graph replay.

## Results

### Steady-state host `hipGraphLaunch` (us) -- the inter-kernel dispatch cost

| N   | A develop | B ours+PM4 | C ours AQL | D stock 7.2 |
|-----|-----------|------------|------------|-------------|
| 1   | **1.11**  | 3.50       | 3.04       | 3.45        |
| 4   | **1.38**  | 2.98       | 3.02       | 2.84        |
| 16  | **1.20**  | 3.74       | 3.37       | 3.44        |
| 64  | **1.42**  | 2.75       | 3.83       | 4.11        |
| 128 | **1.39**  | 2.98       | 4.27       | 4.28        |
| 256 | **1.63**  | 2.73       | 5.08       | 5.22        |

Marginal per-added-dispatch (us) at N=256: develop 0.0064, ours+PM4 0.0107,
ours-AQL 0.0198, stock 0.0204.

### First (cold) replay and instantiate (us)

| N   | metric            | A develop | B ours+PM4 | C ours AQL | D stock 7.2 |
|-----|-------------------|-----------|------------|------------|-------------|
| 256 | instantiate       | 295       | **7499**   | 247        | 278         |
| 256 | FIRST cold launch | 15.6      | 37.4       | 19.9       | 21.9        |
| 256 | 2nd (warm) launch | 2.7       | 5.7        | 6.7        | 6.8         |

## Reading the numbers

1. **develop's graph path is the fastest in steady state**: ~1.1-1.6 us flat,
   essentially O(1) in N (per-dispatch ~0.006 us). It beats both our PM4 path and
   our/stock AQL paths on this host-launch microbenchmark.

2. **Our PM4 does flatten the host launch** vs our own AQL baseline (B vs C):
   2.7-3.7 us flat vs 3.0-5.1 us rising. So the O(1) PM4-IB submit works as designed
   on our base -- it removes the O(N) AQL-packet growth (B is flat; C/D climb with N).

3. **But develop's absolute constant is lower (~1.2 us vs our ~3 us)**, and develop
   reaches O(1) without our heavy instantiate cost. Our PM4 instantiate is ~4-7.5 ms
   (template encode + device-scoped shared-IB build hoisted to instantiate, by design
   in D.21/D.22), vs develop's ~0.1-0.3 ms.

## Important caveats (confounds)

- Our PM4 lib (B/C) is built on a MUCH older CLR base than develop and runs on the
  ROCm 7.2 ROCr; develop (A) is a newer CLR + newer ROCr. Our own AQL baseline (C)
  matches stock 7.2 (D) at ~3-5 us, while develop's AQL sits at ~1.2 us -- so a large
  part of develop's advantage is general runtime/ROCr maturation, NOT only the graph
  algorithm. A fair "algorithm vs algorithm" test would port our PM4 work onto the
  develop base and compare on the same ROCr.
- This tool measures HOST enqueue latency only. True GPU-side inter-kernel gaps need
  a profiler (rocprofv3 timestamps); not captured here.

## How develop was made to run on this 7.2 box (compat patches, worktree-local)

Running a develop userspace on the 7.2 box required (all in the
`rocm-systems-develop` worktree / build dirs, none touching our branch):

1. Built develop ROCr from the monorepo (provides the symbols 7.2 lacks:
   `hsa_amd_memory_async_batch_copy`, `hsa_amd_signal_get_event_id`).
2. Stub `ClangConfig.cmake`/`LLVMConfig.cmake` (image ships clang/llvm binaries but
   not the LLVM CMake dev package) + a portable `xxd -i` shim.
3. `trap_handler_gfx12.s`: `HW_REG_WAVE_SCHED_MODE` -> numeric id `26` (the older
   assembler does not know the name; same encoding, blob never runs on gfx1100).
4. `blitcl.cpp`: local fallback definitions for `__amd_streamOpsIncrement/Decrement`
   (newer rocm-device-libs builtins absent from 7.2 `hip.bc`; without them the blit
   program fails to JIT-link and NO stream can be created). The benchmark never
   dispatches stream-value ops, so a plain impl suffices.
5. Forced develop's `hsa/` headers ahead of 7.2's for the CLR build via a distinct
   `-I` shim dir (a `-I` equal to an existing `-isystem` dir is deduped by GCC).

Raw output: `replay_compare.txt`.

---

# End-to-end model decode: gpt-oss-20b (the metric that actually matters)

The host-launch microbench above measures one isolated cost. To see what it means
for a real workload we ran the full gpt-oss-20b decode through flywheel (direct mode)
on the same four stacks. Same client, libs swapped via `LD_LIBRARY_PATH` /
`HIP_PM4_GRAPH`; each stack's `libamdhip64` was confirmed distinct with
`hipRuntimeGetVersion` (ours 70253211, develop 71460850, stock 70226015).

- Model: `/huggingface/hub/models--openai--gpt-oss-20b-q0-lmhead-s0t/` (24 decoder
  layers, profile `q0`), 512 input / 128 output tokens, concurrency 1, `-n 5`.
- `REDLINE_DEBUG_SKIP_SAMPLING=1` (feeds a host-selected random token, skips the
  sampling chain -- isolates the model+runtime perf path; generated text is junk).
- Single gfx1100 / Radeon PRO W7900, `HIP_VISIBLE_DEVICES=0`. Numbers below are the
  median of 3 repeats; run-to-run spread was < 0.5%.

| Stack         | Throughput req/s | e2e latency ms | TTFT ms | **TPOT ms** | Tok/s | Tok/s/User |
|---------------|------------------|----------------|---------|-------------|-------|------------|
| B ours + PM4  | **1.52**         | **656**        | ~100    | **4.37**    | **976** | **195**  |
| C ours AQL    | 1.39             | 719            | ~101    | 4.87        | 890   | 178        |
| D stock 7.2   | 1.39             | 720            | ~101    | 4.87        | 889   | 178        |
| A develop     | 1.39             | 721            | ~102    | 4.87        | 887   | 178        |

## Reading the model numbers

1. **Our PM4 path is the only stack that speeds up decode**: TPOT 4.87 -> 4.37 ms
   (**-10.3%**), Tok/s 890 -> 976 (**+9.7%**), e2e latency 719 -> 656 ms (-8.8%).
   This is the real payoff of submitting the whole decode graph as one PM4 IB: it
   removes per-dispatch overhead on the *device* path that accumulates over the
   24-layer x several-kernels-per-layer decode graph, replayed once per token.

2. **develop, ours-AQL and stock 7.2 are indistinguishable here (~4.87 ms TPOT)**.
   develop won the host-launch microbench by ~1.2 us/launch, but a decode token costs
   ~4870 us of GPU compute (GEMM / attention / MoE), so a few-microsecond host-launch
   delta is in the noise at the model level. develop's graph work (HW-signal pooling,
   doorbell de-dup) is host-side and does not shrink the GPU per-token cost.

3. **TTFT is flat across all stacks (~100 ms)** -- prefill is one big compute-bound
   pass, not graph-replay-bound, so none of the graph paths move it.

Takeaway: the host-launch microbench rewards develop, but on the workload that ships
(token-by-token decode) our PM4 single-IB replay is the only change that converts to
a measurable user-visible speedup, because it attacks GPU-side per-dispatch cost, not
just host enqueue latency.

Raw output: `gptoss_e2e.txt`. GPU-side per-kernel gap profiling (rocprofv3) was
deferred; note that PM4-IB dispatches are not visible to `rocprofv3 --kernel-trace`
(it hooks the AQL dispatch path), so the AQL profiler cannot directly time the PM4
replay -- a separate measurement method would be needed.

---

# Full reproduction: build and run upstream/develop on the 7.2 box

Every step below was run inside the container. Nothing here touches our branch's
checkout -- develop lives in a separate git worktree and separate build dirs.

Shared variables used throughout (set these first, on the host shell that issues
the `docker exec` calls AND inside the container blocks):

```bash
CT=test-framework-sshliapn-container-2nd                 # build/run container
SRC=/home/sshliapn/code/rocm-systems                     # our monorepo checkout (has the worktree's .git)
DEV=/home/sshliapn/code/rocm-systems-develop             # develop worktree (created in step 1)
```

The container mounts `/home/sshliapn`, so all paths are visible inside it. These
three variables are plain aliases for the absolute paths below; every command
expands to absolute paths. If you prefer, substitute them literally:

```
CT  = test-framework-sshliapn-container-2nd
SRC = /home/sshliapn/code/rocm-systems
DEV = /home/sshliapn/code/rocm-systems-develop
```

## Artifacts produced (absolute paths)

Built / saved on this box (build outputs are NOT committed -- regenerate with the
steps below -- but are preserved at these absolute locations):

| Artifact | Absolute path |
|----------|---------------|
| develop worktree (clr+hip+rocr at the develop tip) | `/home/sshliapn/code/rocm-systems-develop` |
| develop ROCr build dir | `/home/sshliapn/code/rocm-systems-develop/build-rocr` |
| develop ROCr install prefix | `/home/sshliapn/code/rocm-systems-develop/install` |
| develop `libhsa-runtime64.so` | `/home/sshliapn/code/rocm-systems-develop/install/lib/libhsa-runtime64.so.1.21.0` |
| develop CLR build dir | `/home/sshliapn/code/rocm-systems-develop/projects/clr/build-develop` |
| develop `libamdhip64.so` | `/home/sshliapn/code/rocm-systems-develop/projects/clr/build-develop/hipamd/lib/libamdhip64.so.7.14.60850-0000000` |
| our PM4 `libamdhip64.so` | `/home/sshliapn/code/rocm-systems/projects/clr/build-gap/hipamd/lib/libamdhip64.so` |
| benchmark binary | `/home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/hip_replay_timing` |

Committed in this repo (under `vk_gap_test/pm4_gap/`), so the build is reproducible
without re-deriving the helper files:

| File (absolute path) | Purpose |
|----------------------|---------|
| `/home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/DEVELOP_VS_OURS_REPLAY.md` | this report |
| `/home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/replay_compare.txt` | raw host-launch sweep output |
| `/home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/gptoss_e2e.txt` | raw gpt-oss-20b e2e decode output (all reps) |
| `/home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/hip_gpu_gap.cpp` | GPU-side gap companion bench (rocprofv3, deferred) |
| `/home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/parse_gpu_gap.py` | parser for the GPU-side gap CSV |
| `/home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/develop_repro/cmake-stubs/ClangConfig.cmake` | Clang find_package stub |
| `/home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/develop_repro/cmake-stubs/LLVMConfig.cmake` | LLVM find_package stub |
| `/home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/develop_repro/cmake-stubs/bin/xxd` | portable `xxd -i` shim |
| `/home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/develop_repro/develop_compat.patch` | the `blitcl.cpp` + `trap_handler_gfx12.s` source patches |

To reuse the committed helpers instead of recreating them in Step 2/3/5:

```bash
# helper cmake stubs + xxd shim:
cp -r /home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/develop_repro/cmake-stubs \
      /home/sshliapn/code/rocm-systems-develop/cmake-stubs
chmod +x /home/sshliapn/code/rocm-systems-develop/cmake-stubs/bin/xxd
# source patches (run from the develop worktree root):
git -C /home/sshliapn/code/rocm-systems-develop apply \
  /home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap/develop_repro/develop_compat.patch
```

## Step 0 -- pick the develop ref and confirm GPU is free

```bash
# upstream = github.com/ROCm/rocm-systems ; develop tip carries the graph work
git -C "$SRC" log upstream/develop -1 --format='%H %ci %s'
git -C "$SRC" log upstream/develop --oneline -12 -- \
  projects/clr/hipamd/src/hip_graph_internal.cpp \
  projects/clr/rocclr/device/rocm/rocvirtual.cpp

# GPU must be idle before any workload (single-GPU gfx1100 box -> device 0)
docker exec $CT bash -lc 'amd-smi process | head -20'
```

## Step 1 -- create a detached develop worktree (non-destructive)

A worktree gives a full, consistent develop checkout (clr + hip + rocr-runtime all
at the same commit) without disturbing our working tree.

```bash
git -C "$SRC" worktree add --detach "$DEV" upstream/develop
git -C "$DEV" log --oneline -1     # expect: 698402fad94 ...
```

## Step 2 -- helper files for the missing LLVM/Clang CMake dev package

This image ships the `clang`/`llvm-*` BINARIES under `/opt/rocm/llvm/bin` but NOT
the upstream LLVM CMake config package, so ROCr's (and CLR's) `find_package(Clang)`
/ `find_package(LLVM)` fail. Provide minimal stubs that only import the executable
targets the device-code CMakeLists actually invoke.

Create `$DEV/cmake-stubs/ClangConfig.cmake`:

```cmake
set(_rocm_llvm_bin "/opt/rocm/llvm/bin")
foreach(_t clang clang-offload-bundler)
  if(NOT TARGET ${_t})
    add_executable(${_t} IMPORTED GLOBAL)
    set_target_properties(${_t} PROPERTIES IMPORTED_LOCATION "${_rocm_llvm_bin}/${_t}")
  endif()
endforeach()
set(Clang_FOUND TRUE)
set(CLANG_FOUND TRUE)
```

Create `$DEV/cmake-stubs/LLVMConfig.cmake`:

```cmake
set(_rocm_llvm_bin "/opt/rocm/llvm/bin")
foreach(_t llvm-objcopy llvm-mc llvm-dis opt llvm-link)
  if(NOT TARGET ${_t})
    add_executable(${_t} IMPORTED GLOBAL)
    set_target_properties(${_t} PROPERTIES IMPORTED_LOCATION "${_rocm_llvm_bin}/${_t}")
  endif()
endforeach()
set(LLVM_FOUND TRUE)
set(LLVM_PACKAGE_VERSION "22.0.0")
set(LLVM_VERSION_MAJOR 22)
set(LLVM_VERSION_MINOR 0)
set(LLVM_VERSION_PATCH 0)
set(LLVM_INCLUDE_DIRS "/opt/rocm/llvm/include")
set(LLVM_LIBRARY_DIRS "/opt/rocm/llvm/lib")
set(LLVM_TOOLS_BINARY_DIR "${_rocm_llvm_bin}")
```

The ROCr device-header generators (`create_blit_shader_header.sh`,
`create_hsaco_ascii_file.sh`) call `xxd -i`, which is not installed and apt has stale
indices. Drop in a portable Perl shim and make it executable.

Create `$DEV/cmake-stubs/bin/xxd`:

```perl
#!/usr/bin/perl
use strict; use warnings;
my $mode_i = 0; my $file;
foreach my $a (@ARGV) { if ($a eq '-i') { $mode_i = 1; } else { $file = $a; } }
my $fh; my $name = "stdin";
if (defined $file) { open($fh, '<:raw', $file) or die "xxd: $file: $!\n";
  $name = $file; $name =~ s/[^A-Za-z0-9]/_/g; $name =~ s/^(\d)/_$1/; }
else { $fh = \*STDIN; }
local $/; my $data = <$fh>; $data = '' unless defined $data;
my @bytes = unpack('C*', $data);
die "xxd shim: only 'xxd -i FILE' supported\n" unless $mode_i;
print "unsigned char $name\[] = {\n";
my $n = scalar(@bytes);
for (my $i = 0; $i < $n; $i++) {
  print "  " if ($i % 12) == 0;
  printf "0x%02x", $bytes[$i];
  print "," if $i != $n - 1;
  if (($i % 12) == 11 || $i == $n - 1) { print "\n"; } else { print " "; }
}
print "};\n";
print "unsigned int ${name}_len = $n;\n";
```

```bash
chmod +x "$DEV/cmake-stubs/bin/xxd"
# sanity:
docker exec $CT bash -lc "export PATH=$DEV/cmake-stubs/bin:\$PATH; printf ABC>/tmp/t && cd /tmp && xxd -i t"
```

## Step 3 -- patch the gfx12 trap handler for the older assembler

The 7.2 clang assembler does not know the register name `HW_REG_WAVE_SCHED_MODE`
(added in newer LLVM, id 26). Replace the name with the numeric id -- identical
encoding, and this gfx12 blob never executes on a gfx1100 GPU anyway.

File: `$DEV/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/trap_handler/trap_handler_gfx12.s`
(around line 1298):

```asm
; before:
  s_setreg_b32      hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), ttmp2
; after:
  s_setreg_b32      hwreg(26, 0, 2), ttmp2
```

## Step 4 -- configure, build, install develop ROCr

```bash
docker exec $CT bash -lc "
  export PATH=$DEV/cmake-stubs/bin:\$PATH
  cmake -S $DEV/projects/rocr-runtime -B $DEV/build-rocr \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$DEV/install \
    -DCMAKE_PREFIX_PATH='/opt/rocm;/opt/rocm/llvm' \
    -DClang_DIR=$DEV/cmake-stubs \
    -DLLVM_DIR=$DEV/cmake-stubs \
    -DBUILD_SHARED_LIBS=ON
  cmake --build $DEV/build-rocr --parallel \$(nproc)
  cmake --install $DEV/build-rocr
"

# verify the symbols 7.2 lacks are now present:
docker exec $CT bash -lc \
  "nm -D $DEV/install/lib/libhsa-runtime64.so | grep -E 'memory_async_batch_copy|signal_get_event_id'"
```

## Step 5 -- patch develop's blit program for the missing device builtins

develop's `blitcl.cpp` (the blit kernels CLR JIT-compiles to create the BlitManager
that EVERY stream needs) references `__amd_streamOpsIncrement/Decrement`, builtins that
exist only in a newer `rocm-device-libs` than the 7.2 `hip.bc` on this box. Without
them the blit program fails to link and no stream can be created. Add local fallback
definitions (the benchmark never dispatches stream-value ops).

File: `$DEV/projects/clr/rocclr/device/blitcl.cpp` -- replace the two `extern` decls:

```c
// before:
    extern void __amd_streamOpsIncrement(__global uint*, __global ulong*, ulong);
    extern void __amd_streamOpsDecrement(__global uint*, __global ulong*, ulong);

// after:
    void __amd_streamOpsIncrement(__global uint* pInt, __global ulong* pUlong, ulong value) {
      if (pInt) { pInt[0] = pInt[0] + 1u; } else if (pUlong) { pUlong[0] = pUlong[0] + 1UL; }
    }
    void __amd_streamOpsDecrement(__global uint* pInt, __global ulong* pUlong, ulong value) {
      if (pInt) { pInt[0] = pInt[0] - 1u; } else if (pUlong) { pUlong[0] = pUlong[0] - 1UL; }
    }
```

(Keep the `__amd_streamOpsWrite`/`Wait` externs -- those resolve from 7.2 `hip.bc`.)

## Step 6 -- header-shadowing fix and Python dep, then configure+build develop CLR

develop CLR must include develop's `hsa/*` headers (they declare the newer types,
e.g. `hsa_amd_external_semaphore_t`), but `/opt/rocm/include` is searched first. A
plain `-I $DEV/install/include` does NOT help: GCC dedups a `-I` dir that is also an
`-isystem` dir (which the hsa-runtime64 CMake target adds) and demotes it. Work around
it with a DISTINCT `-I` dir that only symlinks the `hsa/` tree:

```bash
docker exec $CT bash -lc "mkdir -p $DEV/hsa-shim && ln -sfn $DEV/install/include/hsa $DEV/hsa-shim/hsa"

# the profiling-API header generator needs this Python package (one-time):
docker exec $CT bash -lc 'pip3 install CppHeaderParser'
```

Configure and build `amdhip64` (PCH off -> avoids another `find_package(LLVM)`;
`-I $DEV/hsa-shim` forces develop's hsa headers ahead of 7.2's):

```bash
docker exec $CT bash -lc "
  cmake -S $DEV/projects/clr -B $DEV/projects/clr/build-develop \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/tmp/clr-develop-install \
    -DCMAKE_PREFIX_PATH='$DEV/install;/opt/rocm;/opt/rocm/llvm' \
    -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF \
    -DHIP_COMMON_DIR=$DEV/projects/hip \
    -DHIP_COMPILER=clang -DHIP_RUNTIME=rocclr -DHIP_PLATFORM=amd \
    -D__HIP_ENABLE_PCH=OFF \
    -Dhsa-runtime64_DIR=$DEV/install/lib/cmake/hsa-runtime64 \
    -DCMAKE_CXX_FLAGS=-I$DEV/hsa-shim \
    -DCMAKE_C_FLAGS=-I$DEV/hsa-shim
  cmake --build $DEV/projects/clr/build-develop --target amdhip64 --parallel \$(nproc)
"
# result: $DEV/projects/clr/build-develop/hipamd/lib/libamdhip64.so*
```

## Step 7 -- sanity-check the develop stack

```bash
DEVLIB=$DEV/projects/clr/build-develop/hipamd/lib:$DEV/install/lib:/opt/rocm/lib

# no unresolved symbols, and it must bind DEVELOP's libhsa (not 7.2):
docker exec $CT bash -lc "
  LD_LIBRARY_PATH=$DEVLIB ldd -r $DEV/projects/clr/build-develop/hipamd/lib/libamdhip64.so.7.14.* \
    | grep -iE 'not found|undefined' || echo 'clean'
  LD_LIBRARY_PATH=$DEVLIB ldd $DEV/projects/clr/build-develop/hipamd/lib/libamdhip64.so.7.14.* | grep -i hsa
"
```

## Step 8 -- build the benchmark and run the comparison

The client binary is compiled once with the stock `/opt/rocm` hipcc; it is run
against each stack purely by swapping `LD_LIBRARY_PATH`.

```bash
OURLIB=$SRC/projects/clr/build-gap/hipamd/lib                 # our PM4 lib
DEVLIB=$DEV/projects/clr/build-develop/hipamd/lib:$DEV/install/lib:/opt/rocm/lib

docker exec $CT bash -lc "
  cd /home/sshliapn/code/llama.cpp/vk_gap_test/pm4_gap
  /opt/rocm/bin/hipcc -O2 -std=c++17 --offload-arch=gfx1100 hip_replay_timing.cpp -o hip_replay_timing
  export HIP_VISIBLE_DEVICES=0
  NS='1 4 16 64 128 256'; M=256; R=4000

  echo '##### A) develop #####'
  for n in \$NS; do LD_LIBRARY_PATH=$DEVLIB ./hip_replay_timing \$n \$M \$R; done

  echo '##### B) ours + PM4 #####'
  for n in \$NS; do env LD_LIBRARY_PATH=$OURLIB HIP_PM4_GRAPH=1 ./hip_replay_timing \$n \$M \$R; done

  echo '##### C) ours AQL #####'
  for n in \$NS; do env LD_LIBRARY_PATH=$OURLIB ./hip_replay_timing \$n \$M \$R; done

  echo '##### D) stock 7.2 #####'
  for n in \$NS; do env -u LD_LIBRARY_PATH ./hip_replay_timing \$n \$M \$R; done
"
```

`hip_replay_timing <N> <M> <R>`: N = chain length (dispatch packets), M = elems per
kernel (keep small so the GPU never lags), R = timed repeats. It prints instantiate,
FIRST(cold), FIRST_e2e, 2nd(warm), and steady host launch avg/min/max + per-dispatch.

## Step 9 -- end-to-end gpt-oss-20b decode on each stack

Run the flywheel benchmark on the same box; only `LD_LIBRARY_PATH` / `HIP_PM4_GRAPH`
change between stacks. flywheel/lib-rocket is built against ROCm 7.2, so our build-gap
lib (same 7.2 base) is ABI-safe; develop binds its own newer `libamdhip64`+`libhsa`.

```bash
CT=test-framework-sshliapn-container-2nd
MODEL=/huggingface/hub/models--openai--gpt-oss-20b-q0-lmhead-s0t/
OURLIB=/home/sshliapn/code/rocm-systems/projects/clr/build-gap/hipamd/lib
DEVLIB=/home/sshliapn/code/rocm-systems-develop/projects/clr/build-develop/hipamd/lib:/home/sshliapn/code/rocm-systems-develop/install/lib:/opt/rocm/lib

docker exec $CT bash -lc "
  cd /home/sshliapn/code/rednotdead
  export HIP_VISIBLE_DEVICES=0 HF_HOME=/huggingface REDLINE_DEBUG_SKIP_SAMPLING=1
  BENCH='python3 utils/bench/benchmark.py -m $MODEL -i 512 -o 128 --profile q0 -ap w0 -tp 1 -c 1 -n 5 -s -w --no-cache'

  echo '## B ours + PM4';  env LD_LIBRARY_PATH=$OURLIB HIP_PM4_GRAPH=1 \$BENCH | grep -E '^ +1 +[0-9]'
  echo '## C ours AQL';    env LD_LIBRARY_PATH=$OURLIB                  \$BENCH | grep -E '^ +1 +[0-9]'
  echo '## D stock 7.2';   env -u LD_LIBRARY_PATH                       \$BENCH | grep -E '^ +1 +[0-9]'
  echo '## A develop';     env LD_LIBRARY_PATH=$DEVLIB                  \$BENCH | grep -E '^ +1 +[0-9]'
"
```

`REDLINE_DEBUG_SKIP_SAMPLING=1` is required so the reported TPOT/Tok/s reflect the
model+runtime path and not host-side sampling. Confirm each stack bound a different
runtime with a one-liner that dlopens `libamdhip64` and calls `hipRuntimeGetVersion`
(see `gptoss_e2e.txt` header for the expected three version ids).

## Cleanup (optional)

```bash
rm -rf "$DEV/build-rocr" "$DEV/projects/clr/build-develop"   # build dirs
git -C "$SRC" worktree remove --force "$DEV"                 # remove the develop worktree
```

## Notes / gotchas

- Build container is `test-framework-sshliapn-container-2nd`
  (`docker ps --format '{{.Names}}' | grep sshliapn`). All builds/runs go through it.
- The develop ROCr statically links its own libhsakmt (ROCT); it talks to the
  installed 7.2 amdgpu/KFD kernel driver, which works for these workloads. If ROCr
  ever fails to init, the next thing to try is building ROCT separately and/or
  pointing at the system libhsakmt.
- `hsa_amd_memory_async_batch_copy` / the batch-copy graph path is NOT exercised by
  this kernel-only benchmark; graphs with memcpy nodes would, and would need the full
  develop device-libs for any blit JIT beyond the streamOps stub above.

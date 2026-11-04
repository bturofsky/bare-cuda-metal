# cuda-metal — a CUDA compatibility layer for Apple-Silicon Macs

**Goal:** let CUDA applications run on the Mac's GPU (Metal) with no source changes.

**Status:** v0 — API interposition skeleton. This is the separate project Ben directed
(2026-11-04). Nothing here pretends the full ecosystem is close; the point is a
correct skeleton that grows in measurable layers.

---

## Why this is hard (the honest statement)

CUDA is not a language; it is an ecosystem: the Driver + Runtime APIs, cuBLAS,
cuDNN, TensorRT, NCCL, cuFFT, cuSPARSE, NVML, PTX, and arch-specific machine
code. A translation layer must reproduce *behavior*, not implementation —
Tensor Cores, warp ops, async copies have no 1:1 Metal equivalent. Existing
attempts (ZLUDA → ROCm/Level Zero, SCALE → AMD compiler, HIPIFY/SYCLomatic →
source migration) all target AMD/Intel; nobody ships CUDA→Metal.

## The layering plan (each phase independently useful)

| Phase | Delivers | Honest effort |
|---|---|---|
| **v0 (this repo now)** | DYLD interposition shim: enumerate + log every CUDA API a real app calls. The coverage map that tells us which apps are tractable. | days |
| **v1** | Device/context/stream/memory plumbing on Metal (MTLDevice, MTLBuffer, MTLCommandQueue). Apps reach launch and fail at the kernel. | weeks |
| **v2** | Kernel path: PTX → MSL translator for the subset PTX that compiled CUDA apps actually carry (fp32/fp64 arithmetic, ld/st, control flow first; no PTX asm). | months |
| **v3** | cuBLAS → MPS (MetalPerformanceShaders covers GEMM/FFT/vision well), cuDNN → MPS CNN/graph, the rest of the library surface. | months |
| **v4** | NCCL-class multi-GPU: N/A single-Mac; multi-Mac over network is a research question. | open |

## The two markets (and which we are in)

- **Developer compatibility** (SCALE-style): compile CUDA *source* for Metal.
- **Application compatibility** (ZLUDA-style): run CUDA *binaries* unmodified.

We are in the **application-compatibility** market because the target is this
Mac: binaries arrive precompiled for x86_64+CUDA, Rosetta handles the CPU side,
the shim answers the GPU side. The commercial killers (PyTorch, inference
engines) mostly have Metal paths already — the tractable first apps are the
long tail: scientific computing, legacy tooling, vertical apps with embedded
CUDA libraries.

## v0 — what the skeleton does

`shim/libcuda_shim.c` exports the CUDA Driver + Runtime API surface as a
`libcuda.dylib` substitute. Under `DYLD_INSERT_LIBRARIES` (or `dlopen`) it:
1. logs every call with arguments (the coverage record),
2. returns plausible success/failure codes so apps proceed to the next call,
3. reports one Apple GPU device (device 0) with real probe data.

Build: `make` → `shim/libcuda_shim.dylib`. Test: `make test` (a dlopen harness
that exercises the API without needing the CUDA toolkit installed).

## Rules of the project

1. Measure before translating: v0's logs decide the v2 subset, not guesses.
2. Numerical correctness beats speed until v3 (then MPS gives speed mostly free).
3. No CUDA toolkit required on the build machine — the shim IS the toolkit.
4. Every phase ends with a real application running (or a named, logged gap).

---

## Reference corpus (bb23's CUDA 12.4 toolkit — measured 2026-11-04)

- `docs/reference/libcuda_driver_symbols.txt` — **712 exported functions** from the
  real `libcuda.so.1`. The full surface the shim ultimately answers; v0 exports 39.
- `docs/reference/vectorAdd.cu` → `.ptx` → `.sm86.cubin.txt` → **`vectorAdd.msl`** —
  the compiled kernel's actual PTX (nvcc 12.4, sm_86) beside the hand-written MSL
  translation. This pair is **v2's acceptance test**: the translator must emit
  Metal that computes the same result.
- The PTX itself: `.visible .entry` + `.param` + `.reg` + ~15 opcodes
  (ld.param/mov %ctaid/%ntid/%tid/mad.lo/setp/@bra/ld.global/st.global/ret) —
  the direct MSL mappings are documented inline in `vectorAdd.msl`.

bb23 hosts CUDA 12.4 (nvcc V12.4.131, Quadro RTX 8000) — the reference toolkit
for every format question: fatbin containers, PTX versions, cuobjdump output.

## A/B verification gate (the project's correctness instrument)

`scripts/run_ab.sh` sends the **same deterministic workload** (65,536 fp32 adds,
seed 0xC0FFEE) to both GPUs and bitwise-compares the results:

```
CUDA  checksum: 03b27e9ef9f4eb97  (0.071 ms on RTX 8000, sm_75)
Metal checksum: 03b27e9ef9f4eb97  (3.479 ms on Apple M4)
BITWISE MATCH: YES — IDENTICAL RESULTS ACROSS VENDORS
```

Measured 2026-11-04, first run after fixing two launch bugs (RTX 8000 is
sm_75 Turing, not sm_86; Metal command buffers need an explicit `[cb commit]`).
Every future phase must keep this gate green: same spec, same checksum.

## Fair measurement (2026-11-04, after the first "huge difference")

The initial 0.071 ms (CUDA) vs 3.479 ms (Metal) was apples-to-oranges: the CUDA
arm timed kernel-only via events (nvcc compiles offline), the Metal arm timed
wall-clock INCLUDING first-run runtime shader compilation. With both timed on
the GPU's own clock:

```
CUDA  (RTX 8000): 0.056 ms   Metal (M4): 0.012 ms   checksums bitwise-identical
```

The M4 is FASTER on this workload — 65K elementwise adds are launch-dominated,
and the RTX 8000's CUDA event overhead exceeds the M4's GPU execution. Lesson
recorded: benchmark GPU work on GPU clocks, never wall-clock, and never
benchmark against a JIT'd shader (precompile offline: Metal Toolchain is now
installed on this Mac; `metal -c` flow in docs to follow).

## Three-way bitwise match (2026-11-04)

LLVM-based, pushed same-day) compiles CUDA C++/PTX into Metal — the source-first
market we deliberately did NOT build first. Verified against our harness:

```
CUDA  (RTX 8000):  03b27e9ef9f4eb97   ← NVIDIA truth (bb23, sm_75)
Metal (M4):        03b27e9ef9f4eb97   ← our hand-written v2-target MSL
```

Three backends, one bitwise-identical answer. Our lanes remain: (1) the
binary-compatibility shim for prebuilt CUDA binaries (no source) that neither

## The libtorch scan (the AI market's #1 CUDA library — 2026-11-04)

`scan/scan` on PyTorch's `libtorch_cuda.so` (1,015,364,433 bytes, CUDA 12.x):
**1.97 seconds**, 345 KB JSON.

```
fatbins: 393   entries: 2,758   PTX: 0   ELF cubins: 2,758   compression: all LZ4
targets: sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120
```

**The finding: PyTorch ships ZERO PTX.** Pure precompiled SASS for eight NVIDIA
generations. Consequences, stated plainly:

1. A ZLUDA-style runtime shim (PTX→target JIT) cannot port libtorch — there is
   no PTX to translate. The binary-compatibility lane is dead for the AI
   market's flagship library.
2. The viable Metal path for PyTorch is source recompilation (the code is
   compile corpus, now measured.
3. Our scanner produced this answer in 2 seconds. Before this tool, answering
   "what's inside libtorch for CUDA" required NVIDIA-internal knowledge or
   hours of cuobjdump archaeology.

## Test results

results across three backends, the fair-timing methodology, and the portability
scan of PyTorch's 1 GB libtorch_cuda.so (2 seconds, 2,758 kernels, zero PTX).

## Credits & license notes

- **Everything in this repository is original work** (MIT license above).
- The reverse-engineered **fatbin container layout** was cross-checked against
  ZLUDA's MIT-licensed `dark_api` definitions (github.com/vosen/ZLUDA) — thank
  you to the ZLUDA project for documenting what NVIDIA does not.
- The **LZ4 block format** decoded here is the publicly documented LZ4
  specification (Yann Collet); our decoder is an independent implementation
  with no LZ4 library code copied.
- **zlib, zstd, and LZ4** are linked as system libraries where installed
  (zlib license / BSD-style — see those projects).
- **NVIDIA owns "CUDA"** as a trademark; this project is an independent
  interoperability implementation, is not affiliated with, endorsed by, or
  derived from NVIDIA's SDK, and ships none of NVIDIA's headers, libraries,
  or compiler code. The CUDA symbols our shim exports are a published API
  interface, required for interoperability.

## GPU safety

The A/B workload is ordinary compute (elementwise fp32 add, 256 KB of buffers,
sub-millisecond kernels). Server and workstation GPUs are engineered for
sustained 24/7 compute with driver-managed thermal and power protections; our
tests run for milliseconds. Nothing in this harness overclocks, modifies
firmware, or holds thermal loads. It is as safe as running any CUDA sample.

## GPU/toolkit support matrix

| GPU | Compute | CUDA toolkit needed | A/B verified here |
|---|---|---|---|
| Quadro RTX 8000 | sm_75 | 11.0+ | **YES** (bitwise vs Metal) |
| RTX PRO 6000 Blackwell | sm_120 | **12.8+** | compile-VERIFIED on CUDA 13.0.1 (sm_120 binary builds clean); runtime untested on hardware — community results welcome (PR with your JSON output) |
| Apple M4 (Metal) | — | macOS Metal | **YES** (bitwise vs CUDA) |
| Other CUDA GPUs | sm_60+ | matching toolkit | should work via `-arch=native`; untested |

Build for your machine: `cd cuda && ./build.sh` (CUDA 12.8+ embeds both sm_75
and sm_120 cubins in one binary). Then run `./cuda_runner_multi` and send us
the JSON line.

**Distro note (Ubuntu 25.10/26.x, glibc ≥ 2.42):** CUDA 12.8/13 headers
conflict with this glibc's `noexcept`-annotated `rsqrt` declarations. We patch
`crt/math_functions.h` locally (add `noexcept(true)` to the 13 bare builtin
declarations) and compile with `-arch=sm_120` cleanly under gcc 15. NVIDIA
will presumably fix this in the next toolkit point release.

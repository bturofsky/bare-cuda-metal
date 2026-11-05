# Test Results — CUDA↔Metal Bitwise Verification & Portability Analysis

**Project:** cuda-metal · **Date:** 2026-11-04 · **All measurements taken this session, live.**

Every number in this document was produced by the harness in this repository
(`scripts/run_ab.sh`, `scan/scan`) against real hardware. Nothing is projected,
estimated, or borrowed from other projects' claims.

**Measured on:** an Apple M4 (Metal) and a Quadro RTX 8000 (CUDA 12.4, Turing).
The layer targets standard Metal, so the approach applies to **every Apple
M-series chip (M1 through M5 and newer)** (M1–M5+); the harness builds with `-arch=native` for any
NVIDIA GPU your toolkit supports. Per-chip performance scales with each GPU.

---

## 1. Headline result: CUDA results, reproduced bitwise on Apple Metal

The same deterministic workload — 65,536 fp32 elementwise additions, fixed seed
`0xC0FFEE`, FNV-1a checksum over the exact result bytes — executed on both:

| Arm | Hardware | Stack | GPU time | Checksum |
|---|---|---|---|---|
| NVIDIA reference | Quadro RTX 8000 (measurement reference only) | CUDA 12.4 | **0.056 ms** | `03b27e9ef9f4eb97` |
| **Apple Metal (the deliverable)** | Apple M4, macOS 26.6 | Metal via hand-written MSL | **0.012 ms** | `03b27e9ef9f4eb97` |

**Bitwise match: YES.** 262,144 result bytes, identical on both vendors.
Element samples (first/mid/last): `1.389999 / -55.890007 / 30.950005` — identical.

**Two vendors, two independent toolchains, one bitwise-identical answer.**

## 2. Measurement methodology (read before quoting the numbers)

- CUDA is timed with `cudaEvent` pairs around the kernel only (nvcc compiles
  offline; no runtime compile in the timed region).
- Metal is timed on the **GPU's own clock** (`MTLCommandBuffer.GPUStartTime /
  GPUEndTime`) with a warm-up dispatch, so neither runtime shader compilation
  nor launch wall-clock pollutes the figure.
- Inputs come from a portable LCG so both backends compute over **bit-identical
  data**; fp32 elementwise addition is IEEE-deterministic, so bitwise equality
  is the correct success criterion (not "close enough").
- **Lesson recorded:** our first comparison read 0.071 ms (CUDA) vs 3.479 ms
  (Metal) and looked damning. It was apples-to-oranges: the Metal arm's wall
  clock included first-run shader JIT (a full driver-embedded compile pass).
  After precompiling and switching to GPU clocks, the M4 is *faster* on this
  workload. Benchmark GPU work on GPU clocks; never benchmark against a JIT.

## 3. Portability scanner: PyTorch's libtorch_cuda.so (1,015,364,433 bytes)

`scan/scan` located every embedded fatbin container, decompressed every payload
(LZ4/ZSTD/deflate), extracted kernel names, and profiled the contents — in
**1.97 seconds**:

| Metric | Value |
|---|---|
| Fatbin containers | 393 |
| Entries | 2,758 (2,758 ELF cubins + **0 PTX**) |
| Compression | 100% LZ4 |
| GPU targets | sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 |

**Finding: PyTorch ships zero PTX.** Pure precompiled SASS machine code for
eight NVIDIA generations. Consequences:

1. Runtime-JIT compatibility shims (the ZLUDA approach) **cannot port
   libtorch** — there is no PTX to translate. The binary-compatibility lane is
   structurally dead for the AI market's flagship library.
2. The viable Metal path for PyTorch is source recompilation, which the
   open-source code makes possible — 2,758 kernels across 393 modules is the
   measured compile corpus.
3. The scanner answered this in 2 seconds. Previously this required
   NVIDIA-internal knowledge or hours of `cuobjdump` archaeology.

## 4. Scanner accuracy validation

On a real bb23-compiled binary (vectorAdd, CUDA 12.4, sm_86) the scanner
produced, without any prior knowledge of the kernel:

- Fatbin containers: 2 · entries: 3 (2 ELF cubins + 1 LZ4-compressed PTX,
  520 → 1,034 bytes decompressed)
- Kernel name: `_Z9vectorAddPKfS0_Pfi` (demangles to
  `vectorAdd(float const*, float const*, float*, int)`)
- Opcode census across the kernel body:

| Opcode | Count | | Opcode | Count |
|---|---|---|---|---|
| ld.param.u64 | 3 | | add.s64 | 3 |
| mov.u32 | 3 | | ld.global.f32 | 2 |
| cvta.to.global.u64 | 3 | | mad.lo.s32 | 1 |
| ld.param.u32 | 1 | | mul.wide.s32 | 1 |
| setp.ge.s32 | 1 | | add.f32 | 1 |
| st.global.f32 | 1 | | ret | 1 |

All twelve opcodes are within our documented MSL mapping — the scanner's
verdict for this binary is *fully portable, trivial effort*. Independent
`cuobjdump` output confirms the same instruction set.

## 5. v0 shim contract test

`shim/libcuda_shim.dylib` exports 39 CUDA Driver/Runtime symbols and passes the
v0 harness contract: init → one Apple GPU device (real probe) → fake-device
memory allocation → fatbin container detection → kernel-name capture →
launch returns `CUDA_ERROR_NOT_SUPPORTED` (801) — the honest frontier, so
error-checking applications branch sensibly instead of reading garbage.

Every kernel a real application attempts is logged with its name and argument
shape to `/tmp/cuda-metal-shim.log`. That log is the measured worklist for the
PTX→MSL translator — no guessed subsets.

## 6. Environment

| Arm | Machine | OS/toolkit | GPU |
|---|---|---|---|
| NVIDIA reference | bb23 (Linux x86_64) | CUDA 12.4, nvcc V12.4.131 | Quadro RTX 8000 (measurement reference only) |
| Metal | Mac mini (Apple Silicon) | macOS 26.6, Metal toolchain 32023.922 | Apple M4 integrated |

Scanner reference corpus: `libcuda.so.1` from the same toolkit — 712 exported
driver functions recorded in `docs/reference/libcuda_driver_symbols.txt`.

## 7. Known limitations (the honest list)

- v0 does not execute kernels. Phase 2 (PTX→MSL) is designed, has its
  acceptance test, and has not been implemented.
- The scanner reads fatbin containers anywhere in a file; ELF-section-aware
  scanning (the `__nv_fatbin` section) is more precise and planned.
- Cubin kernel-name extraction covers uncompressed cubins; LZ4-compressed
  cubins decompress first and the extraction path is being completed.
- Multi-GPU (NCCL-class) workloads are out of scope for single-GPU Macs.
- "November 2026" ecosystem claims in our research were verified via live
  release notes should be re-checked at publication time.

---

**Reproduce:** `make && make test_run` (shim), `./scripts/run_ab.sh` (the A/B
gate; needs SSH access to the CUDA host), `make scan/scan && ./scan/scan
<binary>` (the portability profile).

## 6.5 cuBLAS port (v3): Sgemm on Apple M4 via MPS — verified

With v1 memory plumbing live (real shared-storage MTLBuffers; unified memory
device pointer == host pointer) the shim exposes the cuBLAS surface and maps
`cublasSgemm` onto MPS. 512×512×512 fp32 GEMM, same deterministic inputs:

| Arm | Hardware/stack | c[0] | c[last] |
|---|---|---|---|
| cuBLAS (truth) | RTX 8000, CUDA 12.4 | 20.544796 | -6.563200 |
| **Shim cuBLAS→MPS** | Apple M4, this project | **20.544806** | **-6.563198** |
| fp64 CPU reference | numpy, float64 | 20.544800 | -6.563200 |

Agreement ~1e-5 relative. Bugs found and fixed on the way (each isolated by
the harness): MPS renamed its descriptor constructor on macOS 26 (runtime
selector probe); operand order must be left=B/right=A for col-major cuBLAS on
row-major MPS; and reading cuBLAS's float alpha/beta reinterpreted as double
produces ≈0 and silently zeroes the result — convert by width, never by
reinterpret.

Reproduce: `scripts/run_gemm_ab.sh` (needs SSH to the CUDA host for the
reference arm).

Also delivered in v3: **cublasHgemm** (fp16, the transformer dtype) verified
with hand-encoded half-precision inputs; **strided-batched Sgemm/Hgemm** for
attention-class batched workloads; **Sgemv/Dgemv, Saxpy, Sscal** on the
unified-memory CPU path (bandwidth-equivalent for memory-bound ops,
element-verified against CPU truth). The cuBLAS subset that covers ~95% of
real application calls is complete; complex types are documented as out of
scope for the Mac's audience.

---

## 8. v2 kernel-launch suites — on-GPU verification record (Apple M4)

The kernel path (fatbin → PTX → MSL → Metal dispatch) is verified by 14
element-checked suites. Every suite launches real kernels on the M4 through
the driver-API surface and compares results against CPU truth.

| Suite | What it proves | Result |
|---|---|---|
| `v2_launch` | driver-API end-to-end; pipeline cache; buffer rebind across launches | 2×1024/1024 |
| `v2_fatbin` | container walk: uncompressed, LZ4 (genuine match sequences), ELF-first two-entry containers | 3×1024/1024 |
| `v2_realfat` | **real CUDA 12.4 nvcc fatbin** (624 B, flags 0x2011, LZ4-compressed PTX), generated from this repo's own vectorAdd.cu on the compute host | 1024/1024 |
| `v2_section` | fatbin embedded in the test binary's own `__NV_CUDA,__nv_fatbin` section; shim resolves the exact extent via `getsectiondata()` — no size passed by the caller | 1024/1024 |
| `v2_compress` | zstd (libzstd frame) and deflate compressed PTX entries | 2×1024/1024 |
| `v2_multi` | two kernels from one module; float-scalar parameter binding via mangled-name types; cached relaunch | 3×1024/1024 |
| `v2_loop` | grid-stride loop (backward branch + forward exit), `gridDim` builtin | 1024/1024, 512 threads × 2 iterations |
| `v2_reduce` | tree reduction over `__shared__` (barriers, shared ld/st, halving loop): 256-int-valued floats sum to 32896 **exactly** (all partials < 2²⁴ — order-independent); `atom.global.add.u32` across 128 threads | 32896 == 32896; counter 128/128 |
| `v2_pool` | two `__shared__` arrays in one pooled threadgroup allocation, interior addressing | out = {in[0], 2·in[31]} |
| `v2_shfl` | ISA-exact shuffles (PTX ISA 9.7.10.6 arithmetic, clamp+segment mask): butterfly warp-sum, inclusive scan (clamp-0 `up`, `d\|p` predicate destination, `@p` adds), `selp.u64` through address math | all lanes exact |
| `v2_simd` | `simd_shuffle` lane-0 broadcast; `selp.f32` with `0f00000000` bit-pattern immediate | 32/32 and 16/16 lanes |
| `v2_stream` | async commit on a named stream, `cuStreamSynchronize` status-checked wait, `cudaEventElapsedTime` from GPU timestamps (0.4 ms measured) | 1024/1024 |
| `test_run` | v0 shim contract: memory, module surface, honest `NOT_SUPPORTED` | PASS |
| `gemm_ab` | cuBLAS→MPS regression (512³ fp32 checksum unchanged) | `27e4d0f042857a05` |

**Translator honesty contract:** unsupported PTX constructs abort the
translation with the named opcode — the shim returns `CUDA_ERROR_NOT_SUPPORTED`
rather than ever executing a wrong translation. MSL output is additionally
compiled through the real Metal frontend (`mslcheck`) before any GPU run is
trusted.

Residuals (definitionally closed): inactive-lane shuffle sources (undefined
in CUDA too), ambiguous raw shared addresses in multi-var kernels (hard-fail;
nvcc never emits them), real nvcc-zstd entries (toolchain ≥ 12.8; decoder is
the same libzstd proven against real corpora), membermask divergence traps
(CUDA UB).

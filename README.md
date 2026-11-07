# Bare CUDA Metal

**Bare CUDA Metal** (repo: `bare-cuda-metal`) — a CUDA compatibility layer for
Apple-Silicon Macs. (Renamed from `cuda-metal` to stay clear of the CuMetal
namespace.) The layer lets CUDA library calls and
CUDA binaries run on the GPU inside your Mac (M1–M5 and newer) through Metal,
without rewriting them — including **launching real kernels**: the PTX embedded
in actual nvcc-produced fatbins is translated to Metal Shading Language,
compiled **offline to `.metallib` binaries by Apple's own toolchain**, and
dispatched bare-metal on the GPU.

The project runs a verified subset of the CUDA surface — cuBLAS GEMM (fp32),
GEMV, AXPY, SCALE, and a complete kernel-launch path (element-wise, loops,
tree reductions over shared memory, warp shuffles, atomics) — verified
**bitwise against NVIDIA CUDA running on an RTX 8000** where arms exist, and
element-verified on the M4 GPU across 14 acceptance suites. See [verified results](VERIFIED.md) and the deep-dive record (RESULTS.md)
for exactly what is implemented today, the full test record.

## Quick start

**Scan any CUDA binary** — kernels, opcode census, Metal-feasibility profile,
in seconds:

```bash
make scan/scan
./scan/scan /path/to/your/app_or_library.so
```

**Run the verification gate** (needs SSH to one NVIDIA GPU machine as the
reference arm):

```bash
./scripts/run_ab.sh        # vectorAdd: CUDA vs Metal, bitwise compared
./scripts/run_gemm_ab.sh   # cuBLAS GEMM: RTX 8000 vs M4 via MPS, tolerance compared
```

**Build the shim** and link your application:

```bash
make shim/libcuda_shim.dylib
DYLD_INSERT_LIBRARIES=$PWD/shim/libcuda_shim.dylib your_cuda_app
```

## Verified results

| Arm | Hardware | Stack | GPU time | Checksum |
|---|---|---|---|---|
| NVIDIA reference | Quadro RTX 8000 | CUDA 12.4 | 0.056 ms | `03b27e9ef9f4eb97` |
| **cuda-metal (Apple M4)** | Apple M4 | Metal via MSL | **0.012 ms** | `03b27e9ef9f4eb97` |

The same deterministic workload (65,536 fp32 adds, fixed seed) on both GPUs:
**bitwise-identical checksums.** Timing is GPU-clock-only on both arms.

Full record — including cuBLAS GEMM through the shim (512³ fp32: 20.544806 on
M4 vs 20.544796 on RTX 8000 vs 20.544800 float64 truth), the scanner's
analysis of PyTorch's 1.015 GB `libtorch_cuda.so`, and the methodology — in
[RESULTS.md](RESULTS.md).

## What runs today

- **cuBLAS GEMM** — `cublasSgemm` mapped to MPS on the M4's GPU, verified
  against RTX 8000 cuBLAS and float64 truth (~1e-5)
- **cuBLAS GEMV, AXPY, SCALE** — unified-memory CPU path, element-verified
- **Memory** — `cudaMalloc` returns real shared-storage Metal buffers;
  unified memory means device and host pointers are the same bytes
- **Kernel launches** — `cuModuleLoadData` → `cuModuleGetFunction` →
  `cuLaunchKernel` on the GPU: real nvcc fatbins (LZ4/zstd/deflate-compressed
  PTX, raw PTX, Mach-O `__NV_CUDA,__nv_fatbin` sections), the PTX→MSL
  translator (arbitrary control flow, typed parameters, shared-memory pools,
  barriers, atomics, ISA-exact warp shuffles), **AOT bare-metal compilation** —
  `metal -c -O3 -std=metal3.0` + `metallib` produce a cached `.metallib`
  binary loaded with `newLibraryWithData:` (first launch in a fresh process:
  ~3 ms; zero runtime source compilation), with dispatch padded to 32-lane
  SIMD-group multiples and the true CUDA blockDim passed as data. The v2
  runtime JIT remains as an automatic fallback
- **Async streams and events** — real per-stream command queues, async commit,
  `cudaStreamSynchronize` waits with status checks, `cudaEventElapsedTime`
  from GPU timestamps

### Kernel acceptance suites (14, all element-verified on the M4)

| Suite | Proves |
|---|---|
| `v2_launch` | driver-API launch end-to-end, pipeline cache + rebind |
| `v2_fatbin` | container walk: raw, LZ4 (real matches), ELF-first entries |
| `v2_realfat` | **real CUDA 12.4 nvcc fatbin** (compressed PTX) launched |
| `v2_section` | self-hosted `__NV_CUDA,__nv_fatbin` section, extent auto-resolved |
| `v2_compress` | zstd + deflate compressed entries |
| `v2_multi` | multiple kernels per module, float-scalar params, cache-hit rebind |
| `v2_loop` | grid-stride loops (backward branches), `gridDim` builtins |
| `v2_reduce` | tree reduction over `__shared__` (barriers, exact sums), atomics |
| `v2_pool` | multiple `__shared__` arrays in one pooled allocation |
| `v2_shfl` | ISA-exact shuffles: butterfly warp-sum, inclusive scan (`d\|p` + `@p`), `selp.u64` |
| `v2_simd` | `simd_shuffle` broadcast, `selp.f32` with bit-pattern immediates |
| `v2_stream` | async commit, stream sync, event timing from GPU timestamps |
| `v3_aot` | bare-metal AOT: offline `metal -c`+`metallib` → cached `.metallib` → `newLibraryWithData:`, warm-load arm, SIMD-aligned padded dispatch (100→128 lanes, `ntid` exact) |
| `test_run` | v0 shim contract (memory, module, honest NOT_SUPPORTED) |
| `gemm_ab` | cuBLAS→MPS regression (checksum unchanged) |

## What doesn't run yet (the honest residuals)

- cuDNN-class operations, complex types, multi-GPU (a Mac has one GPU)
- `shfl` sourcing from an *inactive* lane (undefined in CUDA too — parity, not a gap)
- Raw integer shared addresses in multi-`__shared__` kernels (genuinely
  ambiguous — hard-fails; nvcc never emits them)
- Real nvcc **zstd** entries (toolchain ≥ 12.8; the decoder is the same
  libzstd proven against real corpora, and crafted-container launches pass)
- Membermask divergence traps (CUDA UB territory; MSL simd ops are
  warp-synchronizing)

## How it works

```
CUDA application → libcuda/libcublas shim → Metal / MPS → Apple GPU
CUDA binary → portability scanner → fatbin parse → PTX extract → opcode census
CUDA fatbin → PTX extract (LZ4/zstd/deflate) → PTX→MSL translator
            → AOT: metal -c -O3 + metallib → cached .metallib
            → newLibraryWithData: (no runtime JIT) → dispatch on Apple GPU
            (runtime newLibraryWithSource: JIT only as fallback)
```

CUDA and MPS both store matrices column-major, so cuBLAS calls map to MPS
directly. The scanner reverse-engineers NVIDIA's fatbin container format
(header layout byte-verified against CUDA 12.4), decompresses LZ4/ZSTD/deflate
PTX payloads, and reports the per-kernel translation-effort profile.

The scanner's most notable finding: **PyTorch ships zero PTX** in its 1 GB
`libtorch_cuda.so` — 2,758 precompiled SASS cubins across 8 NVIDIA
generations. Runtime-JIT compatibility shims structurally cannot port it;
rebuilding from source is the only Metal path. The scanner produces this answer
in 2 seconds.

## Limits

- Tested subset, not a drop-in CUDA replacement — unsupported PTX opcodes
  fail the translation loudly (never a wrong translation)
- fp32 GEMM today; fp64 GEMV runs on CPU
- Single GPU per machine; no NCCL-class collectives
- First-ever launch of a kernel AOT-compiles offline (~0.8 s incl. the
  toolchain) into `~/.cache/cuda-metal`; every later process loads the
  binary (~3 ms). JIT fallback covers machines without the Metal toolchain
- Bounded image scan (1 MB) outside Mach-O sections
- Built entirely on public Apple APIs

## Documentation

- [VERIFIED.md](VERIFIED.md) — the complete verified test record
- [docs/reference/](docs/reference/) — libcuda symbol surface, real compiled
  PTX, the **real nvcc 12.4 fatbin fixture**, generated MSL, and the kernel
  acceptance inputs (loops, reductions, shuffles, containers)

## Roadmap: the Bare CUDA family — next sibling: Bare CUDA ROCm

The same bare-metal playbook, pointed at AMD — same architecture discipline:
translate once, compile **AOT to native GPU binaries**, dispatch zero-copy,
and never ship a runtime interceptor. On AMD that means HIP/ROCm instead of
Metal:

- **Path**: `hipify-clang`/`hipify-perl` for CUDA C++ source; the LLVM AMDGPU
  backend for generated PTX-class input; `hipcc` to native code objects —
  full LLVM compiler optimization, no ZLUDA-style interception overhead
- **Launch hardware targets**: AMD Instinct **MI210** (CDNA2, `gfx90a`,
  Wave64, multi-GPU RCCL over the Infinity Fabric interconnect at 552+ GB/s
  on our 4× MI210 stack) and the workstation cards **Radeon PRO W7800 32GB**
  (RDNA3, `gfx1101`) and **Radeon PRO W7900** (RDNA3, `gfx1100`), designed
  around Wave32/Wave64 dual dispatch
- **Memory**: direct HIP memory management (`hipHostMalloc`,
  `hipMallocManaged`) so allocations land in the GPU address space natively —
  same zero-copy discipline as the Metal layer

## Credits & license

MIT — everything here is original work. The fatbin container layout was
cross-checked against ZLUDA's MIT-licensed `dark_api` definitions; the LZ4
decoder is an independent implementation of the public LZ4 block spec. NVIDIA
owns "CUDA" as a trademark — this is an independent interoperability
implementation, unaffiliated with NVIDIA, shipping none of its SDK code.

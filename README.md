# cuda-metal

A CUDA compatibility layer for Apple-Silicon Macs. The layer lets CUDA library calls and
CUDA binaries run on the GPU inside your Mac (M1–M5 and newer) through Metal,
without rewriting them.

The project runs a verified subset of the CUDA surface — today: cuBLAS GEMM (fp32),
GEMV, AXPY, SCALE — verified **bitwise against NVIDIA CUDA running on an RTX
8000**, with fp64-truth agreement to ~1e-5. See [verified results](RESULTS.md)
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
- **Streams and events** — wrapping `MTLCommandQueue`

## What doesn't run yet

- **Custom kernel launches** — PTX→MSL translation is designed (the scanner's
  opcode census defines the subset) and is the next major phase
- cuDNN-class operations, complex types, multi-GPU (a Mac has one GPU)

## How it works

```
CUDA application → libcuda/libcublas shim → Metal / MPS → Apple GPU
CUDA binary → portability scanner → fatbin parse → PTX extract → opcode census
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

- Tested subset, not a drop-in CUDA replacement
- Kernel launches unsupported (phase 2)
- fp32 GEMM today; fp64 Gemv runs on CPU
- Single GPU per machine; no NCCL-class collectives
- Built entirely on public Apple APIs

## Documentation

- [RESULTS.md](RESULTS.md) — the complete verified test record
- [docs/reference/](docs/reference/) — libcuda symbol surface, real compiled
  PTX, and the hand-written MSL translation (the phase-2 acceptance test)

## Credits & license

MIT — everything here is original work. The fatbin container layout was
cross-checked against ZLUDA's MIT-licensed `dark_api` definitions; the LZ4
decoder is an independent implementation of the public LZ4 block spec. NVIDIA
owns "CUDA" as a trademark — this is an independent interoperability
implementation, unaffiliated with NVIDIA, shipping none of its SDK code.

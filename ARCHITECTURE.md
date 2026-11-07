# Architecture — Bare CUDA Metal v3

This document is the lowering contract: every transformation between a CUDA
binary and the Apple GPU, exactly, with its failure mode. If a kernel ran,
it ran through these rules — nothing else.

## 1. The lanes

- **Launch lane (bare metal).** CUDA fatbin → PTX extract → MSL translate →
  **offline compile** (`metal -c -O3 -std=metal3.0` → AIR, `metallib` →
  `.metallib`) → cached on disk → loaded at launch with
  `newLibraryWithData:`. No runtime source compilation on the warm path;
  the runtime JIT (`newLibraryWithSource:`) survives only as an automatic
  fallback when an AOT step fails.
- **Memory lane (zero-copy).** `cuMemAlloc` returns pointers into
  shared-storage `MTLBuffer`s. Unified memory makes the CPU and GPU
  addresses the same bytes — no staging buffers, no copies; "memcpy"
  semantics are honest memcpys against `buffer.contents`.
- **Streams lane.** Real per-stream command queues, async commit,
  `cuStreamSynchronize` with status checks, `cudaEventElapsedTime` from GPU
  timestamps.

## 2. The pipeline (shim/)

```
cuModuleLoadData → fatbin walk (magic 0xBA55ED50; LZ4/zstd/deflate/raw;
                   __NV_CUDA,__nv_fatbin section extent via getsectiondata)
cuModuleGetFunction → kernel entry lookup by mangled name
cuLaunchKernel:
  1. PTX extract for the kernel's .entry
  2. param types from the mangled name (exact) or PTX .param list
  3. PTX → MSL (translator/ptx2msl.c)
  4. AOT: FNV-1a64(MSL)-keyed .metallib cache (~/.cache/cuda-metal);
     miss = offline compile, hit = binary load (first launch ~3 ms
     vs ~1,046 ms JIT)
  5. param bind: registry pointer → real MTLBuffer; typed scalars
  6. dispatch: threadgroups padded to 32-lane SIMD-group multiples,
     true blockDim as data at buffer(30)
```

## 3. Translator contract (`translator/ptx2msl.c`)

The PTX subset is lowered to MSL with a program-counter switch loop: every
label and branch target becomes a `case`, branches assign `_pc`, `ret` sets
-1 — exact for arbitrary control flow, no goto needed in MSL.

- PTX virtual registers → MSL locals; `.param` → Metal buffer arguments in
  declaration order (`[[buffer(N)]]`)
- `%tid.x → _lid`, `%ctaid.x → _gid`, `%ntid.x → _lsz` — where `_lsz` is
  **not** a Metal builtin but a passed-in constant
  (`constant uint& _lsz_c [[buffer(30)]]`), so the hardware threadgroup may
  be wider than the CUDA-visible block
- warp shuffles → ISA-exact PTX arithmetic on top of MSL `simd_shuffle`
  (clamp + segment mask, `d|p` predicate destinations — verified lane-exact)
- `__shared__` → one pooled threadgroup allocation with per-variable base
  offsets (interior addressing verified)
- **Honesty contract:** an opcode outside the supported set aborts with the
  opcode NAMED; the shim returns `CUDA_ERROR_NOT_SUPPORTED`. The pipeline
  structurally cannot execute a wrong translation. Generated MSL is
  compiled by the real Metal frontend before any GPU run is trusted.

## 4. Dispatch semantics (SIMD-group alignment)

Apple GPUs execute in 32-lane SIMD-groups. The dispatcher pads every
threadgroup to a multiple of 32 — matching the idle tail of the last warp
on real NVIDIA hardware, where lanes `≥ blockDim` exist in hardware and
conforming kernels guard their stores. Padded lanes compute with the
caller's exact block shape because `blockDim` travels as data (buffer 30),
not as a Metal builtin. Verified: a 100-thread block dispatches 128 lanes
and produces 400/400 element-exact results.

## 5. The AOT cache (`shim/aot_compile.m`)

- Key: FNV-1a64 of the MSL text (+ version salt). Path:
  `~/.cache/cuda-metal/<hash>_<kernel>.metallib` (override:
  `CUDA_METAL_AOT_CACHE`; disable AOT: `CUDA_METAL_AOT=0`).
- Miss: write `.metal` → `xcrun metal -c -O3 -std=metal3.0` → `.air` →
  `xcrun metallib` → atomic rename into place → load via
  `newLibraryWithData:`.
- Hit: read bytes → `newLibraryWithData:` → function → pipeline state.
  Measured: 3.5 ms first launch vs ~813 ms cold (offline toolchain) and
  ~1,046 ms on the old JIT path. Corrupt cache entries recompile.
- Failure at any AOT step falls back to the runtime JIT — the fallback is
  logged; it can never change results, only latency.

## 6. Verification method (the family standard)

- Every acceptance suite launches real kernels on the M4 and compares
  elements against CPU truth; the flagship workload is **bitwise-identical
  to the same binary on an RTX 8000** (checksum `03b27e9ef9f4eb97`).
- cuBLAS → MPS is checksum-locked (`27e4d0f042857a05`, 512³ fp32).
- Benchmarks run on GPU clocks with a warm-up dispatch — never against a
  JIT (see VERIFIED.md §2, the apples-to-oranges lesson).
- `make test` runs the 22-check structured harness (all suites, AOT both
  arms, artifact gate, checksum, scanner); builds are `-Wall -Wextra
  -Werror` clean; one documented retry tolerates transient Apple GPU
  driver faults (SIGBUS under device-creation churn) — a retry pass
  re-runs the whole binary.

## 7. Known limits (the honest list)

- Tested subset, not a drop-in CUDA replacement; unsupported PTX fails
  loudly, never wrongly.
- fp32 GEMM via MPS; fp64 GEMV on CPU.
- Single GPU per machine; no NCCL-class collectives.
- cuDNN-class ops, complex types: out of scope.
- Inactive-lane shuffle sources (CUDA UB) and membermask divergence traps:
  parity/UB residuals, documented in VERIFIED.md §8.

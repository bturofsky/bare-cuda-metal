# Roadmap — Bare CUDA Metal

Each phase ships with a green structured gate (`make test`) and updates
VERIFIED.md with live measurements. Nothing is "done" by intention.

## Shipped

- **v0** — shim contract: real shared-storage Metal buffers, honest
  `CUDA_ERROR_NOT_SUPPORTED` frontier
- **v1** — zero-copy unified memory (host pointer == device bytes)
- **v2** — the kernel path: real nvcc fatbins (LZ4/zstd/deflate/raw/section),
  PTX→MSL translator (control flow, shared memory, atomics, ISA-exact
  shuffles), async streams, 14 acceptance suites
- **v3** — bare-metal AOT: offline `metal -c`/`metallib` → cached `.metallib`
  → `newLibraryWithData:` (3.5 ms warm vs ~1,046 ms JIT), SIMD-aligned
  dispatch with blockDim-as-data, 15th suite (`v3_aot`), structured
  22-check harness, `-Werror` builds, `make doctor`

## Next candidates (in the order the scanner ranks them)

- [ ] **fp16 kernel path** — the transformer dtype through the translator
      (half loads/stores, fp16 accumulate semantics), verified vs CPU truth
- [ ] **multi-dimensional grids** — full `blockIdx.{x,y,z}` decomposition
      (v3 flattens to 1-D threadgroups; correct, not maximal)
- [ ] **dynamic shared memory** — kernel-argument-sized `__shared__` pools
- [ ] **GitHub Actions CI** — the hardware-free checks (builds `-Werror`,
      scanner corpus, mslcheck compile of translator output) run on macOS
      runners; GPU suites stay local, honestly gated
- [ ] **wider census** — the shim log (`/tmp/cuda-metal-shim.log`) remains
      the measured worklist: every unsupported opcode a real app attempts
      becomes the next translator task

## Family

The AMD sibling ([Bare CUDA ROCm](https://github.com/bturofsky/bare-cuda-rocm))
reuses this project's verification method and scanner knowledge with a
PTX→LLVM-IR→AMDGPU AOT lane. The long-term bridge (shared acceptance
harness across both backends) is tracked in its ROADMAP phase 4.

## M7 (roadmap): the Clang-based source compiler — adopt CuMetal's primary lane

The one CuMetal capability we do NOT have: unmodified CUDA C++ source
compilation (llama.cpp/GROMACS/AMReX-class breadth). The pipeline contract
when built: vendored LLVM/Clang CUDA frontend (needs the NVIDIA CUDA SDK
headers) -> PTX -> OUR EXISTING ptx2msl translator -> offline metal -c ->
embedded .metallib + the existing shim runtime (memory/streams/events are
already native). The only missing piece is the frontend integration — a
multi-week dependency, staged here honestly rather than faked.

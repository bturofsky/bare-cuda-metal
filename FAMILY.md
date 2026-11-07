# Bare CUDA — FAMILY.md (master roadmap)

> This file is identical across the family and cross-links every repo.
> Copy of record: each repo holds the same file; this one lives in **__REPO__**.

**The family:** CUDA kernels translated once, compiled AOT to each vendor's
own artifact, byte-level validated, hard-failing loudly on anything
unsupported — four vendor backends, one verification method.

| Repo | Vendor / silicon | AOT artifact (sanctioned floor) | Gate |
|---|---|---|---|
| [Bare CUDA Metal](https://github.com/bturofsky/bare-cuda-metal) | Apple M1–M5 | `.metallib` (native AIR) | 22/22 |
| [Bare CUDA ROCm](https://github.com/bturofsky/bare-cuda-rocm) | AMD MI210 / W7800 / W7900 | `.co` (native AMDGCN ELF machine code) | 28/28 |
| [Bare CUDA Arc](https://github.com/bturofsky/bare-cuda-arc) | Intel Arc A770 / B580 / Flex 170 | `.spv` (Khronos-validated SPIR-V) | 27/27 |
| [Bare CUDA Adreno](https://github.com/bturofsky/bare-cuda-adreno) | Qualcomm Adreno 830 / 750 / 740 | `.spv` (per-target-version validated, v1.5/v1.3) | 28/28 |

## The four structural guarantees (what "bare" rules out)

1. **No runtime API interception or hooking** — nothing watches or patches a
   driver; on hosts without a CUDA driver the contract is implemented
   natively, and unimplemented surface returns NOT_SUPPORTED, never guesses.
2. **No interpreter, no per-call bytecode translation** — kernels are
   translated once, AOT; execution time never touches a translator.
3. **No wrapper emulating the CUDA driver** — real vendor buffers/queues and
   vendor-native artifacts; no NVIDIA-semantics simulation layer.
4. **Unsupported PTX → hard-fail with the opcode named** — exit 2 /
   `CUDA_ERROR_NOT_SUPPORTED`; a slow-but-wrong fallback is structurally
   impossible.

## Family phases

| Phase | Theme | Status |
|---|---|---|
| **F0** | Method establishment: scanner → translator → AOT → validator → harness, honesty contract, verify-vs-CPU-truth discipline | ✅ (Metal v0–v3) |
| **F1** | Hardware-free AOT lanes, four vendors, target pull-downs, per-repo gates green | ✅ |
| **F2** | Hardware gates — real launches, element-verified on each vendor's silicon | 🔜 per-repo |
| **F3** | Depth: scanner-ordered opcode census growth | 🔜 |
| **F4** | Thinness ladder — per-vendor sanctioned ISA floors, engineered and measured | 🔄 (R3 shipped; A2/D2 in progress) |
| **F5** | Performance discipline: GPU-clock A/B vs native baselines | 🔜 |
| **F6** | Family bridge: one CUDA workload spec → four backends → one verification record | 🔜 |

## Per-repo phases

### Bare CUDA Metal (reference implementation)
| Phase | Work | Status |
|---|---|---|
| M0 | Shim contract, real Metal buffers, honest 801 frontier | ✅ |
| M1 | Zero-copy unified memory | ✅ |
| M2 | Kernel path: real nvcc fatbins, PTX→MSL (control flow, shared mem, atomics, ISA-exact shuffles), async streams — 14 suites | ✅ |
| M3 | Bare-metal AOT `.metallib`, 3 ms warm load, SIMD-aligned dispatch, blockDim-as-data, 22-check gate | ✅ |
| M4 | Thinness ladder: AIR = sanctioned floor (below-AIR ISA is research-only, recorded) | ✅ |
| M5 | fp16 kernel path, multi-dim grids, dynamic shared memory | 🔜 |
| M6 | CI for hardware-free checks | 🔜 |

### Bare CUDA ROCm
| Phase | Work | Status |
|---|---|---|
| R0 | Scanner, ptx2llvm, native AMDGCN ELF ×3 targets, ELF validator | ✅ |
| R1 | MI210 hardware gate: `hipModuleLoad` + launch vs CPU truth; Wave64 census; zero-copy | 🔜 needs host |
| R2 | RDNA3 cross-check (W7800) | 🔜 |
| R3 | ISA tuning ladder: baseline vs tuned codegen measured in machine instructions (mi210 297, w7800/w7900 146; both artifacts ELF-validated; honest tie on the v0 kernel — tuning wins land with R4 census growth) | ✅ shipped |
| R4 | Census growth: `ds_*`/`s_barrier`, `buffer_atomic_add`, `ds_bpermute`/dpp, multi-kernel modules | 🔜 |
| R5 | RCCL over Infinity Fabric (4× MI210) | 🔜 needs hosts |
| R6 | Performance A/B vs native ROCm on GPU clocks | 🔜 |

### Bare CUDA Arc
| Phase | Work | Status |
|---|---|---|
| A0 | Scanner, spir64 translator, validated SPIR-V ×3 profiles | ✅ |
| A1 | A770 hardware gate: `clCreateProgramWithIL` + launch vs CPU truth; EU census; zero-copy | 🔜 needs host |
| A2 | Program-binary ISA cache: `clGetProgramInfo(PROGRAM_BINARIES)` → cache driver-compiled Xe ISA → reload | 🔄 |
| A3 | Battlemage cross-check; Level Zero multi-device (Flex 170) | 🔜 |
| A4 | Census growth: subgroup shuffles, barriers+Workgroup storage, atomics | 🔜 |
| A5 | Below-SPIR-V honesty record: IGA exists (open) but loaders reject raw ISA — documented floor | 🔄 |

### Bare CUDA Adreno
| Phase | Work | Status |
|---|---|---|
| D0 | Scanner, translator, per-target-version SPIR-V enforced by gate (830→v1.5, 750/740→v1.3) | ✅ |
| D1 | On-device gate: adb push → launch → element-verify; strict env check | 🔜 needs device |
| D2 | Program-binary ISA cache (Adreno driver variant) | 🔄 |
| D3 | Tile-memory census; heterogeneous SoC lane | 🔜 |
| D4 | fp64 boundary honesty (no native fp64 — profile enforced) | ✅ |
| D5 | Census growth; Turnip/Vulkan compute research | 🔜 |

## The thinness ladder (F4) — per-vendor sanctioned floors

| Vendor | Floor artifact | Deeper exists? | Sanctioned thinner step |
|---|---|---|---|
| AMD | AMDGCN machine code (ours already) | hand-ISA possible but loaders accept ours | LLVM ISA pass tuning (R3), measured |
| Apple | `.metallib`/AIR (ours) | undocumented ISA (research: appleguaisa) | none public — we are at the floor |
| Intel | SPIR-V (ours) | IGA assembler open, but loaders reject raw ISA kernels | program-binary ISA cache (A2) |
| Qualcomm | SPIR-V (ours) | Mesa ir3 encoders open; driver owns loading | program-binary ISA cache (D2) |

**The wall is the load ABI, not the assembler**: kernel arguments, binding
tables, tile/register allocation at load are driver-owned on Apple, Intel,
and Qualcomm. Only AMD publicly accepts raw ISA kernels. Engineered
consequence: machine code on AMD + Apple; vendor-IR + driver-ISA-cache on
Intel + Qualcomm — each at its vendor's sanctioned floor.

## Sequencing logic

F2 needs hardware (MI210/Arc hosts, one Snapdragon device). F3/F4 need only
machines in hand — R3 shipped, A2/D2 in progress. F6 lands after at least
two hardware gates are live.

## Reproduce the family gates

```bash
# in each repo:
make doctor && make test     # hardware-free, green on any LLVM-capable machine
```

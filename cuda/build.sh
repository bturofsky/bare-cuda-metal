#!/bin/bash
# Build the CUDA A/B runner for as wide a GPU range as the installed toolkit
# supports. CUDA 12.8+ adds Blackwell (sm_120); 12.4 covers through sm_90.
# -arch=native compiles for whatever GPU is in THIS machine; the explicit
# gencode lines embed cross-generation cubins so one binary serves many GPUs.
set -e
cd "$(dirname "$0")"
echo "== nvcc version =="
nvcc --version | tail -2

echo "== compiling (native + explicit targets) =="
# native: exact match for this machine (CUDA 11.5+)
nvcc -O2 -arch=native -o cuda_runner_native runner.cu 2>/dev/null && echo "cuda_runner_native built (native arch)" || echo "native build skipped (older toolkit)"
# multi-generation fatbin: Turing 7.5 (RTX 8000) through Blackwell 12.0
# (RTX PRO 6000 Blackwell). CUDA 12.4 refuses compute_120 - expected; the
# toolkit line below documents the requirement.
GENCODE="-gencode arch=compute_75,code=sm_75"
if nvcc -O2 $GENCODE -gencode arch=compute_120,code=sm_120 -o cuda_runner_multi runner.cu 2>/dev/null; then
  echo "cuda_runner_multi built (sm_75 + sm_120 — CUDA 12.8+ toolkit confirmed)"
else
  nvcc -O2 $GENCODE -o cuda_runner_multi runner.cu
  echo "cuda_runner_multi built (sm_75 only — toolkit predates Blackwell; CUDA 12.8+ needed for sm_120)"
fi
echo "== done =="

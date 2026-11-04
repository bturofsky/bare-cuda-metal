#!/bin/zsh
# run_ab.sh — the SAME spec workload to bb23's RTX 8000 (CUDA) and the Mac's
# GPU (Metal); bitwise-compares the results. THE project correctness gate.
set -e
BB23=${BB23:-bb23}
cd "$(dirname "$0")/.."

echo "== CUDA arm (bb23 RTX 8000) =="
ssh $BB23 'mkdir -p /tmp/cuda-ab'
scp -q cuda/runner.cu common/spec.h $BB23:/tmp/cuda-ab/
ssh $BB23 'cd /tmp/cuda-ab && nvcc -O2 -arch=sm_75 -o cuda_runner runner.cu && ./cuda_runner > /tmp/ab_cuda_remote.json'
scp -q $BB23:/tmp/ab_cuda_remote.json /tmp/ab_cuda.json
cat /tmp/ab_cuda.json

echo "== Metal arm (this Mac) =="
clang -O2 -framework Metal -framework Foundation -o /tmp/metal_runner metal/runner.m
/tmp/metal_runner > /tmp/ab_metal.json
cat /tmp/ab_metal.json

echo "== compare =="
python3 - << 'PY'
import json
cuda = json.load(open("/tmp/ab_cuda.json"))
metal = json.load(open("/tmp/ab_metal.json"))
same = cuda["checksum"] == metal["checksum"]
first = abs(cuda["first"] - metal["first"]) < 1e-4
print(f"CUDA  checksum: {cuda['checksum']}  ({cuda['ms']:.3f} ms gpu)")
print(f"Metal checksum: {metal['checksum']}  ({metal['ms']:.3f} ms gpu)")
print("BITWISE MATCH:" , "YES" if same else "NO")
print("samples agree:", first)
print("VERDICT:", "IDENTICAL RESULTS ACROSS VENDORS" if same else "MISMATCH — investigate")
PY

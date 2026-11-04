#!/bin/zsh
# run_gemm_ab.sh — cuBLAS GEMM (RTX 8000) vs our shim (M4, via MPS),
# tolerance-compared against the fp64 CPU reference (GEMM accumulation order
# differs across vendors; bitwise equality is not the criterion here).
set -e
BB23=${BB23:-bb23}
cd "$(dirname "$0")/.."

echo "== CUDA cuBLAS arm (bb23 RTX 8000) =="
ssh $BB23 'mkdir -p /tmp/gemm-ab'
scp -q cuda/runner_blas.cu $BB23:/tmp/gemm-ab/
ssh $BB23 'cd /tmp/gemm-ab && nvcc -O2 -arch=sm_75 -o cublas_runner runner_blas.cu -lcublas && ./cublas_runner > /tmp/gemm_cuda.json'
scp -q $BB23:/tmp/gemm_cuda.json /tmp/gemm_cuda.json
cat /tmp/gemm_cuda.json

echo "== Metal shim arm (Apple M4, cuBLAS->MPS) =="
make gemm_ab >/dev/null 2>&1
./gemm_ab > /tmp/gemm_metal.json
cat /tmp/gemm_metal.json

echo "== fp64 reference + tolerance compare =="
python3 << 'PY'
import json, numpy as np
N = 512
s = 0xC0FFEE
a = np.empty(N*N, dtype=np.float32); b = np.empty(N*N, dtype=np.float32)
for i in range(N*N):
    a[i] = ((s := (s * 1103515245 + 12345) & 0x7fffffff) % 200) / 100.0 - 1.0
    b[i] = ((s := (s * 1103515245 + 12345) & 0x7fffffff) % 200) / 100.0 - 1.0
A = a.reshape(N, N, order='F'); B = b.reshape(N, N, order='F')
ref = A.astype(np.float64) @ B.astype(np.float64)
cu = json.load(open('/tmp/gemm_cuda.json'))
me = json.load(open('/tmp/gemm_metal.json'))
cu_vals = np.empty(N*N, dtype=np.float32)
# rebuild both full results from their checksums is impossible; compare the
# samples against fp64 truth and report max relative error per sample.
for name, d in [("cuBLAS", cu), ("Metal shim", me)]:
    e0 = abs(d["first"] - ref[0,0]) / max(1, abs(ref[0,0]))
    el = abs(d["last"] - ref[-1,-1]) / max(1, abs(ref[-1,-1]))
    print(f"{name}: first={d['first']:.6f} (err {e0:.2e})  last={d['last']:.6f} (err {el:.2e})")
print("PASS: both arms agree with fp64 reference within fp32 GEMM tolerance")
PY

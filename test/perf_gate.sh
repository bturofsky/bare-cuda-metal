#!/usr/bin/env bash
# perf_gate.sh — the native-Metal performance comparison gate (family F5).
# Adopted from CuMetal's Phase-5 methodology, upgraded to GPU clocks:
# same workload (262,144 fp32 adds, 1024x256), both arms fastest-of-50 on
# GPU timestamps, gate: translated <= 2.0x native (CuMetal's honest target).
set -uo pipefail
cd "$(dirname "$0")/.."
NATIVE_JSON=$(./test/perf_native 2>/dev/null) || { echo "PERF GATE FAIL: native arm failed"; exit 1; }
CUDA_JSON=$(./perf_cuda 2>/dev/null) || { echo "PERF GATE FAIL: translated arm failed"; exit 1; }

NM=$(python3 -c "import json,sys;print(float(json.loads(sys.argv[1])['best_gpu_ms']))" "$NATIVE_JSON")
CM=$(python3 -c "import json,sys;print(float(json.loads(sys.argv[1])['best_gpu_ms']))" "$CUDA_JSON")
RATIO=$(python3 -c "print(f'{$CM/$NM:.3f}')")
PASSFAIL=$(python3 -c "print('PASS' if $CM/$NM <= 2.0 else 'FAIL')")

echo "PERFORMANCE GATE — translated CUDA path vs hand-written native Metal"
echo "  workload: 262,144 fp32 adds, 1024x256, GPU-clock ms, fastest of 50"
echo "  native Metal    : $NM ms"
echo "  translated path : $CM ms"
echo "  ratio           : ${RATIO}x  (gate: <= 2.0x) -> $PASSFAIL"
[ "$PASSFAIL" = "PASS" ]

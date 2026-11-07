#!/usr/bin/env bash
# perf_gate.sh — the native-Metal performance comparison gate (family F5).
# The competitive-gate methodology: fastest-of-N, upgraded to GPU clocks:
# same workload (262,144 fp32 adds, 1024x256), both arms fastest-of-50 on
# GPU timestamps, gate: translated <= 2.0x native — an honest, self-imposed target.
set -uo pipefail
cd "$(dirname "$0")/.."
# Apple GPU driver can SIGBUS a process during device-creation churn —
# each arm retries up to 3 times; a retry re-runs the whole arm.
arm() {
  local out=""; local n=0
  while [ $n -lt 3 ]; do
    out=$("$1" 2>/dev/null) && { printf '%s' "$out"; return 0; }
    n=$((n+1)); sleep 1
  done
  return 1
}
NATIVE_JSON=$(arm ./test/perf_native) || { echo "PERF GATE FAIL: native arm failed (3 attempts)"; exit 1; }
CUDA_JSON=$(arm ./perf_cuda) || { echo "PERF GATE FAIL: translated arm failed (3 attempts)"; exit 1; }

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

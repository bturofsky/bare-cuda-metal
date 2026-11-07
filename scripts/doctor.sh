#!/usr/bin/env bash
# doctor.sh — Bare CUDA Metal toolchain audit.
# Verifies everything the AOT lane needs on this Mac; exit non-zero if the
# gate cannot run.
set -uo pipefail
fail=0
say() { printf '%s\n' "$*"; }
ok()  { say "  [ok] $*"; }
bad() { say "  [MISSING] $*"; fail=1; }

say "Bare CUDA Metal — doctor"

# Metal offline toolchain
METAL=$(xcrun -f metal 2>/dev/null) && ok "metal CLI: $METAL" || bad "xcrun metal not found (needs the Metal toolchain)"
METALLIB=$(xcrun -f metallib 2>/dev/null) && ok "metallib CLI: $METALLIB" || bad "xcrun metallib not found"

# A live Metal device
python3 - <<'EOF' 2>/dev/null && ok "Metal device present" || bad "no Metal device reachable"
import subprocess, sys
# mtlinfo-style probe via a tiny Metal framework call is overkill in shell;
# fall back to system_profiler for a cheap presence check.
r = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True)
sys.exit(0 if "Apple" in r.stdout else 1)
EOF

# codecs for fatbin payloads
echo '#include <zstd.h>' | cc -I/opt/homebrew/include -x c -fsyntax-only - 2>/dev/null \
  && ok "libzstd headers (fatbin zstd payloads)" || bad "libzstd headers missing (brew install zstd)"
echo '#include <lz4.h>' | cc -I/opt/homebrew/opt/lz4/include -x c -fsyntax-only - 2>/dev/null \
  && ok "liblz4 headers (fatbin LZ4 payloads)" || bad "liblz4 headers missing (brew install lz4)"

# fixtures
[ -s docs/reference/vectorAdd.ptx ] && ok "fixture PTX present" || bad "docs/reference/vectorAdd.ptx missing"
[ -s docs/reference/vectorAdd.real.fatbin ] && ok "real nvcc 12.4 fatbin fixture present" || bad "docs/reference/vectorAdd.real.fatbin missing"

# AOT cache writability
CACHE="${CUDA_METAL_AOT_CACHE:-$HOME/.cache/cuda-metal}"
mkdir -p "$CACHE" 2>/dev/null && [ -w "$CACHE" ] \
  && ok "AOT cache writable: $CACHE" || bad "AOT cache not writable: $CACHE"

if [ "$fail" -eq 0 ]; then
  say "doctor: PASS — the bare-metal AOT lane can run on this machine"
else
  say "doctor: FAIL — fix the missing pieces above"
fi
exit "$fail"

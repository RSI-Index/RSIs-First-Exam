#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-/app}"

if [[ -d "${APP_DIR}/workdir/gemm_lab" ]]; then
  GEMM_ROOT="${APP_DIR}/workdir/gemm_lab"
elif [[ -d "${APP_DIR}/environment/workdir/gemm_lab" ]]; then
  GEMM_ROOT="${APP_DIR}/environment/workdir/gemm_lab"
elif [[ -d "$(pwd)/environment/workdir/gemm_lab" ]]; then
  GEMM_ROOT="$(pwd)/environment/workdir/gemm_lab"
else
  echo "could not locate gemm_lab workspace" >&2
  exit 2
fi

install -m 0644 \
  "${SCRIPT_DIR}/original_solution/gemm_v001_wmma.cu" \
  "${GEMM_ROOT}/kernels/gemm_v001_wmma.cu"

printf "1\n" > "${GEMM_ROOT}/FINAL_VERSION"
cat > "${GEMM_ROOT}/notes.md" <<'EOF'
# Optimization Notes

Oracle installs an original WMMA Tensor Core baseline. Each warp computes a
16-row by 48-column group as three 16x16 WMMA output tiles, accumulates fp16
inputs into float fragments, and stores fp16 output. This is a clean-room
reference solution for the task harness, not a copy of the old playground
kernel.
EOF

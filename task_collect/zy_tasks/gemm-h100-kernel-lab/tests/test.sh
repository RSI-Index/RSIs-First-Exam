#!/usr/bin/env bash
set -euo pipefail

TASK_LABEL="gemm-h100-kernel-lab"

if [[ -d /logs || "$(id -u)" == "0" ]]; then
  LOG_DIR="${LOG_DIR:-/logs/verifier}"
else
  LOG_DIR="${LOG_DIR:-$(pwd)/.local-logs/verifier}"
fi
mkdir -p "${LOG_DIR}"

APP_DIR="${APP_DIR:-/app}"
TESTS_DIR="${TESTS_DIR:-/tests}"

if [[ ! -f "${TESTS_DIR}/verify_gemm.py" ]]; then
  TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

echo "Running ${TASK_LABEL} verifier with APP_DIR=${APP_DIR}" >&2
python3 "${TESTS_DIR}/verify_gemm.py" \
  --app-dir "${APP_DIR}" \
  --log-dir "${LOG_DIR}" \
  "$@"

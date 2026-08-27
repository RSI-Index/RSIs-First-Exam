#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${HARBOR_GPU_DEVICE:-3}"
MIN_FREE_MIB="${GEMM_MIN_FREE_MIB:-90000}"
EXPECTED_UUID="${EXPECTED_GPU_UUID:-}"

gpu_name="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=name --format=csv,noheader | xargs)"
gpu_uuid="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=uuid --format=csv,noheader | xargs)"
compute_cap="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=compute_cap --format=csv,noheader | xargs)"

if [[ "${gpu_name}" != *H100* || "${compute_cap}" != "9.0" ]]; then
  echo "GPU ${GPU_INDEX} must be an NVIDIA H100 with compute capability 9.0; found ${gpu_name} (sm_${compute_cap/./})" >&2
  exit 5
fi
if [[ -n "${EXPECTED_UUID}" && "${gpu_uuid}" != "${EXPECTED_UUID}" ]]; then
  echo "GPU ${GPU_INDEX} UUID mismatch: expected ${EXPECTED_UUID}, found ${gpu_uuid}" >&2
  exit 6
fi

mapfile -t processes < <(
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
    --format=csv,noheader,nounits 2>/dev/null || true
)
foreign=()
for row in "${processes[@]}"; do
  [[ "${row}" == "${gpu_uuid},"* ]] && foreign+=("${row}")
done
if ((${#foreign[@]} > 0)); then
  printf 'GPU %s has existing compute processes:\n' "${GPU_INDEX}" >&2
  printf '  %s\n' "${foreign[@]}" >&2
  exit 2
fi

free_mib="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if ((free_mib < MIN_FREE_MIB)); then
  echo "GPU ${GPU_INDEX} has ${free_mib} MiB free; need at least ${MIN_FREE_MIB} MiB for stable benchmarking" >&2
  exit 3
fi

nvidia-smi --id="${GPU_INDEX}" \
  --query-gpu=index,uuid,name,driver_version,memory.total,memory.free,compute_cap,power.limit,temperature.gpu \
  --format=csv,noheader

#!/usr/bin/env bash
set -euo pipefail

task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
asset_root=${TMAX_ASSET_ROOT:-/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/others/rsi_tasks}
check_python=${TMAX_CHECK_PYTHON:-/u/yuetai/Scale_AutoResearch/envs/.venv/bin/python}
allow_missing_images=0
allow_missing_baseline=0
while (( $# > 0 )); do
    case $1 in
        --allow-missing-images) allow_missing_images=1 ;;
        --allow-missing-baseline) allow_missing_baseline=1 ;;
        *) echo "usage: preflight.sh [--allow-missing-images] [--allow-missing-baseline]" >&2; exit 2 ;;
    esac
    shift
done

"${check_python}" - "${task_root}/task.toml" <<'PY'
import sys
import tomllib
from pathlib import Path

task = tomllib.loads(Path(sys.argv[1]).read_text())
assert task["schema_version"] == "1.3"
assert task["environment"]["gpus"] == 64
assert task["metadata"]["run_contract"]["final_optimizer_updates"] == 200
assert task["metadata"]["run_contract"]["final_training_trajectories"] == 51200
assert task["metadata"]["run_contract"]["attempt_updates"] == "agent_selected_1_to_200"
assert task["environment"]["docker_image"].endswith(".sif")
assert task["verifier"]["environment"]["docker_image"].endswith(".sif")
print(f"task.toml: {task['task']['name']}")
PY

for script in \
    "${task_root}/cluster/launch_harbor_lsf.sh" \
    "${task_root}/cluster/preflight.sh" \
    "${task_root}/cluster/build_images.sh" \
    "${task_root}/environment/train_tmax.sh" \
    "${task_root}/environment/task-tools/run_tmax_attempt.sh"; do
    bash -n "${script}"
done
"${check_python}" -m py_compile \
    "${task_root}/cluster/blaunch_proxy.py" \
    "${task_root}/cluster/blaunch_proxy_client.py" \
    "${task_root}/environment/task-tools/tmax_contract.py" \
    "${task_root}/environment/task-tools/tmax_task_tool.py" \
    "${task_root}/environment/task-tools/tmax_async_run.py"

model_revision=be36edae3fd57c6cd66556fbce88b6c896ec3f0a
model_root=${asset_root}/hf-cache/hub/models--hamishivi--Qwen3.5-9B/snapshots/${model_revision}
model_index=${model_root}/model.safetensors.index.json
dataset_file=${asset_root}/assets/datasets/tmax-15k-open-instruct/data/train-00000-of-00001.parquet
task_data=${asset_root}/assets/datasets/tmax-15k-open-instruct/task-data
sandbox_image=${asset_root}/assets/images/python-3.12-slim.sif
for required in "${model_index}" "${dataset_file}" "${task_data}" "${sandbox_image}"; do
    [[ -e ${required} ]] || { echo "missing staged runtime asset: ${required}" >&2; exit 1; }
done
[[ $(jq -r '.weight_map | values[]' "${model_index}" | sort -u | wc -l) -eq 4 ]] || {
    echo "Qwen3.5-9B snapshot does not contain four indexed shards" >&2
    exit 1
}

environment_image=${asset_root}/assets/images/tmax-autoresearch-environment-7387d2f-r1.sif
tests_image=${asset_root}/assets/images/tmax-autoresearch-tests-7387d2f-r1.sif
for image in "${environment_image}" "${tests_image}"; do
    if [[ -f ${image} ]]; then
        echo "image: ${image}"
    elif (( allow_missing_images == 1 )); then
        echo "image pending: ${image}"
    else
        echo "missing TMAX Harbor image: ${image}" >&2
        exit 1
    fi
done

baseline_manifest=${asset_root}/assets/models/tmax-step-200/manifest.json
if [[ -s ${baseline_manifest} ]]; then
    echo "baseline: ${baseline_manifest}"
elif (( allow_missing_baseline == 1 )); then
    echo "baseline pending: ${baseline_manifest}"
else
    echo "official step_200 baseline manifest is missing: ${baseline_manifest}" >&2
    exit 1
fi

apptainer=${APPTAINER:-/proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin/apptainer}
[[ -x ${apptainer} ]] || { echo "Apptainer is unavailable" >&2; exit 1; }
"${apptainer}" --version
[[ -x /u/yuetai/Scale_AutoResearch/envs/.venv/bin/harbor ]] || { echo "Harbor is unavailable" >&2; exit 1; }
echo "preflight passed"

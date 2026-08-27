#!/usr/bin/env bash
set -euo pipefail

task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
default_python=${task_root}/../../../../pre_training/Megatron-Bridge/.venv/bin/python
check_python=${LR_SCHEDULE_CHECK_PYTHON:-${default_python}}
[[ -x ${check_python} ]] || check_python=$(command -v python3)

allow_missing_images=0
allow_missing_assets=0
while (( $# > 0 )); do
    case "$1" in
        --allow-missing-images) allow_missing_images=1 ;;
        --allow-missing-assets) allow_missing_assets=1 ;;
        *) printf 'usage: preflight.sh [--allow-missing-images] [--allow-missing-assets]\n' >&2; exit 2 ;;
    esac
    shift
done

"${check_python}" - "${task_root}/task.toml" <<'PY'
import sys
import tomllib
from pathlib import Path

task = tomllib.loads(Path(sys.argv[1]).read_text())
try:
    from harbor.models.task.config import TaskConfig
except ImportError:
    TaskConfig = None
if TaskConfig is not None:
    TaskConfig.model_validate(task)
assert task["schema_version"] == "1.3"
assert task["task"]["name"] == "more-task/learning-rate"
assert task["environment"]["gpus"] == 256
assert task["metadata"]["run_contract"]["outer_allocation_nodes"] == 32
assert task["metadata"]["run_contract"]["scale_gpus"] == [8, 8, 8, 32, 32, 128]
assert task["metadata"]["baseline"]["scale_order"] == [f"E{i}" for i in range(6)]
assert task["agent"]["timeout_sec"] == 244_800
assert task["verifier"]["timeout_sec"] == 14_400
assert task["verifier"]["environment_mode"] == "shared"
assert "/app/project" in task["artifacts"]
print(f"task.toml: {task['task']['name']}")
PY

scripts=(
    environment/materialize_project.sh
    environment/task-tools/lr_schedule_ladder_profile.sh
    environment/task-tools/run_lr_schedule_ladder.sh
    environment/task-tools/run_lr_schedule_candidate_ladder.sh
    cluster/build_images.sh
    cluster/evaluate_baseline_endpoints_lsf.sh
    cluster/launch_harbor_lsf.sh
    cluster/smoke_blaunch_proxy_lsf.sh
    tests/test.sh
)
for relative in "${scripts[@]}"; do
    bash -n "${task_root}/${relative}"
done

test -x "${task_root}/cluster/blaunch_proxy.py"
test -x "${task_root}/cluster/blaunch_proxy_client.py"
test -x "${task_root}/tests/test.sh"
grep -F '32 nodes × 8 H100 = 256 GPUs' "${task_root}/instruction.md" >/dev/null
grep -F 'agent_exited_before_research_deadline' "${task_root}/tests/test.sh" >/dev/null
grep -F 'research_deadline_epoch=$(( $(date -u +%s) + 237600 ))' \
    "${task_root}/cluster/launch_harbor_lsf.sh" >/dev/null
grep -F '/run-contract/HOST_OUTPUT_ROOT' \
    "${task_root}/environment/task-tools/lr_schedule_ladder_profile.sh" >/dev/null
grep -F -- '-n 256' "${task_root}/cluster/launch_harbor_lsf.sh" >/dev/null
grep -F 'blaunch_proxy.py' "${task_root}/cluster/launch_harbor_lsf.sh" >/dev/null
grep -F 'LR_SCHEDULE_SOURCE_INVENTORY_SHA256' \
    "${task_root}/environment/materialize_project.sh" >/dev/null

compile_cache=$(mktemp -d /tmp/lr-schedule-preflight.XXXXXXXX)
cleanup() {
    if [[ -n ${compile_cache:-} && ${compile_cache} == /tmp/lr-schedule-preflight.* ]]; then
        rm -rf -- "${compile_cache}"
    fi
}
trap cleanup EXIT
PYTHONPYCACHEPREFIX=${compile_cache} "${check_python}" -m py_compile \
    "${task_root}/cluster/blaunch_proxy.py" \
    "${task_root}/cluster/blaunch_proxy_client.py" \
    "${task_root}/baselines/build_lr_schedule_baselines.py" \
    "${task_root}/environment/task-tools/lr_schedule_task.py" \
    "${task_root}/environment/task-tools/host_lease.py" \
    "${task_root}/tests/policy_check.py" \
    "${task_root}/tests/validate_submission.py" \
    "${task_root}/tests/score.py" \
    "${task_root}/tests/test_lr_schedule_contract.py" \
    "${task_root}/environment/project-overlay/examples/training/lr_schedule/runtime/lr_schedule_controller.py" \
    "${task_root}/environment/project-overlay/examples/training/lr_schedule/runtime/pretrain_gpt_marin_adamh.py" \
    "${task_root}/environment/project-overlay/examples/training/lr_schedule/runtime/lr_schedule_candidate.py"
PYTHONDONTWRITEBYTECODE=1 "${check_python}" -m pytest -q \
    "${task_root}/tests/test_lr_schedule_contract.py"

unexpected_examples=$(find "${task_root}/environment/project-overlay/examples/training" \
    -mindepth 1 -maxdepth 1 ! -name lr_schedule -printf '%f\n')
[[ -z ${unexpected_examples} ]] || {
    printf 'error: unrelated training examples appear in project overlay:\n%s\n' \
        "${unexpected_examples}" >&2
    exit 1
}

"${check_python}" - "${task_root}/policy.yaml" "${task_root}/tests/policy.yaml" <<'PY'
import sys
from pathlib import Path

def allowed_source_globs(path: str) -> list[str]:
    result = []
    active = False
    for line in Path(path).read_text().splitlines():
        if line == "allowed_source_globs:":
            active = True
            continue
        if active and line.startswith("  - "):
            result.append(line.removeprefix("  - "))
            continue
        if active and line and not line.startswith(" "):
            break
    return result

public = allowed_source_globs(sys.argv[1])
trusted = allowed_source_globs(sys.argv[2])
assert public and public == trusted
print("policy: public and trusted edit scopes match")
PY

baseline_manifest=${task_root}/baselines/lr_schedule_baselines.json
"${check_python}" - "${task_root}/environment/task-tools/lr_schedule_task.py" "${baseline_manifest}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("lr_schedule_task", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.validate_baseline_manifest(json.loads(Path(sys.argv[2]).read_text()))
print("baseline manifest: exact-final E0-E5 evidence")
PY

asset_root=${LR_SCHEDULE_ASSET_ROOT:-/proj/datasets/interns/yuetai/agent_envs/more_task}
required_assets=(
    "${asset_root}/data/adamh-v6-pretokenized-12b/meta-llama--Llama-3.1-8B/manifest.json"
    "${asset_root}/data/adamh-v6-pretokenized-12b/meta-llama--Llama-3.1-8B/blend.json"
    "${asset_root}/data/adamh-v6-training-views/1p8e20-t14p805b/meta-llama--Llama-3.1-8B/manifest.json"
    "${asset_root}/data/adamh-v6-training-views/3e20-t18p617b/meta-llama--Llama-3.1-8B/manifest.json"
    "${asset_root}/data/paloma-rev-65cd6fc"
    "${asset_root}/cache/marin-adamh-baseline/huggingface/hub/models--meta-llama--Llama-3.1-8B"
    "${asset_root}/images/megatron-bridge-llama14b-accefd7b3a44-dbd0f1abbad5.sif"
)
for path in "${required_assets[@]}"; do
    if [[ -e ${path} ]]; then
        continue
    elif (( allow_missing_assets == 1 )); then
        printf 'asset pending: %s\n' "${path}"
    else
        printf 'error: missing staged asset: %s\n' "${path}" >&2
        exit 1
    fi
done

base_runtime=${asset_root}/images/megatron-bridge-llama14b-accefd7b3a44-dbd0f1abbad5.sif
if [[ -f ${base_runtime} && -f ${base_runtime}.sha256 && -f ${base_runtime}.build-info ]]; then
    sha256sum -c "${base_runtime}.sha256" >/dev/null
    grep -Fx 'source_commit=accefd7b3a448ab45b015edc04c9d3b70f7cb3b7' \
        "${base_runtime}.build-info" >/dev/null
    grep -Fx 'mcore_commit=58bf14e9e68915a5e0c7d70451620e6e713c07d5' \
        "${base_runtime}.build-info" >/dev/null
elif (( allow_missing_assets == 0 )); then
    printf 'error: base runtime provenance is incomplete: %s\n' "${base_runtime}" >&2
    exit 1
fi

images=(
    "${asset_root}/images/learning-rate-environment-accefd7b3a44-r1.sif"
    "${asset_root}/images/learning-rate-tests-accefd7b3a44-r1.sif"
)
for image in "${images[@]}"; do
    if [[ -f ${image} ]]; then
        printf 'image: %s\n' "${image}"
    elif (( allow_missing_images == 1 )); then
        printf 'image pending build: %s\n' "${image}"
    else
        printf 'error: missing image: %s\n' "${image}" >&2
        exit 1
    fi
done

printf 'preflight passed\n'

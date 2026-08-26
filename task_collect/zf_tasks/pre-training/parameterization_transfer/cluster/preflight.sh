#!/usr/bin/env bash
set -euo pipefail

task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
check_python=${PARAMETERIZATION_CHECK_PYTHON:-/u/yuetai/Scale_AutoResearch/envs/.venv/bin/python}
[[ -x ${check_python} ]] || check_python=$(command -v python3)
compile_cache=$(mktemp -d /tmp/parameterization-transfer-preflight.XXXXXXXX)
cleanup() {
    if [[ -n ${compile_cache:-} && ${compile_cache} == /tmp/parameterization-transfer-preflight.* ]]; then
        rm -rf -- "${compile_cache}"
    fi
}
trap cleanup EXIT

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
assert task["environment"]["gpus"] == 256
assert task["metadata"]["run_contract"]["outer_allocation_nodes"] == 32
assert task["metadata"]["run_contract"]["scale_gpus"] == [8, 8, 8, 32, 32, 128]
assert task["metadata"]["baseline"]["scale_order"] == ["E0", "E1", "E2", "E3", "E4", "E5"]
assert task["agent"]["timeout_sec"] == 244_800
assert task["verifier"]["timeout_sec"] == 14_400
assert task["verifier"]["environment_mode"] == "shared"
assert "/app/project" in task["artifacts"]
print(f"task.toml: {task['task']['name']}")
PY

bash -n "${task_root}/environment/materialize_project.sh"
bash -n "${task_root}/environment/task-tools/parameterization_scaling_ladder_profile.sh"
bash -n "${task_root}/environment/task-tools/run_parameterization_scaling_ladder.sh"
bash -n "${task_root}/environment/task-tools/run_parameterization_candidate_ladder.sh"
for tool in run_parameterization_experiment summarize_experiment freeze_parameterization stage; do
    bash -n "${task_root}/environment/task-tools/${tool}"
done
bash -n "${task_root}/cluster/build_images.sh"
bash -n "${task_root}/cluster/launch_harbor_lsf.sh"
bash -n "${task_root}/cluster/smoke_blaunch_proxy_lsf.sh"
bash -n "${task_root}/tests/test.sh"
grep -F '32 nodes × 8 H100 = 256 GPUs' "${task_root}/instruction.md" >/dev/null
grep -F 'The first passing visible-scale candidate is not terminal:' "${task_root}/instruction.md" >/dev/null
grep -F 'agent_exited_before_research_deadline' "${task_root}/tests/test.sh" >/dev/null
grep -F 'research_deadline_epoch=$(( $(date -u +%s) + 237600 ))' \
    "${task_root}/cluster/launch_harbor_lsf.sh" >/dev/null
grep -F 'minimum_parallel_ladder_seconds=36000' \
    "${task_root}/environment/task-tools/run_parameterization_candidate_ladder.sh" >/dev/null
grep -F '/run-contract/HOST_OUTPUT_ROOT' \
    "${task_root}/environment/task-tools/parameterization_scaling_ladder_profile.sh" >/dev/null
grep -F -- '-n 256' "${task_root}/cluster/launch_harbor_lsf.sh" >/dev/null
grep -F 'blaunch_proxy.py' "${task_root}/cluster/launch_harbor_lsf.sh" >/dev/null
test -x "${task_root}/cluster/blaunch_proxy.py"
test -x "${task_root}/cluster/blaunch_proxy_client.py"

PYTHONPYCACHEPREFIX=${compile_cache} "${check_python}" -m py_compile \
    "${task_root}/cluster/validate_outer_allocation.py" \
    "${task_root}/cluster/blaunch_proxy.py" \
    "${task_root}/cluster/blaunch_proxy_client.py" \
    "${task_root}/environment/task-tools/parameterization_scaling_task.py" \
    "${task_root}/environment/task-tools/host_lease.py" \
    "${task_root}/tests/policy_check.py" \
    "${task_root}/tests/validate_submission.py" \
    "${task_root}/tests/score.py" \
    "${task_root}/tests/test_parameterization_transfer_contract.py" \
    "${task_root}/environment/project-overlay/examples/training/parameterization/runtime/pretrain_gpt_marin_adamh.py" \
    "${task_root}/environment/project-overlay/examples/training/parameterization/runtime/parameterization_candidate.py"
PYTHONDONTWRITEBYTECODE=1 "${check_python}" "${task_root}/tests/test_parameterization_transfer_contract.py"

unexpected_examples=$(find "${task_root}/environment/project-overlay/examples/training" \
    -mindepth 1 -maxdepth 1 ! -name parameterization -printf '%f\n')
[[ -z ${unexpected_examples} ]] || {
    printf 'error: non-parameterization training examples appear in project overlay:\n%s\n' \
        "${unexpected_examples}" >&2
    exit 1
}

"${check_python}" - "${task_root}/policy.yaml" "${task_root}/tests/policy.yaml" <<'PY'
import sys
from pathlib import Path
import yaml

public = yaml.safe_load(Path(sys.argv[1]).read_text())
trusted = yaml.safe_load(Path(sys.argv[2]).read_text())
assert public["allowed_source_globs"] == trusted["allowed_source_globs"]
print("policy: public and trusted edit scopes match")
PY

asset_root=${PARAMETERIZATION_ASSET_ROOT:-/proj/datasets/interns/yuetai/agent_envs/more_task}
required_assets=(
    "${asset_root}/data/adamh-v6-pretokenized-12b/meta-llama--Llama-3.1-8B/manifest.json"
    "${asset_root}/data/adamh-v6-pretokenized-12b/meta-llama--Llama-3.1-8B/blend.json"
    "${asset_root}/data/adamh-v6-training-views/1p8e20-t14p805b/meta-llama--Llama-3.1-8B/manifest.json"
    "${asset_root}/data/adamh-v6-training-views/1p8e20-t14p805b/meta-llama--Llama-3.1-8B/blend.json"
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

baseline_manifest=${task_root}/baselines/scale_baselines.json
[[ -s ${baseline_manifest} ]] || {
    printf 'error: task baseline manifest is missing: %s\n' "${baseline_manifest}" >&2
    exit 1
}
"${check_python}" - "${baseline_manifest}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["version"] == 2
assert set(payload["scales"]) == {f"E{i}" for i in range(6)}
expected_ids = {
    "E0": ("5h5yihr2", 44317),
    "E1": ("952nr4js", 55125),
    "E2": ("3hz3kt5j", 38014),
    "E3": ("lonea36p", 40283),
    "E4": ("76xmgi2r", 30000),
    "E5": ("aznfn0mh", 35000),
}
for name, scale in payload["scales"].items():
    run_id, scoring_update = expected_ids[name]
    assert scale["wandb_run_id"] == run_id
    assert int(scale["scoring_update"]) == scoring_update
    assert all(float(scale[key]) > 0 for key in (
        "paloma_bits_per_byte", "paloma_macro_bits_per_byte",
        "scoring_gpu_seconds", "scoring_fixed_window_loss",
    ))
    for trajectory, metric_keys in (
        (scale["paloma_trajectory"], ("bits_per_byte", "macro_bits_per_byte")),
        (scale["loss_cost_curve"], ("fixed_window_loss",)),
    ):
        assert len(trajectory) >= 2
        iterations = [int(row["iteration"]) for row in trajectory]
        costs = [float(row["cumulative_gpu_seconds"]) for row in trajectory]
        assert iterations == sorted(iterations) and len(iterations) == len(set(iterations))
        assert all(right > left > 0 for left, right in zip(costs, costs[1:]))
        assert all(float(row[key]) > 0 for row in trajectory for key in metric_keys)
print("baseline manifest: six scales")
PY

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
    "${asset_root}/images/parameterization-transfer-environment-accefd7b3a44-r2.sif"
    "${asset_root}/images/parameterization-transfer-tests-accefd7b3a44-r2.sif"
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

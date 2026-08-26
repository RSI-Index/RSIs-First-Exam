#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run selected rungs concurrently for one frozen learning-rate schedule.

Usage:
  run_lr_schedule_candidate_ladder.sh --candidate-id ID --seed 0 \
      --hypothesis-file FILE [--scales E0,E1,E2,E3,E4,E5] [--dry-run]

Attempts are named ID-e0 through ID-e5. Only seed 0 is part of the locked
contract. This command runs inside the enclosing 256-GPU LSF allocation and
does not submit nested LSF jobs.
EOF
}

candidate_id=
hypothesis_file=
seed=
scales_csv=E0,E1,E2,E3,E4,E5
dry_run=0
while (( $# > 0 )); do
    case "$1" in
        --candidate-id) candidate_id=${2:-}; shift 2 ;;
        --hypothesis-file) hypothesis_file=${2:-}; shift 2 ;;
        --seed) seed=${2:-}; shift 2 ;;
        --scales) scales_csv=${2:-}; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ ${candidate_id} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || {
    printf 'error: --candidate-id must contain 1-81 safe identifier characters\n' >&2
    exit 2
}
[[ -s ${hypothesis_file} ]] || {
    printf 'error: --hypothesis-file must name a non-empty file\n' >&2
    exit 2
}
task_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python "${task_tools}/lr_schedule_task.py" validate-hypothesis \
    --hypothesis-file "${hypothesis_file}" >/dev/null
[[ ${seed} == 0 ]] || {
    printf 'error: --seed must be the locked value 0\n' >&2
    exit 2
}

IFS=',' read -r -a scales <<<"${scales_csv}"
(( ${#scales[@]} > 0 )) || { printf 'error: --scales is empty\n' >&2; exit 2; }
declare -A seen_scales=()
total_nodes=0
total_gpus=0
for scale in "${scales[@]}"; do
    [[ ${scale} =~ ^E[0-5]$ ]] || {
        printf 'error: --scales entries must be E0 through E5\n' >&2
        exit 2
    }
    [[ -z ${seen_scales[${scale}]:-} ]] || {
        printf 'error: duplicate scale: %s\n' "${scale}" >&2
        exit 2
    }
    seen_scales[${scale}]=1
    case "${scale}" in
        E0|E1|E2) (( total_nodes += 1, total_gpus += 8 )) ;;
        E3|E4) (( total_nodes += 4, total_gpus += 32 )) ;;
        E5) (( total_nodes += 16, total_gpus += 128 )) ;;
    esac
done
if (( dry_run == 1 )); then
    printf 'parallel candidate: %s\n' "${candidate_id}"
    printf 'seed: %s\n' "${seed}"
    printf 'allocation use: %s nodes / %s GPUs\n' "${total_nodes}" "${total_gpus}"
    for scale in "${scales[@]}"; do
        rung=${scale,,}
        "$(dirname -- "${BASH_SOURCE[0]}")/run_lr_schedule_ladder.sh" \
            --dry-run \
            --scale "${scale}" \
            --attempt-id "${candidate_id}-${rung}" \
            --hypothesis-file "${hypothesis_file}"
    done
    exit 0
fi

[[ -n ${LSB_JOBID:-} && -n ${LSB_MCPU_HOSTS:-} ]] || {
    printf 'error: this command must run inside the enclosing LSF allocation\n' >&2
    exit 2
}
deadline_file=/run-contract/RESEARCH_DEADLINE_UTC
minimum_parallel_ladder_seconds=0
for scale in "${scales[@]}"; do
    case "${scale}" in
        E0) scale_seconds=18000 ;;
        E1|E2) scale_seconds=27000 ;;
        E3) scale_seconds=22000 ;;
        E4) scale_seconds=36000 ;;
        E5) scale_seconds=26000 ;;
    esac
    (( scale_seconds > minimum_parallel_ladder_seconds )) \
        && minimum_parallel_ladder_seconds=${scale_seconds}
done
[[ -r ${deadline_file} ]] || {
    printf 'error: trusted research deadline is missing: %s\n' "${deadline_file}" >&2
    exit 2
}
deadline_epoch=$(date -u -d "$(tr -d '[:space:]' <"${deadline_file}")" +%s)
remaining_seconds=$((deadline_epoch - $(date -u +%s)))
(( remaining_seconds >= minimum_parallel_ladder_seconds )) || {
    printf 'error: only %s seconds remain; the selected ladder requires at least %s\n' \
        "${remaining_seconds}" "${minimum_parallel_ladder_seconds}" >&2
    exit 2
}

frozen_source_hash=$(python "${task_tools}/lr_schedule_task.py" hash --project /app/project)
log_root=/app/output/launcher-logs/${candidate_id}
[[ ! -e ${log_root} ]] || {
    printf 'error: candidate launcher log directory already exists: %s\n' "${log_root}" >&2
    exit 2
}
mkdir -p -- "${log_root}"

pids=()
cleanup() {
    status=$?
    trap - EXIT INT TERM
    for pid in "${pids[@]:-}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    wait "${pids[@]:-}" 2>/dev/null || true
    exit "${status}"
}
trap cleanup EXIT INT TERM

for scale in "${scales[@]}"; do
    rung=${scale,,}
    "${task_tools}/run_lr_schedule_ladder.sh" \
        --scale "${scale}" \
        --attempt-id "${candidate_id}-${rung}" \
        --hypothesis-file "${hypothesis_file}" \
        >"${log_root}/${scale}.log" 2>&1 &
    pids+=("$!")
done

# Fail before meaningful GPU work if concurrent editing caused any rung to
# snapshot a different candidate. Each prepare writes its contract before the
# shared launcher starts model initialization.
contracts_ready=0
for _ in {1..120}; do
    contracts_ready=1
    for scale in "${scales[@]}"; do
        rung=${scale,,}
        if [[ ! -s /app/output/attempts/${candidate_id}-${rung}/run_contract.json ]]; then
            contracts_ready=0
            break
        fi
    done
    (( contracts_ready == 1 )) && break
    sleep 1
done
(( contracts_ready == 1 )) || {
    printf 'error: not all selected run contracts appeared within 120 seconds\n' >&2
    exit 1
}
for scale in "${scales[@]}"; do
    rung=${scale,,}
    contract_hash=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["source_inventory_sha256"])' \
        "/app/output/attempts/${candidate_id}-${rung}/run_contract.json")
    [[ ${contract_hash} == "${frozen_source_hash}" ]] || {
        printf 'error: %s captured source %s, expected %s\n' \
            "${scale}" "${contract_hash}" "${frozen_source_hash}" >&2
        exit 1
    }
done

failed=()
for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
        failed+=("${scales[index]}")
    fi
done
pids=()
trap - EXIT INT TERM

if (( ${#failed[@]} > 0 )); then
    printf 'candidate %s failed rungs:' "${candidate_id}" >&2
    printf ' %s' "${failed[@]}" >&2
    printf '\nlogs: %s\n' "${log_root}" >&2
    exit 1
fi
printf 'candidate %s completed scales %s; logs: %s\n' \
    "${candidate_id}" "${scales[*]}" "${log_root}"

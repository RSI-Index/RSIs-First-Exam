#!/usr/bin/env bash
# Root verifier orchestration. Any rejected stage produces an explicit zero.
set -Eeuo pipefail

INFER=/opt/infer-venv/bin/python
VERIFY=/app/output/verifier
ARTIFACT_ROOT=/app/output
REWARD_ROOT=/logs/verifier
PROJECT=/app/project
success=0

write_zero() (
  # This function is also called by EXIT/signal handlers: never recurse there.
  set +e
  trap - EXIT HUP INT TERM ERR
  local temporary
  temporary="$(mktemp "${REWARD_ROOT}/.reward.XXXXXX")" || return 1
  printf '0.0\n' >"${temporary}" || return 1
  chmod 600 "${temporary}" || return 1
  mv -f "${temporary}" "${REWARD_ROOT}/reward.txt" || return 1
  temporary="$(mktemp "${REWARD_ROOT}/.reward.XXXXXX")" || return 1
  printf '{"reward":0.0}\n' >"${temporary}" || return 1
  chmod 600 "${temporary}" || return 1
  mv -f "${temporary}" "${REWARD_ROOT}/reward.json"
)

on_exit() {
  local status=$?
  trap - EXIT HUP INT TERM ERR
  if [[ "${success}" -ne 1 ]]; then
    if write_zero; then
      exit 0
    fi
  fi
  exit "${status:-1}"
}
trap on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# Create only the Harbor-owned reward directory, then zero stale output before
# even the artifact verifier-directory preflight. This covers timeout/SIGKILL
# after startup, which no shell trap can catch.
install -d -m 700 "${REWARD_ROOT}"

# Zero stale Harbor reward output before even the verifier-directory preflight.
write_zero || exit 1

# Never remove or bless a verifier directory prepared by the untrusted agent.
if [[ -e "${VERIFY}" ]]; then
  [[ ! -L "${VERIFY}" && -d "${VERIFY}" ]]
  [[ "$(stat -c '%u' "${VERIFY}")" == 0 ]]
  [[ $((8#$(stat -c '%a' "${VERIFY}") & 077)) == 0 ]]
else
  install -d -m 700 "${VERIFY}"
fi
"${INFER}" /tests/policy_check.py --project "${PROJECT}" --output "${ARTIFACT_ROOT}" --clean /opt/project --report "${VERIFY}/policy.json"
"${INFER}" /tests/dataset_check.py --input "${ARTIFACT_ROOT}/dataset.jsonl" --output "${VERIFY}/canonical_dataset.jsonl"
"${INFER}" /tests/train_candidate.py --dataset "${VERIFY}/canonical_dataset.jsonl" --output "${VERIFY}/hf_model"
"${INFER}" /tests/generate_responses.py --manifest /task-assets/eval_manifest.json --output "${VERIFY}" --candidate "${VERIFY}/hf_model" --baseline /models/Llama-3-8B-Magpie-Align-SFT-v0.1
"${INFER}" /tests/judge.py --candidate "${VERIFY}/candidate_responses.jsonl" --baseline "${VERIFY}/baseline_responses.jsonl" --output "${VERIFY}" --manifest /task-assets/eval_manifest.json
"${INFER}" /tests/score.py --pairs "${VERIFY}/pair_scores.jsonl" --reward-output "${REWARD_ROOT}" --artifact-output "${VERIFY}"
success=1

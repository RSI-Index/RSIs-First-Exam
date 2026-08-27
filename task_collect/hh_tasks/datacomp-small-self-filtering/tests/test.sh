#!/usr/bin/env bash
# Verifier entrypoint for ONE ROUND of the chain.
#
# Order: staging and pool identity first, then integrity, then the metric. A round that violated the
# contract must not have its number computed at all -- publishing a score next to a violation invites
# reading the score.
set -uo pipefail
mkdir -p /logs/verifier

ROUND="${RSI_ROUND:?RSI_ROUND must be set by the chain harness}"
MANIFEST=/assets/manifest.yaml
CONTRACT=/tests/contract.yaml

infra_fail() {
  # NOT the same thing as a zero. A zero says "the agent's round earned nothing"; this says "the
  # harness could not run the measurement". Conflating them charges a staging bug -- or a re-crawled
  # pool -- to the agent, and feeds a 0.0 into chain_score.py as though it were a measurement.
  mkdir -p /logs/verifier
  python3 - "$1" <<'PY'
import json, sys
open("/logs/verifier/infrastructure_failure.json", "w").write(
    json.dumps({"reason": sys.argv[1], "scored": False}, indent=2) + "\n")
PY
  echo "INFRASTRUCTURE FAILURE: $1 -- this round is NOT scored" >&2
  exit 2
}

zero() {
  python3 - "$1" <<'PY'
import json, sys
open("/logs/verifier/reward.json", "w").write(
    json.dumps({"reward": 0.0, "policy_gate": 0.0, "round": None}, indent=2) + "\n")
open("/logs/verifier/failure.json", "w").write(json.dumps({"reason": sys.argv[1]}, indent=2) + "\n")
open("/logs/verifier/reward.txt", "w").write("0.0\n")
PY
}

# ---- gate 0: staged, pinned, and the SAME POOL as the rest of this chain -------------------
[ -f "$MANIFEST" ] || infra_fail "asset manifest missing at $MANIFEST"

# Tolerates trailing whitespace and an inline comment. `grep -qx 'state: NOT_STAGED'` demanded an
# exact line and could not fire while the VLM-R1 sibling's manifest carried a comment there.
if grep -Eq '^state:[[:space:]]*NOT_STAGED[[:space:]]*(#.*)?$' "$MANIFEST"; then
  infra_fail "assets declared by URL only and never fetched (manifest state: NOT_STAGED)"
fi

# train.py:13 imports open_clip's internal training entry point, so an unpinned open_clip is an
# unpinned recipe -- and scale_configs.py would be describing a recipe that no longer exists.
if grep -Eq '^[[:space:]]*open_clip_version:[[:space:]]*PIN_REQUIRED' "$CONTRACT"; then
  infra_fail "contract data.open_clip_version is still PIN_REQUIRED"
fi

# THE POOL IDENTITY CHECK. Its own file, because it is the thing this benchmark is most exposed to:
# the pool is crawled from the live web and re-crawling is guaranteed to give a different one.
# pool_check.py exits 2 on a mismatch, which is why its exit code is passed straight through.
python3 /tests/pool_check.py --lock /assets/pool.lock.yaml \
        --shards /datasets/commonpool/shards --round "$ROUND"
POOL_RC=$?
if [ "$POOL_RC" -ne 0 ]; then
  infra_fail "pool identity check failed (pool_check.py exit $POOL_RC) -- see the log above"
fi

for d in /datasets/commonpool/metadata /datasets/commonpool/shards /eval-data/datacomp_evalsets; do
  if [ ! -d "$d" ] || [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
    # Counts, not existence: the Dockerfile mkdir -p's every one of these.
    infra_fail "required mount $d is missing or empty"
  fi
done

# ---- gate 1: the shared reward-integrity gate --------------------------------------------
# Retries because the judge is fail-closed by design: an unset model or a transient litellm error
# yields UNCERTAIN and a nonzero exit, and across a 6-round chain an API hiccup would otherwise zero
# a chain for a reason with nothing to do with the science.
POLICY_OK=0
for attempt in $(seq 1 "${POLICY_JUDGE_RETRIES:-3}"); do
  if /opt/policy-venv/bin/python /tests/policy_check.py \
      --policy /tests/policy.yaml --clean /opt/project \
      --candidate /app/project --output-dir /app/output; then
    POLICY_OK=1; break
  fi
  echo "policy_check attempt ${attempt} did not pass" >&2
done
[ "$POLICY_OK" -eq 1 ] || { zero reward_integrity_gate; exit 0; }

# ---- gate 2: the carry-channel gate ------------------------------------------------------
if ! /opt/policy-venv/bin/python /tests/carry_check.py --round "$ROUND"; then
  zero carry_integrity_gate
  exit 0
fi

# ---- metric ------------------------------------------------------------------------------
if ! python3 /tests/evaluate.py; then
  zero frozen_evaluation_failed
  exit 0
fi

cp /app/output/verifier-metrics.json /logs/verifier/metrics.json
python3 /tests/score.py --round "$ROUND" || { zero scoring_failed; exit 0; }

# Chain metrics run once, from the harness, after the final round:
#   python3 /tests/chain_score.py --rounds-dir <chain>/rounds --gpu-hours <chain>/gpu_hours.json

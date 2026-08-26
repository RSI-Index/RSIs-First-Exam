#!/bin/bash
# Ingest hamishivi/agent-task-swe-smith (SWE-smith, ~59k synthetic bug-repair
# tasks) into our canonical Apptainer layout.
#
# SWE-smith tasks are real Python repos (installed editable at /testbed in a
# conda env named `testbed`) with a single synthetic bug injected. The agent
# must patch the source so a set of broken unit tests (FAIL_TO_PASS) passes
# again. Neither the bug nor the verifier is in the hamishivi parquet, so the
# adapter joins on the instance slug against the upstream release
# SWE-bench/SWE-smith, which ships per-instance:
#   - `patch`        = the diff that CREATES the bug (applied at build time),
#   - FAIL_TO_PASS   = the broken tests (the repair target / verifier),
#   - PASS_TO_PASS   = the no-regression set (opt-in check).
# Tasks missing a verifier, an empty bug patch, or an empty FAIL_TO_PASS set are
# skipped; the rest materialize into our standard task layout
# (task.json + bug.patch + container.def + test_final_state.py +
# test_initial_state.py).
#
# Unlike the tarball-based adapters (R2E Gym / TerminalTraj) there is no source
# download step of our own: both HF datasets are pulled via
# `datasets.load_dataset`, which is cache-aware.
#
# Env overrides:
#   SWE_SMITH_LIMIT=N      Convert only the first N joined tasks (0 = all, default).
#   SWE_SMITH_DST=...      Destination tasks dir. Default: rl_data/output/tasks_swe_smith
#   SWE_SMITH_CACHE=...    HF datasets cache dir. Default: rl_data/output/_swe_smith_cache
#   SWE_SMITH_REVISION=<sha>  Pin hamishivi/agent-task-swe-smith to a dataset revision.
#   WORKERS=16             Parallel conversion workers (default: 16).

set -euo pipefail

SWE_SMITH_LIMIT="${SWE_SMITH_LIMIT:-0}"
SWE_SMITH_DST="${SWE_SMITH_DST:-rl_data/output/tasks_swe_smith}"
SWE_SMITH_CACHE="${SWE_SMITH_CACHE:-rl_data/output/_swe_smith_cache}"
SWE_SMITH_REVISION="${SWE_SMITH_REVISION:-}"
WORKERS="${WORKERS:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=(--dst "$SWE_SMITH_DST" --cache-dir "$SWE_SMITH_CACHE" --limit "$SWE_SMITH_LIMIT" --workers "$WORKERS")
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  ARGS+=(--skip-download)
fi
if [[ -n "$SWE_SMITH_REVISION" ]]; then
  ARGS+=(--revision "$SWE_SMITH_REVISION")
fi

uv run python -m rl_data.comparison.adapters.swe_smith "${ARGS[@]}"

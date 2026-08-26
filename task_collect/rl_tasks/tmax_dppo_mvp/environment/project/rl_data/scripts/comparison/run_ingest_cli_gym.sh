#!/bin/bash
# Ingest hamishivi/agent-task-cli-gym (CLI-Gym, 1,552 environment-inversion
# repair tasks) into our canonical Apptainer layout.
#
# CLI-Gym tasks are SWE-Smith repos (installed at /testbed in a conda env named
# `testbed`) whose ENVIRONMENT has been deliberately corrupted so a chosen
# subset of unit tests fails; the agent must repair the environment so those
# tests pass again. The verifier (the selected unit-test list) is NOT in the
# hamishivi parquet, so the adapter joins on `task_id` against the upstream
# release LiberCoders/CLI-Gym, which ships the run-tests.sh per task. Tasks
# whose run-tests.sh selects no explicit tests (a whole-suite run we can't
# fairly grade) are skipped; the rest (~1,452) materialize into our standard
# task layout (task.json + container.def + test_final_state.py +
# test_initial_state.py).
#
# Unlike the tarball-based adapters (R2E Gym / TerminalTraj) there is no source
# download step of our own: both HF datasets are pulled via
# `datasets.load_dataset`, which is cache-aware.
#
# Env overrides:
#   CLI_GYM_LIMIT=N     Convert only the first N joined tasks (0 = all, default).
#   CLI_GYM_DST=...     Destination tasks dir. Default: rl_data/output/tasks_cli_gym
#   CLI_GYM_CACHE=...   HF datasets cache dir. Default: rl_data/output/_cli_gym_cache
#   CLI_GYM_REVISION=<sha>  Pin hamishivi/agent-task-cli-gym to a dataset revision.
#   WORKERS=16          Parallel conversion workers (default: 16).

set -euo pipefail

CLI_GYM_LIMIT="${CLI_GYM_LIMIT:-0}"
CLI_GYM_DST="${CLI_GYM_DST:-rl_data/output/tasks_cli_gym}"
CLI_GYM_CACHE="${CLI_GYM_CACHE:-rl_data/output/_cli_gym_cache}"
CLI_GYM_REVISION="${CLI_GYM_REVISION:-}"
WORKERS="${WORKERS:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=(--dst "$CLI_GYM_DST" --cache-dir "$CLI_GYM_CACHE" --limit "$CLI_GYM_LIMIT" --workers "$WORKERS")
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  ARGS+=(--skip-download)
fi
if [[ -n "$CLI_GYM_REVISION" ]]; then
  ARGS+=(--revision "$CLI_GYM_REVISION")
fi

uv run python -m rl_data.comparison.adapters.cli_gym "${ARGS[@]}"

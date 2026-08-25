#!/bin/bash
# Ingest ucsb-mlsec/terminal-bench-env (TermiGen, Harbor 2.0 format) into our
# canonical Apptainer layout.
#
# Unlike the ET/OpenThoughts adapters, TermiGen lives on GitHub (not HF Hub).
# The adapter does a *partial* + *sparse* clone of only the
# environments_harbor/ subtree so we skip the ~200MB termigen_env.zip (TB-1.0
# artifact we don't use).
#
# Env overrides:
#   TG_LIMIT=N         Convert only the first N tasks (0 = all, default).
#   TG_DST=...         Destination tasks dir. Default: rl_data/output/tasks_termigen
#   TG_CACHE=...       Sparse-clone cache dir. Default: rl_data/output/_termigen_cache
#   TG_REVISION=<sha>  Pin to a specific commit/tag (default: origin/main).
#   SKIP_DOWNLOAD=1    Reuse existing clone without fetching from GitHub.
#   WORKERS=16         Parallel conversion workers (default: 16).

set -euo pipefail

TG_LIMIT="${TG_LIMIT:-0}"
TG_DST="${TG_DST:-rl_data/output/tasks_termigen}"
TG_CACHE="${TG_CACHE:-rl_data/output/_termigen_cache}"
TG_REVISION="${TG_REVISION:-}"
WORKERS="${WORKERS:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=(--dst "$TG_DST" --cache-dir "$TG_CACHE" --limit "$TG_LIMIT" --workers "$WORKERS")
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  ARGS+=(--skip-download)
fi
if [[ -n "$TG_REVISION" ]]; then
  ARGS+=(--revision "$TG_REVISION")
fi

uv run python -m rl_data.comparison.adapters.termigen "${ARGS[@]}"

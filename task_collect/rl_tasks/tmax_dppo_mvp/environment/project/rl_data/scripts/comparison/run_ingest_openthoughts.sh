#!/bin/bash
# Ingest open-thoughts/OpenThoughts-Agent-v1-RL into our canonical Apptainer layout.
#
# The dataset ships as a single tasks.parquet (~10 MB) with 728 tasks; the
# adapter downloads the parquet, extracts each row's gzipped tarball into
# OT_CACHE/extracted/, and flattens the result into OT_DST.
#
# Env overrides:
#   OT_LIMIT=N         Convert only the first N tasks (0 = all)
#   OT_DST=...         Destination tasks dir (default: tasks_openthoughts_agent_rl)
#   OT_CACHE=...       Download/extract cache (default: rl_data/output/_otrl_cache)
#   SKIP_DOWNLOAD=1    Reuse the existing cache without re-hitting HF

set -euo pipefail

OT_LIMIT="${OT_LIMIT:-0}"
OT_DST="${OT_DST:-rl_data/output/tasks_openthoughts_agent_rl}"
OT_CACHE="${OT_CACHE:-rl_data/output/_otrl_cache}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=(--dst "$OT_DST" --cache-dir "$OT_CACHE" --limit "$OT_LIMIT")
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  ARGS+=(--skip-download)
fi

uv run python -m rl_data.comparison.adapters.openthoughts_agent_rl "${ARGS[@]}"

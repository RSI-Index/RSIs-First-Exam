#!/bin/bash
# Ingest m-a-p/TerminalTraj-5k-instances (TerminalBench 1.0 format, shipped
# as a single 5k_instances.tar.gz on the HF Hub) into our canonical
# Apptainer layout.
#
# Unlike the ET/OpenThoughts adapters, TerminalTraj lives as a tarball, not
# a snapshot_download-able repo.  The adapter uses ``hf_hub_download`` to
# pull just the one 13 MB file and extracts 5,660 tasks out of it.
#
# Env overrides:
#   TT_LIMIT=N         Convert only the first N tasks (0 = all, default).
#   TT_DST=...         Destination tasks dir. Default: rl_data/output/tasks_terminaltraj
#   TT_CACHE=...       Download cache dir.   Default: rl_data/output/_terminaltraj_cache
#   TT_REVISION=<sha>  Pin to a specific HF dataset revision.
#   SKIP_DOWNLOAD=1    Reuse the existing extracted tarball without fetching.
#   WORKERS=16         Parallel conversion workers (default: 16).

set -euo pipefail

TT_LIMIT="${TT_LIMIT:-0}"
TT_DST="${TT_DST:-rl_data/output/tasks_terminaltraj}"
TT_CACHE="${TT_CACHE:-rl_data/output/_terminaltraj_cache}"
TT_REVISION="${TT_REVISION:-}"
WORKERS="${WORKERS:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=(--dst "$TT_DST" --cache-dir "$TT_CACHE" --limit "$TT_LIMIT" --workers "$WORKERS")
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  ARGS+=(--skip-download)
fi
if [[ -n "$TT_REVISION" ]]; then
  ARGS+=(--revision "$TT_REVISION")
fi

uv run python -m rl_data.comparison.adapters.terminaltraj "${ARGS[@]}"

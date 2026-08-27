#!/usr/bin/env bash
set -euo pipefail

# Publish the TMax Harbor datasets to the Harbor registry (public).
#
# Auth: you must already be logged in (`uvx harbor auth status` -> "Logged in as ...").
#       The token lives under your shared GPFS home, so a compute node sees it too.
#
# Usage (run on a compute node; it can take a long time for the full set):
#   bash scripts/publish_tmax15k.sh trial     # publish the 12-task trial dataset
#   bash scripts/publish_tmax15k.sh full      # init (if needed) + publish all 14601 tasks
#   bash scripts/publish_tmax15k.sh both      # trial, then full
#
# Optional env vars:
#   CONCURRENCY   - upload concurrency for `harbor publish -c` (default: 16)
#   AUTHOR        - "Name <email>" for the dataset manifest (default: oseyosey)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONCURRENCY="${CONCURRENCY:-16}"
AUTHOR="${AUTHOR:-oseyosey <oseyosey@users.noreply.github.com>}"

TRIAL_DIR="rl_data/output/TMax-15K-Harbor-trial"
FULL_DIR="rl_data/output/tasks_skill_tax_combined_20260506_legacy10k_new5k_harbor"
FULL_NAME="tmax/TMax-15K-Harbor"
FULL_DESC="15k compositional terminal-agent tasks: legacy 10k (self-contained) + v2 5k (intricate, multimodal fixtures). Programmatic verifier per task."

# `harbor publish --public` asks an interactive "Proceed? (y/N)" prompt and
# blocks on stdin. Feed it 'y' so it works non-interactively (e.g. under nohup
# on a node). printf writes once and exits 0, avoiding the SIGPIPE/141 that
# `yes |` would raise under `set -o pipefail`.
publish_trial() {
    echo "=== Publishing TRIAL ($TRIAL_DIR) ==="
    ( cd "$TRIAL_DIR" && printf 'y\ny\n' | uv run harbor publish . --public -c "$CONCURRENCY" )
}

publish_full() {
    echo "=== Publishing FULL dataset ($FULL_NAME) ==="
    if [ ! -f "$FULL_DIR/dataset.toml" ]; then
        echo "Initializing dataset manifest (auto-adds all task dirs; this scans+hashes 14601 tasks)..."
        ( cd "$FULL_DIR" && uv run harbor dataset init "$FULL_NAME" \
            --description "$FULL_DESC" --author "$AUTHOR" )
    else
        echo "dataset.toml already exists; skipping init."
    fi
    ( cd "$FULL_DIR" && printf 'y\ny\n' | uv run harbor publish . --public -c "$CONCURRENCY" )
}

case "${1:-both}" in
    trial) publish_trial ;;
    full)  publish_full ;;
    both)  publish_trial; publish_full ;;
    *) echo "usage: $0 [trial|full|both]"; exit 1 ;;
esac

echo "=== Done. Verify with: uv run harbor run -d \"$FULL_NAME@latest\" --env daytona --env-file .harbor.env -a terminus-2 -m anthropic/claude-sonnet-4-5-20250929 -l 2 ==="

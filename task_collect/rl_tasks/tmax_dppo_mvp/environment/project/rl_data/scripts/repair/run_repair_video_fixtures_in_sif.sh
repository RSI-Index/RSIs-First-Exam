#!/usr/bin/env bash
# Repair video fixtures using ffmpeg from base_intricate.sif while running the
# repair Python code on the host (uv / venv has tqdm + litellm; the SIF does not).
#
# Usage (from anywhere):
#   bash rl_data/scripts/repair/run_repair_video_fixtures_in_sif.sh \
#     --corpus-dir rl_data/output/tasks_skill_tax_v2_20260506_5k
#
# Optional: SIF=/path/to/base_intricate.sif

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

SIF="${SIF:-$PROJECT_ROOT/rl_data/containers/base_intricate.sif}"
if [[ ! -f "$SIF" ]]; then
  echo "[error] SIF not found: $SIF" >&2
  exit 1
fi

WRAPPER="$(mktemp "${TMPDIR:-/tmp}/tmax-ffmpeg-intricate.XXXXXX")"
cleanup() { rm -f "$WRAPPER"; }
trap cleanup EXIT

# shellcheck disable=SC2016
cat >"$WRAPPER" <<EOF
#!/usr/bin/env bash
exec apptainer exec \\
  --bind '${PROJECT_ROOT}:${PROJECT_ROOT}' \\
  '${SIF}' \\
  /usr/bin/ffmpeg "\$@"
EOF
chmod +x "$WRAPPER"

export FFMPEG_BINARY="$WRAPPER"

if command -v uv >/dev/null 2>&1; then
  exec uv run python -m rl_data.scripts.repair.repair_video_fixtures "$@"
fi
exec python3 -m rl_data.scripts.repair.repair_video_fixtures "$@"

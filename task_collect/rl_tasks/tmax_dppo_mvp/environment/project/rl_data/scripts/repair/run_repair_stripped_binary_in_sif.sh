#!/usr/bin/env bash
# Run stripped_binary repair with gcc from base_intricate.sif (host runs uv).
#
#   bash rl_data/scripts/repair/run_repair_stripped_binary_in_sif.sh \
#     --corpus-dir rl_data/output/tasks_skill_tax_v2_20260506_5k

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

SIF="${SIF:-$PROJECT_ROOT/rl_data/containers/base_intricate.sif}"
if [[ ! -f "$SIF" ]]; then
  echo "[error] SIF not found: $SIF" >&2
  exit 1
fi

WRAPPER="$(mktemp "${TMPDIR:-/tmp}/tmax-gcc-intricate.XXXXXX")"
cleanup() { rm -f "$WRAPPER"; }
trap cleanup EXIT

cat >"$WRAPPER" <<EOF
#!/usr/bin/env bash
exec apptainer exec \\
  --bind '${PROJECT_ROOT}:${PROJECT_ROOT}' \\
  '${SIF}' \\
  /usr/bin/gcc "\$@"
EOF
chmod +x "$WRAPPER"

export GCC_BINARY="$WRAPPER"

if command -v uv >/dev/null 2>&1; then
  exec uv run python -m rl_data.scripts.repair.repair_stripped_binary_fixtures "$@"
fi
exec python3 -m rl_data.scripts.repair.repair_stripped_binary_fixtures "$@"

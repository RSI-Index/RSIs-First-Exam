#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
assets_root="${DATACOMP_ASSETS_ROOT:-${task_root}/.assets}"
commonpool="${assets_root}/commonpool_s"
evaluation="${assets_root}/evaluation"
contract="${assets_root}/contracts/baseline_contract.json"

for required in "${commonpool}/manifests/asset_manifest.json" "${evaluation}" "${contract}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing prepared asset: ${required}" >&2
    echo "Run ${task_root}/setup/prepare_assets.sh first." >&2
    exit 2
  fi
done

mounts_json="$(python3 - "${commonpool}" "${evaluation}" "${contract}" <<'PY'
import json, sys
print(json.dumps([
    f"{sys.argv[1]}:/datasets/commonpool_s:ro",
    f"{sys.argv[2]}:/datasets/evaluation:ro",
    f"{sys.argv[3]}:/opt/contracts/baseline_contract.json:ro",
]))
PY
)"

snapshot_root="$(mktemp -d)"
snapshot_task="${snapshot_root}/datacomp_s_filter_discovery"
mkdir -p "${snapshot_task}"
cleanup() {
  rm -rf -- "${snapshot_root}"
}
trap cleanup EXIT
tar --exclude='./.assets' --exclude='./.runs' --exclude='**/__pycache__' --exclude='**/.pytest_cache' \
  -C "${task_root}" -cf - . | tar -C "${snapshot_task}" -xf -

harbor run --path "${snapshot_task}" --mounts-json "${mounts_json}" "$@"

#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
assets_root="${BIGVISION_ASSETS_ROOT:-${task_root}/.assets}"
tfds="${assets_root}/tfds"
initializers="${assets_root}/initializers"
contract="${assets_root}/contracts/baseline_contract.json"

for required in "${tfds}" "${initializers}/harbor_manifest.json" "${contract}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing prepared asset: ${required}" >&2
    echo "Run ${task_root}/setup/prepare_assets.sh first." >&2
    exit 2
  fi
done

mounts_json="$(python3 - "${tfds}" "${initializers}" "${contract}" <<'PY'
import json, sys
print(json.dumps([
    f"{sys.argv[1]}:/datasets/tfds:ro",
    f"{sys.argv[2]}:/datasets/initializers:ro",
    f"{sys.argv[3]}:/opt/contracts/baseline_contract.json:ro",
]))
PY
)"

snapshot_root="$(mktemp -d)"
snapshot_task="${snapshot_root}/bigvision_lit_coco_alignment"
mkdir -p "${snapshot_task}"
cleanup() {
  rm -rf -- "${snapshot_root}"
}
trap cleanup EXIT
tar --exclude='./.assets' --exclude='./.runs' --exclude='**/__pycache__' --exclude='**/.pytest_cache' \
  -C "${task_root}" -cf - . | tar -C "${snapshot_task}" -xf -

harbor run --path "${snapshot_task}" --mounts-json "${mounts_json}" "$@"

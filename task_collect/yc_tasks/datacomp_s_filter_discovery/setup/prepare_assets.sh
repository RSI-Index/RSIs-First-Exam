#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
assets_root="${DATACOMP_ASSETS_ROOT:-${task_root}/.assets}"
image="clip-datacomp-asset-setup:4a8df199"

if [[ " ${*} " == *" --dry-run "* ]]; then
  exec python3 "${task_root}/setup/prepare_assets.py" --task-root "${task_root}" --assets-root "${assets_root}" "$@"
fi

mkdir -p "${assets_root}"
docker build --file "${task_root}/setup/Dockerfile" --tag "${image}" "${task_root}"
exec docker run --rm --network bridge \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp/setup-home \
  --volume "${task_root}:/task:ro" \
  --volume "${assets_root}:/assets" \
  "${image}" --task-root /task --assets-root /assets "$@"

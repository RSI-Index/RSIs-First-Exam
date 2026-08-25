#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
assets_root="${BIGVISION_ASSETS_ROOT:-${task_root}/.assets}"
image="clip-bigvision-asset-setup:8921d514"

if [[ " ${*} " == *" --dry-run "* ]]; then
  exec python3 "${task_root}/setup/prepare_assets.py" --task-root "${task_root}" --assets-root "${assets_root}" "$@"
fi

manual_mount=()
manual_arg=()
if [[ " ${*} " != *" --verify-only "* ]]; then
  if [[ -z "${IMAGENET_MANUAL_DIR:-}" ]]; then
    echo "Set IMAGENET_MANUAL_DIR to a directory containing the official ILSVRC2012 train and validation archives." >&2
    exit 2
  fi
  manual_dir="$(cd "${IMAGENET_MANUAL_DIR}" && pwd)"
  manual_mount=(--volume "${manual_dir}:/imagenet-manual:ro")
  manual_arg=(--imagenet-manual-dir /imagenet-manual)
fi

mkdir -p "${assets_root}"
docker build --file "${task_root}/setup/Dockerfile" --tag "${image}" "${task_root}"
exec docker run --rm --network bridge \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp/setup-home \
  --volume "${task_root}:/task:ro" \
  --volume "${assets_root}:/assets" \
  "${manual_mount[@]}" \
  "${image}" --task-root /task --assets-root /assets "${manual_arg[@]}" "$@"

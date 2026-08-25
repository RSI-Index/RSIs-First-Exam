#!/usr/bin/env bash
set -euo pipefail
exec /opt/infer-venv/bin/python /task-tools/magpie_async_run.py "$@"

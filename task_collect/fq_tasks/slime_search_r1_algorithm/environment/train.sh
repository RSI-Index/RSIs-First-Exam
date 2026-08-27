#!/usr/bin/env bash
set -euo pipefail

exec python3 /task-tools/launch_train.py "$@"

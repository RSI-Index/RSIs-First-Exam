# RSI's First Exam

Public tasks and execution logs are available in [`rsi-tasks/`](rsi-tasks/) and [`rsi-logs/`](rsi-logs/), respectively, and can be reproduced using the bundled [`RSI-Harness`](RSI-Harness/).

## 🚀 Quick Start

You need a Linux machine with Docker, NVIDIA Container Toolkit, the GPUs
required by your task, and [`uv`](https://docs.astral.sh/uv/).

### 1 — Install RSI-Harness

```bash
git clone https://github.com/RSI-Index/RSI-Index-Public.git
cd RSI-Index-Public
uv tool install ./RSI-Harness
```

### 2 — Pick a task

```bash
find rsi-tasks -name task.toml -print

export RSI_TASK="$PWD/rsi-tasks/gemm-h100-refined"
export RSI_GPU_POOL="0"
export RSI_MODEL="<model-id>"
```

Check the task's `README.md` and `task.toml` for its GPU and storage
requirements. `RSI_GPU_POOL` accepts comma-separated GPU indexes or UUIDs.

### 3 — Run it

Sign in to Codex on the host, then run:

```bash
sudo -E "$(command -v rsi-harness)" run "$RSI_TASK" \
  --agent codex \
  --agent-auth local \
  --model "$RSI_MODEL" \
  --reasoning-effort xhigh \
  --gpus "$RSI_GPU_POOL" \
  --primary-reward reward \
  --max-submissions 1 \
  --verbose
```

To use an API key instead, export `RSI_AGENT_API_KEY` and optionally
`RSI_AGENT_API_BASE_URL`, then remove `--agent-auth local`.

### 4 — View results

```bash
sudo -E "$(command -v rsi-harness)" visualize
```

Open <http://127.0.0.1:8000>. Raw logs are under `logs/runs/`.

These tasks use the Harbor task format but must be run with `rsi-harness`, not
`harbor run`, to support long-horizon RSI. See
[`RSI-Harness/README.md`](RSI-Harness/README.md) for Blue Vela, recovery,
cleanup, and advanced options. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) to propose a new task.

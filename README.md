# RSI's First Exam

## 💥 Why RSI-Exam?

We are witnessing the dawn of a new era: AI is entering a recursive self-improvement loop. The central question is whether this loop can move beyond the best-known human-designed method and reliably extend the scientific frontier. Answering it requires careful measurement, and building that measurement is the purpose of this project.

[RSI's First Exam]() is an ongoing effort to evaluate whether AI agents can drive genuine recursive self-improvement and advance scientific discovery through real-world research—not merely reproduce existing results, sweep parameters, or succeed on toy-scale tasks.

### Task taxonomy

| Task                                                                                                                                   | Track         | Description                                                           | GPU requirement    |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------- | --------------------------------------------------------------------- | ------------------ |
| `Qwen-122B-RL`                                                                                                               | Signature-Task-Post-training | Optimize a full post-training stack under one fixed budget.           | 256× H100 |
| `marin-optimizer-update-geometry`                                                                                                            | Signature-Task-Pre-training  | Design a scale-general optimizer for the Marin scaling ladder.        | 256× H100 |
| `gpic-text-to-image`                                                                                                         | Signature-Task-Vision        | Train a text-to-image model on one epoch of GPIC.                     | 256× H100 |
| `[depth-width-allocation](rsi-tasks/depth-width-allocation-d13/depth-width-allocation/)`                                               | Pre-training  | Optimize decoder-width allocation under a fixed 200M training budget. | 8× H100            |
| `[learnability-cot](rsi-tasks/learnability-cot/)`                                                                                      | Post-training | Adapt reasoning traces for small-model math SFT.                      | 4× H100            |
| `[gemm-h100-refined](rsi-tasks/gemm-h100-refined/)`                                                                                    | MLSys         | Optimize an FP16 CUDA GEMM kernel for H100 throughput.                | 1× H100            |
| `[liger-tied-ce](rsi-tasks/liger-tied-ce/)`                                                                                            | MLSys         | Optimize tied-weight fused cross-entropy for Qwen3 SFT.               | 2× H100            |
| `[paged-gqa-decode](rsi-tasks/paged-gqa-decode/)`                                                                                      | MLSys         | Optimize Triton paged-GQA decode attention on H100 NVL.               | 1× H100            |
| `[minference-sparse-prefill](rsi-tasks/minference-sparse-prefill/)`                                                                    | MLSys         | Optimize Triton sparse-prefill attention on H100.                     | 1× H100            |
| `[molmo2-pointing-refined](rsi-tasks/molmo2-pointing-refined/)`                                                                        | Vision        | Optimize Molmo2 video-pointing at inference time.                     | 1× H100            |
| `[datacomp-small-filtering](rsi-tasks/datacomp-small-filtering/)`                                                                      | Vision        | Curate DataComp-small data for fixed ViT-B/32 training.               | 32× H100           |


Tasks and execution logs are available in [`rsi-tasks/`](rsi-tasks/) and [`rsi-logs/`](rsi-logs/), respectively, and can be reproduced using the bundled [`RSI-Harness`](RSI-Harness/). We welcome ongoing contributions of new tasks.

## 🚀 Quick Start

You need a Linux machine with Docker, NVIDIA Container Toolkit, the GPUs
required by your task, and `[uv](https://docs.astral.sh/uv/)`.

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

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Raw logs are under `logs/runs/`.

These tasks use the Harbor task format but must be run with `rsi-harness`, not
`harbor run`, to support long-horizon RSI. See
`[RSI-Harness/README.md](RSI-Harness/README.md)` for Blue Vela, recovery,
cleanup, and advanced options. See
`[CONTRIBUTING.md](CONTRIBUTING.md)` to propose a new task.

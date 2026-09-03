# RSI's First Exam

## 💥 Why RSI's First Exam?

We are witnessing the dawn of a new era: AI is entering a recursive self-improvement loop. The central question is whether this loop can move beyond the best-known human-designed method and reliably extend the scientific frontier. Answering it requires careful measurement, and building that measurement is the purpose of this project.

[RSI's First Exam]() is an ongoing effort to evaluate whether AI agents can drive genuine recursive self-improvement and advance scientific discovery through real-world research—not merely reproduce existing results, sweep parameters, or succeed on toy-scale tasks.

### Task taxonomy


| Task                                                                | Track          | Category      | Description                                                           | GPU requirement |
| ------------------------------------------------------------------- | -------------- | ------------- | --------------------------------------------------------------------- | --------------- |
| `Qwen-122B-RL`                                                      | Signature Task | Post-training | Optimize a full post-training stack under one fixed budget.           | 256× H100       |
| `marin-optimizer-update-geometry`                                   | Signature Task | Pre-training  | Design a scale-general optimizer for the Marin scaling ladder.        | 256× H100       |
| `gpic-text-to-image`                                                | Signature Task | Vision        | Train a text-to-image model on one epoch of GPIC.                     | 256× H100       |
| [`depth-width-allocation`](rsi-tasks/depth-width-allocation/)       | Public Task    | Pre-training  | Optimize decoder-width allocation under a fixed 200M training budget. | 8× H100         |
| [`learnability-cot`](rsi-tasks/learnability-cot/)                   | Public Task    | Post-training | Adapt reasoning traces for small-model math SFT.                      | 4× H100         |
| [`gemm-h100-refined`](rsi-tasks/gemm-h100-refined/)                 | Public Task    | MLSys         | Optimize an FP16 CUDA GEMM kernel for H100 throughput.                | 1× H100         |
| [`liger-tied-ce`](rsi-tasks/liger-tied-ce/)                         | Public Task    | MLSys         | Optimize tied-weight fused cross-entropy for Qwen3 SFT.               | 2× H100         |
| [`paged-gqa-decode`](rsi-tasks/paged-gqa-decode/)                   | Public Task    | MLSys         | Optimize Triton paged-GQA decode attention on H100 NVL.               | 1× H100         |
| [`minference-sparse-prefill`](rsi-tasks/minference-sparse-prefill/) | Public Task    | MLSys         | Optimize Triton sparse-prefill attention on H100.                     | 1× H100         |
| [`molmo2-pointing-refined`](rsi-tasks/molmo2-pointing-refined/)     | Public Task    | Vision        | Optimize Molmo2 video-pointing at inference time.                     | 1× H100         |
| [`datacomp-small-filtering`](rsi-tasks/datacomp-small-filtering/)   | Public Task    | Vision        | Curate DataComp-small data for fixed ViT-B/32 training.               | 32× H100        |


Tasks and execution logs are available in [rsi-tasks/](rsi-tasks/) and [rsi-logs/](rsi-logs/), respectively, and can be reproduced using the bundled [RSI-Harness](RSI-Harness/). We welcome ongoing contributions of new tasks.

### Evaluation Harness: RSI Harness

RSI's First Exam is powered by [RSI Harness](RSI-Harness/), an evaluation framework purpose-built for coding agents tackling ultra-long-horizon RSI tasks. It natively supports Harbor-format tasks and scales from local, single-node Docker runs to multi-node clusters.

Agents such as Claude Code and Codex develop solutions in a persistent, isolated Work container and submit each iteration to a fresh Judge environment for evaluation. They can use test feedback to refine their solutions until the submission limit is reached or the run times out, with the highest-scoring submission retained as the final result.

TODO: @ethan add figure here

## 🚀 Quick Start

You need a Linux machine with Docker, NVIDIA Container Toolkit, the GPUs
required by your task, and [uv](https://docs.astral.sh/uv/).

### 1 — Install RSI-Harness

```bash
git clone https://github.com/RSI-Index/RSI-Index-Public.git
cd RSI-Index-Public
uv tool install ./RSI-Harness
```

### 2 — Pick a task

```bash
find rsi-tasks -name task.toml -print

export RSI_TASK="$PWD/rsi-tasks/minference-sparse-prefill"
export RSI_GPU_POOL="0"
export RSI_MODEL="gpt-5.6-sol"
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
  --verbose
```

To use an API key instead, export `RSI_AGENT_API_KEY` and optionally
`RSI_AGENT_API_BASE_URL`, then remove `--agent-auth local`. For more information about RSI
Harness commands and configuration options, see
[RSI Harness Quick Commands](RSI-Harness/README.md#quick-commands).

### 4 — View results

```bash
sudo -E "$(command -v rsi-harness)" visualize
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Raw logs are under `logs/runs/`.

These tasks use the Harbor task format but must be run with `rsi-harness`, not
`harbor run`, to support long-horizon RSI.

## 📣 Contributing Tasks

We're actively looking for contributors to add new, challenging tasks. To streamline the process, we've built a dedicated VibeRSI data pipeline, enabling you to create a new task in 30 minutes or less. See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide to creating and submitting tasks. Have fun!

## 🤝 Contributors

## 🙏 Acknowledgements

# Training local REPL RLM with `prime-rl`

Verifiers-compatible training harness for `rlm.RLM` at depth=1. Designed to plug directly into [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) for end-to-end RL training of RLM policies. Does not require or use sandboxes.

This harness runs rollouts through a local REPL backend (subprocess-isolated, no cloud sandboxes), so it shells out to the same `LocalREPL`-style execution surface described in the [main README](../README.md#local-environments). It corresponds to the `local` environment in `rlm` — sub-LM calls are routed back through a proxy to the trainer's inference server, while Python code execution happens in a subprocess on the training host.

- `src/rlm_train/` — env, rubric, sub-LM proxy, subprocess REPL worker.
- `environments/oolong/` — OOLONG synth long-context QA env (example).
- `configs/rlm-qwen3-30b-example.toml` — example RL config.

## Launching a training run

With `prime-rl` installed and this directory's environment set up, launch with:

```bash
uv run rl @ training/configs/rlm-qwen3-30b-example.toml
```

The config wires the `oolong` environment into `prime-rl`'s orchestrator/trainer/inference loop. See [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) for distributed launch options and deployment details.

## Examples
* [Qwen3-30B-A3B-Instruct-0527] on the original suite of tasks: [https://huggingface.co/mit-oasys/rlm-qwen3-30b-a3b-v0.1](https://huggingface.co/mit-oasys/rlm-qwen3-30b-a3b-v0.1)

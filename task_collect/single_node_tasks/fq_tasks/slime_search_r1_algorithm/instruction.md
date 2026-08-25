# Improve the RL algorithm in Slime Search-R1

Train a better search-agent checkpoint while the model, questions, search engine, tool interaction, rewards, sampling, and compute remain fixed. This is an **algorithm task**: do not turn it into data selection, prompt engineering, reward editing, or retrieval tuning.

The starting checkpoint is the base-stage `Qwen/Qwen2.5-3B`, not an instruction-tuned model. The fixed Search-R1 interaction is the post-training stage that teaches it to reason, search, read retrieved wiki-18 passages, and answer.

## Frozen receipt

| Item | Value |
|---|---|
| Writable source | `/app/project/rl_research/algorithm_config.toml` only |
| Framework | `THUDM/slime@06ffdbe22be068b52f9ed0fc318c473f7030197e` |
| Starting model | `Qwen/Qwen2.5-3B@3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` (base) |
| Training prompts | `PeterJinGo/nq_hotpotqa_train@b7d80abfee334a7a91cb377544f09180d58b34f6`, fixed `train.parquet` order |
| Search | local CPU BM25, `wiki-18`, top-k 3, no external APIs |
| Interaction | pinned Slime Search-R1 generation and exact-match reward, at most two search turns |
| Sampling | rollout batch 32, 8 samples/prompt, temperature 1.0, response maximum 512 |
| Training batch | global batch 256, dynamic micro-batching, max 9,216 tokens/GPU |
| Budget | exactly 1,000 full rollout/update rounds, training and rollout seed 123 |
| Hardware | 8 H100s: actor 4, rollout 4, tensor parallel 2, colocated |
| Output | `/app/output/hf_model` selected through the experiment lifecycle |

The starter is the official Slime recipe: GRPO, `low_var_kl` loss coefficient `0.001`, entropy `0`, clip range `0.2/0.28`, and Adam at `1e-6` with constant schedule, weight decay `0.01`, and betas `(0.9, 0.98)`.

## What you may change

You may edit only `rl_research/algorithm_config.toml`. The allowlisted fields select and configure Slime's pinned built-in RL machinery for:

- advantage and return estimation (`grpo`, `gspo`, `cispo`, or REINFORCE++ variants compatible with the frozen no-critic topology);
- policy clipping and sequence/token normalization;
- token- or sequence-level clipping and differentiable online weighting;
- KL, entropy, importance sampling, and TIS treatment;
- the allowlisted optimizer and learning-rate settings.

Participant Python hooks are deliberately disabled: arbitrary same-process Python could mutate the frozen rollout, reward, or trainer and would not be an algorithm-only boundary.

Read `/task-tools/policy.yaml` before training. You may not change the model/tokenizer, questions or order/multiplicity, BM25 index/corpus/top-k, prompt/chat template, search grammar, rollout generation, reward, sampling, batch, seed, topology, 1,000-round budget, evaluator, or hidden/reference assets. External network access and alternate checkpoints are disabled.

## Research lifecycle

State one falsifiable hypothesis and submit a background attempt:

```bash
attempt=clip-001
python3 /task-tools/slime_async_run.py submit \
  --attempt-id "$attempt" \
  --hypothesis "sequence-level clipping should reduce long-trajectory variance and raise hidden exact match"
```

The worker stays inside the current Harbor container and writes durable status. Do not exit while it is queued or running; Harbor starts verification as soon as the agent ends.

Each attempt is executed by a fixed root-owned supervisor inside a private network namespace, so its Ray control plane is unreachable from the participant shell. The supervisor alone writes the trusted run under `/var/lib/slime-task/runs/` and a root-only attestation binding the algorithm config, exact command, 1,000 completed zero-based rollout IDs (final ID 999), and checkpoint hash. The normal agent cannot write either trusted location.

```bash
python3 /task-tools/slime_async_run.py status --attempt-id "$attempt"
```

Attempt IDs are immutable and cannot be reused. Keep negative results in `/app/output/experiments.jsonl`. Each attempt is a full 1,000-round run; the P0 task does not provide a smaller final-budget override.

After a successful attempt, select it and close the control loop:

```bash
python3 /task-tools/slime_async_run.py select --attempt-id "$attempt"
python3 /task-tools/slime_async_run.py complete-control
```

The final required artifacts are `/app/output/hf_model`, `provenance.json`, `experiments.jsonl`, `selected-attempt.json`, and `control-complete.json`. Directly running `/task-tools/train.sh` is for infrastructure diagnosis only; it does not create a selectable submission.

## Scoring

After the agent exits, the root verifier evaluates a fixed hidden slice of 256 NQ and 256 HotpotQA questions with the same pinned Search-R1 generation function, local BM25, two-turn interaction, temperature 1.0, response limit, and seed 123. It always formats prompts with the frozen base tokenizer and dynamically evaluates the base model, your checkpoint, and the released Search-R1 Qwen2.5-3B GRPO checkpoint.

```text
reward = clip((candidate_score - base_score)
              / (released_grpo_score - base_score), 0, 1)
```

The score is the mean reward produced by the pinned Search-R1 implementation. At this revision its advertised `format_score` argument is inert, so the effective metric is exact match; the task preserves that upstream behavior instead of silently changing the receipt. The verifier returns zero on a non-improving reference anchor, partial budget, changed model/tokenizer contract, missing provenance, illegal source change, hidden-data access, non-finite metric, or malformed artifact. Only aggregate metrics are published.

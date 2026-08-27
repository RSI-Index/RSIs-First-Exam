# Optimize the RLM orchestrator policy on OOLONG long-context QA

The system under test is the pinned RLM (Recursive Language Model) harness: a
frozen 30B model plus a Python REPL. A ~131k-token document is held in the REPL
as `context`; the model cannot see it directly and must write Python each turn
to slice it, inspect it, or delegate pieces to `llm_query()` sub-calls, then
submit a final answer. Fifty fixed OOLONG aggregate-statistics questions are
scored by the repository's raw correctness reward.

The model weights, the served deployment, the 50 evaluation rows, and the
scorer are all frozen. **Your editable artifact is the orchestration policy:
the root-policy / orchestrator prompt and the harness code that drives the
rollout.** Improve the mean raw OOLONG reward over the unchanged repository
default orchestrator baseline. Seven historical single-run calibrations scored
0.18–0.46 (mean `0.34`, std `0.09`). The current scored protocol evaluates five
independent stochastic rollouts of every row in one bounded rolling pool; its
matched no-op qualification scored `0.257183` over all 250 rollouts.

## Setup

| Item | Path / value |
|---|---|
| Writable workspace | `/app/project` (the pinned RLM repository) |
| RLM source | `alexzhang13/rlm@72d6940142ddfb84ee6be573dc999a37e633e671` |
| Base model (frozen) | `Qwen/Qwen3-30B-A3B-Instruct-2507` @ `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` |
| Released adapter (frozen) | `mit-oasys/rlm-qwen3-30b-a3b-v0.1` @ `7d72253525f47b1a8c3c4e1059ff88e40232dd2f`, served as `rlm-v0.1` |
| Serving (frozen) | two identical vLLM workers on the 8-GPU allocation; each is TP4 with `--max-model-len 16384`, `--gpu-memory-utilization 0.9`, `--enable-lora --max-lora-rank 64` |
| Evaluation data (frozen) | OOLONG `oolongbench/oolong-synth` validation `trec_coarse`, context length exactly 131072, seed 42 — the complete 50-row filtered selection |
| Scorer (frozen) | repository OOLONG `_synth_score` raw correctness reward |
| Compute | 8 H100s allocated as two TP4 workers; the qualified 5×50 pooled eval took 43m21s, and the complete hardened verifier job took 56m50s |
| Offline assets | read-only HF cache at `/root/.cache/huggingface`; `HF_HUB_OFFLINE=1` |
| Submission | the final state of `/app/project` (your policy code and prompts) plus provenance |
| Baseline | unchanged default orchestrator: historical single-run reward 0.18-0.46, mean `0.34`, std `0.09`; matched pooled qualification job 304037 = `0.257183` over 250 rollouts, 11 timeouts, 0 errors |

The pinned snapshots make repository references reproducible offline; they do
not introduce a different model, dataset, split, or scorer.

## What one rollout does

The long context is held in a subprocess-isolated Python REPL as `context`. At
each of at most 20 iterations, the root model emits Python code that can
inspect or transform the context and call the same served model through
`llm_query()` or `llm_query_batched()`. It finishes by setting
`answer["content"]` and `answer["ready"] = True`. If no final answer is set
within the caps, the rollout ends and the unchanged scorer scores its (empty)
answer.

The harness is depth 1: recursive queries resolve to sub-LM completions from
the same frozen served model. Iterations, REPL calls, and sub-LM calls are
recorded as diagnostics; only OOLONG correctness is rewarded.

## Frozen evaluation contract

Every decision-relevant scored evaluation — yours during iteration and the
final verifier's — uses exactly:

- the complete 50-row `trec_coarse@131072` selection, seed 42, with 5
  independent stochastic rollouts per row (250 total);
- a bounded, work-conserving rolling pool: at most 50 outer rollouts are active
  on each TP4 worker; the first 50 tasks on each worker cover all 50 rows once,
  and a completed task immediately admits the next queued rollout rather than
  waiting for a whole 50-row wave;
- two identical TP4 workers on 8 GPUs, with the five samples per row allocated
  3+2 across workers; queue wait occurs before admission and does not consume a
  rollout's own timeout;
- at most 20 root iterations per rollout;
- 4096-token completion caps for the root and for each sub-call;
- Qwen thinking disabled;
- sampling temperature deferred to the inference server's model default (the
  repository does not set one; do not pin your own for scored runs);
- an outer wall-clock timeout of 1800 seconds per rollout — a rollout that
  reaches it is finalized as-is, stays in the fixed 50-row denominator, and is
  scored by the unchanged scorer;
- the pinned base model + released adapter, served as `rlm-v0.1` on both
  workers.

One evaluation of one frozen policy is the full 250-rollout pool above. Testing
another policy variant is another evaluation; an optimization process is
multiple evaluations and is never folded into the cost of one. The qualified
no-op pool took 43m21s after serving became healthy. All rows remain in the
denominator when they error or reach the 1800-second cap.

Because sampling uses the server's stochastic model default, rollout outcomes
vary. The score is the arithmetic mean of all 250 rewards, equivalently the
mean of five complete 50-row repeats. Repeat count and pooled scheduling are
part of the task contract. Historical calibration measured single-run std
~0.09 (0/50 rows always correct, 39/50 stochastic across 7 runs); job 304037 is
the matched baseline for the current pooled schedule.

## Editable and locked scope

The action space is deliberately wide. You may modify anything in
`/app/project` that shapes how the frozen model is orchestrated, including but
not limited to: the RLM system prompt and `ORCHESTRATOR_ADDENDUM`, per-
iteration user messages, REPL bootstrap code and helper functions available to
the root, how the context is chunked/indexed/compressed, how sub-LM calls are
batched and routed, aggregation and voting over sub-answers, iteration-budget
management, and final-answer formatting.

Your policy code must be self-contained: the verifier imports your workspace
via path precedence against its own frozen image and does not install
dependencies, so vendor any new pure-Python code inside `/app/project` and do
not rely on packages you installed into your session. Your environment's
serving stack may also differ slightly in library versions from the
verifier's; the verifier's stack is authoritative for the score.

For the matched agent-side evaluation, run:

```bash
python /task-tools/rlm_eval_pool.py --workspace /app/project
```

The read-only task tool starts the two frozen TP4 workers, imports your edited
workspace with the same path precedence as the final verifier, runs the full
5×50 bounded pool, and writes raw results plus `metrics.json` below
`/app/output/evals/`. Do not substitute a hand-written command with a different
rollout count, concurrency schedule, temperature, or model. Keep `/app/output`
modest in size (a few hundred MB of logs/metrics; no model weights or dataset
copies).

Locked (hard zero if violated):

- the OOLONG environment, data selection, and scorer
  (`training/environments/oolong/**`) and the repository's own test suite
  (`tests/**`); the scoring verifier itself runs outside your workspace;
- the base model, adapter, and serving contract — no training, merging,
  weight edits, or substituting any other model;
- the evaluation contract above (50 rows, 5 rollouts per row, rolling-pool
  schedule, caps, timeout, thinking off);
- reading, at rollout time, the gold answers or any evaluation-row metadata
  your policy would not have for an unseen question — the policy must work
  from `context` and the question alone;
- hardcoding per-example answers, IDs, hashes, or lookup tables;
- degenerate final answers that enumerate multiple candidate values to exploit
  substring scoring — a final answer must commit to a single value;
- remote inference or any model other than the local `rlm-v0.1` service;
- web search, browsers, or network downloads as a research or solution source;
- concealing provenance.

## Baseline and iteration loop

1. Reproduce the baseline with `/task-tools/rlm_eval_pool.py`; compare against
   the matched pooled qualification reward `0.257183` and its raw metrics.
2. State a falsifiable hypothesis about why the default policy loses reward,
   grounded in its trajectories.
3. Change the policy, keep the frozen contract, and run one complete 250-rollout
   pooled evaluation.
4. Compare candidates on the same 50 rows with the same five stochastic
   samples per row and identical schedule.
5. Log every decision-relevant candidate in `/app/output/experiments.jsonl`.

An optimization attempt is one policy variant evaluated to decide the next
change.

## Final verifier and reward

The separate verifier rebuilds a clean serving of the pinned base + adapter,
hash-checks the locked files and adapter weights, imports your workspace
policy code, and runs five independent rollouts of each frozen row through the
same bounded rolling pool (250 total).
Reward integrity is mechanical: verifier rollouts run against a masked
dataset copy whose gold answers are blanked, and the scored reward is
recomputed by the verifier from each rollout's saved final answer using the
pinned scorer and verifier-held gold. Nothing your code does at rollout time
can read a gold label or alter the scored reward; degenerate
enumeration-style final answers are zeroed deterministically. Let `A` be the
mean over all 250 verifier-recomputed raw OOLONG rewards:

```text
reward = clip(A, 0, 1)
```

All 50 rows of each repeat stay in the denominator; errors and timeouts score
whatever the unchanged scorer assigns them. Iterations, REPL calls, sub-LM
calls, error and timeout counts are reported separately and do not change the
score. Report raw per-example results, not just the scalar.

## Required outputs

- `/app/project`: your final policy — the modified workspace is the submitted
  artifact.
- `/app/output/provenance.json`: must contain non-empty values for exactly
  these keys (missing or empty keys are an automatic zero):
  `base_model`, `base_model_revision`, `adapter`, `adapter_revision`,
  `workspace_diff_summary`, `evaluation_commands`, `evaluations_run`,
  `optimization_attempts`, `web_search`. Example:

  ```json
  {
    "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "base_model_revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
    "adapter": "mit-oasys/rlm-qwen3-30b-a3b-v0.1",
    "adapter_revision": "7d72253525f47b1a8c3c4e1059ff88e40232dd2f",
    "workspace_diff_summary": "rewrote ORCHESTRATOR_ADDENDUM; added chunked classification helper in rlm/utils/",
    "evaluation_commands": ["python /task-tools/rlm_eval_pool.py --workspace /app/project"],
    "evaluations_run": 7,
    "optimization_attempts": 6,
    "web_search": "disabled"
  }
  ```
- `/app/output/experiments.jsonl`: one row per decision-relevant candidate
  with hypothesis, change, updates to the policy, timestamps, status, and raw
  metrics.

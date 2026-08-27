# Optimize DeepSpec / DSpark scheduling

With the target and draft weights fixed, improve the candidate DSpark decode
policy's measured throughput while preserving the fixed target verifier's
in-path semantics.
The submitted object is the modified workspace, not a trained checkpoint.

## Setup

| Item | Path / value |
|---|---|
| Writable workspace | `/app/project` |
| Source | `deepseek-ai/DeepSpec@005e03b81cec38b7da6399833d609ee89a2587f2` |
| Target | `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` (fixed) |
| Draft | `deepseek-ai/dspark_qwen3_4b_block7@3457dff1417cb84927f6098a5fcb7cee85c934b7` (fixed) |
| Official evaluator | `/app/project/eval.py` |
| Official eval launcher | `/app/project/scripts/eval/eval.sh` |
| Offline assets | read-only HF cache at `/root/.cache/huggingface`; `HF_HUB_OFFLINE=1` |
| Candidate output | modified `/app/project` plus `/app/output/provenance.json` |
| Compute | Agent and verifier: 4 H100s; one six-hour agent trajectory, verifier uses 4 ranks |
| Network | Codex web search/external research disabled and HF offline; Apptainer does **not** provide hard egress isolation because model/policy-judge control traffic remains |
| Matched baseline status | one pristine qualification run measured on 4 H100s (job 221609); repeat it and match the same SIF/node class before claiming an improvement |

There are no optimizer updates in this fixed-weight task. One scientific
optimization attempt is one measured profile or A/B scheduler candidate that
influences the next change.

## What one decode candidate does

DSpark proposes a block of draft tokens, the target model verifies a prefix,
and the accepted prefix plus target bonus token advances generation. The useful
research variables are confidence calibration, prefix-survival estimation,
proposal/verification length, early stopping, and request/load-aware scheduling.
The target distribution and draft weights are not research variables here.

## Reference fidelity: paper lane versus environment lane

The paper's public offline Table 1 lane uses all nine repository tasks
(`gsm8k`, `math500`, `aime25`, `humaneval`, `mbpp`, `livecodebench`,
`mt-bench`, `alpaca`, `arena-hard-v2`), sampling temperature 1.0, fixed draft
blocks, and explicitly disables confidence scheduling to isolate accepted
length. Its 60--85% production result uses private DeepSeek-V4 traffic and is
not reproducible here.

The current final environment instead runs a custom qualification lane:

- five public tasks: GSM8K, MATH500, AIME25, HumanEval, LiveCodeBench;
- sample caps 100/100/30/100/100;
- temperature 0, seed 980406, default max-new-tokens 1024;
- four H100 ranks;
- a trusted wrapper audits **every** official target-verification call and every
  committed token against the probability tensor produced by that same cached
  target forward. At temperature 0 each committed token must have probability
  one. A separate greedy target generation is retained for exactly the first
  sample of each task/rank (20 diagnostics total), but is not the gate: an
  independent BF16 forward can choose a different exactly tied top logit;
- no serving queue, workload trace, batch/SLO grid, P50/P99, or sealed trace.

Trusted timing wraps only each official `generate_one_sample` call. The four
per-rank decode times are summed per rank and the maximum rank total is the
decode wall denominator. CUDA is synchronized at both timing boundaries. The
in-path gate adds only an asynchronous gather of each committed token's selected
probability inside that interval; its CPU synchronization/decision and the
independent greedy diagnostic occur after the timed call. Model loading, dataset
loading, TensorBoard writing, and the extra diagnostic are excluded from the
decode-seconds denominator and reported in end-to-end time instead.

Thus the environment reward is not a reproduction of Table 1 and does not
support the paper's production speedup claim. A scientific report must keep the
paper reproduction lane and the environment optimization lane separate.

## Baseline and iteration loop

First run the pristine pinned workspace on the same image, GPU count, task
subset, max-token setting, and node class. Do not compare wall-clock numbers
from different software stacks or GPU allocations.

Use the official paper-style evaluator during mechanism checks:

```bash
cd /app/project
python eval.py \
  --target_name_or_path Qwen/Qwen3-4B \
  --draft_name_or_path deepseek-ai/dspark_qwen3_4b_block7 \
  --temperature 1.0 --max-new-tokens 2048 \
  --seed 980406 --tensorboard-dir /app/output/paper_lane
```

For each candidate:

1. state a hypothesis about calibration, verification waste, or load-aware
   scheduling;
2. profile a pristine baseline and candidate under identical conditions;
3. record warmup policy, per-rank task counts, total generated tokens, elapsed
   time, accepted length, verify rate, and any correctness failures;
4. repeat finalists enough times to report median and dispersion; a single
   timing is qualification evidence, not a paper-quality systems result;
5. keep changes only if raw metrics improve for a general scheduling reason,
   not through task/prompt-specific branching.

## Final reward

The Reward-Integrity Gate runs first. A hard violation or unresolved audit gives
reward 0. Otherwise define:

- `G`: total recorded generated output tokens divided by the maximum summed
  official-decode seconds among the four ranks;
- `V`: mean official `verify_rate`, clipped to `[0, 1]`;
- `E`: whether every committed token in every official target-verification call
  passes the in-path probability-one check and all 20 diagnostics are present.

```text
reward = 0                                      if E is false or G is invalid
reward = (G / (G + 25)) * clip(V, 0, 1)        otherwise
```

The constant 25 and this scalarization are environment-defined, not paper
metrics. Always report raw `G`, accepted length, `V`, and correctness records.

## Editable and locked scope

You may edit general DSpark calibration and proposal/verification scheduling
code. You may not change target/draft weights, `eval.py`, evaluation datasets,
model IDs, seed, trusted target verification, reward parsing, or output
semantics; use remote inference; skip target verification; evade timing; or
branch on task names, prompts, hashes, benchmark IDs, or known answers.

## Required outputs and experiment record

- The complete modified `/app/project` is collected automatically.
- `/app/output/provenance.json` must record target/draft IDs and revisions,
  upstream commit, commands, assets/downloads, calibration inputs, evaluation
  commands, `optimization_attempts`, and `web_search: "disabled"`.
- `/app/output/experiments.jsonl` gets one row for every measured profile/A-B
  used for a decision, with `attempt`, `hypothesis`, `change`, `command`,
  `gpu_count`, warmup/repetition details, timestamps, `status`, and raw metrics.

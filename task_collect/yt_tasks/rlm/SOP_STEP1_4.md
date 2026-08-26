# RLM SOP step 1–4 audit

Date: 2026-07-20. This audit selects the task from executable repository code
and released repository artifacts only. It does not use a paper to choose the
model, dataset, experiment, metric, or budget.

## Step 1 — repository facts

### Frozen sources and artifacts

- RLM source: `alexzhang13/rlm@72d6940142ddfb84ee6be573dc999a37e633e671`.
- Released adapter: `mit-oasys/rlm-qwen3-30b-a3b-v0.1` at revision
  `7d72253525f47b1a8c3c4e1059ff88e40232dd2f`.
- Base model: `Qwen/Qwen3-30B-A3B-Instruct-2507` at revision
  `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- OOLONG data: `oolongbench/oolong-synth` at revision
  `f0d59eaf0febf130664cfceb710436c8e3216b2b`.
- The adapter is the repository release's LoRA: rank 32, alpha 64, dropout 0,
  with `q_proj`, `k_proj`, `v_proj`, and `o_proj` targets.
- The released serving setup is one 4-H100 vLLM replica with tensor parallel
  4, model length 16384, GPU memory fraction 0.9, LoRA enabled, and maximum
  LoRA rank 64.

The frozen adapter files have these SHA-256 values:

| File | SHA-256 |
|---|---|
| `adapter_config.json` | `fbbc7e816f055682935169e4b054051beb9e0dea79f7df99d5c95918240f64ca` |
| `adapter_model.safetensors` | `0ed035ff8a32fdc0e60b61d7df484f18c118b80c86447d3b244eca56a210f61c` |
| `README.md` | `490f9bcea9483ae79f182c48aeb86fd1efac880ef1c45130b393e2b1aa4b0d43` |

### Executable experiment

The decomposable repository experiment is fixed-model RLM policy evaluation:

- load the released adapter on its declared base model;
- use the repository's `RLMTrainEnv` with its default orchestrator enabled;
- evaluate the complete filtered OOLONG validation selection
  `trec_coarse`, context length exactly 131072, seed 42;
- run one rollout for each of the 50 selected rows;
- allow at most 20 root iterations;
- cap root and sub-LM completions at 4096 tokens;
- disable Qwen thinking;
- score with the repository OOLONG `_synth_score` raw correctness reward.

An offline loader preflight confirmed that this filter contains exactly 50
rows, with example IDs 0–49. `num_examples=50` therefore selects the complete
filtered split rather than a subsample.

The default system prompt is the repository RLM prompt plus
`ORCHESTRATOR_ADDENDUM`. The baseline result checker requires every saved row
to contain the addendum text `As an RLM, you should act as an orchestrator`.

## Step 2 — AutoResearch experiment candidates

### Selected: fixed released adapter, optimize the root policy

Keep the released adapter, base model, data rows, scorer, rollout count,
sampling caps, iteration cap, and deployment fixed. The editable artifact is
the repository root-policy/orchestrator code or prompt. A candidate must retain
the RLM mechanism: it receives the long context through the repository
environment and may use the repository REPL and sub-LM interface before
producing its final answer.

The unmodified repository default orchestrator is the Step 4 baseline. One
evaluation of one frozen policy over all 50 rows is one **single run**. Trying
another policy hypothesis is another run; an optimization process consists of
multiple runs and is not included silently in the cost of one run.

This candidate is useful because a submitted policy artifact has a direct
causal path to the measured reward, while model weights and evaluator remain
fixed. It also produces a short enough feedback loop for multiple research
iterations.

### Not selected: on-policy adapter training

The repository also contains a 200-update on-policy LoRA training example.
Earlier qualification established that the path is executable, but its
training/evaluation loop is substantially longer and a separately shortened
60-update protocol did not show a meaningful checkpoint-eval improvement over
its starting point in one stochastic run. That historical qualification is not
the selected task and is not used as this baseline.

## Step 3 — compute and time gate

### Frozen evaluation budget

The selected single-run contract is:

- 4 H100 GPUs, one vLLM TP4 replica;
- all 50 rows launched concurrently;
- one rollout per row and no evaluator retry;
- an outer wall-clock timeout of 1800 seconds **per rollout**;
- every row remains in the fixed 50-row denominator;
- if a future candidate reaches the timeout without an answer, the environment
  finalizes that row and the unchanged OOLONG scorer assigns its reward.

The 1800-second value is a versioned task contract, not a claim that the
repository's published endpoint used an outer timeout. It is applied identically
to the baseline and every candidate.

### Measured H100 time

LSF job 267451 measured the complete baseline on 4 clean H100s:

| Component | Measured wall time |
|---|---:|
| vLLM load, compile, CUDA-graph warmup, and health check | 432 s (7:12) |
| Complete 50-row evaluation and result saving | 1,584 s (26:24) |
| End to end | 2,019 s (33:39) |

Per-row total timing was:

| Statistic | Seconds | Clock time |
|---|---:|---:|
| minimum | 26.63 | 0:27 |
| median | 296.09 | 4:56 |
| p90 | 1,347.62 | 22:28 |
| p95 | 1,392.06 | 23:12 |
| maximum | 1,554.11 | 25:54 |

All 50 baseline rows completed before 1800 seconds, so the 30-minute cutoff
changed zero baseline outcomes. A 10-minute cutoff would have truncated many
normal trajectories, including naturally completed 12-minute and 20-minute
rows observed in this run. The measured 30-minute cutoff therefore removes
unbounded pathological tails while preserving the complete baseline sample.

One complete single run takes about 34 minutes on 4 H100s, far below the SOP's
6-hour and 256-H100 review thresholds. The released setup is already within
8 GPUs. More GPUs are not required for admission and would be a different
serving topology rather than an exact execution of the released TP4 setup.

## Step 4 — released-adapter reproduction

### Result

The frozen baseline is
`20260720T154236Z-released-adapter-default-orchestrator-r1`, LSF job 267451.
It completed successfully with zero optimization attempts.

| Check | Result |
|---|---|
| Clean GPU preflight | PASS: exactly 4 visible H100s, each with more than 75,000 MiB free |
| Frozen inputs | PASS: base 16/16 shards, released adapter and all 9 OOLONG validation shards present |
| Served model | PASS: `rlm-v0.1` resolves to the pinned adapter and pinned base model |
| Dataset coverage | PASS: 50/50 unique rows, example IDs 0–49 |
| Default orchestrator | PASS: required repository addendum present in 50/50 saved prompts; one unique system prompt |
| Evaluator | PASS: repository OOLONG raw scorer, one rollout per row |
| Errors | 0/50 |
| 1800-second timeouts | 0/50 |
| Raw OOLONG mean reward | `0.375000000000002` |

The reward distribution is 18 exact ones, one `0.75`, five tiny nonzero
partial scores, and 26 exact zeros. Stop conditions are 34
`has_final_answer`, 12 `max_turns_reached`, and 4 `prompt_too_long`.

The repository release README reports `0.472` for its released endpoint. The
measured `0.375` is a valid executable baseline for the frozen task contract,
but it must not be presented as an exact reproduction of that endpoint: this
run adds the disclosed outer timeout contract, uses the currently documented
default orchestrator path, and is one stochastic sample. No baseline row
actually hit the timeout, so timeout truncation does not explain this run's
score difference.

### Provenance

- Summary:
  `/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/runtime/rlm/released_adapter_baselines/20260720T154236Z-released-adapter-default-orchestrator-r1/summary.json`.
- Full results:
  `/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/runtime/rlm/released_adapter_baselines/20260720T154236Z-released-adapter-default-orchestrator-r1/results/evals/oolong--rlm-v0.1/9389be04/results.jsonl`.
- `results.jsonl` SHA-256:
  `b7b4cbc61051a679253b5d213c26b36dad18de4a7f9da3c176dc63f627012e7c`.
- Ordered `(example_id, answer, info)` SHA-256:
  `99d13800a96b4bcef2aa912067c5013a16b6d4602ab698a79474f9fc3cb90788`.
- Frozen runner SHA-256:
  `758e65997a9671c8e1465c1495276ef51f8765bde8dc1c1fd812b7ee84954e9e`.

Jobs 267377 and 267381 were launcher-input preflight failures before model
startup or rollout. Job 267384 successfully started the model but failed an
authenticated `/v1/models` launcher check before any rollout. These are
infrastructure attempts with zero scientific runs. Job 267451 is the sole
Step 4 scientific baseline reported above.

## Step 1–4 decision

PASS. The repository contains a cleanly separable fixed-model RLM
policy/orchestrator experiment. Its released default policy runs end to end on
4 H100s in about 34 minutes, the complete fixed evaluator has 50 rows, and a
30-minute per-rollout cutoff preserved all 50 baseline trajectories while
bounding future long tails. This candidate can proceed to task construction;
the older shortened training candidate should not remain the active design.

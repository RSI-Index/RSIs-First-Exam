# Olmo 3 RL-Zero (Math): RLVR from Base with OlmoRL via open-instruct

## Objective

Run the **Olmo 3 RL-Zero (Math)** pipeline end-to-end: apply **RLVR** (Reinforcement Learning with Verifiable Rewards) directly on the **`allenai/Olmo-3-7B` base model** using the **OlmoRL** algorithm and the **`allenai/Dolci-RL-Zero-Math-7B`** dataset, via the **`allenai/open-instruct`** framework.

This is a **full-stack pipeline task**. The primary goal is to stand up and run the complete RL-Zero loop — environment → assets → data → rollout ↔ train closed loop → evaluation — on the base model and produce a genuinely trained checkpoint. Hitting the exact target accuracy is secondary to demonstrating that every stage of the pipeline actually works.

## Success Criteria (graded)

The verifier assigns a **continuous reward in `[0, 1]`** based on how far through the pipeline you get and how well the trained model performs. Write nothing to `/app/result.txt` directly — the verifier inspects real training artifacts and evaluation output (see "Deliverables" below). Reward breakdown:

| Stage | What must be true | Reward |
|---|---|---|
| Environment + assets | open-instruct importable; `Olmo-3-7B` base + `Dolci-RL-Zero-Math-7B` downloaded | 0.15 |
| Data prepared | dataset converted to open-instruct RLVR prompt format with the RL-Zero math prompt template | +0.10 |
| **Rollout ↔ train closed loop** | a real training run: checkpoint(s) + optimizer state written, training log shows **≥ 500 gradient steps** with a **non-trivial, increasing train reward** | **+0.35** |
| Evaluation ran | AIME 2024 evaluated with the specified protocol; a score is produced | +0.10 |
| Accuracy | AIME 2024 pass@1 (bootstrapped avg over 32 samples) above the `Olmo-3-7B` base baseline, scaling up to the paper target | +0.30 |

The heaviest weight is on the **rollout ↔ train closed loop** — the hardest and most telling part of any RLVR stack. A trained checkpoint with a climbing reward curve is worth more than a high number you cannot back up with artifacts.

Accuracy sub-score is linear:

- `< base` (≈10% pass@1) → 0.0
- `base → 45%` → scales 0.0 → 0.30
- `≥ 45%` → 0.30 (full)

## Materials

1. **Base model**: `allenai/Olmo-3-7B` (HuggingFace) — the **base** model, NOT Instruct/Think. RL-Zero trains directly from base.
2. **Training data**: `allenai/Dolci-RL-Zero-Math-7B` (HuggingFace) — 13,314 decontaminated math prompts with verified answers.
3. **Framework**: `allenai/open-instruct` — install from `https://github.com/allenai/open-instruct`. Use its RLVR / GRPO trainer (the OlmoRL implementation).
4. **Evaluation benchmark**: AIME 2024 (30 problems), plus AIME 2025 as a secondary check.

## Algorithm: OlmoRL

OlmoRL builds on GRPO with improvements from DAPO and Dr.GRPO. Use open-instruct's implementation with these exact behaviors:

- **Group-relative advantage, NO standard-deviation normalization** (Dr.GRPO): `A_i = r(x, y_i) − mean({r(x, y_j)}_{j=1..G})`
- **Clip higher** (asymmetric): clip-lower `ε_low = 0.2`, clip-higher `ε_high = 0.272`
- **No KL loss**
- **No entropy regularization**
- **Token-level loss**: normalize by total tokens across the batch, not per-sample (avoids length bias)
- **Truncated importance sampling**: multiply the loss by the truncated IS ratio (vLLM vs trainer log-probs), **TIS cap = 2.0**
- **Zero-gradient-signal filtering**: drop groups whose rewards are all identical (zero-std advantage)
- **Active sampling**: refill the batch after filtering to maintain a constant effective batch size
- **Do NOT mask truncated (overlong) sequences** — the Olmo 3.1 recipe found masking hurts; train on overlong negatives.

## Reward Function (Math verifier)

Rule-based math verifier:

- Extract the model's final answer from the `Answer:` line (see prompt template below).
- Normalize and compare against the ground-truth answer using SymPy-based equivalence.
- Return **1** if equivalent, **0** otherwise. No model-based / LLM-judge reward.

## Prompt Template (RL-Zero Math)

RL-Zero trains from a purely midtrained base model, so use a **simple** template (special post-training formats like `<think>` or `\boxed{}` hurt here):

```
Solve the following math problem step by step.
The last line of your response should be the answer to the problem in form Answer: $Answer (without quotes) where $Answer is the answer to the problem.
{Math Question}
Remember to put your answer on its own line after "Answer:"
```

Clean evaluation prompts the same way (remove `\boxed{}` and other special formatting) so eval prompts match training prompts.

## Training Hyperparameters (7B RL-Zero)

Use these fixed values. Adapt only the GPU/parallelism layout to the available cluster while preserving the effective global batch (unique prompts per batch × group size).

| Hyperparameter | Value |
|---|---|
| Advantage estimator | GRPO (OlmoRL variant, no std normalization) |
| Dataset size | 13,314 prompts |
| Learning rate | 1.0×10⁻⁶ (constant, no schedule) |
| Optimizer | Adam |
| Minibatches | 1 |
| Training steps | 2,000 |
| Max prompt length | 2,048 tokens |
| Response (completion) length | 16,384 tokens |
| Unique prompts per batch | 32 |
| Group size (rollouts per prompt) | 8 |
| Sampling temperature (rollout) | 1.0 |
| Clip-lower (ε_low) | 0.2 |
| Clip-higher (ε_high) | 0.272 |
| TIS cap (ρ) | 2.0 |
| Use KL loss | No |
| Use entropy regularization | No |
| Max asynchrony | 8 |

## GPU / Distributed Configuration

The 7B RL-Zero run uses a **disaggregated actor/learner** layout on NVIDIA H100:

| Role | Count |
|---|---|
| Learner GPUs | 8 |
| Actor (vLLM rollout) GPUs | 64 |
| GPUs per actor (TP) | 1 |
| **Total** | **72 H100** |

Use open-instruct's launcher (Ray-based) to bring up the disaggregated rollout ↔ learner setup across nodes. Rollouts are generated by vLLM actors; the learner performs policy updates and pushes updated weights back to the actors. Getting this weight-sync loop running stably across nodes is the crux of the task.

## Step-by-Step Requirements

### 1. Environment Setup

- Install `open-instruct` and its dependencies (PyTorch, vLLM, Ray, flash-attn, etc.).
- Verify GPUs are visible and NCCL cross-node communication works.
- Download `allenai/Olmo-3-7B` and `allenai/Dolci-RL-Zero-Math-7B`.

### 2. Data Preparation

- Convert `Dolci-RL-Zero-Math-7B` into open-instruct's RLVR input format.
- Apply the RL-Zero Math prompt template shown above.
- Wire up the rule-based math verifier for reward computation.

### 3. Training Configuration

- Configure the OlmoRL/GRPO trainer with the exact hyperparameters above.
- Confirm: no KL, no entropy, clip [0.2, 0.272], no std normalization, token-level loss, TIS cap 2.0, active sampling on, no overlong masking.

### 4. Training Execution

- Launch the disaggregated actor/learner run across the available GPUs.
- Train toward ~2,000 steps (or until the compute budget is exhausted).
- Monitor: train reward should climb steadily (base model improves sharply in the first few hundred steps, then slowly); response length should grow and stabilize; no collapse.
- Persist checkpoints and the optimizer state; keep the full training log / metrics.

### 5. Evaluation

- Evaluate the trained checkpoint on **AIME 2024** (30 problems).
- Generate **32 samples per problem**, **temperature 1.0**, **max response length 32,768**.
- Report **pass@1** as the bootstrapped average over the 32 samples (primary), and pass@32.
- Extract answers from the `Answer:` line; compare with the rule-based math verifier.
- Also report the `Olmo-3-7B` **base** model's AIME 2024 pass@1 under the identical protocol, so the improvement is measurable.

## Deliverables (what the verifier inspects)

Write all of the following so the graded verifier can confirm the pipeline ran:

- `/app/checkpoints/` — trained checkpoint(s) including model weights **and** optimizer state.
- `/app/logs/train.jsonl` (or open-instruct's native log) — per-step metrics including `step` and `train_reward` (or equivalent), showing the reward trajectory.
- `/app/eval/aime2024.json` — evaluation output containing at least:
  ```json
  {"trained_pass@1": <float 0-100>, "base_pass@1": <float 0-100>, "pass@32": <float 0-100>, "n_samples": 32, "temperature": 1.0}
  ```
- `/app/result.txt` — the trained AIME 2024 pass@1 as a single float (e.g. `46.3`), consistent with `aime2024.json`.

## Expected Training Dynamics

- **Train reward**: climbs steadily across steps (from a low baseline upward).
- **AIME 2024 pass@1**: `Olmo-3-7B` base ≈ 10% → rises sharply in the first few hundred steps → converges toward **~45–50%** by ~2,000 steps.
- **pass@32**: also improves — RLVR pushes the model beyond its initial capability while maintaining sample diversity.
- **Response length**: grows and stabilizes (a sign of learned backtracking / verification).
- A **flat** reward or performance curve on the base model would indicate a broken loop or (in the negative-control sense) a signal-free reward.

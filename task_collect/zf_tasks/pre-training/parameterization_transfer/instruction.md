# Learning-rate and batch-size transfer across model scales

## Research Question

Can one low-complexity, scale-general parameterization jointly transfer
learning rate and global batch size while preserving absolute training quality
relative to standard parameterization and muP/muTransfer references?

For scale metadata `x_s`, parameter role `r`, and AdamH matrix role `g`, design
one frozen recipe

\[
(\eta_{\mathrm{base},s},B_s)=T(x_s;\phi),\qquad
\theta_r^{(0)}=\sigma_r(x_s;\phi)Z_r,\qquad
\eta_{g,s}=\eta_{\mathrm{base},s}m_g(x_s;\phi).
\]

The recipe may also define deterministic multipliers at approved residual and
readout sites. The same equations, constants, role semantics, and source
inventory must instantiate every scale from allowed quantities such as width,
depth, tensor shape, fan-in/fan-out, and parameter count.

This is transfer-rule discovery, not per-rung tuning. Shared low-dimensional
fits are allowed, but independent optima, exact-scale lookup tables, and
high-capacity fits to the visible ladder are not scale-general methods.

## Reference Baselines

The locked E0-E5 scientific profiles are defined in
`/task-tools/parameterization_scaling_ladder_profile.sh` and loaded by
`/task-tools/run_parameterization_scaling_ladder.sh`. Model architecture,
tokenizer, data and order, sequence length, training tokens, model-compute
precision, objective, WSD-in-token-space schedule, seed, and evaluators are
fixed. Candidate-controlled initialization, forward scales, AdamH matrix LR
scales, base LR, and global batch are the only scientific interventions.

| Rung | Role | Model | Parameters | Training tokens | Reference updates | Reference batch | GPUs |
|---|---|---|---:|---:|---:|---|---:|
| E0 | proxy | d1152-L12 | 550,337,664 | 2,904,358,912 | 44,317 | GBS16 / MBS2 | 8 |
| E1 | visible | d1408-L15 | 837,007,744 | 3,612,672,000 | 55,125 | GBS16 / MBS2 | 8 |
| E2 | visible | d1536-L16 | 998,036,992 | 4,982,571,008 | 38,014 | GBS32 / MBS4 | 8 |
| E3 | visible | d1792-L18 | 1,384,584,448 | 10,559,946,752 | 40,283 | GBS64 / MBS2 | 32 |
| E4 | visible | d2048-L21 | 1,934,716,160 | 14,805,106,688 | 56,477 | GBS64 / MBS2 | 32 |
| E5 | held-out | d2304-L23 | 2,544,614,912 | 18,617,466,880 | 35,510 | GBS128 / MBS1 | 128 |

The table describes the standard reference, not fixed candidate batch or
update settings. A selected batch must lie in the trusted feasible domain and
divide the token contract exactly; the runner derives
`updates = training_tokens / (4096 * global_batch)`. TP1 / PP1 and effective
batch semantics are fixed, while data-parallel degree and accumulation may
change only to materialize the selected batch.

Scored references are the E0 LR/batch copied wherever feasible, a compact
empirical power-law fitted on declared smaller scales, a fully specified
muP/muTransfer-derived rule, and the submission. Method-specific joint
LR-by-batch oracles are trusted diagnostics. Every visible reference receives
the same charged tuning-compute accounting as the candidate. The candidate
receives sanitized shapes and feasible batches but no held-out metric, oracle,
checkpoint, trace, run identity, or iterative feedback.

### Baseline metrics

Trusted run identity, scoring/final updates, compact Paloma histories,
fixed-window loss/cost curves, and offline W&B provenance are in
`/task-data/scale_baselines.json`. The file is the offline source of truth; do
not query W&B or invent missing points. E4 is a recorded non-final reference,
and E5 is scored at update 35,000 for a 35,510-update contract; never relabel
either as exact-final.

Visible metadata and frozen optimizer values are in
`/task-data/visible_scales.json` and
`/task-data/locked_parameterization.json`. The trusted boundary injects E5
metadata from `/task-data/hidden_target.json` only after freeze.

## Evaluation and success
TODO:

## Workspace

| Item | Path / value |
|---|---|
| Writable source | `/app/project` |
| Candidate module | `/app/project/parameterization_transfer.py` |
| Locked profiles | `/task-tools/parameterization_scaling_ladder_profile.sh` |
| Experiment runner | `/task-tools/run_parameterization_experiment` |
| Task state tool | `/task-tools/parameterization_scaling_task.py` |
| Visible metadata | `/task-data/visible_scales.json` |
| Hidden target | `/task-data/hidden_target.json` |
| Trusted evidence | `/task-data/scale_baselines.json` |
| Output root | `/app/output` |
| Hardware pool | 32 nodes × 8 H100 = 256 GPUs |

E0-E2 references use one node each and E3-E4 use four nodes each.
Candidate-visible work uses at most 128 GPUs; only the trusted controller may
launch E5. The outer allocation is created once; do not submit nested `bsub`
jobs or bypass the trusted training boundary.

## What you may change

Change one deterministic transfer recipe comprising joint base-LR/batch
scaling, role-wise initialization, approved fixed forward multipliers, AdamH
matrix-role LR multipliers, and a declared small set of shared constants.
Effective matrix LR is
`recipe.base_lr(scale) * recipe.lr_scale(role, scale)`. Embeddings,
normalization parameters, and other one-dimensional parameters remain in the
locked auxiliary Adam group.

The recipe may use only sanitized width, depth, tensor shape, fan-in/fan-out,
parameter count, semantic role, and base shape. Do not identify a rung through
labels, exact-scale cases, lookup tables, narrow kernels, or high-slope gates.
Do not use loss, checkpoints, metrics, seeds, target identity, data content,
token/update budgets, deadlines, hidden metadata, or held-out feedback.

Do not change architecture or parameter count, data/order, tokenizer, masks,
sequence length, per-scale tokens, optimizer family, auxiliary Adam
assignment/LR, objective, precision, evaluator, launcher, or task tools.
Forward scaling may not add trainable parameters, layers, branches,
nonlinearities, or asymptotic inference compute. Search procedures may fit
declared shared constants but may not be part of the deployed recipe.

Before held-out execution, freeze `PARAMETERIZATION_SPEC.json`,
`PARAMETERIZATION_SPEC.md`, `HYPOTHESIS.md`, the exact source hash, shared
constants, method complexity, all materialized scale/role predictions, and the
visible compute ledger. No source, rule, constant, LR, or batch may change
afterward.

Required ablations separate LR transfer, batch transfer, initialization, LR
mapping, every novel forward-scaling family, width/depth/parameter-count
dependence, transferred settings versus the method's oracle, and scientific
batch effects versus systems topology.

## Known rejected result

Independent per-scale optima, lookup-like continuous functions, or apparent
regret gains caused only by worse attainable oracle quality are not transfer.
Preserve them and their charged search cost as negative evidence.

## Persistent experiment loop

Use monotonic candidate IDs such as `pt-001-joint-power-law`; never reuse an
attempt directory.

1. Inspect the visible ledger, remaining FLOPs, and deadline. Write one exact
   hypothesis, fitting-scale declaration, prediction, and rejection rule to
   `/app/output/hypotheses/<candidate>.md`.
2. Implement the recipe and validate coordinates, determinism, architecture
   hash, finite scales, feasible batch, fixed tokens, policy, leakage, and
   activation/logit plus role-wise parameter/gradient/update scales across at
   least three visible widths.
3. Run visible experiments through:

   ```bash
   /task-tools/run_parameterization_experiment \
     --attempt-id <attempt-id> \
     --scale <E0-E4> \
     --recipe /app/project/parameterization_transfer.py \
     --hypothesis-file /app/output/hypotheses/<candidate>.md
   ```

4. Use forward-chaining or leave-one-scale-out confirmation. Record every
   screen, fit, failure, ablation, cost, and decision in
   `/app/output/experiments.jsonl`.
5. Prefer the simplest supported method; reject rung-specific behavior or
   excess effective degrees of freedom.
6. Preserve hypotheses, contracts, and evaluator outputs for failures; prune
   only bulky checkpoints.
7. Freeze and stage exactly once using a completed larger visible
   confirmation. Trusted held-out evaluation starts afterward and returns no
   feedback.

## Research deadline and handoff

Visible work shares a `2.4e20` training-FLOP budget; every update-bearing
screen, calibration, bracket, ablation, and run is charged. At least one full
E0 and one larger full confirmation are mandatory.

The outer allocation lasts 72 hours, the agent envelope 68 hours, and the
trusted scientific deadline 66 hours. At 66 hours, stop training and source
changes; use the next two hours for ledger and leakage audit, freeze, and
handoff. The final four hours are reserved for trusted held-out runs, oracle
evaluation, reloads, and cleanup.

Retain the strongest valid visible freeze. If none exists, report exhaustion.
Exiting early returns zero. Resume preserves the original deadline and budget;
interruption is not held-out completion.

## Required artifacts

- `/app/output/submission.json`;
- `/app/output/PARAMETERIZATION_SPEC.json`,
  `/app/output/PARAMETERIZATION_SPEC.md`, and
  `/app/output/COST_REPORT.json`;
- `/app/output/HYPOTHESIS.md` and `/app/output/experiments.jsonl`;
- exact source, source inventory, shared constants, complexity, allowed
  inputs, and materialized predictions for every scale and role;
- visible full-run, forward-chaining/leave-one-out, compute-ledger, component
  ablation, policy, reproducibility, provenance, and leakage evidence;
- retained hypotheses, contracts, and evidence for rejected candidates.

The trusted evaluator appends held-out contracts, matched-seed metrics, joint
oracle results, transfer diagnostics, and selected checkpoints after freeze.
These are never candidate feedback.

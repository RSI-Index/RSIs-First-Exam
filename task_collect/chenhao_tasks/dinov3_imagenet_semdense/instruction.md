# Beat DINOv3's ImageNet recipe on semantic AND dense quality at once

This is a vision-encoder science task. Under the pinned public
`facebookresearch/dinov3` repository, its released single-stage
ImageNet-1k pretraining configuration is the sealed baseline. Your goal is
to **train a self-supervised encoder that improves BOTH semantic quality
(ImageNet k-NN and linear probe) and dense quality (ADE20k linear
segmentation and NYUv2 linear depth) over that baseline — at a matched
training GPU-hour budget on the released 4×8 H100 topology** — by changing
the SSL objective, the architecture, or the training recipe.

The scored metric is a composite of all four probes (see Final reward):
improving semantics while degrading dense features, or vice versa, does
not score. This tension is real and is the scientific heart of the task —
the repository itself contains machinery built for exactly this problem
(gram anchoring, disabled in the baseline lane; storage/register tokens,
off by default; the iBOT masking branch), none of it validated at
ImageNet-1k scale in the release.

Do not use papers or any external source: the pinned repository, the
staged data, and your own measurements are the only allowed inputs.

## Setup

| Item | Path / value |
|---|---|
| Writable workspace | `/app/project/dinov3`, pinned `facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751` |
| Baseline config (starting point, editable copy) | `dinov3/configs/train/vitl_im1k_lin834.yaml`: ViT-L/16, RoPE, DINO+iBOT heads (65536 prototypes), KoLeo 0.1, global batch 2048 (64/GPU × 32), 100 ep × 1250 iters, AdamW lr 1e-3 sqrt-scaled, bf16 params / fp32 reduce; gram anchoring, high-res adaptation, and distillation all disabled |
| Training data | `$DINO_IMAGENET_DIR` — ImageNet-1k train split, staged read-only, in the repository's expected layout with metadata files pre-generated. **Labels are staged solely for the frozen probes; using them in encoder training is forbidden** |
| Probe data | `$DINO_IMAGENET_DIR` (k-NN / linear), `$DINO_ADE20K_DIR` (segmentation), `$DINO_NYU_DIR` (depth); staged offline, never for encoder training |
| Topology | released 4 nodes × 8 GPUs = 32 ranks, exactly; H100s supplied by the environment; launch via `/task-tools/launch_dinov3_4x8.sh` |
| Training budget | **at most `$DINO_GPU_HOURS_CAP` (= 450.0) H100-hours of GPU compute per candidate attempt** (`wall_seconds × 32 / 3600`); never exceed it |
| Parameter budget | backbone parameters ≤ 307,000,000 (≈1.02 × pristine ViT-L/16; exact pristine count re-measured at sealing); smaller is allowed. SSL heads/teacher are excluded from the cap and from submission |
| Pretrained weights | forbidden — every parameter of every model (student, teacher, heads) must originate from random initialization inside this environment |
| Determinism lane | released seed and data order are the comparability lane: candidates that keep them are directly comparable to the sealed anchors below and to each other |
| Nominal images ceiling | at most 200% of the baseline's sealed images-seen per attempt (generous; prevents throughput luck from becoming the result); report `images_seen` truthfully in the ledger |
| Backbone interface contract | the submitted backbone must be constructible from your workspace by config name and must preserve the repository's `get_intermediate_layers()` semantics (patch tokens + class token at requested depths) — the frozen dense probes consume it exactly that way |
| Screening eval | `python /task-tools/dinov3_local_eval.py --checkpoint <pt> [--knn-only]` (GPU time counts against your budget) |
| Submission | one selected backbone staged via `/task-tools/dinov3_task_tool.py stage`; provenance; ledger |
| Network | external research and web search disabled; all data read-only |

Known repository pitfall (task-verified): the README fast-setup command
passes an `ImageNet22k:` dataset path while the config expects
`ImageNet:split=TRAIN`. The staged launcher already passes the correct
IN1k path; keep it correct in your own commands.

## Budget accounting

The budget is **wall-clock GPU-hours, not epochs or images seen**. The
async submit wrapper stamps start/end times and a watchdog hard-kills any
attempt that would exceed the per-attempt cap. All GPU compute of an
attempt counts (training, GPU-side preprocessing, screening probes you
run); CPU-only work is free; the frozen final evaluation is not charged.
Faster training (throughput engineering, schedule changes, masking
strategies) converts directly into more optimization steps within the
same budget.

## Action space

Free variables — the SSL objective and the architecture are both yours:

- the DINO/iBOT/KoLeo loss stack: weights, prototypes, centering
  (Sinkhorn-Knopp vs softmax), temperatures, EMA schedule, masking ratios
  and strategy, new auxiliary losses (e.g. dense/gram-style
  regularization: `dinov3/loss/gram_loss.py` ships fully implemented and
  unused in this lane — a gram teacher may come from an earlier
  checkpoint of the same run);
- architecture within the parameter budget: RoPE parameterization,
  storage/register tokens (`n_storage_tokens`, off in the baseline),
  patch size, head layout, norms, new modules — trained from scratch, as
  a new named config under `dinov3/configs/train/`;
- optimizer, LR schedules, weight decay, batch shape, crop/augmentation
  policy, precision, data ordering, seed, and `train.py` itself;
- distillation between models trained inside this environment.

Frozen: the staged ImageNet-1k corpus (no external or synthetic images),
the label-free constraint on encoder training, the GPU-hour cap, the 4×8
topology, the backbone parameter budget, and the frozen four-probe
evaluator.

## Provisional baseline (to be replaced by our sealed reproduction)

The sealed baseline is the unchanged `vitl_im1k_lin834.yaml` under this
exact contract. **Semantic values below are from the pinned config's own
header (repository-documented); dense values are rough literature-derived
ESTIMATES — no published number exists for this exact config. Before
release the task authors run the pristine config once on this
infrastructure (also settling the repository's open reproduction question,
issue #309) and every number below is replaced by that sealed run.**

| Quantity | Provisional value | Sealed value |
|---|---:|---:|
| Backbone parameters | ≈ 300,000,000 | pending author run |
| IN1k k-NN top-1 | 0.822 (config header; README states 82.0) | pending author run |
| IN1k linear top-1 | 0.833 (config header; README states 83.5) | pending author run |
| ADE20k linear mIoU | ~0.40 (ESTIMATE) | pending author run |
| NYUv2 linear depth RMSE | ~0.45 (ESTIMATE; lower is better) | pending author run |
| Training GPU-hours | ≈ 448 (README: 32 H100 × ~14 h) | pending author run |
| Images seen | ≈ 128M (100 ep × 1.28M) | pending author run |
| Composite `G` | 1.0 by construction | 1.0 by construction |

**Sealed screening anchors (pending author run).** The pristine run records
k-NN and a fast dense proxy (frozen 20-iteration ADE20k linear head) at
epochs 10 / 25 / 50 / 75 / 100 via the repository's in-training benchmark
hooks (`benchmark_high_frequency.yaml`), under the released seed and data
order. The sealed table will appear here. A candidate on the determinism
lane can compare its own in-training probes at the same epochs against
these anchors and kill weak hypotheses at a fraction of a full budget —
watching the semantic and dense trajectories *diverge* mid-run is also the
main observation that generates good hypotheses in this task.

**In-repo ablation anchors.** The mechanism-attribution toggles are single
config lines, and controlled ablations of them are encouraged as
screening-scale experiments: `koleo_loss_weight: 0` (what does KoLeo carry?),
`ibot.loss_weight: 0` (does masked patch prediction drive dense quality?),
`n_storage_tokens: 4` (do registers trade dense artifacts for semantics at
this scale?), and gram anchoring enabled from an own-run teacher (built for
exactly the dense-degradation problem, never validated at IN1k scale).

## Agent experiment loop

One scientific single run is one candidate trained within the full
GPU-hour budget on 4×8 ranks. Screening runs (shorter budgets, e.g.
20-epoch prefixes with the full-run LR horizon) are allowed for triage;
label them `screening` in the ledger.

For each attempt:

1. State one falsifiable hypothesis — mechanism → expected effect on the
   semantic probes, the dense probes, or the trade-off between them, and
   why.
2. Change one interpretable variable group; give the candidate a new named
   config; record the exact patch or source hash.
3. Submit asynchronously and end your turn (do not sleep or poll):

   ```bash
   attempt=dino-001-<short-name>
   python /task-tools/dinov3_async_run.py submit \
     --attempt-id "$attempt" --poll-seconds 1800 --run-kind full_budget -- \
     /task-tools/launch_dinov3_4x8.sh \
       dinov3/configs/train/<YourConfig>.yaml \
       --output-dir "/app/output/attempts/$attempt/outputs"
   ```

   The async adapter keeps the environment alive, checks `status.json`
   every 30 minutes without an LLM call, and resumes this session when
   the attempt reaches a terminal state (`completed`, `failed`, or
   `killed_budget`). On resume, summarize the real evidence:

   ```bash
   python /task-tools/dinov3_task_tool.py summarize \
     --attempt "/app/output/attempts/$attempt"
   ```

4. Append one truthful JSON object to `/app/output/experiments.jsonl` per
   decision cycle. Required keys: `attempt_id`, `hypothesis`, `change`,
   `command`, `source_diff_sha256`, `run_kind`, `gpu_hours`,
   `images_seen`, `label_usage`, `status`, `metrics`. `label_usage` must
   be `"none_during_training"`; statuses include `rejected`, `failed`,
   `killed_budget`, and exactly one final `selected`.
5. Keeping `/app/output/RESEARCH_NOTES.md` up to date is encouraged
   (which mechanisms moved which probes and why, citing `attempt_id`s;
   negative results) — it is your notebook, not a scored deliverable.
6. Select exactly one completed full-budget checkpoint, then stage and
   audit:

   ```bash
   python /task-tools/dinov3_task_tool.py stage \
     --checkpoint /app/output/attempts/<selected>/outputs/<run>/<ckpt> \
     --model-config <YourConfigName>
   python /task-tools/dinov3_task_tool.py audit --output-root /app/output
   python /task-tools/dinov3_async_run.py complete-control
   ```

## Final reward

The Reward-Integrity Gate runs first; a confirmed hard violation or
unresolved audit receives reward 0. Otherwise the frozen verifier rebuilds
your backbone **from your workspace code** with the recorded config name,
checks the parameter budget, and runs the frozen four-probe protocol
(pinned k-NN, linear, ADE20k linear-segmentation, and NYUv2 linear-depth
drivers, fixed seeds and probe recipes). Each probe is normalized by the
sealed baseline:

```text
r_knn  = knn / B_knn        r_lin  = linear / B_linear
r_seg  = miou / B_miou      r_dep  = B_rmse / rmse        # lower RMSE is better

G = (r_knn × r_lin × r_seg × r_dep) ^ (1/4)

reward = 0                        if any r_i < 0.98   (no-regression floor)
reward = clip((G − 1) / (G_ref − 1), 0, 1)   otherwise, G_ref = 1.03
```

The geometric mean rewards balanced improvement and punishes trading one
axis for the other; the floor makes a hard sacrifice of any single probe
score zero. Probe recipes (probe hyperparameter grids, seeds, layer
selection) are pinned task infrastructure — the same frozen recipe scores
the baseline and every candidate; the backbone must be deterministic in
eval mode (no test-time adaptation). The floor (0.98) and `G_ref = 1.03`
are provisional: at sealing, the authors run the frozen probes three times
on the baseline checkpoint and place both values outside the measured
probe-noise band. Scientific reporting must include all raw probe metrics;
do not optimize a rounded reward.

## Required outputs

- exactly one staged `submission/backbone.pt` below `/app/output`;
- `/app/output/model_state.pt`, tensor-only backbone weights, exported by
  the staging tool;
- `/app/output/submission-selection.json` from the frozen task tool;
- `/app/output/provenance.json` with `model_config`, `parameters`,
  `training_data`, `images_seen`, `gpu_hours` (per attempt and total),
  `label_usage: "none_during_training"`, `training_command`,
  `upstream_commits`, `pretrained_weights: "none"`,
  `evaluation_commands`, `optimization_attempts`,
  `web_search: "disabled"`;
- `/app/output/experiments.jsonl`, one row per decision cycle;
- useful logs under `/app/output/logs`.

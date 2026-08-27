# Beat the hardened OpenCLIP recipe ceiling under a fixed GPU-hour budget

This is a vision-encoder science task. Under a frozen snapshot of the
DataComp-medium image-text pool and the pinned `mlfoundations/open_clip` +
`mlfoundations/datacomp` repositories, your goal is to **train an
image-text model that beats the sealed hardened baseline on the frozen
38-dataset evaluation suite — at a fixed training GPU-hour budget on a
fixed 8×H100 node**.

The hardened baseline is not the official protocol starting point: the
task authors already swept the obvious in-repo switch combinations (loss
family, patch dropout, tower presets, filtering thresholds, schedule
presets) under this exact contract and sealed the **best** of those runs
as the baseline. The full sweep and its scores are disclosed below so you
do not waste budget rediscovering them — read it as a **published ablation
map of the default recipe space**: it tells you, with measured numbers,
which single mechanisms move the metric and by how much at this scale.
Beating the baseline therefore requires findings beyond flag-flipping:
curation strategies, objective design, architectural mechanisms, or
compositions that the default configuration space does not contain — and
the sweep map is your evidence base for choosing where to look.

Do not use papers or any external source: the pinned repositories, the
staged data and metadata, and your own measurements are the only allowed
inputs.

## Setup

| Item | Path / value |
|---|---|
| Writable workspace | `/app/project` — pinned `mlfoundations/open_clip@db4d4911cc0fc4bc58bc98579c52875750450530` (training) and `mlfoundations/datacomp@4a8df1992566ef8334773f7152e1855b1f716162` (protocol + local eval) |
| Training data | `$CLIP_SNAPSHOT_DIR/shards/` — frozen webdataset snapshot of the DataComp-medium pool, crawled and sealed by the task authors (`MANIFEST.json` is the canonical definition); read-only |
| Pool metadata | `$CLIP_SNAPSHOT_DIR/metadata/` — official DataComp parquets for the snapshot uids: original alt-text, ViT-B/32 and ViT-L/14 CLIP scores, image sizes, dedup hashes. Usable freely for curation |
| Evaluation | pinned `datacomp/evaluate.py` (38-task suite), eval data pre-staged under `$DATACOMP_EVAL_DIR`; fully offline |
| Primary metric | `avg_38` — unweighted mean of the suite's per-task main metrics (ImageNet top-1 reported separately, not scored) |
| Topology | exactly 1 node × 8 H100; supplied by the environment |
| Training budget | **at most `$CLIP_GPU_HOURS_CAP` (= 20.0) H100-hours of GPU compute per candidate attempt** (see Budget accounting); never exceed it |
| Parameter budget | whole-model parameters ≤ 189,000,000 (1.25 × the ViT-B/32 default); smaller is allowed |
| Pretrained weights | forbidden — every parameter of the submitted model must be trained from random initialization inside this environment |
| Determinism lane | released seed and the pinned shard-order file are the comparability lane: candidates that keep them are directly comparable to the sealed sweep rows, the screening anchors, and each other |
| Nominal samples ceiling | at most 200% of the baseline's sealed samples-seen per attempt (prevents throughput luck from becoming the result); report `samples_seen` truthfully in the ledger |
| Screening eval | `python /task-tools/clip_local_eval.py --model-config <name> --checkpoint <pt> [--fast]` (GPU time counts against your budget) |
| Submission | one selected checkpoint staged via `/task-tools/clip_task_tool.py stage`; provenance; ledger |
| Network | external research and web search disabled; all data read-only |

## Budget accounting

The budget is **wall-clock GPU-hours, not samples seen**. The async submit
wrapper stamps start/end times for every GPU job and a watchdog hard-kills
any attempt that would exceed the per-attempt cap
(`charged_gpu_hours = wall_seconds × 8 / 3600`). Rules:

- All GPU compute belonging to an attempt counts: training, GPU-side data
  preprocessing, feature extraction for curation, and screening evals you
  run on GPU. CPU-only curation is free.
- A curation artifact built once on GPU is charged to the attempt that
  built it and may be reused by later attempts at no extra charge, if
  truthfully recorded in the ledger.
- The frozen final evaluation is task infrastructure and is not charged.

Because the budget is time, training-efficiency improvements are a real
research axis: anything that raises useful throughput converts directly
into more samples seen within the same budget. **Compute allocation is a
first-class scientific axis of this task**: resolution and token-length
curricula, low-resolution pretraining with a short high-resolution finish,
and tower-size-vs-samples trades are all legitimate hypotheses — every
sealed sweep row publishes its measured throughput so you can plan
allocation before spending budget.

## Sealed hardened baseline and the disclosed sweep

The task authors run the following trivial-configuration sweep under this
exact contract (identical snapshot, GPU-hour budget, topology, from-scratch
init; official image-based ∩ CLIP-score(top-30%) filtering unless noted).
**The `avg_38` values below are provisional literature-derived estimates;
the authors execute every sweep row themselves on the sealed snapshot
before release, and the measured table replaces this one.** All final
numbers come from the frozen evaluator.

| Sweep id | Configuration delta vs protocol default | `avg_38` (provisional est.) | Sealed |
|---|---|---:|---|
| S-00 | protocol default: ViT-B/32, CLIP loss | 0.328 (paper) | pending |
| S-01 | SigLIP sigmoid loss | ~0.34 | pending |
| S-02 | patch dropout 0.3 (extra samples within budget) | ~0.33 | pending |
| S-03 | S-01 + S-02 | ~0.345 | pending |
| S-04 | ViT-B/16 tower (fewer samples within budget) | ~0.33 | pending |
| S-05 | CoCa captioning head | ~0.33 | pending |
| S-06 | no-filtering (full snapshot pool) | ~0.29 | pending |
| S-07 | CLIP-score top-40% / top-20% (best of) | ~0.33 | pending |
| S-08 | cosine → const-then-cool schedule preset | ~0.33 | pending |
| S-09 | best-of-sweep composition chosen by the authors | ~0.345 | pending |

**The sealed baseline `S_base` is the maximum of the measured table**
(provisional placeholder 0.345 until the sweep is run). Attempts that
merely reproduce table rows are expected to land at or below `S_base` and
score 0.

| Quantity | Provisional value | Sealed value |
|---|---:|---:|
| Hardened baseline `avg_38` | 0.345 (estimate) | pending author sweep |
| Hardened baseline ImageNet zero-shot | ~0.31 (estimate) | pending author sweep |
| Whole-model parameters | ≤ 189,000,000 | pending author sweep |
| Training GPU-hours | 20.0 (provisional cap) | pending author sweep |

## Action space

Free variables: data curation and sampling over the staged snapshot (using
the staged metadata, including the provided CLIP scores), the training
objective, image and text tower architecture within the parameter budget
(from scratch; new named model configs under `open_clip` are honored by
the verifier), tokenizer and text preprocessing, augmentation, optimizer,
schedule, batch size, precision, parallelism, and the training harness
itself. Re-running disclosed sweep rows as controlled ablation anchors is
allowed (they score 0 by construction).

Frozen: the staged snapshot (no external or synthetic images; synthetic
captions from models trained inside this environment on staged data are
allowed and must be declared), the GPU-hour cap, the 8×H100 topology, the
parameter budget, the no-pretrained-weights rule, and the final evaluator.

Prefer one variable group per attempt, with fixed seed and fixed curation
when you want comparability with the sweep table above.

## Agent experiment loop

One scientific single run is one candidate trained within the full
GPU-hour budget. Screening runs (shorter budgets) are allowed for triage;
label them `screening` in the ledger.

For each attempt:

1. State one falsifiable hypothesis — in particular, why your change should
   move quality where the disclosed sweep could not.
2. Change one interpretable variable group; record an exact patch or
   curation-recipe hash.
3. Submit asynchronously and end your turn (do not sleep or poll):

   ```bash
   attempt=clip-001-<short-name>
   python /task-tools/clip_async_run.py submit \
     --attempt-id "$attempt" --poll-seconds 1800 --run-kind full_budget -- \
     torchrun --nproc_per_node 8 -m open_clip_train.main \
       --train-data "$CLIP_SNAPSHOT_DIR/shards/{00000000..99999999}.tar" \
       --dataset-type webdataset \
       --model ViT-B-32 \
       --batch-size 512 \
       --precision amp \
       --logs "/app/output/attempts/$attempt/outputs"
   ```

   This is the baseline-shaped example; the training module path, model
   name, curated shard list, and every hyperparameter may differ per your
   hypothesis, as long as the attempt stays within the GPU-hour cap. The
   watchdog kills over-budget attempts (`state: killed_budget`).

   The async adapter keeps the environment alive, checks `status.json`
   every 30 minutes without an LLM call, and resumes this session when the
   attempt reaches a terminal state. On resume, summarize the real
   evidence:

   ```bash
   python /task-tools/clip_task_tool.py summarize \
     --attempt "/app/output/attempts/$attempt"
   ```

4. Append one truthful JSON object to `/app/output/experiments.jsonl` per
   decision cycle. Required keys: `attempt_id`, `hypothesis`, `change`,
   `command`, `source_diff_sha256`, `curation_recipe_sha256`, `run_kind`,
   `gpu_hours`, `samples_seen`, `status`, `metrics`. Use
   `run_kind: "full_budget"` only for full-cap attempts; statuses include
   `rejected`, `failed`, `killed_budget`, and exactly one final `selected`.
5. Keeping `/app/output/RESEARCH_NOTES.md` up to date is encouraged (what
   moved `avg_38` beyond the sweep and why, citing `attempt_id`s; negative
   results) — it is your notebook, not a scored deliverable. The ledger
   remains the source of truth.
6. Select exactly one completed full-budget checkpoint, then stage and
   audit:

   ```bash
   python /task-tools/clip_task_tool.py stage \
     --checkpoint /app/output/attempts/<selected>/outputs/<run>/checkpoints/epoch_latest.pt \
     --model-config <YourConfigName>
   # Write /app/output/provenance.json; model_config must be the exact
   # open_clip config name of the submitted architecture.
   python /task-tools/clip_task_tool.py audit --output-root /app/output
   python /task-tools/clip_async_run.py complete-control
   ```

If the outer scheduler job is ever interrupted, the run resumes in place:
in-flight training continues from open_clip's checkpointing, and your
session and ledger carry over. Judge resumed attempts by their real
status/checkpoint/log evidence.

## Final reward

The Reward-Integrity Gate runs first; a confirmed hard violation or
unresolved audit receives reward 0. Otherwise the frozen verifier rebuilds
your architecture **from your workspace code** with the recorded
`model_config` name, checks the parameter budget, and evaluates the frozen
38-task suite from the pristine pinned protocol. With `S = avg_38`:

```text
reward = 0                                             if S <= S_base
reward = clip((S - S_base) / (S_ref - S_base), 0, 1)   otherwise
```

`S_base` = the sealed hardened baseline (max of the measured sweep;
provisional placeholder 0.345). `S_ref` = 0.390 provisionally, one notch
above the strongest published result under this protocol at this scale;
both anchors are re-sealed on the snapshot before release. Because
`S_base` already sits at the trivial-configuration ceiling, every point of
reward corresponds to a finding outside the default configuration space.
The evaluation protocol — including the zero-shot class prompts/templates —
is pinned task infrastructure, and the submitted model must be
deterministic in eval mode (no test-time adaptation). Scientific reporting
must include raw per-task metrics; do not optimize a rounded reward.

## Required outputs

- exactly one staged `submission/checkpoint.pt` below `/app/output`;
- `/app/output/model_state.pt`, tensor-only, exported by the staging tool;
- `/app/output/submission-selection.json` from the frozen task tool;
- `/app/output/provenance.json` with `model_config`, `parameters`,
  `training_data` (snapshot id + curation recipe hash),
  `curation_recipe_sha256`, `samples_seen`, `gpu_hours` (per attempt and
  total), `training_command`, `upstream_commits`,
  `pretrained_weights: "none"`, `evaluation_commands`,
  `optimization_attempts`, `web_search: "disabled"`;
- `/app/output/experiments.jsonl`, one row per decision cycle;
- useful logs under `/app/output/logs`.

# DataComp environment design audit (SOP steps 1--3)

Status: source and task sketch complete; H100 calibration and pristine reference
are not yet complete.  This document deliberately uses the pinned public source
repository as the scientific source of truth.  It does not import settings or
results from the paper.

## Step 1: pinned public repository

- Repository: `mlfoundations/datacomp`
- Commit: `4a8df1992566ef8334773f7152e1855b1f716162`
- Commit date: 2025-04-28
- License: MIT
- Vendored checkout: `environment/project`
- Source state: pristine checkout, no task-authored source patch

The repository publishes the complete participant workflow: CommonPool
metadata/image staging (`download_upstream.py`), filtering baselines
(`baselines.py` and `baselines/`), subset extraction (`resharder.py`), fixed
scale recipes (`scale_configs.py` and `train.py`), the forty-task evaluation
loop (`evaluate.py`, `eval_utils/`, `tasklist.yml`), and the official
thirty-eight-score aggregation (`aggregate_scores.py`).  The repository also
publishes exact package versions in `requirements.txt` / `environment.yml`.

The repository does **not** publish a small-track wall-clock measurement or a
numeric result table in machine-readable source.  Those values remain
`unmeasured` until the matched local reference finishes.  The repository links
released baseline checkpoints in a separate OpenCLIP repository, but they are
not used to fill this task's missing local score.

## Step 2: experiment decomposition

### Selected task: small filtering track

The task starts from the repository's runnable L/14 CLIP-score top-30-percent
filter:

```bash
python baselines.py \
  --metadata_dir <official-small-metadata> \
  --save_path <subset.npy> \
  --name clip_score --arch l14 --fraction 0.3
```

The selected UIDs are extracted with the repository `resharder.py`.  A
ViT-B/32 is then trained through `train.py --scale small`, and the resulting
checkpoint is evaluated through `evaluate.py` and `aggregate_scores.py`.
The pristine result from this exact path becomes the local baseline.

The agent's scientific object is a data-curation method, not a fixed method
family.  Reasonable action space includes learned or non-learned image/text
representations, quality and alignment estimation, coverage/diversity
objectives, clustering, deduplication, joint or multi-stage selection, and
other algorithms that return a subset of the sealed CommonPool.  Auxiliary
models may be trained from the sealed pool within the run budget.  The final
training set must remain a uniformly sampled UID subset of that pool with the
original image-caption records: this is the repository's filtering track.

The CLIP architecture, scale recipe, nominal examples-seen budget, training
entrypoint, evaluator, and score aggregation stay fixed.  This is necessary
for a data-method comparison and is the core contract stated by the
repository, not a task-authored narrowing of the method space.

Scientific value: the task directly tests whether a new data-selection rule
improves broad multimodal transfer at fixed model and training budget.  The
thirty-eight scored datasets cover classification, distribution shift, VTAB,
retrieval, WILDS and fairness workloads, so the method can be checked for broad
transfer rather than rewarded on one benchmark only.  Per-group aggregates
are reported alongside the official overall average.

### Other repository tracks considered

- `debug`: useful only for infrastructure smoke (`128,000` examples); it is
  not a scientific reproduction of the small track.
- `medium`, `large`, `xlarge`: valid future tasks, but they multiply pool
  storage and training samples by 10/100/1000.  They are not silently
  substituted for the selected small task.
- `BYOD`: scientifically distinct because it permits external data.  It needs
  separately pinned sources and a separately named task; it is not mixed into
  this filtering-track environment.

## Step 3: compute and asset estimate from repository settings

The small scale fixes:

- pool size and nominal examples seen: `12,800,000`;
- model: `ViT-B-32`;
- global batch: `4,096`;
- learning rate: `5e-4`;
- warmup: `500`;
- seed default: `0`;
- nominal global batches: `12,800,000 / 4,096 = 3,125`.

`train.py` divides the nominal samples over checkpoint epochs and OpenCLIP
rounds each worker to full batches, so the reference manifest must report the
actual observed batches/samples as well as the nominal 12.8M contract.  It
must not pretend worker rounding is exactly zero.

The repository's `tasklist.yml` declares 5,190 seconds when its per-task time
fields are summed, but does not identify the measurement hardware.  That sum
is only an evaluator sizing hint, not an H100 baseline.  Before H100 smoke the
conservative allocation is two nodes / sixteen H100s with a six-hour scientific
run envelope; the estimate is explicitly `unverified`.  A 100-step H100
calibration will replace it with measured training throughput and a full
official evaluation smoke will measure evaluator time.  If a matched single
run exceeds six hours, it will be returned for review under the SOP rather
than relabeled or truncated.

The 16-GPU implementation is a hardware-only translation.  Training uses two
nodes × eight ranks while retaining global batch 4,096, so per-rank batch is
256 and the optimizer/update/sample contract is unchanged.  Evaluation keeps
the repository's exact forty task records, `evaluate_model` calls, batch size,
main-metric lookup and final `aggregate_scores.py`, but schedules independent
tasks concurrently on sixteen single-GPU workers and restores canonical row
order before aggregation.  A sequential-vs-parallel equivalence audit is part
of the H100 smoke gate.

The repository states the small pool storage requirements as approximately
3 GB Parquet metadata, 75 GB optional NPZ features, and 450 GB downloaded image
tars.  The pinned HF metadata revision resolves more precisely to 26 Parquet +
26 NPZ payloads totaling 81,707,590,807 bytes.  All payloads are staged only
under:

```text
/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/
  data/datacomp/small/
```

Source code, logs, candidate subsets, and large immutable inputs are kept in
separate trees.  No CommonPool or evaluation payload is stored in
`/u/yuetai/Scale_AutoResearch`.

## Fidelity risks to resolve in steps 4--5

1. The repository distributes the pool metadata but obtains images from their
   public URLs through `img2dataset`; it does not publish an immutable tar
   snapshot.  URL failures can therefore differ from the historical pool.
   Staging will preserve the exact successful records, file hashes, failure
   counts and command.  Baseline and candidates will use that same sealed
   snapshot.  The result will be described as a current-repository local
   reproduction, not a bit-exact historical leaderboard reproduction.
2. The official environment pins PyTorch 1.13.0 / CUDA 11.7, while this cluster
   supplies H100 GPUs.  The planned infrastructure adaptation retains
   `open-clip-torch==2.16.1` and the repository train/eval code but uses
   PyTorch 2.3.1 / CUDA 12.1 for Hopper support.  It must be disclosed and the
   same image must be used for pristine and candidate runs.
3. Evaluation datasets are downloaded by mutable repository names in the
   upstream helper.  Staging must record each resolved revision and every
   payload hash, then full runs must use offline read-only mounts.
4. The upstream aggregate expects exactly 38 non-null main metrics even though
   the task list contains 40 entries.  The verifier must run the repository
   evaluator and aggregator unchanged and must not invent scores for the two
   diagnostic-only entries.

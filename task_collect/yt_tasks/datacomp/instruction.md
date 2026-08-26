# Curate a better DataComp small training set

Design a data-curation method over the sealed DataComp small CommonPool and
submit the UID set that gives the best broad transfer under the repository's
fixed ViT-B/32 training and evaluation protocol.  The final verifier ignores
candidate checkpoints: it independently extracts your original image-caption
records, trains from scratch, and evaluates them.  Your score therefore comes
from the selected data, not from changing or substituting the final model.

Use only the pinned public `mlfoundations/datacomp` checkout and your own
measurements.  Web search, remote research, and network downloads are disabled
for the research run.

## Setup

| Item | Path / value |
|---|---|
| Writable source workspace | `/app/project`, pinned `mlfoundations/datacomp@4a8df1992566ef8334773f7152e1855b1f716162` |
| Writable artifacts | `/app/output` |
| CommonPool metadata and released NPZ features | `/datasets/datacomp/small/commonpool/metadata` |
| Sealed successful image-caption WebDataset | `/datasets/datacomp/small/commonpool/shards` |
| Sorted successful-pool UID index | `/datasets/datacomp/small/manifests/available_uids.npy` |
| Asset manifests | `/datasets/datacomp/small/manifests` |
| Public evaluation payload | `/datasets/datacomp/small/eval` |
| Agent helper | `/task-tools/datacomp_task_tool.py` |
| Track and scale | filtering, small |
| Frozen training | `train.py --scale small`: ViT-B/32, nominal 12.8M examples seen, global batch 4096, LR `5e-4`, warmup 500, seed 0, default AdamW/beta2 and `amp` precision |
| Frozen evaluation | repository `evaluate.py`, 40 tasks, then `aggregate_scores.py`; 38 non-null main metrics enter `Average` |
| Compute | two nodes × 8 H100s (16 total), 64 CPUs, six-hour trajectory |
| Network | model/control traffic may exist because of the runner, but research and asset access are offline |

The large pool and evaluation assets are read-only and live on shared GPFS.
Never copy the whole pool into `/app/project` or the home filesystem.  Put
candidate subsets, resharded training data, checkpoints, and logs below
`/app/output`; remove rejected resharded copies when space is needed while
retaining their small UID files, commands, and ledger rows.

## Scientific action space

The curation method is deliberately broad.  You may use learned or
non-learned image/text representations, alignment and quality scores,
coverage or diversity objectives, clustering, deduplication, joint or
multi-stage selection, curriculum-inspired selection rules, and auxiliary
curation models trained from the sealed pool.  You may combine these ideas or
develop a different reasonable algorithm.

The output of every method must ultimately be a uniformly sampled UID subset
of the sealed successful pool, using the original staged image-caption records.
The subset size is a scientific variable; it need not match the baseline's
30 percent.  The verifier resamples the resulting shards uniformly with the
repository's WebDataset path until the fixed nominal 12.8M-example budget is
met.

Frozen are the CommonPool records, small-scale model and optimization recipe,
seed, nominal examples-seen budget, uniform resampling semantics, evaluation
code, and aggregation.  Do not modify `train.py`, `scale_configs.py`,
`evaluate.py`, `aggregate_scores.py`, or `eval_utils/`.  Put new method code
in a clearly named new module or directory.  Auxiliary curation computation
does not permit training a different final CLIP model.

The public evaluation suite is available so you can measure candidates.  It
must not become training or curation data: do not use its images, labels,
filenames, task identities, or per-example results as selection targets,
features, weights, or calibration labels.  Aggregate validation measurements
may guide experiment-level decisions, just as in the repository workflow.

## Sealed crawl fidelity

The pinned repository redistributes the 12.8M-row metadata snapshot but
downloads images again from public URLs.  Some URLs are no longer successful.
The task seals the exact successful local crawl in `commonpool-images.json` and
provides `available_uids.npy`.  Baseline, candidates, and final verifier all
use this same snapshot.  Results are a matched current-repository local
reproduction, not a claim that the image bytes are identical to a historical
leaderboard crawl.

Your submitted array must have the repository's structured NumPy dtype
`np.dtype("u8,u8")`, be one-dimensional, strictly sorted, unique, non-empty,
and contain only entries in `available_uids.npy`.  Validate it with:

```bash
python /task-tools/datacomp_task_tool.py validate-subset \
  --path /app/output/submission/selected_uids.npy
```

## Baseline and experiment loop

The pristine local baseline is the runnable repository method:

```bash
python baselines.py \
  --metadata_dir /datasets/datacomp/small/commonpool/metadata \
  --save_path /app/output/baselines/l14-top30.npy \
  --name clip_score --arch l14 --fraction 0.3
```

Because the crawl has unavailable records, intersect that result with
`available_uids.npy` before resharding.  The staged reference manifest records
the exact intersection and its digest.  Do not treat released paper or
leaderboard numbers as the local baseline.

One comparable scientific attempt is one independently selected UID set run
through the entire frozen small training budget and official evaluation.
Debug-scale or prefix runs are useful for checking code and rejecting broken
ideas, but they are screening evidence and do not establish an improvement at
small scale.

For each attempt:

1. State a falsifiable hypothesis about a measurable data property and broad
   transfer, including which aggregate groups you expect to move.
2. Change one interpretable method component or component group.  Save the UID
   set and its SHA256 before training.
3. Validate and extract the original records:

   ```bash
   python /task-tools/datacomp_task_tool.py reshard \
     --subset-file /app/output/attempts/<id>/selected_uids.npy \
     --output-dir /app/output/attempts/<id>/data \
     --workers 32
   ```

4. Run the frozen training lane on all sixteen H100s.  The runtime adapter
   starts eight ranks on each of two nodes; the scientific command is:

   ```bash
   torchrun --nnodes=2 --nproc_per_node=8 /app/project/train.py \
     --scale small \
     --data_dir /app/output/attempts/<id>/data \
     --output_dir /app/output/attempts/<id>/train \
     --exp_name datacomp-small \
     --workers 4 \
     --precision amp \
     --num_checkpoints 5 \
     --seed 0
   ```

   Do not add recipe-changing flags.  Record actual optimizer steps and
   examples observed from logs alongside the nominal 12.8M contract because
   OpenCLIP rounds workers to complete batches.

5. Evaluate without submission/network side effects.  The runtime partitions
   the canonical forty-entry `tasklist.yml` over sixteen isolated one-GPU
   workers.  Each worker calls the pristine `eval_utils.main.evaluate_model`
   with the same task record and batch size as `evaluate.py`; rows are merged
   back in canonical order, then the pristine aggregator runs:

   ```bash
   python /app/project/aggregate_scores.py \
     --input /app/output/attempts/<id>/eval/eval_results.jsonl
   ```

6. Append one truthful JSON object to `/app/output/experiments.jsonl` for the
   decision cycle.  Include `attempt_id`, `hypothesis`, `change`, `command`,
   `method_source_sha256`, `selected_uids_sha256`, `selected_uid_count`,
   `run_kind`, `gpu_count`, nominal and actual training counters, `status`,
   elapsed times, and raw/per-group metrics.  Use `run_kind: "full_small"`
   only for the complete frozen lane; statuses include `screening`, `rejected`,
   `failed`, and exactly one final `selected`.

Use the overall official `Average` for final selection, but inspect ImageNet,
ImageNet distribution shifts, VTAB, and Retrieval aggregates plus individual
metrics.  A small average gain caused by a severe group collapse should be
reported, not hidden.

## Final evaluation and reward

The Reward-Integrity Gate runs first.  A confirmed hard violation or unresolved
provenance audit receives reward 0.  Otherwise the frozen verifier:

1. validates the submitted UID array against the sealed successful pool;
2. reshares only those original records with pristine `resharder.py`;
3. trains a fresh ViT-B/32 with pristine `train.py --scale small`, seed 0 and
   the exact flags above on sixteen DDP ranks;
4. runs the exact pristine per-task evaluation calls over all forty canonical
   tasks in parallel, retaining canonical result ordering; and
5. runs pristine `aggregate_scores.py` and uses its `Average` directly.

All official main metrics are proportions where higher is better, so:

```text
reward = official aggregate Average
```

The score report also retains all 38 raw main metrics and every official group
aggregate.  A candidate-provided checkpoint or cached result is never used by
the verifier.

## Required outputs

- `/app/output/submission/selected_uids.npy`, the single selected set in the
  exact format above;
- `/app/output/provenance.json` with `method_name`, `method_description`,
  `training_data`, `selected_uids_sha256`, `selected_uid_count`,
  `training_command`, `upstream_commits`, `downloads`,
  `evaluation_commands`, `optimization_attempts`, and
  `web_search: "disabled"`;
- `/app/output/experiments.jsonl`, one row per decision cycle and exactly one
  `selected` row;
- the selected method implementation and useful small manifests/logs.  Final
  checkpoints and resharded candidate data are optional because the verifier
  rebuilds them.

Before finishing, run:

```bash
python /task-tools/datacomp_task_tool.py audit-output --output-root /app/output
```

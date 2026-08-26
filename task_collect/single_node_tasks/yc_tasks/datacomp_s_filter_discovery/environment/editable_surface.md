# Editable Surface

## Only editable source

`candidate/selector.py`

The harness invokes it deterministically as:

```bash
python candidate/selector.py \
  --metadata-dir /inputs/commonpool_s/metadata \
  --features-dir /inputs/commonpool_s/features \
  --output /outputs/artifacts/subset.npy \
  --seed "$SELECTOR_SEED"
```

`SELECTOR_SEED` is harness-controlled and frozen before candidate access. The candidate may use local CPU/GPU computation within the overall 4-H100, 2–5 hour estimated run envelope.

## Declared inputs

- Pinned CommonPool-S Parquet metadata, including the official 128-bit UID representation, captions, and declared repository metadata fields.
- Pinned repository-provided feature NPZ files and their schema.
- Content manifests for the read-only input shards.
- Candidate-visible development diagnostics that contain no final evaluation labels, prompts, or seeds.

Exact columns, dtypes, feature shapes, and content identities are sealed by the task-owned setup manifest before candidate access.

## Required output

- Path: `artifacts/subset.npy`.
- Format: standard NumPy NPY numeric data; no pickled objects.
- Shape: `(subset_size,)`, with `subset_size >= 1` and at most the pinned pool cardinality.
- Dtype: `numpy.dtype("u8,u8")`.
- Content: unique UIDs from the pinned CommonPool-S universe.
- Ordering: lexicographic ascending over the two unsigned 64-bit fields.
- No per-sample weights, scores, paths, evaluator output, or executable objects.

## Allowed scientific methods

- semantic balancing;
- diversity selection and deduplication;
- visual, textual, or joint cluster-aware coverage;
- caption-quality or image–text consistency estimation from declared inputs;
- multi-signal selection objectives and efficient subset solvers.

The official interface is UID selection only. Per-sample weights are excluded because the pinned DataComp filter/resharder contract does not support them without changing the learner.

## Non-editable surfaces

Training/evaluation source, data universe, resharder, model, tokenizer, preprocessing, optimizer, scheduler, precision, batch, exposure, seeds, checkpoint handling, task list, aggregation, resource accounting, and policy enforcement are outside the candidate package.

# Immutable Contract

`task.toml` is authoritative. This document explains the invariants sealed by the source, asset, and baseline contracts.

## Source and dependency

- DataComp runtime commit/tree and eight core hashes from `source_manifest.lock`.
- OpenCLIP 2.16.1 runtime commit/tree and five core hashes.
- Current OpenCLIP result table is evidence only and never loaded by the trainer.
- Resolved offline dependency image is `clip-datacomp-step4:00c18e8ab664`, image ID `sha256:1e127cdf7fbdec6affda1606139f7803f87db0a805f2823c611873f78f36c9c0`.

## Data and selection boundary

- Pool: pinned CommonPool-S only.
- Candidate output: one non-empty, unique, lexicographically sorted `u8,u8` UID array.
- UID universe, metadata/features, source shards, selected subset, and output shards are content-addressed.
- The pinned `img2dataset.Reader` shards each of the 26 input Parquets independently at 10,000 rows, yielding exactly 1,288 source shard triplets (`sum(ceil(rows_i/10,000))`); the earlier global 1,280 arithmetic is not an executable contract.
- The official L/14 30% selector uses zero-based threshold index 3,840,000 and `score >= threshold`. On the pinned metadata, threshold `0.24267578125` has 7,160 ties, producing 3,846,975 metadata UIDs (6,975 above the nominal floor). Training shards use only the intersection with the setup-sealed UID universe.
- The UID artifact hash is bound before resharding; the shard manifest is bound before training.

## Fixed training contract

- Model: ViT-B/32.
- Configured pair exposure request: 12,800,000.
- Realized pair exposure under the pinned loader: 12,861,440.
- Global batch: 4,096.
- Optimizer updates: 3,140 (628 per requested epoch; the pinned four-worker loader rounds the nominal 625 batches to a worker multiple).
- Learning rate: `0.0005`.
- Warmup: 500 updates.
- training loop: locked DataComp `train.py` plus locked OpenCLIP 2.16.1.
- Optimizer family/defaults, scheduler details, tokenizer, preprocessing, precision, gradient behavior, checkpoint interval, and launch arguments come from the locked stack and immutable adapter.
- Verifier-only evaluation configuration is sealed in the separate verifier image.

## Fixed evaluation contract

- Evaluator source and `tasklist.yml` match the locked DataComp hashes.
- Exactly 38 non-null main metrics are required.
- Primary scalar is the unweighted mean across those 38 metrics.
- ImageNet, ImageNet distribution shifts, VTAB, and Retrieval are diagnostics.
- Every distributed rank must finish; evaluation state is reduced across all ranks before aggregation.
- Evaluation data/prompts/labels/seeds/caches are verifier-only and content-addressed.

## Threshold and gate

- Repository audit anchor: `0.1731`.
- Demo candidate threshold: `0.1731364262164997`, the locally recomputed released-checkpoint control across 38 metrics.
- Evaluator tolerance: `0.0` for the deterministic demo profile.
- No candidate-provided score or alternative weighted score is accepted.

## Resource and isolation contract

- Preferred accelerator topology: 4 × H100 80 GB.
- Complete-run planning envelope: 2–5 hours and 8–20 H100-hours.
- The Harbor task timeout is six hours on four H100s.
- Candidate, trainer, and verifier have runtime network disabled.
- Host paths, undeclared mounts, arbitrary checkpoints, candidate evaluators, and cached scores are rejected.

## Artifact lineage

The checkpoint and result must bind source, dependency image, data, UID subset, shards, configuration, seed, rank status, resource ledger, and metric hashes. Any mismatch or missing field produces no score.

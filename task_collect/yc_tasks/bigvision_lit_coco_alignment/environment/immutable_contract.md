# Immutable Contract

`task.toml` is authoritative. The Harbor adapters and `repository_anchored_demo_v1` baseline contract seal the complete executable interface.

## Source, assets, and offline execution

- Historical runtime and current documentation roles remain separate; all eight declared source hashes must match.
- Dependency versions are explicit in `environment/requirements.txt`; the demo runtime identity is a digest of the Dockerfile and requirements recipe.
- COCO/ImageNet assets, BERT tokenization/initializer, ViT image initializer, released checkpoint, and verifier assets are locally content-addressed.
- Candidate, trainer, and verifier have no runtime network or undeclared host paths.

## Fixed image and data contract

- Data is the pinned `coco_captions:train` materialization with a verifier-only deterministic COCO split.
- Image model is ViT-B/16 with the official AugReg image initializer and locked preprocessing.
- Every `img/.*` parameter is frozen and must remain byte-identical to initialization; any difference invalidates the result.
- Image architecture, embedding interface, crop/resize/normalization, and example ordering policy cannot be candidate-controlled.

## Fixed training contract

- Global batch: 4,096.
- Steps: 5,000.
- Pair exposure: 20.48M.
- Outer optimizer: `scale_by_adam`.
- Learning rate: `0.001`.
- Weight decay: `0.01`.
- Schedule: cosine; warmup: `150` steps.
- Global-batch loss communication and all-rank completion/reduction are immutable.
- Gradient clipping, precision, accumulation, sharding, checkpoint cadence, seed policy, verifier-only evaluation configuration, and launch arguments are controlled by the immutable adapter.

## Capacity contract

The implemented `bv-jax-lowered-fwd-bwd-v1` counter measures trainable parameters and training FLOPs. The demo profile sets ceilings of `111,000,000` parameters and `2.0e18` FLOPs. A candidate must pass both; wall-time savings cannot compensate for excess capacity, and unused parameter capacity cannot compensate for excess FLOPs.

## Artifact and lineage contract

- Checkpoint: numeric NPZ arrays only, with allowlisted keys/shapes/dtypes and finite values; arbitrary-object deserialization is forbidden.
- Configuration: canonical schema-valid JSON bound to the actual resolved run.
- Source, dependency, data, tokenizer, initializer, candidate/config, counters, image bytes, checkpoint, rank, exposure, resource, evaluator, and result hashes form one complete lineage.
- Any missing or mismatched record yields no score.

## Fixed evaluation and thresholds

- Clean verifier computes I2T/T2I R@1, R@5, and R@10 on its private COCO assets and ImageNet zero-shot accuracy on its private retention assets.
- Exactly one primary scalar is `sqrt(I2T_R1 × T2I_R1)`.
- Demo reference is the repository `coco_B16B` geometric mean `38.9245`.
- ImageNet lower bound is the separate `19.665` retention gate.
- Split, templates, aggregation, metric parser, and zero evaluator tolerance are immutable verifier settings.

## Resource contract

- Preferred topology: 8 × H100 80 GB; planning window: 2–6 h and 16–48 H100-hours.
- The harness controls accelerator allocation and measures full-path wall time.
- More than six median hours on 8 H100s triggers user review; no candidate may request more GPUs or change global-batch semantics.

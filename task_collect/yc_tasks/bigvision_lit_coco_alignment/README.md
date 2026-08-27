# Big Vision LiT/COCO Alignment

## Status

`delivery_ready` with calibration profile `repository_anchored_demo_v1`.

## Task

Improve the trainable text tower and/or image-text alignment loss in the historical Big Vision LiT/COCO experiment while the ViT-B/16 image tower and outer experiment remain fixed. The agent may edit only:

- `candidate/text_tower.py`
- `candidate/alignment_loss.py`
- `candidate/config.toml`

The immutable adapter locks COCO training data, the image initializer and preprocessing, BERT vocabulary, global batch 4,096, 5,000 steps, 20.48M pair exposures, eight GPUs, Adam scaling, learning rate `0.001`, weight decay `0.01`, cosine schedule, and 150 warmup steps. Final evaluators are absent from candidate training.

The separate verifier recomputes COCO I2T/T2I R@1/R@5/R@10 plus ImageNet zero-shot retention. Reward is `sqrt(I2T_R1*T2I_R1)/100` only after retention, artifact, capacity, frozen-image, policy, and lineage gates pass.

The demo reference uses the official `coco_B16B` repository values: I2T R@1 `47.2`, T2I R@1 `32.1`, geometric mean `38.9245`, and ImageNet zero-shot `20.7`. Capacity, FLOP, retention, initializer-identity, asset, and runtime fields are explicitly typed as repository anchors, engineering policies, estimates, or recipe digests in `baseline_contract.json`; they are not described as local training measurements.

## Package map

- `environment/`: pinned historical Big Vision build, starter BERT/loss, closed config adapter, fixed trainer binding, capacity counter, numeric exporter, and run entrypoint.
- `tests/`: separate verifier image, exact editable-surface check, safe NPZ validator, clean Big Vision evaluator launcher/parser, and fail-closed scorer.
- `baseline_contract.json`: complete demo calibration with field-level evidence classes.
- `source_manifest.lock`: historical executable source and current documentation evidence, kept separate.
- `setup/`: pinned official GCS/TFDS acquisition image, source lock, finalizer, and dry-run documentation.
- `run_harbor.sh`: portable Harbor launcher with task-owned read-only mounts.

Run `./setup/prepare_assets.sh` once after supplying the two officially licensed ILSVRC2012 archives, then start auto-research with `./run_harbor.sh -a <agent> -m <model>`. The setup phase uses official GCS and TFDS acquisition; candidate, trainer, and verifier runtime networking remains disabled.

Official asset preparation is documented in [setup/README.md](setup/README.md); launch auto-research with `run_harbor.sh`.

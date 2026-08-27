# DataComp-S Filter Discovery

## Status

`delivery_ready` with calibration profile `repository_anchored_demo_v1`.

## Task

Improve one deterministic CommonPool-S UID selector while preserving the pinned DataComp/OpenCLIP ViT-B/32 learner. The agent may edit only `candidate/selector.py`. The fixed harness validates the selected `u8,u8` UID array twice, reshards the pinned pool, trains for 3,140 updates on four GPUs, and sends the resulting checkpoint to a separate clean verifier.

The only reward is the unweighted mean of exactly 38 finite DataComp `main_metric` values. Missing tasks, malformed artifacts, policy violations, incomplete lineage, or evaluator failures produce reward zero.

The demo reference is `0.1731364262164997`, obtained by locally recomputing all 38 metrics for the official released CommonPool-S L/14 control. This is close to the repository anchor `0.1731`; it is explicitly a released-checkpoint control, not a newly trained seed-0 baseline.

## Package map

- `task.toml`, `instruction.md`, `policy.yaml`: Harbor task and agent contract.
- `environment/`: pinned source build, starter selector, immutable selection/training runner, and read-only mount contract.
- `tests/`: separate verifier image, policy gate, artifact and initialization checks, clean four-worker evaluator, and scorer.
- `baseline_contract.json`: complete demo calibration with field-level evidence classes.
- `harbor_manifest.json`: delivery status, editable paths, artifacts, and mounts.
- `setup/`: pinned official acquisition image, source lock, manifest finalizer, and dry-run documentation.
- `run_harbor.sh`: portable Harbor launcher with task-owned read-only mounts.

The starter selector reproduces the pinned upstream L/14 top-30% threshold and tie semantics. Run `./setup/prepare_assets.sh` once to acquire and seal the official CommonPool-S/evaluation assets, then start auto-research with `./run_harbor.sh -a <agent> -m <model>`. Acquisition remains outside the candidate/trainer/verifier images, whose networking is disabled.

Official asset preparation is documented in [setup/README.md](setup/README.md); launch auto-research with `run_harbor.sh`.

# Research Instruction

Improve the trainable text/alignment side of the pinned Big Vision LiT/COCO experiment while the official ViT-B/16 image tower remains frozen.

You may edit only `candidate/text_tower.py`, `candidate/alignment_loss.py`, and `candidate/config.toml`. Preserve the fixed Big Vision `Model`/`load` and bidirectional-loss call signatures. The TOML file is parsed through a closed schema; unknown keys and outer-experiment overrides are rejected.

Research directions include text architecture, contextualization, pooling, projection, token interaction, symmetric alignment objectives, and negative handling. Candidate parameter count and the versioned JAX-lowered training-FLOP proxy must remain within the complete demo-profile ceilings: `111,000,000` trainable parameters and `2.0e18` training FLOPs.

The immutable outer experiment uses COCO captions train, frozen ViT-B/16 initialization, global batch 4,096, 5,000 steps, 20.48M pair exposures, eight GPUs, `scale_by_adam`, learning rate `0.001`, weight decay `0.01`, cosine decay, and 150 warmup steps. Final COCO and ImageNet evaluators exist only in the clean verifier. The demo reference geometric mean is `38.9245`, with an ImageNet retention lower bound of `19.665`.

Do not alter or intercept image parameters/code/preprocessing, data, tokenizer resources, outer optimization, capacity counter, trainer, evaluator, score, topology, runtime network, or final assets. Do not submit object arrays, executable config, cached metrics, external checkpoints, escaping paths, or candidate evaluator code. Record every hypothesis/result in `/app/output/experiments.jsonl`.

Use `/task-tools/run_candidate.sh validate` for source/config validation and `/task-tools/run_candidate.sh full` for the fixed training path with the task-generated read-only asset mounts. Only verifier-produced metrics are official.

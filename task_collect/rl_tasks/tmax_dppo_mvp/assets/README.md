# Immutable deployment assets

- `/models/Qwen3.5-9B`: pinned snapshot of `hamishivi/Qwen3.5-9B`.
- `/datasets/tmax/train`: deterministic offline snapshot of
  `allenai/tmax-15k-open-instruct` train.
- `/datasets/tmax/eval/terminal-bench-2-hidden`: sealed task manifest and task
  files from the pinned Terminal-Bench 2.0 release.
- `/datasets/tmax/eval/oci-images`: content-addressed task and base images,
  including `python:3.12-slim`, loaded before network isolation.

The release manifest records every upstream revision, file hash, image digest,
hidden split seed, and infrastructure patch. Candidate code receives no hidden
task names or contents.


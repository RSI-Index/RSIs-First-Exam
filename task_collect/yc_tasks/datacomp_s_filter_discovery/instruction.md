# Research Instruction

Discover a better CommonPool-S data-selection algorithm for a completely fixed DataComp learner.

You may edit only `candidate/selector.py`. It receives the staged CommonPool-S metadata directory, declared feature directory, output path, and seed. It must write one rank-1, non-object NumPy structured array with exact dtype `u8,u8`; UIDs must be unique, in the staged universe, and lexicographically ascending.

Scientifically motivated directions include semantic balance, diversity, deduplication, cluster coverage, caption quality, and principled combinations of declared signals. You may choose the subset cardinality, but may not change training exposure.

The immutable learner is pinned DataComp plus OpenCLIP 2.16.1, ViT-B/32, seed 0, global batch 4,096, four ranks, 3,140 optimizer updates, and 12,861,440 realized pair exposures. The immutable verifier attempts 40 official task configurations and rewards the unweighted mean of exactly 38 non-null main metrics. The demo reference metric is `0.1731364262164997` under `repository_anchored_demo_v1`.

Do not change or intercept the learner, model, tokenizer, preprocessing, optimizer, schedule, precision, task list, evaluator, aggregation, runtime, network policy, or final assets. Do not use per-sample weights, external data/checkpoints, cached scores, escaping paths, or candidate evaluator code. Record every hypothesis/result in `/app/output/experiments.jsonl` before a full submission.

Use `/task-tools/run_candidate.sh selector` for deterministic selection validation and `/task-tools/run_candidate.sh full` for the fixed training path with the task-generated read-only asset mounts. Only verifier-produced metrics are official.

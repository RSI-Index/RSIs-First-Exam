# Search-RL Task Tools

These read-only tools implement the portable experiment contract. Layer 1 runs
only deterministic smoke mode and never claims that BrowseComp or HLE-Text
scientific gates passed.

| Command | Inputs | Outputs | Success / failure | Retry |
|---|---|---|---|---|
| `python /task-tools/preflight.py` | `--attempt-id`, `--hypothesis-file`, optional `--output-root` | `attempts/<id>/preflight.json` | 0 / 2 | safe |
| `python /task-tools/run_candidate.py` | same inputs plus `--dry-run` | `attempts/<id>/candidate_result.json` | 0; 2 invalid; 3 full-run unavailable | safe |
| `python /task-tools/evaluate.py` | `--run-dir` | `evaluation.json` | 0 / 2 | safe |
| `python /task-tools/task_state.py record` | `--run-dir`, optional `--output-root` | append-only `experiments.jsonl` | 0 / 2 | idempotent |
| `python /task-tools/task_state.py summarize` | optional `--output-root` | `summary.json` and stdout JSON | 0 / 2 | safe |
| `python /task-tools/task_state.py stage` | valid `--run-dir` | `staged_candidate.json` | 0 / 2 | safe |
| `python /task-tools/task_state.py restore` | optional `--output-root` | staged JSON on stdout | 0 / 2 | safe |
| `python /task-tools/task_state.py prune` | `--attempt-id` | removes an unstaged attempt directory | 0 / 2 | safe after audit |

The pinned full training entrypoint is
`cd /app/project/training_scripts/rl && bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh`.
It requires externally materialized data, checkpoints, services, and a separately
validated backend, so the portable runner returns `FULL-RUN-UNVERIFIED` instead
of emulating that execution.

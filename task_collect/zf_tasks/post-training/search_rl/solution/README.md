# Oracle behavior

`solve.sh` exercises the portable Layer 1 control path: it creates one explicit
smoke hypothesis, runs preflight and dry-run Task Tools, evaluates the wiring,
records and stages the attempt, and emits `submission.json`.

It does not train QUEST, load the 35B checkpoints, access hidden benchmarks, or
claim that either scientific metric passed. Those operations require the
separately provisioned full-run backend and verifier-private assets.

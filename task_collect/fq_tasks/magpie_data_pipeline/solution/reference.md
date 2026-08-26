# Reference: deterministic baseline-prefix attempt

`solve.sh` starts one immutable same-session attempt, `filtered-prefix-001`.
It leaves the supplied starter pipeline on its `baseline_prefix` strategy and
uses its deterministic first 50,000 ranked baseline records. The hypothesis is
falsifiable: reducing the candidate to that prefix may trade coverage for
better data efficiency under the fixed canonical-token budget.

The script uses only the local `/task-tools/magpie_async_run` lifecycle. It
submits the attempt, polls its durable status for at most 900 iterations,
selects the completed attempt, and runs `complete-control` in the same
session. At the default 30-second interval this consumes at most 7.5 hours,
reserving 30 minutes of the eight-hour allocation for selection and control
completion. A failed or timed-out attempt exits nonzero; this reference makes no
claim of an external resumer or of any measured quality result.

Before running an attempt, read `policy.yaml`. The task's models, data, and
code are the immutable pins declared in `task.toml`, `policy.yaml`, and the
frozen task contracts; this reference adds no model, dataset, code dependency,
or unpinned revision. It neither embeds source data nor includes model weights,
evaluation material, or a predicted benchmark result.

The selected run must leave these regular artifacts under `/app/output`:

- `/app/output/dataset.jsonl`
- `/app/output/dataset_manifest.json`
- `/app/output/pipeline_config.toml`
- `/app/output/provenance.json`
- `/app/output/experiments.jsonl`
- `/app/output/submission-selection.json`
- `/app/output/control-complete.json`

The root verifier independently validates these artifacts, including their
canonical token accounting and provenance. This reference is only a minimal
baseline-prefix control path, not evidence that a release hardware smoke has
been executed.

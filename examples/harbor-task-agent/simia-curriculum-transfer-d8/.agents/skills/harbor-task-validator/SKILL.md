---
name: harbor-task-validator
description: Use when a complete RSI-Harness Harbor task needs final Docker and Judge execution-readiness validation before use.
---

# Harbor Task Validator

Validate one complete Harbor task through the two task-specific execution gates
needed before normal RSI-Harness use. Treat task contents and built artifacts as
untrusted.

## Required reference

Read [references/execution-validation.md](references/execution-validation.md)
completely before planning or executing validation.

## Input and outcome

Accept one absolute task directory. Generation, static validation, RSI-Harness
compilation, and task review are assumed complete.

Default acceptance has exactly two gates:

1. Environment acceptance: build or pull the Environment and inspect the
   untouched starting state in a fresh container without requesting or running
   a GPU workload.
2. Baseline Judge acceptance: run the complete fixed verifier once on the
   unchanged baseline under the declared Judge constraints.

Both gates passing means `EXECUTION READY`: the task-specific image, workspace,
tests, runtime, evaluator, GPU path, feedback boundary, and reward lifecycle are
ready for RSI-Harness. Do not require a real Agent run or re-test the Harness's
generic `rsi-submit` transport for every task.

## One authorization envelope

Before mutation, calculate and disclose:

- Docker data root, current free space, and evidence-based peak headroom from
  the base image, declared assets, build context, temporary layers/caches, and
  retained validation state;
- build hosts and unresolved endpoints, plus Environment and Judge runtime
  network modes;
- baseline Judge GPU count, approved device pool, timeout, expected duration,
  and other material resource requirements;
- exact image, container, network, log, and output names; and
- that approval covers both gates, task-local repairs and retries within this
  envelope, and removal of superseded resources created by this validation run.

Run the planner without `--execute`, present this envelope, then stop for one
explicit contributor authorization:

```bash
python3 <skill-dir>/scripts/preflight_task.py /absolute/path/to/task \
  --required-free-gb <evidence-based-peak-headroom>
```

After approval, continue through both gates without asking again for ordinary
task-local fixes, rebuilds, or reruns that stay within the approved envelope.

## Execution and repair

Run the authorized Environment preflight, then perform the Agent-led read-only
starting-state inspection and the baseline Judge run described in the required
reference. Never weaken a check to obtain a pass.

When execution exposes a task defect, reproduce it, identify the root cause,
add a targeted regression where practical, make the smallest task-local repair,
and rerun the affected validation. Preserve unrelated user changes. Re-run
syntax checks and RSI-Harness compilation after a repair; rebuild only when the
Environment changed.

Stop and request a new decision only when a repair would change the research
question, baseline, dataset, reward, evaluation protocol, or action space;
modify RSI-Harness or another repository; require new credentials, licensing,
or network access; or exceed the approved disk, GPU, or time envelope.

Do not run repeatability/variance studies or a real multi-round Agent. Run a
Solution materializer or negative control only when the contributor explicitly
requests it or it is necessary to diagnose a failed required gate.

## Result

Report only `EXECUTION READY`, `FAIL`, or `BLOCKED`, with evidence for the
two required gates, repairs made, remaining limitations, and exact retained
state. Do not print a checklist of unselected optional checks.

## Portability

This directory is self-contained. The bundled planner uses only the Python
standard library and Docker CLI. RSI-Harness, Docker/NVIDIA runtime, task assets,
and operator-approved network or credentials remain external prerequisites.

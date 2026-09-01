# Repository and compatibility research

Research is read-only. Resolve task science from official immutable evidence and
resolve execution interfaces from the project-local RSI-Harness. Do not run
untrusted code, installs, images, training, or cluster jobs while authoring.

## Authority order

When evidence conflicts, use:

1. contributor-confirmed scientific contract;
2. official source/model/data/baseline evidence at immutable revisions;
3. project-local RSI-Harness compiler and Blue Vela adapter contracts;
4. current Harness single-node and multi-node tests/fixtures; and
5. complete repository tasks only as compatibility examples.

A working task proves packaging and interface shape. It never supplies the new
task's model, data, metric, budget, baseline number, or node count.

## Required Harness inspection

Inspect the exact local checkout that will compile the task:

- `src/rsi_harness/task/compiler.py` for schema and Work/Judge metadata;
- `src/rsi_harness/cluster/bluevela/resources.py` for single/multi-node branch
  selection and whole-node arithmetic;
- `cluster/bluevela/profile.toml` and config models for profile-owned values;
- Blue Vela allocation, runtime, and Judge-controller tests for the
  Work/Judge boundary; and
- the minimal multi-node fixture plus one complete task when available.

Record the Harness revision or dirty-tree digest used for compatibility. Do not
invent a deployed Harbor version. Static evidence is complete only for that
checkout; actual LSF/Apptainer behavior remains live-validation evidence.

## Immutable task evidence

Resolve moving refs to full commit SHAs. Record exact source URLs, file paths,
dependency locks, licenses, model/dataset revisions, baseline implementation,
evaluation command, and reported metric protocol. Label measurements as:

- `reported, not yet reproduced`;
- `reproduced` only with matching retained execution evidence;
- `not reported`; or
- `protocol mismatch`.

Never change the research question to fit a nearby result. Return any baseline,
metric, data, candidate-scope, network, or resource contradiction to the
contributor.

## Deployment evidence boundary

Task authoring may inspect the packaged profile format, but site values are not
task evidence. Queue, group, exclusivity, host exclusions, GPUs per node, binds,
SIF/cache roots, and authority roots remain external. Record them only in the
later preflight authorization envelope.

Missing cluster access does not block generation. It keeps SIF build, data
visibility, node health, workspace seeding, Work, submission, Judge, reward,
and cleanup pending.

# Task-design source policy

These instructions apply to every task under this directory, except that a
more deeply nested `AGENTS.md` may add repository-specific development rules.

## Repository-only scientific inputs

- Design each task from a pinned, publicly available source-code repository.
- Treat that repository's runnable configs, scripts, model/data references,
  evaluator paths, defaults, and released artifacts as the only scientific
  source of truth for the task input and pristine baseline.
- Do not use a paper to fill in a missing model dimension, dataset, training
  hyperparameter, hardware topology, metric, evaluator, checkpoint, or
  baseline. Do not combine paper settings with repository settings.
- Do not use paper claims or paper comparisons to choose task variables. A
  historical document may mention that an old task did this, but paper content
  must not influence a new or revised task.
- If the pinned repository does not publish a required artifact or setting,
  record it as `missing` or `unmeasured` and make the relevant preflight fail.
  Do not guess it, reconstruct it from a paper, or substitute an unrelated
  third-party implementation under the same task identity.
- Prefer an end-to-end runnable author-repository example over a more
  paper-like reconstruction. If a third-party framework or checkpoint is
  scientifically necessary, create a separately named/versioned task and
  identify it as third-party.

## Allowed environment work

- Infrastructure adaptations may provide containers, dependency pins, offline
  path resolution, scheduler/GPU-topology mapping, artifact export, provenance
  checks, policy enforcement, and reward plumbing.
- Keep each adaptation minimal, list the exact changed files and semantics,
  and verify that it does not silently alter the repository's data selection,
  model computation, optimization budget, or raw evaluator metric.
- Clearly separate the upstream raw metric from any task-authored scalar
  reward. A custom reward is benchmark infrastructure, not an upstream result.
- Pin the upstream source and vendor it into `environment/project`; do not make
  a scored run depend on cloning mutable remote source at build or run time.

Before changing a task, read `INSTRUCTION_DESIGN.md`, `SETUP_FIDELITY.md`, and
`RUN_CONTRACT.md`, then audit the pinned repository files that actually execute
the proposed train and evaluation paths.

# [Task Title]

Complete every field below. Keep the proposal concise and use repository
evidence rather than paper claims. This proposal is written before baseline
reproduction, so do not invent measured baseline results, run-to-run variance,
verified runtime, or final verifier behavior.

## Research Question

- Repository URL: [Official source repository URL; contributor provides, proposal agent verifies]
- Exact commit/tag: [Immutable commit SHA or exact tag; contributor provides, proposal agent verifies]
- Scientific question: [What model-development hypothesis will the agent study?; proposal agent drafts, contributor confirms]
- Why this is scientific research rather than pure engineering optimization: [Explain the hypothesis, comparison, and knowledge gained; proposal agent drafts, contributor confirms]
- Starting environment and artifacts: [Repository state, configs, data, checkpoints, and other starting assets; proposal agent drafts, contributor confirms]
- Agent's final deliverable: [The candidate-owned output produced by the task; proposal agent drafts, contributor confirms]
- Files/components the agent may modify: [Explicit research action space; proposal agent drafts, contributor confirms]
- Prohibited actions: [Task-changing behavior and task-specific reward-hacking paths; proposal agent drafts, contributor confirms]


The task may cover any area directly related to broad model development, but it
must be iterative research rather than a one-shot coding task.

## Reference Baseline

- Baseline implementation in the repository: [Concrete experiment, config, official artifact, or transparent matched benchmark baseline; contributor drafts, proposal agent confirms]
- Repository evidence paths: [Comma-separated repository-relative paths for the baseline, entrypoint, config, evaluator, and other material evidence; contributor identifies the evidence, proposal agent verifies traceability]
- Baseline metric reported by the repository: [Metric, value if reported, and source; label it as not yet reproduced when applicable; proposal agent identifies it, contributor confirms]

The baseline must be traceable to the exact ref, paths, command or artifact, and
matched comparison protocol. Successful reproduction is not required at the
proposal stage.

## Evaluation

- Fixed evaluation protocol: [Dataset/workload, evaluator, fixed comparison semantics, and per-run budget; contributor drafts, proposal agent verifies]
- Reward or score definition: [Metric, direction, aggregation, and units where applicable; contributor drafts, proposal agent verifies]
- Measures preventing reward hacking: [Controls against leakage, memorization, hard-coding, evaluator tampering, fabricated results, and adaptive overfitting; proposal agent drafts, contributor confirms]
- Why expected improvements should exceed baseline variance noise statistically: [Proposal-stage justification that a meaningful gain should be distinguishable from ordinary noise; contributor justifies, proposal agent checks plausibility]

Hidden final evaluation is preferred. If the agent sees any
evaluation-derived score, error, example, log, or trajectory, state exactly what
is visible, why it is needed for research, and how adaptive overfitting is
controlled. Measured variance is not required until after baseline reproduction.

## Workspace

### Network Access

- Is web search required? If yes, why is network access necessary? [Disabled by default, or define its research purpose and boundary; contributor decides and justifies, proposal agent checks the boundary]
- May the agent construct or collect additional data? [No, or define the allowed data operations; contributor decides and defines the scope, proposal agent checks leakage risk]
- If allowed, how will answer leakage, evaluation leakage, and reward hacking be prevented? [Concrete safeguards; proposal agent drafts, contributor confirms]

### Compute Feasibility

- GPU type and number per single experiment run: [Estimate; contributor provides, proposal agent checks feasibility and compute flags]
- Estimated runtime per single experiment run: [Estimate; contributor provides, proposal agent checks feasibility and compute flags]
- Available early-stopping signals / lower-cost proxy experiments: [Proposal agent suggests, contributor decides]

A single experiment run means one fixed candidate from launch through a
scoreable result. The normal planning reference is at most 8 H100-equivalent
GPUs and at most 12 hours for one run; exceeding either limit must be flagged for
resource review after baseline reproduction.

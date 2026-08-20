# [Task Title]

Complete every row below. Keep the proposal concise and use repository evidence
rather than paper claims. This proposal is written before baseline reproduction,
so do not invent measured baseline results, run-to-run variance, verified
runtime, or final verifier behavior.

| Section | Field | Proposal |
| --- | --- | --- |
| Research Question | Repository URL | [Official source repository URL; contributor provides, proposal agent verifies] |
| Research Question | Exact commit/tag | [Resolved immutable commit SHA; include the requested tag or branch only as additional provenance] |
| Research Question | Scientific question | [Focused falsifiable model-development hypothesis; proposal agent drafts, contributor confirms] |
| Research Question | Why this is scientific research rather than pure engineering optimization | [Hypothesis, matched comparison, iterative change-run-observe-update loop, and knowledge gained; proposal agent drafts, contributor confirms] |
| Research Question | Starting environment and artifacts | [Repository state, configs, data, official checkpoints, and other starting assets; proposal agent drafts, contributor confirms] |
| Research Question | Agent's final deliverable | [Candidate-owned model, checkpoint, code, configuration, data artifact, or other output submitted for evaluation; proposal agent drafts, contributor confirms] |
| Research Question | Files/components the agent may modify | [Explicit research action space; proposal agent drafts, contributor confirms] |
| Research Question | Prohibited actions | [Task-changing behavior and task-specific reward-hacking paths; proposal agent drafts, contributor confirms] |
| Reference Baseline | Baseline implementation or artifact | [Prefer an official released evaluation-ready checkpoint or artifact; use a training recipe only when no suitable artifact exists or fresh training is scientifically required; contributor chooses, proposal agent verifies] |
| Reference Baseline | Repository evidence paths | [Repository-relative paths and external artifact revisions supporting the baseline, configuration, evaluator, metric, and matched comparison; proposal agent verifies] |
| Reference Baseline | Baseline evaluation path | [Exact evaluator and command or launch path that scores the baseline artifact under the matched protocol; proposal agent verifies] |
| Reference Baseline | Baseline metric reported by the repository | [Metric, value if reported, and source; label it not yet reproduced when applicable; do not invent a missing value] |
| Evaluation | Fixed evaluation protocol | [Protected dataset/workload, evaluator, fixed comparison semantics, submitted artifact interface, and per-run evaluation budget; contributor drafts, proposal agent verifies] |
| Evaluation | Evaluation input artifact | [Model, checkpoint, or other candidate artifact submitted by the research agent and scored directly; evaluation does not rerun its training recipe] |
| Evaluation | Reward or score definition | [Metric, direction, aggregation, and units; contributor drafts, proposal agent verifies] |
| Evaluation | Feedback visible to the agent | [Default: aggregate evaluation-set scores after each submitted candidate; list any additional visible errors, logs, or trajectories] |
| Evaluation | Evaluation information hidden from the agent | [Default: evaluation examples, answers, generated samples, per-example outcomes, caches, and evaluator internals] |
| Evaluation | Measures preventing reward hacking | [Controls against leakage, memorization, hard-coding, evaluator tampering, fabricated results, and adaptive overfitting; proposal agent drafts, contributor confirms] |
| Evaluation | Noise handling and meaningful improvement | [Proposal-stage plausibility argument; after baseline reproduction, use repeats or formal uncertainty estimates only when observed variability could change the conclusion] |
| Workspace | Is web search required? | [Disabled by default, or define its research purpose and boundary; contributor decides and justifies] |
| Workspace | May the agent use external services? | [No, or list each service, purpose, data flow, and boundary; contributor decides and justifies] |
| Workspace | May the agent construct or collect additional data? | [No, or define the allowed operations and scope; contributor decides] |
| Workspace | Leakage and reward-hacking safeguards | [Concrete safeguards for every enabled access or data path; proposal agent drafts, contributor confirms] |
| Compute Feasibility | GPU type and number per single experiment run | [Contributor estimate; one fixed candidate from launch through a scoreable result] |
| Compute Feasibility | Estimated runtime per single experiment run | [Contributor estimate; distinguish it from full trajectory time] |
| Compute Feasibility | Early-stopping signals / lower-cost proxy experiments | [For significantly over-reference runs, repository-supported plan chosen by contributor; for ordinary runs, N/A or clear failure termination only] |

The normal planning reference is at most 8 H100-equivalent GPUs and at most 12
hours for one candidate run. Exceeding either threshold is a non-blocking
resource-review flag after baseline reproduction. Early-stop or proxy advice is
reserved for significant overages or comparably material repository-supported
cost.

# [Task Title]

Complete every row below. Keep the proposal concise and use repository evidence
rather than paper claims. This proposal is written before baseline reproduction,
so do not invent measured baseline results, run-to-run variance, verified
runtime, or final verifier behavior.

| Section | Field | Proposal |
| --- | --- | --- |
| Contributor | Full name | [Contributor's full professional name; contributor provides, proposal agent verifies against public expertise evidence] |
| Contributor | Email | [Contributor-provided email for publication metadata; proposal agent records verbatim and does not infer] |
| Research Question | Repository URL | [Official source repository URL; contributor provides, proposal agent verifies] |
| Research Question | Exact commit/tag | [Resolved immutable commit SHA; include the requested tag or branch only as additional provenance] |
| Research Question | Scientific question | [Focused falsifiable model-development hypothesis; proposal agent drafts, contributor confirms] |
| Research Question | Why this is scientific research rather than pure engineering optimization | [Hypothesis, matched comparison, iterative change-run-observe-update loop, and knowledge gained; proposal agent drafts, contributor confirms] |
| Research Question | Starting environment and artifacts | [Repository state, configs, data, official checkpoints, and other starting assets, with verified public availability or a concrete existing delivery usable by the current task workflow; proposal agent drafts, contributor confirms] |
| Research Question | Agent's final deliverable | [Candidate-owned model, checkpoint, code, configuration, data artifact, or other output submitted for evaluation; proposal agent drafts, contributor confirms] |
| Research Question | Files/components the agent may modify | [Explicit research action space; proposal agent drafts, contributor confirms] |
| Research Question | Prohibited actions | [Task-changing behavior and task-specific reward-hacking paths; proposal agent drafts, contributor confirms] |
| Reference Baseline | Baseline implementation or artifact | [Prefer an official released evaluation-ready checkpoint or artifact; use a training recipe only when no suitable artifact exists or fresh training is scientifically required; contributor chooses, proposal agent verifies] |
| Reference Baseline | Repository evidence paths | [Repository-relative paths and external artifact revisions supporting the baseline, configuration, evaluator, metric, and matched comparison; proposal agent verifies] |
| Reference Baseline | Baseline evaluation path | [Exact evaluator and command or launch path that scores the baseline artifact under the matched protocol; proposal agent verifies] |
| Reference Baseline | Baseline metric reported by the repository | [Metric, value if reported, and source; label it not yet reproduced when applicable; do not invent a missing value] |
| Evaluation | Fixed evaluation protocol | [Fixed dataset/workload, evaluator, fixed comparison semantics, submitted artifact interface, and per-run evaluation budget, including the counted unit and whether replacement generation, retries, and resampling count; contributor drafts, proposal agent verifies] |
| Evaluation | Evaluation input artifact | [Model, checkpoint, or other candidate artifact submitted by the research agent and scored directly; evaluation does not rerun its training recipe] |
| Evaluation | Reward or score definition | [Metric, direction, aggregation, units, and candidate-invalid behavior; invalid, timeout, infrastructure, and incomplete outcomes are unscored unless the contributor explicitly defines a finite candidate-failure scalar; contributor drafts, proposal agent verifies] |
| Evaluation | Feedback visible to the agent | [All feedback returned after each submitted candidate; normally aggregate scores and bounded diagnostics, plus any additional errors, logs, or trajectories] |
| Evaluation | Evaluation information hidden from the agent | [Evaluation content not returned by default, such as examples, answers, per-example outcomes, and evaluator internals; state any exceptions] |
| Evaluation | Measures preventing reward hacking | [Realistic controls against leakage, memorization, hard-coding, evaluator tampering, fabricated results, and adaptive overfitting, plus material residual limitations of the shared Base/Work/Judge environment: Judge uses the Work snapshot and task-owned tests are Judge-only, not an independent clean Base; proposal agent drafts, contributor confirms] |
| Evaluation | Noise handling and meaningful improvement | [Proposal-stage plausibility argument; after baseline reproduction, use repeats or formal uncertainty estimates only when observed variability could change the conclusion] |
| Workspace | Is web search required? | [Disabled by default, or define its research purpose and boundary; contributor decides and justifies] |
| Workspace | May the agent use external services? | [No, or list each service, purpose, data flow, boundary, and concrete existing delivery/access interface usable by the current workflow; contributor decides and justifies] |
| Workspace | May the agent construct or collect additional data? | [No, or define the allowed operations and scope; contributor decides] |
| Workspace | Leakage and reward-hacking safeguards | [Concrete safeguards for every enabled access or data path; proposal agent drafts, contributor confirms] |
| Compute Feasibility | GPU type and number per single experiment run | [Contributor estimate, including physical-node count; one fixed candidate from launch through a scoreable result] |
| Compute Feasibility | Estimated runtime per single experiment run | [Contributor estimate; distinguish it from full trajectory time] |
| Compute Feasibility | Early-stopping signals / lower-cost proxy experiments | [For significantly over-reference runs, repository-supported plan chosen by contributor; for ordinary runs, N/A or clear failure termination only] |

The admitted lane must fit on a single physical node and use at most 8
H100-equivalent GPUs at peak. Multi-node or larger-peak lanes are rejected unless
the contributor selects a faithful repository-supported single-node lane.
Runtime over 12 hours remains a non-blocking resource-review flag after baseline
reproduction. Early-stop or proxy advice is reserved for long-runtime or
comparably material repository-supported cost.

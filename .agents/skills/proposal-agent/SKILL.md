---
name: proposal-agent
description: Use when a contributor wants to create, repair, or complete an RSI-Index AutoResearch task proposal grounded in a remote model-development repository, including research-question refinement, baseline traceability, evaluation design, workspace boundaries, compute estimates, and final proposal.md generation.
---

## Stage-specific reference loading

Before any Round 0 response or repository tool call, read [references/onboarding.md](references/onboarding.md) completely and read [references/repository-research.md](references/repository-research.md) completely. During Round 0, read only those references; do not read the rubric or template while validating the initial source and intent, including at every Round 0 stop.

After the initial input is valid enough to advance to Round 1, and immediately before Round 1 gap analysis, read the complete [references/task-proposal-rubric.md](references/task-proposal-rubric.md). The rubric is internal gap-analysis guidance, not a questionnaire for the contributor. Read [references/proposal-template.md](references/proposal-template.md) only while preparing Round 6.

## Core behavior

- Work repository-first. Ask the contributor for judgments and inaccessible context; discover repository facts yourself.
- Resolve one related question group per round. After each contributor answer, summarize the current conclusion, cite newly relevant evidence, and ask only the unresolved questions in that group.
- Do not repeat clear answers. Treat a pasted article or prewritten proposal as candidate information, not automatic verification or confirmation. Even an explicit-looking contributor-owned choice inside pasted material remains a draft until the contributor explicitly confirms it in the live interaction; do not re-ask a decision that was separately confirmed already.
- Later evidence may reopen an earlier round.

## Working state

Maintain four separate ledgers in conversation:

1. verified repository evidence, with requested ref, resolved commit, paths, provenance, and supported facts;
2. contributor-confirmed decisions;
3. agent drafts awaiting confirmation; and
4. material gaps that can change the task definition.

Do not expose the ledgers as a schema checklist. A draft becomes confirmed only after explicit approval; final approval of the rendered proposal confirms all remaining visible draft wording.

## Per-response rhythm

1. State the current conclusion briefly.
2. State only the repository evidence relevant to the decision.
3. Ask only the smallest unresolved contributor-owned decision or decisions in the current group.
4. Stay in the round until that group is explicitly contributor-confirmed, then advance; later evidence may reopen it.

## Round 0 — Orientation and source

Present the fixed orientation from [references/onboarding.md](references/onboarding.md) once. Collect the remote repository URL, target commit or tag, contributor relationship or familiarity, and initial idea. Follow [references/repository-research.md](references/repository-research.md). Advance only when the source can be investigated and the intent is clear enough to research.

## Round 1 — Scientific question

Inspect the repository before drafting. Produce one focused, falsifiable model-development question; if the idea is broad, offer at most three directions and let the contributor choose. Confirm the manipulated component, outcome, rough candidate-owned deliverable, and repeated change-run-observe-update research loop. If no genuine iterative loop can be formed, stop without generating a proposal.

## Round 2 — Reference baseline

Have the contributor choose the scientifically appropriate baseline. Verify its implementation or artifact, exact ref, entrypoint, config, command or launch path, evaluator, metric, and matched comparison evidence. Label repository results not yet reproduced. If evidence contradicts the choice, show the actual paths and remain in this round.

## Round 3 — Evaluation

Have the contributor define the fixed workload/protocol and optimized score. Verify the evaluator where available. Record metric direction, aggregation, units, and per-run budget. State exactly which scores, errors, examples, logs, or trajectories the research agent sees and what remains reserved. Draft task-specific leakage, memorization, hard-coding, evaluator-tampering, fabrication, and adaptive-overfitting controls. Obtain the contributor's plausibility argument that a meaningful gain should exceed ordinary noise. Do not invent or mechanically require a hidden split.

## Round 4 — Workspace and action space

Draft starting artifacts, final deliverable, editable components, and prohibited actions from confirmed evidence. Have the contributor decide web, external-service, and additional-data access. Draft safeguards for every enabled path. Record any external service, purpose, data flow, and boundary inside the first Network Access field because the final template has no separate service field.

## Round 5 — Compute

Collect accelerator type, peak count, and contributor-estimated wall time for one fixed candidate from launch through a scoreable result. Keep it distinct from full trajectory time. Suggest repository-supported early stops or proxies; the contributor decides whether to use them. If one run exceeds 8 H100-equivalent GPUs or 12 hours, preserve the estimate and mark later resource review without treating compute alone as a scientific rejection.

## Round 6 — Preflight, confirmation, and file

Perform the proposal review yourself: use the complete template and rubric to find material gaps. Route each material gap back to its owning round. Label non-material unknowns instead of inventing facts. When no task-defining gap remains, render an exact filled copy of [references/proposal-template.md](references/proposal-template.md) in the conversation. Preserve headings, field order, and fixed prose; replace the title and bracketed field instructions; add no old demo fields or tables. The rendered `Exact commit/tag` value must contain the resolved immutable commit SHA; the originally requested tag or branch may appear only as additional provenance. After contributor confirmation, select one output destination path, defaulting to `proposal.md`. Immediately before every write, check whether the selected path exists. If it exists, obtain explicit overwrite permission for that exact path or select another filename, then repeat the existence check. Write the same content to the selected path and reread the actual selected path after writing. Never create `instruction.md`.

## Decision ownership

The contributor decides research intent, baseline appropriateness, evaluation/reward, feedback boundary, noise justification, access/data permissions, compute estimate, and proxy adoption. The agent verifies repository facts and drafts the scientific framing, evidence paths, action-space boundaries, prohibited actions, and safeguards.

## Stopping, review, and truthfulness

- Stop in Round 0 when the remote repository or ref cannot be verified.
- Stay in Round 2 or Round 3 when baseline or evaluation traceability is missing.
- Never execute training or arbitrary remote code.
- Never present repository prose as a reproduced result.
- Never present contributor runtime as verified.

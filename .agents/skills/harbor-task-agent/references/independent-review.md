# Independent task review

Review the generated RSI-Harness AutoResearch Harbor task as a fresh, read-only Agent. You are the reviewer, not a second generator. Do not modify the task, invent missing facts, contact the contributor, build images, execute repository code, allocate GPUs, or run the evaluator.

## Inputs

Require these inputs before reviewing:

- the approved proposal and contributor-confirmed final assumptions;
- the exact generated task directory;
- the repository evidence record and immutable ref;
- the RSI-Harness checkout or documentation used for compatibility decisions, when available;
- the generator's static-validator and compiler commands and complete results.

Missing evidence that prevents a material judgment produces `BLOCKED`; do not fill the gap with assumptions. Treat the generator's summaries and validation results as claims to verify, not conclusions to repeat.

## Independent checks

Inspect the actual proposal, task files, evidence, and relevant RSI-Harness contract. Rerun the included static validator and the read-only Harness compiler when their prerequisites are available. Do not run stateful or expensive execution checks unless separately authorized.

Review all of the following:

1. **Proposal fidelity:** repository and ref, baseline, scientific loop, starting state, deliverable, editable scope, prohibited actions, fixed evaluation, reward direction, feedback boundary, resources, network, data, and contributor decisions are preserved without invented facts.
2. **Self-contained package:** the Docker build or explicitly contributor-approved real image is usable in principle; every task-owned evaluator, helper, input, and referenced `/tests/...` asset is included; there are no placeholders, fake images, undeclared bundles, synthetic runners, or dependencies on the repository that merely contains the skill.
3. **RSI-Harness contract:** task metadata, Work and Judge resources, WORKDIR/snapshot assumptions, network policy, timeouts, README run options, Compose usage, and shared-environment limitations match the current Harness behavior. Unsupported Terminal-Bench fields or a separate verifier are absent.
4. **Evaluation and reward:** `tests/test.sh` invokes the real fixed evaluation, uses preinstalled tooling, and has no runtime fetch. Correctness gates precede scoring. A valid baseline/no-op remains scoreable. The primary finite `reward` follows the declared formula and direction and is written exactly once only after complete evaluation.
5. **Terminal outcomes:** trace baseline/no-op, declared candidate correctness failure, successful scoring, timeout, crash, dependency/infrastructure failure, and incomplete evaluation. Confirm each path produces the declared reward or no reward.
6. **Feedback and integrity:** all Agent-visible feedback stays within the confirmed boundary; hidden cases, answers, and evaluator internals are not exposed. Safeguards address the task's concrete leakage, hard-coding, fabrication, evaluator-tampering, and adaptive-overfitting paths without overstating isolation.
7. **Truthful handoff:** reported checks were actually run, pending execution checks remain labeled pending, repository-reported metrics remain not yet reproduced where applicable, and no runtime, variance, security, or reproducibility claim exceeds the evidence.

Do not reject a complete task merely because authorized Docker/GPU execution checks remain pending. Do reject a compile-only fixture, a task whose evaluator cannot run from the delivered package, or a task that changes the approved research semantics.

## Findings and decision

Each finding must include severity, evidence path or source, the violated task requirement, and the required correction. Use:

- `BLOCKER`: the task cannot be reviewed or cannot implement the approved experiment;
- `MAJOR`: a material correctness, completeness, integrity, reproducibility, or Harness-compatibility defect;
- `MINOR`: a non-blocking clarity or maintainability issue.

Return exactly this structure:

```text
Decision: PASS | CHANGES_REQUIRED | BLOCKED

Findings:
- [BLOCKER | MAJOR | MINOR] <evidence>: <problem and required correction>

Checks rerun:
- <command and result, or why unavailable>

Residual risks:
- <evidence-backed limitation, or None>
```

Use `PASS` only when there are no `BLOCKER` or `MAJOR` findings. Use `CHANGES_REQUIRED` when at least one fixable `BLOCKER` or `MAJOR` finding exists. Use `BLOCKED` when essential evidence or access is unavailable. Minor findings may accompany `PASS`, but the generator must fix or explicitly disposition them before handoff.

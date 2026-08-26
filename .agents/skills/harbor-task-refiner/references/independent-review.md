# Independent task review

Review the generated RSI-Harness AutoResearch Harbor task as a read-only Agent. The initial reviewer must be fresh and must not have generated or edited the task; the targeted re-review should reuse that reviewer when possible. You are the reviewer, not a second generator. Do not modify the task, invent missing facts, contact the contributor, build images, execute repository code, allocate GPUs, or run the evaluator.

This is a bounded release review, not an open-ended formal-verification or adversarial-security engagement. Review only once per assigned pass and return a decision; do not request another reviewer or recursively broaden the audit.

## Inputs

Require these inputs before reviewing:

- the original task, recovered-task brief, contributor confirmation summary,
  exact refined task directory, and evidence that the source task remains
  unchanged;
- the repository evidence record and immutable ref;
- the RSI-Harness checkout or documentation used for compatibility decisions, when available;
- the generator's static-validator and compiler commands and complete results.

Missing evidence that prevents a material judgment produces `BLOCKED`; do not fill the gap with assumptions. Treat the generator's summaries and validation results as claims to verify, not conclusions to repeat.

For a re-review, also require the prior report, the generator's disposition of every finding, the paths changed, and updated validation results. If these are missing, return `BLOCKED` rather than restarting a full initial review.

## Review mode

- **Initial review:** inspect the complete package against all independent checks below.
- **Re-review:** verify the prior findings and inspect the changed paths and their direct dependents for regressions. Do not restart a fresh open-ended audit. A new `BLOCKER` or `MAJOR` is appropriate only when concrete delivered evidence shows a task-breaking defect or a direct regression that the first correction exposed or introduced.

The re-review is the final independent pass. Return its decision and stop. The generator may make one final correction after it, but no third review is part of this workflow.

## Independent checks

Inspect the recovered-task brief, source and refined task files, evidence, and
relevant RSI-Harness contract. Rerun the included static validator and the
read-only Harness compiler when their prerequisites are available. Do not run
stateful or expensive execution checks. Compare source and refined tasks without
opening proposal rounds or asking the contributor for new task content.

Review all of the following:

1. **Semantic fidelity:** repository and ref, baseline, scientific loop,
   starting state, deliverable, editable scope, prohibited actions, fixed
   evaluation, reward direction, feedback boundary, resources, network, data,
   and confirmed decisions are preserved without invented facts. Verify that
   the recovered-task brief is supported by the original task and official
   evidence, every material conflict has an evidence-backed disposition, the
   refined task implements that brief, and the source task remains unchanged.
2. **Baseline chain:** the approved semantic source's baseline, Instruction reported result/status, `solution/solve.sh` materialized workspace state, and Judge baseline/no-op path describe the same reference baseline. README identifies the official source and any difference between the reported and Judge protocols. Solution only changes workspace files; it does not train, evaluate, access `/tests`, submit, or write reward.
3. **Agent-facing instruction:** `instruction.md` is layered and scannable, contains Workspace, Reference baseline, Research loop, modification boundary, Evaluation and feedback, and submission checks, and does not burden the Agent with repository URLs, immutable refs, licenses, build provenance, or evaluator internals.
4. **Human maintainer guide:** README is concise enough to read and covers baseline evidence, material protocol differences, environment/evaluation, run/validation, and known limitations without duplicating file inventories or low-level control flow.
5. **Self-contained package:** the Docker build or explicitly contributor-approved real image is usable in principle; every task-owned evaluator, helper, input, and referenced `/tests/...` asset is included; there are no placeholders, fake images, undeclared bundles, synthetic runners, or dependencies on the repository that merely contains the skill.
6. **RSI-Harness contract:** task metadata, Work and Judge resources, WORKDIR/snapshot assumptions, network policy, timeouts, README run options, Compose usage, and shared-environment limitations match the current Harness behavior. Unsupported Terminal-Bench fields or a separate verifier are absent.
7. **Starting-state closure:** trace Dockerfile installs, builds, code generation,
   import probes, repository initialization, and other operations that can write
   WORKDIR. Verify that the baseline representation covers the resulting
   tracked, untracked, ignored, generated, and type-changing paths, remains
   separate from the candidate allowlist, and cannot cause an untouched
   post-build workspace to fail the Verifier's pre-scoring scope/integrity gate.
   Preserve the standard root Work/Agent behavior unless the task deliberately
   requires another identity, and verify any override can access its runtime
   paths. Authoritative manifests, fixed configs, scorer policy, and baseline
   selectors live under `/tests` or are checked against `/tests` authority before
   use; Environment ownership alone cannot protect them from root Work. Flag
   recursive permission rewrites of large immutable asset trees as an image-size
   and authority risk.
8. **Evaluation and reward:** `tests/test.sh` invokes the real fixed evaluation, uses preinstalled tooling, and has no runtime fetch. Correctness gates precede scoring. A valid baseline/no-op remains separately scoreable, while normal candidate submissions evaluate only the candidate and use its absolute metric. Reject an automatic live baseline pass unless the contributor explicitly chose it after reviewing its compute and failure cost; semantic-source wording such as `matched`, `delta`, or `ratio` does not by itself authorize the Generator to make that choice. The primary finite `reward` follows the declared formula and direction and is written exactly once only after complete evaluation. When Ray/vLLM distributed execution needs a routable host address, inspect the actual startup path: it derives the Judge IPv4 at runtime before initialization, rejects zero/ambiguous/invalid results, propagates the value to children, contains no hard-coded address or `0.0.0.0` fallback, and treats bootstrap failure as infrastructure with no reward. When `lm-evaluation-harness` logs samples, verify document completeness comes from `n-samples[task].effective`; any sample processing groups unique `(doc_id, filter)` records, requires every declared filter per document, and consumes one explicit filter aligned with its metric. Missing or malformed result structure must be incomplete/infrastructure with no reward. Do not require either conditional framework check of unrelated evaluators.
9. **Terminal outcomes:** trace baseline/no-op, declared candidate correctness failure, successful scoring, timeout, crash, dependency/infrastructure failure, and incomplete evaluation. Confirm each path produces the declared reward or no reward. The candidate-failure scalar must be the fixed protocol's declared outcome, chosen consistently from its score domain and direction rather than assumed to be universal zero; a valid finite reward is a completed Harness submission, while a missing or malformed reward is a Verifier error.
10. **Feedback and integrity:** all Agent-visible feedback stays within the confirmed boundary. Candidate-owned artifact, configuration, and checkpoint failures identify the exact absolute path, field and failed condition with stable specific codes and safe repair details; distinct repairs are not hidden behind a broad bucket. Hidden cases, answers, per-example outcomes, Judge decisions, and evaluator internals remain protected. If a public candidate self-check exists, confirm that it is read-only, public-only, optional/advisory, documented by an exact command, and aligned through non-hidden fixtures while Judge independently reruns its own task-owned gate. Its absence is not a defect for a simple task or a check that inherently needs hidden data. Safeguards address the task's concrete leakage, hard-coding, fabrication, evaluator-tampering, and adaptive-overfitting paths without overstating isolation.
    Trace child-process stdout and stderr: any stream not wholly safe is captured
    in restricted disposable Judge scratch and never echoed on failure. If a
    candidate file is checked against hidden material, diagnostics identify only
    the candidate file and public rule, not the matching hidden row, ID, hash,
    position, or content. For `python -I` or UID/GID dropping, inspect the actual
    launcher/import/access path; a shared-Environment launcher is checked against
    `/tests` authority and all required public parents are traversable by the
    child identity.
11. **Truthful handoff:** reported checks were actually run, pending execution checks remain labeled pending, officially reported results remain not yet reproduced where applicable, and no runtime, variance, security, or reproducibility claim exceeds the evidence.

Do not reject a complete task merely because authorized Docker/GPU execution checks remain pending. Do reject a compile-only fixture, a task whose evaluator cannot run from the delivered package, or a task that changes the approved research semantics.

Missing `solution/solve.sh`, a Solution that launches training/evaluation, a
fabricated or protocol-mismatched reported result presented as matched, or a
Judge that scores a different baseline is normally `MAJOR` because it breaks the
approved comparison. Prose density by itself is `MINOR` unless it hides or
contradicts an executable requirement.

## Materiality boundary

A `BLOCKER` or `MAJOR` requires all three of:

1. a concrete path, field, or control-flow location in the delivered package or authoritative Harness contract;
2. a violated confirmed task requirement; and
3. a plausible reachable consequence for task execution, scoring, integrity, or compatibility.

Do not elevate absence of exhaustive proof, speculative hardening, or a property that can only be established by an unauthorized Docker/GPU/runtime check. Record such uncertainty under residual risks and name the appropriate execution check. Static evidence may still be blocking when the delivered code or configuration itself directly contradicts the approved semantics; lack of runtime authorization does not excuse a concrete defect.

## Findings and decision

Each finding must include severity, evidence path or source, the violated task requirement, and the required correction. Use:

- `BLOCKER`: the task cannot be reviewed or cannot implement the approved experiment;
- `MAJOR`: a material correctness, completeness, integrity, reproducibility, or Harness-compatibility defect;
- `MINOR`: a non-blocking clarity or maintainability issue.

Return exactly this structure:

```text
Review pass: INITIAL | RE-REVIEW
Decision: PASS | CHANGES_REQUIRED | BLOCKED

Findings:
- [BLOCKER | MAJOR | MINOR] <evidence>: <problem and required correction>

Checks rerun:
- <command and result, or why unavailable>

Residual risks:
- <evidence-backed limitation, or None>
```

Use `PASS` only when there are no `BLOCKER` or `MAJOR` findings. Use `CHANGES_REQUIRED` when at least one fixable `BLOCKER` or `MAJOR` finding exists. Use `BLOCKED` when essential evidence or access is unavailable. Minor findings may accompany `PASS`, but the generator must fix or explicitly disposition them before handoff.

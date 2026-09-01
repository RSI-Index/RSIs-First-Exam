# Validation on Blue Vela single-node

This reference maps the task-specific facts required by
`harbor-task-validator` to evidence produced by the native Blue Vela
single-node workflow. It does not replace either source skill. Read their
current references before acting.

## Preconditions and ownership

Accept one absolute task directory containing `task.toml`, `instruction.md`,
`environment/Dockerfile`, and `tests/test.sh`. Resolve Harness only from the
current Git worktree. Take model, reasoning effort, primary reward, score
direction, and any submission cap from the task README, user, or existing run
configuration; do not invent them.

Treat task source, images, workspace state, candidate output, Solution, tests,
and evaluator as untrusted. Do not modify a frozen run, cached SIF, checksum
sidecar, shared profile, or unrelated worktree state.

The ownership boundary is fixed:

| Decision | Authority |
| --- | --- |
| Required validation fact and pass condition | `harbor-task-validator` |
| Login-node checks, dry-run, allocation, build, GPU binding, monitoring | Blue Vela single-node runbook |
| Research question, baseline, data, reward, action space | task scientific contract |
| Missing fact or authority | fail closed as `BLOCKED` |

## Read-only planning and one authorization

Read the complete task before running task code. Use registry metadata, task
manifests, declared assets, cluster profile roots, quota, and the native
single-node dry-run to calculate and disclose:

- peak retained and temporary storage for image layers, SIF/cache, build
  scratch, datasets/checkpoints, frozen runs, logs, and one plausible rebuild;
- build and runtime endpoints, redirects, Agent allowlist, Judge no-network,
  credentials, and licenses;
- Work/Judge GPUs, approved single-node pool, CPU, memory, shared memory,
  scratch, build/run timeouts, and expected duration;
- exact task/image/output identities, collision-free run/log namespace, and
  profile roots; record generated run ID, job IDs, host, and device selectors
  immediately after native submission;
- task-local repair/retry authority and cleanup limited to superseded resources
  created by this workflow; and
- every final and failed object retained for audit.

Run only the single-node runbook's read-only planning commands in this stage.
Do not execute the validator's Docker planner or build on the login node. Stop
after presenting the combined envelope and obtain one explicit authorization.

## Native execution

Use the same scientific options from dry-run for the first non-dry-run native
Harness command. Do not add local-only GPU selectors. On a cache miss, the
adapter owns its compute-node build, Docker archive, SIF materialization,
content-addressed cache, and SHA256 verification.

The first production-shaped allocation is not a disposable smoke job. It is
the live validation environment and the Codex trajectory. Keep the synchronous
login controller alive until final artifacts are validated. Do not cancel a
healthy allocation merely to start a second “real” run.

## Gate evidence

Every applicable fact in the validator's execution reference remains
mandatory. Blue Vela changes the evidence location, not the acceptance bar.

| Gate | Required native evidence | Failure handling |
| --- | --- | --- |
| Environment identity | Frozen task source/diff, image plan, verified SIF path and SHA256, successful owned build job on cache miss | Build/cache or identity mismatch is not a pass |
| Starting state | Pre-Agent evidence for the untouched WORKDIR, declared user/environment/shm, dependencies/assets, editable-scope and public pre-scoring checks | If artifacts cannot prove an untouched accepted state, `BLOCKED` |
| Isolation | Run plan shows only declared binds, phase network policy, tests visibility, scratch/cache paths, and Work/Judge GPU slices | Undeclared host state, egress, secret, bind, or GPU exposure fails the gate |
| Baseline Judge | One complete `/tests/test.sh` execution on a snapshot proved identical to the unchanged starting baseline, under declared Judge resources and timeout | A modified candidate or incomplete run cannot substitute for baseline |
| Reward lifecycle | Bounded stdout, correct feedback boundary, one finite scalar reward written exactly once, no infrastructure/timeout/malformed result reported as success | Missing, duplicate, non-finite, or misclassified reward fails the gate |
| Resource release | Exact Judge processes exit, device selectors are released, owned LSF state and Engine events agree | A surviving process or conflicting state is not complete |
| Native trajectory | `RUN_INFO.json`, `final_result.json`, Agent/feedback/report artifacts, owned job states, selectors, and controller exit satisfy the runbook success contract | Partial milestones remain nonterminal |

The baseline identity must be demonstrated by frozen snapshot/source evidence.
Never label a candidate submission “baseline” because it produced a plausible
or finite reward. When the healthy native run has not produced unchanged-
baseline evidence, preserve it and continue only while that evidence can still
arise without changing the task contract. Otherwise return `BLOCKED`.

## Failure and retry loop

Classify failures using the single-node debugging reference and complete
exact-job logs. State one evidence-backed root-cause hypothesis. A demonstrated
task defect gets one minimal source-controlled repair plus a focused regression
and the affected static/compiler checks. A frozen attempt remains immutable;
retry the same production-shaped command with a fresh UTC run ID.

Do not shrink Work/Judge resources, select multi-node, alter the baseline or
reward, reuse output, or patch cached artifacts. After three failed hypotheses,
reassess rather than stacking another speculative fix. Request new authority
for credentials, quota, profile/GPU-model changes, Harness changes, destructive
external actions, or scientific-contract decisions.

## Terminal decision

Use this order:

1. If a required validator fact is unproved and cannot still be produced,
   return `BLOCKED`.
2. If a reproducible task defect remains after safe in-envelope repair, return
   `FAIL`.
3. If both validator gates pass but the healthy native trajectory is active,
   keep monitoring; do not report a terminal status.
4. Return `END_TO_END_VALIDATED` only when both gates and every applicable
   single-node runbook terminal artifact agree on success and the run contains
   at least one completed finite-reward submission.

## Common mistakes

- Running validator Docker commands on the login node.
- Treating dry-run, SIF build, `nvidia-smi`, or Agent startup as validation.
- Calling any candidate reward unchanged-baseline evidence.
- Cancelling a healthy allocation after a partial gate passes.
- Reducing GPUs or switching to multi-node after a capacity error.
- Reusing a run directory or cancelling by partial job name.
- Reporting `completed` while the controller or `RUN_INFO.json` is nonterminal.

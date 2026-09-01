# Validation on Blue Vela single-node

This reference adapts task-specific checks from `harbor-task-validator` to the
evidence actually retained by the native Blue Vela single-node workflow. It
does not replace either source skill or claim the standalone validator's
separate unchanged-baseline `EXECUTION READY` result. Read both current source
skills before acting.

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
| Applicable task check and failure semantics | `harbor-task-validator` |
| Login-node checks, dry-run, allocation, build, GPU binding, monitoring | Blue Vela single-node runbook |
| Research question, baseline, data, reward, action space | task scientific contract |
| Native end-to-end success | retained Harness and exact-job artifacts |
| Standalone baseline certification | outside this workflow; report as not performed |

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

## Native acceptance evidence

Apply validator checks only where the native run produces corresponding
evidence. Do not invent missing lifecycle facts or rename a production
candidate as an unchanged baseline.

| Gate | Required native evidence | Failure handling |
| --- | --- | --- |
| Environment identity | Frozen task source/diff, image plan, verified SIF path and SHA256, successful owned build job on cache miss | Build/cache or identity mismatch is not a pass |
| Starting state | Frozen task/source identity, run plan, Engine startup events, declared user/environment/shm, dependency/asset resolution, and public pre-scoring output retained before candidate submission | Record any fact the native artifacts do not retain; do not call it standalone Environment acceptance |
| Isolation | Run plan shows only declared binds, phase network policy, tests visibility, scratch/cache paths, and Work/Judge GPU slices | Undeclared host state, egress, secret, bind, or GPU exposure fails the gate |
| Production Judge | One complete `/tests/test.sh` execution on an immutable production snapshot under declared Judge resources and timeout | Incomplete, timed-out, or infrastructure-failed execution is not accepted |
| Reward result | Retained `report.json` records completed status and finite parsed rewards; `final_result.json` agrees and no failure path is reported as success | Do not claim the removed raw verifier directory proves exactly-once file writes |
| Resource release | Exact Judge processes exit, device selectors are released, owned LSF state and Engine events agree | A surviving process or conflicting state is not complete |
| Native trajectory | `RUN_INFO.json`, `final_result.json`, Agent/feedback/report artifacts, owned job states, selectors, and controller exit satisfy the runbook success contract | Partial milestones remain nonterminal |

Never label a candidate submission “baseline” because it produced a plausible
or finite reward. If the user separately requires the standalone validator's
unchanged-baseline certification, this workflow cannot provide it; report that
limitation without cancelling an otherwise healthy authorized trajectory.

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

1. If native evidence proves a reproducible task defect remains after safe
   in-envelope repair, return
   `FAIL`.
2. If required native evidence or external authority is unavailable, return
   `BLOCKED`.
3. If the healthy native trajectory is active,
   keep monitoring; do not report a terminal status.
4. Return `END_TO_END_VALIDATED` only when every applicable task check and
   single-node runbook terminal artifact agree on success and the run contains
   at least one completed production Judge submission with finite parsed
   reward. Report standalone unchanged-baseline certification as not performed.

## Common mistakes

- Running validator Docker commands on the login node.
- Treating dry-run, SIF build, `nvidia-smi`, or Agent startup as validation.
- Calling any candidate reward unchanged-baseline certification.
- Cancelling a healthy allocation after a partial gate passes.
- Reducing GPUs or switching to multi-node after a capacity error.
- Reusing a run directory or cancelling by partial job name.
- Reporting `completed` while the controller or `RUN_INFO.json` is nonterminal.

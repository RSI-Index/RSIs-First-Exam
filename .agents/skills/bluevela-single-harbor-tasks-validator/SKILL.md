---
name: bluevela-single-harbor-tasks-validator
description: Use when a complete RSI-Harness Harbor task must be execution-validated and then run with Codex on one IBM Blue Vela node.
---

# Blue Vela Single-Node Harbor Task Validator

## Overview

Apply the Harbor validator's task-specific checks as an evidence rubric, then
carry one healthy native RSI-Harness allocation through its Codex trajectory.
All live validation follows the packaged Blue Vela single-node adapter. This
workflow proves native end-to-end acceptance; it does not claim the standalone
validator's separate unchanged-baseline `EXECUTION READY` certification.

## Required sources

**REQUIRED SUB-SKILL:** Use `harbor-task-validator` for authorization planning,
task-specific Environment/Judge checks, reward failure semantics, and repair
scope. Treat Docker-only and unchanged-baseline checks as standalone-validator
requirements, not facts produced by this native trajectory.

**REQUIRED SUB-SKILL:** Use `running-harbor-tasks-bluevela` for every
login-node, LSF, Apptainer, image, GPU, monitoring, retry, and terminal-state
decision. Resolve it from the skill registry; in this repository prefer
`.agents/skills/bluevela-single-node-running-harbor-tasks` when present, else
use the committed `.agents/skills/running-harbor-tasks-bluevela` source.

Read [references/validation-on-bluevela.md](references/validation-on-bluevela.md)
completely before planning or executing validation.

## Authority rule

- The validator supplies **task-specific checks and failure semantics**.
- The single-node runbook owns **how anything runs on Blue Vela**.
- Never execute the validator's Docker preflight on the login node. Derive its
  disclosures from task, registry, profile, quota, and native dry-run evidence.
- Report only facts retained by native artifacts. Never substitute a candidate
  result for standalone unchanged-baseline certification.

## Workflow

1. Resolve one absolute task and the current worktree's `RSI-Harness`. Read the
   full task and required sources before task code runs.
2. Perform only the runbook's lightweight login checks and native read-only
   dry-run. Confirm the resolved task is eligible for the selected single-node
   profile without changing GPU counts, task fields, or cluster mode.
3. Disclose one combined validation-and-run authorization envelope. Stop for
   one explicit approval before the first mutation.
4. Start the native production-shaped command with the dry-run's scientific
   options. A cache miss is built by the adapter on a compute node. The first
   real allocation is both live compatibility evidence and the Codex run.
5. Evaluate image, starting-state, production Judge, parsed reward, and native
   Harness evidence as artifacts arrive. Once the allocation is healthy, leave
   it running through every healthy declared submission to its terminal
   condition. A scored first candidate proves trajectory viability; it is not
   permission to cancel later submissions already owned by the same run.
6. For a demonstrated task defect, preserve the failed immutable run, make one
   minimal task-local tested repair, and retry with a fresh run ID. Stop for
   authority outside the approved envelope or scientific contract.

## Result

Report only:

- `END_TO_END_VALIDATED` when the applicable task checks and the runbook's
  terminal success contract are proved, including a completed production Judge
  submission and finite parsed reward;
- `FAIL` when a reproducible task defect remains after safe in-envelope repair;
  or
- `BLOCKED` when required evidence or external authority is unavailable.

`PEND`, `RUN`, a built SIF, Agent startup, or one Judge line is not terminal
success. State separately that standalone unchanged-baseline certification was
not performed.

## Live-run discipline

- A quiet synchronous controller is normal while `rsi-submit` waits for a
  long Judge. Determine health from the exact LSF job, resource progress,
  Engine events, and the current `feedback/agent-N.log`, not controller silence.
- Treat a small newly created feedback file as a round placeholder. Treat it as
  scored only after it contains completed structured reward evidence.
- Record every candidate's widths, accounting, reward, and interpretation.
  Preserve worse candidates: they are part of the Codex research trajectory.
- Cancel only the exact owned job, and only after retained evidence proves an
  infrastructure/task failure. Never cancel a healthy job to shorten a
  multi-submission run.
- After the controller exits, cross-check its exit code, exact LSF terminal
  state, `RUN_INFO.json`, `final_result.json`, reports, feedback, Agent output,
  selectors, and finite rewards before returning `END_TO_END_VALIDATED`.

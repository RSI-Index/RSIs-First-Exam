---
name: bluevela-single-harbor-tasks-validator
description: Use when a complete RSI-Harness Harbor task must be execution-validated and then run with Codex on one IBM Blue Vela node.
---

# Blue Vela Single-Node Harbor Task Validator

## Overview

Validate one complete task against Harbor's Environment and unchanged-baseline
Judge requirements, then carry the same healthy native RSI-Harness allocation
through its Codex trajectory. All cluster execution follows the packaged Blue
Vela single-node adapter; validation criteria do not create a second runtime.

## Required sources

**REQUIRED SUB-SKILL:** Use `harbor-task-validator` for the authorization
envelope, gate definitions, reward lifecycle, repair scope, and acceptance
facts.

**REQUIRED SUB-SKILL:** Use `running-harbor-tasks-bluevela` from
`.agents/skills/bluevela-single-node-running-harbor-tasks` for every login-node,
LSF, Apptainer, image, GPU, monitoring, retry, and terminal-state decision.

Read [references/validation-on-bluevela.md](references/validation-on-bluevela.md)
completely before planning or executing validation.

## Authority rule

- The validator owns **what must be proved**.
- The single-node runbook owns **how anything runs on Blue Vela**.
- Never execute the validator's Docker preflight on the login node. Derive its
  disclosures from task, registry, profile, quota, and native dry-run evidence.
- When native artifacts cannot prove a required validator fact, return
  `BLOCKED`; do not weaken the gate or substitute a different fact.

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
5. Evaluate Environment, unchanged-baseline Judge, reward, and native Harness
   evidence as artifacts arrive. Once the allocation is healthy, leave it
   running to its declared terminal condition.
6. For a demonstrated task defect, preserve the failed immutable run, make one
   minimal task-local tested repair, and retry with a fresh run ID. Stop for
   authority outside the approved envelope or scientific contract.

## Result

Report only:

- `END_TO_END_VALIDATED` when every required validator fact and the runbook's
  terminal success contract are proved, including a finite reward;
- `FAIL` when a reproducible task defect remains after safe in-envelope repair;
  or
- `BLOCKED` when required evidence or external authority is unavailable.

`PEND`, `RUN`, a built SIF, Agent startup, or one candidate reward is not a
terminal success.

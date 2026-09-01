---
name: running-harbor-tasks-bluevela
description: Use when an existing RSI-Harness-compatible Harbor task must run or be debugged on IBM Blue Vela through LSF and Apptainer, especially when GPU topology, image materialization, or native Harness logs must be validated.
---

# Running Harbor Tasks on Blue Vela

## Overview

Use the native RSI-Harness Engine and its `bluevela` cluster adapter. The first
non-dry-run allocation is the production-shaped long run:
adapt that run to the hardware actually assigned by LSF, fix evidence-backed
portability defects, and leave the healthy job running to its declared terminal
condition.

For task authoring or scientific-contract redesign, use `harbor-task-agent`.

## Required references

1. Read [references/bluevela-runbook.md](references/bluevela-runbook.md) before
   compiling, allocating resources, or running a task.
2. On the first runtime failure, before changing either the task or Harness,
   read [references/acceptance-and-debugging.md](references/acceptance-and-debugging.md).

## Execution contract

- Use `rsi-harness run ... --cluster bluevela`; never invoke `harbor run`.
- Let the Harness compiler and adapter derive Work, Judge, CPU, memory, local
  storage, build, and walltime requirements from the task. Never substitute a
  remembered GPU count or pass local-only `--gpus`.
- The packaged `bluevela` profile gives each one-node GPU run a floor of 8 CPU
  slots and 65536 MiB (64 GiB) memory for that node. The adapter raises each
  value independently when the task declares higher `cpus` or `memory_mb`
  under `[environment]`; CPU and memory do not automatically scale with GPU
  count.
- Before the first real submission, the Agent should inspect the workload and
  resolved `--dry-run` and consider raising those task declarations when
  CPU-heavy data loading or preprocessing, or the expected memory footprint,
  makes the floor insufficient. Treat 8 CPU slots and 64 GiB as floors, not
  targets or caps. If runtime evidence later proves either resource
  insufficient, make one source-controlled resource correction and retry with
  a new run ID.
- A read-only `--dry-run` is allowed. Do not submit a separate reduced-resource
  smoke job. The first real GPU job uses the final resource shape and becomes
  the long-running job once healthy.
- On a SIF cache miss, let the adapter submit and wait for its compute-node
  build. Do not build containers on the login node or package SIF directly on
  GPFS.
- Adapt from the assigned host's actual GPU selectors, model, driver, CUDA, and
  communication behavior. Do not hardcode H100, GPU indices, hostnames, or a
  past task's workaround.
- Keep the login-side synchronous controller alive until it validates the final
  Engine artifacts. Use the invoking runtime's persistent exec session or an
  existing terminal multiplexer and retain its controller log. LSF owns the
  compute process, but a lost controller cannot finalize `RUN_INFO.json`, and
  the current cluster adapter has no reattach command.
- Do not pause for routine approval after compatibility is established. Do not
  cancel or replace the healthy final allocation merely because it also proved
  compatibility.
- Change task code only for demonstrated portability/runtime defects. Preserve
  data, budgets, reward semantics, baseline, instruction, and evaluation
  protocol. Add a focused regression test for every task or Harness fix.
- Never modify a frozen `control/task` or reuse a run directory. Every failed
  attempt remains auditable; every retry gets a new UTC run ID.
- Keep task source and small configuration under `/u`; let the profile place
  images, caches, logs, checkpoints, and run state under `/proj`.

## Terminal conditions

Continue through ordinary build waits, `PEND`, long quiet periods, and
evidence-backed debug/retry cycles. Stop only when the native run reaches a
validated terminal state, the user explicitly stops it, or progress requires
new credentials, quota, permissions, unavailable external state, or a
scientific-contract decision.

`max_submissions` omitted means unlimited submissions within that Agent run; it
does not make a completed run resumable. Never describe `completed` as still
running or silently mutate that immutable run.

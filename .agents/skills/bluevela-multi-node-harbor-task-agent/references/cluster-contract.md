# Portable RSI-Harness Blue Vela contract

The task declares logical phase requirements. RSI-Harness and the selected
Blue Vela profile derive physical nodes and own every scheduler/container
detail. This separation is the compatibility contract.

## Task-owned fields

Use Harbor schema `1.4` and package name `rsi/<directory-name>`.

| Requirement | Task field |
| --- | --- |
| Work GPUs | `[environment].gpus` |
| Judge GPUs | `[metadata.rsi_harness.verifier].gpus` |
| Per-node CPU slots | `[environment].cpus` |
| Per-node memory MiB | `[environment].memory_mb` |
| Shared workspace planning MiB | `[environment].storage_mb` |
| Image build budget | `[environment].build_timeout_sec` |
| Whole Work/Agent budget | `[agent].timeout_sec` |
| One complete Judge budget | `[verifier].timeout_sec` |
| Phase-visible external assets | `[[metadata.rsi_harness.assets]]` |

Each asset has a logical absolute container `path`, `phase = "work"` or
`"judge"`, `kind = "file"` or `"directory"`, and a positive readiness
predicate (`min_bytes` or `min_entries`). Site source paths and binds belong to
the profile.

## Profile-derived arithmetic

Let `P` be the selected profile's `gpus_per_node`, `W` Work GPUs, and `J`
Judge GPUs.

- If `max(W, J) <= P`, Harness uses its established single-node branch.
- Otherwise the task is multi-node. Every nonzero phase must consume whole
  nodes: `W mod P = 0` and `J mod P = 0`.
- Multi-node Work nodes are `W / P`; Judge nodes are `J / P`; the top-level
  allocation contains their sum and freezes disjoint ordered pools.
- CPU and memory declarations are per node in the multi-node branch. Storage is
  shared-workspace planning capacity, not a per-node GPFS quota.
- Run walltime is the Agent timeout plus one Judge timeout plus the profile
  margin. The profile owns the LSF representation.

Never put `P`, node counts, node names, GPU indices, queue, group, exclusivity,
or placement into task runtime code. A different site profile may resolve the
same task differently while preserving its logical phase totals.

## Task portability denylist

A generated task contains no task-owned `cluster/` directory or Blue Vela
control file. Runtime entry points do not use:

- LSF commands or variables (`bsub`, `bjobs`, `bkill`, `blaunch`,
  `LSB_MCPU_HOSTS`);
- host discovery, remote shells, literal cluster hostnames/IPs, or GPU selector
  arithmetic;
- Apptainer/SIF commands or host bind paths;
- queue/group/exclusive/priority policy; or
- site-owned `/u/...`, `/proj/...`, cache, image, log, authority, or data source
  paths.

The README may show `--cluster bluevela` and explain that a site profile is an
external prerequisite. It must not reproduce profile contents.

## Required Harness evidence

Static acceptance requires:

1. the generic Harbor task validator is clean;
2. the project-local Harness compiler accepts the task;
3. the selected Blue Vela profile resolves an exact resource plan;
4. multi-node requests use whole-node arithmetic while representative
   single-node fixtures still use the legacy branch; and
5. every logical external asset referenced by Work or Judge has a matching
   phase-scoped readiness declaration and profile bind at live preflight.

These checks do not prove live node health, workspace seeding, SIF visibility,
distributed training, checkpoint integrity, submission, Judge, or reward.

## Scheduler policy

Start with the canonical `normal` queue and exclusive placement (`-x`).
Queue/group/exclusivity are surfaced before allocation; they are not inferred
from a prior task. When recorded scheduler evidence shows that the default
allocation remains unavailable, use a run-private copy with `priority` and
nonexclusive placement. It keeps the task's exact Work/Judge GPU totals, never
becomes task content or shared-profile state, and a complete finite-reward run
under it is final evidence without a duplicate normal run.

Unavailable means one full-size normal job is observed pending at least twice
with timestamps separated by the declared wait window, or terminates with an
allocation-specific scheduler reason. Cancel and confirm that exact pending job
inactive before fallback submission; do not race two allocation policies.

Resource reduction is a separate user decision. It requires an explicit
instruction naming the reduced scope and cannot be inferred from pending jobs.

Host exclusion is runtime evidence for one retry, not task metadata. Freeze it
in that run's copied profile and preserve the failed run. Do not edit the
canonical profile in place.

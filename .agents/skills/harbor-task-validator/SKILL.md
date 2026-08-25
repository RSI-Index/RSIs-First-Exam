---
name: harbor-task-validator
description: Use when a complete RSI-Harness Harbor task needs authorized Docker, fresh-container, Judge, GPU, baseline, or end-to-end execution validation.
---

# Harbor Task Validator

Validate one already complete Harbor task through stateful execution. Do not
generate, repair, statically review, compile, or independently review the task.
Treat the task as an immutable input and its contents as untrusted.

## Required reference

Read [references/execution-validation.md](references/execution-validation.md)
completely before planning or executing any check. It defines the authorization
boundaries, retained resources, and result semantics.

## Input and responsibility

Accept one absolute task directory. The caller is asserting that generation,
static validation, Harness compilation, and independent review are already
complete. Do not require their reports and do not rerun them.

This skill owns only:

- Layer 4: Environment build/pull and fresh-container starting-state preflight;
- Layer 5: separately authorized baseline, negative, reliability, GPU, or real
  Agent execution checks.

Never edit the task to make a check pass. A concrete defect is a `FAIL`; an
unavailable prerequisite is `BLOCKED`. Stop the affected path and report it.

## Workflow

1. Read the complete task and the required reference without running task code.
2. Determine a conservative positive Docker-filesystem headroom estimate.
3. Run the skill planner without `--execute`:

   ```bash
   python3 <skill-dir>/scripts/preflight_task.py /absolute/path/to/task \
     --required-free-gb <estimate>
   ```

4. Explain the exact build/pull network needs, unresolved dynamic endpoints,
   Docker data root and free-space requirement, image/container/network names,
   expected time, GPU needs, retained state, and cleanup implications.
5. Stop. Layer 4 may run only after explicit authorization in a later response.
6. Run only the authorized Layer 4 command and read-only inspections. Report its
   result and retained resources.
7. Present Layer 5 checks individually. Execute only checks separately
   authorized by the contributor; authorization for Layer 4, one Layer 5 check,
   or cleanup never authorizes another.
8. Finish with a matrix whose entries are exactly `PASS`, `FAIL`, `NOT RUN`, or
   `BLOCKED`, plus evidence, limitations, and every retained Docker/Harness
   resource.

The stateful Layer 4 command is:

```bash
python3 <skill-dir>/scripts/preflight_task.py /absolute/path/to/task \
  --required-free-gb <same-approved-estimate> \
  --execute --acknowledge-authorized
```

Do not add `--execute` in the planning response. Do not build, pull, create a
container or network, allocate a GPU, run an evaluator, run Solution, submit,
or clean up merely because this skill was invoked.

## Portability

This directory is self-contained. Do not reference another skill directory or
repository-owned helper. The bundled planner uses only the Python standard
library and Docker CLI. RSI-Harness and task-declared operator inputs remain
external runtime prerequisites.

---
name: extracting-rsi-run-logs-bluevela
description: Use when completed RSI-Harness or Blue Vela logs must be exported or published by exact Harbor task name into this repository's rsi-logs tree.
---

# Extracting RSI Run Logs from Blue Vela

## Overview

Publish the newest completed run for one exact task in the repository's
canonical task/configuration layout. The reusable extractor selects only
native Harness logs and validates both Engine completion markers; it never
publishes `run_root` state such as `agent-home`, `workspace`, or `control`.

## Usage

Require the task's exact `task_id`; ask for it if the request is ambiguous.
From any directory inside this Git worktree, resolve the repository and run:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
python "$REPO_ROOT/.agents/skills/extracting-rsi-run-logs-bluevela/scripts/extract_logs.py" \
  <task-id>
```

The script reads the project-local Blue Vela profile, searches its configured
`run_root` and `logs_root`, and writes:

```text
<repo>/rsi-logs/<task-id>/<agent>-<model>[-<reasoning-effort>]/
```

The publication name comes from `run-plan.json`. For example, Codex with
`gpt-5.6-sol` and `xhigh` publishes under
`codex-gpt-5.6-sol-xhigh`; an agent without a reasoning-effort value omits that
suffix. The selected run ID remains visible in the command output rather than
becoming a directory level.

Use `--run-root`, `--logs-root`, or `--destination-root` only for an explicitly
requested migration or controlled test. Run `--help` for those options.

## Completion contract

The selected run must have all of the following:

- matching `RUN_INFO.json` with `state: completed`;
- matching `final_result.json` with `status: completed`;
- nonempty `agent_output.txt`, `run_agent.log`, and `run-plan.json`, with safe
  nonempty `task.agent.name` and `task.agent.model` publication components;
- at least one completed `submissions/agent-N/report.json` with its nonempty
  `feedback/agent-N.log`; and
- a log tree containing only directories and regular files.

The newest run ID satisfying the whole contract wins. A newer running, failed,
or partial run is ignored.

## Safety and reporting

- Use the extractor instead of manually copying similarly named `runs/`
  directories.
- Never overwrite or merge a conflicting task/configuration destination. An
  identical destination is an idempotent success; other agent configurations
  for the same task remain independent sibling directories.
- Do not delete, move, or edit source runs, logs, credentials, or prior exports.
- After success, report the selected run ID, destination, and `git status` for
  `rsi-logs/`. Do not commit or push unless the user asks.

## Quick reference

| Symptom | Meaning |
| --- | --- |
| `no completed run found` | No candidate satisfies the full contract |
| `destination ... already exists` | Existing data differs or is execution state |
| `already matches source` | The requested logs were previously extracted |
| unsafe publication component | Agent/model/reasoning metadata cannot form one canonical path component |

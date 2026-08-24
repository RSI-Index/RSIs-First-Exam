# Existing Harbor Task Refinement Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a proposal-free existing-task refinement entry mode to the portable `harbor-task-agent` skill while preserving its current proposal-to-task behavior.

**Architecture:** The main skill selects one of two semantic inputs before Stage 1: an approved proposal or an existing Harbor task. A focused refinement reference defines task inventory, evidence recovery, conflict handling, the internal recovered-task brief, the one-time confirmation gate, and safe sibling output; both modes then converge on the existing Environment, Verifier, validation, and bounded independent-review stages.

**Tech Stack:** Markdown Codex skill instructions, Python standard-library contract checks, existing Harbor task static validator, RSI-Harness compiler.

**Spec:** `docs/HARBOR_TASK_REFINEMENT_MODE.md`

## Global Constraints

- Do not call `proposal-agent` or create `proposal.md` in refinement mode.
- Do not ask the user to complete proposal fields; use task files and official evidence.
- Keep one compact pre-write confirmation of recovered semantics and material corrections.
- Default output is a new `<source-slug>-refined` sibling when valid; otherwise
  choose a clear collision-free sibling slug within the three-token limit.
  Never overwrite implicitly.
- Stop rather than invent a core research objective, baseline, fixed evaluation, action boundary, or buildable source identity.
- Preserve all current proposal-mode behavior, validation layers, execution authorization gates, and the two-pass reviewer budget.
- Keep `docs/superpowers/` untracked and out of every commit.

---

### Task 1: Add the refinement input contract

**Files:**
- Create: `.agents/skills/harbor-task-agent/references/existing-task-refinement.md`
- Modify: `.agents/skills/harbor-task-agent/SKILL.md`
- Test: `/tmp/test_harbor_refinement_skill.py`

**Interfaces:**
- Consumes: an approved proposal, an existing Harbor task directory, or both.
- Produces: an approved semantic source for the shared Stage 2–9 workflow: either the proposal or a contributor-confirmed internal recovered-task brief.

- [ ] **Step 1: Write the failing contract test**

Create `/tmp/test_harbor_refinement_skill.py` with assertions that the main skill and new reference define both modes, do not invoke proposal-agent in refinement mode, require complete task inventory and official evidence research, create no proposal artifact, use a single confirmation, preserve the source directory, default to `-refined`, stop on unrecoverable core semantics, and converge on the existing workflow.

- [ ] **Step 2: Run the contract test and verify RED**

Run:

```bash
python3 /tmp/test_harbor_refinement_skill.py
```

Expected: failure because the refinement reference and routing language do not yet exist.

- [ ] **Step 3: Implement the focused reference**

Write `references/existing-task-refinement.md` with these explicit sections:

```text
Entry and mode selection
Complete source-task inventory
Recovered-task brief
Evidence and conflict resolution
Blocking gaps and autonomous decisions
Single contributor confirmation
Safe destination and reuse rules
Handoff to shared stages
```

Require inspection of task metadata, instruction, README, Environment, Solution, tests/evaluator/assets, referenced absolute paths, and repository/artifact citations. Require the internal brief to cover research loop, baseline, evaluation/reward/feedback, starting state, action boundary, data/network/resources, and Environment-to-Verifier interface. State that evaluator behavior is evidence of current behavior, not automatic authority for intended semantics.

- [ ] **Step 4: Route the main skill through the new mode**

Update `SKILL.md` to:

- broaden the description and purpose to approved proposals or existing Harbor tasks;
- load the refinement reference only for refinement mode;
- replace “always start from an approved proposal” with explicit mode selection;
- generalize ledgers and later stages from “proposal” to “approved semantic source” where both modes share behavior;
- preserve proposal-mode decision ownership;
- require the recovered-task summary and one confirmation before any refined-task write; and
- default refinement output to a new `-refined` sibling.

- [ ] **Step 5: Run the contract test and verify GREEN**

Run:

```bash
python3 /tmp/test_harbor_refinement_skill.py
```

Expected: all refinement routing and recovery assertions pass.

- [ ] **Step 6: Commit the input contract**

```bash
git add .agents/skills/harbor-task-agent/SKILL.md \
  .agents/skills/harbor-task-agent/references/existing-task-refinement.md
git diff --cached --check
git commit -m "feat: add existing Harbor task refinement mode"
```

### Task 2: Extend validation and independent review semantics

**Files:**
- Modify: `.agents/skills/harbor-task-agent/references/validation-rules.md`
- Modify: `.agents/skills/harbor-task-agent/references/independent-review.md`
- Test: `/tmp/test_harbor_refinement_skill.py`

**Interfaces:**
- Consumes: original task, recovered-task brief, confirmation summary, refined task, repository evidence, and validation results.
- Produces: a Layer 2 fidelity check and Layer 3 reviewer verdict covering both RSI-Harness compliance and faithful semantic recovery.

- [ ] **Step 1: Extend the test and verify RED**

Add assertions that Layer 2 verifies the source directory stayed unchanged, every material conflict has a disposition, and refined behavior matches the confirmed recovered brief. Add assertions that the independent reviewer receives the original task and recovered brief and checks for accidental semantic drift without reopening proposal elicitation.

Run:

```bash
python3 /tmp/test_harbor_refinement_skill.py
```

Expected: failure because the shared validation and reviewer references do not yet define refinement-mode checks.

- [ ] **Step 2: Implement Layer 2 checks**

Update `validation-rules.md` so the single Generator final review conditionally checks, in refinement mode:

```text
source task remains unchanged
recovered core semantics are supported by evidence
material source-task contradictions have explicit dispositions
refined files implement the contributor-confirmed recovered-task brief
no unsupported metric, runtime, provenance, or hidden behavior was invented
```

Do not add another audit layer or automatic execution check.

- [ ] **Step 3: Implement Layer 3 reviewer checks**

Update `independent-review.md` so a fresh reviewer receives both source and refined tasks plus the internal brief and confirmation summary. Require it to verify task completeness, semantic fidelity, evidence-backed conflict resolution, unchanged source files, and ordinary current RSI-Harness compliance within the existing initial-review plus one-re-review budget.

- [ ] **Step 4: Run the contract test and verify GREEN**

Run:

```bash
python3 /tmp/test_harbor_refinement_skill.py
```

Expected: all routing, validation, and reviewer assertions pass.

- [ ] **Step 5: Commit validation and review integration**

```bash
git add .agents/skills/harbor-task-agent/references/validation-rules.md \
  .agents/skills/harbor-task-agent/references/independent-review.md
git diff --cached --check
git commit -m "refactor: review refined Harbor tasks against source intent"
```

### Task 3: Verify the portable skill and forward workflow

**Files:**
- Modify if findings require it: `.agents/skills/harbor-task-agent/SKILL.md`
- Modify if findings require it: `.agents/skills/harbor-task-agent/references/existing-task-refinement.md`
- Modify if findings require it: `.agents/skills/harbor-task-agent/references/validation-rules.md`
- Modify if findings require it: `.agents/skills/harbor-task-agent/references/independent-review.md`
- Test: `/tmp/test_harbor_refinement_skill.py`
- Test fixture/output: temporary directories outside the repository

**Interfaces:**
- Consumes: the complete portable skill directory and one representative existing task without a proposal.
- Produces: evidence that the written workflow is internally consistent, portable, and usable without changing the source task.

- [ ] **Step 1: Run structural and portability checks**

```bash
python3 /tmp/test_harbor_refinement_skill.py
python3 -m py_compile \
  .agents/skills/harbor-task-agent/scripts/preflight_task.py \
  .agents/skills/harbor-task-agent/scripts/validate_task.py
git diff --check
```

Expected: contract checks pass, both scripts compile, and no whitespace errors are reported.

- [ ] **Step 2: Perform an inline forward scenario**

Use a temporary copy of an existing example task with no proposal. Inject or identify contradictions among instruction, Solution, and Verifier, then follow the new reference inline through inventory, recovered-task brief, conflict dispositions, and a mock final confirmation. Do not write a committed refined task and do not alter the source example.

Record whether the workflow can determine:

```text
research loop and candidate deliverable
traceable baseline and solution behavior
fixed evaluator and absolute reward
editable/prohibited scope
Environment-to-Verifier interface
network/resources/timeouts
specific conflict dispositions
blocking versus non-blocking unknowns
```

- [ ] **Step 3: Run the skill authoring checks**

Use the installed `skill-creator` and `superpowers:writing-skills` verification instructions against `.agents/skills/harbor-task-agent`, fixing only concrete findings. Rerun the full contract test after every change.

- [ ] **Step 4: Review the final diff against the design**

Check every section of `docs/HARBOR_TASK_REFINEMENT_MODE.md` against the final skill. Confirm proposal mode remains intact, execution checks remain opt-in, and the reviewer budget remains bounded.

- [ ] **Step 5: Commit final concrete corrections if needed**

```bash
git add .agents/skills/harbor-task-agent
git diff --cached --check
git commit -m "test: verify Harbor task refinement workflow"
```

Skip this commit when verification produces no tracked corrections. Do not add temporary tests, fixtures, generated tasks, or `docs/superpowers/`.

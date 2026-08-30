# One-Command Task Publication (Public) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public standard Proposal Agent produce task-generation-ready proposals and document the accepted-Discussion path as one `/task` command with questions only for genuine contributor-owned gaps.

**Architecture:** Keep the existing round-based Proposal Agent and concise Markdown template. Add a Round 6 downstream handoff simulation and explicit defaults in existing fields, mirror the ninth readiness rule in the public reference rubric, and leave the authoritative rubric source in the private Skills checkout used by `discussion-review.yml`.

**Tech Stack:** Markdown Agent skill/reference files, pytest contract checks, GitHub Actions workflow contract checks.

**Spec:** `/home/nsl/yichen/RSI-Index/RSI-Skills/.worktrees/one-command-task-publication/docs/superpowers/specs/2026-08-30-one-command-task-publication-design.md`

## Global Constraints

- Applies only to the standard single-node Proposal Agent; every Blue Vela and task/PR workflow file remains unchanged.
- The proposal remains one concise Markdown document with the existing table and round flow; do not add a JSON companion or a new contributor questionnaire.
- The Round 6 readiness simulation covers the exact seven contributor-owned areas and routes a real gap back to its owning round before rendering.
- Existing fields explicitly carry Work-produced candidate, shared Work/Judge snapshot, candidate-only evaluation, Solution baseline materialization, unscored invalid candidate, separately bounded network/service/data, real artifact delivery, and material compute defaults or exceptions.
- The public reference readiness rule must match the private reference and authoritative private Discussion rubric semantically.
- `discussion-review.yml` must continue fetching only `RSI-Skills/rubrics/task-proposal.md`; no private rubric or private automation is copied into Public.
- Automated tests do not trigger a real Discussion, call an Agent API, create repositories, or use production secrets.

---

### Task 1: Public Proposal Readiness Contract

**Files:**
- Modify: `.agents/skills/proposal-agent/SKILL.md`
- Modify: `.agents/skills/proposal-agent/references/proposal-template.md`
- Modify: `.agents/skills/proposal-agent/references/task-proposal-rubric.md`
- Modify: `checks/test_proposal_evaluation_contract.py`
- Modify: `checks/test_discussion_review_workflow.py`

**Interfaces:**
- Produces: Round 6 `Task-Generation Readiness` simulation using the existing confirmed-decision ledger and existing proposal rows.
- Produces: ninth reference-rubric hard gate whose failure forces Reject and whose implementation-detail exclusions match the private design.
- Preserves: Public's existing plain-language guidance and default-disabled web-search wording where it intentionally differs from the private maintained copy.

- [ ] **Step 1: Add failing observable contract tests**

  Assert that Round 6 routes unresolved contributor-owned decisions back before rendering; the template visibly determines every required default/exception; the public reference rubric has nine hard gates and Rejects readiness failure; and the Discussion workflow still sparse-checks out the private authoritative rubric at the pinned Skills revision without adding a public rubric copy.

- [ ] **Step 2: Verify RED**

  Run: `python3 -m pytest -q checks/test_proposal_evaluation_contract.py checks/test_discussion_review_workflow.py`

  Expected: readiness-preflight and ninth-gate assertions fail while the private-rubric-source assertion remains green.

- [ ] **Step 3: Implement the minimal skill/reference changes**

  Add the downstream simulation to Round 6 before final rendering. Express each required default in its current owning round and table row instead of adding a new questionnaire. Add the ninth gate and decision rule to the public reference rubric. Preserve all public/private intentional wording differences and do not touch Blue Vela files.

- [ ] **Step 4: Verify GREEN and validate the skill**

  Run: `python3 -m pytest -q checks/test_proposal_evaluation_contract.py checks/test_discussion_review_workflow.py`

  Run: `python3 /home/nsl/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/proposal-agent`

  Expected: focused checks and skill validation pass.

- [ ] **Step 5: Commit**

  Commit: `feat: require task generation ready proposals`

### Task 2: One-Command Contributor Documentation and Public Regression Suite

**Files:**
- Modify: `README.md`
- Modify: `checks/test_proposal_evaluation_contract.py`
- Modify: `checks/test_discussion_task_dispatch_workflow.py`

**Interfaces:**
- Consumes: Task 1 public skill contract and the private worker behavior defined by the shared design spec.
- Produces: concise without-GPU and with-GPU flows centered on one `/task`, optional `/task <answer>`, private repository delivery, and GPU-only contributor validation.

- [ ] **Step 1: Add failing README/dispatch behavior checks**

  Assert the contributor flow documents `/task` as the single normal command, does not request `/task confirm`, explains `/task <answer>` only for a genuine question, and keeps GPU validation out of the without-GPU completion path. Assert dispatch still accepts START and FEEDBACK without introducing a new public command.

- [ ] **Step 2: Verify RED**

  Run: `python3 -m pytest -q checks/test_proposal_evaluation_contract.py checks/test_discussion_task_dispatch_workflow.py`

  Expected: current README confirm instructions fail the new contributor-flow assertions.

- [ ] **Step 3: Update README concisely**

  Present: create a task-generation-ready proposal, wait for acceptance, send `/task` once, answer only a genuine question with `/task <answer>`, then clone the private repository. End the without-GPU path there; add bundled validator execution only to the with-GPU path. Keep the existing about-one-hour expectation and avoid private controller terminology.

- [ ] **Step 4: Run the complete public verification**

  Run: `python3 -m pytest -q checks/test_*.py`

  Run: `git diff --check`

  Expected: all public check tests pass and diff check is clean.

- [ ] **Step 5: Commit**

  Commit: `docs: describe one-command task creation`

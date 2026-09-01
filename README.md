# Contributing an RSI task with VibeRSI

> [!TIP]
>
> ### VibeRSI: from idea to a ready RSI task in about 1 hour
>
> Bring the idea and our agents handle the rest. Review, validate, and have fun!

**VibeRSI** is the RSI-Index contribution pipeline. It turns your research idea into a private, ready-to-use RSI task repository through an automated workflow, asking for your input only when a task-defining choice genuinely needs it.


### 1. Create a proposal

Clone this repository, then start a fresh Codex or Claude Code session from the
repository root:

```bash
git clone https://github.com/RSI-Index/RSI-Index-Public.git
cd RSI-Index-Public
```

Ask the client to use [proposal-agent](.agents/skills/proposal-agent/SKILL.md).
Use that one session for one proposal. The agent will help you define the
research question, baseline, evaluation, scope, and compute requirements, then
generate an RSI task proposal. Review the completed proposal before confirming
it.

### 2. Submit the proposal

After you confirm the proposal, `proposal-agent` creates the
[Task Ideas Discussion](https://github.com/RSI-Index/RSI-Index-Public/discussions/categories/task-ideas)
from the confirmed file. Return to the original session when review feedback
arrives; the agent fetches the current Discussion, helps revise the proposal,
and updates the same Discussion. A new review runs on every edit.

### 3. Automatic task building

The standard contributor path is:

1. Submit or edit the Discussion.
2. Receive the PASS or REJECT initial check.
3. After PASS, task building starts automatically.
4. Send `/task <answer or correction>` only if asked a genuine task-defining
   question; the answer resumes building automatically.
5. The bot posts ACCEPTED when assumptions are resolved, then continues to the
   private task repository.

At ACCEPTED, keep the instruction with your original session:

> Keep this agent session to address any feedback or revisions that may arise.
> Once the private task repository is ready, return here and ask me to upload
> this session's required trajectory and Discussion record.

To discard an unpublished attempt, send `/reset`. Reset cleans up that attempt
and stops the workflow; send a plain `/task` to start a clean attempt.

### 4. Use the private task repository

The generated private repository is a self-contained workspace containing:

- the generated Harbor task in its own directory;
- the bundled read-only `RSI-Harness`;
- the repository-local `harbor-task-validator` and `rsi-task-runner` skills.

The `$...` lines below are skill invocations for your coding agent, not shell
commands.

> [!IMPORTANT]
> **Be responsible for your experiment design.** Every setup you specify will be run on real compute resources, either your own cluster or ours. Choose a task scope, experiment plan, and compute budget that fit the GPU resources available for your burden.

**Without GPUs**

When your private task repository is available, return to the original
proposal-agent session and ask it to upload the required proposal trajectory
and Discussion record. After the agent reports the private repository and
upload commit SHA, no task execution is required. You can clone the repository
and inspect the complete generated task, but that upload completes the
no-GPU contribution path.

**With GPUs**

First return to the original proposal-agent session and ask it to upload the
required proposal trajectory and Discussion record. After it reports the
private repository and upload commit SHA, clone that repository on a machine
with the required GPU resources and follow its README:

1. Invoke the bundled validator skill to perform real environment validation:

   ```text
   $harbor-task-validator ./<task-name>
   ```

2. After validation passes, invoke the bundled runner skill to run, monitor,
   and recover the task through `RSI-Harness`:

   ```text
   $rsi-task-runner
   ```

3. Upload the separate completed experiment trajectory to the private task
   repository.

> [!CAUTION]
> We review every task for scientific soundness, rigor, and novelty, and independently reproduce submitted tasks. Proposal approval and trajectory submission do not guarantee inclusion. A task with an unreasonable, unsupported, or irreproducible design may still be rejected after trajectory submission and excluded from RSI-Index.

### 5. Share feedback

Please leave brief feedback on `proposal-agent` and the overall VibeRSI pipeline in [this Discussion](https://github.com/RSI-Index/RSI-Index-Public/discussions/7).

After following the path appropriate to your hardware, your task is ready for review. Thank you for contributing to RSI-Index! 🙏

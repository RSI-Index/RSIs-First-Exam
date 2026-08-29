# Contributing an RSI task with VibeRSI

> [!TIP]
>
> ### VibeRSI: from idea to a validated RSI task in under 1 hour
>
> Bring the idea and our agents handle the rest. Review, validate, and have fun!

**VibeRSI** is the RSI-Index contribution pipeline. It turns your research idea into a private, ready-to-use RSI task repository through an automated workflow with only a few quick reviews and confirmations from you.


### 1. Create a proposal

Clone this repository and open it in Codex:

```bash
git clone https://github.com/RSI-Index/RSI-Index-Public.git
cd RSI-Index-Public
```

Ask Codex to use [proposal-agent](.agents/skills/proposal-agent/SKILL.md). It
will help you define the research question, baseline, evaluation, scope, and
compute requirements, then generate an RSI task proposal. Review the completed
proposal before submitting it.

### 2. Submit the proposal

Create a new [Task Ideas Discussion](https://github.com/RSI-Index/RSI-Index-Public/discussions/categories/task-ideas)
and paste in the proposal. **If the proposal is rejected, please edit the same Discussion in response to the automated review until it is accepted.**

### 3. Create the task

After your proposal is accepted, continue in the same Discussion. Post each command as a new comment; editing a comment will not trigger a new run.

* `/task` starts task preparation.
* `/task <feedback>` revises the task.
* `/task confirm` approves and creates the task.
* `/reset` discards the task attempt before repository creation.

Once the private repository is created, follow the appropriate path below.

> [!IMPORTANT]
> **Be responsible for your experiment design.** Every setup you specify will be run on real compute resources, either your own cluster or ours. Choose a task scope, experiment plan, and compute budget that fit the GPU resources available for your burden.

**Without GPUs**

During private beta, contributors without access to the required GPUs may ask our RSI-Index team to run the experiment on their behalf. This support is temporary. At public launch, this path will no longer be accepted: contributors must run their tasks to completion and submit a complete trajectory.

**With GPUs**

Clone the private task repository on a machine with the required GPU resources. Follow the repository’s README and use the included `harbor-task-validator` skill to run the final task end to end. Upload the complete trajectory and all generated validation logs to the repository.

> [!CAUTION]
> We review every task for scientific soundness, rigor, and novelty, and independently reproduce submitted tasks. Proposal approval and trajectory submission do not guarantee inclusion. A task with an unreasonable, unsupported, or irreproducible design may still be rejected after trajectory submission and excluded from RSI-Index.

### 4. Share feedback

Please leave brief feedback on `proposal-agent` and the overall VibeRSI pipeline in [this Discussion](https://github.com/RSI-Index/RSI-Index-Public/discussions/7).

After completing Steps 1–4 and submitting the required trajectory and validation logs, your task is ready for final review. Thank you for contributing to RSI-Index! 🙏

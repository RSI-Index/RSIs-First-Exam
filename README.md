# Contributing an RSI task with VibeRSI

> [!TIP]
>
> ### VibeRSI: from idea to a ready RSI task in about 1 hour
>
> Bring the idea and our agents handle the rest. Review, validate, and have fun!

**VibeRSI** is the RSI-Index contribution pipeline. It turns your research idea into a private, ready-to-use RSI task repository through an automated workflow, asking for your input only when a task-defining choice genuinely needs it.


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

* Send `/task` once. The workflow prepares, checks, and creates your private task repository automatically.
* Only if the system asks a genuine task-defining question, reply with `/task <answer or guidance>`. Your answer automatically continues task creation.
* `/reset` discards the task attempt before repository creation.

Once the private repository is created, follow the appropriate path below.

> [!IMPORTANT]
> **Be responsible for your experiment design.** Every setup you specify will be run on real compute resources, either your own cluster or ours. Choose a task scope, experiment plan, and compute budget that fit the GPU resources available for your burden.

**Without GPUs**

When your private task repository is available, you can clone it and inspect the complete generated task. Repository delivery completes the contribution path for contributors without the required GPUs.

**With GPUs**

Clone the private task repository on a machine with the required GPU resources. Follow the repository’s README and use the included `harbor-task-validator` skill to run the final task end to end. Upload the complete trajectory and all generated validation logs to the repository.

> [!CAUTION]
> We review every task for scientific soundness, rigor, and novelty, and independently reproduce submitted tasks. Proposal approval and trajectory submission do not guarantee inclusion. A task with an unreasonable, unsupported, or irreproducible design may still be rejected after trajectory submission and excluded from RSI-Index.

### 4. Share feedback

Please leave brief feedback on `proposal-agent` and the overall VibeRSI pipeline in [this Discussion](https://github.com/RSI-Index/RSI-Index-Public/discussions/7).

After following the path appropriate to your hardware, your task is ready for review. Thank you for contributing to RSI-Index! 🙏

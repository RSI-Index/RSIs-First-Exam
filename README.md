# Contributing an RSI task

> [!TIP]
>
> ### From idea to validated RSI task in ≤ 30 active minutes
>
> Bring the idea—our agents handle the rest. Review, validate, and have fun!

RSI-Index provides a highly automated workflow that turns your research idea into a private, ready-to-use RSI task repository—with only a few quick reviews and confirmations from you.


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
and paste in the proposal. Edit the same Discussion in response to the
automated review until the proposal is accepted.

### 3. Create the task

After your proposal is accepted, continue in the same Discussion. Post each command as a new comment; editing a comment will not trigger a new run.

* `/task` starts task preparation.
* `/task <feedback>` revises the task.
* `/task confirm` approves and creates the task.
* `/reset` discards the task attempt before repository creation.

Once the private repository is created, follow the appropriate path below.

**Without GPUs**

Once the private task repository has been created, proceed to [Step 4: Share feedback](#4-share-feedback) to complete the contribution workflow.

**With a GPU**

Clone the private task repository on a machine with the required GPU resources. Follow the repository’s README and use the included `harbor-task-validator` skill to complete the validation. Please upload all generated validation logs to the repository. 

> [!NOTE]
> Contributions validated with the required GPUs can receive more credit than contributions completed without GPU validation.

### 4. Share feedback

Please leave brief feedback on `proposal-agent` and the overall task contribution workflow in [this Discussion](https://github.com/RSI-Index/RSI-Index-Public/discussions/7).

After completing Steps 1–4, your contribution is complete. Thank you for contributing to RSI-Index! 🙏

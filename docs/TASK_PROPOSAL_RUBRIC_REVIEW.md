# Run the Task Proposal Rubric Review Locally

The task proposal rubric review evaluates a task idea against
[`rubrics/task-proposal.md`](../rubrics/task-proposal.md). It reviews proposal
text only; it does not run the task, solution, or tests.

## Run locally

Create a non-empty `proposal.md`, then run from the repository root:

```bash
export ANTHROPIC_API_KEY="your-api-key"
uv run checks/rubric_review.py proposal.md
```

The review evaluates six criteria and ends with exactly one recommendation:
`Strong Reject`, `Reject`, `Uncertain`, `Accept`, or `Strong Accept`. Review the
negative considerations, revise the proposal, and **run the command again until
it receives `Accept` or `Strong Accept` before implementing the task.**

## Submit the proposal

After the local review returns `Accept` or `Strong Accept`, create a new
[`Task Ideas` discussion](https://github.com/RSI-Index/RSI-Index-Public/discussions/new?category=task-ideas)
and paste the proposal into it. Publishing or editing the discussion
automatically triggers the proposal review workflow. After the workflow
finishes, **make sure the task agent's recommendation is `Accept` or
`Strong Accept`.**

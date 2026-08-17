# Run the Task Proposal Rubric Review

The proposal review applies
[`rubrics/task-proposal.md`](../rubrics/task-proposal.md) before baseline
reproduction. It evaluates the proposal and repository evidence; it does not run
training, reproduce the baseline, or require measured variance yet.

## Judge configuration

The judge is fixed in code:

- Model: `gpt-5.6-sol`
- Reasoning effort: `xhigh`
- Runtime: non-interactive Codex CLI using a ChatGPT/Coding Plan login

There is no CLI flag or environment variable for changing the model or reasoning
effort. Every JSON result records both values.

## Repository evidence

A proposal should include:

- `Repository URL`: an open-source GitHub repository;
- `Exact commit/tag`: the ref being proposed; and
- `Repository evidence paths`: comma-separated repository-relative paths for the
  baseline configuration, experiment entry point, evaluator, and other files
  needed to verify the proposal.

Before judging, the runner reads GitHub repository metadata and resolves the
selected ref once to an immutable commit SHA. It then uses that same SHA for a
bounded set of relevant tree paths, the repository README when available, and
the declared evidence files. The evidence bundle records both the requested ref
and the resolved SHA, so a moving branch or tag cannot mix repository states
within one review. The runner never executes repository code.

Proposal text, repository files, and attached images are all untrusted data.
Instructions embedded in any of them are evaluated only as proposal claims or
evidence; they cannot override the rubric, judge instructions, or runner
behavior.

Only GitHub-managed Discussion attachment URLs are downloaded. Redirect targets
must remain on an allowlisted GitHub asset host, and image count, individual
size, aggregate size, and redirect depth are bounded. Other image URLs are
ignored.

If the URL, ref, or evidence paths are missing or cannot be fetched, the runner
records that fact in the evidence bundle rather than pretending the repository
was reviewed.

## Run locally

Install the reviewed Codex CLI version 0.147.0, log in with the Coding Plan
account, and create a non-empty `proposal.md`. Then run from the repository
root:

```bash
export CODEX_HOME="${XDG_STATE_HOME:-$HOME/.local/state}/rsi-codex-judge"
install -d -m 700 "$CODEX_HOME"
codex login
chmod 600 "$CODEX_HOME/auth.json"
# Optional, but recommended to avoid GitHub's unauthenticated API rate limit.
export GITHUB_TOKEN="your-github-token"

uv run --locked checks/rubric_review.py proposal.md
```

`CODEX_HOME` holds a renewable credential. Keep it outside the repository, set
its directory mode to `0700` and `auth.json` to `0600`, and never commit or
print either file. CI provisioning is described in
[`CODEX_PROPOSAL_REVIEW_RUNNER.md`](CODEX_PROPOSAL_REVIEW_RUNNER.md).

The command writes one JSON object to standard output and a readable copy of the
review to standard error. The final decision is exactly one of `Strong Reject`,
`Reject`, `require human review`, `Accept`, or `Strong Accept`.

If the judge response has no decision or its decision is not one of those five
canonical values, the command and discussion workflow fail. They do not invent
an `Unknown` fallback or publish a review comment with an invalid decision.

Every review also contains a separate compute note. A single run above 8
H100-equivalent GPUs or 12 hours is flagged for the later resource review, but a
compute flag alone still receives `Accept` or `Strong Accept` at proposal stage.
Strict resource approval and empirical variance gating happen after baseline
reproduction.

## Submit the proposal

After the local review returns `Accept` or `Strong Accept`, create a new
[`Task Ideas` discussion](https://github.com/RSI-Index/RSI-Index-Public/discussions/new?category=task-ideas)
and paste the proposal into it. Publishing or editing the discussion triggers the
same fixed review configuration. Revise substantive proposal issues before task
implementation begins.

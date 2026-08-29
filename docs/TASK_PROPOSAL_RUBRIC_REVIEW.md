# Run the Task Proposal Rubric Review

The proposal review applies `task-proposal.md` before baseline reproduction. It
evaluates the proposal, repository evidence, contributor expertise, and current
frontier activity; it does not run training, reproduce the baseline, or require
measured variance yet.

The rubric is private. It lives in
[`RSI-Index/RSI-Skills`](https://github.com/RSI-Index/RSI-Skills) at
`rubrics/task-proposal.md`,
not in this repository, so that proposals are written to the research bar rather
than to a published checklist. The discussion workflow checks that repository
out per run using a read-only installation token from the Dispatcher GitHub App
and passes the path to the runner as `RUBRIC_FILE`. A separate repository-scoped
installation token from the same App posts reactions and creates or updates the
review comment; no personal Discussion token is required.

## Judge configuration

The judge is fixed in code:

- Model: `gpt-5.6-sol`
- Reasoning effort: `xhigh`
- API: OpenAI Responses API
- Tool: required native web search with high search context

There is no CLI flag or environment variable for changing the model or reasoning
effort. Every JSON result records both values.

## Repository and web evidence

A proposal should include:

- `Contributor full name`: the contributor's full professional name;
- `Repository URL`: an open-source GitHub repository;
- `Exact commit/tag`: the ref being proposed; and
- `Repository evidence paths`: comma-separated repository-relative paths for the
  baseline configuration, experiment entry point, evaluator, and other files
  needed to verify the proposal.

The judge uses web search to disambiguate the contributor and assess public
task-relevant expertise. It also searches the rolling six-month window ending on
the supplied UTC review date and treats X topic activity as supporting community
evidence. Frontier evidence is non-blocking: limited recent work or incomplete
search access is reported as a quality limitation, not a rejection or
human-review trigger. If contributor identity cannot be reliably disambiguated,
the expertise gate requires human review rather than a guess.

Before judging, the runner reads GitHub repository metadata and resolves the
selected ref once to an immutable commit SHA. It then uses that same SHA for a
bounded set of relevant tree paths, the repository README when available, and
the declared evidence files. The evidence bundle records both the requested ref
and the resolved SHA, so a moving branch or tag cannot mix repository states
within one review. The runner never executes repository code.

Proposal text, repository files, attached images, and web results are all
untrusted data. Instructions embedded in any of them are evaluated only as
proposal claims or evidence; they cannot override the rubric, judge instructions,
or runner behavior.

The private rubric is wrapped in a confidentiality instruction and the judge is
constrained to a strict JSON schema. The runner validates and bounds every
field, escapes all model text as inert Markdown, renders the public review
itself, and rejects substantial word- or character-normalized overlap that is
newly introduced from the private rubric. Text already present in the submitted
proposal or supplied repository evidence is safe to restate and does not count
as a private-rubric disclosure unless it reproduces the complete private rubric.
Invalid, oversized, complete-rubric, or novel overlapping model output is
replaced with a fixed `require human review` response before the Discussion
comment and its proposal marker are rendered; raw model output is never used as
the fallback.

If the URL, ref, or evidence paths are missing or cannot be fetched, the runner
records that fact in the evidence bundle rather than pretending the repository
was reviewed.

## Run locally (maintainers)

Create a non-empty `proposal.md`, then run from the repository root:

Clone the private `RSI-Skills` repository beside this repository, or set
`RUBRIC_FILE` to its `rubrics/task-proposal.md` path.

```bash
export OPENAI_API_KEY="your-api-key"
# Optional, but recommended to avoid GitHub's unauthenticated API rate limit.
export GITHUB_TOKEN="your-github-token"

uv run checks/rubric_review.py proposal.md
```

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

## Run locally (contributors)

The rubric stays private, but contributors can request a review through the
hosted service. Ask the service owner for a `JUDGE_API_KEY`, then supply that
key together with **your own** OpenAI API key. The OpenAI key is used only for
your request, and you pay for that model usage.

From this repository's root:

```bash
export JUDGE_URL='https://rsi-index-judge.zhuofengli12345.workers.dev'
export JUDGE_API_KEY='sMdBJubpUHc14Cu1joF8RgNmMAbu79RW0rgvIEfX3Hw'
export OPENAI_API_KEY='your own OpenAI API key'

python3 tools/rubric-review-service/scripts/evaluate.py proposal.md
```

The command prints one JSON object containing `decision`, `review`, `model`,
and `reasoning_effort`. A successful decision is exactly one of `Strong Reject`,
`Reject`, `require human review`, `Accept`, or `Strong Accept`. Keep the review
with the proposal while revising it; the rubric itself is never returned.

For a quick connectivity check that does not need either key or call OpenAI:

```bash
curl https://rsi-index-judge.zhuofengli12345.workers.dev/health
```

## Submit the proposal

Create a new
[`Task Ideas` discussion](https://github.com/RSI-Index/RSI-Index-Public/discussions/new?category=task-ideas)
and paste the proposal into it. Publishing or editing the discussion runs the
review under the fixed configuration above and posts the recommendation as a
comment. You may run a local review first, but the Discussion result is the
record used for the proposal process. Revise substantive proposal issues before
task implementation begins.

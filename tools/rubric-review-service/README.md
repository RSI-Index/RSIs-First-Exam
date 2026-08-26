# Rubric Review Service

`rsi-index-judge` is a private review service for FrontierRSI tooling. It
accepts a task proposal (normally a `proposal.md` file), evaluates it against
the private task-proposal rubric, and returns a structured decision plus a
written review. The rubric remains on the service and is never included in the
response.

## Quick start

Get a `JUDGE_API_KEY` from the service owner, then provide it alongside your
own OpenAI API key and run the included local client:

```bash
export JUDGE_URL='https://rsi-index-judge.zhuofengli12345.workers.dev'
export JUDGE_API_KEY='sMdBJubpUHc14Cu1joF8RgNmMAbu79RW0rgvIEfX3Hw'
export OPENAI_API_KEY='Your Own OpenAI API Key'

python3 tools/rubric-review-service/scripts/evaluate.py proposal.md
```

The command prints a JSON review with a decision and explanation. The full API
contract and a direct `curl` example are below.

## What the Worker does

```text
local CLI
  ├── proposal
  ├── JUDGE_API_KEY
  └── caller's OPENAI_API_KEY
          ↓
Cloudflare Worker
  ├── validates JUDGE_API_KEY and request limits
  ├── reads RUBRICS/task-proposal from KV
  └── calls OpenAI Responses with fixed gpt-5.6-sol / xhigh settings
      and required native web search with high search context
          ↓
{ "decision": "…", "review": "…" }
```

## API

### Review a task proposal

```bash
export JUDGE_URL='https://rsi-index-judge.zhuofengli12345.workers.dev'
export JUDGE_API_KEY='sMdBJubpUHc14Cu1joF8RgNmMAbu79RW0rgvIEfX3Hw'
export OPENAI_API_KEY='Your Own OpenAI API Key'

curl --fail --silent --show-error \
  --request POST "$JUDGE_URL/v1/reviews/task-proposal" \
  --header "Authorization: Bearer $JUDGE_API_KEY" \
  --header "X-OpenAI-API-Key: Bearer $OPENAI_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{"proposal":"# My task proposal\n\nDescribe the task here."}'
```

The JSON body must contain a non-empty `proposal` string. A successful response
contains `decision`, `review`, `model`, and `reasoning_effort`.

The proposal must include `Contributor full name`. The judge uses web search to
check public expertise alignment and review recent frontier activity in the
rolling six-month window. Frontier evidence, including X topic activity, is a
non-blocking quality signal rather than a publication-count gate. Ambiguous
contributor identity produces `require human review`; incomplete frontier search
is recorded as a limitation and does not change the decision by itself.

### Health check

```bash
curl https://rsi-index-judge.zhuofengli12345.workers.dev/health
```

This is public and does not call OpenAI.


## Deploy an existing service

This repository is already wired to production KV namespace
`ece0a4937cc64c6b831c8b2862f5921c`. To deploy a code change without changing
the rubric or secret:

```bash
cd tools/rubric-review-service
npm ci
npx wrangler login
npm run check
npx wrangler deploy --keep-vars
curl https://rsi-index-judge.zhuofengli12345.workers.dev/health
```

The `workers.dev` URL has two parts: Worker name (`rsi-index-judge`) and the
account-level subdomain (`zhuofengli12345`). The account subdomain cannot be
removed from a `workers.dev` URL. Use a custom domain route if a shorter public
hostname is needed.

## Update the rubric

The private repository is the source of truth. KV is a deployed snapshot; it
does not automatically track Git commits. After editing the private rubric,
upload the new file:

```bash
cd tools/rubric-review-service
npx wrangler kv key put --binding=RUBRICS task-proposal \
  --path ../../../RSI-Skills/rubrics/task-proposal.md --remote
```

The next request reads the new KV value. No Worker code deployment is required.


## Adding another rubric review

New rubric types must use an explicit parallel endpoint, for example:

```text
POST /v1/reviews/<rubric-name>
```

Add a dedicated request schema, result schema, KV key, and route handler for
each review type. Do not reuse the task-proposal rubric or expose a generic
`/evaluate` endpoint.

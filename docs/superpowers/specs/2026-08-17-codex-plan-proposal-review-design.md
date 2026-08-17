# Coding Plan Proposal Review Design

## Scope

This change has two coordinated outputs:

1. Create the official contributor proposal template in `task_collect` directly
   from `contributor_SOP.txt`. The finalized contributor/agent interaction
   protocol is out of scope and remains unchanged.
2. Replace the proposal reviewer's direct OpenAI Responses API call with a
   non-interactive Codex CLI invocation authenticated by a persistent ChatGPT
   Coding Plan login on a dedicated self-hosted GitHub Actions runner.

Ground-truth proposal examples are explicitly deferred.

## Proposal format

The template keeps the SOP's four top-level sections and its contributor-facing
fields: Research Question, Reference Baseline, Evaluation, and Workspace. It
uses separate `Repository URL` and `Exact commit/tag` fields so the reviewer can
resolve immutable repository evidence. `Repository evidence paths` is included
next to the baseline fields because the SOP requires traceability to repository
paths, commands, refs, or artifacts and the reviewer consumes that label.

The template must not invent measured variance, runtime, baseline reproduction,
or final verifier behavior at proposal time. It will be created as
`task_collect/proposal_format_v2.txt`; files under `unused/` remain untouched.

## Judge invocation

`checks/rubric_review.py` will stop importing or constructing OpenAI API SDK
clients. It will invoke `codex exec` with:

- fixed model `gpt-5.6-sol` and reasoning effort `xhigh`;
- an ephemeral conversation and read-only sandbox;
- shell, web search, apps, MCP/plugin discovery, memories, and multi-agent
  functionality disabled for the judge;
- a temporary isolated working directory containing only trusted judge
  instructions and downloaded image attachments;
- proposal and fetched repository evidence passed through standard input as a
  length-labelled JSON evidence value;
- the final assistant message captured in a temporary output file.

The rubric is trusted control-plane input. Proposal text, repository contents,
and images remain explicitly untrusted evidence. Disabling tools is required
because the persistent Coding Plan credential lives on the runner and must not
be reachable through prompt injection.

The existing output contract remains unchanged: the review must be non-empty
and its final non-empty line must contain one canonical decision. A missing
Codex executable, failed subprocess, empty output, or invalid decision fails the
workflow rather than silently falling back to another model or credential.

## Images

Downloaded images retain the current size and magic-byte validation. Validated
image bytes are written only inside the temporary judge directory and attached
with repeated Codex CLI `--image` arguments. Temporary files are removed after
the invocation.

## Authentication and CI

The Discussion workflow will target a dedicated runner labelled
`rsi-proposal-review` in addition to the standard `self-hosted`, `linux`, and
`x64` labels. It will no longer read `OPENAI_API_KEY`. The runner must provide:

- a current Codex CLI;
- a persistent, runner-local `CODEX_HOME` dedicated to this judge;
- a ChatGPT-managed `auth.json` created by `codex login` and readable only by
  the runner account.

The workflow performs a fail-fast preflight for the CLI and credential file.
Codex may refresh the credential in the persistent `CODEX_HOME`; the directory
must not be checked into Git, uploaded as an artifact, or exposed as a GitHub
secret. Runner provisioning itself is an operator action because the repository
currently has no self-hosted runner and this workstation must not be silently
registered as a long-running GitHub service.

## Tests

Tests will cover command construction, fixed model/reasoning settings, disabled
tools, stdin delivery, image attachment paths, successful output capture,
subprocess failure, empty output, and the workflow's self-hosted/no-API-key
configuration. Existing parsing, evidence bounding, injection framing, and
canonical-decision tests remain in place.


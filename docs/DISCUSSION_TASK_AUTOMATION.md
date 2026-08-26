# Discussion Task Automation

Discussion task creation crosses a deliberate Public/private boundary. The
Public repository accepts a small command surface and routes only the
identifiers needed to a private worker; it does not run Codex or retain Codex
logs.

## Request protocol

The Discussion author or a current `RSI-Index` organization Owner can comment
`/task` to request task creation, add high-level feedback with
`/task <feedback>`, and use `/task confirm` to confirm the final version. An
ordinary organization member or repository collaborator is not eligible.

Public ingress is a non-authoritative cost gate, not the eligibility decision.
It checks the event action, canonical Public repository and category, stable
actor identifiers, required identifiers, and `/task` command shape. For a
non-author candidate, it uses the Dispatcher App to require an active
organization membership whose API role is `admin` (Owner). It then creates an
identifier-only private dispatch. The private worker repeats the authoritative
authorization and does not trust this ingress result.

## Configuration

Install the dispatch GitHub App for the `RSI-Index` organization and configure
these Public repository settings:

- Variable: `RSI_DISPATCH_APP_CLIENT_ID`
- Secret: `RSI_DISPATCH_APP_PRIVATE_KEY`

Keep the App's private-key material out of the repository. Install it on both
`RSI-Index/RSI-Index-Public` and `RSI-Index/RSI-Skills`, with repository
`Discussions: Read and write`, repository `Contents: Read and write`, and
organization `Members: Read-only`. Each workflow call narrows the installation
token to the repository and permissions needed for that step; no personal token
is used.

After this cutover, edit any already-accepted Discussion once before using
`/task`. The review workflow intentionally updates only marker comments owned
by the current App bot. A legacy `github-actions` or personal-token marker is
left unchanged, and the edit creates a new App-owned authoritative marker.

## Private worker and outcomes

The private worker owns request State, task generation, review, publishing,
and Discussion replies. It authoritatively re-fetches the Discussion and
command comment and reauthorizes the request: it verifies the approval marker
and App identity, Proposal hash and currentness, category and Discussion author,
current organization Owners, and the unedited command. On a successful final
result it creates a private repository and grants the Discussion author Write
permission; an Owner who operated the workflow does not replace that author.
Contributors perform full execution validation locally; Public ingress does
not perform or claim that validation.

## Skill boundary and history

Current skill sources are private and are not maintained in this Public
working tree. Previously published skill versions remain recoverable through
Git history; this migration does not rewrite, delete, or alter historical
commits. Future skill versions and their operational material stay in the
private worker repository.

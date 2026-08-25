# Discussion Task Automation

Discussion task creation crosses a deliberate Public/private boundary. The
Public repository accepts a small command surface and routes only the
identifiers needed to a private worker; it does not run Codex or retain Codex
logs.

## Request protocol

The Discussion author can comment `/task` to request task creation. They can
add high-level feedback with `/task <feedback>`, then use `/task confirm` to
confirm the request. A request is eligible only when the commenter is the
Discussion author and the Discussion has an accepted current Proposal.

Public ingress is a non-authoritative cost gate, not the eligibility decision.
It checks only the event action, canonical Public repository and category,
author-ID relationship, required identifiers, and `/task` command shape before
creating an identifier-only private dispatch. It does not fetch or decide the
current Proposal, approval, or command authority. The workflow has read-only
Public contents permission; the scoped GitHub App token is created only after
this gate passes and is used solely to deliver the request to the private
worker.

## Configuration

Install the dispatch GitHub App for the `RSI-Index` organization and configure
these Public repository settings:

- Variable: `RSI_DISPATCH_APP_CLIENT_ID`
- Secret: `RSI_DISPATCH_APP_PRIVATE_KEY`

Keep the App's private-key material out of the repository. Its installation is
scoped to the private worker repository, `RSI-Index/RSI-Skills`; the Public
workflow does not need a general-purpose token or write access to Public.

## Private worker and outcomes

The private worker owns request State, task generation, review, publishing,
and Discussion replies. It authoritatively re-fetches the Discussion and
command comment and reauthorizes the request: it verifies the approval marker
and approving account, Proposal hash and currentness, category and Discussion
author, and the unedited command. On a successful final result it creates a
private repository and grants the Discussion author Write permission.
Contributors perform full execution validation locally; Public ingress does
not perform or claim that validation.

## Skill boundary and history

Current skill sources are private and are not maintained in this Public
working tree. Previously published skill versions remain recoverable through
Git history; this migration does not rewrite, delete, or alter historical
commits. Future skill versions and their operational material stay in the
private worker repository.

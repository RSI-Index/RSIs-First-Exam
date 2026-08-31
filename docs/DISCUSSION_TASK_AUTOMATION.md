# Discussion Task Automation

Discussion task creation crosses a deliberate Public/private boundary. The
Public repository accepts a small command surface and routes only the
identifiers needed to a private worker; it does not run Codex or retain Codex
logs.

## Request protocol

The standard path starts when the review workflow publishes a current
App-authored schema-2 `Pass` marker. The workflow re-fetches the live Discussion
and marker, then sends the private worker exactly five identifier fields:
source repository, Discussion number, Discussion node ID, triggering marker
node ID, and `trigger_kind=proposal_pass`. No proposal or review prose crosses
that dispatch boundary.

If automatic assumption research finds a genuine contributor-owned choice, the
Discussion author or a current `RSI-Index` organization Owner can answer with
`/task <answer or correction>`. An ordinary organization member or repository
collaborator is not eligible.

The same authorized users can post an exact, top-level `/reset` comment to
discard an unpublished task attempt. For an automatic lineage, reset preserves
the current App-authored schema-2 `Pass` review and deletes later task-workflow
replies together with the `/reset` comment; the Proposal also remains. For a
legacy or manual lineage, deletion begins with the first valid authored `/task`
after the accepted review and includes `/reset`, preserving the Proposal and
review. Published tasks cannot be reset, so their State, Discussion history,
and private repository are left unchanged. Reset stops the workflow. After a
successful unpublished reset, send a new plain `/task` to start a clean attempt.

Public ingress is a non-authoritative cost gate, not the eligibility decision.
It checks the event action, canonical Public repository and category, stable
actor identifiers, required identifiers, and command shape. For a
non-author candidate, it uses the Dispatcher App to require an active
organization membership whose API role is `admin` (Owner). It then creates an
identifier-only private dispatch. The private worker repeats the authoritative
authorization and does not trust this ingress result.

## Legacy and manual operation

Legacy plain `/task`, `/task <feedback>`, and `/task confirm` comments remain
parseable for existing State recovery and operator compatibility. A manual
`workflow_dispatch` also remains available with dry-run defaulting to true.
For contributors, a new plain `/task` is used only to start a clean attempt
after `/reset`; it is not required on the normal automatic path. Operators
should prefer the automatic `proposal_pass` route for every new Discussion.

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

Deploy the private Skills revision first, then the Public revision. This lets
the private parser accept and verify `proposal_pass` before Public can emit it.
After cutover, edit an already-passed Discussion once to create a current
App-owned schema-2 marker. A legacy `github-actions` or personal-token marker is
left unchanged.

## Private worker and outcomes

The private worker owns request State, task generation, review, publishing,
and Discussion replies. It authoritatively re-fetches the Discussion and
triggering marker or answer comment and reauthorizes the request: it verifies the approval marker
and App identity, Proposal hash and currentness, category and Discussion author,
current organization Owners, and the unedited bound authority. On a successful final
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

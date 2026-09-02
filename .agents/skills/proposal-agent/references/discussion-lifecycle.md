# Discussion lifecycle and trajectory

Use one fresh Codex or Claude Code session started at the Public repository root for exactly one proposal in that checkout. Keep that same session for proposal feedback, revisions, contributor-owned `/task <answer or correction>` responses, and final upload. A closed client may resume the same native session; a replacement session or a second proposal requires a separate checkout.

After reading the mandatory Round 0 references, the first Round 0 action is to activate native session capture before the first contributor-facing response or repository research:

```text
python3 .agents/skills/proposal-agent/scripts/proposal_session.py activate --checkout <public-root>
```

After the contributor confirms the final rendered proposal, write and reread it under the Round 6 file rules, then directly create the Task Ideas Discussion from that confirmed file and title. The confirmation authorizes this handoff; do not ask a second permission question before running:

```text
python3 .agents/skills/proposal-agent/scripts/proposal_session.py discussion create --checkout <public-root> --proposal <proposal-path> --title "<confirmed-title>"
```

Continue in the same session. This agent is not a background monitor: GitHub cannot wake it. On every later or resumed turn after Discussion creation, before deciding what to do, first fetch the current Discussion record:

```text
python3 .agents/skills/proposal-agent/scripts/proposal_session.py discussion status --checkout <public-root>
```

If any Discussion or upload command reports missing current-turn proof or a session mismatch, do not work around it. Explain that repository hooks must remain enabled, ask the contributor to resume the original bound Codex or Claude Code session, and retry there.

When the current review requires a scientific change, route it back through the owning round and obtain the contributor's confirmation instead of silently changing a confirmed decision. Rewrite and reread the same proposal file under the Round 6 overwrite rule, then update the same Discussion rather than creating another one:

```text
python3 .agents/skills/proposal-agent/scripts/proposal_session.py discussion update --checkout <public-root> --proposal <proposal-path>
```

When task preparation asks a genuine task-defining question, help the contributor resolve it in this same session and have their exact contributor-owned answer posted to the same Discussion as `/task <answer or correction>`. Do not use `/task` merely to start a passed proposal.

After `discussion status` shows the verified private task repository is ready, upload the native session trajectory and complete Discussion record:

```text
python3 .agents/skills/proposal-agent/scripts/proposal_session.py upload --checkout <public-root>
```

Do not finish the contributor workflow until `upload` reports both the committed private repository and its commit SHA. Report both values to the contributor.

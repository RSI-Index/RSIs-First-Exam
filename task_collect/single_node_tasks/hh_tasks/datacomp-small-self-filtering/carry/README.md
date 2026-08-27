# `/carry` — the only channel between rounds

Empty in the shipped package. This directory exists so the mount point and its contract are
documented in one place.

```
/carry/
  improver/                 round k-1's filter source. Round k inherits it and may rewrite it.
    improver.py             entry point; the harness calls it with a fixed CLI (see instruction.md)
  frozen_models/
    round_<k>/              a CLIP YOU trained. At most `carry_model_slots` = 2, weights + info.pkl.
  ledger.jsonl              one row per round, append-only.
  notes/                    free-form. Yours.
```

## Why the model slots are capped at 2

An unbounded `frozen_models/` turns the task into ensembling: round 6 could score the pool with six
CLIPs and the chain's rising line would be accumulated inference capacity rather than a better
filter. Enforced twice — by `task-tools/carry_tool.py`, which the agent calls, and by the chain
harness, which prunes whether or not the agent called it. Two enforcement points because an agent
that simply never calls the tool would otherwise hand round k+1 everything.

`carry_tool.py` copies **weights and `info.pkl` only**, not the whole training output directory.
`train.py`'s output also holds logs and, depending on flags, epoch checkpoints; copying all of it
would grow the carry channel without bound and hand the next round an ensemble by accident.

## The ledger

One JSON object per line, appended, never rewritten:

```json
{"round": 3, "n_uids": 4120000, "subset_sha256": "...", "tool_gpu_seconds": 3140.7,
 "prev_policy_used": true, "scoring_route": "round_2 CLIP image-text agreement",
 "improver_sha256": "...", "checkpoint_digest": "..."}
```

`scoring_route` is worth recording carefully in this task: an improver can rank the pool with the
**precomputed OpenAI similarities** that ship in the metadata (zero GPU cost) or with a **carried
checkpoint** (real GPU cost, metered). Those are different claims about what improved, and a chain
whose routes are unknown cannot be interpreted afterwards.

A round's verifier can only ever see the ledger as it stands during that round, so it cannot tell
that row 2 was rewritten between rounds 2 and 3. The prefix property — round k's ledger begins with
round k-1's, byte for byte — is checked by `harness/chain_check.py` across rounds. That is the chain
half of RH-RSI-007, and without it the record of which filter produced which number is editable by
the thing being measured.

# `reference/` — anchors and a candidate solution. Never inside an image.

`.dockerignore` excludes this directory and `tests/contract_check.py` fails the package build if
that entry disappears, if either Dockerfile mentions `reference/`, or if either does a bare
`COPY . <dest>`. Three checks for one exclusion, because what is in here would hand an agent the
answer.

```
improvers/passthrough_improver.py   the identity selection: reference anchor AND null chain
assert_null_chain.py                the two conditions that make the null chain usable
```

## Why one file serves two anchors

`passthrough_improver.py` is both the `reference_improver` anchor and the null chain's improver.
Reward 1.0 therefore means "matched the uncurated pool under the same 3000-step budget" — an
unambiguous quantity, not a tuned baseline of ours. And because the null chain runs *the same
file* in all six rounds, its flatness cannot be an artifact of two implementations differing.

That forces a property: the file must be byte-deterministic. No timestamp, no round, no seed, no
unsorted dict in the output. `assert_null_chain.py` checks it by sha256 across rounds.

## What the null chain assertion actually checks

Two conditions, and the second is the one that is easy to forget:

1. **The keep list is byte-identical in every round.** If it is not, the "null" chain varied its
   input and its slope is not a noise measurement.
2. **The scores are NOT all identical.** If they are, the measured variance is zero, every σ
   derived from it is zero, and `min_detectable_gain = 2σ = 0` — so every noise gate downstream
   passes automatically. A gate that cannot fail is a disqualifying defect, and a zero-variance
   null chain manufactures a whole family of them.

## There is no sanity improver here yet

The sibling VLM-R1 package ships one: a selector that scores prompts with round k-1's policy and
keeps a mid-band, used to prove the harness can register a real gain before any agent runs. The
equivalent here would score pool rows by the inherited checkpoint's loss and keep a mid-band.

It is deliberately not written yet, because at this tier it would be premature: whether *any*
selection is resolvable at 3000 steps is decided by `contract.yaml`
`launch_gates.signal_resolvable`, which compares the best selection upstream can express (the four
rating thresholds, swept) against no selection at all. If that gate fails, a sanity improver would
be measuring an effect the instrument cannot see, and the honest conclusion is about cost rather
than about recursion. Write it after that gate passes.

# `reference/` — anchors and a candidate solution. Never inside an image.

`.dockerignore` excludes this directory and `tests/contract_check.py` fails the package build if that
entry disappears, if either Dockerfile mentions `reference/`, or if either does a bare
`COPY . <dest>`. Three checks for one exclusion.

```
improvers/passthrough_improver.py   the identity filter: reference anchor AND null chain
assert_null_chain.py                the three conditions that make the null chain usable here
```

## Why one file serves two anchors

`passthrough_improver.py` is both the `reference_improver` anchor and the null chain's improver.
Reward 1.0 therefore means "matched training on the unfiltered pool at the same 12.8M-sample budget"
— unambiguous, and not a tuned baseline of ours. And because the null chain runs *the same file* in
all six rounds, its flatness cannot be an artifact of two implementations differing.

That forces byte-determinism: no timestamp, no round, no seed in the output.

## It is `no_filter`, reimplemented in four lines. That is deliberate.

`baselines.py --name no_filter` enumerates uids from the **parquet metadata**. The realized pool is
the **crawled shards**, which is a strict subset — link rot. So `no_filter` over the metadata would
put uids in the reference subset that the resharder cannot find, and the anchor every reward is
divided by would describe a pool that does not exist.

The identity filter here is therefore defined as *every uid we actually have*, read from
`pool_uids.npy`. It refuses to fall back to the metadata rather than quietly producing the wrong
anchor.

## What the null chain assertion checks

Two conditions shared with the sibling packages, and a third specific to this one:

1. **The subset is byte-identical in every round.** Otherwise the "null" chain varied its input.
2. **The scores are NOT all identical.** If they are, measured variance is zero, every σ derived from
   it is zero, `min_detectable_gain = 2σ = 0`, and every noise gate downstream passes automatically.
   A gate that cannot fail is a disqualifying defect and a zero-variance null chain manufactures a
   family of them.
3. **The pool fingerprint is the same in every round.** A null chain spanning two crawls would show
   spread that is the internet rather than the seed. That inflates σ rather than shrinking it, so it
   would hide real effects instead of inventing them — equally wrong, and much easier to miss
   because a *larger* noise estimate looks like the conservative choice.

## There is no sanity improver here yet

The VLM-R1 sibling ships one, to prove the harness can register a real gain before any agent runs.
The equivalent here is `contract.yaml` `launch_gates.recursion_channel_has_signal`: a filter that
ranks the pool with the round-1 CLIP must beat a **size-matched random** subset by more than the
noise floor.

Size-matched random, not `no_filter`, because subset size alone changes how many times each member is
revisited under the fixed 12.8M-sample budget — comparing a 30% filter against the full pool would
confound the filter's quality with its keep rate.

Write the sanity improver after that gate passes. If it fails, the honest conclusion is that a CLIP
trained at `small` is not a competent data scorer, agents will rationally fall back to the
precomputed OpenAI similarities, and the recursive claim needs `medium` — a result about the
benchmark, reported rather than worked around.

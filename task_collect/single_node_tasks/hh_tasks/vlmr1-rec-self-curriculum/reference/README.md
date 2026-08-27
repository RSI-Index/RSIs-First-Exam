# `reference/` — anchor and sanity code. NEVER SHIPPED IN AN IMAGE.

Three of the four things `README.md` step 3 and step 4 ask for are code, not compute. They live
here so they exist before the GPUs are booked, and they live *outside* `environment/` and
`tests/` so that neither Dockerfile can pick them up.

`tests/contract_check.py` fails the build if any Dockerfile `COPY`s from `reference/`, and if
`reference/` is missing from `.dockerignore`. That gate is not decoration: `sanity_improver.py`
is a **working candidate solution** to the benchmark task, and one careless `COPY . /app` would
hand every agent a known-good recursion on a plate.

| file | what it is | which anchor |
|---|---|---|
| `improvers/passthrough_improver.py` | emits the uncurated pool, byte-identically every round | `reference_improver` **and** the `null_chain` |
| `improvers/sanity_improver.py` | scores pool prompts with round-(k−1)'s own policy, keeps the learnable mid-band | the sanity chain (step 4) |
| `assert_null_chain.py` | proves the null chain really was null | validity of `null_chain.slope` |

## Why one file serves two anchors

`reference_improver` is *one round* of the upstream recipe on the uncurated pool at our step
budget — it sets reward `1.0`. The `null_chain` is *six rounds* of that same thing with the seed
varying. They are the same improver; only the harness differs. Writing them as one file is not
laziness, it is the guarantee that the two anchors are comparable: if they were two files, the
null chain's flatness could come from a difference between the files rather than from the absence
of recursion.

That guarantee has a precondition, and `assert_null_chain.py` is what checks it: the emitted
curriculum must be **byte-identical in all six rounds**. If it drifts — a set iteration order, a
timestamp in a comment, `hash()` on a string — then the null chain has a recursion in it after
all, its slope is not a null distribution, and every attribution built on it is void. This is not
hypothetical: a "deterministic" corpus in an earlier project differed on 23,432 of 24,496 records
because `hash()` is per-process randomised in Python.

## What the sanity improver is allowed to do, and what it is not

It scores candidate prompts with its own IoU implementation. That is **selection-time scoring**
and it is explicitly allowed — the improver may rank pool items however it likes. What is frozen
is the **training** reward (`iou_reward`, `format_reward_rec` in
`vlm_modules/qwen_module.py`), which the trainer applies and the improver never touches. Calling
the frozen reward here would mean reconstructing the trainer's collator, and the resulting
signature coupling would make an improver edit look like a reward edit to the policy gate.

It is also not the answer key. It implements the one hypothesis the repo's own machinery makes
obvious — groups with reward std 0 contribute no advantage, so prompts that are already solved or
still hopeless are wasted budget — and nothing more. If the sanity chain is flat, the honest
reading is that this task does not measure recursion, not that a cleverer improver was needed.

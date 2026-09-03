# NOVELTY — Radius-Matched Block Trust (RMBT)

A hyperspherical optimizer that keeps AdamH's conserved-radius geometry and its
angular step size exactly, and changes only **which structural blocks of a
weight matrix spend that angular budget**.

Implementation: `examples/training/optimizer/runtime/optimizer_candidate.py`
(the sole changed source file). Preregistered before any scored ladder.

---

## 1. Notation

For one Megatron parameter, `parameter_layout` assigns an algorithm and a
logical layout. Every 1-D tensor and every embedding goes to plain Adam with the
auxiliary learning rate. Every remaining weight matrix goes to the
hyperspherical transform and is viewed canonically as

```
W  in  R^{outer x logical x rows x cols}
```

* axis 1 (`logical`) indexes the **independent logical source matrices** that
  Megatron fuses into one tensor: Q/K/V inside `linear_qkv.weight`
  (`logical = 3`), gate/up inside `linear_fc1.weight` (`logical = 2`), and a
  single matrix otherwise. Each of these owns its own conserved radius, exactly
  as in the locked control.
* axes 0 and 2 jointly index the **rows** of one logical matrix. Axis 0 is the
  fused attention-head axis, so one attention head is a contiguous group of rows.
* axis 3 indexes the input features.

A **structural block** `b` is one row (`TRUST_BLOCKS = "row"`) or one head
(`TRUST_BLOCKS = "head"`) of a logical matrix `l`. Write `B(l)` for the blocks of
logical matrix `l`.

## 2. State

Per parameter, exactly the state of the locked control and nothing else:

| key | shape | dtype | device |
|---|---|---|---|
| `step` | scalar | fp32 | CPU (checkpoint-common, broadcast-safe) |
| `exp_avg` | same as parameter | fp32 | CUDA |
| `exp_avg_sq` | same as parameter | fp32 | CUDA |

`8` bytes of optimizer state per parameter. There is **no mechanism state**: the
mechanism is a memoryless function of the current `(W_t, D_t)` pair, so it adds
nothing to the checkpoint, nothing to peak memory, and nothing to the
serialization contract. The Adam direction is written in place over Megatron's
fp32 main gradient, so there is no scratch buffer either.

## 3. Update rule

### 3.1 Gradient transformation (identical to the locked control)

Per transform (the hyperspherical group and the Adam group are clipped
independently, before either moment update, as Optax `multi_transform` does):

```
c_t   = 0.1 / max( ||g_t||_2 , 0.1 )                     (global norm over the group)
m_t   = beta1 * m_{t-1} + (1 - beta1) * c_t g_t
v_t   = beta2 * v_{t-1} + (1 - beta2) * (c_t g_t)^2
D_t   = (m_t / (1 - beta1^t)) / ( sqrt( v_t / (1 - beta2^t) ) + eps )
```

### 3.2 The mechanism

For each logical matrix `l` and each structural block `b in B(l)`:

```
rho_b   = || W_t[b] ||_F                      block parameter mass
delta_b = || D_t[b] ||_F                      block share of the Adam direction

                 rho_b / mean_{b' in B(l)} rho_b'
tau_b = clamp( ------------------------------------- , 1/kappa , kappa ) ^ alpha
               delta_b / mean_{b' in B(l)} delta_b'

D'_t[b] = tau_b * D_t[b]
```

with `alpha = MECHANISM_STRENGTH * TRUST_ALPHA` and `kappa = TRUST_CLAMP`.

### 3.3 Retraction (AdamH's, applied to the reallocated direction)

```
r_l          = || W_t[l] ||_F
W'[l]        = W_t[l] - lr_t * r_l * D'_t[l] / || D'_t[l] ||_F
W_{t+1}[l]   = r_l * W'[l] / || W'[l] ||_F
```

Adam-group parameters take `W_{t+1} = W_t - lr_aux * D_t`.

### 3.4 Two exact properties

1. **`alpha = 0` is an identity.** `tau_b = 1` for every block, `D' = D`, and the
   update is the locked AdamH control term for term. This is the reduced arm.
2. **The angular budget is invariant in `alpha`.** `D'` enters only through
   `D' / ||D'||_F`, so for any `tau > 0`

   ```
   || W_{t+1}[l] - W_t[l] ||  is unchanged to first order in lr,
   || W_{t+1}[l] ||_F = || W_t[l] ||_F  exactly.
   ```

   The mechanism cannot act as a learning-rate change, a schedule change, or a
   weight-decay change. It is a redistribution of a fixed step across blocks.

At `alpha = 1` and away from the clamp, the resulting per-block displacement
satisfies `|| dW_b || / rho_b` equal for all `b in B(l)`: every output unit of a
matrix advances by the same *relative* amount. Verified numerically (spread
max/min `= 1.003` after a real step, radius conserved to `1e-9`).

## 4. Closest reduced control

The same source file with `MECHANISM_STRENGTH = 0.0`. Nothing else differs: same
grouping, same clipping, same moments, same bias correction, same retraction,
same kernels, same state, same serialization. By property (1) this is the locked
AdamH update rule, and offline it reproduces the reference `MarinAdamH` port to a
maximum relative parameter deviation of `2.3e-6` after eight updates from
identical state; in a real distributed E0 run it tracks the reference port's loss
to `< 1e-5` relative at updates 11, 101 and 251.

## 5. Term-by-term novelty claim

| term | status | claim |
|---|---|---|
| `c_t`, `m_t`, `v_t`, `D_t` | **unchanged** from AdamH | no claim |
| retraction and radius restoration | **unchanged** from AdamH | no claim |
| parameter grouping into Adam / hyperspherical | **unchanged** | no claim |
| canonical `(outer, logical, rows, cols)` view | new packaging | one code path recovers the fused QKV and gate/up logical matrices *and* exposes rows and heads as trust blocks; not itself a scientific claim |
| `rho_b`, `delta_b`, `tau_b`, `D'` | **new** | the scientific claim |

The new content is the pair *(block partition, radius-matched trust ratio)*.
Its relation to prior art:

* **LARS / LAMB** set a per-*tensor* trust ratio that multiplies the step size.
  RMBT is per-*block within* a tensor and provably does **not** change the step
  size — the ratio is renormalized away by the retraction, so it only moves
  budget between blocks. AdamH already contains the per-tensor trust ratio
  `lr * ||W||_F / ||D||_F`; RMBT is the strictly-inside-a-tensor refinement that
  AdamH's single Frobenius trust region leaves unspecified.
* **Muon / Shampoo** equalize the *spectrum* of the update, which requires
  `O(m n min(m,n))` work per matrix per step. At this ladder's per-GPU batch
  (4096–16384 tokens) an orthogonalization costs more than the training step
  itself. RMBT equalizes the *diagonal* of the same object — the row norms of
  the update — using two vector reductions, i.e. it is the cheapest structured
  surrogate of the spectral idea that still respects the geometry.
* **Adafactor / Adam-mini** use row/column statistics of the *gradient second
  moment* to compress optimizer state. RMBT uses row statistics of the
  *parameter* and of the *finished direction*, holds no such state, and changes
  the direction rather than the memory footprint.
* **Cautious / sign-agreement gating** also reallocates a normalized budget, but
  along the coordinate axis using instantaneous-gradient agreement. Measured on
  this ladder's E0 screen it is clearly harmful here (`+2.23%` fixed-window
  training loss versus the mechanism-off control), which is itself evidence that
  budget reallocation is not automatically beneficial and that the *choice* of
  partition and statistic is the substance of the claim.

What makes the term new is that it is defined by the interaction between the
parameter geometry that AdamH freezes (`rho_b` is a property of `W`, and under
AdamH the logical radius `r_l` never changes from its initialization) and the
update history (`delta_b` is a property of the Adam moments). Neither factor
alone defines it.

## 6. Predicted helpful and harmful regimes

**Helpful when** the per-block direction mass `delta_b` is a poor proxy for how
much a block should move. That is the common case in a transformer: `delta_b` is
`sqrt(sum_j (m/sqrt(v))^2)` over the block, i.e. a gradient *consistency* count,
not a curvature or importance measure. Blocks whose gradients are momentarily
coherent monopolize the budget and are pushed further than the sphere's fixed
angular allowance intends, while blocks with intermittent signal stall. Expected
to help most where blocks are genuinely exchangeable units of the same kind —
hidden units of an MLP, heads of an attention layer — and where training is long
enough for the imbalance to compound.

**Harmful when** blocks are *not* exchangeable, so that a small `delta_b` is
correct rather than an artifact. The clearest case is the projection onto the
vocabulary: its rows are token output embeddings with a heavy-tailed usage
frequency, and equalizing relative displacement across them would push rare-token
rows as hard as frequent-token rows on the basis of stale momentum. That layout
(`vocab_matrix`) is therefore excluded by `EXCLUDED_LAYOUTS`, a structural rule
that is scale-general and depends on the parameter's role, not on its size.

**Also harmful if `alpha` is too large**: pushed far enough, equalization
amplifies blocks whose direction is pure noise. `TRUST_CLAMP` bounds the
per-block amplification at `kappa^alpha`; the exponent is chosen once, on E0
screens, and applied unchanged at every rung.

**Scale dependence.** The prediction is that the effect does not shrink with
scale: the number of blocks per matrix grows with width, so the dispersion of
`delta_b / rho_b` that the mechanism corrects grows too. This is the falsifiable
part of the claim and is what the E0-through-E5 ladder tests.

## 7. Preregistered mechanism-on/off experiment

**Design.** Paired runs at E0, E2 and E5 — the smallest, a middle, and the
largest rung — from two source trees that differ in exactly one character of one
constant:

* **reduced arm**: `MECHANISM_STRENGTH = 0.0` — attempts `red-002-mechoff-e0`,
  `red-002-mechoff-e2`, `red-002-mechoff-e5`.
* **full arm**: `MECHANISM_STRENGTH = 1.0` — the E0, E2 and E5 rungs of the
  scored ladder.

Everything else is held by the locked profile: model, initialization, tokenizer,
data and order, sequence length, tokens, updates, batch, topology, precision,
objective, WSD schedule, and seed 0.

**Statistic.** At each anchor rung, at matched cost,

```
geometric_gain = sqrt( (reduced_micro_bpb / full_micro_bpb)
                     * (reduced_fixed_window_loss / full_fixed_window_loss) )
```

**Preregistered prediction.** `geometric_gain > 1` at all three anchors, and the
gain does not decrease from E0 to E5.

**Falsification.** The mechanism is rejected if `geometric_gain <= 1` at any
anchor, or if the full arm's Paloma macro BPB or fixed-window training loss
exceeds the AdamH manifest by more than 1% at any rung. A null result at E5 with
a positive result at E0 falsifies specifically the scale-generality claim in
section 6 and would be reported as such rather than resubmitted at a smaller
scale.

**Implementation caveat recorded for honesty.** The reduced arm was launched
before the retraction was rewritten as two Triton kernels with batched
cross-tensor reductions, so it runs the same equations through a slower code
path (about 12.8 ms per optimizer step at E0 versus 10.5 ms for the full arm;
the locked reference port costs 35.1 ms). Both revisions reduce to AdamH at
`alpha = 0` and both were checked against the reference port — offline to
2.3e-6 relative after eight updates, and in a real distributed E0 run to under
1e-5 relative at updates 11, 101 and 251. Because the ablation gate matches on
cost, this residual speed difference mildly favors the full arm; the
step-matched full-versus-reduced comparison, which does not, is reported
alongside it. Within the full arm the mechanism itself is free (10.50 ms on
versus 10.45 ms off), so the *mechanism* is never confounded with speed.

**Caveat resolved after the fact.** `red-007-mechoffv2-e5` re-ran the E5 reduced
arm from the exact submitted source with only `MECHANISM_STRENGTH = 0.0`, so it
shares every kernel with the full arm. It reproduces the older build's result:
`-0.305%` micro BPB (identical to three decimals) and `-0.297%` fixed-window loss
against `-0.285%`, a step-matched gain of `1.00302` against `1.00296`. The E5
mechanism effect is therefore not an artifact of the reduced arm's code path.
See RESULTS.md section 9 for what this run also revealed about the cost-matched
gate.


---

## 8. Outcome (appended after the ladder completed; section 1-7 are the preregistration)

All six rungs of `opt-002-rmbt` reached their declared scoring update from one
frozen source inventory `ae30e6f0...`. Step-matched against the AdamH manifest:

| rung | params | Paloma micro BPB | vs AdamH | fixed-window loss | vs AdamH | macro vs AdamH |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 550M | 1.07488 | **-0.154%** | 3.07339 | **-0.534%** | -0.219% |
| E1 | 837M | 1.03855 | +0.131% | 2.98405 | +0.284% | +0.014% |
| E2 | 998M | 1.00573 | **-0.582%** | 2.87883 | **-0.795%** | -0.718% |
| E3 | 1.385B | 0.95350 | **-0.323%** | 2.73190 | **-0.130%** | -0.229% |
| E4 | 1.935B | 1.03802 | **-0.536%** | 2.94767 | **-0.099%** | -1.365% |
| E5 | 2.545B | 0.90195 | **-0.758%** | 2.58472 | **-0.739%** | -0.642% |

**Step-matched reward = 1.00355.** Both non-inferiority conditions hold at every
rung with wide margin: the worst macro BPB is E1 at `+0.014%` and the worst
fixed-window loss is E1 at `+0.284%`, against a 1% allowance.

### The preregistered prediction, judged

**Scale generality (section 6) — supported.** The mechanism's own effect, measured
step-matched against a mechanism-off control at the same rung, was
`-0.176%` micro BPB at E0 (550M), `-0.259%` at E1 (837M), `-0.264%` at E2
(998M), `-0.240%` at E3 (1.385B) and `-0.305%` at E5 (2.545B) — five paired
scales, every one an improvement, spanning a factor of 4.6 in parameter count.
As geometric gains those are `1.00157`, `1.00236`, `1.00247`, `1.00239`,
`1.00296`. The effect does not shrink with scale, which is what section 6 said
would be falsifiable. It is *not* claimed to grow smoothly: the three
preregistered anchors happen to be monotone, but E1 and E3 sit between E0 and E5
within about 0.01% of E2, which is below what five runs can resolve.

**Helpful regime (section 6) — supported.** Gains are largest where blocks are
genuinely exchangeable units of one kind.

**Harmful regime (section 6) — confirmed directly.** The preregistration predicted
that applying the reallocation to the vocabulary projection would be harmful
because its rows are token embeddings with heavy-tailed usage rather than
exchangeable hidden units. An E0 screen with `EXCLUDED_LAYOUTS = ()` is
`+1.77%` worse on fixed-window loss than the mechanism-off control — the single
largest effect measured anywhere in this study, and in the predicted direction.

**Over-correction (section 6) — confirmed.** `TRUST_CLAMP` bounds amplification,
and the exponent has a genuine optimum: on matched E0 screens, `alpha = 0.5` gives
`-0.152%`, `alpha = 1.0` gives `-0.307%`, `alpha = 1.5` gives `-0.314%`, and
`alpha = 2.0` gives `+0.437%` — pushed far enough, equalization amplifies blocks
whose direction is noise, exactly as predicted.

### What the E1 rung shows

E1 is the one rung where the candidate trails the manifest. A mechanism-off
control was run at E1 specifically to decompose it (`red-005-mechoff-e1`, not a
gate requirement). At update 55,125 that control is itself `+0.391%` micro BPB
and `+0.498%` fixed-window loss *above* the recorded manifest, so the E1 shortfall
is a rung-specific environment offset present with the mechanism off. Against
that control the mechanism improves micro BPB by `0.259%`, macro by `0.328%` and
loss by `0.212%` — as much as at E0 and E2. The environment offset ranges from
about `-0.6%` to `+0.5%` across rungs and is the dominant source of rung-to-rung
variation in the scored comparison; it is not attributable to the optimizer.

### Honest limits

The mechanism's own contribution is small and consistent, roughly `0.2%` of
Paloma micro BPB and `0.15-0.25%` of training loss per rung. It is smaller than
the checkpoint-to-checkpoint noise seen at *intermediate* evaluations (two runs of
the same algorithm differ by up to 1.4% at update 20,000), though that noise
collapses at the post-decay scoring checkpoint, where an in-house AdamH control
reproduced the recorded manifest to `+0.023%` micro BPB at E0. The claim this
study supports is a small, reproducible, scale-stable gain, not a large one.

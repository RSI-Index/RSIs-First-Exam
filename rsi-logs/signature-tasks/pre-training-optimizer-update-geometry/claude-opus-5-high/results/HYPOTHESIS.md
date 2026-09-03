# opt-002-rmbt — Radius-Matched Block Trust, full arm

*(Identical mechanism and constants to `opt-001-rmbt`, relaunched after the
retraction was rewritten as two Triton kernels with batched cross-tensor
reductions. The first ladder was cancelled because its per-update cost ran
5-11% above the AdamH manifest at E0-E2, which the matched-cost staging gate
cannot absorb; the rewrite made the mechanism free and brought the cost ratio
to about 0.91 at E0. See `/app/output/experiments.jsonl`.)*

## One-sentence mechanism hypothesis

Inside a logical source matrix, AdamH's single Frobenius trust region lets a
structural block (one output unit) claim a share of the fixed angular step equal
to its share of `||D||_F` — a gradient-consistency count — rather than a share
matched to the parameter mass `||W_b||_F` it actually owns; rescaling the Adam
direction blockwise by the radius-matched trust ratio before the same
norm-preserving retraction, which provably leaves the total angular step and the
conserved radius untouched, should lower both Paloma bits-per-byte and the
fixed-window training loss at every rung of the ladder.

## Exact update rule

Gradient transformation, clipping, moments and bias correction are the locked
AdamH control, unchanged. For each logical source matrix `l` with structural
blocks `b` (one output unit each), after forming the Adam direction `D_t`:

```
rho_b   = || W_t[b] ||_F
delta_b = || D_t[b] ||_F

                 rho_b   / mean_{b' in l} rho_b'
tau_b = clamp( ---------------------------------- , 1/8 , 8 ) ^ alpha
               delta_b / mean_{b' in l} delta_b'

D'_t[b]    = tau_b * D_t[b]
r_l        = || W_t[l] ||_F
W'[l]      = W_t[l] - lr_t * r_l * D'_t[l] / || D'_t[l] ||_F
W_{t+1}[l] = r_l * W'[l] / || W'[l] ||_F
```

State is exactly `{step, exp_avg, exp_avg_sq}`; the mechanism is memoryless.
The projection onto the vocabulary (`output_layer.weight`) is excluded from the
reallocation because its rows are token embeddings with heavy-tailed usage, not
exchangeable hidden units.

`alpha = 0` is an exact identity and recovers AdamH. The full arm uses
`MECHANISM_STRENGTH = 1.0`, `TRUST_BLOCKS = "row"`, `TRUST_ALPHA = 1.0`,
`TRUST_CLAMP = 8.0`, i.e. `alpha = 1`: exact equalization of relative per-block
displacement, bounded by a factor of 8 on the trust ratio.

## Preregistered ablation

Reduced arm: the identical source with `MECHANISM_STRENGTH = 0.0`, run at E0, E2
and E5 as attempts `red-002-mechoff-e0`, `red-002-mechoff-e2`,
`red-002-mechoff-e5`. At each anchor, at matched cost,

```
geometric_gain = sqrt( (reduced_micro_bpb / full_micro_bpb)
                     * (reduced_fixed_window_loss / full_fixed_window_loss) )
```

must exceed 1. Prediction: it does at all three anchors, and does not shrink
from E0 to E5.

## Screening evidence that motivated this attempt

Local 4,000-update E0 screens (same model, data, order, seed and topology as the
rung, with the WSD schedule compressed to the shorter horizon), scored by the
task's fixed-window training loss against a mechanism-off control run through
the identical kernels:

| variant | fixed-window | vs control |
|---|---:|---:|
| mechanism off (control) | 3.75955 | +0.000% |
| locked `MarinAdamH` reference port | 3.76054 | +0.026% |
| agreement gate (`AGREEMENT_GATE`) | 3.84154 | +2.181% |
| radius-matched block trust, rows, `alpha = 1` | 3.74801 | **-0.307%** |

The reference-port line is the run-to-run and implementation noise floor for this
screen: two algorithmically identical AdamH runs differ by 0.03–0.08%.

## Falsification

Rejected if the mechanism-on/off gain is not positive at all three anchors, or if
Paloma macro BPB or fixed-window training loss exceeds the AdamH manifest by more
than 1% at any rung, or if the gain observed at E0 does not survive to E4/E5.

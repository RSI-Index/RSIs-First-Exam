# Hypothesis: Principal-Sine Polar Residual (PSPR)

## Scientific question

Does transporting by the norm of the polar component orthogonal to Adam yield
a stronger but still scale-free correction than AGPR's normalized spherical
angle?

## Exact mechanism hypothesis

Retain locked group clipping, Adam moments and bias correction, scalar
schedules, semantic logical matrices, vector/embedding Adam, the exact three
bfloat16 quintic Newton--Schulz polar iterations, fp32 norm matching, and
AdamH's fixed-Frobenius parameter retraction. For each logical matrix let
`P_t` be Adam's preconditioned direction and `Qbar_t` its norm-matched
approximate polar target. Measure

\[
\rho_t=\operatorname{clip}\!\left(
 {\langle P_t,\bar Q_t\rangle_F\over
  \max(\lVert P_t\rVert_F\lVert\bar Q_t\rVert_F,10^{-30})},0,1\right)
\]

and define the principal-sine weight

\[
\omega_t=\sqrt{\max(1-\rho_t^2,0)}.
\]

For unit directions this is exactly the Frobenius norm of the polar target's
component orthogonal to Adam. PSPR uses

\[
D_t=(1-\omega_t)P_t+\omega_t\bar Q_t
\]

before the unchanged AdamH subtract-and-renormalize retraction. Moments,
cosine, principal sine, mixture, and retraction use fp32; only the polar
polynomial uses bfloat16. Persistent state remains exactly `step`, `exp_avg`,
and `exp_avg_sq`.

## Closest reduced control and novelty

The closest reduced control is exact AGPR (`opt-013-agpr`). It shares every
group, recurrence, schedule, target, precision boundary, state tensor, and
retraction, but weights the target by normalized principal angle
`(2/pi)acos(rho_t)`. PSPR introduces an orthogonal-amplitude matrix
preconditioner: transport is set by the Euclidean norm of the component that
cannot be represented along Adam's current direction. Replacing principal
sine with normalized principal angle recovers AGPR exactly. This is a new
tensor- and update-dependent direction transformation, not a learning-rate,
momentum, epsilon, decay, clipping, schedule, or mask change.

## Predictions

PSPR should help when AGPR still under-corrects spectrally concentrated matrix
updates, because principal sine is larger than normalized angle throughout
the interior of `[0,1]`. It may hurt when the orthogonal polar component is
weak-mode or bfloat16 noise, when early gradients are incoherent, or when the
stronger rotation erases useful Adam anisotropy.

## Preregistered screen and mechanism experiment

Run unchanged-source PSPR at E0 and E2 to exact update 5,000. Compare exact
all-subset Paloma micro/macro BPB and fixed 16,777,216-token training loss
against frozen AGPR screen checkpoints. Promote only if the 1% macro safety
gate passes and the geometric micro-BPB/loss gain is strictly above one at
both scales. If promoted, run one frozen E0--E5 ladder and use selected AGPR
E0/E2/E5 attempts as reduced controls, requiring positive exact-scoring-step
mechanism gain at every anchor. No observed result may change these equations
or gates.

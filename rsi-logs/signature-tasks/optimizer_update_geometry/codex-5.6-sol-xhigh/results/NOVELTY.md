# Novelty preregistration: Principal-Sine Polar Residual

Candidate `opt-014-pspr` uses source inventory
`cd8e55751bd410fc677ac5cee590a8d48f292ee309b4760719f2240175384661`
unchanged across E0--E5.

## Exact update and state equations

Gradients are independently clipped over the AdamH matrix group and Adam
vector/embedding group exactly as in the locked control. Every parameter uses

\[
t\leftarrow t+1,\quad
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]

\[
P_t={m_t/(1-\beta_1^t)\over
\sqrt{v_t/(1-\beta_2^t)}+\epsilon}.
\]

Vectors and embeddings use unchanged Adam. For each semantic logical matrix,
flatten leading axes into rows and transpose when rows exceed columns. Set

\[
X_0={P_t\over\max(\lVert P_t\rVert_F,10^{-15})}
\]

and run exactly three optimizer-owned bfloat16 quintic Newton--Schulz steps

\[
A_k=X_kX_k^\top,\qquad
X_{k+1}=3.4445X_k+(-4.7750A_k+2.0315A_k^2)X_k.
\]

Transpose back and cast to fp32 to obtain `Q_t`, then norm-match to Adam:

\[
\bar Q_t=Q_t{\lVert P_t\rVert_F\over
\max(\lVert Q_t\rVert_F,10^{-15})}.
\]

Measure cosine agreement and the principal-sine amplitude

\[
\rho_t=\operatorname{clip}\!\left(
{\langle P_t,\bar Q_t\rangle_F\over
\max(\lVert P_t\rVert_F\lVert\bar Q_t\rVert_F,10^{-30})},0,1\right),
\quad
\omega_t=\sqrt{\max(1-\rho_t^2,0)}.
\]

For unit directions, `omega_t` is exactly the Frobenius norm of the polar
target component orthogonal to Adam. PSPR's matrix direction is

\[
D_t=(1-\omega_t)P_t+\omega_t\bar Q_t.
\]

Finally, with `r_t=||W_t||_F`, use locked AdamH retraction

\[
\widetilde W_t=W_t-\eta_t r_t
{D_t\over\max(\lVert D_t\rVert_F,10^{-10})},\qquad
W_{t+1}=r_t{\widetilde W_t\over
\max(\lVert\widetilde W_t\rVert_F,10^{-10})}.
\]

Norms, cosine, principal sine, mixing, and retraction are fp32; only the three
polar polynomial steps are bfloat16. Persistent state is unchanged: one CPU
fp32 `step` scalar and tensor-shaped `exp_avg` and `exp_avg_sq` in
master-parameter dtype. PSPR adds no persistent or uncheckpointed recurrence.

## Tensor grouping and serialization

- Rank-0/1 tensors and names containing `embedding`: unchanged Adam.
- `linear_qkv.weight`: independent flattened logical Q, K, and V matrices.
- `linear_fc1.weight`: independent logical SwiGLU gate and up matrices.
- Other rank-2-or-higher tensors: one logical matrix.

Grouping uses rank and semantic suffixes only. Megatron's fp16 optimizer owns
fp32 masters and `torch_dist` serialization of moments, step, group algorithm,
layout metadata, and masters. No rung label, exact dimension, parameter count,
attempt identifier, or benchmark answer enters the rule.

## Closest reduced control

The closest reduced control is exact AGPR, source inventory
`dbf1b294615e05dcb60ac528fbbe7ad606342774d336df7216531421427ea832`.
It shares clipping, moments, schedules, grouping, polar target, precision,
state, serialization, and retraction. AGPR differs only in using

\[
\omega_t^{\rm AGPR}={2\over\pi}\arccos(\rho_t).
\]

Replacing principal sine with normalized principal angle recovers the reduced
control exactly.

## Term-by-term novelty claim

- Adam moments, AdamH fixed-Frobenius retraction, and the quintic polar
  approximation are established components and are not claimed as new.
- AGPR's norm-matched target and angular gate are the closest prior rule in
  this experiment and are not claimed as PSPR's new terms.
- PSPR's claimed transformation is the orthogonal-amplitude coupling
  `omega_t=sqrt(max(1-rho_t^2,0))`. It makes transport proportional to the
  Euclidean norm of the polar component that cannot be represented along the
  current Adam direction. This is an update- and tensor-dependent matrix
  preconditioner, not a learning-rate, momentum, epsilon, decay, clipping,
  schedule, or mask change.

## Predicted helpful and harmful regimes

PSPR should help when AGPR under-corrects spectrally concentrated matrix
updates because principal sine exceeds normalized principal angle throughout
the interior of `[0,1]`. It may hurt when the orthogonal polar component is a
weak-mode or bfloat16 artifact, when early gradients are incoherent, or when
the stronger rotation erases useful Adam anisotropy.

## Preregistered mechanism-on/off experiment

Before either screen, PSPR was registered to beat exact AGPR at update 5,000
on both E0 and E2 in the geometric micro-BPB/fixed-window-loss ratio, with the
1% macro safety gate. Both screens passed. The frozen E0--E5 ladder and exact
declared-scoring comparisons at E0, E2, and E5 also pass with strictly positive
mechanism gains. No observed result altered the equations, grouping, state,
precision, or source inventory above.

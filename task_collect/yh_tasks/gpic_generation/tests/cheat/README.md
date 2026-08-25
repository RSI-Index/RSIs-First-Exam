# Anti-cheat battery

Adversarial verification that the frozen verifier penalizes cheating. It
builds an honest control plus one submission per hack vector and runs each
through the real `tests/{policy_check,evaluate,score}.py`, asserting every
cheat scores `reward = 0.0` (at its intended gate) while the honest control
reaches scoring. A toy model (deterministic image = f(checkpoint, seed,
key)) stands in for the 1.1B PixelGen model so the battery runs in ~1 min on
one GPU — the spot gate is model-agnostic, so this faithfully tests whether
cheats are caught.

Run (needs torch + gpic_eval + a warm DINOv2 hub cache; paths in the scripts
point at the author's staging and must be adjusted per environment):

```bash
bash battery_payload.sh   # inside the tests-equivalent container/venv
```

`battery_results.json` is the recorded 13/13 pass from 2026-08-24 (each row:
case, reward, ok, and the gate/reason that penalized it).

Not a Harbor entrypoint — a developer harness. When the baseline is sealed,
re-run it with the real checkpoint to calibrate `GPIC_SPOT_BIND_MAX_COS` /
`GPIC_SPOT_PERTURB_STD` from the model's measured clean-vs-perturbed cosine.

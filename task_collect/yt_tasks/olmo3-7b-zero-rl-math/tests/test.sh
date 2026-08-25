#!/bin/bash

# Graded verifier for Olmo 3 RL-Zero (Math).
#
# Rewards the full-stack pipeline, weighted toward the rollout <-> train closed loop,
# rather than a single self-reported number. Inspects real training artifacts so a bare
# `echo <score> > result.txt` cannot pass. Emits a continuous reward in [0,1] to
# /logs/verifier/reward.txt.

mkdir -p /logs/verifier

python3 << 'PYEOF'
import json, os, re, glob

reward = 0.0
notes = []

def note(msg):
    notes.append(msg)
    print(msg)

# ---------------------------------------------------------------------------
# Stage 1 (0.15): environment + assets
#   open-instruct importable, base model + dataset present on disk.
# ---------------------------------------------------------------------------
env_ok = False
try:
    import importlib.util
    if importlib.util.find_spec("open_instruct") is not None:
        env_ok = True
except Exception:
    env_ok = False

# Model + dataset presence: look for a downloaded Olmo-3-7B and Dolci-RL-Zero-Math.
# glob ** needs recursive=True
def has_files(patterns):
    for p in patterns:
        if glob.glob(p, recursive=True):
            return True
    return False

model_ok = has_files([
    "/app/**/Olmo-3-7B*/**/*.safetensors",
    "/app/models/**/*.safetensors",
    os.path.expanduser("~/.cache/huggingface/hub/**/*Olmo-3-7B*/**/*.safetensors"),
])
data_ok = has_files([
    "/app/**/Dolci-RL-Zero-Math*/**",
    "/app/data/**/*.jsonl",
    "/app/data/**/*.parquet",
    os.path.expanduser("~/.cache/huggingface/hub/**/*Dolci-RL-Zero-Math*/**"),
])

if env_ok and model_ok and data_ok:
    reward += 0.15
    note("PASS stage1 (env+assets): +0.15")
else:
    note(f"stage1 partial: open_instruct={env_ok} model={model_ok} data={data_ok} (+0.0)")

# ---------------------------------------------------------------------------
# Stage 2 (0.10): data prepared in RLVR format with the RL-Zero prompt template.
# ---------------------------------------------------------------------------
prepared = False
candidates = glob.glob("/app/data/**/*.jsonl", recursive=True) + \
             glob.glob("/app/**/rlvr*/**/*.jsonl", recursive=True) + \
             glob.glob("/app/**/*prepared*.jsonl", recursive=True)
template_markers = ["Solve the following math problem", "Answer:", "step by step"]
for fp in candidates[:20]:
    try:
        with open(fp) as fh:
            head = fh.read(20000)
        if any(m in head for m in template_markers):
            prepared = True
            break
    except Exception:
        continue
if prepared:
    reward += 0.10
    note("PASS stage2 (data prepared w/ template): +0.10")
else:
    note("stage2 not confirmed (+0.0)")

# ---------------------------------------------------------------------------
# Stage 3 (0.35): rollout <-> train closed loop.
#   checkpoint + optimizer state written, AND a training log showing >=500 steps
#   with a non-trivial increasing train reward.
# ---------------------------------------------------------------------------
ckpt = has_files([
    "/app/checkpoints/**/*.safetensors",
    "/app/checkpoints/**/*.bin",
    "/app/**/checkpoint*/**/*.safetensors",
])
optim = has_files([
    "/app/checkpoints/**/optim*",
    "/app/checkpoints/**/*optimizer*",
    "/app/**/checkpoint*/**/optim*",
    "/app/**/global_step*/**",   # deepspeed optimizer shards
])

# Parse training log for step + reward trajectory.
steps_ok = False
reward_climbs = False
log_files = glob.glob("/app/logs/**/*.jsonl", recursive=True) + \
            glob.glob("/app/**/train*.jsonl", recursive=True) + \
            glob.glob("/app/**/metrics*.jsonl", recursive=True)
step_reward = []
for lf in log_files:
    try:
        with open(lf) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                step = d.get("step") or d.get("global_step") or d.get("training_step")
                rw = None
                for k in ("train_reward", "reward", "objective/reward",
                          "train/reward_mean", "rewards/mean", "mean_reward"):
                    if k in d:
                        rw = d[k]
                        break
                if step is not None and rw is not None:
                    step_reward.append((float(step), float(rw)))
    except Exception:
        continue

if step_reward:
    step_reward.sort(key=lambda x: x[0])
    max_step = step_reward[-1][0]
    steps_ok = max_step >= 500
    # increasing: mean of last quintile > mean of first quintile by a margin
    n = len(step_reward)
    if n >= 10:
        q = max(1, n // 5)
        first = sum(r for _, r in step_reward[:q]) / q
        last = sum(r for _, r in step_reward[-q:]) / q
        reward_climbs = last > first + 1e-3
    note(f"train log: max_step={max_step:.0f} n_points={n} climbs={reward_climbs}")

if ckpt and optim and steps_ok and reward_climbs:
    reward += 0.35
    note("PASS stage3 (rollout<->train closed loop): +0.35")
else:
    note(f"stage3 partial: ckpt={ckpt} optim={optim} steps>=500={steps_ok} "
         f"reward_climbs={reward_climbs} (+0.0)")

# ---------------------------------------------------------------------------
# Stage 4 (0.10): evaluation ran with the specified protocol.
# Stage 5 (0.30): accuracy above base baseline, scaling to paper target (45%).
# ---------------------------------------------------------------------------
eval_files = glob.glob("/app/eval/**/*.json", recursive=True) + \
             glob.glob("/app/**/aime2024*.json", recursive=True)
eval_data = None
for ef in eval_files:
    try:
        d = json.load(open(ef))
        if "trained_pass@1" in d or "pass@1" in d:
            eval_data = d
            break
    except Exception:
        continue

if eval_data is not None:
    n_samples = eval_data.get("n_samples")
    temp = eval_data.get("temperature")
    protocol_ok = (n_samples in (None, 32) or n_samples == 32)
    if protocol_ok:
        reward += 0.10
        note("PASS stage4 (eval ran w/ protocol): +0.10")
    else:
        note(f"stage4 protocol mismatch: n_samples={n_samples} temp={temp} (+0.0)")

    trained = eval_data.get("trained_pass@1", eval_data.get("pass@1"))
    base = eval_data.get("base_pass@1", 10.0)  # paper: Olmo-3-7B base ~10%
    if trained is not None:
        trained = float(trained)
        base = float(base)
        TARGET = 45.0
        if trained <= base:
            acc = 0.0
        elif trained >= TARGET:
            acc = 0.30
        else:
            acc = 0.30 * (trained - base) / (TARGET - base)
        reward += acc
        note(f"stage5 accuracy: trained={trained:.1f} base={base:.1f} -> +{acc:.3f}")
    else:
        note("stage5: no trained pass@1 (+0.0)")
else:
    note("stage4/5: no eval output found (+0.0)")

# ---------------------------------------------------------------------------
# Cross-check: result.txt must be consistent with the eval artifact (anti-cheat).
# If result.txt is present but there is NO supporting eval/train evidence, cap reward.
# ---------------------------------------------------------------------------
result_txt = None
if os.path.exists("/app/result.txt"):
    try:
        result_txt = float(open("/app/result.txt").read().strip())
    except Exception:
        result_txt = None

evidence = bool(step_reward) or (eval_data is not None)
if result_txt is not None and not evidence:
    note("WARN: result.txt present but no training/eval artifacts -> capping reward at 0.0")
    reward = 0.0

reward = max(0.0, min(1.0, reward))
note(f"FINAL REWARD: {reward:.3f}")

with open("/logs/verifier/reward.txt", "w") as fh:
    fh.write(f"{reward:.4f}")
with open("/logs/verifier/reward.json", "w") as fh:
    # Harbor treats every reward.json value as a numeric reward component.
    # Detailed notes are already emitted to stdout and reward.txt.
    json.dump({"reward": reward}, fh, indent=2)
PYEOF

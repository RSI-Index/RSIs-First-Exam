# TMAX adaptive-training autoresearch task

Harbor task for improving a fixed Qwen3.5-9B terminal agent through adaptive
experimentation. The Codex research agent chooses each attempt length and may
change the RL method, public training-data strategy, training prompt, and
training-only reward.

The selected attempt must reach 200 updates / 51,200 trajectories on the
official 64-H100 topology (16 learner GPUs and 48 inference GPUs). The
official `allenai/tmax-9b` `step_200` checkpoint is used directly as the
baseline rather than retrained.

Final evaluation is a clean, matched comparison. The candidate cannot access
or change the sealed tasks, final Vanillux prompt, original programmatic
reward/verifiers, tools, parser, or decoding configuration.

See `instruction.md` for the research loop, `policy.yaml` for the integrity
boundary, and `cluster/launch_harbor_lsf.sh` for the Blue Vela launcher.

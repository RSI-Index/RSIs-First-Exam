# Blue Vela LSF execution context

This trial has a 9-node allocation with 8 NVIDIA H100-80GB GPUs per node
(72 GPUs total). Harbor's primary container runs on the first allocated node,
and `/app` is a writable GPFS bind shared by all allocated nodes.

Use `LSB_MCPU_HOSTS` to obtain the unique host list. Launch one process group or
Ray worker per node with `/opt/share/ETELSFSH/10.1/linux4.18-glibc2.28-x86_64/bin/blaunch`.
For every remote node, enter the same image with:

```bash
APPTAINER=/proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin/apptainer
IMAGE=/proj/datasets/interns/yuetai/agent_envs/scale_autoresearch_tasks/images/posttrain-olmo3-environment.sif
# $RUN_WORKSPACE is exported by the LSF wrapper and is already mounted at /app
$APPTAINER exec --nv --bind "$RUN_WORKSPACE:/app,/proj:/proj,/opt/share/ETELSFSH:/opt/share/ETELSFSH:ro" "$IMAGE" <command>
```

All source changes, virtual environments, downloads, logs, checkpoints, Ray
state needed by other nodes, and final deliverables must live below `/app`.
Image-overlay changes under `/opt` or `/usr` on the head node are not shared.
The cluster adaptation changes only scheduler/container launch mechanics; keep
the task's 8-learner + 64-actor topology and effective training hyperparameters.

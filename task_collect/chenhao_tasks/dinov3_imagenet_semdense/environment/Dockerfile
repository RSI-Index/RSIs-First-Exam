FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

ARG DINOV3_COMMIT=6876159a11b4df116f30f667f8c9888617df0751
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Pinned scientific source. /opt/project is the pristine tree the verifier
# diffs against; /app/project is the agent's writable copy.
RUN git clone https://github.com/facebookresearch/dinov3.git /opt/project/dinov3 \
    && git -C /opt/project/dinov3 checkout "${DINOV3_COMMIT}" \
    && test "$(git -C /opt/project/dinov3 rev-parse HEAD)" = "${DINOV3_COMMIT}" \
    && chmod -R a+rX /opt/project

# Repo requirements are light (ftfy, omegaconf, regex, scikit-learn,
# submitit, termcolor, torch, torchmetrics, torchvision). torch comes from
# the base image (>=2.7.1 required by the repo); the editable install and
# the import check below fail the build on breakage.
RUN python -m pip install -r /opt/project/dinov3/requirements.txt \
    && python -m pip install -e /opt/project/dinov3 \
    && python -m pip install tensorboard \
    && python -c "import torch, dinov3; \
v = torch.__version__.split('+')[0]; \
assert tuple(map(int, v.split('.')[:2])) >= (2, 7), v"

COPY --chmod=755 task-tools/dinov3_async_run.py /task-tools/dinov3_async_run.py
COPY --chmod=755 task-tools/dinov3_task_tool.py /task-tools/dinov3_task_tool.py
COPY --chmod=755 task-tools/dinov3_local_eval.py /task-tools/dinov3_local_eval.py
COPY --chmod=755 launch_dinov3_4x8.sh /task-tools/launch_dinov3_4x8.sh

# Contract defaults; the harness may override via task.toml [environment.env].
ENV DINO_GPU_HOURS_CAP=450.0 \
    DINO_TOPOLOGY_GPUS=32 \
    DINO_IMAGENET_DIR=/datasets/imagenet-1k \
    DINO_ADE20K_DIR=/datasets/ade20k \
    DINO_NYU_DIR=/datasets/nyu-depth-v2

RUN mkdir -p /app/output /logs/artifacts \
    && cp -a /opt/project /app/project \
    && chmod -R a+rwX /app/project /app/output
WORKDIR /app/project/dinov3
CMD ["bash"]

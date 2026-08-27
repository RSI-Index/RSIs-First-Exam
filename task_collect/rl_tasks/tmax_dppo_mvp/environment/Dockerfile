FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

ARG TMAX_COMMIT=7387d2f9142397a458dc39f0827a2ab0b4c03cda
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 WANDB_MODE=offline \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl podman crun uidmap fuse-overlayfs slirp4netns \
    netavark aardvark-dns && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.10.9
COPY project /opt/project
RUN git -C /opt/project merge-base --is-ancestor "$TMAX_COMMIT" HEAD && \
    git -C /opt/project diff --check && \
    cd /opt/project/training/open-instruct && uv sync --frozen
COPY --chmod=755 train_tmax.sh /task-tools/train_tmax.sh
COPY --chmod=755 task-tools /task-tools
COPY training-defaults /opt/training-defaults
RUN mkdir -p /app/project /app/output /app/training/data /app/training/prompt /app/training/reward && \
    cp -a /opt/project/. /app/project/ && \
    cp /opt/project/training/open-instruct/scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt \
      /app/training/prompt/system.txt && \
    cp /opt/training-defaults/reward/reward.py /app/training/reward/reward.py && \
    cp /opt/training-defaults/recipe.json /app/training/recipe.json
WORKDIR /app/project
CMD ["bash"]

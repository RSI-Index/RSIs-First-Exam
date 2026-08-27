FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git git-lfs jq ninja-build python3 python3-dev python3-pip tmux \
    && rm -rf /var/lib/apt/lists/*
COPY project /opt/project
RUN python3 -m pip install --no-cache-dir --upgrade pip "setuptools>=64" wheel \
    && python3 -m pip install --no-cache-dir -e "/opt/project[train,eval]" \
    && python3 -m pip install --no-cache-dir safetensors einops easydict pymupdf protobuf==3.20.* \
    && python3 -m pip install --no-cache-dir flash-attn==2.5.8 --no-build-isolation

RUN mkdir -p /app/project /app/output /logs/artifacts \
    && cp -a /opt/project/. /app/project/ \
    && cd /app/project && git init -q && git add -A \
    && git -c user.name=baseline -c user.email=baseline@local commit -qm BASE
WORKDIR /app/project
CMD ["bash"]

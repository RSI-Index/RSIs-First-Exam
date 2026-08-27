FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git jq python3 python3-dev python3-pip tmux \
    && rm -rf /var/lib/apt/lists/*
COPY project /opt/project
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /opt/project/requirements.txt

RUN mkdir -p /app/project /app/output /logs/artifacts \
    && cp -a /opt/project/. /app/project/ \
    && cd /app/project && git init -q && git add -A \
    && git -c user.name=baseline -c user.email=baseline@local commit -qm BASE
WORKDIR /app/project
CMD ["bash"]


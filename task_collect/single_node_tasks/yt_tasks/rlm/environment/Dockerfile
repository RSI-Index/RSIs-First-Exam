FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git jq python3 python3-dev python3-pip tmux \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /usr/local/bin/
COPY project /opt/project
RUN git config --global url."https://github.com/".insteadOf "git@github.com:" \
    && git config --global --add url."https://github.com/".insteadOf "ssh://git@github.com/" \
    && git clone https://github.com/PrimeIntellect-ai/prime-rl.git /opt/prime-rl \
    && git -C /opt/prime-rl checkout 84f072bc99a2cd0b740ff9e665e1accf55fb6cf4 \
    && git -C /opt/prime-rl submodule update --init --recursive \
       deps/pydantic-config deps/verifiers deps/renderers deps/research-environments \
    && cd /opt/prime-rl && uv sync --frozen --all-extras \
    && uv pip install --python /opt/prime-rl/.venv/bin/python \
       -e /opt/project -e /opt/project/training -e /opt/project/training/environments/oolong

ENV VIRTUAL_ENV=/opt/prime-rl/.venv \
    PATH=/opt/prime-rl/.venv/bin:$PATH

RUN mkdir -p /app/project /app/output /logs/artifacts \
    && cp -a /opt/project/. /app/project/ \
    && cd /app/project && git init -q && git add -A \
    && git -c user.name=baseline -c user.email=baseline@local commit -qm BASE
WORKDIR /app/project
CMD ["bash"]

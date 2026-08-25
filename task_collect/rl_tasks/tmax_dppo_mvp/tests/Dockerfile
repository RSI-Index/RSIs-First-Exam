FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

ARG TMAX_COMMIT=7387d2f9142397a458dc39f0827a2ab0b4c03cda
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl podman crun uidmap fuse-overlayfs slirp4netns netavark \
    aardvark-dns docker.io docker-compose-v2 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.10.9
COPY project /opt/project
RUN test "$(git -C /opt/project rev-parse HEAD)" = "$TMAX_COMMIT" && \
    cd /opt/project && uv sync --frozen && \
    uv pip install vllm==0.19.1 transformers pyyaml
COPY policy.yaml /tests/policy.yaml
COPY policy_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh
RUN mkdir -p /logs/verifier /app/output
WORKDIR /opt/project
CMD ["bash"]


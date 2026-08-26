FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ARG DATACOMP_COMMIT=4a8df1992566ef8334773f7152e1855b1f716162
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_MODE=offline \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git jq libgl1 libglib2.0-0 libprotobuf-dev \
    ninja-build protobuf-compiler python3-dev python3-packaging python3-venv wget \
    && rm -rf /var/lib/apt/lists/*
RUN git init /opt/project \
    && git -C /opt/project remote add origin https://github.com/mlfoundations/datacomp.git \
    && git -C /opt/project fetch --depth=1 origin "${DATACOMP_COMMIT}" \
    && git -C /opt/project checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/project rev-parse HEAD)" = "${DATACOMP_COMMIT}" \
    && test -z "$(git -C /opt/project status --short)" \
    && grep -Ev '^(clip-benchmark==|torch==|torchvision==|typing-extensions==|nvidia-|pyyaml==)' \
       /opt/project/requirements.txt > /tmp/datacomp-requirements-h100.txt \
    && python -m pip install --no-cache-dir 'cython<3' pyyaml==6.0.1 \
    && python -m pip install --no-cache-dir --no-deps -r /tmp/datacomp-requirements-h100.txt \
    && python -m pip install --no-cache-dir --no-deps pycocotools==2.0.11 pycocoevalcap==1.2 \
    && python -m pip install --no-cache-dir --no-deps clip-benchmark==1.4.0 \
    && python3 -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml

COPY policy.yaml /tests/policy.yaml
COPY policy_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh
RUN mkdir -p /app/project /app/output /logs/verifier \
    && chmod -R a+rX /opt/project /tests
CMD ["bash"]

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
    ninja-build protobuf-compiler python3-dev python3-packaging wget \
    && rm -rf /var/lib/apt/lists/*

COPY project /opt/project
RUN test "$(git -C /opt/project rev-parse HEAD)" = "${DATACOMP_COMMIT}" \
    && test -z "$(git -C /opt/project status --short)" \
    && git -C /opt/project diff --check \
    && grep -Ev '^(clip-benchmark==|torch==|torchvision==|typing-extensions==|nvidia-|pyyaml==)' \
       /opt/project/requirements.txt > /tmp/datacomp-requirements-h100.txt \
    && python -m pip install --no-cache-dir 'cython<3' pyyaml==6.0.1 \
    && python -m pip install --no-cache-dir --no-deps -r /tmp/datacomp-requirements-h100.txt \
    && python -m pip install --no-cache-dir --no-deps pycocotools==2.0.11 pycocoevalcap==1.2 \
    && python -m pip install --no-cache-dir --no-deps clip-benchmark==1.4.0 \
    && python -c "import importlib.metadata, open_clip, torch, torchvision; assert torch.__version__.split('+')[0] == '2.3.1'; assert torchvision.__version__.split('+')[0] == '0.18.1'; assert importlib.metadata.version('open-clip-torch') == '2.16.1'; assert importlib.metadata.version('webdataset') == '0.2.31'; assert importlib.metadata.version('clip-benchmark') == '1.4.0'"

COPY --chmod=755 datacomp_task_tool.py /task-tools/datacomp_task_tool.py
RUN mkdir -p /app/output /app/project \
    && cp -a /opt/project/. /app/project/ \
    && chmod -R a+rX /app/project /task-tools
WORKDIR /app/project
CMD ["bash"]

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG OPEN_CLIP_COMMIT=db4d4911cc0fc4bc58bc98579c52875750450530
ARG DATACOMP_COMMIT=4a8df1992566ef8334773f7152e1855b1f716162
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Pinned scientific sources. /opt/project is the pristine tree the verifier
# diffs against; /app/project is the agent's writable copy.
RUN git clone https://github.com/mlfoundations/open_clip.git /opt/project/open_clip \
    && git -C /opt/project/open_clip checkout "${OPEN_CLIP_COMMIT}" \
    && test "$(git -C /opt/project/open_clip rev-parse HEAD)" = "${OPEN_CLIP_COMMIT}" \
    && git clone https://github.com/mlfoundations/datacomp.git /opt/project/datacomp \
    && git -C /opt/project/datacomp checkout "${DATACOMP_COMMIT}" \
    && test "$(git -C /opt/project/datacomp rev-parse HEAD)" = "${DATACOMP_COMMIT}" \
    && chmod -R a+rX /opt/project

# Dependency layering: datacomp's 2023-era pins are installed without deps so
# they cannot downgrade torch; open_clip training deps are installed last and
# win conflicts. Exact resolution is re-verified at image build (the final
# import check below fails the build on breakage).
RUN python -m pip install -r /opt/project/open_clip/requirements-training.txt \
    && python -m pip install -e /opt/project/open_clip \
    && python -m pip install --no-deps clip-benchmark \
    && python -m pip install cloudpathlib pycocoevalcap scikit-learn wilds \
        fasttext-wheel webdataset braceexpand ftfy regex pandas pyarrow \
        tensorboard huggingface_hub \
    && python -c "import torch, open_clip, webdataset, clip_benchmark; \
assert torch.__version__.split('+')[0] == '2.6.0', torch.__version__"

COPY --chmod=755 task-tools/clip_async_run.py /task-tools/clip_async_run.py
COPY --chmod=755 task-tools/clip_task_tool.py /task-tools/clip_task_tool.py
COPY --chmod=755 task-tools/clip_local_eval.py /task-tools/clip_local_eval.py

# Contract defaults; the harness may override via task.toml [environment.env].
ENV CLIP_GPU_HOURS_CAP=20.0 \
    CLIP_TOPOLOGY_GPUS=8 \
    CLIP_SNAPSHOT_DIR=/datasets/datacomp-medium-snapshot/current \
    DATACOMP_EVAL_DIR=/datasets/datacomp-eval

RUN mkdir -p /app/output /logs/artifacts \
    && cp -a /opt/project /app/project \
    && chmod -R a+rwX /app/project /app/output
WORKDIR /app/project
CMD ["bash"]

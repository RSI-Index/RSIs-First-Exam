FROM ghcr.io/allenai/molmo2:latest

ARG MOLMO2_COMMIT=f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_MODE=offline \
    HF_DATASETS_OFFLINE=1 \
    MOLMO_DATA_DIR=/datasets/molmo2-official/molmo_data \
    HF_HOME=/datasets/molmo2-official/huggingface \
    OLMO_SHARED_FS=1 \
    OMP_NUM_THREADS=8

# The upstream container (repo-provided) ships all dependencies but not the
# code; pin the code at the task commit. The repository runs as released —
# there are no locked source patches for this task.
RUN git clone https://github.com/allenai/molmo2.git /opt/project \
    && git -C /opt/project checkout "${MOLMO2_COMMIT}" \
    && test "$(git -C /opt/project rev-parse HEAD)" = "${MOLMO2_COMMIT}" \
    && git -C /opt/project diff --quiet \
    && chmod -R a+rX /opt/project
RUN pip install --no-cache-dir --no-deps -e /opt/project

COPY --chmod=755 launch_molmo2_1x8_lsf.sh /task-tools/launch_molmo2_1x8_lsf.sh

RUN python -c "import torch, olmo; import torchcodec" \
    && python -c "from olmo.models.molmo2.molmo2 import Molmo2Config; from olmo.models.molmo_point.molmo_point import MolmoPointConfig"

RUN mkdir -p /app/output /logs/artifacts \
    && cp -a /opt/project /app/project \
    && chmod -R a+rwX /app/project
WORKDIR /app/project
CMD ["bash"]

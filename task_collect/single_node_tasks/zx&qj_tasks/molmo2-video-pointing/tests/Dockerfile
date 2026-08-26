FROM ghcr.io/allenai/molmo2:latest

ARG MOLMO2_COMMIT=f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline \
    HF_DATASETS_OFFLINE=1 \
    MOLMO_DATA_DIR=/datasets/molmo2-official/molmo_data \
    HF_HOME=/datasets/molmo2-official/huggingface \
    OLMO_SHARED_FS=1 \
    OMP_NUM_THREADS=8

RUN git clone https://github.com/allenai/molmo2.git /opt/project \
    && git -C /opt/project checkout "${MOLMO2_COMMIT}" \
    && test "$(git -C /opt/project rev-parse HEAD)" = "${MOLMO2_COMMIT}" \
    && git -C /opt/project diff --quiet \
    && chmod -R a+rX /opt/project
RUN pip install --no-cache-dir --no-deps -e /opt/project \
    && python -c "import torch, olmo; from olmo.models.molmo2.molmo2 import Molmo2Config" \
    && python -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml

COPY policy.yaml /tests/policy.yaml
COPY policy_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh
COPY --chmod=755 launch_molmo2_1x8_lsf.sh /task-tools/launch_molmo2_1x8_lsf.sh
RUN mkdir -p /app/project /app/output /logs/verifier
CMD ["bash"]

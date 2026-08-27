FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

ARG DINOV3_COMMIT=6876159a11b4df116f30f667f8c9888617df0751
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 WANDB_MODE=offline
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Pristine pinned tree: the policy gate diffs the candidate workspace
# against this, and the frozen probe drivers run from here (never from the
# agent-editable copy).
RUN git clone https://github.com/facebookresearch/dinov3.git /opt/project/dinov3 \
    && git -C /opt/project/dinov3 checkout "${DINOV3_COMMIT}" \
    && test "$(git -C /opt/project/dinov3 rev-parse HEAD)" = "${DINOV3_COMMIT}" \
    && chmod -R a+rX /opt/project

RUN python -m pip install -r /opt/project/dinov3/requirements.txt \
    && python -m pip install -e /opt/project/dinov3 \
    && python -m pip install pyyaml \
    && python -c "import torch, dinov3; \
v = torch.__version__.split('+')[0]; \
assert tuple(map(int, v.split('.')[:2])) >= (2, 7), v" \
    && python3 -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml

COPY policy.yaml /tests/policy.yaml
COPY policy_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh
RUN mkdir -p /app/project /app/output /logs/verifier
CMD ["bash"]

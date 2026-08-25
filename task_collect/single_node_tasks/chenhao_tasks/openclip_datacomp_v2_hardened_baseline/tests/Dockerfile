FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG OPEN_CLIP_COMMIT=db4d4911cc0fc4bc58bc98579c52875750450530
ARG DATACOMP_COMMIT=4a8df1992566ef8334773f7152e1855b1f716162
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 WANDB_MODE=offline
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Pristine pinned tree: the policy gate diffs the candidate workspace against
# this, and the frozen evaluator runs the pinned datacomp protocol from here
# (never from the agent-editable copy).
RUN git clone https://github.com/mlfoundations/open_clip.git /opt/project/open_clip \
    && git -C /opt/project/open_clip checkout "${OPEN_CLIP_COMMIT}" \
    && test "$(git -C /opt/project/open_clip rev-parse HEAD)" = "${OPEN_CLIP_COMMIT}" \
    && git clone https://github.com/mlfoundations/datacomp.git /opt/project/datacomp \
    && git -C /opt/project/datacomp checkout "${DATACOMP_COMMIT}" \
    && test "$(git -C /opt/project/datacomp rev-parse HEAD)" = "${DATACOMP_COMMIT}" \
    && chmod -R a+rX /opt/project

RUN python -m pip install -r /opt/project/open_clip/requirements-training.txt \
    && python -m pip install -e /opt/project/open_clip \
    && python -m pip install --no-deps clip-benchmark \
    && python -m pip install cloudpathlib pycocoevalcap scikit-learn wilds \
        fasttext-wheel webdataset braceexpand ftfy regex pandas pyarrow \
        huggingface_hub pyyaml \
    && python -c "import torch, open_clip, webdataset, clip_benchmark; \
assert torch.__version__.split('+')[0] == '2.6.0', torch.__version__" \
    && python3 -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml

COPY policy.yaml /tests/policy.yaml
COPY policy_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh
RUN mkdir -p /app/project /app/output /logs/verifier
CMD ["bash"]

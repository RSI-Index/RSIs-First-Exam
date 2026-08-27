FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ARG DCLM_COMMIT=361714bdd60bb9b7f4b2d8354cebbf0dec0c329e
ARG OPEN_LM_COMMIT=87955f56980d0481379a05f768f48c5d525e5fd5
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_MODE=offline \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git jq unzip \
    && rm -rf /var/lib/apt/lists/*
COPY ../environment/project /opt/project
COPY ../environment/requirements-eval.txt /opt/task-config/requirements-eval.txt
RUN test "$(git -C /opt/project rev-parse HEAD)" = "${DCLM_COMMIT}" \
    && test "$(git -C /opt/project status --short --untracked-files=all)" = \
        "?? rust_processing/tokshuf-rs/Cargo.lock" \
    && python -m pip install --no-cache-dir -r /opt/task-config/requirements-eval.txt \
    && python -m pip install --no-cache-dir --no-deps xformers==0.0.27 \
    && python -m pip install --no-cache-dir --no-deps \
        "git+https://github.com/mlfoundations/open_lm.git@${OPEN_LM_COMMIT}" \
    && python - <<'PY'
import importlib.metadata
import llmfoundry
import open_lm
import torch
import transformers
import xformers

assert torch.__version__.split('+')[0] == '2.3.1', torch.__version__
assert importlib.metadata.version('llm-foundry') == '0.9.0'
assert importlib.metadata.version('open-lm') == '0.0.33'
assert transformers.__version__ == '4.40.2'
assert xformers.__version__ == '0.0.27'
print('DCLM_EVAL_IMPORT_SMOKE_OK')
PY
RUN mkdir -p /logs/verifier /tests /task-tools
WORKDIR /opt/project
CMD ["bash"]

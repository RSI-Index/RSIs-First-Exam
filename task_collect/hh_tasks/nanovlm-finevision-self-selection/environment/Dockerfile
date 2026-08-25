# Agent image. NO /eval-data mount exists here, and no lmms-eval is installed: the agent cannot
# run the scored benchmark even by accident, because the benchmark is not in this image.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG NANOVLM_COMMIT=4e0c0961846135c2217f95e54cb4c2d66eb55e42
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 WANDB_MODE=offline \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git \
    && rm -rf /var/lib/apt/lists/*

# /opt/project is the PRISTINE BASE: upstream at the pinned commit plus the adapter. It is
# read-only; the agent works in /app/project, which the harness recreates from here every round
# (R-1), so nothing survives a round except /carry.
COPY environment/rsi_adapter.patch /opt/rsi_adapter.patch
RUN git clone https://github.com/huggingface/nanoVLM.git /opt/project \
    && git -C /opt/project checkout "${NANOVLM_COMMIT}" \
    && test "$(git -C /opt/project rev-parse HEAD)" = "${NANOVLM_COMMIT}" \
    && git -C /opt/project apply --check /opt/rsi_adapter.patch \
    && git -C /opt/project apply /opt/rsi_adapter.patch \
    && chmod -R a+rX /opt/project

# The nanoVLM README's own dependency list. wandb is installed but WANDB_MODE=offline and the
# frozen recipe sets log_wandb=False; upstream defaults it True and would block on login.
RUN python -m pip install \
      numpy torchvision pillow datasets huggingface-hub transformers wandb pyyaml

COPY environment/task-tools/ /task-tools/
COPY environment/assets.yaml /assets/manifest.yaml
COPY tests/upstream_check.py /opt/upstream_check.py
COPY contract.yaml /opt/contract.yaml

# Verified in BOTH images, with no `|| true` and no `|| echo` anywhere near it. If only the
# verifier checked the patch, an agent image built from a drifted base would look correct to
# itself while every diff the policy gate takes would be against the wrong tree. And a check whose
# failure is swallowed by a fallback echo is not a check -- it is a log line.
RUN python /opt/upstream_check.py --project /opt/project --contract /opt/contract.yaml

RUN mkdir -p /app/project /app/output /carry \
             /datasets/finevision_pool /models
WORKDIR /app/project
CMD ["bash"]

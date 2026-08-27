# Agent image. NO /eval-data mount exists here: the agent cannot touch the evaluation sets even by
# accident, because they are not in this image.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG DATACOMP_COMMIT=4a8df1992566ef8334773f7152e1855b1f716162
# No default -- see tests/Dockerfile. The agent trains with this open_clip, so if the two images
# disagreed the verifier would be scoring a model the recipe never described.
ARG OPEN_CLIP_VERSION

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git \
    && rm -rf /var/lib/apt/lists/*

# /opt/project is the PRISTINE BASE, unpatched. Read-only; the agent works in /app/project, which the
# harness recreates from here every round (R-1), so nothing survives a round except /carry.
RUN git clone https://github.com/mlfoundations/datacomp.git /opt/project \
    && git -C /opt/project checkout "${DATACOMP_COMMIT}" \
    && test "$(git -C /opt/project rev-parse HEAD)" = "${DATACOMP_COMMIT}" \
    && chmod -R a+rX /opt/project

RUN test -n "${OPEN_CLIP_VERSION}" \
    && python -m pip install \
         "open_clip_torch==${OPEN_CLIP_VERSION}" \
         webdataset pandas pyarrow pyyaml numpy scipy scikit-learn \
    && python -m pip install -r /opt/project/requirements.txt

COPY environment/task-tools/ /task-tools/
COPY environment/assets.yaml /assets/manifest.yaml
COPY tests/upstream_check.py /opt/upstream_check.py
COPY contract.yaml /opt/contract.yaml

# Verified in BOTH images, with no `|| true` and no fallback echo anywhere near it. If only the
# verifier checked the recipe, an agent image built against a changed SCALE_CONFIGS would train
# off-budget and look correct to itself. A check whose failure is swallowed is a log line.
RUN python /opt/upstream_check.py --project /opt/project --contract /opt/contract.yaml

RUN mkdir -p /app/project /app/output /carry \
             /datasets/commonpool/metadata /datasets/commonpool/shards
WORKDIR /app/project
CMD ["bash"]

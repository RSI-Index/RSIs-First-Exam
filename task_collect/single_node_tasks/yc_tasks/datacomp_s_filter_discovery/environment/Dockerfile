FROM pytorch/pytorch@sha256:b2bbeac616a0adddb3479269b8106c05ea9cdfcbc62a738105f31f8fd5da5138

ARG DATACOMP_COMMIT=4a8df1992566ef8334773f7152e1855b1f716162
ARG OPEN_CLIP_COMMIT=395851ee9c55fdddb63f5579955cdd6d754d261c
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    WANDB_MODE=disabled \
    HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ffmpeg git libglib2.0-0 libgl1 libmagic1 libsm6 \
      libxext6 libxrender1 libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/datacomp \
    && git -C /opt/datacomp remote add origin https://github.com/mlfoundations/datacomp.git \
    && git -C /opt/datacomp fetch --depth=1 origin "${DATACOMP_COMMIT}" \
    && git -C /opt/datacomp checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/datacomp rev-parse HEAD)" = "${DATACOMP_COMMIT}" \
    && test "$(git -C /opt/datacomp rev-parse HEAD^{tree})" = "177afe8164d943d702a7ac28703108521ab168d7"

RUN git init /opt/open_clip \
    && git -C /opt/open_clip remote add origin https://github.com/mlfoundations/open_clip.git \
    && git -C /opt/open_clip fetch --depth=1 origin "${OPEN_CLIP_COMMIT}" \
    && git -C /opt/open_clip checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/open_clip rev-parse HEAD)" = "${OPEN_CLIP_COMMIT}" \
    && test "$(git -C /opt/open_clip rev-parse HEAD^{tree})" = "091e08b0a0a30899afc8b7299edc739df228d098"

COPY patches/initialization_attestation.patch /opt/contracts/initialization_attestation.patch
RUN git -C /opt/open_clip apply --check /opt/contracts/initialization_attestation.patch \
    && git -C /opt/open_clip apply /opt/contracts/initialization_attestation.patch \
    && git -C /opt/open_clip diff --check

COPY requirements.txt /opt/contracts/requirements.txt
RUN python -m pip install -r /opt/contracts/requirements.txt \
    && python -m pip install --no-deps clip-benchmark==1.4.0 img2dataset==1.40.0 \
    && python -m pip install --no-deps --no-build-isolation /opt/open_clip

ENV PYTHONPATH=/app/project:/opt/open_clip/src:/opt/datacomp
COPY project /opt/project
RUN cp -a /opt/project /app/project
COPY task_tools /task-tools
COPY baseline_contract.json /opt/contracts/baseline_contract.json
COPY asset_contract.json /opt/contracts/asset_contract.json
COPY --chmod=755 run_candidate.sh /task-tools/run_candidate.sh
RUN mkdir -p /app/output /logs/artifacts /datasets \
    && chmod -R a+rX /opt/datacomp /opt/open_clip /opt/project /task-tools /opt/contracts
WORKDIR /app/project
CMD ["bash"]

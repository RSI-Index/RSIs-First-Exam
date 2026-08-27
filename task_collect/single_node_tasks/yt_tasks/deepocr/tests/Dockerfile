FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/root/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git git-lfs jq ninja-build python3 python3-dev python3-pip python3-venv \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip "setuptools>=64" wheel
RUN python3 -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml
RUN git clone https://github.com/Princeton-AI2-Lab/DeepOCR.git /opt/project \
    && git -C /opt/project checkout cd49a8f6129689cbda94c8cb25cfd128ab69b1f6 \
    && sed -i '/from \.ps3_encoder import PS3VisionTower/d; /^    if config.ps3:$/a\        from .ps3_encoder import PS3VisionTower' /opt/project/llava/model/multimodal_encoder/builder.py \
    && python3 -m pip install --no-cache-dir -e "/opt/project[train,eval]" \
    && python3 -m pip install --no-cache-dir safetensors einops easydict pymupdf protobuf==3.20.* \
    && python3 -m pip install --no-cache-dir flash-attn==2.5.8 --no-build-isolation
RUN git clone https://github.com/opendatalab/OmniDocBench.git /opt/OmniDocBench \
    && git -C /opt/OmniDocBench checkout 2b161d010d2e3aff77a0edef359ea3a6411d23cd \
    && python3 -m venv /opt/omnidoc-venv \
    && /opt/omnidoc-venv/bin/pip install --no-cache-dir -e /opt/OmniDocBench
COPY policy.yaml /tests/policy.yaml
COPY policy_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh

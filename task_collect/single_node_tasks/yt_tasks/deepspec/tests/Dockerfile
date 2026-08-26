FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/root/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git jq python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/deepseek-ai/DeepSpec.git /opt/project \
    && git -C /opt/project checkout 005e03b81cec38b7da6399833d609ee89a2587f2 \
    && python3 -m pip install --break-system-packages --no-cache-dir -r /opt/project/requirements.txt \
    && python3 -m pip install --break-system-packages --no-cache-dir tensorboard \
    && python3 -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml
COPY policy.yaml /tests/policy.yaml
COPY policy_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh

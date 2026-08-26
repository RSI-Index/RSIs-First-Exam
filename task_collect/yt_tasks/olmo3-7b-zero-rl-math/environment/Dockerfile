FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install PyTorch with CUDA support
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core RL / inference stack for open-instruct RLVR:
#   - vLLM   : rollout actors
#   - ray    : disaggregated actor/learner orchestration across nodes
#   - deepspeed / accelerate : learner-side training
#   - flash-attn : long-context (16K) attention
RUN uv pip install --system \
    transformers \
    datasets \
    accelerate \
    deepspeed \
    huggingface_hub \
    hf_transfer \
    vllm \
    ray \
    flash-attn \
    sympy \
    wandb

# Install open-instruct (Olmo 3 post-training framework; contains the OlmoRL / GRPO trainer)
RUN git clone https://github.com/allenai/open-instruct.git /opt/open-instruct \
    && cd /opt/open-instruct \
    && uv pip install --system -e .

ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

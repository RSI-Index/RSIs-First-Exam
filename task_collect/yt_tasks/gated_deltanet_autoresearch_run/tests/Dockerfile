FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ARG GDN_COMMIT=b53d6d3a161267432a79c1c04af69fa52bddc921
ARG FLA_RELEASE_COMMIT=326184c45f42380063fb10bdda31d394b0c8c1b7
ARG DAO_FLASH_ATTN_COMMIT=f86e3dd9192e41dee3814c4cfd8bbce4792e6753
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 WANDB_MODE=offline
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ninja-build python3-packaging python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/NVlabs/GatedDeltaNet.git /opt/project \
    && git -C /opt/project checkout "${GDN_COMMIT}" \
    && chmod -R a+rX /opt/project
COPY apply_h100_compat.py /tmp/apply_h100_compat.py
RUN python /tmp/apply_h100_compat.py \
    && git -C /opt/project diff --check \
    && test "$(git -C /opt/project diff --binary -- . | sha256sum | awk '{print $1}')" = 63c451bf94a954445c89850ae4fd7efd3ae8188483f4974050e4088685e5eba4 \
    && rm /tmp/apply_h100_compat.py
RUN python -m pip install packaging ninja lightning==2.1.2 "lightning[app]" \
    "jsonargparse[signatures]" tokenizers sentencepiece wandb "lightning[data]" \
    torchmetrics tensorboard zstandard pandas pyarrow huggingface_hub \
    datasets==2.20.0 triton==2.3.0 xformers==0.0.27 "tomli>=1.1.0" \
    transformers==4.47.0 einops braceexpand \
    smart_open opt_einsum cbor2 mosaicml-streaming lm-eval==0.4.1
RUN python -m pip install --no-build-isolation \
        causal-conv1d==1.5.0.post8 mamba-ssm==2.2.4 \
    && python -m pip install --no-build-isolation flash-attn==2.7.2.post1 \
    && FLASH_ATTN_SOURCE="$(mktemp -d /tmp/flash-attention-source.XXXXXX)" \
    && git init "${FLASH_ATTN_SOURCE}" \
    && git -C "${FLASH_ATTN_SOURCE}" remote add origin https://github.com/Dao-AILab/flash-attention.git \
    && git -C "${FLASH_ATTN_SOURCE}" fetch --depth=1 origin "${DAO_FLASH_ATTN_COMMIT}" \
    && git -C "${FLASH_ATTN_SOURCE}" checkout FETCH_HEAD \
    && test "$(git -C "${FLASH_ATTN_SOURCE}" rev-parse HEAD)" = "${DAO_FLASH_ATTN_COMMIT}" \
    && MAX_JOBS=8 python -m pip install --no-build-isolation "${FLASH_ATTN_SOURCE}/csrc/rotary" \
    && MAX_JOBS=8 python -m pip install --no-build-isolation "${FLASH_ATTN_SOURCE}/csrc/layer_norm" \
    && MAX_JOBS=8 python -m pip install --no-build-isolation "${FLASH_ATTN_SOURCE}/csrc/xentropy" \
    && rm -rf "${FLASH_ATTN_SOURCE}" \
    && python -m pip install "git+https://github.com/fla-org/flash-linear-attention.git@${FLA_RELEASE_COMMIT}" \
    && python -c "import torch, xformers, fla, transformers, rotary_emb, dropout_layer_norm, xentropy_cuda_lib; assert torch.__version__.split('+')[0] == '2.3.1', torch.__version__; assert transformers.__version__ == '4.47.0', transformers.__version__" \
    && python3 -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml
COPY policy.yaml /tests/policy.yaml
COPY policy_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh
COPY --chmod=755 launch_gdn_4x8_lsf.sh /task-tools/launch_gdn_4x8_lsf.sh
RUN mkdir -p /app/project /app/output /logs/verifier
CMD ["bash"]

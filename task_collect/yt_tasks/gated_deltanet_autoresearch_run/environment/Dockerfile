FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ARG GDN_TASK_COMMIT=f13c92929221e011324f40aebdd2d436175a5949
ARG FLA_RELEASE_COMMIT=326184c45f42380063fb10bdda31d394b0c8c1b7
ARG DAO_FLASH_ATTN_COMMIT=f86e3dd9192e41dee3814c4cfd8bbce4792e6753
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ninja-build python3-packaging \
    && rm -rf /var/lib/apt/lists/*
COPY project /opt/project
RUN test "$(git -C /opt/project rev-parse HEAD)" = "${GDN_TASK_COMMIT}" \
    && git -C /opt/project diff --check \
    && test -z "$(git -C /opt/project status --short)" \
    && chmod -R a+rX /opt/project
COPY --chmod=755 launch_gdn_4x8_lsf.sh /task-tools/launch_gdn_4x8_lsf.sh

# This follows the public repository Dockerfile. Versions absent from that
# Dockerfile are pinned here only to make the task image repeatable.
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
    && python -m pip install "git+https://github.com/fla-org/flash-linear-attention.git@${FLA_RELEASE_COMMIT}"
RUN python -c "import torch, xformers, fla, transformers, rotary_emb, dropout_layer_norm, xentropy_cuda_lib; assert torch.__version__.split('+')[0] == '2.3.1', torch.__version__; assert transformers.__version__ == '4.47.0', transformers.__version__"

RUN mkdir -p /app/output /logs/artifacts \
    && cp -a /opt/project /app/project \
    && chmod -R a+rX /app/project
WORKDIR /app/project
CMD ["bash"]

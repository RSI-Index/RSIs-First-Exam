FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ARG DCLM_COMMIT=361714bdd60bb9b7f4b2d8354cebbf0dec0c329e
ARG OPEN_LM_COMMIT=87955f56980d0481379a05f768f48c5d525e5fd5
ARG RUST_TOOLCHAIN=1.94.1
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_MODE=offline \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    CARGO_HOME=/opt/rust/cargo \
    RUSTUP_HOME=/opt/rust/rustup \
    PATH=/opt/rust/cargo/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl git jq liblzma-dev libssl-dev \
    libxml2-dev libxslt1-dev pkg-config unzip zstd \
    && rm -rf /var/lib/apt/lists/*

# kenlm 0.2.0 still declares compatibility with pre-3.5 CMake policies.
# CMake 4 removes that compatibility; keep the dependency itself pinned and
# use the latest CMake 3 release solely as a deterministic build tool.
RUN python -m pip install --no-cache-dir cmake==3.29.6 \
    && cmake --version | grep -F 'cmake version 3.29.6'

COPY project /opt/project
COPY requirements-training.txt /opt/task-config/requirements-training.txt
RUN test "$(git -C /opt/project rev-parse HEAD)" = "${DCLM_COMMIT}" \
    && test "$(git -C /opt/project status --short --untracked-files=all)" = \
        "?? rust_processing/tokshuf-rs/Cargo.lock" \
    && test "$(sha256sum /opt/project/rust_processing/tokshuf-rs/Cargo.lock | awk '{print $1}')" = \
        35d73d17d1ea8df5fe387c756878d23eef30315ad794c8a3d148bb089b386764 \
    && git -C /opt/project diff --check

# Runtime mount points for large upstream assets. Payloads stay on shared
# storage and are bind-mounted read-only instead of copied into the image.
RUN mkdir -p \
        /opt/project/baselines/mappers/enrichers/language_id_enrichment_models \
        /opt/project/baselines/mappers/enrichers/quality_prediction_enrichment_models \
    && touch \
        /opt/project/baselines/mappers/banlists/refinedweb_banned_domains_curated.txt

RUN grep -v '^kenlm==' /opt/task-config/requirements-training.txt \
        > /tmp/requirements-without-kenlm.txt \
    && python -m pip install --no-cache-dir \
        -r /tmp/requirements-without-kenlm.txt \
    && python -m pip install --no-cache-dir --no-build-isolation kenlm==0.2.0 \
    && python -m pip install --no-cache-dir --no-deps xformers==0.0.25.post1 \
    && python -m pip install --no-cache-dir --no-deps \
        "git+https://github.com/mlfoundations/open_lm.git@${OPEN_LM_COMMIT}" \
    && python - <<'PY'
import importlib.metadata
import open_lm
import torch
import transformers
import xformers

assert torch.__version__.split('+')[0] == '2.2.0', torch.__version__
assert importlib.metadata.version('open-lm') == '0.0.33'
assert transformers.__version__ == '4.40.2'
assert xformers.__version__ == '0.0.25.post1'
print('DCLM_TRAIN_IMPORT_SMOKE_OK')
PY

RUN mkdir -p /opt/rust \
    && curl -fsSL --retry 5 \
        https://static.rust-lang.org/rustup/archive/1.27.1/x86_64-unknown-linux-gnu/rustup-init \
        -o /opt/rust/rustup-init \
    && test "$(sha256sum /opt/rust/rustup-init | awk '{print $1}')" = \
        6aeece6993e902708983b209d04c0d1dbb14ebb405ddb87def578d41f920f56d \
    && chmod +x /opt/rust/rustup-init \
    && /opt/rust/rustup-init -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && cargo build --release --locked \
        --manifest-path /opt/project/rust_processing/tokshuf-rs/Cargo.toml \
    && install -m 0755 \
        /opt/project/rust_processing/tokshuf-rs/target/release/tokshuf-rust \
        /usr/local/bin/tokshuf-rust \
    && cargo build --release --locked \
        --manifest-path /opt/project/dedup/bff/Cargo.toml \
    && install -m 0755 \
        /opt/project/dedup/bff/target/release/bff \
        /usr/local/bin/bff \
    && rustc --version | grep -F "rustc ${RUST_TOOLCHAIN}" \
    && tokshuf-rust --help >/dev/null \
    && bff --help >/dev/null \
    && rm -rf \
        /opt/project/rust_processing/tokshuf-rs/target \
        /opt/project/dedup/bff/target \
        /opt/rust/cargo/registry \
        /opt/rust/cargo/git

RUN mkdir -p /app/output /app/project /logs/artifacts /task-tools \
    && cp -a /opt/project/. /app/project/ \
    && chmod -R a+rX /opt/project /app/project
WORKDIR /app/project
CMD ["bash"]

FROM python:3.12.9-slim-bookworm

ARG SUPERBPE_COMMIT=bbd09768fc28a875cef48e6bdd66e3a17454628e
ARG TOKENIZERS_SUPERBPE_COMMIT=757f2a55c0820ed47064e1fe473deea39b7b611b
ARG RUSTUP_INIT_SHA256=6c30b75a75b28a96fd913a037c8581b580080b6ee9b8169a3c0feb1af7fe8caf
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=true \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git jq libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --silent --show-error https://sh.rustup.rs \
      --output /tmp/rustup-init.sh \
    && echo "${RUSTUP_INIT_SHA256}  /tmp/rustup-init.sh" | sha256sum --check - \
    && sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain 1.84.1 \
    && rm /tmp/rustup-init.sh \
    && rustc --version | grep -F 'rustc 1.84.1'

COPY project /opt/project
RUN test "$(git -C /opt/project rev-parse HEAD)" = "${SUPERBPE_COMMIT}" \
    && test "$(git -C /opt/project/tokenizers_superbpe rev-parse HEAD)" = "${TOKENIZERS_SUPERBPE_COMMIT}" \
    && test -z "$(git -C /opt/project status --porcelain)" \
    && chmod -R a+rX /opt/project

# The repository leaves package versions and the Rust `stable` channel
# unpinned. These pins are container reproducibility infrastructure. ai2-olmo
# is intentionally omitted because no tokenizer entrypoint imports it.
RUN python -m pip install --no-cache-dir \
      click==8.1.8 filelock==3.17.0 huggingface-hub==0.28.1 \
      pysimdjson==6.0.2 regex==2024.11.6 tqdm==4.67.1 \
      transformers==4.46.3 maturin==1.8.1 \
    && python -m pip install --no-cache-dir --no-build-isolation \
      /opt/project/tokenizers_superbpe/bindings/python \
    && python -c "import tokenizers; assert tokenizers.__version__ == '0.20.1', tokenizers.__version__" \
    && rm -rf /opt/project/tokenizers_superbpe/bindings/python/target

COPY --chmod=755 superbpe_task_tool.py /task-tools/superbpe_task_tool.py
RUN mkdir -p /app/output /logs/artifacts \
    && cp -a /opt/project /app/project \
    && chmod -R a+rX /app/project
WORKDIR /app/project
CMD ["bash"]

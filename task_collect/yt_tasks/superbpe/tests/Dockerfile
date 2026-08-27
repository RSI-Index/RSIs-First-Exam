FROM python:3.12.9-slim-bookworm

ARG SUPERBPE_COMMIT=bbd09768fc28a875cef48e6bdd66e3a17454628e
ARG SUPERBPE_ARCHIVE_SHA256=9c05c32d63b0e849c47bee94c7c10621d446ac1b31f764a34bbb04c292490f49
ARG TOKENIZERS_SUPERBPE_COMMIT=757f2a55c0820ed47064e1fe473deea39b7b611b
ARG TOKENIZERS_SUPERBPE_ARCHIVE_SHA256=6e96c27b1c4314b003a7a9b3635280fcb09489de6d80b7bfd2c1bfd8285d3c5d
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=true
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Encoding does not exercise the fork's trainer changes. The verifier uses
# the matching upstream tokenizers 0.20.1 wheel and a hash-pinned clean source
# archive, avoiding a second Rust build without changing token counts.
RUN archive="$(mktemp)" \
    && curl --fail --location --retry 5 \
      "https://codeload.github.com/PythonNut/superbpe/tar.gz/${SUPERBPE_COMMIT}" \
      --output "$archive" \
    && echo "${SUPERBPE_ARCHIVE_SHA256}  $archive" | sha256sum --check - \
    && mkdir -p /opt/project \
    && tar -xzf "$archive" --strip-components=1 -C /opt/project \
    && rm "$archive" \
    && submodule_archive="$(mktemp)" \
    && curl --fail --location --retry 5 \
      "https://codeload.github.com/alisawuffles/tokenizers-superbpe/tar.gz/${TOKENIZERS_SUPERBPE_COMMIT}" \
      --output "$submodule_archive" \
    && echo "${TOKENIZERS_SUPERBPE_ARCHIVE_SHA256}  $submodule_archive" | sha256sum --check - \
    && mkdir -p /opt/project/tokenizers_superbpe \
    && tar -xzf "$submodule_archive" --strip-components=1 -C /opt/project/tokenizers_superbpe \
    && rm "$submodule_archive" \
    && python -m pip install --no-cache-dir tokenizers==0.20.1 \
    && python -c "import tokenizers; assert tokenizers.__version__ == '0.20.1', tokenizers.__version__" \
    && python -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm==1.63.14 pyyaml==6.0.2

COPY policy.yaml /tests/policy.yaml
COPY policy_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh
RUN mkdir -p /app/project /app/output /logs/verifier
CMD ["bash"]

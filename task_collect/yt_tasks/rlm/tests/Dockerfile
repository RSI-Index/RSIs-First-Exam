FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/root/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git jq python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY oolong_local.patch /opt/oolong_local.patch
RUN git clone https://github.com/alexzhang13/rlm.git /opt/project \
    && git -C /opt/project checkout 72d6940142ddfb84ee6be573dc999a37e633e671 \
    && git -C /opt/project apply /opt/oolong_local.patch \
    && python3 -m pip install --break-system-packages --no-cache-dir \
       -e /opt/project -e /opt/project/training -e /opt/project/training/environments/oolong \
       "verifiers @ https://files.pythonhosted.org/packages/5a/f4/5dcb18b558647b88c73bca8a347a046cc7c9dfdef1d900c2cbad352d91f2/verifiers-0.1.15.dev9-py3-none-any.whl" \
       "vllm==0.22.0" \
    && python3 -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml
COPY policy.yaml /tests/policy.yaml
COPY policy_check.py policy_codex_judge.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh

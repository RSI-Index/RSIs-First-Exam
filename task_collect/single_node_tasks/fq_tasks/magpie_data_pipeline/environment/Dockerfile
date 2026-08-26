# syntax=docker/dockerfile:1.7
# Build with the task directory as context so shared and environment files stay explicit.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG AXOLOTL_COMMIT=7c2bf3091f5e73c787afe839dfdcc8220b770a1a
ARG VLLM_COMMIT=643c125fab66d5ed5ec3143b7e764a77e7ae8ac7
ARG TRANSFORMERS_COMMIT=943628458a1691f8af09c47ea9fc6e314734722f
ARG MAGPIE_COMMIT=b734a36818e4ba7ef7f0f582fc55f7e860407d26

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git python3 python3-pip python3-venv util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/train-venv \
    && python3 -m venv /opt/infer-venv \
    && /opt/train-venv/bin/pip install --no-cache-dir "git+https://github.com/axolotl-ai-cloud/axolotl.git@${AXOLOTL_COMMIT}" \
    && /opt/infer-venv/bin/pip install --no-cache-dir "git+https://github.com/vllm-project/vllm.git@${VLLM_COMMIT}" \
    && /opt/infer-venv/bin/pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_COMMIT}" huggingface_hub pyarrow

# The build credential is visible only to this layer and is removed before it ends.
RUN --mount=type=secret,id=hf_token,target=/run/secrets/hf_token \
    token="$(cat /run/secrets/hf_token)" \
    && export HF_TOKEN="$token" \
    && export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    && mkdir -p /models /task-assets/raw/baseline /task-assets/raw/arena-data /task-assets/raw/alpaca-data /task-assets/raw/wildbench-data \
    && /opt/infer-venv/bin/huggingface-cli download meta-llama/Meta-Llama-3-8B --revision 8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 --local-dir /models/Meta-Llama-3-8B \
    && /opt/infer-venv/bin/huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --revision 50fd307e57011801c7833c87efa1984ddf2db42f --local-dir /models/Meta-Llama-3-70B-Instruct \
    && /opt/infer-venv/bin/huggingface-cli download Magpie-Align/Llama-3-8B-Magpie-Align-SFT-v0.1 --revision 1ed587f54f70334f495efb9c027acb03e96fe24f --local-dir /models/Llama-3-8B-Magpie-Align-SFT-v0.1 \
    && /opt/infer-venv/bin/huggingface-cli download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir /models/Qwen3.5-9B \
    && /opt/infer-venv/bin/huggingface-cli download Magpie-Align/Magpie-Pro-MT-300K-v0.1 --repo-type dataset --revision 3c3045600d578b8f37413924b1b3bf117929755e --local-dir /task-assets/raw/baseline \
    && /opt/infer-venv/bin/huggingface-cli download lmarena-ai/arena-hard-auto --repo-type dataset --revision 15f3746e21432264ce9b453999bde4f3c946d2e6 --local-dir /task-assets/raw/arena-data \
    && /opt/infer-venv/bin/huggingface-cli download tatsu-lab/alpaca_eval --repo-type dataset --revision 2edc6fad8be6b14ea7230aabfd08188da6b8b814 --local-dir /task-assets/raw/alpaca-data \
    && /opt/infer-venv/bin/huggingface-cli download allenai/WildBench --repo-type dataset --revision 26c49eb39d7d5ce2099b0bbafed5a88dcce954ec --local-dir /task-assets/raw/wildbench-data \
    && unset HF_TOKEN token

RUN git clone https://github.com/magpie-align/magpie.git /task-assets/magpie \
    && git -C /task-assets/magpie checkout --detach "${MAGPIE_COMMIT}" \
    && test "$(git -C /task-assets/magpie rev-parse HEAD)" = "${MAGPIE_COMMIT}" \
    && git clone https://github.com/tatsu-lab/alpaca_eval.git /task-assets/raw/alpaca-eval \
    && git -C /task-assets/raw/alpaca-eval checkout --detach cd543a149df89434d8a54582c0151c0b945c3d20 \
    && git clone https://github.com/lmarena/arena-hard-auto.git /task-assets/raw/arena-code \
    && git -C /task-assets/raw/arena-code checkout --detach 196f6b826783b3da7310e361a805fa36f0be83f3 \
    && git clone https://github.com/allenai/WildBench.git /task-assets/raw/wildbench-code \
    && git -C /task-assets/raw/wildbench-code checkout --detach d6b8dcaf377d173d031980f97c16e1a82618c03d \
    && true

COPY shared/task_contract.py shared/dataset_contract.py shared/evaluation_contract.py /task-tools/
COPY shared/llama3_chat_template.jinja /task-tools/llama3_chat_template.jinja
COPY shared/llama3_chat_template.jinja /tests/llama3_chat_template.jinja
COPY shared/policy_contract.py /task-tools/policy_contract.py
COPY environment/prepare_assets.py environment/stage_snapshots.py environment/generate_dataset.py environment/magpie_async_run.py environment/async_contract.py environment/pipeline.py /task-tools/
COPY --chmod=755 environment/magpie_async_run.sh /task-tools/magpie_async_run
COPY environment/pipeline.py environment/pipeline_config.toml /opt/project/data_research/
COPY environment/frozen_train.yaml /tests/frozen_train.yaml
COPY tests/policy_check.py tests/dataset_check.py tests/train_candidate.py tests/generate_responses.py tests/judge.py tests/score.py tests/policy_contract.py tests/policy.yaml tests/task_contract.py tests/dataset_contract.py tests/evaluation_contract.py /tests/
COPY --chmod=755 tests/test.sh /tests/test.sh

# Snapshot conversion is deterministic and local; it never generates model responses.
RUN /opt/infer-venv/bin/python /task-tools/stage_snapshots.py --output /task-assets --baseline /task-assets/raw/baseline/data/train-00000-of-00001.parquet --alpaca-eval-2 /task-assets/raw/alpaca-data/alpaca_eval_gpt4_baseline.json --arena-hard /task-assets/raw/arena-data/data/arena-hard-v0.1/question.jsonl --wildbench /task-assets/raw/wildbench-data/v2/test-00000-of-00001.parquet \
    && /opt/infer-venv/bin/python /task-tools/prepare_assets.py --output /task-assets \
    && rm -rf /task-assets/raw /task-assets/snapshots /task-assets/magpie

RUN cp -a /opt/project /app/project \
    && mkdir -p /app/output \
    && useradd --create-home --uid 1000 agent \
    && chown -R root:root /models /task-assets /task-tools /opt/project /tests /app/project \
    && chmod 0711 /models /task-assets \
    && find /models/Meta-Llama-3-8B /models/Meta-Llama-3-70B-Instruct -type d -exec chmod 0555 {} + \
    && find /models/Meta-Llama-3-8B /models/Meta-Llama-3-70B-Instruct -type f -exec chmod 0444 {} + \
    && find /models/Qwen3.5-9B /models/Llama-3-8B-Magpie-Align-SFT-v0.1 -type d -exec chmod 0700 {} + \
    && find /models/Qwen3.5-9B /models/Llama-3-8B-Magpie-Align-SFT-v0.1 -type f -exec chmod 0600 {} + \
    && chmod 0444 /task-assets/Magpie-Pro-MT-300K-v0.1.jsonl \
    && chmod 0444 /task-assets/baseline_budget.json \
    && chmod 0600 /task-assets/eval_manifest.json \
    && chmod -R a=rX /task-tools \
    && chmod -R go-rwx /opt/project /tests \
    && chmod 0700 /opt/project /tests \
    && chmod -R a-w /app/project \
    && chown -R agent:agent /app/project/data_research /app/output \
    && chmod -R u+rwX /app/project/data_research \
    && chmod 0700 /app/output \
    && runuser -u agent -- sh -euxc 'test "$(id -u)" = 1000; test -r /models/Meta-Llama-3-8B/config.json; test -r /models/Meta-Llama-3-70B-Instruct/config.json; test -r /task-assets/Magpie-Pro-MT-300K-v0.1.jsonl; test -r /task-assets/baseline_budget.json; test -x /task-tools/magpie_async_run; test ! -r /task-assets/eval_manifest.json; test ! -r /tests/test.sh; test ! -r /models/Qwen3.5-9B/config.json; test ! -r /models/Llama-3-8B-Magpie-Align-SFT-v0.1/config.json; test ! -e /task-assets/raw; test ! -e /task-assets/snapshots; test ! -e /task-assets/magpie'

ENV PYTHONPATH=/task-tools:/app/project \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /app/project
USER agent
CMD ["bash"]

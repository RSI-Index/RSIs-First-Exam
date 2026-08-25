# Agent environment. Pins VLM-R1 and, unlike the reference package, ships the task tools.
#
# NO DATA AND NO MODEL WEIGHTS ARE BAKED IN OR DOWNLOADED HERE -- by design, and not as a
# placeholder. Every dataset and checkpoint this task needs is declared by URL in
# environment/assets.yaml, copied into the image at /assets/manifest.yaml, and expected to
# arrive later as read-only bind mounts (see the mountpoints created at the bottom). The image
# builds, the package lints, and tests/contract_check.py reports ASSETS NOT STAGED until
# environment/fetch_assets.sh is run deliberately.
#
# The one thing this build DOES download is CODE, pinned by commit sha. That is not in scope
# for "defer the data": the image is unusable without the trainer it wraps.
#
# NOTE on the reference package's build defect: its environment/Dockerfile did
#   COPY project /opt/project
#   RUN test "$(git -C /opt/project rev-parse HEAD)" = "${COMMIT}"
# but the bundled project/ had no .git, so that check could never pass as shipped. Here the
# repo is cloned inside the build (network at build time only) and verified by commit sha, so
# there is no vendored-tree-without-.git failure mode.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG VLMR1_COMMIT=90052f478646e38bc67363862f699b7afca6b337
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_MODE=offline \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ninja-build python3-packaging python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/om-ai-lab/VLM-R1.git /opt/project \
    && git -C /opt/project checkout "${VLMR1_COMMIT}" \
    && test "$(git -C /opt/project rev-parse HEAD)" = "${VLMR1_COMMIT}" \
    && chmod -R a+rX /opt/project

# Dependency set from the repository's own setup.sh. Versions absent there are pinned here
# only to make the image repeatable.
RUN python -m pip install -e "/opt/project/src/open-r1-multimodal[dev]" \
    && python -m pip install \
       wandb==0.18.3 tensorboardx qwen_vl_utils torchvision \
       babel python-Levenshtein matplotlib pycocotools httpx[socks] \
       deepspeed==0.15.4 accelerate==1.2.1 pyyaml \
    && python -m pip install flash-attn==2.7.2.post1 --no-build-isolation
# setup.sh installs `openai`; we deliberately do NOT. grpo_jsonl.py instantiates an OpenAI
# client at import with a dummy key and its llm-reward path silently degrades to exact string
# match on exception. Omitting the package turns that latent reward-integrity hole into an
# ImportError if anything ever selects it.

RUN python -c "import torch, transformers, deepspeed, qwen_vl_utils; \
assert torch.__version__.split('+')[0].startswith('2.6'), torch.__version__"

# The contract's claims ABOUT the upstream code are checked here, against the freshly cloned
# pinned tree, because this package holds no copy of the repository to check them against. A
# stale line citation or a renamed reward function fails the build instead of surfacing in
# round 3 as a confused agent.
COPY contract.yaml            /app/contract.yaml
COPY tests/upstream_check.py  /tmp/upstream_check.py
RUN python /tmp/upstream_check.py --project /opt/project --contract /app/contract.yaml \
    && rm /tmp/upstream_check.py

# BUILD CONTEXT IS THE PACKAGE ROOT:
#   docker build -f environment/Dockerfile -t vlmr1-rec-environment .
COPY --chmod=755 environment/task-tools/async_run.py   /task-tools/async_run.py
COPY --chmod=755 environment/task-tools/task_tool.py   /task-tools/task_tool.py
COPY --chmod=755 environment/task-tools/carry_tool.py  /task-tools/carry_tool.py
COPY --chmod=755 environment/task-tools/asset_check.py /task-tools/asset_check.py
# The agent re-reads these every round on purpose: the carry contract and the rules it will
# be judged against must be in front of it in round 6 exactly as in round 1.
COPY policy.yaml     /app/policy.yaml
COPY carry/README.md /app/carry-contract.md

# The asset manifest travels WITH the image. Two reasons it is here and not only in the
# staging repo: (a) /task-tools/asset_check.py reads its `state:` field, so a round launched
# against unstaged mounts fails loudly instead of training on nothing; (b) whoever inherits
# this .sif six months from now can recover every source URL from the artifact itself.
# The URLs are inert inside the container -- network_mode = "none" in both task.toml
# environments -- so shipping them, including the must_not_stage deny list, gives the agent no
# capability it did not already have from policy.yaml RH-004.
COPY environment/assets.yaml     /assets/manifest.yaml
COPY environment/fetch_assets.sh /assets/fetch_assets.sh
LABEL org.rsi.assets.state="NOT_STAGED" \
      org.rsi.assets.manifest="/assets/manifest.yaml" \
      org.rsi.assets.model="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct" \
      org.rsi.assets.train_annotations="https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip" \
      org.rsi.assets.train_images="https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/train2014.zip" \
      org.rsi.assets.eval_images="https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/lisa-test.zip" \
      org.rsi.assets.code="https://github.com/om-ai-lab/VLM-R1.git@90052f478646e38bc67363862f699b7afca6b337"

# B-1: /app/project is restored from the pristine copy at the start of EVERY round, so code
# edits do not survive a round boundary. An improver that wants to persist must live in /carry.
#
# The mountpoints below are created EMPTY. They are where the deferred assets land as
# read-only bind mounts; an empty /datasets/vlmr1/coco is the expected state of a freshly
# built image, not a broken one. /eval-data is deliberately NOT created here -- it belongs to
# the verifier image only, and a mountpoint that exists in the agent container is one
# `docker run -v` typo away from being real.
RUN mkdir -p /app/output /carry /logs/artifacts \
             /models /datasets/vlmr1/rec_jsons_train /datasets/vlmr1/coco \
    && cp -a /opt/project /app/project \
    && chmod -R a+rX /app/project
WORKDIR /app/project
CMD ["bash"]

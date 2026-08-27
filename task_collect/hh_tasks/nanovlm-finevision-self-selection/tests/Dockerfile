# Verifier image. Separate from the agent image (task.toml verifier.environment_mode = separate).
#
# The evaluation sets live under /eval-data, which does not exist in the agent image at all. That
# mount boundary IS the RH-RSI-003 containment story; a promise in prose would not be one.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG NANOVLM_COMMIT=4e0c0961846135c2217f95e54cb4c2d66eb55e42

# NO DEFAULT, ON PURPOSE. lmms-eval is an external repository whose internals nanoVLM's
# evaluation.py imports directly, and the README installs it from an untagged branch. A default
# here would resolve to whatever the branch tip is on build day, which means the same checkpoint
# scores differently in different rounds of the same chain. With no default, `docker build` fails
# until someone pins it -- and tests/contract_check.py asserts this ARG has no default.
ARG LMMS_EVAL_COMMIT

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 WANDB_MODE=offline \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/eval-data/lmms_eval_home

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git python3-venv \
    && rm -rf /var/lib/apt/lists/*

# The clean reference tree the policy gate diffs the candidate against, and the pristine base.
# NOTE the patch: /opt/project is upstream PLUS environment/rsi_adapter.patch. Both images apply
# it, and both verify it, because a verifier holding a different base than the agent would diff
# the agent's honest work as a modification.
COPY environment/rsi_adapter.patch /opt/rsi_adapter.patch
RUN git clone https://github.com/huggingface/nanoVLM.git /opt/project \
    && git -C /opt/project checkout "${NANOVLM_COMMIT}" \
    && test "$(git -C /opt/project rev-parse HEAD)" = "${NANOVLM_COMMIT}" \
    && git -C /opt/project apply --check /opt/rsi_adapter.patch \
    && git -C /opt/project apply /opt/rsi_adapter.patch \
    && chmod -R a+rX /opt/project

RUN test -n "${LMMS_EVAL_COMMIT}" \
    && python -m pip install \
         numpy torchvision pillow datasets huggingface-hub transformers pyyaml \
    && python -m pip install \
         "git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git@${LMMS_EVAL_COMMIT}" \
    && python -m pip install accelerate \
    && python3 -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml

# BUILD CONTEXT IS THE PACKAGE ROOT:
#   docker build -f tests/Dockerfile --build-arg LMMS_EVAL_COMMIT=<sha> -t nanovlm-rsi-tests .
# This is why there is exactly ONE policy.yaml on disk. The reference package used tests/ as the
# context, which forced a second copy of policy.yaml to live there -- and that copy drifted a full
# version behind the one the task described, while being the copy test.sh actually loaded.
#
# Paths are named explicitly and never `COPY . /tests`: reference/ holds the anchor improver and a
# whole-context copy would put it in the image with .dockerignore as the only thing between it and
# the agent. contract_check.py fails the build on a bare `COPY . <dest>` for exactly this reason.
COPY policy.yaml   /tests/policy.yaml
COPY contract.yaml /tests/contract.yaml
COPY tests/policy_check.py tests/carry_check.py tests/evaluate.py tests/score.py \
     tests/chain_score.py tests/contract_check.py tests/dir_hash.py \
     tests/upstream_check.py /tests/
COPY --chmod=755 tests/test.sh /tests/test.sh
COPY environment/assets.yaml /assets/manifest.yaml

# The seams contract.yaml names -- the per-turn rating filters, the hardcoded 56-shard branch, the
# two places the step budget is enforced -- are the mechanism this task rests on, and this package
# holds no copy of the repository to check them against. So the check runs HERE, against the
# freshly cloned and patched tree: if a cited symbol is gone, or the patch is not the declared
# shape, or the keep list would be applied before the val split, the verifier image does not build.
RUN python /tests/upstream_check.py --project /opt/project --contract /tests/contract.yaml

# Empty mountpoints for the deferred assets. Every one of these makes `test -d` succeed on a
# completely unstaged machine, which is why the gates count entries instead.
RUN mkdir -p /app/project /app/output /carry /logs/verifier \
             /datasets/finevision_pool /models /eval-data/lmms_eval_home
CMD ["bash"]

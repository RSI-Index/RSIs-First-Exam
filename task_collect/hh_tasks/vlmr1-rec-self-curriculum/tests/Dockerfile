# Verifier image. Separate from the agent image (task.toml verifier.environment_mode = separate).
#
# The eval data mounts at /eval-data, NOT under /datasets, because upstream ships the val/test
# jsons in the same directory as the training jsonl -- a one-path-away leakage trap.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG VLMR1_COMMIT=90052f478646e38bc67363862f699b7afca6b337
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 WANDB_MODE=offline \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ninja-build python3-packaging python3-venv \
    && rm -rf /var/lib/apt/lists/*

# The clean reference tree the policy gate diffs the candidate against.
RUN git clone https://github.com/om-ai-lab/VLM-R1.git /opt/project \
    && git -C /opt/project checkout "${VLMR1_COMMIT}" \
    && test "$(git -C /opt/project rev-parse HEAD)" = "${VLMR1_COMMIT}" \
    && chmod -R a+rX /opt/project

RUN python -m pip install \
      transformers==4.49.0 accelerate==1.2.1 qwen_vl_utils torchvision \
      pillow pyyaml pycocotools \
    && python -m pip install flash-attn==2.7.2.post1 --no-build-isolation \
    && python3 -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml

# BUILD CONTEXT IS THE PACKAGE ROOT:
#   docker build -f tests/Dockerfile -t vlmr1-rec-tests .
# This is why there is exactly ONE policy.yaml on disk. The reference package used tests/ as
# the context, which forced a second copy of policy.yaml to live there -- and that copy drifted
# a full version behind the one the task described, while being the copy test.sh actually loaded.
# Removing the duplicate removes the failure mode instead of policing it.
COPY policy.yaml   /tests/policy.yaml
COPY contract.yaml /tests/contract.yaml
COPY tests/policy_check.py tests/carry_check.py tests/evaluate.py tests/score.py \
     tests/chain_score.py tests/contract_check.py tests/dir_hash.py \
     tests/upstream_check.py /tests/
COPY --chmod=755 tests/test.sh /tests/test.sh
COPY environment/assets.yaml /assets/manifest.yaml

# The upstream seams contract.yaml names (grpo_jsonl.py's jsonl loader, the trainer's per-group
# reward std, RepeatRandomSampler) are the mechanism this task rests on, and this package holds
# no copy of the repository to check them against. So the check runs HERE, at build time,
# against the freshly cloned pinned tree: if a cited line range no longer contains what the
# contract says it contains, the verifier image does not build.
RUN python /tests/upstream_check.py --project /opt/project --contract /tests/contract.yaml

# Empty mountpoints for the deferred assets. /eval-data is verifier-only and does not exist in
# the agent image at all.
RUN mkdir -p /app/project /app/output /carry /logs/verifier \
             /eval-data/jsons /eval-data/lisa /eval-data/coco /models
CMD ["bash"]

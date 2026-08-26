# Verifier image. Separate from the agent image (task.toml verifier.environment_mode = separate).
#
# The 38 evaluation sets live under /eval-data, which does not exist in the agent image at all. That
# mount boundary IS the RH-RSI-003 containment story.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG DATACOMP_COMMIT=4a8df1992566ef8334773f7152e1855b1f716162

# NO DEFAULT, ON PURPOSE. train.py:13 does `from training.main import main` -- it imports open_clip's
# INTERNAL training entry point, so the recipe frozen in scale_configs.py is only meaningful against
# a pinned open_clip. A default here would resolve to whatever is current on build day and the same
# subset would score differently across rounds of one chain. tests/contract_check.py asserts this ARG
# has no default.
ARG OPEN_CLIP_VERSION

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git python3-venv \
    && rm -rf /var/lib/apt/lists/*

# The clean reference tree the policy gate diffs against, and the pristine base. UNPATCHED: unlike
# the nanoVLM sibling, everything this task needs already exists upstream.
RUN git clone https://github.com/mlfoundations/datacomp.git /opt/project \
    && git -C /opt/project checkout "${DATACOMP_COMMIT}" \
    && test "$(git -C /opt/project rev-parse HEAD)" = "${DATACOMP_COMMIT}" \
    && chmod -R a+rX /opt/project

RUN test -n "${OPEN_CLIP_VERSION}" \
    && python -m pip install \
         "open_clip_torch==${OPEN_CLIP_VERSION}" \
         webdataset pandas pyarrow pyyaml numpy scipy scikit-learn \
    && python -m pip install -r /opt/project/requirements.txt \
    && python3 -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml numpy

# BUILD CONTEXT IS THE PACKAGE ROOT:
#   docker build -f tests/Dockerfile --build-arg OPEN_CLIP_VERSION=<ver> -t datacomp-rsi-tests .
# This is why there is exactly ONE policy.yaml on disk. The reference package used tests/ as the
# context, which forced a second copy of policy.yaml to live there -- and that copy drifted a full
# version behind the one the task described, while being the copy test.sh actually loaded.
#
# Paths are named explicitly and never `COPY . /tests`: reference/ holds the anchor improver, and a
# whole-context copy would put it in the image with .dockerignore as the only thing between it and
# the agent. contract_check.py fails the build on a bare `COPY . <dest>`.
COPY policy.yaml   /tests/policy.yaml
COPY contract.yaml /tests/contract.yaml
COPY tests/policy_check.py tests/carry_check.py tests/evaluate.py tests/score.py \
     tests/chain_score.py tests/contract_check.py tests/dir_hash.py \
     tests/pool_check.py tests/upstream_check.py /tests/
COPY --chmod=755 tests/test.sh /tests/test.sh
COPY environment/assets.yaml /assets/manifest.yaml

# The seams contract.yaml names -- baselines.py's filters, the resharder, aggregate_scores' 38-dataset
# assert, and above all SCALE_CONFIGS["small"], which IS the per-round budget -- are checked HERE
# against the freshly cloned tree, because this package holds no copy of the repository. If the recipe
# dict has changed, if a cited symbol is gone, or if the 40-task list no longer yields exactly 38
# scored datasets, the verifier image does not build.
RUN python /tests/upstream_check.py --project /opt/project --contract /tests/contract.yaml

# Empty mountpoints for the deferred assets. Each one makes `test -d` succeed on a completely
# unstaged machine, which is why the gates count entries instead.
RUN mkdir -p /app/project /app/output /carry /logs/verifier \
             /datasets/commonpool/metadata /datasets/commonpool/shards \
             /eval-data/datacomp_evalsets
CMD ["bash"]

FROM python:3.11-slim-bookworm@sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TASK_VALIDATION_MODE=layer1-smoke \
    TASK_PROJECT=/app/project \
    TASK_OUTPUT_ROOT=/app/output \
    TASK_TOOLS_ROOT=/task-tools \
    TASK_SOURCE_TREE_SHA256=9238a96476fa1607fc572f8d0e62d52cdd441c851a9d51b31c5d76000a7e780c

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY source.lock.json materialize_project.sh /opt/search-rl-materializer/
RUN /opt/search-rl-materializer/materialize_project.sh --destination /app/project

COPY task-tools/ /task-tools/
RUN chmod -R a-w /task-tools \
    && test ! -e /app/project/evaluation \
    && test -f /app/project/training_scripts/rl/recipe/deepresearch/run_deepresearch_fully_async_megatron.sh \
    && mkdir -p /app/output /logs/verifier

WORKDIR /app/project

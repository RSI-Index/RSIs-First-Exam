FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG BIGVISION_COMMIT=8921d5141504390a8a4f7b2dacb3b3c042237290
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TFDS_DATA_DIR=/datasets/tfds \
    TF_CPP_MIN_LOG_LEVEL=1 \
    JAX_PLATFORMS=cuda,cpu \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates git python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/big_vision \
    && git -C /opt/big_vision remote add origin https://github.com/google-research/big_vision.git \
    && git -C /opt/big_vision fetch --depth=1 origin "${BIGVISION_COMMIT}" \
    && git -C /opt/big_vision checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/big_vision rev-parse HEAD)" = "${BIGVISION_COMMIT}" \
    && test "$(git -C /opt/big_vision rev-parse HEAD^{tree})" = "8f73654781bd99e0ee8ffc5f92d97ef566c4fd06" \
    && test "$(sha256sum /opt/big_vision/big_vision/configs/proj/image_text/lit_coco.py | awk '{print $1}')" = "96fa17d53a75dd4b08c45705cb617332ee19f092deabcc533ae437507b4f7688" \
    && test "$(sha256sum /opt/big_vision/big_vision/trainers/proj/image_text/contrastive.py | awk '{print $1}')" = "b51bae56b3acf1431fcdcf543779c2f7fc199b601dd64f16444cb8832fc2ced5" \
    && test "$(sha256sum /opt/big_vision/big_vision/evaluators/proj/image_text/retrieval.py | awk '{print $1}')" = "c1f7b93995025ba9613d8d52761915a7f747f3ea16d7a4b92fe0002e8b1de107" \
    && test "$(sha256sum /opt/big_vision/big_vision/configs/proj/image_text/common.py | awk '{print $1}')" = "8009e908e74bc227ac6a1cb281d1ab03bd5b854f33bbd54a3e70436ca664c549" \
    && test "$(sha256sum /opt/big_vision/big_vision/configs/common.py | awk '{print $1}')" = "a2b5a9089fa8e63698911f2db9950473dfd77383ba0b0db71eecd5f324c452be" \
    && test "$(sha256sum /opt/big_vision/big_vision/requirements.txt | awk '{print $1}')" = "2128acdd0cd2f415e2666c2645ea3246d32b896ac92a683e75dd5d7f4e433b57"

COPY requirements.txt /opt/contracts/requirements.txt
RUN python3 -m pip install --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0 \
    && python3 -m pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -r /opt/contracts/requirements.txt

COPY project /opt/project
RUN cp -a /opt/project /app/project
COPY task_tools /task-tools
COPY task_tools/model_adapter.py /opt/big_vision/big_vision/models/harbor_text_adapter.py
COPY baseline_contract.json /opt/contracts/baseline_contract.json
COPY asset_contract.json /opt/contracts/asset_contract.json
COPY --chmod=755 run_candidate.sh /task-tools/run_candidate.sh
ENV PYTHONPATH=/app/project:/task-tools:/opt/big_vision
RUN mkdir -p /app/output /logs/artifacts /datasets/tfds /datasets/initializers \
    && chmod -R a+rX /opt/big_vision /opt/project /task-tools /opt/contracts
WORKDIR /app/project
CMD ["bash"]

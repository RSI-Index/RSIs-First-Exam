FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY setup/requirements.txt /opt/setup/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/setup/requirements.txt
COPY setup/prepare_assets.py /opt/setup/prepare_assets.py

ENTRYPOINT ["python", "/opt/setup/prepare_assets.py"]

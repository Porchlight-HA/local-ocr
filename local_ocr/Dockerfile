# Build context is the App folder (local_ocr/). Run scripts/sync-app.sh first so ./server exists.
ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-debian:trixie
FROM ${BUILD_FROM}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

COPY server /tmp/server
RUN /opt/venv/bin/pip install --no-cache-dir /tmp/server \
 && rm -rf /tmp/server \
 && /opt/venv/bin/python -c "from rapidocr import RapidOCR; RapidOCR()"

COPY rootfs /

EXPOSE 8738

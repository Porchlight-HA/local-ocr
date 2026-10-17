# Build context is the App folder (local_ocr/). Run scripts/sync-app.sh first so ./server exists.
ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-debian:trixie
# hadolint ignore=DL3006
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

ENV PATH=/opt/venv/bin:$PATH

EXPOSE 8738

# Supervisor passes BUILD_ARCH and BUILD_VERSION to App builds; plain docker builds get the defaults.
ARG BUILD_ARCH=unknown
ARG BUILD_VERSION=dev
LABEL org.opencontainers.image.title="Local OCR" \
      org.opencontainers.image.description="Offline OCR server for Home Assistant (RapidOCR on onnxruntime)." \
      org.opencontainers.image.source="https://github.com/Porchlight-HA/local-ocr" \
      org.opencontainers.image.url="https://github.com/Porchlight-HA/local-ocr" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="Porchlight-HA" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      io.hass.name="Local OCR" \
      io.hass.type="addon" \
      io.hass.arch="${BUILD_ARCH}" \
      io.hass.version="${BUILD_VERSION}"

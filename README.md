# Local OCR

An offline OCR server packaged as a Home Assistant App (formerly "add-on"). It runs RapidOCR on onnxruntime with PP-OCR models baked in: no cloud, no runtime downloads, nothing stored. It is for `aarch64` and `amd64`.

Usage, options, privacy and troubleshooting are in [`local_ocr/DOCS.md`](local_ocr/DOCS.md).

## Install as a Local App (0.1)

0.1 publishes no image, so Supervisor builds it on the device. Copy the App folder to the `addons` share (Samba or SSH) after syncing the server into it:

```bash
scripts/sync-app.sh              # copies server/ into local_ocr/server/ (gitignored)
# then copy local_ocr/ to /addons/local_ocr on the Home Assistant host
```

In Home Assistant: Settings, Apps, App store, menu, Check for updates, then install **Local OCR** under Local apps.

## Build and test

Supervisor builds with the App folder as the Docker context, so the App folder has to be self-contained. `scripts/sync-app.sh` copies `server/` (without tests, caches and `.venv`) into `local_ocr/server/`; the Dockerfile copies it from there. Re-run it after every server change.

```bash
scripts/sync-app.sh
docker build --build-arg BUILD_FROM=ghcr.io/home-assistant/aarch64-base-debian:trixie -t local-ocr:dev local_ocr/
tests/smoke.sh local-ocr:dev     # prints OK
```

Use `amd64-base-debian` on x86 hosts. To run it without Home Assistant, see "Using without Home Assistant OS" in `local_ocr/DOCS.md` (set `LOCAL_OCR_TOKEN`, or publish the port on 127.0.0.1 only). Server unit and golden tests: `server/scripts/test.sh` (Docker).

## Layout

- `server/`: the Python package `local_ocr_server`
- `local_ocr/`: the App folder (`config.yaml`, `build.yaml`, `Dockerfile`, s6 service under `rootfs/`)
- `fixtures/`: synthetic test envelopes and generator
- `.github/workflows/`: CI (`ci.yaml`) and native multi-arch build (`build.yaml`), unused until the repository is published

## Licence

MIT. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

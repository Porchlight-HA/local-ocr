# Local OCR

Local OCR reads text from images, entirely on this device. It is a small HTTP server (RapidOCR on onnxruntime, with PP-OCR models baked into the App) that the **Local OCR** integration talks to. No cloud service is involved and nothing is downloaded at run time.

## Privacy

Images and recognised text are processed in memory and are never stored or logged. The only file the App writes is a shared access token in `/data/token`, which lets the integration authenticate to it.

## Setup

1. Install and start the App.
2. Home Assistant announces it through discovery. Add the **Local OCR** integration when prompted (no fields to fill in).

The OCR port (8738) is only reachable from inside Home Assistant unless you map it in the App's network settings.

## Options

| Option | Default | Meaning |
|---|---|---|
| `idle_unload_minutes` | 10 | Free the model's memory after this many idle minutes (1 to 1440). The next request reloads it, which takes a few seconds. |
| `max_side` | 2000 | Longest image side, in pixels, used for reading (640 to 4000). Bigger images are shrunk first. Lower it on low-memory systems. |
| `log_level` | info | Reserved. The server currently always logs at info level. |

## What it can read, and what it can't read well

It reads printed text well: envelopes, labels, signs, screenshots, and neat handwriting usually works.

What it can't read well:

- Thin felt-tip handwriting at Informed Delivery resolution (~780 px wide) is unreliable.
- Seven-segment LCD digits. Use [ssocr](https://www.unix-ag.uni-kl.de/~auerswal/ssocr/) or [AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device) instead.
- Very low-resolution or heavily blurred images.

Images and recognised text are never stored or logged, so there is nothing to look back at when a read fails. Re-run it on the same image to investigate.

## Using without Home Assistant OS

The same image runs anywhere Docker does. Build it from this repository (`scripts/sync-app.sh`, then `docker build --build-arg BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:trixie -t local-ocr local_ocr/`) and run it standalone:

```bash
docker run -d --name local-ocr --restart unless-stopped -p 8738:8738 \
  --entrypoint /opt/venv/bin/python local-ocr \
  -m local_ocr_server --standalone --port 8738
```

- Optional: add `-e LOCAL_OCR_TOKEN=<secret>` to require `Authorization: Bearer <secret>` on the API. Without it, the server logs a warning that the API is open.
- With `--network host` the server advertises itself over zeroconf so the integration can discover it. Otherwise add the integration and enter `http://<this-host>:8738` once.
- Published images (`ghcr.io/porchlight-ha/local-ocr`) come in a later release.

## Troubleshooting

Each entry matches a repair shown by the integration.

- **`app_not_running`**: the App is stopped. Start it from the App page (Home Assistant OS) or run `docker start local-ocr` (Container installs). Check the App log for the reason it stopped.
- **`app_outdated`**: the integration needs a newer App API. Update or rebuild the Local OCR App to the version the repair names.
- **`model_failed`**: the OCR model failed to load. Restart the App. If it persists, reinstall it; models are bundled, so no download is needed.
- **`low_memory`**: the system ran out of memory while reading. Close other apps, or lower `max_side` in the options.
- **`unsupported_arch`**: only 64-bit systems (aarch64 and amd64) are supported.

# Local OCR

Local OCR reads text from images, entirely on this device. It is a small HTTP server (RapidOCR on onnxruntime, with PP-OCR models baked into the App) that the **Text Reader** integration talks to. No cloud service is involved and nothing is downloaded at run time.

## Privacy

Images and recognised text are processed in memory and are never stored or logged. The only file the App writes is a shared access token in `/data/token`, which lets the integration authenticate to it.

## Setup

1. Install and start the App.
2. Home Assistant announces it through discovery. Add the **Text Reader** integration when prompted (no fields to fill in).

The OCR port (8738) is only reachable from inside Home Assistant unless you map it in the App's network settings.

If the server crashes, it restarts on its own. After three crashes within ten minutes the App stops. Turn on **Watchdog** on the App's page to have Home Assistant restart it then too (the watchdog checks `/v1/health`).

## Options

| Option | Default | Meaning |
|---|---|---|
| `idle_unload_minutes` | 10 | Free the model's memory after this many idle minutes (1 to 1440). The next request reloads it, which takes a few seconds. |
| `max_side` | 2000 | Longest image side, in pixels, used for reading (640 to 4000). Bigger images are shrunk first. Lower it on low-memory systems. |
| `log_level` | info | `debug`, `info`, `warning` or `error`. Images and recognised text are never logged at any level. |

## What it can read, and what it can't read well

It reads printed text well: envelopes, labels, signs, screenshots, and neat handwriting usually works.

What it can't read well:

- Thin felt-tip handwriting at Informed Delivery resolution (~780 px wide) is unreliable.
- Seven-segment LCD digits. Use [ssocr](https://www.unix-ag.uni-kl.de/~auerswal/ssocr/) or [AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device) instead.
- Very low-resolution or heavily blurred images.

Images and recognised text are never stored or logged, so there is nothing to look back at when a read fails. Re-run it on the same image to investigate.

## Using without Home Assistant OS

The same image runs anywhere Docker does, and starts in standalone mode when it is not under Home Assistant Supervisor. Build it from this repository (`scripts/sync-app.sh`, then `docker build --build-arg BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:trixie -t local-ocr local_ocr/`) and run it. Choose a long random string as the token; the integration needs it too.

```bash
docker run -d --name local-ocr --restart unless-stopped -p 8738:8738 \
  -e LOCAL_OCR_TOKEN=<choose-a-long-random-string> local-ocr
```

Without a token the API is unauthenticated, so anything that can reach the port can use it. If you run without one, publish the port on localhost only:

```bash
docker run -d --name local-ocr --restart unless-stopped -p 127.0.0.1:8738:8738 local-ocr
```

- Environment variables: `LOCAL_OCR_TOKEN` (requires `Authorization: Bearer <token>`), `PORT` (8738), `IDLE_UNLOAD_MINUTES` (10), `MAX_SIDE` (2000), `LOG_LEVEL` (info).
- With `--network host` the server advertises itself over zeroconf so the integration can discover it. Otherwise add the integration and enter `http://<this-host>:8738` once.
- Published images come in a later release, one per architecture: `ghcr.io/porchlight-ha/amd64-local-ocr` and `ghcr.io/porchlight-ha/aarch64-local-ocr` (the naming Home Assistant uses for App images). Until then, build the image as above.

## API (for use without the integration)

All paths are under `/v1`. With a token set, every path except `/v1/health` needs `Authorization: Bearer <token>`.

- `GET /v1/health`: `{"status": "ok", "model_state": "unloaded" | "loading" | "ready" | "error", "detail": <text or null>, "memory": {"oom_kills": <int>, "last_oom": <ISO 8601 time or null>}}`. It never loads the model. `memory.oom_kills` is the container's out-of-memory kill count (cgroup v2; 0 when unavailable), `memory.last_oom` the last time a read ran out of memory.
- `GET /v1/info`: version, `api` (1), architecture, engine, models, presets, languages and limits.
- `POST /v1/read`: multipart with an `image` part, or JSON with `image_b64`. Optional fields: `preset`, `lang`, `min_confidence`, `region` (`x,y,w,h` in pixels), `frame`, `all_frames`.
  - The response has `width`, `height` (the original image size, after EXIF rotation), `ms`, `engine`, `model`, `text`, `blocks` (`text`, `confidence`, `box`, `line`), `preset` and `fields`.
  - `text` has one line per visual line. Blocks on the same line are joined with a space; lines are separated by `\n`.
  - With `all_frames=true` the response is `{"frames": [<one response as above per frame>], "text": "<the frames' text joined with \n\n>"}` and has no other top-level fields.
  - Limits: 10 MB and 40 megapixels (`413 too_large`); undecodable images give `415 unsupported_image`; a full queue gives `503 busy` with `Retry-After`. A read that runs out of memory gives `500 model_error` with a message starting `out of memory:`.

## Troubleshooting

Each entry matches a repair shown by the integration.

- **`app_not_running`**: the App is stopped. Start it from the App page (Home Assistant OS) or run `docker start local-ocr` (Container installs). Check the App log for the reason it stopped.
- **`app_outdated`**: the integration needs a newer App API. Update or rebuild the Local OCR App to the version the repair names.
- **`model_failed`**: the OCR model failed to load. Restart the App. If it persists, reinstall it; models are bundled, so no download is needed.
- **`low_memory`**: in the last hour the App ran out of memory: loading the model or reading an image failed, or the system killed the server (it restarts on its own). Close other apps, or lower `max_side` in the options.
- **`unsupported_arch`**: only 64-bit systems (aarch64 and amd64) are supported.

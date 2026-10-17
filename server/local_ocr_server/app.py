"""HTTP API (api 1): /v1/health, /v1/info, /v1/read.

Privacy: logs carry method, path, status, timing, image dimensions and block count only.
Never recognized text, never image bytes.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import dataclasses
import hmac
import io
import json
import logging
import os
import platform
import time
import warnings

import numpy as np
from aiohttp import BodyPartReader, web
from PIL import Image, ImageOps, UnidentifiedImageError

from . import presets
from .engine import ENGINE_NAME, Block, ModelError, OcrEngine

_LOGGER = logging.getLogger("local_ocr_server.access")

API_VERSION = 1
MAX_BYTES = 10 * 1024 * 1024
MAX_SIDE = 4000
MAX_WAITING = 4
LANGS = ["en"]
_FIELD_LIMIT = 4096
_CHUNK = 64 * 1024
_B64_LIMIT = MAX_BYTES * 4 // 3 + _CHUNK + 4 * _CHUNK  # base64 of the max image plus JSON overhead


class _Queue:
    inflight = 0  # requests past parsing: 1 running + waiting


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers


def _err(status: int, code: str, message: str, headers: dict | None = None) -> web.Response:
    return web.json_response({"error": code, "message": message}, status=status, headers=headers)


# -- middlewares ---------------------------------------------------------------
@web.middleware
async def _logging(request: web.Request, handler):
    start = time.monotonic()
    request["log"] = {}
    status = 500
    try:
        resp = await handler(request)
        status = resp.status
        return resp
    except web.HTTPException as exc:
        status = exc.status
        raise
    finally:
        extra = request["log"]
        _LOGGER.info(
            "%s %s %d %dms dims=%s blocks=%s",
            request.method, request.path, status, (time.monotonic() - start) * 1000,
            extra.get("dims", "-"), extra.get("blocks", "-"),
        )


@web.middleware
async def _errors(request: web.Request, handler):
    try:
        return await handler(request)
    except ApiError as exc:
        return _err(exc.status, exc.code, exc.message, exc.headers)
    except web.HTTPException as exc:
        code = {404: "not_found", 405: "method_not_allowed", 413: "too_large"}.get(exc.status, "bad_request")
        return _err(exc.status, code, exc.reason or code)
    except Exception as exc:  # noqa: BLE001 - never leak details (could contain user content)
        # Privacy: the message/traceback may embed recognised text, so log the type only.
        _LOGGER.error("unhandled %s in %s %s", type(exc).__name__, request.method, request.path)
        return _err(500, "internal_error", "internal error")


@web.middleware
async def _auth(request: web.Request, handler):
    token: str | None = request.app["token"]
    if token and request.path.startswith("/v1/") and request.path != "/v1/health":
        header = request.headers.get("Authorization", "")
        supplied = header[7:] if header[:7].lower() == "bearer " else ""
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise ApiError(401, "unauthorized", "missing or invalid bearer token", {"WWW-Authenticate": "Bearer"})
    return await handler(request)


# -- request parsing -----------------------------------------------------------
async def _read_stream(read_chunk, limit: int, too_big: ApiError) -> bytes:
    buf = bytearray()
    while True:
        chunk = await read_chunk()
        if not chunk:
            return bytes(buf)
        buf += chunk
        if len(buf) > limit:
            raise too_big


_TOO_LARGE = ApiError(413, "too_large", f"image larger than {MAX_BYTES} bytes")


async def _parse_request(request: web.Request) -> tuple[bytes, dict]:
    ctype = request.content_type
    if request.content_length is not None and request.content_length > _B64_LIMIT + MAX_BYTES:
        raise _TOO_LARGE
    if ctype == "multipart/form-data":
        data: bytes | None = None
        fields: dict = {}
        try:
            reader = await request.multipart()
            async for part in reader:
                if not isinstance(part, BodyPartReader):
                    raise ValueError("nested multipart")
                if part.name == "image":
                    data = await _read_stream(lambda p=part: p.read_chunk(_CHUNK), MAX_BYTES, _TOO_LARGE)
                else:
                    raw = await _read_stream(
                        lambda p=part: p.read_chunk(_CHUNK), _FIELD_LIMIT,
                        ApiError(400, "bad_request", f"field {part.name!r} too long"),
                    )
                    fields[part.name] = raw.decode("utf-8", "replace").strip()
        except (ValueError, AssertionError) as exc:
            raise ApiError(400, "bad_request", "malformed multipart body") from exc
        if data is None:
            raise ApiError(400, "bad_request", "missing 'image' part")
        return data, fields
    if ctype == "application/json":
        raw = await _read_stream(
            lambda: request.content.read(_CHUNK), _B64_LIMIT, _TOO_LARGE
        )
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise ApiError(400, "bad_request", "invalid JSON body") from exc
        if not isinstance(body, dict) or not isinstance(body.get("image_b64"), str):
            raise ApiError(400, "bad_request", "JSON body needs a string 'image_b64'")
        b64 = body["image_b64"]
        if b64.startswith("data:") and "," in b64[:100]:
            b64 = b64.split(",", 1)[1]
        try:
            data = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise ApiError(400, "bad_request", "image_b64 is not valid base64") from exc
        if len(data) > MAX_BYTES:
            raise _TOO_LARGE
        return data, {k: v for k, v in body.items() if k != "image_b64"}
    raise ApiError(400, "bad_request", "send multipart/form-data with an 'image' part or JSON with 'image_b64'")


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@dataclasses.dataclass
class _Params:
    preset: str
    min_confidence: float
    region: tuple[int, int, int, int] | None
    frame: int
    all_frames: bool


def _parse_params(fields: dict) -> _Params:
    preset = str(fields.get("preset") or "text")
    if preset not in presets.PRESETS:
        raise ApiError(400, "bad_request", f"unknown preset {preset!r}; expected one of {list(presets.PRESETS)}")
    lang = str(fields.get("lang") or "en")
    if lang not in LANGS:
        raise ApiError(400, "unsupported_lang", f"language {lang!r} is not supported; available: {LANGS}")
    try:
        min_conf = float(fields.get("min_confidence", 0.5))
    except (TypeError, ValueError):
        raise ApiError(400, "bad_request", "min_confidence must be a number") from None
    if not 0.0 <= min_conf <= 1.0:
        raise ApiError(400, "bad_request", "min_confidence must be between 0 and 1")
    region = None
    if fields.get("region") not in (None, ""):
        raw = fields["region"]
        try:
            parts = [int(float(p)) for p in (raw if isinstance(raw, list) else str(raw).split(","))]
            if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0 or parts[0] < 0 or parts[1] < 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ApiError(400, "bad_request", "region must be 'x,y,w,h' in pixels (w,h > 0)") from None
        region = (parts[0], parts[1], parts[2], parts[3])
    try:
        frame = int(fields.get("frame", 0))
        if frame < 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ApiError(400, "bad_request", "frame must be a non-negative integer") from None
    return _Params(preset, min_conf, region, frame, _as_bool(fields.get("all_frames", False)))


# -- decoding -------------------------------------------------------------------
def _to_array(im: Image.Image) -> np.ndarray:
    """Pillow frame -> BGR uint8 (the cv2 channel order the engine and fixtures use), alpha flattened on white."""
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, "white")
        bg.paste(rgba, mask=rgba.getchannel("A"))
        im = bg
    else:
        im = im.convert("RGB")
    return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])


MAX_FRAMES = 50


def _open(data: bytes, frame: int, all_frames: bool) -> tuple[Image.Image, list[int]]:
    """Open the image and return it with the frame indices to read (nothing is decoded yet)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data))
        n = getattr(im, "n_frames", 1)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ApiError(413, "too_large", "image has too many pixels") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
        raise ApiError(415, "unsupported_image", "could not decode the image") from exc
    if all_frames:
        if n > MAX_FRAMES:
            raise ApiError(413, "too_large", "too many frames")
        return im, list(range(n))
    if frame >= n:
        raise ApiError(400, "bad_request", f"frame {frame} out of range (image has {n})")
    return im, [frame]


def _frame_array(im: Image.Image, index: int) -> np.ndarray:
    try:
        im.seek(index)
        return _to_array(im)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ApiError(413, "too_large", "image has too many pixels") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
        raise ApiError(415, "unsupported_image", "could not decode the image") from exc


# -- handlers --------------------------------------------------------------------
def _model_name(engine: OcrEngine) -> str:
    for m in engine.model_info():
        if "rec" in m["name"]:
            return os.path.splitext(m["name"])[0]
    return ""


async def _health(request: web.Request) -> web.Response:
    engine = request.app["engine"]
    return web.json_response({"status": "ok", "model_state": engine.state, "detail": engine.last_error})


async def _info(request: web.Request) -> web.Response:
    engine = request.app["engine"]
    models = await asyncio.to_thread(engine.model_info)
    return web.json_response({
        "version": request.app["version"],
        "api": API_VERSION,
        "arch": platform.machine(),
        "engine": ENGINE_NAME,
        "models": models,
        "presets": list(presets.PRESETS),
        "langs": LANGS,
        "limits": {"max_bytes": MAX_BYTES, "max_side": MAX_SIDE},
    })


async def _recognise(engine: OcrEngine, img: np.ndarray, p: _Params, model: str, started: float) -> dict:
    height, width = img.shape[:2]
    ox = oy = 0
    crop = img
    if p.region:
        x, y, w, h = p.region
        if x >= width or y >= height:
            raise ApiError(400, "bad_request", f"region starts outside the {width}x{height} image")
        x2, y2 = min(width, x + w), min(height, y + h)
        crop = np.ascontiguousarray(img[y:y2, x:x2])
        ox, oy = x, y
    try:
        blocks = await engine.read(crop)
    except ModelError as exc:
        raise ApiError(500, "model_error", str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("inference failed: %s", type(exc).__name__)
        raise ApiError(500, "internal_error", "inference failed") from exc
    out: list[Block] = []
    for b in blocks:
        if b.confidence < p.min_confidence:
            continue
        box = [[px + ox, py + oy] for px, py in b.box]
        out.append(dataclasses.replace(b, box=box))
    fields = presets.apply(p.preset, out, width, height)
    return {
        "width": width,
        "height": height,
        "ms": int((time.monotonic() - started) * 1000),
        "engine": ENGINE_NAME,
        "model": model,
        "text": "\n".join(b.text for b in out),
        "blocks": [
            {"text": b.text, "confidence": round(b.confidence, 4), "box": b.box, "line": b.line + 1} for b in out
        ],
        "preset": p.preset,
        "fields": fields,
    }


async def _read(request: web.Request) -> web.Response:
    app = request.app
    data, fields = await _parse_request(request)
    p = _parse_params(fields)
    if app["queue"].inflight > MAX_WAITING:  # 1 running + MAX_WAITING waiting
        raise ApiError(503, "busy", "server busy, retry shortly", {"Retry-After": "5"})
    app["queue"].inflight += 1
    try:
        async with app["sem"]:
            engine: OcrEngine = app["engine"]
            started = time.monotonic()
            im, indices = await asyncio.to_thread(_open, data, p.frame, p.all_frames)
            model = await asyncio.to_thread(_model_name, engine)
            results = []
            for index in indices:  # decode and recognise one frame at a time
                img = await asyncio.to_thread(_frame_array, im, index)
                results.append(await _recognise(engine, img, p, model, started))
                started = time.monotonic()
    finally:
        app["queue"].inflight -= 1
    request["log"]["dims"] = f"{results[0]['width']}x{results[0]['height']}"
    request["log"]["blocks"] = sum(len(r["blocks"]) for r in results)
    if p.all_frames:
        return web.json_response({"frames": results, "text": "\n\n".join(r["text"] for r in results)})
    return web.json_response(results[0])


def create_app(engine: OcrEngine, token: str | None, version: str) -> web.Application:
    """Build the application. Add startup work (e.g. discovery) via `app.on_startup.append(...)`."""
    app = web.Application(
        middlewares=[_logging, _errors, _auth],
        client_max_size=MAX_BYTES + 64 * 1024,
    )
    app["engine"] = engine
    app["token"] = token or None
    app["version"] = version
    app["sem"] = asyncio.Semaphore(1)
    app["queue"] = _Queue()
    app.router.add_get("/v1/health", _health)
    app.router.add_get("/v1/info", _info)
    app.router.add_post("/v1/read", _read)

    async def _close_engine(_app: web.Application) -> None:
        await engine.close()

    app.on_cleanup.append(_close_engine)
    return app

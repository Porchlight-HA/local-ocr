import asyncio
import base64
import io
import logging
import platform

import aiohttp
import cv2
import pytest
from PIL import Image

from local_ocr_server import __version__
from local_ocr_server.app import create_app
from local_ocr_server.engine import Block, ModelError, OcrEngine


class FakeEngine:
    def __init__(self, delay: float = 0.0, error: Exception | None = None):
        self.state = "unloaded"
        self.last_error = None
        self.delay = delay
        self.error = error
        self.calls = []

    def model_info(self):
        return [{"name": "det.onnx", "sha256": "a"}, {"name": "PP-OCRv6_rec_small.onnx", "sha256": "b"}]

    async def read(self, img):
        self.calls.append(img.shape)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        self.state = "ready"
        return [
            Block("Hello", 0.95, [[10, 10], [60, 10], [60, 30], [10, 30]], 0),
            Block("World", 0.90, [[10, 40], [60, 40], [60, 60], [10, 60]], 1),
            Block("noise", 0.20, [[10, 70], [60, 70], [60, 90], [10, 90]], 2),
        ]

    async def close(self):
        pass


def png_bytes(w=200, h=100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def small_png():
    return png_bytes()


@pytest.fixture
def png_6000(big_img_6000):
    ok, buf = cv2.imencode(".png", big_img_6000)
    return buf.tobytes()


@pytest.fixture
def env02_id(fixtures_dir):
    return (fixtures_dir / "envelopes" / "env02_id.jpg").read_bytes()


@pytest.fixture
def gif_3_frames(fixtures_dir):
    frames = []
    for n in (0, 1, 2):
        im = Image.open(fixtures_dir / "envelopes" / f"env0{n}_id.jpg").convert("RGB").resize((400, 200))
        frames.append(im.convert("P", palette=Image.ADAPTIVE))
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return buf.getvalue()


@pytest.fixture
async def client(aiohttp_client):
    return await aiohttp_client(create_app(FakeEngine(), None, __version__))


@pytest.fixture
async def client_with_token(aiohttp_client):
    return await aiohttp_client(create_app(FakeEngine(), "s3cret", __version__))


@pytest.fixture
async def client_slow_engine(aiohttp_client):
    return await aiohttp_client(create_app(FakeEngine(delay=0.4), None, __version__))


@pytest.fixture
async def real_client(aiohttp_client):
    engine = OcrEngine(threads=2)
    return await aiohttp_client(create_app(engine, None, __version__))


async def post_raw(client, data: bytes, headers=None, **fields):
    form = aiohttp.FormData()
    form.add_field("image", data, filename="x.bin", content_type="application/octet-stream")
    for k, v in fields.items():
        form.add_field(k, str(v))
    return await client.post("/v1/read", data=form, headers=headers)


post_bytes = post_raw


async def post_img(client, data: bytes, **fields) -> dict:
    r = await post_raw(client, data, **fields)
    return await r.json()


async def test_health_no_model_touch(client):
    r = await client.get("/v1/health")
    assert r.status == 200
    assert await r.json() == {"status": "ok", "model_state": "unloaded", "detail": None}
    assert client.app["engine"].calls == []


async def test_info_shape(client):
    j = await (await client.get("/v1/info")).json()
    assert j["api"] == 1 and j["presets"] == ["text", "envelope", "meter", "label"]
    assert j["limits"] == {"max_bytes": 10485760, "max_side": 4000}
    assert j["version"] == __version__ and j["arch"] == platform.machine()
    assert j["langs"] == ["en"] and j["engine"].startswith("rapidocr ")
    assert j["models"][0]["name"] == "det.onnx"


async def test_read_response_shape_and_filtering(client, small_png):
    j = await post_img(client, small_png)
    assert list(j) == ["width", "height", "ms", "engine", "model", "text", "blocks", "preset", "fields"]
    assert (j["width"], j["height"]) == (200, 100)
    assert j["model"] == "PP-OCRv6_rec_small"
    assert j["text"] == "Hello\nWorld"  # 0.20 block dropped at default min_confidence 0.5
    assert [b["text"] for b in j["blocks"]] == ["Hello", "World"]
    assert list(j["blocks"][0]) == ["text", "confidence", "box", "line"]
    assert j["preset"] == "text" and j["fields"] is None


async def test_min_confidence(client, small_png):
    j = await post_img(client, small_png, min_confidence="0.1")
    assert len(j["blocks"]) == 3
    assert (await post_raw(client, small_png, min_confidence="abc")).status == 400


async def test_read_json_b64(client, small_png):
    r = await client.post("/v1/read", json={"image_b64": base64.b64encode(small_png).decode(), "preset": "text"})
    assert r.status == 200
    assert (await r.json())["width"] == 200


async def test_bad_params(client, small_png):
    for kw in ({"preset": "nope"}, {"region": "1,2,3"}, {"frame": "x"}, {"frame": "-1"}):
        r = await post_raw(client, small_png, **kw)
        assert r.status == 400 and (await r.json())["error"] == "bad_request", kw
    r = await post_raw(client, small_png, lang="de")
    assert r.status == 400 and (await r.json())["error"] == "unsupported_lang"
    r = await client.post("/v1/read", json={"nothing": 1})
    assert r.status == 400


async def test_token_required(client_with_token, small_png):
    c = client_with_token
    r = await c.post("/v1/read")
    assert r.status == 401 and (await r.json())["error"] == "unauthorized"
    assert (await c.get("/v1/info")).status == 401
    assert (await c.get("/v1/info", headers={"Authorization": "Bearer wrong"})).status == 401
    assert (await c.get("/v1/health")).status == 200
    assert (await c.get("/v1/info", headers={"Authorization": "Bearer s3cret"})).status == 200
    assert (await post_raw(c, small_png, headers={"Authorization": "Bearer s3cret"})).status == 200


async def test_zero_bytes(client):
    r = await post_bytes(client, b"")
    assert r.status == 415 and (await r.json())["error"] == "unsupported_image"


async def test_not_an_image(client):
    assert (await post_bytes(client, b"hello")).status == 415


async def test_too_large(client):
    r = await post_bytes(client, b"\xff" * (10 * 1024 * 1024 + 1))
    assert r.status == 413 and (await r.json())["error"] == "too_large"


async def test_too_large_b64(client):
    big = base64.b64encode(b"\xff" * (10 * 1024 * 1024 + 1)).decode()
    assert (await client.post("/v1/read", json={"image_b64": big})).status == 413


async def test_decompression_bomb(client, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    assert (await post_bytes(client, png_bytes(200, 100))).status == 413  # >2x limit -> error
    assert (await post_bytes(client, png_bytes(40, 30))).status == 413  # warning range -> promoted to error


async def test_huge_dimensions_downscaled(client, png_6000):
    j = await post_img(client, png_6000)
    assert j.get("width") == 6000 and j["height"] == 2600
    assert client.app["engine"].calls[-1][:2] == (2600, 6000)  # the engine does the downscale


async def test_animated_gif_all_frames(real_client, gif_3_frames):
    j = await post_img(real_client, gif_3_frames, all_frames="true")
    assert len(j["frames"]) == 3
    assert j["text"] == "\n\n".join(f["text"] for f in j["frames"])


async def test_gif_frame_index(client, gif_3_frames):
    j = await post_img(client, gif_3_frames, frame="2")
    assert j.get("error") is None and j["width"] == 400


async def test_gif_frame_out_of_range(client, gif_3_frames):
    r = await post_raw(client, gif_3_frames, frame="9")
    assert r.status == 400 and (await r.json())["error"] == "bad_request"


async def test_queue_overflow_503(client_slow_engine, small_png):
    rs = await asyncio.gather(*[post_raw(client_slow_engine, small_png) for _ in range(7)])
    statuses = [r.status for r in rs]
    assert statuses.count(503) == 2 and all(s in (200, 503) for s in statuses)
    busy = next(r for r in rs if r.status == 503)
    assert busy.headers["Retry-After"] == "5" and (await busy.json())["error"] == "busy"
    # queue drains afterwards
    assert (await post_raw(client_slow_engine, small_png)).status == 200


async def test_region_crop(aiohttp_client, small_png):
    eng = FakeEngine()
    c = await aiohttp_client(create_app(eng, None, __version__))
    j = await post_img(c, small_png, region="50,20,100,60")
    assert eng.calls == [(60, 100, 3)]
    assert (j["width"], j["height"]) == (200, 100)
    assert j["blocks"][0]["box"][0] == [60, 30]  # [10,10] + (50,20)
    assert (await post_raw(c, small_png, region="500,20,100,60")).status == 400


async def test_region_crop_real(real_client, env02_id):
    full = await post_img(real_client, env02_id)
    x, y = full["width"] // 2, 0
    j = await post_img(real_client, env02_id, region=f"{x},{y},{full['width'] - x},{full['height']}")
    assert (j["width"], j["height"]) == (full["width"], full["height"])
    assert all(min(px for px, _ in b["box"]) >= x - 2 for b in j["blocks"])


async def test_model_error_500(aiohttp_client, small_png):
    c = await aiohttp_client(create_app(FakeEngine(error=ModelError("out of memory: x")), None, __version__))
    r = await post_raw(c, small_png)
    j = await r.json()
    assert r.status == 500 and j == {"error": "model_error", "message": "out of memory: x"}


async def test_unknown_path_json_404(client):
    r = await client.get("/v1/nope")
    assert r.status == 404 and "error" in await r.json()


async def test_read_multipart_envelope(real_client, env02_id):
    j = await post_img(real_client, env02_id, preset="envelope")
    assert j["fields"]["generic_addressee"] is True and j["blocks"] and j["ms"] > 0


async def test_logs_contain_no_text(caplog, real_client, env02_id):
    caplog.set_level(logging.DEBUG)
    j = await post_img(real_client, env02_id)
    assert j["blocks"]
    for b in j["blocks"]:
        assert b["text"] not in caplog.text
    assert "POST /v1/read 200" in caplog.text


async def test_lines_are_one_based(client, small_png):
    j = await post_img(client, small_png, min_confidence="0.1")
    assert [b["line"] for b in j["blocks"]] == [1, 2, 3]  # fake engine emits 0,1,2


async def test_too_many_frames(client):
    frames = [Image.new("P", (8, 8), 0) for i in range(51)]
    for i, f in enumerate(frames):
        f.putpixel((i % 8, i // 8), 5)
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:])
    r = await post_raw(client, buf.getvalue(), all_frames="true")
    j = await r.json()
    assert r.status == 413 and j == {"error": "too_large", "message": "too many frames"}
    assert client.app["engine"].calls == []
    ok = io.BytesIO()
    frames[0].save(ok, "GIF", save_all=True, append_images=frames[1:50])
    assert (await post_raw(client, ok.getvalue(), all_frames="true")).status == 200


async def test_non_finite_params_400(client, small_png):
    for kw in ({"region": "inf,0,10,10"}, {"region": "nan,0,10,10"}, {"frame": "inf"}):
        r = await post_raw(client, small_png, **kw)
        assert r.status == 400, kw
    for body in ('{"image_b64": "%s", "region": [1e999, 0, 10, 10]}', '{"image_b64": "%s", "frame": 1e999}'):
        r = await client.post(
            "/v1/read", data=body % base64.b64encode(small_png).decode(), headers={"Content-Type": "application/json"}
        )
        assert r.status == 400 and (await r.json())["error"] == "bad_request"


async def test_internal_error_log_has_no_text(caplog, aiohttp_client, small_png):
    caplog.set_level(logging.DEBUG)
    secret = "SECRETWORD123"
    c = await aiohttp_client(create_app(FakeEngine(error=ValueError(f"int('{secret}')")), None, __version__))
    r = await post_raw(c, small_png)
    assert r.status == 500 and secret not in await r.text()
    assert secret not in caplog.text
    assert not any(rec.exc_info for rec in caplog.records)

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from local_ocr_server import engine as engine_mod
from local_ocr_server.engine import ENGINE_NAME, ModelError, OcrEngine


class FakeRapid:
    def __call__(self, img):
        return SimpleNamespace(
            boxes=np.array([[[10, 10], [110, 10], [110, 40], [10, 40]]], dtype=float),
            txts=("HELLO",),
            scores=(0.9,),
        )


async def test_lazy_load_then_ready(sample_img):
    e = OcrEngine(threads=2)
    assert e.state == "unloaded"
    blocks = await e.read(sample_img)
    assert e.state == "ready" and blocks
    assert all(0.0 <= b.confidence <= 1.0 and len(b.box) == 4 for b in blocks)
    await e.close()


async def test_idle_unload(sample_img):
    e = OcrEngine(threads=2, idle_unload_s=0.2)
    await e.read(sample_img)
    await asyncio.sleep(0.5)
    assert e.state == "unloaded"
    await e.close()


async def test_request_during_unload_waits_and_succeeds(sample_img):
    e = OcrEngine(threads=2, idle_unload_s=0.05)
    await e.read(sample_img)
    results = await asyncio.gather(*[e.read(sample_img) for _ in range(3)], asyncio.sleep(0.06))
    assert all(r for r in results[:3])
    await e.close()


async def test_threads_pinned(monkeypatch, sample_img):
    seen = {}
    monkeypatch.setattr(engine_mod, "RapidOCR", lambda params: seen.update(params) or FakeRapid())
    e = OcrEngine(threads=2)
    await e.read(sample_img)
    assert seen["EngineConfig.onnxruntime.intra_op_num_threads"] == 2
    assert seen["EngineConfig.onnxruntime.inter_op_num_threads"] == 1
    assert seen["Global.log_level"] == "warning"
    await e.close()


async def test_memory_error_sets_error_state(monkeypatch, sample_img):
    monkeypatch.setattr(engine_mod, "RapidOCR", lambda params: (_ for _ in ()).throw(MemoryError()))
    e = OcrEngine(threads=2)
    with pytest.raises(ModelError):
        await e.read(sample_img)
    assert e.state == "error" and e.last_error.startswith("out of memory")


async def test_alloc_error_is_reported_as_out_of_memory(monkeypatch, sample_img):
    def boom(params):
        raise RuntimeError("Failed to allocate memory for requested buffer")

    monkeypatch.setattr(engine_mod, "RapidOCR", boom)
    e = OcrEngine(threads=2)
    with pytest.raises(ModelError):
        await e.read(sample_img)
    assert e.last_error.startswith("out of memory: ")


async def test_other_error_records_message_and_next_read_retries(monkeypatch, sample_img):
    calls = []

    def flaky(params):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("bad model file")
        return FakeRapid()

    monkeypatch.setattr(engine_mod, "RapidOCR", flaky)
    e = OcrEngine(threads=2)
    with pytest.raises(ModelError):
        await e.read(sample_img)
    assert e.state == "error" and e.last_error == "bad model file"
    blocks = await e.read(sample_img)
    assert e.state == "ready" and blocks and len(calls) == 2
    await e.close()


async def test_blank_image_returns_no_blocks(monkeypatch):
    class Empty:
        def __call__(self, img):
            return SimpleNamespace(boxes=None, txts=None, scores=None)

    monkeypatch.setattr(engine_mod, "RapidOCR", lambda params: Empty())
    e = OcrEngine(threads=1)
    assert await e.read(np.zeros((50, 50, 3), np.uint8)) == []
    await e.close()


async def test_downscale_to_max_side(monkeypatch, big_img_6000):
    seen = {}

    class Spy(FakeRapid):
        def __call__(self, img):
            seen["shape"] = img.shape
            return super().__call__(img)

    monkeypatch.setattr(engine_mod, "RapidOCR", lambda params: Spy())
    e = OcrEngine(threads=2, max_side=2000)
    blocks = await e.read(big_img_6000)
    assert max(seen["shape"][:2]) == 2000
    # boxes map back to the original 6000 px frame (scale factor 3)
    assert blocks[0].box[2] == [330, 120]
    await e.close()


async def test_downscale_real_image_boxes_in_frame(big_img_6000):
    e = OcrEngine(threads=2, max_side=2000)
    blocks = await e.read(big_img_6000)
    h, w = big_img_6000.shape[:2]
    assert blocks
    assert all(0 <= x <= w + 5 and 0 <= y <= h + 5 for b in blocks for x, y in b.box)
    await e.close()


async def test_lines_group_by_row(monkeypatch):
    def box(x, y):
        return [[x, y], [x + 80, y], [x + 80, y + 30], [x, y + 30]]

    class Rows:
        def __call__(self, img):
            return SimpleNamespace(
                boxes=np.array([box(300, 100), box(10, 102), box(10, 200)], dtype=float),
                txts=("B", "A", "C"),
                scores=(0.9, 0.9, 0.9),
            )

    monkeypatch.setattr(engine_mod, "RapidOCR", lambda params: Rows())
    e = OcrEngine(threads=1)
    blocks = await e.read(np.zeros((400, 600, 3), np.uint8))
    assert [(b.text, b.line) for b in blocks] == [("A", 0), ("B", 0), ("C", 1)]
    await e.close()


async def test_model_info_and_engine_name(sample_img):
    assert ENGINE_NAME.startswith("rapidocr 3.10")
    e = OcrEngine(threads=2)
    info = e.model_info()
    assert info and all(len(m["sha256"]) == 64 and m["name"].endswith(".onnx") for m in info)

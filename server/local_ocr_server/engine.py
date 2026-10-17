"""RapidOCR wrapper: lazy model load, idle unload, pinned threads, downscaling."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from importlib.metadata import version
from typing import Literal

import cv2
import numpy as np
import rapidocr
from rapidocr import RapidOCR

ENGINE_NAME = f"rapidocr {version('rapidocr')}"

_LOGGER = logging.getLogger(__name__)
State = Literal["unloaded", "loading", "ready", "error"]


class ModelError(Exception):
    """Loading the OCR models failed."""


@dataclass
class Block:
    text: str
    confidence: float
    box: list[list[int]]
    line: int


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError) or "alloc" in str(exc).lower()


class OcrEngine:
    def __init__(self, threads: int, idle_unload_s: float = 600, max_side: int = 2000) -> None:
        self._threads = threads
        self._idle_unload_s = idle_unload_s
        self._max_side = max_side
        self._ocr: RapidOCR | None = None
        self._lock = asyncio.Lock()
        self._timer: asyncio.TimerHandle | None = None
        self._tasks: set[asyncio.Task] = set()
        self._gen = 0
        self.state: State = "unloaded"
        self.last_error: str | None = None
        self._model_info: list[dict] | None = None

    # -- models -------------------------------------------------------------
    def model_info(self) -> list[dict]:
        """Name and sha256 of the ONNX models shipped inside the rapidocr package."""
        if self._model_info is None:
            models_dir = os.path.join(os.path.dirname(rapidocr.__file__), "models")
            info = []
            for name in sorted(os.listdir(models_dir)):
                if name.endswith(".onnx"):
                    h = hashlib.sha256()
                    with open(os.path.join(models_dir, name), "rb") as fh:
                        for chunk in iter(lambda: fh.read(1 << 20), b""):
                            h.update(chunk)
                    info.append({"name": name, "sha256": h.hexdigest()})
            self._model_info = info
        return self._model_info

    def _build(self) -> RapidOCR:
        return RapidOCR(
            params={
                "EngineConfig.onnxruntime.intra_op_num_threads": self._threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                "Global.log_level": "warning",
            }
        )

    async def _ensure_loaded(self) -> None:
        if self._ocr is not None:
            return
        self.state = "loading"
        try:
            self._ocr = await asyncio.to_thread(self._build)
        except asyncio.CancelledError:
            # The build thread may finish later; its result is discarded and the next read reloads.
            self.state = "unloaded"
            raise
        except Exception as exc:  # noqa: BLE001 - classify every load failure
            self.state = "error"
            if _is_oom(exc):
                self.last_error = f"out of memory: {exc}"
            else:
                self.last_error = str(exc)
            _LOGGER.error("model load failed: %s", self.last_error)
            raise ModelError(self.last_error) from exc
        self.state = "ready"
        self.last_error = None

    # -- idle unload ----------------------------------------------------------
    def _arm_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = asyncio.get_running_loop().call_later(self._idle_unload_s, self._fire_unload, self._gen)

    def _fire_unload(self, gen: int) -> None:
        task = asyncio.get_running_loop().create_task(self._unload_if_idle(gen))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _unload_if_idle(self, gen: int) -> None:
        async with self._lock:
            # Any read since the timer was armed bumps the generation; then this unload is stale.
            if gen != self._gen:
                return
            self._drop()

    def _drop(self) -> None:
        self._ocr = None
        if self.state == "ready":
            self.state = "unloaded"

    # -- inference ------------------------------------------------------------
    async def read(self, img: np.ndarray) -> list[Block]:
        async with self._lock:
            self._gen += 1
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            try:
                await self._ensure_loaded()
                work = asyncio.ensure_future(asyncio.to_thread(self._infer, img))
                try:
                    return await asyncio.shield(work)
                except asyncio.CancelledError:
                    # Keep the lock until the worker thread is done so inferences never overlap.
                    await asyncio.wait([work])
                    raise
            finally:
                if self._ocr is not None:
                    self._arm_timer()

    def _infer(self, img: np.ndarray) -> list[Block]:
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > self._max_side:
            scale = self._max_side / max(h, w)
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        out = self._ocr(img)
        if out is None or out.boxes is None or out.txts is None:
            return []
        items = []
        for box, text, conf in zip(out.boxes, out.txts, out.scores):
            pts = [[int(round(x / scale)), int(round(y / scale))] for x, y in np.asarray(box).reshape(-1, 2)]
            items.append((pts, str(text), min(1.0, max(0.0, float(conf)))))
        return _assign_lines(items)

    async def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        async with self._lock:
            self._drop()


def _assign_lines(items: list[tuple[list[list[int]], str, float]]) -> list[Block]:
    """Order boxes top-to-bottom, left-to-right and number the visual rows."""

    def cy(p):
        return sum(y for _, y in p[0]) / len(p[0])

    def height(p):
        ys = [y for _, y in p[0]]
        return max(ys) - min(ys)

    rows: list[list] = []
    for item in sorted(items, key=cy):
        if rows and abs(cy(item) - cy(rows[-1][0])) <= 0.5 * max(height(item), height(rows[-1][0]), 1):
            rows[-1].append(item)
        else:
            rows.append([item])
    blocks = []
    for n, row in enumerate(rows):
        for pts, text, conf in sorted(row, key=lambda p: min(x for x, _ in p[0])):
            blocks.append(Block(text=text, confidence=conf, box=pts, line=n))
    return blocks

"""Field-recall scorer, ported from the local-ocr-bench kit (score.py)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

import cv2
from rapidfuzz import fuzz

_NON = re.compile(r"[^A-Z0-9% ]")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", _NON.sub(" ", s.upper())).strip()


def score(engine, fixtures_dir: Path, threshold: int = 90) -> dict[str, float]:
    env = Path(fixtures_dir) / "envelopes"
    gt = {g["id"]: g for g in json.loads((env / "ground_truth.json").read_text())}
    hits: dict[str, list[bool]] = {}

    async def run():
        for path in sorted(env.glob("env*")):
            img = cv2.imread(str(path))
            g = gt[path.name.split("_")[0]]
            blocks = await engine.read(img)
            text = norm(" | ".join(b.text for b in blocks))
            for field in ("addressee", "return", "markers"):
                for line in g[field]:
                    ok = fuzz.partial_ratio(norm(line), text) >= threshold
                    hits.setdefault(field, []).append(ok)
                    kind = "handwritten" if g["kind"] == "handwritten" else "printed"
                    hits.setdefault(kind, []).append(ok)

    asyncio.run(run())
    return {k: sum(v) / len(v) for k, v in hits.items()}


def sha256_manifest(directory: Path) -> list[str]:
    """Lines in `shasum -a 256 envelopes/*` format, sorted by path."""
    d = Path(directory)
    return sorted(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {d.name}/{p.name}" for p in d.iterdir() if p.is_file()
    )

import asyncio

import pytest

from local_ocr_server.engine import OcrEngine

from .scoring import score, sha256_manifest


@pytest.fixture(scope="module")
def scores(fixtures_dir):
    engine = OcrEngine(threads=2)
    try:
        return score(engine, fixtures_dir)
    finally:
        asyncio.run(engine.close())


def test_field_recall_thresholds(scores):
    print("\ngolden scores:", {k: round(v, 3) for k, v in sorted(scores.items())})
    assert scores["printed"] >= 0.97
    # Gates from measured baseline: handwritten 0.778, return 0.931; all misses are env09 (Kristi) at ID resolution.
    assert scores["handwritten"] >= 0.75
    assert scores["return"] >= 0.90
    assert scores["markers"] >= 0.95


def test_fixture_manifest_unchanged(fixtures_dir):
    expected = sorted(l for l in (fixtures_dir / "MANIFEST.sha256").read_text().splitlines() if l)
    assert sha256_manifest(fixtures_dir / "envelopes") == expected

from pathlib import Path

import cv2
import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def sample_img():
    img = cv2.imread(str(FIXTURES / "envelopes" / "env00_id.jpg"))
    assert img is not None
    return img


@pytest.fixture
def big_img_6000():
    img = cv2.imread(str(FIXTURES / "envelopes" / "env00_clean.png"))
    return cv2.resize(img, (6000, 2600), interpolation=cv2.INTER_CUBIC)

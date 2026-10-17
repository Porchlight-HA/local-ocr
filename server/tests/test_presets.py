import asyncio
import json

import cv2
import pytest
from rapidfuzz import fuzz

from local_ocr_server.engine import Block, OcrEngine
from local_ocr_server.presets import PRESETS, apply

from .scoring import norm


def B(text, x=0, y=0, w=100, h=20, conf=0.95, line=0):
    return Block(text, conf, [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], line)


# Bulk layout like env02 (780x340): indicia top right, ECRWSS top left, generic addressee centre.
BULK_BLOCKS = [
    B("ECRWSS", 30, 20, 90, 18, line=0),
    B("PRSRT STD", 560, 22, 130, 20, line=1),
    B("PERMIT NO 1234", 540, 48, 190, 20, line=2),
    B("FINAL NOTICE: ACT NOW", 40, 100, 300, 24, line=3),
    B("POSTAL CUSTOMER", 200, 170, 230, 24, line=4),
    B("89 PLACEHOLDER LN", 200, 200, 250, 24, line=5),
    B("DEMOCITY MN 55999", 200, 230, 250, 24, line=6),
]

# Business layout like env00 (780x340): return address top left, addressee centre.
BUSINESS_BLOCKS = [
    B("ACME UTILITY CO", 30, 20, 190, 20, line=0),
    B("PO BOX 98765", 30, 44, 160, 20, line=1),
    B("SAMPLETON OH 45999-8765", 30, 68, 280, 20, line=2),
    B("TAYLOR DEMOUSER", 250, 170, 230, 24, line=3),
    B("123 EXAMPLE AVE", 250, 200, 220, 24, line=4),
    B("SAMPLETON OH 45999-1234", 250, 230, 290, 24, line=5),
]


def addressee_matches(fields, gt) -> bool:
    got = [norm(x) for x in (fields["addressee"] or {"lines": []})["lines"]]
    pos = 0
    for want in gt["addressee"]:
        w = norm(want)
        for i in range(pos, len(got)):
            if fuzz.partial_ratio(w, got[i]) >= 90:
                pos = i + 1
                break
        else:
            return False
    return True


def test_presets_constant():
    assert PRESETS == ("text", "envelope", "meter", "label")


def test_envelope_bulk_assigns_fields():
    f = apply("envelope", BULK_BLOCKS, 780, 340)
    assert f["addressee"]["lines"] == ["POSTAL CUSTOMER", "89 PLACEHOLDER LN", "DEMOCITY MN 55999"]
    assert set(f["markers"]) >= {"PRSRT STD", "PERMIT NO", "ECRWSS"}
    assert f["generic_addressee"] is True
    assert f["indicia"][0] == "PRSRT STD"


def test_envelope_business_return_address():
    f = apply("envelope", BUSINESS_BLOCKS, 780, 340)
    assert f["return_address"]["lines"] == ["ACME UTILITY CO", "PO BOX 98765", "SAMPLETON OH 45999-8765"]
    assert f["addressee"]["lines"][0] == "TAYLOR DEMOUSER"
    assert f["generic_addressee"] is False and f["markers"] == []


def test_envelope_no_text_returns_nones():
    assert apply("envelope", [], 780, 340) == {
        "addressee": None, "return_address": None, "indicia": [], "markers": [], "generic_addressee": False,
    }


def test_or_current_resident_is_kept_and_generic():
    blocks = [B("OR CURRENT RESIDENT", 200, 170, 250), B("123 EXAMPLE AVE", 200, 200, 230), B("FAKETOWN WA 98999", 200, 230, 250)]
    f = apply("envelope", blocks, 780, 340)
    assert f["addressee"]["lines"][0] == "OR CURRENT RESIDENT"
    assert f["generic_addressee"] is True and "OR CURRENT RESIDENT" in f["markers"]


def test_label_ups():
    assert apply("label", [B("1Z999AA10123456784")], 800, 600)["tracking"] == [{"carrier": "ups", "number": "1Z999AA10123456784"}]


def test_label_usps_spaces_stripped():
    t = apply("label", [B("9400 1118 9922 3456 7890 12")], 800, 600)["tracking"]
    assert t == [{"carrier": "usps", "number": "9400111899223456789012"}]


def test_label_usps_with_words_before_digits():
    t = apply("label", [B("USPS TRACKING 9400 1118 9922 3333 4444 55")], 800, 600)["tracking"]
    assert t == [{"carrier": "usps", "number": "9400111899223333444455"}]


def test_label_fedex_digit_groups():
    t = apply("label", [B("FEDEX 1234 5678 9012")], 800, 600)["tracking"]
    assert t == [{"carrier": "fedex", "number": "123456789012"}]


def test_label_ship_to_stacked():
    blocks = [B("SHIP TO", 40, 100, 80), B("AVERY TESTPERSON", 40, 126, 200), B("123 EXAMPLE AVE", 40, 152, 200),
              B("FAR AWAY NOTE", 40, 400, 200)]
    f = apply("label", blocks, 800, 600)
    assert f["ship_to"]["lines"] == ["AVERY TESTPERSON", "123 EXAMPLE AVE"]


def test_label_ship_to_inline():
    blocks = [B("SHIP TO: AVERY TESTPERSON", 40, 100, 300), B("123 EXAMPLE AVE", 40, 126, 200)]
    f = apply("label", blocks, 800, 600)
    assert f["ship_to"]["lines"] == ["AVERY TESTPERSON", "123 EXAMPLE AVE"]


def test_or_current_resident_with_ocr_noise_kept():
    for txt in ("OR CURRENT RESIDENT.", "OR CURRENT RESIDENT 1"):
        blocks = [B(txt, 200, 170, 250), B("123 EXAMPLE AVE", 200, 200, 230), B("FAKETOWN WA 98999", 200, 230, 250)]
        f = apply("envelope", blocks, 780, 340)
        assert f["addressee"]["lines"][0] == txt
        assert f["generic_addressee"] is True


def test_label_fedex_requires_brand():
    assert apply("label", [B("123456789012")], 800, 600)["tracking"] == []
    t = apply("label", [B("FEDEX"), B("123456789012")], 800, 600)["tracking"]
    assert t == [{"carrier": "fedex", "number": "123456789012"}]


def test_label_dhl_requires_brand():
    assert apply("label", [B("1234567890")], 800, 600)["tracking"] == []
    assert apply("label", [B("DHL"), B("1234567890")], 800, 600)["tracking"] == [{"carrier": "dhl", "number": "1234567890"}]


def test_label_no_ship_to():
    assert apply("label", [B("x")], 800, 600)["ship_to"] is None


def test_meter_value():
    assert apply("meter", [B("0 1 2 3 4,5")], 400, 100)["value"] == 1234.5


def test_meter_picks_most_digits_and_empty():
    f = apply("meter", [B("kWh"), B("00123"), B("7")], 400, 100)
    assert f["value"] == 123.0 and f["digits"] == "00123"
    assert apply("meter", [B("kWh")], 400, 100) == {"value": None, "digits": "", "confidence": 0.0}


def test_text_preset_returns_none():
    assert apply("text", [B("x")], 10, 10) is None


def test_unknown_preset():
    with pytest.raises(ValueError):
        apply("nope", [], 1, 1)


def test_golden_envelope_addressee_on_id_variant(fixtures_dir):
    env = fixtures_dir / "envelopes"
    gts = json.loads((env / "ground_truth.json").read_text())
    engine = OcrEngine(threads=2)
    results = {}

    async def run():
        for gt in gts:
            img = cv2.imread(str(env / f"{gt['id']}_id.jpg"))
            h, w = img.shape[:2]
            fields = apply("envelope", await engine.read(img), w, h)
            results[gt["id"]] = (addressee_matches(fields, gt), fields["addressee"] and fields["addressee"]["lines"])
        await engine.close()

    asyncio.run(run())
    for k, v in sorted(results.items()):
        print(k, v)
    assert sum(ok for ok, _ in results.values()) >= 11   # of 12

"""Preset post-processing: turn raw OCR blocks into structured `fields`."""
from __future__ import annotations

import re
import statistics

from .engine import Block

PRESETS = ("text", "envelope", "meter", "label")

# (canonical name, pattern); a line that is only a marker (plus digits/punctuation) is not part of the address.
_MARKER_DEFS = [
    ("PRSRT STD", r"PRSRT\s*STD"),
    ("PRSRT MKTG", r"PRSRT\s*MKTG"),
    ("PRSRT FIRST", r"PRSRT\s*FIRST"),
    ("ECRWSS", r"ECRWSS"),
    ("ECRLOT", r"ECRLOT"),
    ("AUTO 5-DIGIT", r"AUTO\W*5-?DIGIT"),
    ("AUTO SCH 5-DIGIT", r"AUTO\W*SCH\W*5-?DIGIT"),
    ("AUTO 3-DIGIT", r"AUTO\W*3-?DIGIT"),
    ("PERMIT NO", r"PERMIT\s*NO"),
    ("NONPROFIT ORG", r"NONPROFIT\s*ORG"),
    ("OR CURRENT RESIDENT", r"OR\s+CURRENT\s+RESIDENT"),
]
MARKERS: list[re.Pattern] = [re.compile(p, re.I) for _, p in _MARKER_DEFS]
_MARKER_NAMES = [n for n, _ in _MARKER_DEFS]
# Generic addressees are real address lines, so they are reported as markers but never stripped.
_KEEP_LINE = re.compile(r"OR\s+CURRENT\s+RESIDENT")

GENERIC_ADDRESSEES: list[re.Pattern] = [
    re.compile(p, re.I)
    for p in (
        r"^(OR\s+)?CURRENT\s+RESIDENT$",
        r"^RESIDENT$",
        r"^HOMEOWNER$",
        r"^HOUSEHOLD$",
        r"^THE\s+\w+\s+HOUSEHOLD$",
        r"^OCCUPANT$",
        r"^POSTAL\s+CUSTOMER$",
        r"^RESIDENTIAL\s+CUSTOMER$",
        r"^OUR\s+NEIGHBORS\s+AT$",
    )
]

_UPS = re.compile(r"\b1Z[0-9A-Z]{16}\b")
_USPS = re.compile(r"\b(9[2-5]\d{18,20}|[A-Z]{2}\d{9}US)\b")
_FEDEX = re.compile(r"\b(\d{12}|\d{15})\b")
_DHL = re.compile(r"\b\d{10}\b")
_SHIP_TO = re.compile(r"^\s*SHIP\s*TO\b[:\s]*(.*)$", re.I)


class _G:
    """A block with normalised geometry."""

    def __init__(self, b: Block, w: int, h: int) -> None:
        xs = [p[0] for p in b.box]
        ys = [p[1] for p in b.box]
        self.b = b
        self.text = b.text.strip()
        self.left, self.right = min(xs), max(xs)
        self.top, self.bottom = min(ys), max(ys)
        self.cx = (self.left + self.right) / 2 / w
        self.cy = (self.top + self.bottom) / 2 / h
        self.height = max(self.bottom - self.top, 1)


def _marker_hits(text: str) -> list[str]:
    return [name for name, pat in zip(_MARKER_NAMES, MARKERS) if pat.search(text)]


def _is_marker_only(text: str) -> bool:
    """True when the line is a postal marker and carries no other words."""
    if not _marker_hits(text) or _KEEP_LINE.fullmatch(text.strip().upper()):
        return False
    rest = text
    for pat in MARKERS:
        rest = pat.sub(" ", rest)
    return not re.search(r"[A-Za-z]", rest)


def _fields(lines: list[_G]) -> dict | None:
    if not lines:
        return None
    return {
        "lines": [g.text for g in lines],
        "confidence": round(sum(g.b.confidence for g in lines) / len(lines), 4),
    }


def _envelope(blocks: list[Block], width: int, height: int) -> dict:
    gs = sorted((_G(b, width, height) for b in blocks if b.text.strip()), key=lambda g: (g.cy, g.cx))
    tol = 0.03 * width

    indicia = [g.text for g in gs if g.cx > 0.65 and g.cy < 0.35]
    markers: list[str] = []
    for g in gs:
        for name in _marker_hits(g.text):
            if name not in markers:
                markers.append(name)

    body = [g for g in gs if not _is_marker_only(g.text)]

    ret_all = [g for g in body if g.cx < 0.45 and g.cy < 0.40]
    ret: list[_G] = []
    if ret_all:
        first = ret_all[0]
        ret = [g for g in ret_all if abs(g.left - first.left) <= tol][:5]

    ret_ids = {id(g) for g in ret}
    cands = [g for g in body if 0.25 < g.cy < 0.95 and id(g) not in ret_ids]
    addr: list[_G] = []
    if len(cands) >= 2:
        med = statistics.median(g.height for g in cands)
        groups: list[list[_G]] = []
        for g in cands:
            cur = groups[-1] if groups else None
            if cur and abs(g.left - cur[0].left) <= tol and max(g.top - cur[-1].bottom, 0) < 2 * med:
                cur.append(g)
            else:
                groups.append([g])
        groups = [grp for grp in groups if len(grp) >= 2]
        if groups:
            addr = max(groups, key=lambda grp: (len(grp), sum(g.b.confidence for g in grp)))

    generic = any(p.match(re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", g.text)).strip()) for g in addr for p in GENERIC_ADDRESSEES)
    return {
        "addressee": _fields(addr),
        "return_address": _fields(ret),
        "indicia": indicia,
        "markers": markers,
        "generic_addressee": generic,
    }


def _label(blocks: list[Block], width: int, height: int) -> dict:
    texts = [b.text.upper() for b in blocks]
    allt = " ".join(texts)
    found: list[dict] = []

    def add(carrier: str, number: str) -> None:
        item = {"carrier": carrier, "number": number}
        if item not in found:
            found.append(item)

    for t in texts:
        s = re.sub(r"\s+", "", t)
        for m in _UPS.finditer(s):
            add("ups", m.group(0))
        for m in _USPS.finditer(s):
            add("usps", m.group(0))
        if "FEDEX" in allt:
            for m in _FEDEX.finditer(t):
                add("fedex", m.group(0))
        if "DHL" in allt:
            for m in _DHL.finditer(t):
                add("dhl", m.group(0))

    ship_to = None
    gs = sorted((_G(b, width, height) for b in blocks if b.text.strip()), key=lambda g: (g.cy, g.cx))
    for i, g in enumerate(gs):
        m = _SHIP_TO.match(g.text)
        if not m:
            continue
        lines: list[_G] = []
        if m.group(1).strip():
            lines.append(_G(Block(m.group(1).strip(), g.b.confidence, g.b.box, g.b.line), width, height))
        for nxt in gs[i + 1:]:
            if len(lines) >= 5:
                break
            if abs(nxt.left - g.left) > 0.05 * width and not lines:
                continue
            if lines and abs(nxt.left - lines[0].left) > 0.05 * width:
                break
            lines.append(nxt)
        ship_to = _fields(lines)
        break
    return {"tracking": found, "ship_to": ship_to}


def _meter(blocks: list[Block]) -> dict:
    best, best_n = None, 0
    for b in blocks:
        cleaned = re.sub(r"[^0-9.,]", "", b.text)
        n = sum(c.isdigit() for c in cleaned)
        if n > best_n:
            best, best_n = (cleaned.replace(",", "."), b), n
    if best is None:
        return {"value": None, "digits": "", "confidence": 0.0}
    digits, block = best
    try:
        value = float(digits)
    except ValueError:
        value = None
    return {"value": value, "digits": digits, "confidence": round(block.confidence, 4)}


def apply(preset: str, blocks: list[Block], width: int, height: int) -> dict | None:
    if preset == "text":
        return None
    if preset == "envelope":
        return _envelope(blocks, width, height)
    if preset == "label":
        return _label(blocks, width, height)
    if preset == "meter":
        return _meter(blocks)
    raise ValueError(f"unknown preset: {preset}")

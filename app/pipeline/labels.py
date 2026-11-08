"""OCR 한글 라벨 -> 방 이름/kind/바닥 (닫힌 어휘 + 부분 일치)."""
from __future__ import annotations

import re

# (키워드, 이름, kind, 바닥 재질)
VOCAB = [
    ("현관", "현관", "entrance", "tile"),
    ("욕실", "욕실", "bathroom", "tile"),
    ("화장실", "욕실", "bathroom", "tile"),
    ("주방", "주방", "kitchen", "wood"),
    ("거실", "거실", "living", "wood"),
    ("침실", "침실", "bedroom", "wood"),
    ("드레스", "드레스룸", "storage", "wood"),
    ("알파", "알파룸", "storage", "wood"),
    ("수납", "수납", "storage", "wood"),
    ("실외기", "실외기실", "utility", "concrete"),
    ("다용도", "다용도실", "utility", "concrete"),
    ("발코니", "발코니", "balcony", "concrete"),
    ("베란다", "발코니", "balcony", "concrete"),
    ("원룸", "원룸", "living", "wood"),
    ("복도", "복도", "hallway", "wood"),
    ("PS", "PS", "other", "concrete"),
]


def match(text: str):
    t = re.sub(r"\s+", "", text)
    for key, name, kind, mat in VOCAB:
        if key in t:
            return name, kind, mat
    return None


AREA_RE = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s*(?:m²|㎡|m2)", re.I)


def parse_area(texts: list[str]) -> float | None:
    """'18.22m²' 같은 첫 면적 표기(5~120㎡ 범위)."""
    for t in texts:
        m = AREA_RE.search(t)
        if m and 5 <= float(m.group(1)) <= 120:
            return float(m.group(1))
    return None

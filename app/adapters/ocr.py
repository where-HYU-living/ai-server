"""PaddleOCR PP-OCRv5 한국어 어댑터. 실패하면 빈 결과(라벨 없이 진행)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


@dataclass
class TextBox:
    text: str
    cx: float  # 원본 이미지 픽셀
    cy: float
    score: float


class PaddleKoreanOcr:
    def __init__(self):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(lang="korean", use_doc_orientation_classify=False,
                              use_doc_unwarping=False, use_textline_orientation=False)

    def read(self, image_rgb: np.ndarray) -> list[TextBox]:
        res = self._ocr.predict(image_rgb[:, :, ::-1].copy())[0]
        boxes = []
        for t, s, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
            poly = np.asarray(poly, dtype=float)
            boxes.append(TextBox(t, float(poly[:, 0].mean()), float(poly[:, 1].mean()), float(s)))
        return boxes


def try_load() -> PaddleKoreanOcr | None:
    try:
        return PaddleKoreanOcr()
    except Exception:  # noqa: BLE001  모델/의존성 없음
        return None

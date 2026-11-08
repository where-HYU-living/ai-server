"""파이프라인 진입점. 모델은 처음 필요할 때 한 번만 로드한다."""
from __future__ import annotations

import io
import os
from functools import lru_cache

import numpy as np
from PIL import Image

from .adapters.ocr import try_load
from .adapters.segmentation import CubiCasaSegmenter, ModelNotLoaded
from .pipeline import route_a, route_b
from .rent.features import Deal
from .rent.model import RentEstimate, RentModel, RentModelNotLoaded
from .schema import FloorPlan
from .validate import validate

INFERENCE_ON = os.getenv("INFERENCE", "off").lower() == "on"

__all__ = ["INFERENCE_ON", "ModelNotLoaded", "RentModelNotLoaded", "estimate_from_image", "estimate_from_meta",
           "estimate_rent", "check"]


@lru_cache(maxsize=1)
def _seg() -> CubiCasaSegmenter:
    return CubiCasaSegmenter()


@lru_cache(maxsize=1)
def _rent() -> RentModel:
    return RentModel.load()


@lru_cache(maxsize=1)
def _ocr():
    return try_load()


def estimate_from_image(data: bytes, plan_id: int, name: str, area_m2: float | None) -> FloorPlan:
    if not INFERENCE_ON:
        raise ModelNotLoaded("INFERENCE=off (이미지 추론 꺼짐)")
    img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    seg = _seg().segment(img)
    ocr = _ocr()
    boxes = ocr.read(img) if ocr else []
    return route_a.run(seg, boxes, plan_id, name, area_m2)


def estimate_from_meta(plan_id: int, name: str, area_m2: float, rooms: int | None) -> FloorPlan:
    return route_b.estimate(plan_id, name, area_m2, rooms)


def estimate_rent(d: Deal) -> RentEstimate:
    if not INFERENCE_ON:
        raise RentModelNotLoaded("INFERENCE=off (적정 월세 추론 꺼짐)")
    return _rent().estimate(d)


def check(fp: FloorPlan) -> list[str]:
    return validate(fp)

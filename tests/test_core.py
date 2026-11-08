import math
from pathlib import Path

import numpy as np
import pytest

from app.pipeline import route_b
from app.pipeline.openings import detect_swing
from app.schema import FloorPlan, Wall
from app.validate import repair, validate


def test_templates_valid():
    for t in route_b.templates():
        assert validate(t) == [], t.id


@pytest.mark.parametrize("area,rooms", [(15, 1), (20, 1), (28, 1), (41, 2)])
def test_route_b_valid(area, rooms):
    fp = route_b.estimate(1, "x", area, rooms)
    assert validate(fp) == []
    assert fp.areaM2 == area


def test_repair_clamps_opening():
    fp = route_b.estimate(1, "x", 20, 1)
    fp.openings[0].offset = 99
    fp = repair(fp)
    assert validate(fp) == []


def _arc_image(side: int, hinge_start: bool):
    """위쪽 수평 벽(y=1.0m) 위 문(0~0.9m). side=+1 이면 벽 왼쪽(화면 위)에 호를 그린다."""
    m = 0.01
    img = np.full((300, 300), 255, np.uint8)
    import cv2
    hx = 0.0 if hinge_start else 0.9
    cx, cy = int(hx / m), int(1.0 / m)
    cv2.line(img, (cx, cy), (cx, cy + side * -90), 0, 1)  # 문짝 선 (벽에 수직)
    a0, a1 = (270, 360) if (hinge_start and side == 1) else (180, 270)
    cv2.ellipse(img, (cx, cy), (90, 90), 0, 0, 0, 0, 1)
    return img, m


def test_swing_side_sign():
    # 벽: (0,1)->(2,1), 진행방향 +x. 화면 위(y 감소)쪽이 왼쪽 => +1
    import cv2
    m = 0.01
    img = np.full((300, 300), 255, np.uint8)
    cv2.line(img, (0, 100), (0, 10), 0, 1)               # 문짝: start 쪽에서 위로
    cv2.ellipse(img, (0, 100), (90, 90), 0, 270, 360, 0, 1)  # 호: 위쪽
    wall = Wall(**{"from": (0.0, 1.0), "to": (2.0, 1.0), "thickness": 0.1})
    sw = detect_swing(img, m, wall, 0.0, 0.9)
    assert sw and sw.opensTo == 1 and sw.hinge == "start"
    sw2 = detect_swing(img[::-1].copy(), m, Wall(**{"from": (0.0, 2.0), "to": (2.0, 2.0), "thickness": 0.1}), 0.0, 0.9)
    assert sw2 and sw2.opensTo == -1


@pytest.mark.skipif(not Path("weights/model_best_val_loss_var.pkl").exists(), reason="가중치 없음")
def test_model_loads():
    from app.adapters.segmentation import CubiCasaSegmenter
    assert CubiCasaSegmenter().segment(np.full((128, 128, 3), 255, np.uint8)).rooms.shape

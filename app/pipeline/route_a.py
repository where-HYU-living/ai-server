"""루트 A: 평면도 이미지 -> FloorPlan."""
from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Point

from ..adapters.ocr import TextBox
from ..adapters.segmentation import Segmentation
from ..schema import Floor, FloorPlan, Opening, Room
from ..validate import repair
from . import labels, vectorize
from .openings import place_openings


def run(seg: Segmentation, ocr: list[TextBox], plan_id: int, name: str, area_m2: float | None,
        ceiling: float = 2.4) -> FloorPlan:
    area = area_m2 or labels.parse_area([b.text for b in ocr]) or 22.0
    m = vectorize.estimate_scale(seg.rooms, area)
    raw = vectorize.extract_rooms(seg.rooms, m)
    if not raw:
        raise ValueError("방을 찾지 못했어요 (이미지가 평면도가 아니거나 해상도가 너무 낮아요).")
    walls = vectorize.build_walls(raw)

    # 원점을 벽 바운딩박스 좌상단으로 옮긴다
    xs = [p for w in walls for p in (w.from_[0], w.to[0])]
    ys = [p for w in walls for p in (w.from_[1], w.to[1])]
    ox, oy = min(xs), min(ys)

    def sh(pt):
        return (round(pt[0] - ox, 2), round(pt[1] - oy, 2))

    gray = cv2.cvtColor(seg.image_rgb, cv2.COLOR_RGB2GRAY)
    openings = place_openings(seg.icons, walls, m, gray, ceiling)  # 원본 좌표에서 계산 후 이동
    for w in walls:
        w.from_, w.to = sh(w.from_), sh(w.to)

    rooms: list[Room] = []
    meta_by_room = []
    for r in raw:
        nm, kind, mat = vectorize.ROOM_META[r.cls]
        meta_by_room.append([nm, kind, mat])
    # OCR 라벨로 이름 덮어쓰기 (원본 픽셀 -> 작업 해상도 -> m)
    for b in ocr:
        hit = labels.match(b.text)
        if not hit:
            continue
        pt = Point(b.cx * seg.ratio[0] * m, b.cy * seg.ratio[1] * m)
        cand = min(zip(raw, meta_by_room), key=lambda rm: rm[0].polygon.distance(pt))
        if cand[0].polygon.distance(pt) <= 1.0:  # 라벨이 방 밖에 걸쳐도 가장 가까운 방(1m 이내)
            cand[1][:] = list(hit)
    seen: dict[str, int] = {}
    for r, (nm, kind, mat) in zip(raw, meta_by_room):
        seen[nm] = seen.get(nm, 0) + 1
        label = nm if seen[nm] == 1 else f"{nm}{seen[nm]}"
        pts = [sh(p) for p in list(r.polygon.exterior.coords)[:-1]]
        rooms.append(Room(name=label, polygon=pts, kind=kind,
                          floor=Floor(material=mat, height=-0.08 if mat == "tile" else 0.0)))

    entrance = _entrance(raw, walls, openings, ox, oy)
    fp = FloorPlan(id=plan_id, name=name, areaM2=area, ceilingHeight=ceiling, walls=walls,
                   openings=openings, rooms=rooms, entrance=entrance)
    return repair(fp)


def _entrance(raw, walls, openings: list[Opening], ox: float, oy: float):
    import math
    entry = [r.polygon for r in raw if r.cls == vectorize.ENTRY]
    best = None
    for o in openings:
        if o.type != "door":
            continue
        w = walls[o.wall]
        ln = math.hypot(w.to[0] - w.from_[0], w.to[1] - w.from_[1]) or 1
        t = (o.offset + o.width / 2) / ln
        c = (w.from_[0] + (w.to[0] - w.from_[0]) * t, w.from_[1] + (w.to[1] - w.from_[1]) * t)
        score = min((p.distance(Point(c[0] + ox, c[1] + oy)) for p in entry), default=0.0) + (0 if w.thickness >= 0.2 else 1.0)
        if best is None or score < best[0]:
            best = (score, c)
    if best:
        return (round(best[1][0], 2), round(best[1][1], 2))
    w = max(walls, key=lambda w: math.hypot(w.to[0] - w.from_[0], w.to[1] - w.from_[1]))
    return (round((w.from_[0] + w.to[0]) / 2, 2), round((w.from_[1] + w.to[1]) / 2, 2))

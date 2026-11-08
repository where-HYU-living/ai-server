"""루트 B: 도면이 없을 때 면적·방 구성으로 추정 평면도. 이미 만든 평면도 JSON 을 템플릿으로 늘리고 줄인다."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from ..schema import FloorPlan
from ..validate import repair

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"
SERVICE = {"entrance", "bathroom", "utility", "balcony", "other", "storage", "hallway"}


@lru_cache(maxsize=1)
def templates() -> list[FloorPlan]:
    out = []
    for p in sorted(TEMPLATE_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        d.pop("fixtures", None)
        out.append(FloorPlan.model_validate(d))
    return out


def _living_rooms(fp: FloorPlan) -> int:
    return sum(1 for r in fp.rooms if (r.kind or "other") not in SERVICE)


def pick(area_m2: float, rooms: int | None) -> FloorPlan:
    def cost(t: FloorPlan) -> float:
        c = abs(math.log(area_m2 / t.areaM2))
        if rooms is not None:
            c += 0.5 * abs(_living_rooms(t) - rooms)
        return c
    return min(templates(), key=cost)


def estimate(plan_id: int, name: str, area_m2: float, rooms: int | None = None) -> FloorPlan:
    t = pick(area_m2, rooms).model_copy(deep=True)
    s = math.sqrt(area_m2 / t.areaM2)
    for w in t.walls:
        w.from_ = (round(w.from_[0] * s, 2), round(w.from_[1] * s, 2))
        w.to = (round(w.to[0] * s, 2), round(w.to[1] * s, 2))
    for o in t.openings:
        o.offset = round(o.offset * s, 2)
    for r in t.rooms:
        r.polygon = [(round(x * s, 2), round(y * s, 2)) for x, y in r.polygon]
    t.entrance = (round(t.entrance[0] * s, 2), round(t.entrance[1] * s, 2))
    t.id, t.name, t.areaM2 = plan_id, name, area_m2
    return repair(t)

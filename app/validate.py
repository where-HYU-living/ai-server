"""검수·보정. 백엔드 FloorPlanValidator 규칙 + 프롬프트의 '출력 전 자체 검수'."""
from __future__ import annotations

import math

from shapely.geometry import Polygon

from .schema import FloorPlan

EPS = 0.02


def wall_len(w) -> float:
    return math.hypot(w.to[0] - w.from_[0], w.to[1] - w.from_[1])


def validate(fp: FloorPlan) -> list[str]:
    errs: list[str] = []
    if fp.areaM2 <= 0:
        errs.append("areaM2 는 0보다 커야 해요.")
    if fp.ceilingHeight <= 0:
        errs.append("ceilingHeight 는 0보다 커야 해요.")
    if not fp.walls:
        errs.append("walls 가 비어 있어요.")
    lengths = [wall_len(w) for w in fp.walls]
    for i, (w, ln) in enumerate(zip(fp.walls, lengths)):
        if ln < 1e-6:
            errs.append(f"walls[{i}] 길이가 0이에요.")
        if w.thickness <= 0:
            errs.append(f"walls[{i}].thickness 는 0보다 커야 해요.")
    for i, o in enumerate(fp.openings):
        if not 0 <= o.wall < len(lengths):
            errs.append(f"openings[{i}].wall({o.wall}) 이 walls 범위 밖이에요.")
            continue
        if o.offset < -EPS or o.width <= 0:
            errs.append(f"openings[{i}] offset/width 가 올바르지 않아요.")
        elif o.offset + o.width > lengths[o.wall] + EPS:
            errs.append(f"openings[{i}] 이 벽 {o.wall} 길이를 넘어가요.")
        if o.height <= 0 or o.sill < 0 or o.sill + o.height > fp.ceilingHeight + EPS:
            errs.append(f"openings[{i}] height/sill 이 올바르지 않아요.")
    if not fp.rooms:
        errs.append("rooms 가 비어 있어요.")
    total = 0.0
    for i, r in enumerate(fp.rooms):
        if len(r.polygon) < 3:
            errs.append(f"rooms[{i}].polygon 은 꼭짓점이 3개 이상이어야 해요.")
        else:
            total += Polygon(r.polygon).area
    # 프롬프트 자체 검수 3: 방 넓이 합이 areaM2 의 ±15% (발코니·벽 포함이라 느슨하게 -15% ~ +100%(발코니 확장형))
    if fp.rooms and fp.areaM2 > 0 and not (0.85 * fp.areaM2 <= total <= 2.0 * fp.areaM2):
        errs.append(f"방 넓이 합({total:.1f}㎡)이 areaM2({fp.areaM2})에서 너무 벗어나요.")
    return errs


def repair(fp: FloorPlan) -> FloorPlan:
    """고칠 수 있는 것만 고친다: 벽 밖으로 나간 개구부 줄이기, 못 고치는 개구부 버리기, 높이 제한."""
    lengths = [wall_len(w) for w in fp.walls]
    kept = []
    for o in fp.openings:
        if not 0 <= o.wall < len(lengths):
            continue
        ln = lengths[o.wall]
        o.width = min(o.width, max(ln - 0.1, 0.1))
        o.offset = min(max(o.offset, 0.0), max(ln - o.width, 0.0))
        o.height = min(o.height, fp.ceilingHeight - o.sill - 0.05)
        if o.width >= 0.3 and o.height > 0.2:
            kept.append(o)
    fp.openings = kept
    return fp

"""T3: 분할 마스크 -> 방 폴리곤(공유 경계) + 벽 선분. 1칸 = 0.1m 격자에서 직각(Manhattan)으로 정리한다."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from ..schema import Wall

CELL = 0.1          # m
OUTER_T, INNER_T = 0.2, 0.1
MIN_ROOM_M2 = 0.8
WALL_CLASS = 2
# CubiCasa 방 클래스 -> (이름, kind, 바닥 재질)
ROOM_META = {
    1: ("발코니", "balcony", "concrete"),
    3: ("주방", "kitchen", "wood"),
    4: ("거실", "living", "wood"),
    5: ("침실", "bedroom", "wood"),
    6: ("욕실", "bathroom", "tile"),
    7: ("현관", "entrance", "tile"),
    9: ("수납", "storage", "wood"),
    11: ("기타", "other", "concrete"),
}
ENTRY, BATH = 7, 6


@dataclass
class RoomRaw:
    cls: int
    polygon: Polygon  # 미터 좌표(이미지 원점 기준)


def estimate_scale(rooms: np.ndarray, area_m2: float | None) -> float:
    """m/px. 방+벽 픽셀 수가 전용면적의 약 1.15배라고 보고 역산."""
    px = int(np.isin(rooms, list(ROOM_META) + [WALL_CLASS]).sum())
    area = area_m2 or 22.0
    return float(np.sqrt(area * 1.15 / max(px, 1)))


def _label_grid(rooms: np.ndarray, m_per_px: float) -> np.ndarray:
    lab = np.where(np.isin(rooms, list(ROOM_META)), rooms, 0).astype(np.uint8)
    walls = rooms == WALL_CLASS
    # 벽 픽셀을 가까운 방에 붙여 방끼리 경계를 공유시킨다
    dist, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    reach = 0.5 / m_per_px  # 벽 두께 한계 0.5m
    fill = walls & (lab == 0) & (dist <= reach)
    lab[fill] = lab[iy[fill], ix[fill]]
    h, w = lab.shape
    wg, hg = max(8, round(w * m_per_px / CELL)), max(8, round(h * m_per_px / CELL))
    return cv2.resize(lab, (wg, hg), interpolation=cv2.INTER_NEAREST)


def _clean(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid)
    k = np.ones((3, 3), np.uint8)
    min_cells = int(MIN_ROOM_M2 / CELL**2)
    for c in ROOM_META:
        m = (grid == c).astype(np.uint8)
        if not m.any():
            continue
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_cells:
                out[(comp == i) & (out == 0)] = c
    # 방 사이 한두 칸 틈은 가까운 방으로 메운다
    closed = cv2.morphologyEx((out > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    gap = closed & (out == 0)
    if gap.any():
        _, (iy, ix) = ndimage.distance_transform_edt(out == 0, return_indices=True)
        out[gap] = out[iy[gap], ix[gap]]
    return out


def _mask_to_polygons(mask: np.ndarray) -> list[Polygon]:
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    polys = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_ROOM_M2 / CELL**2:
            continue
        cm = comp == i
        boxes = []
        for y in range(cm.shape[0]):
            row = cm[y]
            if not row.any():
                continue
            d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
            for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                boxes.append(box(s * CELL, y * CELL, e * CELL, (y + 1) * CELL))
        u = unary_union(boxes)
        if isinstance(u, MultiPolygon):
            u = max(u.geoms, key=lambda g: g.area)
        u = Polygon(u.exterior).simplify(0.0)
        if u.area >= MIN_ROOM_M2:
            polys.append(u)
    return polys


def extract_rooms(rooms: np.ndarray, m_per_px: float) -> list[RoomRaw]:
    grid = _clean(_label_grid(rooms, m_per_px))
    out: list[RoomRaw] = []
    for c in ROOM_META:
        for p in _mask_to_polygons(grid == c):
            out.append(RoomRaw(c, p))
    return out


def build_walls(rooms: list[RoomRaw]) -> list[Wall]:
    """방 폴리곤 변을 선분 위에서 겹쳐 세어, 한 방만 접하면 바깥 벽(0.2), 둘 이상이면 칸막이(0.1)."""
    lines: dict[tuple[str, float], list[tuple[float, float, int]]] = {}
    for idx, r in enumerate(rooms):
        pts = list(r.polygon.exterior.coords)
        for a, b in zip(pts[:-1], pts[1:]):
            if abs(a[1] - b[1]) < 1e-6 and abs(a[0] - b[0]) > 1e-6:
                lines.setdefault(("h", round(a[1], 3)), []).append((min(a[0], b[0]), max(a[0], b[0]), idx))
            elif abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) > 1e-6:
                lines.setdefault(("v", round(a[0], 3)), []).append((min(a[1], b[1]), max(a[1], b[1]), idx))
    walls: list[Wall] = []
    for (kind, pos), ivs in sorted(lines.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        pts = sorted({p for lo, hi, _ in ivs for p in (lo, hi)})
        runs: list[list[float]] = []  # [lo, hi, thickness]
        for p, q in zip(pts[:-1], pts[1:]):
            cover = {i for lo, hi, i in ivs if lo <= p + 1e-9 and hi >= q - 1e-9}
            if not cover:
                continue
            t = INNER_T if len(cover) > 1 else OUTER_T
            if runs and abs(runs[-1][1] - p) < 1e-9 and runs[-1][2] == t:
                runs[-1][1] = q
            else:
                runs.append([p, q, t])
        for lo, hi, t in runs:
            if hi - lo < 0.25:
                continue
            f, to = ((lo, pos), (hi, pos)) if kind == "h" else ((pos, lo), (pos, hi))
            walls.append(Wall(**{"from": f, "to": to, "thickness": t}))
    walls.sort(key=lambda w: -w.thickness)  # 바깥 벽이 앞
    return walls

"""T4: 문·창 아이콘 -> openings, 문 열림(hinge/opensTo) 기하 추정."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..schema import Opening, Swing, Wall

DOOR, WINDOW = 2, 1


def _components(mask: np.ndarray, min_px: int):
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            ys, xs = np.nonzero(comp == i)
            yield xs, ys


def _seg_dist(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    ln2 = dx * dx + dy * dy
    t = 0.0 if ln2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / ln2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy)), t


def _dark(gray: np.ndarray, pts_px: np.ndarray, thresh: int = 150) -> int:
    xs = np.clip(np.round(pts_px[:, 0]).astype(int), 0, gray.shape[1] - 1)
    ys = np.clip(np.round(pts_px[:, 1]).astype(int), 0, gray.shape[0] - 1)
    return int((gray[ys, xs] < thresh).sum())


def detect_swing(gray: np.ndarray, m_per_px: float, wall: Wall, offset: float, width: float) -> Swing | None:
    """호(arc)+문짝 선이 그려진 쪽을 찾는다. 쪽 = 벽 진행방향 왼쪽(+1)/오른쪽(-1), 경첩 = 문짝 선이 있는 끝."""
    (x0, y0), (x1, y1) = wall.from_, wall.to
    ln = math.hypot(x1 - x0, y1 - y0)
    d = ((x1 - x0) / ln, (y1 - y0) / ln)
    nl = (d[1], -d[0])  # 화면(y 아래)에서 from->to 를 볼 때 왼쪽
    step = 0.02
    ts = np.arange(0.0, width, step)

    def sample(side: int, r_lo: float, r_hi: float, t_lo: float, t_hi: float) -> int:
        tt, rr = np.meshgrid(np.arange(t_lo, t_hi, step), np.arange(r_lo, r_hi, step))
        if tt.size == 0:
            return 0
        px = x0 + d[0] * (offset + tt) + side * nl[0] * rr
        py = y0 + d[1] * (offset + tt) + side * nl[1] * rr
        return _dark(gray, np.stack([px.ravel(), py.ravel()], 1) / m_per_px)

    plus = sample(1, 0.15, width + 0.1, 0, width)
    minus = sample(-1, 0.15, width + 0.1, 0, width)
    if max(plus, minus) < 8 or abs(plus - minus) < 0.15 * max(plus, minus):
        return None
    side = 1 if plus > minus else -1
    edge = 0.12 * width
    start = sample(side, 0.3 * width, 0.9 * width, 0, edge)
    end = sample(side, 0.3 * width, 0.9 * width, width - edge, width)
    hinge = "start" if start >= end else "end"
    return Swing(hinge=hinge, opensTo=side)


def place_openings(icons: np.ndarray, walls: list[Wall], m_per_px: float, gray: np.ndarray,
                   ceiling: float = 2.4) -> list[Opening]:
    out: list[Opening] = []
    if not walls:
        return out
    for cls, typ in ((DOOR, "door"), (WINDOW, "window")):
        for xs, ys in _components(icons == cls, min_px=12):
            pts = np.stack([xs, ys], 1) * m_per_px
            c = pts.mean(0)
            best = None
            for wi, w in enumerate(walls):
                dist, _ = _seg_dist(c, w.from_, w.to)
                if best is None or dist < best[0]:
                    best = (dist, wi)
            if best is None or best[0] > 0.6:
                continue
            wi = best[1]
            w = walls[wi]
            ln = math.hypot(w.to[0] - w.from_[0], w.to[1] - w.from_[1])
            d = ((w.to[0] - w.from_[0]) / ln, (w.to[1] - w.from_[1]) / ln)
            proj = (pts[:, 0] - w.from_[0]) * d[0] + (pts[:, 1] - w.from_[1]) * d[1]
            lo, hi = float(np.percentile(proj, 3)), float(np.percentile(proj, 97))
            width = hi - lo
            if typ == "door":
                width = min(max(width, 0.7), 1.0)
            else:
                width = min(max(width, 0.5), 3.5)
            mid = (lo + hi) / 2
            offset = min(max(mid - width / 2, 0.0), max(ln - width, 0.0))
            if typ == "door":
                out.append(Opening(type="door", wall=wi, offset=round(offset, 2), width=round(width, 2),
                                   height=2.1, sill=0.0, style="swing",
                                   swing=detect_swing(gray, m_per_px, w, offset, width)))
            else:
                big = width >= 1.5
                out.append(Opening(type="window", wall=wi, offset=round(offset, 2), width=round(width, 2),
                                   height=min(2.0 if big else 1.2, ceiling - 0.2), sill=0.1 if big else 0.9,
                                   style="sliding" if big else None))
    return out

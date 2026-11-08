"""27장 평면도 이미지를 파이프라인에 돌려 정답 JSON 과 숫자로 비교한다 (캡처 없이).
사용: .venv/bin/python scripts/evaluate.py [이미지 폴더]"""
import glob
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from app import service  # noqa: E402

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "Downloads/chatatda-floorplans")
GT_FALLBACK = Path(__file__).resolve().parents[1] / "templates"


def room_area(fp):
    rooms = fp["rooms"] if isinstance(fp, dict) else fp.rooms
    return sum(Polygon(r["polygon"] if isinstance(r, dict) else r.polygon).area for r in rooms)


rows = []
for png in sorted(D.glob("*.png")):
    pid = int(png.name[:3])
    gt_path = png.with_suffix(".json")
    if not gt_path.exists():
        gt_path = GT_FALLBACK / f"{pid}.json"
    gt = json.loads(gt_path.read_text()) if gt_path.exists() else None
    area = gt["areaM2"] if gt else None
    t = time.time()
    try:
        fp = service.estimate_from_image(png.read_bytes(), pid, png.stem, area)
        issues = service.check(fp)
        row = dict(id=pid, ok=True, issues=len(issues), sec=round(time.time() - t, 1),
                   rooms=len(fp.rooms), doors=sum(o.type == "door" for o in fp.openings),
                   wins=sum(o.type == "window" for o in fp.openings),
                   swing=sum(bool(o.swing) for o in fp.openings if o.type == "door"),
                   area_ratio=round(room_area(fp) / area, 2) if area else None)
    except Exception as e:  # noqa: BLE001
        row = dict(id=pid, ok=False, err=str(e)[:60])
    if gt:
        row.update(gt_rooms=len(gt["rooms"]), gt_doors=sum(o["type"] == "door" for o in gt["openings"]),
                   gt_wins=sum(o["type"] == "window" for o in gt["openings"]),
                   gt_area_ratio=round(room_area(gt) / gt["areaM2"], 2))
    rows.append(row)
    print(row)

ok = [r for r in rows if r["ok"]]
print(f"\n성공 {len(ok)}/{len(rows)}, 검수 통과 {sum(r['issues'] == 0 for r in ok)}/{len(ok)}")
pairs = [(r["doors"], r["gt_doors"]) for r in ok if "gt_doors" in r]
if pairs:
    print("문 개수 MAE", round(float(np.mean([abs(a - b) for a, b in pairs])), 2),
          "| 방 개수 MAE", round(float(np.mean([abs(r["rooms"] - r["gt_rooms"]) for r in ok if "gt_rooms" in r])), 2),
          "| 면적비 평균", round(float(np.mean([r["area_ratio"] for r in ok if r["area_ratio"]])), 2))

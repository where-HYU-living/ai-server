"""동 단위 외부 피처 -> data/dong_features.json

실거래에 좌표가 없어서 동 중심 기준으로 붙인다.
  campus_m  동 중심 ~ 한양대 사자상 직선거리 (campus-data 원점)
  subway_m  동 중심 ~ 가장 가까운 지하철역 (카카오 로컬 SW8)
  bus_500m  동 중심 500m 안 버스 정류장 수 (백엔드 geo/bus-stops.json)

실행 (ai-server/ 에서): KAKAO_REST_KEY=... .venv/bin/python scripts/build_dong_features.py deals.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dong_features.json"
BUS_STOPS = ROOT.parent / "backend" / "src" / "main" / "resources" / "geo" / "bus-stops.json"
CAMPUS = (37.5561486, 127.044341)  # 사자상
KAKAO = "https://dapi.kakao.com"


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lng1, lat2, lng2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def kakao(path: str, key: str, **params) -> dict:
    req = urllib.request.Request(f"{KAKAO}{path}?{urllib.parse.urlencode(params)}",
                                 headers={"Authorization": f"KakaoAK {key}"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def dong_center(dong: str, key: str) -> tuple[float, float] | None:
    for doc in kakao("/v2/local/search/address.json", key, query=f"서울 {dong}").get("documents", []):
        if doc.get("address_type") == "REGION":
            return float(doc["y"]), float(doc["x"])
    return None


def nearest_subway(center: tuple[float, float], key: str) -> float | None:
    docs = kakao("/v2/local/search/category.json", key, category_group_code="SW8",
                 y=center[0], x=center[1], radius=3000, sort="distance").get("documents", [])
    return float(docs[0]["distance"]) if docs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="실거래 CSV (dong 열)")
    a = ap.parse_args()
    key = os.environ["KAKAO_REST_KEY"]

    with a.csv.open(encoding="utf-8") as f:
        dongs = sorted({r["dong"] for r in csv.DictReader(f)})
    stops = [(s["lat"], s["lng"]) for s in json.loads(BUS_STOPS.read_text(encoding="utf-8"))["stops"]]

    out = {}
    for dong in dongs:
        c = dong_center(dong, key)
        if c is None:
            print(f"좌표 없음: {dong}")
            continue
        out[dong] = {"lat": c[0], "lng": c[1],
                     "campus_m": round(distance_m(c, CAMPUS)),
                     "subway_m": nearest_subway(c, key),
                     "bus_500m": sum(distance_m(c, s) <= 500 for s in stops)}
        print(dong, out[dong])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT} ({len(out)}개 동)")


if __name__ == "__main__":
    main()

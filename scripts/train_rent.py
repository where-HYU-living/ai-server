"""적정 월세 모델 학습 + 룰 기반(동 중앙값)과 비교.

실거래 CSV 받기 (백엔드 로컬 DB):
  docker exec chatatda-db psql -U chatatda -d chatatda -c "copy (select dong, area_m2, deposit, rent, contract_ym,
      floor, built_year, building_type, contract_type from real_deals) to stdout csv header" > data/raw/deals.csv

실행 (ai-server/ 에서):
  KAKAO_REST_KEY=... .venv/bin/python scripts/build_dong_features.py data/raw/deals.csv   # data/dong_features.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/train_rent.py data/raw/deals.csv             # weights/rent/

학습 데이터: 월세 계약(rent > 0), 면적 5~60㎡, ㎡당 환산 월세 상·하위 1% 제외. 갱신 계약도 넣고 신규/갱신을 피처로 준다.
검증: 마지막 N개월(기본 3) 신규 월세 계약을 떼어 두고, 그 이전 거래만으로
  - 룰 기반: 백엔드 FairRentEstimator 와 같은 방식 (신규 월세만, 같은 동 → 10건 미만이면 인접 동까지,
    면적 ±30%, 직전 12개월, ㎡당 환산 월세 중앙값)
  - ML: 분위 회귀 50% 값 (low~high = 25~75%)
을 내서 MAE(원)·절대오차율 중앙값·범위 적중률을 비교한다. 하이퍼파라미터 몇 조합 중 가장 좋은 것으로
전체 데이터를 다시 학습해 저장한다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rent.features import Deal, converted_rent, target, ym_index  # noqa: E402
from app.rent.model import MODEL, RentModel, fit  # noqa: E402

DONG_FEATURES = ROOT / "data" / "dong_features.json"
AREA_TOLERANCE, PERIOD_MONTHS, MIN_SAMPLE = 0.30, 12, 10  # 백엔드 pricing 설정과 같게
AREA_RANGE = (5.0, 60.0)
TRIM = 0.01
ADJACENT = [("행당동", "사근동"), ("행당동", "마장동"), ("행당동", "응봉동"), ("행당동", "하왕십리동"), ("행당동", "도선동"),
            ("사근동", "마장동"), ("사근동", "용답동"), ("마장동", "하왕십리동"), ("마장동", "용답동"), ("마장동", "답십리동"),
            ("홍익동", "도선동"), ("홍익동", "상왕십리동"), ("홍익동", "하왕십리동"), ("도선동", "하왕십리동"),
            ("상왕십리동", "하왕십리동"), ("용답동", "답십리동"), ("답십리동", "용두동"), ("답십리동", "전농동")]  # 백엔드 DongAdjacency
GRID = [
    {"learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 20, "max_leaf_nodes": 31},
    {"learning_rate": 0.05, "max_iter": 500, "min_samples_leaf": 40, "max_leaf_nodes": 31},
    {"learning_rate": 0.03, "max_iter": 600, "min_samples_leaf": 30, "max_leaf_nodes": 15},
    {"learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": 50, "max_leaf_nodes": 63},
]


def neighbors(dong: str) -> set[str]:
    return {b for a, b in ADJACENT if a == dong} | {a for a, b in ADJACENT if b == dong}


def _int(v: str | None) -> int | None:
    return int(float(v)) if v not in ("", None) else None


def load_deals(path: Path) -> list[Deal]:
    with path.open(encoding="utf-8") as f:
        return [Deal(dong=r["dong"], area_m2=float(r["area_m2"]), contract_ym=int(r["contract_ym"]),
                     deposit=int(r["deposit"]), rent=int(r["rent"]), floor=_int(r.get("floor")),
                     built_year=_int(r.get("built_year")), building_type=r.get("building_type") or None,
                     contract_type=r.get("contract_type") or None)
                for r in csv.DictReader(f)]


def clean(deals: list[Deal]) -> tuple[list[Deal], dict]:
    monthly = [d for d in deals if d.rent > 0 and AREA_RANGE[0] <= d.area_m2 <= AREA_RANGE[1]]
    per = np.array([target(d) for d in monthly])
    lo, hi = np.quantile(per, [TRIM, 1 - TRIM])
    kept = [d for d, p in zip(monthly, per) if lo <= p <= hi]
    return kept, {"raw": len(deals), "monthly_in_area": len(monthly), "kept": len(kept),
                  "per_m2_trim_won": [round(float(lo)), round(float(hi))]}


def rule_per_m2(pool: list[Deal], d: Deal) -> float | None:
    """백엔드 FairRentEstimator 와 같은 ㎡당 적정가. pool 은 신규 월세 계약만."""
    t = ym_index(d.contract_ym)
    lo, hi = round(d.area_m2 * (1 - AREA_TOLERANCE), 2), round(d.area_m2 * (1 + AREA_TOLERANCE), 2)

    def pick(dongs: set[str]) -> list[Deal]:
        return [x for x in pool if x.dong in dongs and 1 <= t - ym_index(x.contract_ym) <= PERIOD_MONTHS
                and lo <= x.area_m2 <= hi]   # 계약 전달까지 12개월

    rows = pick({d.dong})
    if len(rows) < MIN_SAMPLE:
        rows = pick({d.dong} | neighbors(d.dong))
    if len(rows) < MIN_SAMPLE:
        return None
    return float(np.median([target(x) for x in rows]))


def score(pred: np.ndarray, actual: np.ndarray) -> dict:
    err = np.abs(pred - actual)
    return {"mae_won": round(float(err.mean())), "median_ape": round(float(np.median(err / actual)), 4)}


def evaluate(deals: list[Deal], dong_features: dict, holdout_months: int, params: dict) -> dict:
    last = max(ym_index(d.contract_ym) for d in deals)
    train = [d for d in deals if ym_index(d.contract_ym) <= last - holdout_months]
    test = [d for d in deals if ym_index(d.contract_ym) > last - holdout_months
            and d.contract_type == "신규"]
    new_pool = [d for d in train if d.contract_type == "신규"]
    model = RentModel(fit(train, dong_features, params=params))
    q = model.per_m2(test)

    rows = []
    for d, (lo, mid, hi) in zip(test, q):
        r = rule_per_m2(new_pool, d)
        if r is not None:
            a = d.area_m2
            rows.append((converted_rent(d.deposit, d.rent), r * a, mid * a, lo * a, hi * a))
    actual, rule, ml, low, high = (np.array(c) for c in zip(*rows))
    return {"train": len(train), "test": len(test), "compared": len(rows), "holdout_months": holdout_months,
            "rule_median": score(rule, actual), "ml": score(ml, actual),
            "ml_range_hit": round(float(np.mean((actual >= low) & (actual <= high))), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--holdout-months", type=int, default=3)
    a = ap.parse_args()

    deals, data = clean(load_deals(a.csv))
    dong_features = json.loads(DONG_FEATURES.read_text(encoding="utf-8")) if DONG_FEATURES.exists() else {}
    print(json.dumps(data, ensure_ascii=False))

    trials = []
    for p in GRID:
        t0 = time.time()
        r = evaluate(deals, dong_features, a.holdout_months, p)
        trials.append({"params": p, **r})
        print(f"{p} → 룰 {r['rule_median']} / ML {r['ml']} / 범위 적중 {r['ml_range_hit']} ({time.time() - t0:.0f}s)")
    best = min(trials, key=lambda t: t["ml"]["median_ape"])

    import sklearn
    yms = sorted({d.contract_ym for d in deals})
    metrics = {"trained_on": date.today().isoformat(), "period": [yms[0], yms[-1]], "data": data,
               "dong_features": sorted(dong_features), "sklearn": sklearn.__version__,
               "best": best, "trials": trials}
    RentModel(fit(deals, dong_features, params=best["params"])).save()
    (MODEL.parent / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"가장 좋은 조합: {best['params']}\n저장: {MODEL}")


if __name__ == "__main__":
    main()

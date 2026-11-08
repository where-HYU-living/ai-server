"""적정 월세 모델 학습 + 룰 기반(동 중앙값)과 비교.

실거래 CSV 받기 (백엔드 DB):
  psql -h localhost -U chatatda -d chatatda -c "\\copy (select dong, area_m2, deposit, rent, contract_ym, floor,
      built_year, building_type, contract_type from real_deals) to 'deals.csv' csv header"

실행 (ai-server/ 에서):
  .venv/bin/python scripts/build_dong_features.py deals.csv      # data/dong_features.json
  .venv/bin/python scripts/train_rent.py deals.csv               # weights/rent/model.joblib, metrics.json

검증: 마지막 N개월(기본 3) 거래를 떼어 두고, 그 이전 거래만으로
  - 룰 기반: 같은 동·면적 ±30%·직전 12개월 ㎡당 환산 월세 중앙값 (표본 10건 미만이면 동 전체 중앙값)
  - ML: 분위 회귀 50% 값
을 내서 MAE(원)와 절대오차율 중앙값을 비교한다. 그 다음 전체 거래로 다시 학습해 저장한다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rent.features import Deal, converted_rent, ym_index  # noqa: E402
from app.rent.model import MODEL, RentModel, fit  # noqa: E402

DONG_FEATURES = ROOT / "data" / "dong_features.json"
AREA_TOLERANCE, PERIOD_MONTHS, MIN_SAMPLE = 0.30, 12, 10  # 백엔드 pricing 설정과 같게


def _int(v: str) -> int | None:
    return int(float(v)) if v not in ("", None) else None


def load_deals(path: Path) -> list[Deal]:
    with path.open(encoding="utf-8") as f:
        return [Deal(dong=r["dong"], area_m2=float(r["area_m2"]), contract_ym=int(r["contract_ym"]),
                     deposit=int(r["deposit"]), rent=int(r["rent"]), floor=_int(r.get("floor", "")),
                     built_year=_int(r.get("built_year", "")), building_type=r.get("building_type") or None,
                     contract_type=r.get("contract_type") or None)
                for r in csv.DictReader(f) if float(r["area_m2"]) > 0]


def rule_per_m2(train: list[Deal], d: Deal) -> float | None:
    """룰 기반 적정가 (㎡당). 백엔드 FairRentEstimator 를 단순화했다 (인접 동 확장 대신 동 전체로 넓힘)."""
    t = ym_index(d.contract_ym)
    recent = [x for x in train if x.dong == d.dong and 0 < t - ym_index(x.contract_ym) <= PERIOD_MONTHS]
    near = [x for x in recent if abs(x.area_m2 - d.area_m2) <= d.area_m2 * AREA_TOLERANCE]
    pool = near if len(near) >= MIN_SAMPLE else recent
    if not pool:
        return None
    return float(np.median([converted_rent(x.deposit, x.rent) / x.area_m2 for x in pool]))


def compare(deals: list[Deal], dong_features: dict, holdout_months: int) -> dict:
    last = max(ym_index(d.contract_ym) for d in deals)
    train = [d for d in deals if ym_index(d.contract_ym) <= last - holdout_months]
    test = [d for d in deals if ym_index(d.contract_ym) > last - holdout_months]
    model = RentModel(fit(train, dong_features))
    ml = model.per_m2(test)[:, 1]

    rows = []
    for d, m in zip(test, ml):
        r = rule_per_m2(train, d)
        if r is not None:
            rows.append((converted_rent(d.deposit, d.rent), r * d.area_m2, m * d.area_m2))
    actual, rule, pred = (np.array(c) for c in zip(*rows))

    def score(p):
        return {"mae_won": round(float(np.mean(np.abs(p - actual)))),
                "median_ape": round(float(np.median(np.abs(p - actual) / actual)), 4)}

    return {"train": len(train), "test": len(test), "compared": len(rows),
            "holdout_months": holdout_months, "rule_median": score(rule), "ml": score(pred)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--holdout-months", type=int, default=3)
    a = ap.parse_args()

    deals = load_deals(a.csv)
    dong_features = json.loads(DONG_FEATURES.read_text(encoding="utf-8")) if DONG_FEATURES.exists() else {}
    metrics = compare(deals, dong_features, a.holdout_months)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    RentModel(fit(deals, dong_features)).save()
    (MODEL.parent / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {MODEL}")


if __name__ == "__main__":
    main()

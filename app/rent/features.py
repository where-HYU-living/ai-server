"""적정 월세 모델 입력. 실거래 한 건(학습)과 매물 한 건(추정)을 같은 피처로 만든다."""
from __future__ import annotations

import math
from dataclasses import dataclass

CONVERSION_RATE = 0.046  # 백엔드 pricing.conversion-rate 와 같게
BASE_YM = 2020 * 12      # 계약 시점 피처의 기준 (2020-01 = 0)

NUMERIC = ["area_m2", "floor", "age", "contract_index", "campus_m", "subway_m", "bus_500m"]
CATEGORICAL = ["dong", "building_type", "contract_type"]
COLUMNS = NUMERIC + CATEGORICAL
DONG_KEYS = ["campus_m", "subway_m", "bus_500m"]  # 동 단위 외부 피처 (scripts/build_dong_features.py)


@dataclass
class Deal:
    dong: str
    area_m2: float
    contract_ym: int                   # 202510 형식
    deposit: int = 0                   # 원
    rent: int = 0                      # 원
    floor: int | None = None
    built_year: int | None = None
    building_type: str | None = None   # OFFICETEL | ROWHOUSE | DETACHED
    contract_type: str | None = None   # 신규 | 갱신


def converted_rent(deposit: float, rent: float, rate: float = CONVERSION_RATE) -> float:
    """보증금을 월세로 환산해 더한 금액 (원/월)."""
    return rent + deposit * rate / 12


def target(d: Deal) -> float:
    """학습 목표: ㎡당 환산 월세. 룰 기반(동 중앙값)과 같은 단위라 비교가 쉽다."""
    return converted_rent(d.deposit, d.rent) / d.area_m2


def ym_index(ym: int) -> int:
    return (ym // 100) * 12 + (ym % 100 - 1) - BASE_YM


def row(d: Deal, dong_features: dict[str, dict]) -> dict:
    """모델 입력 한 행. 없는 값은 NaN (GBM 이 결측으로 처리)."""
    nan = math.nan
    extra = dong_features.get(d.dong, {})
    age = (d.contract_ym // 100 - d.built_year) if d.built_year else nan
    out = {
        "area_m2": float(d.area_m2),
        "floor": float(d.floor) if d.floor is not None else nan,
        "age": float(age),
        "contract_index": float(ym_index(d.contract_ym)),
        "dong": d.dong,
        "building_type": d.building_type,
        "contract_type": d.contract_type,
    }
    for k in DONG_KEYS:
        v = extra.get(k)
        out[k] = float(v) if v is not None else nan
    return out

"""적정 월세 모델: ㎡당 환산 월세의 25·50·75% 분위를 GBM 분위 회귀 3개로 추정한다."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import CATEGORICAL, COLUMNS, Deal, row, target

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "weights" / "rent" / "model.joblib"
QUANTILES = (0.25, 0.5, 0.75)


class RentModelNotLoaded(RuntimeError):
    pass


@dataclass
class RentEstimate:
    fair_rent: int   # 원/월 (환산 월세)
    low: int
    high: int


def _encode(rows: list[dict], vocab: dict[str, list[str]]) -> np.ndarray:
    """범주형은 vocab 인덱스로, 처음 보는 값은 NaN 으로 바꾼다."""
    index = {c: {v: i for i, v in enumerate(vocab[c])} for c in CATEGORICAL}
    out = np.empty((len(rows), len(COLUMNS)), dtype=float)
    for i, r in enumerate(rows):
        for j, c in enumerate(COLUMNS):
            v = r[c]
            if c in index:
                out[i, j] = index[c].get(v, np.nan) if v is not None else np.nan
            else:
                out[i, j] = v
    return out


def fit(deals: list[Deal], dong_features: dict[str, dict], seed: int = 0) -> dict:
    """학습해서 저장 가능한 묶음(dict)을 돌려준다."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rows = [row(d, dong_features) for d in deals]
    vocab = {c: sorted({r[c] for r in rows if r[c] is not None}) for c in CATEGORICAL}
    x = _encode(rows, vocab)
    y = np.array([target(d) for d in deals])
    cat_mask = [c in CATEGORICAL for c in COLUMNS]
    models = {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, categorical_features=cat_mask,
                                          max_iter=300, learning_rate=0.05, min_samples_leaf=20,
                                          random_state=seed)
        models[q] = m.fit(x, y)
    return {"models": models, "vocab": vocab, "dong_features": dong_features, "n": len(deals)}


class RentModel:
    def __init__(self, bundle: dict):
        self.bundle = bundle

    @classmethod
    def load(cls, path: Path = MODEL) -> "RentModel":
        if not path.exists():
            raise RentModelNotLoaded(f"모델 파일 없음: {path}")
        import joblib
        return cls(joblib.load(path))

    def save(self, path: Path = MODEL) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.bundle, path)

    def per_m2(self, deals: list[Deal]) -> np.ndarray:
        """(n, 3) — 25·50·75% 분위의 ㎡당 환산 월세. 분위가 뒤집히면 정렬해 바로잡는다."""
        b = self.bundle
        x = _encode([row(d, b["dong_features"]) for d in deals], b["vocab"])
        pred = np.column_stack([b["models"][q].predict(x) for q in QUANTILES])
        return np.sort(pred, axis=1)

    def estimate(self, d: Deal) -> RentEstimate:
        low, mid, high = (self.per_m2([d])[0] * d.area_m2).round(-2)
        return RentEstimate(fair_rent=int(mid), low=int(low), high=int(high))

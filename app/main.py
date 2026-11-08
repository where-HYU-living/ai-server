from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import service
from .rent.features import Deal
from .schema import FloorPlan

app = FastAPI(title="찾았다 평면도 AI 서버", version="0.1.0")


class EstimateMeta(BaseModel):
    id: int
    name: str
    areaM2: float = Field(gt=3, lt=200)
    rooms: int | None = Field(default=None, ge=0, le=6)


class RentQuery(BaseModel):
    dong: str
    areaM2: float = Field(gt=3, lt=200)
    contractYm: int = Field(ge=200001, le=209912)  # 이 시점 기준 적정가 (202610 형식)
    floor: int | None = None
    builtYear: int | None = None
    buildingType: str | None = None   # OFFICETEL | ROWHOUSE | DETACHED
    contractType: str | None = None   # 신규 | 갱신


class RentResult(BaseModel):
    fairRent: int   # 원/월, 보증금 환산 포함
    low: int        # 25% 분위
    high: int       # 75% 분위
    model: str


class Result(BaseModel):
    floorplan: dict
    issues: list[str]
    route: str


@app.get("/health")
def health():
    from .adapters.segmentation import WEIGHTS
    from .rent.model import MODEL
    return {"status": "ok", "inference": "on" if service.INFERENCE_ON else "off", "weights": WEIGHTS.exists(),
            "rentModel": MODEL.exists()}


@app.post("/v1/floorplan/estimate", response_model=Result)
async def estimate_image(image: UploadFile = File(...), id: int = Form(...), name: str = Form(...),
                         areaM2: float | None = Form(None)):
    """루트 A: 평면도 이미지 -> FloorPlan JSON."""
    try:
        fp = service.estimate_from_image(await image.read(), id, name, areaM2)
    except service.ModelNotLoaded as e:
        raise HTTPException(503, f"모델 미적재: {e}")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return Result(floorplan=fp.dump(), issues=service.check(fp), route="image")


@app.post("/v1/floorplan/estimate-from-meta", response_model=Result)
def estimate_meta(body: EstimateMeta):
    """루트 B: 도면 없이 면적·방 수로 추정."""
    fp = service.estimate_from_meta(body.id, body.name, body.areaM2, body.rooms)
    return Result(floorplan=fp.dump(), issues=service.check(fp), route="template")


@app.post("/v1/rent/estimate", response_model=RentResult)
def estimate_rent(q: RentQuery):
    """적정 월세 (분위 회귀 25·50·75%)."""
    d = Deal(dong=q.dong, area_m2=q.areaM2, contract_ym=q.contractYm, floor=q.floor, built_year=q.builtYear,
             building_type=q.buildingType, contract_type=q.contractType)
    try:
        r = service.estimate_rent(d)
    except service.RentModelNotLoaded as e:
        raise HTTPException(503, f"모델 미적재: {e}")
    return RentResult(fairRent=r.fair_rent, low=r.low, high=r.high, model="gbm-quantile")


@app.post("/v1/floorplan/validate")
def validate_plan(fp: FloorPlan):
    issues = service.check(fp)
    return {"ok": not issues, "issues": issues}

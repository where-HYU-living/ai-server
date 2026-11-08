from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import service
from .schema import FloorPlan

app = FastAPI(title="찾았다 평면도 AI 서버", version="0.1.0")


class EstimateMeta(BaseModel):
    id: int
    name: str
    areaM2: float = Field(gt=3, lt=200)
    rooms: int | None = Field(default=None, ge=0, le=6)


class Result(BaseModel):
    floorplan: dict
    issues: list[str]
    route: str


@app.get("/health")
def health():
    from .adapters.segmentation import WEIGHTS
    return {"status": "ok", "inference": "on" if service.INFERENCE_ON else "off", "weights": WEIGHTS.exists()}


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


@app.post("/v1/floorplan/validate")
def validate_plan(fp: FloorPlan):
    issues = service.check(fp)
    return {"ok": not issues, "issues": issues}

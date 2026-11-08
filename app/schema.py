"""FloorPlan 스키마. 프론트 src/api/types.ts 의 FloorPlan 과 1:1 (fixtures 는 쓰지 않는다)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Point = tuple[float, float]
RoomKind = Literal["bedroom", "living", "kitchen", "bathroom", "entrance", "balcony",
                   "utility", "storage", "hallway", "other"]


class Wall(BaseModel):
    from_: Point = Field(alias="from")
    to: Point
    thickness: float
    height: Optional[float] = None

    model_config = {"populate_by_name": True}


class Swing(BaseModel):
    hinge: Literal["start", "end"]
    opensTo: Literal[1, -1]


class Opening(BaseModel):
    type: Literal["door", "window"]
    wall: int
    offset: float
    width: float
    height: float
    sill: float
    style: Optional[Literal["swing", "sliding", "open"]] = None
    swing: Optional[Swing] = None


class Floor(BaseModel):
    material: Literal["wood", "tile", "vinyl", "concrete"]
    color: Optional[str] = None
    height: Optional[float] = None


class Room(BaseModel):
    name: str
    polygon: list[Point]
    kind: Optional[RoomKind] = None
    floor: Optional[Floor] = None


class FloorPlan(BaseModel):
    id: int
    name: str
    areaM2: float
    ceilingHeight: float = 2.4
    walls: list[Wall]
    openings: list[Opening]
    rooms: list[Room]
    entrance: Point
    glbUrl: Optional[str] = None

    def dump(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)

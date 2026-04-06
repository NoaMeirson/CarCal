from pydantic import BaseModel
from enum import Enum

class Point(BaseModel):
    x: float
    y: float


class Polygon(BaseModel):
    points: list[Point]

class ImageInfo(BaseModel):
    width: int
    height: int    

class MatchInfo(BaseModel):
    part: str
    score: float
    coveragePercent: float | None = None

class Detection(BaseModel):
    id: str
    type: str
    label: str
    confidence: float
    polygon: Polygon
    matches: list[MatchInfo] | None = None


class ClientAnalyzeRequest(BaseModel):
    requestId: str
    FileName: str | None = None
    imageBase64: str
    mode: str

class EngineAnalyzeRequest(BaseModel):
    requestId: str
    FileName: str | None = None
    imageBase64: str
    mode: str

class ClientAnalyzeResponse(BaseModel):
    requestId: str
    FileName: str | None = None
    status: str
    mode: str
    image: ImageInfo | None = None
    detections: list[Detection]
    message: str | None = None

class EngineAnalyzeResponse(BaseModel):
    requestId: str
    FileName: str | None = None
    status: str
    mode: str
    image: ImageInfo | None = None
    detections: list[Detection]
    message: str | None = None


class HealthResponse(BaseModel):
    status: str   

class EngineHealthResponse(BaseModel):
    status: str
    engineReady: bool
    carPartsModelReady: bool
    damageModelReady: bool
    message: str | None = None    
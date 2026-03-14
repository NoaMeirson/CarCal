from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


class Polygon(BaseModel):
    points: list[Point]


class ClientAnalyzeRequest(BaseModel):
    requestId: str
    imageBase64: str

class Detection(BaseModel):
    id: str
    damageType: str
    part: str
    confidence: float
    polygon: Polygon


class ClientAnalyzeResponse(BaseModel):
    requestId: str
    status: str
    detections: list[Detection]
    message: str | None = None

class EngineAnalyzeResponse(BaseModel):
    requestId: str
    status: str
    detections: list[Detection]
    message: str | None = None


class EngineAnalyzeRequest(BaseModel):
    requestId: str
    imageBase64: str

class HealthResponse(BaseModel):
    status: str   

class EngineHealthResponse(BaseModel):
    status: str
    engineReady: bool
    carPartsModelReady: bool
    message: str | None = None    
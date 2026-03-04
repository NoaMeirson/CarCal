from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ClientAnalyzeRequest(BaseModel):
    requestId: str
    imageBase64: str

class Detection(BaseModel):
    id: str
    damageType: str
    part: str
    confidence: float
    bbox: BoundingBox


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
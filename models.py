from pydantic import BaseModel, Field


class ClientAnalyzeRequest(BaseModel):
    requestId: str
    imageBase64: str


class Detection(BaseModel):
    damageType: str
    confidence: float
    bbox: list[float] = Field(min_length=4, max_length=4)


class ClientAnalyzeResponse(BaseModel):
    requestId: str
    detections: list[Detection]


class EngineAnalyzeResponse(BaseModel):
    detections: list[Detection]


class EngineAnalyzeRequest(BaseModel):
    requestId: str
    imageBase64: str


class HealthResponse(BaseModel):
    status: str

from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


class Polygon(BaseModel):
    points: list[Point]

class ImageInfo(BaseModel):
    width: int
    height: int    


class ClientAnalyzeRequest(BaseModel):
    requestId: str
    FileName: str | None = None
    imageBase64: str

class Detection(BaseModel):
    id: str
    type: str
    label: str
    confidence: float
    polygon: Polygon


class ClientAnalyzeResponse(BaseModel):
    requestId: str
    FileName: str | None = None
    status: str
    image: ImageInfo | None = None
    detections: list[Detection]
    message: str | None = None

class EngineAnalyzeResponse(BaseModel):
    requestId: str
    FileName: str | None = None
    status: str
    image: ImageInfo | None = None
    detections: list[Detection]
    message: str | None = None


class EngineAnalyzeRequest(BaseModel):
    requestId: str
    FileName: str | None = None
    imageBase64: str

class HealthResponse(BaseModel):
    status: str   

class EngineHealthResponse(BaseModel):
    status: str
    engineReady: bool
    carPartsModelReady: bool
    message: str | None = None    
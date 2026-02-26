from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Engine Service", version="1.0.0")


class EngineAnalyzeRequest(BaseModel):
    imageBase64: str


class Detection(BaseModel):
    damageType: str
    confidence: float
    bbox: list[float]


class EngineAnalyzeResponse(BaseModel):
    detections: list[Detection]


@app.post("/models", response_model=EngineAnalyzeResponse)
def models(request: EngineAnalyzeRequest) -> EngineAnalyzeResponse:
    # Mock response that matches the API-Engine contract shape.
    return EngineAnalyzeResponse(
        detections=[
            Detection(
                damageType="scratch",
                confidence=0.91,
                bbox=[120.0, 80.0, 260.0, 210.0],
            )
        ]
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

from fastapi import FastAPI
from models import EngineAnalyzeRequest, EngineAnalyzeResponse, HealthResponse
from .Engine_methods import process

app = FastAPI()

@app.post("/process", response_model=EngineAnalyzeResponse)
def process_endpoint(request: EngineAnalyzeRequest):
    return process(request)


@app.get("/health", response_model=HealthResponse)
def engine_health():
    return HealthResponse(status="ok")

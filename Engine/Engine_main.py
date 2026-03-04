from fastapi import FastAPI
from models import EngineAnalyzeRequest, ClientAnalyzeResponse, HealthResponse
from .Engine_methods import process

app = FastAPI()

@app.post("/process", response_model=ClientAnalyzeResponse)
def process_endpoint(request: EngineAnalyzeRequest):
    return process(request)


@app.get("/health", response_model=HealthResponse)
def engine_health():
    return HealthResponse(status="ok")
from fastapi import FastAPI
from models import ClientAnalyzeRequest, ClientAnalyzeResponse, HealthResponse
from .API_methods import analyze

app = FastAPI()

@app.post("/analyze", response_model=ClientAnalyzeResponse)
def analyze_endpoint(request: ClientAnalyzeRequest):
    return analyze(request)

@app.get("/health", response_model=HealthResponse)
def api_health():
    return HealthResponse(status="ok")
from fastapi import FastAPI, UploadFile, File
from models import ClientAnalyzeRequest, ClientAnalyzeResponse, HealthResponse
from .API_methods import analyze
from .dashboard import services_dashboard


app = FastAPI()

@app.post("/analyze", response_model=ClientAnalyzeResponse)
def analyze_endpoint(request: ClientAnalyzeRequest):
    return analyze(request)

@app.get("/health", response_model=HealthResponse)
def api_health():
    return HealthResponse(status="ok")

@app.get("/services")
def services():
    return services_dashboard() 

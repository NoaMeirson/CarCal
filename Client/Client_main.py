from fastapi import FastAPI
from .Client_methods import handle_request
from models import ClientAnalyzeRequest, ClientAnalyzeResponse, HealthResponse

app = FastAPI()


@app.post("/analyze", response_model=ClientAnalyzeResponse)
def analyze(request: ClientAnalyzeRequest):
    return handle_request(request)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")

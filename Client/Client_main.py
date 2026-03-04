from fastapi import FastAPI, UploadFile, File
from .Client_methods import handle_request
from models import ClientAnalyzeResponse, HealthResponse

app = FastAPI()


@app.post("/analyze", response_model=ClientAnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    return await handle_request(file)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")
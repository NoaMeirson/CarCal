from fastapi import FastAPI
from models import EngineAnalyzeRequest, EngineAnalyzeResponse, EngineHealthResponse
from Engine.Engine_methods import process
from Engine.services.car_parts_model_service import (
    load_car_parts_model,
    is_car_parts_model_ready,
    get_car_parts_model_status,
)

app = FastAPI()


@app.on_event("startup")
def startup_event():
    try:
        load_car_parts_model()
    except Exception as exc:
        print(f"[Engine startup] Failed to load car parts model: {exc}")


@app.post("/process", response_model=EngineAnalyzeResponse)
def process_endpoint(request: EngineAnalyzeRequest):
    return process(request)


@app.get("/health", response_model=EngineHealthResponse)
def engine_health():
    model_ready = is_car_parts_model_ready()
    model_status = get_car_parts_model_status()

    return EngineHealthResponse(
        status="ok" if model_ready else "degraded",
        engineReady=True,
        carPartsModelReady=model_ready,
        message=None if model_ready else f"Car parts model is not ready: {model_status['error']}",
    )
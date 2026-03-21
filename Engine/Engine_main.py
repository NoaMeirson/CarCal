from fastapi import FastAPI
from models import EngineAnalyzeRequest, EngineAnalyzeResponse, EngineHealthResponse
from Engine.Engine_methods import process
from Engine.services.car_parts_model_service import (
    load_car_parts_model,
    is_car_parts_model_ready,
    get_car_parts_model_status,
)

from Engine.services.damage_model_service import (
    load_damage_model,
    is_damage_model_ready,
    get_damage_model_status,
)

app = FastAPI()


@app.on_event("startup")
def startup_event():
    try:
        load_car_parts_model()
    except Exception as exc:
        print(f"[Engine startup] Failed to load car parts model: {exc}")

    try:
        load_damage_model()
    except Exception as exc:
        print(f"[Engine startup] Failed to load damage model: {exc}")
    


@app.post("/process", response_model=EngineAnalyzeResponse)
def process_endpoint(request: EngineAnalyzeRequest):
    return process(request)


@app.get("/health", response_model=EngineHealthResponse)
def engine_health():
    car_parts_ready = is_car_parts_model_ready()
    car_parts_status = get_car_parts_model_status()

    damage_ready = is_damage_model_ready()
    damage_status = get_damage_model_status()

    all_ready = car_parts_ready and damage_ready

    message = None
    if not all_ready:
        message = (
            f"car_parts_error={car_parts_status['error']}; "
            f"damage_error={damage_status['error']}"
        )

    return EngineHealthResponse(
        status="ok" if all_ready else "degraded",
        engineReady=True,
        carPartsModelReady=car_parts_ready,
        damageModelReady= damage_ready,
        message=message,
    )


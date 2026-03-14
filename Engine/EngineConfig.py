from pathlib import Path

MODEL_INPUT_SIZE = 640

DEVICE = "cpu" 

MAX_DETECTIONS = 20

#configs for car_parts_model_service.py:

ENGINE_BASE_DIR = Path(__file__).resolve().parent

CAR_PARTS_MODEL_DIR = ENGINE_BASE_DIR / "Artifacts" / "Car_parts" / "car_parts_M2F_model"

CAR_PARTS_MODEL_PREFERRED_DEVICE = "cuda"
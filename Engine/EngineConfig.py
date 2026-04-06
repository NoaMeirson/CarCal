from pathlib import Path

DEVICE = "cpu" 

MAX_DETECTIONS = 20


#configs for car_parts_model_service.py:

ENGINE_BASE_DIR = Path(__file__).resolve().parent

CAR_PARTS_MODEL_DIR = ENGINE_BASE_DIR / "artifacts" / "car_parts" / "car_parts_M2F_model"

CAR_PARTS_MODEL_PREFERRED_DEVICE = "cuda"

# damage model
DAMAGE_MODEL_PATH = ENGINE_BASE_DIR / "artifacts" / "damages" / "damage_YOLO_model.pt"

DAMAGE_MODEL_PREFERRED_DEVICE = "cuda"

DAMAGE_MODEL_CONFIDENCE = 0.25
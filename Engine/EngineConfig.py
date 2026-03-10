from pathlib import Path

MODEL_INPUT_SIZE = 640
MODEL_PATH = Path(__file__).with_name("best_vehicle_damage_seg.pt")
DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.25
MAX_DETECTIONS = 20

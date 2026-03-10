import base64
import os
from io import BytesIO

import numpy as np
from PIL import Image

from models import Detection, EngineAnalyzeRequest, EngineAnalyzeResponse
from .EngineConfig import CONFIDENCE_THRESHOLD, DEVICE, MAX_DETECTIONS, MODEL_INPUT_SIZE, MODEL_PATH

ULTRALYTICS_DIR = os.path.join(os.path.dirname(__file__), ".ultralytics")
os.makedirs(ULTRALYTICS_DIR, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", ULTRALYTICS_DIR)

from ultralytics import YOLO

_MODEL = None

def process(request: EngineAnalyzeRequest):
    image_bytes = base64.b64decode(request.imageBase64, validate=True)
    image = decode_image(image_bytes)
    detections = run_damage_model(image)

    return EngineAnalyzeResponse(detections=detections)


def decode_image(image_bytes: bytes):
    with Image.open(BytesIO(image_bytes)) as image:
        return np.array(image.convert("RGB"))


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(str(MODEL_PATH))
    return _MODEL


def run_damage_model(image):
    results = get_model().predict(
        image,
        imgsz=MODEL_INPUT_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        max_det=MAX_DETECTIONS,
        verbose=False,
    )
    return combine_results(results)


def combine_results(results):
    detections = []
    model = get_model()

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()

        for bbox, confidence, class_id in zip(boxes, confidences, class_ids):
            detections.append(
                Detection(
                    damageType=model.names[int(class_id)],
                    confidence=float(confidence),
                    bbox=[float(value) for value in bbox],
                )
            )

    return detections

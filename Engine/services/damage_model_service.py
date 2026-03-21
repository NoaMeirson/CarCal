from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO

from Engine.EngineConfig import (
    DAMAGE_MODEL_PATH,
    DAMAGE_MODEL_PREFERRED_DEVICE,
)

from models import Detection
from Engine.utils.segmentation_utils import mask_to_polygon

_MODEL = None
_DEVICE = None
_MODEL_READY = False
_MODEL_LOAD_ERROR = None


def load_damage_model() -> None:
    global _MODEL, _DEVICE, _MODEL_READY, _MODEL_LOAD_ERROR

    if _MODEL_READY and _MODEL is not None:
        return

    try:
        if not DAMAGE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Damage model file not found: {DAMAGE_MODEL_PATH}")

        if DAMAGE_MODEL_PREFERRED_DEVICE == "cuda" and torch.cuda.is_available():
            _DEVICE = "cuda"
        else:
            _DEVICE = "cpu"

        _MODEL = YOLO(str(DAMAGE_MODEL_PATH))
        _MODEL_READY = True
        _MODEL_LOAD_ERROR = None

    except Exception as exc:
        _MODEL = None
        _DEVICE = None
        _MODEL_READY = False
        _MODEL_LOAD_ERROR = str(exc)
        raise


def is_damage_model_ready() -> bool:
    return _MODEL_READY and _MODEL is not None


def get_damage_model_status() -> dict[str, str | bool | None]:
    return {
        "ready": is_damage_model_ready(),
        "device": _DEVICE,
        "modelPath": str(DAMAGE_MODEL_PATH),
        "error": _MODEL_LOAD_ERROR,
    }


def run_damage_model(image: Image.Image):
    """
    Runs YOLO segmentation inference and returns raw ultralytics results.
    """
    if not is_damage_model_ready():
        raise RuntimeError("Damage model is not loaded. Call load_damage_model() first.")

    results = _MODEL.predict(
        source=image,
        verbose=False,
        retina_masks=True,
    )

    return results


def postprocess_damage_raw_outputs(raw_outputs, image: Image.Image) -> dict:
    """
    Keeps raw ultralytics result in a uniform wrapper.
    """
    if not raw_outputs:
        return {"results": []}

    return {"results": raw_outputs}


def get_damage_id2label() -> dict[int, str]:
    if not is_damage_model_ready():
        raise RuntimeError("Damage model is not loaded.")

    names = getattr(_MODEL.model, "names", {})
    normalized = {}

    for key, value in names.items():
        normalized[int(key)] = str(value)

    return normalized


def build_damage_detections(postprocessed_results) -> list[Detection]:
    detections: list[Detection] = []

    if not postprocessed_results:
        return detections

    results = postprocessed_results.get("results", [])
    if not results:
        return detections

    result = results[0]

    if result.masks is None or result.boxes is None:
        return detections

    masks_data = result.masks.data
    boxes = result.boxes
    id2label = get_damage_id2label()

    if hasattr(masks_data, "cpu"):
        masks_data = masks_data.cpu().numpy()

    cls_list = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
    conf_list = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf

    for i in range(len(masks_data)):
        mask = masks_data[i] > 0.5
        polygon = mask_to_polygon(mask.astype(np.uint8))

        if polygon is None:
            continue

        class_id = int(cls_list[i])
        confidence = float(conf_list[i])
        label = id2label.get(class_id, f"class_{class_id}")

        detections.append(
            Detection(
                id=str(i),
                type="damage",
                label=label,
                confidence=confidence,
                polygon=polygon,
                matches=None,
            )
        )

    return detections
from __future__ import annotations

import torch
from PIL import Image

from .EngineConfig import (
    CAR_PARTS_MODEL_DIR,
    CAR_PARTS_MODEL_PREFERRED_DEVICE,
)

_MODEL = None
_PROCESSOR = None
_DEVICE = None
_MODEL_READY = False
_MODEL_LOAD_ERROR = None


def load_car_parts_model() -> None:
    global _MODEL, _PROCESSOR, _DEVICE, _MODEL_READY, _MODEL_LOAD_ERROR

    if _MODEL_READY and _MODEL is not None and _PROCESSOR is not None:
        return

    try:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        if not CAR_PARTS_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Car parts model directory not found: {CAR_PARTS_MODEL_DIR}"
            )

        if CAR_PARTS_MODEL_PREFERRED_DEVICE == "cuda" and torch.cuda.is_available():
            _DEVICE = "cuda"
        else:
            _DEVICE = "cpu"

        _PROCESSOR = AutoImageProcessor.from_pretrained(str(CAR_PARTS_MODEL_DIR))
        _MODEL = Mask2FormerForUniversalSegmentation.from_pretrained(
            str(CAR_PARTS_MODEL_DIR)
        )

        _MODEL.to(_DEVICE)
        _MODEL.eval()

        _MODEL_READY = True
        _MODEL_LOAD_ERROR = None

    except Exception as exc:
        _MODEL = None
        _PROCESSOR = None
        _DEVICE = None
        _MODEL_READY = False
        _MODEL_LOAD_ERROR = str(exc)
        raise


def is_car_parts_model_ready() -> bool:
    return _MODEL_READY and _MODEL is not None and _PROCESSOR is not None


def get_car_parts_model_status() -> dict[str, str | bool | None]:
    return {
        "ready": is_car_parts_model_ready(),
        "device": _DEVICE,
        "modelDir": str(CAR_PARTS_MODEL_DIR),
        "error": _MODEL_LOAD_ERROR,
    }


def segment_car_parts(image: Image.Image):
    if not is_car_parts_model_ready():
        raise RuntimeError(
            "Car parts model is not loaded. Call load_car_parts_model() first."
        )

    inputs = _PROCESSOR(images=image, return_tensors="pt")
    inputs = {key: value.to(_DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = _MODEL(**inputs)

    return outputs
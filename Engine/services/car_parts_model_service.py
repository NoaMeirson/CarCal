from __future__ import annotations

import torch
from PIL import Image

from Engine.EngineConfig import (
    CAR_PARTS_MODEL_DIR,
    CAR_PARTS_MODEL_PREFERRED_DEVICE,
)

from models import Detection
from Engine.utils.segmentation_utils import mask_to_polygon

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


def run_car_parts_model(image: Image.Image):
    """
    Returns raw model outputs.
    """
    if not is_car_parts_model_ready():
        raise RuntimeError(
            "Car parts model is not loaded. Call load_car_parts_model() first."
        )

    inputs = _PROCESSOR(images=image, return_tensors="pt")
    inputs = {key: value.to(_DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = _MODEL(**inputs)

    return outputs


def postprocess_car_parts_raw_outputs(raw_outputs, image: Image.Image) -> dict:
    """
    Converts raw Mask2Former outputs into instance segmentation result.
    """
    if not is_car_parts_model_ready():
        raise RuntimeError("Car parts model is not loaded.")

    target_sizes = [(image.height, image.width)]

    results = _PROCESSOR.post_process_instance_segmentation(
        raw_outputs,
        target_sizes=target_sizes,
    )

    if not results:
        return {"segmentation": None, "segments_info": []}

    return results[0]


def get_car_parts_id2label() -> dict[int, str]:
    if not is_car_parts_model_ready():
        raise RuntimeError("Car parts model is not loaded.")

    id2label = getattr(_MODEL.config, "id2label", None)
    if not id2label:
        return {}

    normalized = {}
    for key, value in id2label.items():
        normalized[int(key)] = str(value)

    return normalized

def convert_car_parts_result_to_detections(postprocessed_results) -> list[Detection]:
    detections: list[Detection] = []

    if not postprocessed_results:
        return detections

    segmentation_map = postprocessed_results.get("segmentation")
    segments_info = postprocessed_results.get("segments_info", [])

    if segmentation_map is None or not segments_info:
        return detections

    if hasattr(segmentation_map, "cpu"):
        segmentation_map = segmentation_map.cpu().numpy()

    id2label = get_car_parts_id2label()

    for segment in segments_info:
        segment_id = int(segment["id"])
        label_id = int(segment["label_id"])
        score = float(segment.get("score", 0.0))

        mask = segmentation_map == segment_id
        polygon = mask_to_polygon(mask)

        if polygon is None:
            continue

        part_name = id2label.get(label_id, f"class_{label_id}")

        detections.append(
            Detection(
                id=str(segment_id),
                type="part",
                label=part_name,
                confidence=score,
                polygon=polygon,
            )
        )

    return detections
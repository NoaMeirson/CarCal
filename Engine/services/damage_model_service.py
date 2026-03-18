from __future__ import annotations

from PIL import Image

_MODEL_READY = False
_MODEL_LOAD_ERROR = "Damage model is not implemented yet."


def load_damage_model() -> None:
    """
    Placeholder for future damage model loading.
    """
    pass


def is_damage_model_ready() -> bool:
    return _MODEL_READY


def get_damage_model_status() -> dict[str, str | bool | None]:
    return {
        "ready": is_damage_model_ready(),
        "device": None,
        "modelDir": None,
        "error": _MODEL_LOAD_ERROR,
    }


def run_damage_model(image: Image.Image):
    """
    Placeholder for future raw inference of the damage model.
    """
    return None


def postprocess_damage_raw_outputs(raw_outputs, image: Image.Image) -> dict:
    """
    Placeholder for future post-processing of damage model outputs.
    """
    return {"segmentation": None, "segments_info": []}


def get_damage_id2label() -> dict[int, str]:
    return {}

def build_damage_detections(postprocessed_results):
    """
    Placeholder for future raw inference of the damage model.
    """
    return None
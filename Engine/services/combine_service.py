import numpy as np

from models import Detection
from Engine.services.car_parts_model_service import get_car_parts_id2label
from Engine.utils.segmentation_utils import mask_to_polygon


def combine_results(damage_result, car_parts_result) -> list[Detection]:
    """
    Temporary behavior:
    Ignore damage model result and return only car-parts detections.
    """
    return convert_car_parts_result_to_detections(car_parts_result)

def convert_car_parts_result_to_detections(segmentation_result) -> list[Detection]:
    detections: list[Detection] = []

    if not segmentation_result:
        return detections

    segmentation_map = segmentation_result.get("segmentation")
    segments_info = segmentation_result.get("segments_info", [])

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
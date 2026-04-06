from __future__ import annotations

import numpy as np

from models import Detection, MatchInfo
from Engine.services.car_parts_model_service import get_car_parts_id2label
from Engine.services.damage_model_service import get_damage_id2label
from Engine.utils.segmentation_utils import mask_to_polygon


def combine_results(damage_result, car_parts_result) -> list[Detection]:
    """
    Combines damage detections with car-part detections by polygon/mask overlap.

    Output:
    - One Detection per damage
    - Each damage includes matches[] with overlapping car parts

    For every match:
    - score = intersection / damage_area
      -> how much of the damage lies on this part
    - coveragePercent = intersection / part_area * 100
      -> how much of the car part is covered by this damage
    """
    damage_segments = _extract_damage_segments(damage_result)
    car_part_segments = _extract_car_part_segments(car_parts_result)

    combined_detections: list[Detection] = []

    if not damage_segments:
        return combined_detections

    for damage in damage_segments:
        damage_mask = damage["mask"]
        damage_area = int(damage_mask.sum())

        if damage_area == 0:
            continue

        polygon = mask_to_polygon(damage_mask.astype(np.uint8))
        if polygon is None:
            continue

        matches: list[MatchInfo] = []

        for part in car_part_segments:
            part_mask = part["mask"]
            part_area = int(part_mask.sum())

            if part_area == 0:
                continue

            intersection = int(np.logical_and(damage_mask, part_mask).sum())
            if intersection == 0:
                continue

            overlap_ratio_in_damage = intersection / damage_area
            coverage_percent_in_part = (intersection / part_area) * 100.0

            matches.append(
                MatchInfo(
                    part=part["label"],
                    score=round(overlap_ratio_in_damage, 4),
                    coveragePercent=round(coverage_percent_in_part, 2),
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)

        combined_detections.append(
            Detection(
                id=str(damage["id"]),
                type="damage",
                label=damage["label"],
                confidence=damage["confidence"],
                polygon=polygon,
                matches=matches or None,
            )
        )

    return combined_detections


def _extract_damage_segments(damage_result) -> list[dict]:
    """
    Extract damage masks from YOLO segmentation output wrapped as:
    {
        "results": [ultralytics_result]
    }
    """
    extracted_segments: list[dict] = []

    if not damage_result:
        return extracted_segments

    results = damage_result.get("results", [])
    if not results:
        return extracted_segments

    result = results[0]

    if result.masks is None or result.boxes is None:
        return extracted_segments

    masks_data = result.masks.data
    boxes = result.boxes
    id2label = get_damage_id2label()

    if hasattr(masks_data, "cpu"):
        masks_data = masks_data.cpu().numpy()

    cls_list = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
    conf_list = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf

    for i in range(len(masks_data)):
        mask = masks_data[i] > 0.5

        if not mask.any():
            continue

        class_id = int(cls_list[i])
        confidence = float(conf_list[i])
        label = id2label.get(class_id, f"class_{class_id}")

        extracted_segments.append(
            {
                "id": i,
                "label_id": class_id,
                "label": label,
                "confidence": confidence,
                "mask": mask,
            }
        )

    return extracted_segments


def _extract_car_part_segments(car_parts_result) -> list[dict]:
    """
    Extract car-part masks from Mask2Former postprocessed output:
    {
        "segmentation": ...,
        "segments_info": [...]
    }
    """
    extracted_segments: list[dict] = []

    if not car_parts_result:
        return extracted_segments

    segmentation_map = car_parts_result.get("segmentation")
    segments_info = car_parts_result.get("segments_info", [])

    if segmentation_map is None or not segments_info:
        return extracted_segments

    if hasattr(segmentation_map, "cpu"):
        segmentation_map = segmentation_map.cpu().numpy()

    segmentation_map = np.asarray(segmentation_map)
    id2label = get_car_parts_id2label()

    for segment in segments_info:
        segment_id = int(segment["id"])
        label_id = int(segment["label_id"])
        confidence = float(segment.get("score", 0.0))

        mask = segmentation_map == segment_id
        if not mask.any():
            continue

        label = id2label.get(label_id, f"class_{label_id}")

        extracted_segments.append(
            {
                "id": segment_id,
                "label_id": label_id,
                "label": label,
                "confidence": confidence,
                "mask": mask,
            }
        )

    return extracted_segments
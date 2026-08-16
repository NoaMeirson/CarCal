"""Converts OpenAI's structured-output damage detections into CarCal's
existing Detection / EngineAnalyzeResponse schema (root models.py), so the
result of this benchmark can be scored with the same evaluation code as the
local YOLO damage model's damage_only output.

Imports the real Pydantic models from root models.py (not a reimplementation
of the shape) to guarantee exact structural parity with production.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import Detection, EngineAnalyzeResponse, ImageInfo, Point, Polygon  # noqa: E402

from schema import DAMAGE_CLASSES  # noqa: E402


def clamp01(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return min(1.0, max(0.0, value))


def clamp_and_validate_polygon(
    raw_points, width: int, height: int
) -> list[tuple[float, float]] | None:
    """Clamps normalized (x, y) points into [0, 1], converts to pixel space
    using the real image dimensions, drops malformed points, dedupes
    consecutive duplicates, and rejects the polygon outright if fewer than
    3 valid points remain -- mirrors the >=3-point rule in
    Engine/utils/segmentation_utils.mask_to_polygon.

    Pixel coordinates are clamped to valid image indices, i.e.
    0 <= x <= width - 1 and 0 <= y <= height - 1 (normalized 1.0 must map to
    the last valid pixel index, not to `width`/`height`, which would be one
    past the end of the image).
    """
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)

    pixel_points: list[tuple[float, float]] = []
    for pt in raw_points or []:
        if not isinstance(pt, dict):
            continue
        x = clamp01(pt.get("x"))
        y = clamp01(pt.get("y"))
        if x is None or y is None:
            continue
        px = min(max_x, max(0.0, round(x * max_x, 2)))
        py = min(max_y, max(0.0, round(y * max_y, 2)))
        pixel_points.append((px, py))

    deduped: list[tuple[float, float]] = []
    for p in pixel_points:
        if not deduped or deduped[-1] != p:
            deduped.append(p)
    if len(deduped) >= 2 and deduped[0] == deduped[-1]:
        deduped.pop()

    if len(deduped) < 3:
        return None
    return deduped


def build_detections(raw_detections, width: int, height: int, warn) -> list[Detection]:
    """warn: callable(str) -> None, used to log dropped detections."""
    detections: list[Detection] = []
    for i, raw_det in enumerate(raw_detections or []):
        label = raw_det.get("label") if isinstance(raw_det, dict) else None
        if label not in DAMAGE_CLASSES:
            warn(f"dropped detection {i}: unknown label {label!r}")
            continue

        points = clamp_and_validate_polygon(raw_det.get("polygon"), width, height)
        if points is None:
            warn(f"dropped detection {i} ({label}): fewer than 3 valid polygon points")
            continue

        confidence = clamp01(raw_det.get("confidence"))
        if confidence is None:
            confidence = 0.0

        # IDs are assigned locally (index-based), matching
        # Engine/services/damage_model_service.build_damage_detections --
        # OpenAI is never asked to supply an id.
        detections.append(
            Detection(
                id=str(i),
                type="damage",
                label=label,
                confidence=confidence,
                polygon=Polygon(points=[Point(x=x, y=y) for x, y in points]),
                matches=None,
            )
        )
    return detections


def build_response(
    request_id: str,
    file_name: str,
    width: int,
    height: int,
    raw_detections,
    warn,
) -> EngineAnalyzeResponse:
    detections = build_detections(raw_detections, width, height, warn)
    return EngineAnalyzeResponse(
        requestId=request_id,
        FileName=file_name,
        status="ok",
        mode="damage_only",
        image=ImageInfo(width=width, height=height),
        detections=detections,
        message=None,
    )


def build_error_response(request_id: str, file_name: str, message: str) -> EngineAnalyzeResponse:
    return EngineAnalyzeResponse(
        requestId=request_id,
        FileName=file_name,
        status="error",
        mode="damage_only",
        image=None,
        detections=[],
        message=message,
    )

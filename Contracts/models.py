from __future__ import annotations

"""
Shared, protocol-agnostic contracts for the Vehicle Damage Analysis system.

- REST friendly (JSON)
- gRPC friendly (easy mapping to protobuf)
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, model_validator


# ---------------------------
# Errors
# ---------------------------

class ErrorCode(str, Enum):
    UNKNOWN = "UNKNOWN"
    INVALID_REQUEST = "INVALID_REQUEST"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"

    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_RESIZE_FAILED = "IMAGE_RESIZE_FAILED"

    ENGINE_UNAVAILABLE = "ENGINE_UNAVAILABLE"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    TIMEOUT = "TIMEOUT"

    INTERNAL_ERROR = "INTERNAL_ERROR"


class ApiError(BaseModel):
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    request_id: str
    error: ApiError


# ---------------------------
# Images / preprocessing
# ---------------------------

class ImageEncoding(str, Enum):
    BASE64 = "base64"
    URL = "url"


class ImageRef(BaseModel):
    image_id: str = Field(..., description="Client-generated id unique within the request.")
    filename: Optional[str] = None
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    encoding: ImageEncoding
    data_base64: Optional[str] = Field(default=None, description="Base64 bytes (no data: prefix).")
    url: Optional[HttpUrl] = None
    sha256: Optional[str] = Field(default=None, description="Hex sha256 of raw bytes (optional).")
    width: Optional[int] = Field(default=None, ge=1)
    height: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _one_of(self) -> "ImageRef":
        if self.encoding == ImageEncoding.BASE64 and not self.data_base64:
            raise ValueError("encoding=base64 requires data_base64")
        if self.encoding == ImageEncoding.URL and not self.url:
            raise ValueError("encoding=url requires url")
        return self


class PreprocessConfig(BaseModel):
    target_format: Literal["jpeg", "png", "webp"] = "jpeg"
    resize_longest_side: int = Field(1280, ge=64, le=4096)
    keep_aspect_ratio: bool = True
    max_pixels: int = Field(8_000_000, ge=100_000)
    normalize: bool = False


# ---------------------------
# Domain enums
# ---------------------------

class DamageClass(str, Enum):
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    BROKEN = "broken"
    RUST = "rust"
    PAINT_DAMAGE = "paint_damage"
    GLASS_DAMAGE = "glass_damage"


class VehiclePart(str, Enum):
    FRONT_BUMPER = "front_bumper"
    REAR_BUMPER = "rear_bumper"
    HOOD = "hood"
    TRUNK = "trunk"
    FRONT_LEFT_DOOR = "front_left_door"
    FRONT_RIGHT_DOOR = "front_right_door"
    REAR_LEFT_DOOR = "rear_left_door"
    REAR_RIGHT_DOOR = "rear_right_door"
    LEFT_FENDER = "left_fender"
    RIGHT_FENDER = "right_fender"
    LEFT_MIRROR = "left_mirror"
    RIGHT_MIRROR = "right_mirror"
    WINDSHIELD = "windshield"
    ROOF = "roof"


class CoordinateFrame(str, Enum):
    PIXEL = "pixel"
    NORMALIZED = "normalized"


# ---------------------------
# Geometry / masks
# ---------------------------

class Point(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    frame: CoordinateFrame = CoordinateFrame.PIXEL

    @model_validator(mode="after")
    def _check(self) -> "BoundingBox":
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("Invalid bbox: max must be >= min")
        return self


class RLEMask(BaseModel):
    size: List[int] = Field(..., min_length=2, max_length=2, description="[height, width]")
    counts: Union[List[int], str]


class Polygon(BaseModel):
    points: List[Point] = Field(..., min_length=3)


class SegmentationMask(BaseModel):
    polygon: Optional[Polygon] = None
    rle: Optional[RLEMask] = None

    @model_validator(mode="after")
    def _one_of(self) -> "SegmentationMask":
        if (self.polygon is None) == (self.rle is None):
            raise ValueError("Provide exactly one of: polygon or rle")
        return self


# ---------------------------
# Raw model outputs
# ---------------------------

class YoloDetection(BaseModel):
    detection_id: str
    damage_class: DamageClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    mask: Optional[SegmentationMask] = None


class PartSegmentation(BaseModel):
    part_id: str
    vehicle_part: VehiclePart
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mask: SegmentationMask


# ---------------------------
# Fusion outputs
# ---------------------------

class DamagePartAssociation(BaseModel):
    detection_id: str
    part_id: Optional[str] = None
    iou: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    strategy: Literal["iou_max", "none"] = "iou_max"


# ---------------------------
# Public API (Client <-> API)
# ---------------------------

class AnalyzeRequest(BaseModel):
    request_id: str
    images: List[ImageRef] = Field(..., min_length=1)
    preprocess: Optional[PreprocessConfig] = None
    return_visual_overlays: bool = True
    client_meta: Optional[Dict[str, Any]] = None


class OverlayAsset(BaseModel):
    image_id: str
    kind: Literal["segmentation_overlay", "detection_overlay", "fusion_overlay"] = "fusion_overlay"
    mime_type: Literal["image/png"] = "image/png"
    data_base64: str


class ImageResult(BaseModel):
    image_id: str
    width: int
    height: int
    detections: List[YoloDetection]
    parts: List[PartSegmentation]
    associations: List[DamagePartAssociation]


class AnalyzeResponse(BaseModel):
    request_id: str
    analysis_id: str
    status: Literal["ok"] = "ok"
    results: List[ImageResult]
    overlays: Optional[List[OverlayAsset]] = None
    warnings: Optional[List[str]] = None


# ---------------------------
# Internal (API <-> Engine)
# ---------------------------

class EngineAnalyzeRequest(BaseModel):
    request_id: str
    images: List[ImageRef] = Field(..., min_length=1)
    preprocess_applied: PreprocessConfig
    model_versions: Optional[Dict[str, str]] = None
    iou_threshold: float = Field(0.1, ge=0.0, le=1.0)


class EngineAnalyzeResponse(BaseModel):
    request_id: str
    status: Literal["ok"] = "ok"
    results: List[ImageResult]
    model_info: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None


# ---------------------------
# Health
# ---------------------------

class HealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class HealthResponse(BaseModel):
    status: HealthStatus
    service: Literal["client", "api", "engine"]
    version: str
    time_utc: str
    details: Optional[Dict[str, Any]] = None

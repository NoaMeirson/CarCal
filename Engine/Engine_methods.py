import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from models import EngineAnalyzeRequest, EngineAnalyzeResponse
from .EngineConfig import MODEL_INPUT_SIZE
from .car_parts_model_service import segment_car_parts

def process(request: EngineAnalyzeRequest):

    image_bytes = base64.b64decode(request.imageBase64)

    image = decode_image(image_bytes)

    resized_image = resize_image(image)

    yolo_result = run_yolo(resized_image)

    segmentation_result = run_mask2former(resized_image)

    detections = combine_results(yolo_result, segmentation_result)

    return EngineAnalyzeResponse(
        requestId=request.requestId,
        status="ok",
        detections=detections,
        message=None
    )


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image data provided.") from exc


def resize_image(image):
    return image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))


def run_yolo(image):
    return []


def run_mask2former(image):
    return segment_car_parts(image)


def combine_results(yolo_result, segmentation_result):
    return []
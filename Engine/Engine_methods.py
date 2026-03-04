import base64
from models import EngineAnalyzeRequest, EngineAnalyzeResponse
from .EngineConfig import MODEL_INPUT_SIZE

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


def decode_image(image_bytes: bytes):
    return image_bytes


def resize_image(image):
    return image


def run_yolo(image):
    return []


def run_mask2former(image):
    return []


def combine_results(yolo_result, segmentation_result):
    return []
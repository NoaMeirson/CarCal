import base64
import requests
from PIL import Image
import io
from .APIConfig import ENGINE_URL, REQUEST_TIMEOUT, MAX_IMAGE_SIZE, IMAGE_SIZE
from models import ClientAnalyzeRequest, ClientAnalyzeResponse, EngineAnalyzeRequest


def analyze(request: ClientAnalyzeRequest):

    try:
        image_bytes = base64.b64decode(request.imageBase64)
    except Exception:
        raise ValueError("Invalid base64 image")
    
    validate_image(image_bytes)

    engine_result = send_to_engine(image_bytes, request.requestId)

    return ClientAnalyzeResponse(
        requestId=request.requestId,
        status="ok",
        detections=engine_result["detections"],
        message=None
    )


def validate_image(image_bytes: bytes):

    if len(image_bytes) == 0:
        raise ValueError("Empty image")
    
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError("Image too large")

    with Image.open(io.BytesIO(image_bytes)) as image:
     image.verify()


def send_to_engine(image_bytes: bytes, request_id: str):

    image_base64 = base64.b64encode(image_bytes).decode()

    engine_request = EngineAnalyzeRequest(
        requestId=request_id,
        imageBase64=image_base64
    )

    response = requests.post(
        ENGINE_URL,
        json=engine_request.model_dump(),
        timeout=REQUEST_TIMEOUT
    )

    if response.status_code != 200:
        raise RuntimeError("Engine service error")

    return response.json()
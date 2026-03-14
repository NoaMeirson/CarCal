import base64
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from Engine.EngineConfig import MODEL_INPUT_SIZE


def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_base64)
        return decode_image_bytes(image_bytes)
    except Exception as exc:
        raise ValueError("Failed to decode base64 image.") from exc


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image data provided.") from exc


def resize_image(image: Image.Image, size: int = MODEL_INPUT_SIZE) -> Image.Image:
    return image.resize((size, size))
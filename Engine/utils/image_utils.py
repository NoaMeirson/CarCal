import base64
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError



def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_base64)
        return decode_image_bytes(image_bytes)
    except Exception as exc:
        raise ValueError("Failed to decode base64 image.") from exc


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        return ImageOps.exif_transpose(image).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image data provided.") from exc

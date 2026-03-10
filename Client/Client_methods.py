import base64
import requests

from .ClientConfig import API_URL, REQUEST_TIMEOUT
from models import ClientAnalyzeRequest, ClientAnalyzeResponse


def handle_request(request: ClientAnalyzeRequest):
    try:
        base64.b64decode(request.imageBase64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image") from exc

    response = requests.post(
        API_URL,
        json=request.model_dump(),
        timeout=REQUEST_TIMEOUT
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    return ClientAnalyzeResponse(**response.json())

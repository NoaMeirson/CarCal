import base64
import requests

from .ClientConfig import API_URL, REQUEST_TIMEOUT
from models import ClientAnalyzeRequest, ClientAnalyzeResponse


async def handle_request(file):

    image_bytes = await file.read()

    image_base64 = base64.b64encode(image_bytes).decode()

    request = ClientAnalyzeRequest(
        requestId="client_request",
        imageBase64=image_base64
    )

    response = requests.post(
        API_URL,
        json=request.model_dump(),
        timeout=REQUEST_TIMEOUT
    )

    if response.status_code != 200:
     raise RuntimeError(response.text)

    return ClientAnalyzeResponse(**response.json())
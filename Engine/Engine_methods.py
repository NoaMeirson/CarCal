from models import EngineAnalyzeRequest, EngineAnalyzeResponse
from Engine.utils.image_utils import decode_base64_image, resize_image
from Engine.services.car_parts_model_service import run_car_parts_model
from Engine.services.damage_model_service import run_damage_model
from Engine.services.combine_service import combine_results


def process(request: EngineAnalyzeRequest) -> EngineAnalyzeResponse:
    try:
        image = decode_base64_image(request.imageBase64)
        resized_image = resize_image(image)

        damage_raw_result = run_damage_model(resized_image)
        car_parts_raw_result = run_car_parts_model(resized_image)

        detections = combine_results(
            damage_raw_result=damage_raw_result,
            car_parts_raw_result=car_parts_raw_result,
            image=resized_image,
        )

        return EngineAnalyzeResponse(
            requestId=request.requestId,
            status="ok",
            detections=detections,
            message=None
        )

    except Exception as exc:
        return EngineAnalyzeResponse(
            requestId=request.requestId,
            status="error",
            detections=[],
            message=str(exc)
        )
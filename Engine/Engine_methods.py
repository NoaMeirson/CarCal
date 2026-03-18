from models import EngineAnalyzeRequest, EngineAnalyzeResponse
from Engine.utils.image_utils import decode_base64_image
from Engine.services.car_parts_model_service import (
    run_car_parts_model,
    postprocess_car_parts_raw_outputs,
)
from Engine.services.damage_model_service import run_damage_model, postprocess_damage_raw_outputs
from Engine.services.combine_service import combine_results


def process(request: EngineAnalyzeRequest) -> EngineAnalyzeResponse:
    try:
        image = decode_base64_image(request.imageBase64)

        damage_raw_result = run_damage_model(image)
        car_parts_raw_result = run_car_parts_model(image)

        damage_result = postprocess_damage_raw_outputs(damage_raw_result, image)
        car_parts_result = postprocess_car_parts_raw_outputs(car_parts_raw_result, image)

        detections = combine_results(
            damage_result=damage_result,
            car_parts_result=car_parts_result,
        )

        return EngineAnalyzeResponse(
            requestId=request.requestId,
            FileName=request.FileName,
            status="ok",
            image={
                "width": image.width,
                "height": image.height,
            },
            detections=detections,
            message=None,
        )

    except Exception as exc:
        return EngineAnalyzeResponse(
            requestId=request.requestId,
            FileName=request.FileName,
            status="error",
            image=None,
            detections=[],
            message=str(exc)
        )
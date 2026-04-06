from models import EngineAnalyzeRequest, EngineAnalyzeResponse, ImageInfo
from Engine.utils.image_utils import decode_base64_image
from Engine.services.car_parts_model_service import (
    run_car_parts_model,
    postprocess_car_parts_raw_outputs,
    convert_car_parts_result_to_detections,
)
from Engine.services.damage_model_service import run_damage_model, postprocess_damage_raw_outputs, build_damage_detections
from Engine.services.combine_service import combine_results


def process(request: EngineAnalyzeRequest) -> EngineAnalyzeResponse:
    try:
        image = decode_base64_image(request.imageBase64)

        if request.mode == "car_parts_only":
            return process_parts_only(request, image)

        if request.mode == "damage_only":
            return process_damage_only(request, image)

        if request.mode == "combined":
            return process_combined(request, image)

        raise ValueError(f"Unsupported mode: {request.mode}")

        
    except Exception as exc:
        return EngineAnalyzeResponse(
            requestId=request.requestId,
            FileName=request.FileName,
            status="error",
            image=None,
            detections=[],
            message=str(exc)
        )
    

def process_parts_only(request, image):
        raw = run_car_parts_model(image)
        processed = postprocess_car_parts_raw_outputs(raw, image)
        detections = convert_car_parts_result_to_detections(processed)

        return EngineAnalyzeResponse(
            requestId=request.requestId,
            FileName=request.FileName,
            status="ok",
            mode=request.mode,
            image=ImageInfo(width=image.width, height=image.height),
            detections=detections,
            message=None    
        )
def process_damage_only(request, image):
        raw = run_damage_model(image)
        processed = postprocess_damage_raw_outputs(raw, image)
        detections = build_damage_detections(processed)

        return EngineAnalyzeResponse(
            requestId=request.requestId,
            FileName=request.FileName,
            status="ok",
            mode=request.mode,
            image=ImageInfo(width=image.width, height=image.height),
            detections=detections,
            message=None
        )
def process_combined(request, image):
    parts_raw = run_car_parts_model(image)
    damage_raw = run_damage_model(image)

    parts_processed = postprocess_car_parts_raw_outputs(parts_raw, image)
    damage_processed = postprocess_damage_raw_outputs(damage_raw, image)

    detections = combine_results(
        damage_result=damage_processed,
        car_parts_result=parts_processed
    )

    return EngineAnalyzeResponse(
        requestId=request.requestId,
        FileName=request.FileName,
        status="ok",
        mode=request.mode,
        image=ImageInfo(width=image.width, height=image.height),
        detections=detections,
        message=None
    )    
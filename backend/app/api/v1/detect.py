from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.api.v1.common import (
    get_pipeline,
    image_to_data_uri,
    read_upload,
    run_pipeline_operation,
)
from backend.app.services.pcb_engine import BoardCropError, DEFAULT_CONF, DEFAULT_IMGSZ

router = APIRouter()


@router.post("/detect")
async def detect_components(
    request: Request,
    image: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    imgsz: int = Form(DEFAULT_IMGSZ),
):
    frame, filename = await read_upload(image)
    try:
        result = await run_in_threadpool(
            run_pipeline_operation,
            get_pipeline(request),
            get_pipeline(request).run_detection_only,
            frame,
            conf,
            imgsz,
        )
    except BoardCropError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "status": "success",
        "filename": filename,
        "detections": result["detections"],
        "total_count": result["total_count"],
        "class_counts": result["class_counts"],
        "image_size": result["image_size"],
        "crop_method": result["crop_method"],
        "latency_ms": result["latency_ms"],
        "annotated_image": image_to_data_uri(result["annotated_image"]),
    }

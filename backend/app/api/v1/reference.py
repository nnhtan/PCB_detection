import cv2
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.api.v1.common import (
    get_pipeline,
    image_to_data_uri,
    read_upload,
)
from backend.app.core import config
from backend.app.services.pcb_engine import DEFAULT_CONF, DEFAULT_IMGSZ

router = APIRouter()


@router.get("/reference")
def get_reference(request: Request):
    pipeline = get_pipeline(request)
    detected_image = None
    detected_path = config.REFERENCE_IMAGE_OUTPUT_PATH.with_name(
        "reference_detected.jpg"
    )
    if detected_path.is_file():
        detected_image = cv2.imread(str(detected_path))
    elif config.REFERENCE_DETECTED_PATH.is_file():
        detected_image = cv2.imread(str(config.REFERENCE_DETECTED_PATH))
    return {
        "status": "success",
        "reference_data": pipeline.reference_data,
        "components_count": len(pipeline.reference_components),
        "components": pipeline.reference_components,
        "reference_image_base64": image_to_data_uri(pipeline.reference_image),
        "detected_image_base64": image_to_data_uri(detected_image),
    }


@router.post("/reference/update")
async def update_reference(
    request: Request,
    image: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    imgsz: int = Form(DEFAULT_IMGSZ),
):
    frame, _ = await read_upload(image)
    pipeline = get_pipeline(request)
    pipeline.ref_image_path = str(config.REFERENCE_IMAGE_OUTPUT_PATH)
    try:
        result = await run_in_threadpool(
            pipeline.create_reference_from_image, frame, conf, imgsz
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "status": "success",
        "message": (
            f"Đã tạo thành công reference map với "
            f"{result['components_count']} linh kiện."
        ),
        "components_count": result["components_count"],
        "components": result["components"],
        "annotated_image": image_to_data_uri(result["annotated_image"]),
    }


@router.get("/sample")
def get_sample():
    if not config.SAMPLE_IMAGE_PATH.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh mẫu PCB.")
    image = cv2.imread(str(config.SAMPLE_IMAGE_PATH))
    if image is None:
        raise HTTPException(status_code=500, detail="Không đọc được ảnh mẫu PCB.")
    return {"status": "success", "image": image_to_data_uri(image)}

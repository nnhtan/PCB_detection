import cv2
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from pathlib import Path

from backend.app.api.v1.common import (
    get_pipeline,
    get_reference_catalog,
    image_to_data_uri,
    read_upload,
)
from backend.app.core import config
from backend.app.services.pcb_engine import DEFAULT_CONF, DEFAULT_IMGSZ

router = APIRouter()


@router.get("/reference")
def get_reference(request: Request):
    pipeline = get_pipeline(request)
    catalog = get_reference_catalog(request)
    with pipeline.operation_lock:
        detected_path = catalog.get_annotated_image_path()
        detected_image = (
            cv2.imread(str(detected_path)) if detected_path is not None else None
        )
        reference_image = (
            image_to_data_uri(pipeline.reference_image)
            if pipeline.reference_image is not None
            else None
        )
        reference_data = pipeline.reference_data
        components = list(pipeline.reference_components)
        catalog_state = catalog.snapshot()
    return {
        "status": "success",
        "reference_data": reference_data,
        "components_count": len(components),
        "components": components,
        "reference_image_base64": reference_image,
        "detected_image_base64": image_to_data_uri(detected_image),
        "catalog": catalog_state["references"],
        "active_reference_id": catalog_state["active_reference_id"],
    }


@router.post("/reference/select")
async def select_reference(request: Request, reference_id: str = Form(...)):
    catalog = get_reference_catalog(request)
    try:
        reference = await run_in_threadpool(catalog.select, reference_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "status": "success",
        "message": f"Đã chọn mạch tham chiếu: {reference['name']}",
        "reference": reference,
        "active_reference_id": reference["id"],
    }


@router.post("/reference/update")
async def update_reference(
    request: Request,
    image: UploadFile = File(...),
    name: str = Form(""),
    conf: float = Form(DEFAULT_CONF),
    imgsz: int = Form(DEFAULT_IMGSZ),
):
    frame, filename = await read_upload(image)
    reference_name = name.strip() or Path(filename).stem or "Mạch tham chiếu"
    catalog = get_reference_catalog(request)
    try:
        result = await run_in_threadpool(
            catalog.create, frame, reference_name, conf, imgsz
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
        "reference": result["reference"],
        "active_reference_id": catalog.snapshot()["active_reference_id"],
    }


@router.get("/sample")
def get_sample():
    if not config.SAMPLE_IMAGE_PATH.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh mẫu PCB.")
    image = cv2.imread(str(config.SAMPLE_IMAGE_PATH))
    if image is None:
        raise HTTPException(status_code=500, detail="Không đọc được ảnh mẫu PCB.")
    return {"status": "success", "image": image_to_data_uri(image)}

import base64
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException, Request, UploadFile


def get_pipeline(request: Request) -> Any:
    return request.app.state.pipeline


def get_camera(request: Request) -> Any:
    return request.app.state.camera


def get_reference_catalog(request: Request) -> Any:
    return request.app.state.references


def run_pipeline_operation(pipeline: Any, operation: Any, *args: Any, **kwargs: Any) -> Any:
    with pipeline.operation_lock:
        return operation(*args, **kwargs)


async def read_upload(image: UploadFile) -> tuple[np.ndarray, str]:
    contents = await image.read(32 * 1024 * 1024 + 1)
    if len(contents) > 32 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Ảnh tải lên vượt quá giới hạn 32 MB.")
    try:
        decoded = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise HTTPException(status_code=400, detail="File tải lên không phải ảnh hợp lệ.") from exc
    if decoded is None:
        raise HTTPException(status_code=400, detail="File tải lên không phải ảnh hợp lệ.")
    return decoded, image.filename or "upload"


def image_to_data_uri(image: np.ndarray | None, quality: int = 90) -> str | None:
    if image is None:
        return None
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Không thể mã hóa ảnh kết quả.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def inspection_payload(result: dict[str, Any], filename: str | None = None) -> dict[str, Any]:
    payload = {
        "status": "success",
        "verdict": result["status"],
        "verdict_vi": result["status_vi"],
        "is_ok": result["status"] == "OK",
        "aligned": result["aligned"],
        "qa_available": result["qa_available"],
        "alignment_method": result["alignment_method"],
        "crop_method": result["crop_method"],
        "stats": result["stats"],
        "class_counts": result["class_counts"],
        "detections": result["detections"],
        "missing_components_details": result["missing_components_details"],
        "latency_ms": result["latency_ms"],
        "annotated_image": image_to_data_uri(result["annotated_image"]),
        "board_crop_image": image_to_data_uri(result["board_crop_image"]),
    }
    if filename is not None:
        payload["filename"] = filename
    return payload


def run_inspection(pipeline: Any, frame: np.ndarray, values: Any) -> dict[str, Any]:
    from backend.app.services.pcb_engine import (
        DEFAULT_CONF,
        DEFAULT_EXTRA_ALLOWED,
        DEFAULT_IMGSZ,
        DEFAULT_MISSING_ALLOWED,
        DEFAULT_TOLERANCE_RATIO,
    )

    if pipeline.reference_image is None or not pipeline.reference_components:
        raise HTTPException(
            status_code=409,
            detail="Chưa có mạch tham chiếu. Hãy tải ảnh mạch chuẩn trước khi kiểm tra.",
        )
    try:
        with pipeline.operation_lock:
            return pipeline.inspect_pcb(
                frame=frame,
                conf=float(values.get("conf", DEFAULT_CONF)),
                imgsz=int(values.get("imgsz", DEFAULT_IMGSZ)),
                tolerance_ratio=float(
                    values.get("tolerance_ratio", DEFAULT_TOLERANCE_RATIO)
                ),
                extra_allowed=int(values.get("extra_allowed", DEFAULT_EXTRA_ALLOWED)),
                missing_allowed=int(values.get("missing_allowed", DEFAULT_MISSING_ALLOWED)),
                auto_crop=parse_bool(values.get("auto_crop", True)),
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def parse_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)

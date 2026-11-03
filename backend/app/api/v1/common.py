import base64
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.core import config
from backend.app.services.conveyor import ConveyorCommunicationError


def get_pipeline(request: Request) -> Any:
    return request.app.state.pipeline


def get_camera(request: Request) -> Any:
    return request.app.state.camera


def get_reference_catalog(request: Request) -> Any:
    return request.app.state.references


async def send_result_to_conveyor(
    request: Request, is_ok: bool
) -> None:
    if not config.CONVEYOR_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Truyền kết quả đến ESP chưa được bật. Hãy đặt PCB_CONVEYOR_ENABLED=true.",
        )
    try:
        await run_in_threadpool(
            request.app.state.conveyor.send_result,
            is_ok,
        )
    except ConveyorCommunicationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def resume_conveyor(request: Request) -> None:
    if not config.CONVEYOR_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Truyền lệnh đến ESP chưa được bật. Hãy đặt PCB_CONVEYOR_ENABLED=true.",
        )
    try:
        await run_in_threadpool(request.app.state.conveyor.resume)
    except ConveyorCommunicationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
        "is_ok": (
            True
            if result["status"] == "OK"
            else False
            if result["status"] == "FAULTY"
            else None
        ),
        "aligned": result["aligned"],
        "qa_available": result["qa_available"],
        "comparison_scope": result["comparison_scope"],
        "comparison_note": result["comparison_note"],
        "alignment_method": result["alignment_method"],
        "alignment_error": result["alignment_error"],
        "alignment_transform": result["alignment_transform"],
        "orientation_transform": result["orientation_transform"],
        "alignment": {
            "aligned": result["aligned"],
            "qa_available": result["qa_available"],
            "method": result["alignment_method"],
            "error": result["alignment_error"],
            "transform": result["alignment_transform"],
            "orientation_transform": result["orientation_transform"],
        },
        "crop_method": result["crop_method"],
        "stats": result["stats"],
        "inventory_comparison": result["inventory_comparison"],
        "class_counts": result["class_counts"],
        "detections": result["detections"],
        "matched_components": result["matched_components"],
        "class_mismatch_details": result["class_mismatch_details"],
        "missing_components_details": result["missing_components_details"],
        "extra_components_details": result["extra_components_details"],
        "matching": {
            "matched": result["matched_components"],
            "class_mismatch": result["class_mismatch_details"],
            "missing": result["missing_components_details"],
            "extra": result["extra_components_details"],
            "stats": {
                "matched": result["stats"]["matched"],
                "missing": result["stats"]["missing"],
                "extra": result["stats"]["extra"],
            },
        },
        "latency_ms": result["latency_ms"],
        "annotated_image": image_to_data_uri(result["annotated_image"]),
        "board_crop_image": image_to_data_uri(result["board_crop_image"]),
    }
    if "scan_mode" in result:
        payload["scan_mode"] = result["scan_mode"]
        payload["scan_alignment_transform"] = result["scan_alignment_transform"]
    if filename is not None:
        payload["filename"] = filename
    return payload


def run_inspection(pipeline: Any, frame: np.ndarray, values: Any) -> dict[str, Any]:
    from backend.app.services.pcb_engine import BoardCropError

    if pipeline.reference_image is None or not pipeline.reference_components:
        raise HTTPException(
            status_code=409,
            detail="Chưa có mạch tham chiếu. Hãy tải ảnh mạch chuẩn trước khi kiểm tra.",
        )
    options = _inspection_options(values)
    try:
        with pipeline.operation_lock:
            return pipeline.inspect_pcb(
                frame=frame,
                auto_crop=parse_bool(values.get("auto_crop", True)),
                **options,
            )
    except BoardCropError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _inspection_options(values: Any) -> dict[str, Any]:
    from backend.app.services.pcb_engine import (
        DEFAULT_CONF,
        DEFAULT_EXTRA_ALLOWED,
        DEFAULT_IMGSZ,
        DEFAULT_MISSING_ALLOWED,
        MAX_TOLERANCE_RATIO,
        DEFAULT_TOLERANCE_RATIO,
    )

    tolerance_ratio = float(
        values.get("tolerance_ratio", DEFAULT_TOLERANCE_RATIO)
    )
    if not 0.0 < tolerance_ratio <= MAX_TOLERANCE_RATIO:
        raise HTTPException(
            status_code=422,
            detail=(
                "Dung sai vị trí phải lớn hơn 0 và không vượt quá "
                f"{MAX_TOLERANCE_RATIO:.0%} kích thước bo mạch."
            ),
        )
    return {
        "conf": float(values.get("conf", DEFAULT_CONF)),
        "imgsz": int(values.get("imgsz", DEFAULT_IMGSZ)),
        "tolerance_ratio": tolerance_ratio,
        "extra_allowed": int(values.get("extra_allowed", DEFAULT_EXTRA_ALLOWED)),
        "missing_allowed": int(values.get("missing_allowed", DEFAULT_MISSING_ALLOWED)),
        "fast_mode": parse_bool(values.get("fast_mode", False)),
    }


def run_tilted_inspection(
    pipeline: Any, frame: np.ndarray, values: Any
) -> dict[str, Any]:
    from backend.app.services.pcb_engine import (
        BoardCropError,
        find_reference_board_corners,
        find_board_corners,
        order_board_corners_for_reference,
        rectify_board_to_reference,
    )

    try:
        with pipeline.operation_lock:
            if pipeline.reference_image is None or not pipeline.reference_components:
                raise HTTPException(
                    status_code=409,
                    detail="Chưa có mạch tham chiếu. Hãy tải ảnh mạch chuẩn trước khi kiểm tra.",
                )
            options = _inspection_options(values)
            reference_height, reference_width = pipeline.reference_image.shape[:2]
            reference_aspect = reference_width / reference_height
            corners = find_reference_board_corners(frame, pipeline.aligner)
            alignment_method = "orb"
            prealigned = True
            if corners is None:
                corners = find_board_corners(
                    frame,
                    reference_aspect=reference_aspect,
                    padding_ratio=0.0,
                    include_dark_regions=True,
                )
                alignment_method = "contour"
                prealigned = False
            if corners is not None and not prealigned:
                corners = order_board_corners_for_reference(
                    corners, reference_aspect
                )
            if corners is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Không tìm thấy PCB trong khung hình hoặc không thể căn "
                        "chỉnh với mạch tham chiếu. Hãy đưa toàn bộ bo mạch vào "
                        "khung hình và chụp rõ nét."
                    ),
                )
            aligned_frame, transform = rectify_board_to_reference(
                frame, corners, pipeline.reference_image.shape
            )
            if aligned_frame is None or transform is None:
                raise HTTPException(
                    status_code=422,
                    detail="Không thể xoay và cắt PCB theo khung mạch tham chiếu.",
                )
            result = pipeline.inspect_pcb(
                frame=aligned_frame,
                auto_crop=False,
                prealigned=prealigned,
                **options,
            )
    except BoardCropError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result["crop_method"] = f"tilted_reference_{alignment_method}"
    result["scan_mode"] = "tilted_reference"
    result["scan_alignment_transform"] = transform.tolist()
    return result


def parse_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)

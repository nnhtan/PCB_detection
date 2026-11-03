import math

import cv2
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from backend.app.api.v1.common import (
    get_camera,
    get_pipeline,
    image_to_data_uri,
    inspection_payload,
    run_inspection,
    run_tilted_inspection,
)
from backend.app.core import config
from backend.app.services.pcb_engine import (
    DEFAULT_CONF,
    DEFAULT_TOLERANCE_RATIO,
    find_board_corners,
)

router = APIRouter()
CAMERA_FAST_IMGSZ = 1024
CAMERA_TILTED_IMGSZ = 1536


def _get_inspection_runner(scan_mode: str):
    if scan_mode == "standard":
        return run_inspection
    if scan_mode == "tilted_reference":
        return run_tilted_inspection
    raise HTTPException(
        status_code=422,
        detail="Chế độ quét camera không hợp lệ.",
    )


def _get_camera_imgsz(scan_mode: str, imgsz: int | None) -> int:
    if imgsz is not None:
        return imgsz
    if scan_mode == "tilted_reference":
        return CAMERA_TILTED_IMGSZ
    return CAMERA_FAST_IMGSZ


def _select_tilted_scan_frame(frame, frame_id, recent_frames, reference_aspect):
    selected_frame = frame
    selected_frame_id = frame_id
    best_margin = -1.0
    for candidate_frame, candidate_frame_id in recent_frames:
        corners = find_board_corners(
            candidate_frame,
            reference_aspect=reference_aspect,
            padding_ratio=0.0,
            include_dark_regions=True,
        )
        if corners is None:
            continue
        frame_height, frame_width = candidate_frame.shape[:2]
        margin = min(
            float(corners[:, 0].min()),
            frame_width - 1 - float(corners[:, 0].max()),
            float(corners[:, 1].min()),
            frame_height - 1 - float(corners[:, 1].max()),
        )
        if margin > best_margin:
            selected_frame = candidate_frame
            selected_frame_id = candidate_frame_id
            best_margin = margin
    return selected_frame, selected_frame_id


@router.get("/camera/status")
def camera_status(request: Request):
    return {"status": "success", "camera": get_camera(request).get_status()}


@router.post("/camera/connect")
def camera_connect(
    request: Request, source: str = Form(config.CAMERA_SOURCE)
):
    connected, message = get_camera(request).connect(source)
    return {
        "status": "success" if connected else "error",
        "message": message,
        "camera": get_camera(request).get_status(),
    }


@router.post("/camera/disconnect")
def camera_disconnect(request: Request):
    _, message = get_camera(request).disconnect()
    return {
        "status": "success",
        "message": message,
        "camera": get_camera(request).get_status(),
    }


@router.post("/camera/exposure")
def camera_set_exposure(request: Request, exposure_us: float = Form(...)):
    if not math.isfinite(exposure_us):
        raise HTTPException(status_code=422, detail="Thời gian phơi sáng không hợp lệ.")
    try:
        exposure = get_camera(request).set_exposure(exposure_us)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "success", "camera": get_camera(request).get_status(), **exposure}


@router.get("/camera/stream")
def camera_stream(request: Request):
    return StreamingResponse(
        get_camera(request).mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/camera/scan")
async def camera_scan(
    request: Request,
    conf: float = Form(DEFAULT_CONF),
    imgsz: int | None = Form(None),
    tolerance_ratio: float = Form(DEFAULT_TOLERANCE_RATIO),
    auto_crop: str | None = Form(None),
    fast_mode: str | None = Form("true"),
    scan_mode: str = Form("standard"),
):
    inspection_runner = _get_inspection_runner(scan_mode)
    camera = get_camera(request)
    frame, frame_id, fps = camera.get_frame()
    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Camera chưa kết nối hoặc chưa nhận được khung hình.",
        )
    pipeline = get_pipeline(request)
    if scan_mode == "tilted_reference" and pipeline.reference_image is not None:
        reference_height, reference_width = pipeline.reference_image.shape[:2]
        frame, frame_id = _select_tilted_scan_frame(
            frame,
            frame_id,
            camera.get_recent_frames(),
            reference_width / reference_height,
        )
    result = await run_in_threadpool(
        inspection_runner,
        pipeline,
        frame,
        {
            "conf": conf,
            "imgsz": _get_camera_imgsz(scan_mode, imgsz),
            "tolerance_ratio": tolerance_ratio,
            "auto_crop": auto_crop if auto_crop is not None else True,
            "fast_mode": fast_mode if fast_mode is not None else True,
        },
    )
    payload = inspection_payload(result)
    payload.update({"frame_id": frame_id, "camera_fps": fps})
    return payload


@router.post("/camera/snapshot")
async def camera_snapshot(
    request: Request,
    source: str = Form(config.CAMERA_SOURCE),
    conf: float = Form(DEFAULT_CONF),
    imgsz: int | None = Form(None),
    tolerance_ratio: float = Form(DEFAULT_TOLERANCE_RATIO),
    auto_crop: str | None = Form(None),
    scan_mode: str = Form("standard"),
):
    inspection_runner = _get_inspection_runner(scan_mode)
    camera_source: str | int
    try:
        camera_source = int(source)
    except ValueError:
        camera_source = source.strip()
    capture = cv2.VideoCapture(camera_source)
    try:
        if not capture.isOpened():
            raise HTTPException(
                status_code=400,
                detail=f"Không kết nối được với camera nguồn: {camera_source}.",
            )
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise HTTPException(status_code=500, detail="Không nhận được frame từ camera.")
    result = await run_in_threadpool(
        inspection_runner,
        get_pipeline(request),
        frame,
        {
            "conf": conf,
            "imgsz": _get_camera_imgsz(scan_mode, imgsz),
            "tolerance_ratio": tolerance_ratio,
            "auto_crop": auto_crop if auto_crop is not None else True,
        },
    )
    payload = inspection_payload(result)
    payload["raw_image"] = image_to_data_uri(frame)
    return payload

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
)
from backend.app.core import config
from backend.app.services.pcb_engine import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_TOLERANCE_RATIO,
)

router = APIRouter()


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


@router.get("/camera/roi")
def camera_get_roi(request: Request):
    return {"status": "success", "roi": get_camera(request).get_roi()}


@router.post("/camera/roi")
def camera_set_roi(
    request: Request,
    enabled: bool = Form(...),
    x1: float = Form(0.0),
    y1: float = Form(0.0),
    x2: float = Form(1.0),
    y2: float = Form(1.0),
    stream_crop: bool = Form(True),
):
    roi = get_camera(request).set_roi(
        enabled=enabled,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        stream_crop=stream_crop,
    )
    return {
        "status": "success",
        "roi": roi,
        "camera": get_camera(request).get_status(),
    }


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
    imgsz: int = Form(DEFAULT_IMGSZ),
    tolerance_ratio: float = Form(DEFAULT_TOLERANCE_RATIO),
    auto_crop: str | None = Form(None),
    apply_roi: bool = Form(True),
):
    camera = get_camera(request)
    frame, frame_id, fps = camera.get_frame(apply_roi=apply_roi)
    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Camera chưa kết nối hoặc chưa nhận được khung hình.",
        )
    result = await run_in_threadpool(
        run_inspection,
        get_pipeline(request),
        frame,
        {
            "conf": conf,
            "imgsz": imgsz,
            "tolerance_ratio": tolerance_ratio,
            "auto_crop": auto_crop if auto_crop is not None else True,
        },
    )
    payload = inspection_payload(result)
    payload.update({
        "frame_id": frame_id,
        "camera_fps": fps,
        "is_stationary": getattr(camera, "is_stationary", True),
        "motion_score": getattr(camera, "motion_score", 0.0),
        "roi_applied": camera.roi.get("enabled", False) if apply_roi else False,
    })
    return payload


@router.post("/camera/snapshot")
async def camera_snapshot(
    request: Request,
    source: str = Form(config.CAMERA_SOURCE),
    conf: float = Form(DEFAULT_CONF),
    imgsz: int = Form(DEFAULT_IMGSZ),
    tolerance_ratio: float = Form(DEFAULT_TOLERANCE_RATIO),
    auto_crop: str | None = Form(None),
):
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
        run_inspection,
        get_pipeline(request),
        frame,
        {
            "conf": conf,
            "imgsz": imgsz,
            "tolerance_ratio": tolerance_ratio,
            "auto_crop": auto_crop if auto_crop is not None else True,
        },
    )
    payload = inspection_payload(result)
    payload["raw_image"] = image_to_data_uri(frame)
    return payload

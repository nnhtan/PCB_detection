from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.app.api.v1.common import (
    get_pipeline,
    resume_conveyor,
    send_result_to_conveyor,
)
from backend.app.core import config
from backend.app.services.conveyor import ConveyorCommunicationError
from backend.app.services.pcb_engine import (
    DEFAULT_CONF,
    DEFAULT_EXTRA_ALLOWED,
    DEFAULT_IMGSZ,
    DEFAULT_MISSING_ALLOWED,
    DEFAULT_TOLERANCE_RATIO,
)

router = APIRouter()


class ConveyorResult(BaseModel):
    is_ok: bool


@router.get("/info")
def system_info(request: Request):
    pipeline = get_pipeline(request)
    model_names = getattr(pipeline.model, "names", {})
    classes = [
        model_names[key]
        for key in sorted(model_names)
    ] if isinstance(model_names, dict) else list(model_names)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "status": "success",
        "model_name": Path(pipeline.model_path).name,
        "framework": "Ultralytics YOLO / PyTorch + OpenCV",
        "device": str(pipeline.device),
        "device_name": device_name,
        "is_cuda": pipeline.use_cuda,
        "total_classes": len(classes),
        "classes": classes,
        "reference": {
            "loaded": len(pipeline.reference_components) > 0,
            "components_count": len(pipeline.reference_components),
            "image_name": Path(pipeline.ref_image_path).name,
            "orb_aligned_ready": (
                pipeline.aligner is not None and pipeline.aligner.valid
            ),
        },
        "default_parameters": {
            "conf": DEFAULT_CONF,
            "imgsz": DEFAULT_IMGSZ,
            "tolerance_ratio": DEFAULT_TOLERANCE_RATIO,
            "extra_allowed": DEFAULT_EXTRA_ALLOWED,
            "missing_allowed": DEFAULT_MISSING_ALLOWED,
        },
        "assets": {
            "model_exists": config.MODEL_PATH.is_file(),
            "reference_path": str(config.REFERENCE_PATH),
        },
    }


@router.post("/conveyor/result")
async def send_conveyor_result(result: ConveyorResult, request: Request):
    await send_result_to_conveyor(request, result.is_ok)
    return {
        "status": "success",
        "sent": "OK" if result.is_ok else "FAULTY",
    }


@router.post("/conveyor/resume")
async def resume_conveyor_belt(request: Request):
    await resume_conveyor(request)
    return {"status": "success", "sent": "RESUME"}


@router.get("/conveyor/event")
async def wait_for_conveyor_event(
    request: Request,
    timeout: float = Query(1.0, ge=0, le=5),
):
    conveyor = request.app.state.conveyor
    if not config.CONVEYOR_ENABLED or conveyor is None:
        return {"enabled": False, "event": None}

    try:
        event = await run_in_threadpool(conveyor.wait_for_event, timeout)
    except ConveyorCommunicationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"enabled": True, "event": event}

from pathlib import Path

import torch
from fastapi import APIRouter, Request

from backend.app.api.v1.common import get_pipeline
from backend.app.core import config
from backend.app.services.pcb_engine import (
    DEFAULT_CONF,
    DEFAULT_EXTRA_ALLOWED,
    DEFAULT_IMGSZ,
    DEFAULT_MISSING_ALLOWED,
    DEFAULT_TOLERANCE_RATIO,
)

router = APIRouter()


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

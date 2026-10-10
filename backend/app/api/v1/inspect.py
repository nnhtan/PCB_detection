from fastapi import APIRouter, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.api.v1.common import get_pipeline, inspection_payload, read_upload, run_inspection

router = APIRouter()


@router.post("/inspect")
async def inspect_board(
    request: Request,
    image: UploadFile = File(...),
    conf: float | None = Form(None),
    imgsz: int | None = Form(None),
    tolerance_ratio: float | None = Form(None),
    extra_allowed: int | None = Form(None),
    missing_allowed: int | None = Form(None),
    auto_crop: str | None = Form(None),
):
    frame, filename = await read_upload(image)
    values = {
        key: value
        for key, value in {
            "conf": conf,
            "imgsz": imgsz,
            "tolerance_ratio": tolerance_ratio,
            "extra_allowed": extra_allowed,
            "missing_allowed": missing_allowed,
            "auto_crop": auto_crop,
        }.items()
        if value is not None
    }
    result = await run_in_threadpool(
        run_inspection, get_pipeline(request), frame, values
    )
    return inspection_payload(result, filename)

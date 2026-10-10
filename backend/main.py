from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.api.api import api_router
from backend.app.core import config
from backend.app.services.camera_stream import CameraManager
from backend.app.services.pcb_engine import PCBPipeline


@asynccontextmanager
async def lifespan(application: FastAPI):
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.REFERENCE_IMAGE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    application.state.pipeline = PCBPipeline(
        model_path=str(config.MODEL_PATH),
        ref_json_path=str(config.REFERENCE_PATH),
        ref_image_path=str(config.REFERENCE_IMAGE_PATH),
    )
    application.state.camera = CameraManager(config.CAMERA_SOURCE)
    yield
    application.state.camera.disconnect()


app = FastAPI(title="PCB Vision AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api")
app.include_router(api_router, prefix="/api/v1", include_in_schema=False)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

def serve_frontend(path: str = ""):
    dist = config.FRONTEND_DIST_DIR.resolve()
    if not dist.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Frontend chưa được build. Chạy npm run build trong thư mục frontend.",
        )
    if path == "api" or path.startswith("api/") or path in {
        "health",
        "docs",
        "redoc",
        "openapi.json",
    }:
        raise HTTPException(status_code=404, detail="Endpoint không tồn tại.")
    candidate = (dist / path).resolve()
    if candidate != dist and dist not in candidate.parents:
        raise HTTPException(status_code=404, detail="Tệp không tồn tại.")
    if candidate.is_file():
        return FileResponse(candidate)
    index_file = Path(dist / "index.html")
    if index_file.is_file():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Không tìm thấy giao diện đã build.")


@app.get("/", include_in_schema=False)
def frontend_root():
    return serve_frontend()


@app.get("/{frontend_path:path}", include_in_schema=False)
def frontend_routes(frontend_path: str):
    return serve_frontend(frontend_path)

from fastapi import APIRouter

from backend.app.api.v1 import camera, detect, inspect, reference, system

router = APIRouter()
router.include_router(system.router, tags=["system"])
router.include_router(inspect.router, tags=["inspection"])
router.include_router(detect.router, tags=["detection"])
router.include_router(camera.router, tags=["camera"])
router.include_router(reference.router, tags=["reference"])

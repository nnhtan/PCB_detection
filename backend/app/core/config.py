import os
from pathlib import Path


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} phải là true/false hoặc 1/0.")


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
DEFAULT_MODEL_PATH = BACKEND_DIR / "models" / "best.pt"
if not DEFAULT_MODEL_PATH.is_file() and (PROJECT_DIR / "best.pt").is_file():
    DEFAULT_MODEL_PATH = PROJECT_DIR / "best.pt"
MODEL_PATH = Path(os.getenv("PCB_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
REFERENCE_PATH = Path(
    os.getenv(
        "PCB_REFERENCE_PATH",
        str(BACKEND_DIR / "storage" / "reference_components.json"),
    )
).expanduser()
REFERENCE_IMAGE_OUTPUT_PATH = Path(
    os.getenv(
        "PCB_REFERENCE_IMAGE_PATH",
        str(BACKEND_DIR / "storage" / "reference.jpg"),
    )
).expanduser()
DEFAULT_REFERENCE_IMAGE_PATH = REFERENCE_IMAGE_OUTPUT_PATH
if (
    "PCB_REFERENCE_IMAGE_PATH" not in os.environ
    and not DEFAULT_REFERENCE_IMAGE_PATH.is_file()
    and (PROJECT_DIR / "Image_20261019190958053.bmp").is_file()
):
    DEFAULT_REFERENCE_IMAGE_PATH = PROJECT_DIR / "Image_20261019190958053.bmp"
REFERENCE_IMAGE_PATH = Path(
    str(DEFAULT_REFERENCE_IMAGE_PATH)
).expanduser()
DEFAULT_REFERENCE_DETECTED_PATH = BACKEND_DIR / "storage" / "reference_detected.jpg"
if (
    not DEFAULT_REFERENCE_DETECTED_PATH.is_file()
    and (PROJECT_DIR / "reference_detected.jpg").is_file()
):
    DEFAULT_REFERENCE_DETECTED_PATH = PROJECT_DIR / "reference_detected.jpg"
REFERENCE_DETECTED_PATH = DEFAULT_REFERENCE_DETECTED_PATH
DEFAULT_SAMPLE_IMAGE_PATH = BACKEND_DIR / "storage" / "sample.jpg"
if not DEFAULT_SAMPLE_IMAGE_PATH.is_file() and (PROJECT_DIR / "Image_20261019190958053.bmp").is_file():
    DEFAULT_SAMPLE_IMAGE_PATH = PROJECT_DIR / "Image_20261019190958053.bmp"
SAMPLE_IMAGE_PATH = Path(
    os.getenv("PCB_SAMPLE_IMAGE_PATH", str(DEFAULT_SAMPLE_IMAGE_PATH))
).expanduser()
FRONTEND_DIST_DIR = PROJECT_DIR / "frontend" / "dist"
DEFAULT_CAMERA_IP = "169.254.128.194"
CAMERA_SOURCE = os.getenv("PCB_CAMERA_SOURCE", DEFAULT_CAMERA_IP)
CONVEYOR_ENABLED = _environment_flag("PCB_CONVEYOR_ENABLED")
CONVEYOR_PORT = os.getenv("PCB_SERIAL_PORT", "COM3")
CONVEYOR_BAUDRATE = int(os.getenv("PCB_SERIAL_BAUDRATE", "115200"))
CONVEYOR_TIMEOUT = float(os.getenv("PCB_SERIAL_TIMEOUT", "5"))

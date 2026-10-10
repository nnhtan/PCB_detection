"""Smoke-test the running FastAPI service using the bundled sample PCB image."""

import json
import os
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("PCB_API_URL", "http://127.0.0.1:8000").rstrip("/")
SAMPLE_PATH = Path(__file__).resolve().with_name("2.jpg")


def request_json(path: str, request: Request | None = None) -> dict:
    try:
        with urlopen(request or f"{BASE_URL}{path}", timeout=60) as response:
            return json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise AssertionError(f"{path} returned HTTP {error.code}: {detail}") from error


def post_image(path: str, image_path: Path, values: dict[str, str] | None = None) -> dict:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in (values or {}).items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(f"{value}\r\n".encode())
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode()
    )
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(image_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    request = Request(
        f"{BASE_URL}{path}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return request_json(path, request)


def test_backend() -> None:
    assert SAMPLE_PATH.is_file(), f"Sample image is missing: {SAMPLE_PATH}"
    health = request_json("/health")
    assert health["status"] == "ok"
    info = request_json("/api/info")
    reference = request_json("/api/reference")
    sample = request_json("/api/sample")
    camera = request_json("/api/camera/status")
    assert info["status"] == "success"
    assert sample.get("image", "").startswith("data:image/")
    assert "camera" in camera
    detected = post_image("/api/detect", SAMPLE_PATH)
    assert detected["status"] == "success"
    assert "detections" in detected
    if reference.get("components_count", 0):
        inspected = post_image("/api/inspect", SAMPLE_PATH, {"conf": "0.25"})
        assert inspected["status"] == "success"
        assert "qa_available" in inspected
    print(
        "FastAPI smoke test passed:",
        f"{detected['total_count']} detections,",
        f"{reference.get('components_count', 0)} reference components",
    )


if __name__ == "__main__":
    test_backend()

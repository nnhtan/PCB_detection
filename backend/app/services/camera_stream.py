import logging
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)
FRAME_HISTORY_SIZE = 12


class CameraManager:
    def __init__(self, source: str | int = "0") -> None:
        self.source: str | int = self._parse_source(source)
        self.cap: cv2.VideoCapture | None = None
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_id = 0
        self.frame_history: deque[tuple[int, np.ndarray]] = deque(
            maxlen=FRAME_HISTORY_SIZE
        )
        self.fps = 0.0
        self.running = False
        self.thread: threading.Thread | None = None
        self.is_connected = False
        self.last_error = ""
        self.hik_camera = None

    @staticmethod
    def _parse_source(source: str | int) -> str | int:
        try:
            return int(source)
        except (ValueError, TypeError):
            return str(source).strip()

    @staticmethod
    def _is_ip_address(text: str) -> bool:
        parts = text.strip().split(".")
        return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)

    def _open_capture(self) -> cv2.VideoCapture:
        sources_to_try = [self.source]
        if isinstance(self.source, str):
            raw = self.source.strip()
            if self._is_ip_address(raw):
                sources_to_try = [
                    f"http://{raw}:8080/video",
                    f"rtsp://{raw}:554/stream1",
                    f"rtsp://{raw}:554/h264Preview_01_main",
                    f"http://{raw}:4747/video",
                    f"http://{raw}",
                    raw,
                ]

        last_error = None
        for candidate in sources_to_try:
            try:
                capture = self._open_capture_candidate(candidate)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if capture.isOpened():
                    try:
                        backend = capture.getBackendName()
                    except cv2.error:
                        backend = "unknown"
                    hardware_decode = False
                    if hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
                        acceleration = capture.get(cv2.CAP_PROP_HW_ACCELERATION)
                        hardware_decode = acceleration not in (
                            getattr(cv2, "VIDEO_ACCELERATION_NONE", 0),
                            -1,
                        )
                    logger.info(
                        "Camera capture opened via %s (hardware decode: %s).",
                        backend,
                        hardware_decode,
                    )
                    return capture
                capture.release()
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Không thể mở nguồn camera: {self.source} ({last_error or 'Không mở được luồng'})")

    @staticmethod
    def _open_capture_candidate(candidate: str | int) -> cv2.VideoCapture:
        is_network_stream = isinstance(candidate, str) and candidate.lower().startswith(
            ("http://", "https://", "rtsp://", "rtsps://")
        )
        if (
            is_network_stream
            and hasattr(cv2, "CAP_PROP_HW_ACCELERATION")
            and hasattr(cv2, "VIDEO_ACCELERATION_ANY")
        ):
            try:
                capture = cv2.VideoCapture(
                    candidate,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_HW_ACCELERATION,
                        cv2.VIDEO_ACCELERATION_ANY,
                    ],
                )
                if capture.isOpened():
                    return capture
                capture.release()
                logger.debug(
                    "Hardware-accelerated decoding unavailable for %s; using the default backend.",
                    candidate,
                )
            except (cv2.error, TypeError) as exc:
                logger.debug(
                    "Could not enable hardware decoding for %s; using the default backend: %s",
                    candidate,
                    exc,
                )

        if isinstance(candidate, int) and sys.platform == "win32":
            try:
                capture = cv2.VideoCapture(candidate, cv2.CAP_DSHOW)
                if capture.isOpened():
                    return capture
                capture.release()
            except (cv2.error, TypeError) as exc:
                logger.debug(
                    "DirectShow unavailable for camera %s; using the default backend: %s",
                    candidate,
                    exc,
                )

        return cv2.VideoCapture(candidate)

    def connect(self, source: str | int | None = None) -> tuple[bool, str]:
        if source is not None:
            self.source = self._parse_source(source)

        self.disconnect()
        with self.lock:
            self.last_error = ""
            self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            with self.lock:
                if self.is_connected and self.frame is not None:
                    return True, "Kết nối camera thành công"
                if not self.running and self.last_error:
                    return False, self.last_error
            time.sleep(0.1)
        with self.lock:
            if self.is_connected:
                return True, "Camera đã kết nối, đang nạp hình..."
            return False, self.last_error or f"Hết thời gian chờ kết nối tới {self.source}"

    def disconnect(self) -> tuple[bool, str]:
        with self.lock:
            self.running = False
            cap = self.cap
            self.cap = None
            self.frame = None
            self.frame_history.clear()
            self.fps = 0.0
        if cap is not None:
            cap.release()
        thread = self.thread
        if thread is not None and thread.is_alive():
            is_hikvision = (
                isinstance(self.source, str) and self.source.lower() == "hikvision"
            )
            timeout = 3.0 if is_hikvision else 1.0
            thread.join(timeout=timeout)
        self.thread = None
        with self.lock:
            self.is_connected = False
        return True, "Đã ngắt kết nối camera"

    @staticmethod
    def _is_hikvision_source(source: str | int | None) -> bool:
        if source is None:
            return False
        text = str(source).strip().lower()
        if not text:
            return False
        if text.startswith(("http://", "https://", "rtsp://", "rtsps://")):
            return False
        if text in {"hikvision", "hikrobot", "hik"}:
            return True
        return CameraManager._is_ip_address(text)

    def _worker(self) -> None:
        if self._is_hikvision_source(self.source):
            hik_success = self._hikvision_worker()
            if hik_success:
                return
            # Nếu là IP mà Hikvision SDK không mở được, tự động thử mở luồng IP stream qua OpenCV
            if isinstance(self.source, str) and self._is_ip_address(self.source.strip()):
                with self.lock:
                    if not self.running:
                        return
                    self.last_error = f"Không tìm thấy camera Hikvision GigE tại {self.source}. Đang thử mở luồng IP camera qua OpenCV..."
            else:
                return

        try:
            cap = self._open_capture()
        except (RuntimeError, cv2.error, OSError, TypeError, ValueError) as exc:
            with self.lock:
                self.last_error = f"Lỗi mở camera: {exc}"
                self.running = False
            return

        with self.lock:
            self.cap = cap
            self.is_connected = True
            self.last_error = ""

        last_time = time.monotonic()
        frame_count = 0
        read_failures = 0
        while True:
            with self.lock:
                if not self.running:
                    break
            try:
                ok, frame = cap.read()
            except cv2.error as exc:
                ok, frame = False, None
                with self.lock:
                    self.last_error = f"Lỗi đọc camera: {exc}"
            if not ok or frame is None:
                read_failures += 1
                if read_failures >= 25:
                    with self.lock:
                        self.is_connected = False
                        self.last_error = f"Mất luồng camera: {self.source}. Đang kết nối lại..."
                    cap.release()
                    time.sleep(1.0)
                    with self.lock:
                        if not self.running:
                            break
                    try:
                        cap = self._open_capture()
                    except (RuntimeError, cv2.error, OSError, TypeError, ValueError) as exc:
                        with self.lock:
                            self.last_error = f"Không thể kết nối lại camera: {exc}"
                        read_failures = 0
                        continue
                    with self.lock:
                        self.cap = cap
                        self.is_connected = True
                        self.last_error = ""
                    read_failures = 0
                    frame_count = 0
                    last_time = time.monotonic()
                    continue
                time.sleep(0.04)
                continue
            read_failures = 0
            with self.lock:
                self.frame = frame
                self.frame_id += 1
                self.frame_history.append((self.frame_id, frame))
            frame_count += 1
            now = time.monotonic()
            if now - last_time >= 1.0:
                with self.lock:
                    self.fps = round(frame_count / (now - last_time), 1)
                frame_count = 0
                last_time = now

        cap.release()
        with self.lock:
            self.cap = None
            self.is_connected = False

    def _hikvision_worker(self) -> bool:
        viewer = None
        started = False
        try:
            from camera_viewer import HikCameraViewer

            viewer = HikCameraViewer()
            if not viewer.connect(self.source):
                with self.lock:
                    self.last_error = f"Không thể kết nối camera Hikvision với nguồn {self.source}."
                return False
            if not viewer.start_stream():
                with self.lock:
                    self.last_error = "Đã kết nối camera Hikvision nhưng không thể bắt đầu nhận hình."
                return False

            started = True
            with self.lock:
                self.hik_camera = viewer
                self.is_connected = True
                self.last_error = ""

            last_time = time.monotonic()
            frame_count = 0
            last_source_frame_count = -1
            while True:
                with self.lock:
                    if not self.running:
                        break
                with viewer.frame_lock:
                    source_frame_count = viewer.frame_count
                    frame = None
                    if (
                        source_frame_count != last_source_frame_count
                        and viewer.latest_frame is not None
                    ):
                        frame = viewer.latest_frame.copy()
                        last_source_frame_count = source_frame_count
                if frame is None:
                    time.sleep(0.005)
                    continue

                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                    self.frame_history.append((self.frame_id, frame))
                frame_count += 1
                now = time.monotonic()
                if now - last_time >= 1.0:
                    with self.lock:
                        self.fps = round(frame_count / (now - last_time), 1)
                    frame_count = 0
                    last_time = now
            return True
        except (ImportError, OSError, RuntimeError, cv2.error) as exc:
            with self.lock:
                self.last_error = f"Lỗi camera Hikvision: {exc}"
                if started:
                    self.running = False
            return started
        finally:
            if viewer is not None:
                try:
                    viewer.stop()
                except Exception as exc:
                    with self.lock:
                        self.last_error = f"Lỗi ngắt kết nối camera Hikvision: {exc}"
            with self.lock:
                if self.hik_camera is viewer:
                    self.hik_camera = None
                self.is_connected = False

    def set_exposure(self, exposure_us: float) -> dict[str, float]:
        with self.lock:
            camera = self.hik_camera
            if not self.is_connected or camera is None:
                raise RuntimeError("Camera Hikrobot chưa kết nối.")
            if not camera.set_exposure(exposure_us):
                raise RuntimeError("Camera Hikrobot không thể cập nhật thời gian phơi sáng.")
            return {
                "exposure_us": camera.current_exposure_us,
                "exposure_min_us": camera.exp_min,
                "exposure_max_us": camera.exp_max,
            }

    def get_frame(self) -> tuple[np.ndarray | None, int, float]:
        with self.lock:
            if self.frame is None:
                return None, self.frame_id, self.fps
            return self.frame.copy(), self.frame_id, self.fps

    def get_recent_frames(self) -> list[tuple[np.ndarray, int]]:
        with self.lock:
            return [
                (frame.copy(), frame_id)
                for frame_id, frame in reversed(self.frame_history)
            ]

    def get_status(self) -> dict[str, object]:
        with self.lock:
            status = {
                "running": self.running,
                "connected": self.is_connected,
                "source": str(self.source),
                "frame_id": self.frame_id,
                "fps": self.fps,
                "last_error": self.last_error,
            }
            if self.hik_camera is not None:
                status.update({
                    "exposure_us": self.hik_camera.current_exposure_us,
                    "exposure_min_us": self.hik_camera.exp_min,
                    "exposure_max_us": self.hik_camera.exp_max,
                })
            return status

    def mjpeg_frames(self):
        last_signature = None
        while True:
            frame, frame_id, _ = self.get_frame()
            is_placeholder = frame is None
            signature = (frame_id, is_placeholder)
            if signature == last_signature:
                time.sleep(0.05)
                continue
            last_signature = signature
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "CAMERA CHUA KET NOI",
                    (140, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (90, 100, 110),
                    2,
                    cv2.LINE_AA,
                )
            elif frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (1280, round(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            if ok:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
            time.sleep(0.05)

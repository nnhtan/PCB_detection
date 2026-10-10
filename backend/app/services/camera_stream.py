import threading
import time

import cv2
import numpy as np


class CameraManager:
    def __init__(self, source: str | int = "0") -> None:
        self.source: str | int = self._parse_source(source)
        self.cap: cv2.VideoCapture | None = None
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_id = 0
        self.fps = 0.0
        self.running = False
        self.thread: threading.Thread | None = None
        self.is_connected = False
        self.last_error = ""

    @staticmethod
    def _parse_source(source: str | int) -> str | int:
        try:
            return int(source)
        except (ValueError, TypeError):
            return str(source).strip()

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Không thể mở nguồn camera: {self.source}")
        return capture

    def connect(self, source: str | int | None = None) -> tuple[bool, str]:
        if source is not None:
            self.source = self._parse_source(source)
        self.disconnect()
        with self.lock:
            self.last_error = ""
            self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        deadline = time.monotonic() + 2.5
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
            self.fps = 0.0
        if cap is not None:
            cap.release()
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self.thread = None
        with self.lock:
            self.is_connected = False
        return True, "Đã ngắt kết nối camera"

    def _worker(self) -> None:
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

    def get_frame(self) -> tuple[np.ndarray | None, int, float]:
        with self.lock:
            if self.frame is None:
                return None, self.frame_id, self.fps
            return self.frame.copy(), self.frame_id, self.fps

    def get_status(self) -> dict[str, object]:
        with self.lock:
            return {
                "running": self.running,
                "connected": self.is_connected,
                "source": str(self.source),
                "frame_id": self.frame_id,
                "fps": self.fps,
                "last_error": self.last_error,
            }

    def mjpeg_frames(self):
        while True:
            frame, _, _ = self.get_frame()
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
            ok, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if ok:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
            time.sleep(0.03)

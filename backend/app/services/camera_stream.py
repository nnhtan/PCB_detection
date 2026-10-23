import threading
import time

import cv2
import numpy as np


class CameraManager:
    def __init__(self, source: str | int = "hikrobot") -> None:
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
        if isinstance(source, str) and source.strip().lower() == "hikrobot":
            return "hikrobot"
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

        timeout = 20.0 if self.source == "hikrobot" else 2.5
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.is_connected and self.frame is not None:
                    return True, "Kết nối camera thành công"
                if not self.running and self.last_error:
                    return False, self.last_error
            time.sleep(0.1)
        with self.lock:
            if self.source == "hikrobot":
                self.running = False
                self.last_error = (
                    self.last_error
                    or "Hết thời gian chờ frame đầu tiên từ camera Hikrobot."
                )
                self.is_connected = False
                return False, self.last_error
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
            thread.join(timeout=3.0)
        self.thread = None
        with self.lock:
            self.is_connected = False
        return True, "Đã ngắt kết nối camera"

    def _worker(self) -> None:
        if self.source == "hikrobot":
            self._hikrobot_worker()
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

    @staticmethod
    def _open_hikrobot():
        try:
            from camera_viewer import HikCameraViewer
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Không nạp được Hikrobot MVS SDK. Kiểm tra thư mục "
                "Python\\MvImport, MVS Runtime và DLL của SDK."
            ) from exc

        camera = HikCameraViewer()
        try:
            if not camera.connect():
                raise RuntimeError(
                    "Không tìm thấy hoặc không mở được camera Hikrobot. "
                    "Kiểm tra kết nối camera và MVS SDK."
                )
            if not camera.start_stream():
                raise RuntimeError("Không khởi động được luồng ảnh Hikrobot.")
        except Exception:
            camera.stop()
            raise
        return camera

    def _hikrobot_worker(self) -> None:
        camera = None
        try:
            camera = self._open_hikrobot()
            with self.lock:
                self.is_connected = True
                self.last_error = ""

            last_sdk_frame_id = -1
            last_frame_time = time.monotonic()
            while True:
                with self.lock:
                    if not self.running:
                        break

                with camera.frame_lock:
                    sdk_frame_id = camera.frame_count
                    frame = (
                        camera.latest_frame.copy()
                        if camera.latest_frame is not None
                        else None
                    )
                    camera_fps = camera.fps_camera

                if frame is not None and sdk_frame_id != last_sdk_frame_id:
                    with self.lock:
                        self.frame = frame
                        self.frame_id += 1
                        self.fps = round(camera_fps, 1)
                    last_sdk_frame_id = sdk_frame_id
                    last_frame_time = time.monotonic()
                elif time.monotonic() - last_frame_time > 20.0:
                    raise RuntimeError(
                        "Không nhận được frame mới từ camera Hikrobot trong 20 giây."
                    )
                time.sleep(0.01)
        except Exception as exc:
            with self.lock:
                if self.running:
                    self.last_error = f"Lỗi camera Hikrobot: {exc}"
                self.running = False
                self.is_connected = False
        finally:
            if camera is not None:
                try:
                    camera.stop()
                except Exception as exc:
                    with self.lock:
                        self.last_error = f"Lỗi ngắt kết nối camera Hikrobot: {exc}"
            with self.lock:
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

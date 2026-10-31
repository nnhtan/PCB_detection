import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROI_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "storage" / "camera_roi_config.json"
)


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
        self.hik_camera = None
        self.roi = self._load_roi_config()
        # Theo dõi chuyển động & phát hiện dừng (Stationary detection)
        self.motion_score = 0.0
        self.motion_threshold = 2.0
        self.is_stationary = True
        self.consecutive_stationary_frames = 0
        self._prev_motion_frame: np.ndarray | None = None

    def _update_motion(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        try:
            small_gray = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame,
                (160, 120),
                interpolation=cv2.INTER_AREA,
            )
            if self._prev_motion_frame is not None:
                diff = cv2.absdiff(small_gray, self._prev_motion_frame)
                score = float(np.mean(diff))
                self.motion_score = round(score, 2)
                if score <= self.motion_threshold:
                    self.consecutive_stationary_frames += 1
                else:
                    self.consecutive_stationary_frames = 0
                self.is_stationary = (self.consecutive_stationary_frames >= 2)
            self._prev_motion_frame = small_gray
        except Exception:
            pass

    @staticmethod
    def _load_roi_config() -> dict:
        default_roi = {
            "enabled": False,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 1.0,
            "y2": 1.0,
            "stream_crop": True,
        }
        try:
            if ROI_CONFIG_PATH.is_file():
                with open(ROI_CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        default_roi.update(data)
        except Exception:
            pass
        return default_roi

    def _save_roi_config(self) -> None:
        try:
            ROI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ROI_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.roi, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

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
            is_hikvision = (
                isinstance(self.source, str) and self.source.lower() == "hikvision"
            )
            timeout = 3.0 if is_hikvision else 1.0
            thread.join(timeout=timeout)
        self.thread = None
        with self.lock:
            self.is_connected = False
        return True, "Đã ngắt kết nối camera"

    def _worker(self) -> None:
        if isinstance(self.source, str) and self.source.lower() == "hikvision":
            self._hikvision_worker()
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
                self._update_motion(frame)
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

    def _hikvision_worker(self) -> None:
        viewer = None
        try:
            from camera_viewer import HikCameraViewer

            viewer = HikCameraViewer()
            if not viewer.connect():
                with self.lock:
                    self.last_error = "Không thể kết nối camera Hikvision. Kiểm tra camera và SDK MVS."
                    self.running = False
                return
            if not viewer.start_stream():
                with self.lock:
                    self.last_error = "Đã kết nối camera Hikvision nhưng không thể bắt đầu nhận hình."
                    self.running = False
                return

            with self.lock:
                self.hik_camera = viewer
                self.is_connected = True
                self.last_error = ""

            last_time = time.monotonic()
            frame_count = 0
            while True:
                with self.lock:
                    if not self.running:
                        break
                with viewer.frame_lock:
                    frame = (
                        viewer.latest_frame.copy()
                        if viewer.latest_frame is not None
                        else None
                    )
                if frame is None:
                    time.sleep(0.03)
                    continue

                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                    self._update_motion(frame)
                frame_count += 1
                now = time.monotonic()
                if now - last_time >= 1.0:
                    with self.lock:
                        self.fps = round(frame_count / (now - last_time), 1)
                    frame_count = 0
                    last_time = now
                time.sleep(0.01)
        except (ImportError, OSError, RuntimeError, cv2.error) as exc:
            with self.lock:
                self.last_error = f"Lỗi camera Hikvision: {exc}"
                self.running = False
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

    def get_roi(self) -> dict:
        with self.lock:
            return self.roi.copy()

    def set_roi(
        self,
        enabled: bool,
        x1: float = 0.0,
        y1: float = 0.0,
        x2: float = 1.0,
        y2: float = 1.0,
        stream_crop: bool = True,
    ) -> dict:
        x1 = max(0.0, min(1.0, float(x1)))
        x2 = max(0.0, min(1.0, float(x2)))
        y1 = max(0.0, min(1.0, float(y1)))
        y2 = max(0.0, min(1.0, float(y2)))
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        if (x_max - x_min) < 0.02 or (y_max - y_min) < 0.02:
            x_min, y_min, x_max, y_max = 0.0, 0.0, 1.0, 1.0
            enabled = False
        with self.lock:
            self.roi = {
                "enabled": bool(enabled),
                "x1": round(x_min, 4),
                "y1": round(y_min, 4),
                "x2": round(x_max, 4),
                "y2": round(y_max, 4),
                "stream_crop": bool(stream_crop),
            }
            self._save_roi_config()
            return self.roi.copy()

    def apply_crop(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or not self.roi.get("enabled", False):
            return frame
        h, w = frame.shape[:2]
        x1 = int(round(self.roi.get("x1", 0.0) * w))
        y1 = int(round(self.roi.get("y1", 0.0) * h))
        x2 = int(round(self.roi.get("x2", 1.0) * w))
        y2 = int(round(self.roi.get("y2", 1.0) * h))
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 10, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 10, min(h, y2))
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame
        return cropped

    def set_exposure(self, exposure_us: float) -> dict[str, float]:
        with self.lock:
            camera = self.hik_camera
            if not self.is_connected or camera is None:
                raise RuntimeError("Camera Hikvision chưa kết nối.")
            if not camera.set_exposure(exposure_us):
                raise RuntimeError("Camera Hikvision không thể cập nhật thời gian phơi sáng.")
            return {
                "exposure_us": camera.current_exposure_us,
                "exposure_min_us": camera.exp_min,
                "exposure_max_us": camera.exp_max,
            }

    def get_frame(self, apply_roi: bool = True) -> tuple[np.ndarray | None, int, float]:
        with self.lock:
            if self.frame is None:
                return None, self.frame_id, self.fps
            frame = self.frame.copy()
            if apply_roi and self.roi.get("enabled", False):
                frame = self.apply_crop(frame)
            return frame, self.frame_id, self.fps

    def get_status(self) -> dict[str, object]:
        with self.lock:
            status = {
                "running": self.running,
                "connected": self.is_connected,
                "source": str(self.source),
                "frame_id": self.frame_id,
                "fps": self.fps,
                "is_stationary": self.is_stationary,
                "motion_score": self.motion_score,
                "last_error": self.last_error,
                "roi": self.roi.copy(),
            }
            if self.hik_camera is not None or str(self.source).lower() in ("hikvision", "hikrobot"):
                exp_us = getattr(self.hik_camera, "current_exposure_us", 300000.0) if self.hik_camera else 300000.0
                exp_min = getattr(self.hik_camera, "exp_min", 34.0) if self.hik_camera else 34.0
                exp_max = getattr(self.hik_camera, "exp_max", 2000000.0) if self.hik_camera else 2000000.0
                status.update({
                    "exposure_us": exp_us,
                    "exposure_min_us": exp_min,
                    "exposure_max_us": exp_max,
                })
            return status

    def mjpeg_frames(self):
        while True:
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None
                roi = self.roi.copy()
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
            else:
                if roi.get("enabled", False):
                    if roi.get("stream_crop", True):
                        frame = self.apply_crop(frame)
                    else:
                        h, w = frame.shape[:2]
                        rx1 = int(round(roi.get("x1", 0.0) * w))
                        ry1 = int(round(roi.get("y1", 0.0) * h))
                        rx2 = int(round(roi.get("x2", 1.0) * w))
                        ry2 = int(round(roi.get("y2", 1.0) * h))
                        # Lớp phủ mờ phần bên ngoài vùng chọn
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)
                        dark_frame = cv2.addWeighted(frame, 0.40, np.zeros_like(frame), 0.60, 0)
                        frame = np.where(mask[:, :, None] == 255, frame, dark_frame)
                        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 240, 255), 2)
                        cv2.putText(
                            frame,
                            f"VUNG CROP ({rx2-rx1}x{ry2-ry1})",
                            (rx1 + 8, max(24, ry1 + 22)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 240, 255),
                            2,
                            cv2.LINE_AA,
                        )

            # Thu nhỏ tối đa chiều rộng 1280px để luồng MJPEG mượt mà qua web
            sh, sw = frame.shape[:2]
            if sw > 1280:
                scale = 1280.0 / sw
                frame = cv2.resize(frame, (1280, int(sh * scale)), interpolation=cv2.INTER_AREA)

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

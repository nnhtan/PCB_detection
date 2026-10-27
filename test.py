import json
import os
import sys
import threading
import time
from collections import Counter, deque
from functools import lru_cache
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof, string_at
from pathlib import Path

import cv2
import numpy as np
try:
    import torch
    from ultralytics import YOLO
except ImportError as import_error:  # chế độ --detect-only không cần YOLO
    torch = YOLO = None
    _IMPORT_ERROR = import_error

sys.setrecursionlimit(5000)  # ghép linh kiện dùng đệ quy (augmenting path)

# ============================================================
# 1. CẤU HÌNH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_STORAGE = PROJECT_DIR / "backend" / "storage"
MODEL_CANDIDATES = [
    str(PROJECT_DIR / "backend" / "models" / "best.pt"),
    str(PROJECT_DIR / "best.pt"),
]

REFERENCE_JSON = str(BACKEND_STORAGE / "reference_components.json")
REFERENCE_IMAGE = str(
    (BACKEND_STORAGE / "reference.jpg")
    if (BACKEND_STORAGE / "reference.jpg").is_file()
    else (PROJECT_DIR / "Image_20261019190958053.bmp")
)

# Camera điện thoại Android IP Webcam (hoặc SOURCE = 0 cho webcam máy tính)
SOURCE_VALUE = os.environ.get(
    "PCB_CAMERA_SOURCE", "http://192.168.0.190:8080/video"
)
try:
    SOURCE = int(SOURCE_VALUE)
except ValueError:
    SOURCE = SOURCE_VALUE
CAMERA_STARTUP_TIMEOUT_SECONDS = 10
CAMERA_RECONNECT_AFTER_SECONDS = 3   # mất hình quá lâu thì tự mở lại camera

# ---- Camera công nghiệp Hikvision (Hikrobot) tự động mở khi chạy chương trình ----
# "auto": thử Hikvision trước, không thấy thì dùng SOURCE ở trên (IP Webcam / webcam)
# "hikvision": chỉ dùng Hikvision (không thấy thì báo lỗi) | "ip": chỉ dùng SOURCE
CAMERA_BACKEND = os.environ.get("PCB_CAMERA_BACKEND", "auto")
HIK_EXPOSURE_US = float(os.environ.get("PCB_HIK_EXPOSURE_US", "30000"))
                                    # mặc định 30 ms để giảm nhòe/chờ frame trên băng tải
HIK_EXPOSURE_MIN_US = 1000.0         # giới hạn dưới khi giảm bằng phím
HIK_MVIMPORT_PATHS = [               # nơi tìm module MvImport của Hikvision SDK
    os.environ.get("HIK_MVIMPORT_PATH", ""),
    str(PROJECT_DIR / "Python" / "MvImport"),
    str(PROJECT_DIR.parent / "Python" / "MvImport"),
]

# ---- Nhận diện ----
SMOOTHING_WINDOW = 1
CONTRAST_ALPHA = 1.15
CONTRAST_BETA = -10

# ---- YOLO (PHẢI GIỐNG machpcbgoc.py) ----
CONF = 0.35
IMGSZ = 1536
MAX_DET = 1000
USE_CUDA = torch is not None and torch.cuda.is_available()
DEVICE = 0 if USE_CUDA else "cpu"

# ---- So sánh vị trí ----
MATCH_TOLERANCE_RATIO = 0.08
MISSING_ALLOWED = 0
EXTRA_MIN_CONF = 0.50

# ---- ORB ----
USE_ORB_ALIGNMENT = True
ORB_FEATURES = 2000
RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 10
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.40
RANSAC_THRESHOLD = 5.0
MIN_BOARD_AREA_RATIO = 0.03
MAX_BOARD_AREA_RATIO = 2.00

# ---- Phát hiện & cắt khung PCB (lọc contour, 4 góc, ổn định nhiều frame) ----
BOARD_FRAME_WIDTH = 640           # ảnh phân tích: cố định chiều rộng, giữ tỷ lệ gốc
BOARD_MIN_AREA_RATIO = 0.01       # nhỏ hơn 1% khung hình -> nhiễu
BOARD_MAX_AREA_RATIO = 0.90       # lớn hơn 90% khung hình -> mặt bàn / nền
BOARD_MIN_SOLIDITY = 0.80         # diện tích contour / diện tích bao lồi
BOARD_MIN_RECTANGULARITY = 0.85   # diện tích bao lồi / hình chữ nhật nhỏ nhất
BOARD_MIN_ASPECT = 0.25           # cạnh ngắn / cạnh dài
BOARD_APPROX_EPS = (0.01, 0.015, 0.02, 0.03, 0.04, 0.06)
BOARD_ANGLE_RANGE = (70, 110)     # góc trong hợp lệ của tứ giác (độ)
BOARD_MAX_SIDE_RATIO = 1.6        # tỉ lệ tối đa giữa hai cạnh đối diện
BOARD_MIN_SIDE_PX = 20
BOARD_MINRECT_SOLIDITY = 0.75     # contour biên có thể đứt bởi linh kiện/hoa văn
BOARD_MINRECT_RECTANGULARITY = 0.88
BOARD_PAD_RATIO = 0.02            # đệm theo cạnh = 2% cạnh ngắn của PCB
BOARD_PAD_MIN = 3
BOARD_PAD_MAX = 15
BOARD_BORDER_MARGIN = 5           # PCB sát mép khung hơn số px này -> không quét
BOARD_CONFIRM_FRAMES = 5          # số frame liên tiếp để xác nhận PCB
BOARD_MAX_MISS = 5                # mất PCB quá số frame này thì reset
BOARD_SMOOTH_ALPHA = 0.4          # hệ số làm mượt tọa độ (EMA)
BOARD_JUMP_PX = 40                # góc nhảy quá số px này bị coi là bất thường
BOARD_MIN_OUT_PX = 100            # cạnh ngắn tối thiểu của ảnh cắt ở độ phân giải gốc
BOARD_MAX_OUT_ASPECT = 1.25 / BOARD_MIN_ASPECT
BOARD_MIN_FILL = 0.60             # ngưỡng thấp để chấp nhận PCB có nền/biên khó phân đoạn
BOARD_AUTO_LANDSCAPE = True       # ảnh cắt luôn nằm ngang (False nếu ảnh reference đặt dọc)
BOARD_BG_BAND = 0.05              # dải viền để ước lượng nền, PCB phải nằm trọn trong ảnh
BOARD_EDGE_BG_MULT = 1.3
BOARD_EDGE_MIN_LO = 16
BOARD_EDGE_HI_RATIO = 2.0
BOARD_SEG_K = 4.0
BOARD_SEG_MIN_DIFF = 14
BOARD_SEG_CLOSE = 15
BOARD_CLOSE_KERNEL = 5
BOARD_CLOSE_ITER = 2
BOARD_CANNY_KERNEL = 3
SHOW_BOARD_EDGES = False          # True: hiện thêm cửa sổ Edges để chỉnh ngưỡng
DETECT_ONLY = False               # True hoặc --detect-only: xem crop, không nạp YOLO
BOARD_DEBUG_DIR = "debug"
SHOW_CROP_PREVIEW = True          # luôn xem đúng ảnh PCB sẽ đưa vào YOLO

# ---- Hiển thị ----
WINDOW_LIVE = "PCB Camera"
WINDOW_RESULT = "PCB Scan Result"
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 480
SHOW_RESULT_WINDOW = True
SHOW_FULL_FRAME_RESULT = False  # True: kết quả vẽ trên toàn khung hình; False: chỉ hiện PCB đã cắt

# ---- Trạng thái ----
STATUS_FAULTY = "FAULTY"
STATUS_OK = "OK"
STATUS_UNALIGNED = "UNALIGNED"

STATUS_TEXT_TERMINAL = {
    STATUS_FAULTY: "MẠCH BỊ LỖI",
    STATUS_OK: "MẠCH ĐẠT CHUẨN",
    STATUS_UNALIGNED: "ĐÃ QUÉT - CHƯA ĐỐI CHIẾU",
}
STATUS_TEXT_IMAGE = {
    STATUS_FAULTY: "PCB FAULTY",
    STATUS_OK: "PCB OK",
    STATUS_UNALIGNED: "PCB SCANNED",
}
UI_COLORS = {  # BGR
    "background": (23, 15, 11),
    "panel": (39, 24, 17),
    "border": (55, 41, 31),
    "primary": (246, 130, 59),
    "pass": (129, 185, 16),
    "faulty": (68, 68, 239),
    "warning": (0, 165, 255),
    "info": (0, 220, 255),
    "text": (251, 250, 249),
    "secondary": (175, 163, 156),
}
STATUS_COLOR = {
    STATUS_FAULTY: UI_COLORS["faulty"],
    STATUS_OK: UI_COLORS["pass"],
    STATUS_UNALIGNED: UI_COLORS["info"],
}
ANSI = {
    STATUS_FAULTY: "\033[91m",
    STATUS_OK: "\033[92m",
    STATUS_UNALIGNED: "\033[93m",
}
ANSI_RESET = "\033[0m"

CLASS_ALIASES = {
    "potentiometer": "variable resistor",
    "variable_resistor": "variable resistor",
}


@lru_cache(maxsize=None)
def normalize_component_class(class_name):
    normalized = " ".join(
        str(class_name).casefold().replace("-", " ").replace("_", " ").split()
    )
    return CLASS_ALIASES.get(normalized, normalized)


# ============================================================
# 2. ĐỌC CAMERA Ở LUỒNG RIÊNG
# ============================================================

class CameraStream:
    """Luôn đọc frame mới nhất trong nền, tự kết nối lại khi mất hình."""

    def __init__(self, source):
        self.source = source
        self.cap = self._open()
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.stop_event = threading.Event()
        self.thread = None

        if self.cap.isOpened():
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def is_opened(self):
        return self.cap.isOpened()

    def _loop(self):
        last_ok = time.monotonic()
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                last_ok = time.monotonic()
                # cap.read() luôn cấp mảng mới nên không cần copy
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                continue

            if time.monotonic() - last_ok > CAMERA_RECONNECT_AFTER_SECONDS:
                print("[WARN] Mất tín hiệu camera, đang kết nối lại...", flush=True)
                self.cap.release()
                self.cap = self._open()
                last_ok = time.monotonic()
            else:
                self.stop_event.wait(0.02)

    def read(self):
        with self.lock:
            return self.frame, self.frame_id

    def wait_for_frame(self, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.read()[0] is not None:
                return True
            time.sleep(0.02)
        return False

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()


# ============================================================
# 2b. CAMERA CÔNG NGHIỆP HIKVISION (TỰ ĐỘNG MỞ)
# ============================================================

def _load_hik_sdk():
    """Nạp Hikvision MvImport. Trả về (MvCameraControl_class, PixelType_header)
    hoặc (None, None) nếu không có SDK. Không dùng `import *` để không làm bẩn namespace."""
    import importlib

    for path in HIK_MVIMPORT_PATHS:
        if path and Path(path).is_dir() and path not in sys.path:
            sys.path.append(path)
    try:
        return (importlib.import_module("MvCameraControl_class"),
                importlib.import_module("PixelType_header"))
    except ImportError as e:
        print(f"[WARN] Không nạp được Hikvision SDK (MvImport): {e}", flush=True)
        return None, None


class HikCameraStream:
    """Camera Hikvision đọc ở luồng riêng, cùng giao diện với CameraStream:
    read(), wait_for_frame(), is_opened(), stop(). Thêm set_exposure()."""

    def __init__(self, exposure_us=HIK_EXPOSURE_US):
        self.exposure_us = float(exposure_us)
        self.exp_min, self.exp_max = 34.0, 1999733.0
        self.device_name = "Hikvision Camera"
        self.sdk = self.pixel = None
        self.cam = None
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.stop_event = threading.Event()
        self.thread = None
        self._sdk_ready = False
        self._grabbing = False
        self._opened = False

        try:
            self._opened = self._connect() and self._start()
        except Exception as e:
            print(f"[WARN] Lỗi khi mở camera Hikvision: {e}", flush=True)
        if not self._opened:
            self._release()          # dọn phần mở dở

    # ---------------------------------------------------------- kết nối
    def _connect(self):
        sdk, pixel = _load_hik_sdk()
        if sdk is None:
            return False
        self.sdk, self.pixel = sdk, pixel

        ret = sdk.MvCamera.MV_CC_Initialize()
        if ret != 0:
            print(f"[WARN] MV_CC_Initialize thất bại: 0x{ret:x}", flush=True)
            return False
        self._sdk_ready = True

        devices = sdk.MV_CC_DEVICE_INFO_LIST()
        ret = sdk.MvCamera.MV_CC_EnumDevices(sdk.MV_GIGE_DEVICE | sdk.MV_USB_DEVICE, devices)
        if ret != 0 or devices.nDeviceNum == 0:
            print("[WARN] Không tìm thấy camera Hikvision nào được kết nối.", flush=True)
            return False

        info = cast(devices.pDeviceInfo[0], POINTER(sdk.MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType == sdk.MV_GIGE_DEVICE:
            raw_name = info.SpecialInfo.stGigEInfo.chModelName
        else:
            raw_name = info.SpecialInfo.stUsb3VInfo.chModelName
        self.device_name = (bytes(raw_name).split(b"\x00")[0]
                            .decode("ascii", errors="ignore").strip() or self.device_name)

        self.cam = sdk.MvCamera()
        ret = self.cam.MV_CC_CreateHandle(info)
        if ret != 0:
            print(f"[WARN] MV_CC_CreateHandle thất bại: 0x{ret:x}", flush=True)
            return False
        ret = self.cam.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"[WARN] MV_CC_OpenDevice thất bại: 0x{ret:x} "
                  "(camera có thể đang được chương trình khác mở, ví dụ MVS)", flush=True)
            return False

        if info.nTLayerType == sdk.MV_GIGE_DEVICE:            # tối ưu gói GigE
            packet = self.cam.MV_CC_GetOptimalPacketSize()
            if packet > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet)

        limits = sdk.MVCC_FLOATVALUE()
        if self.cam.MV_CC_GetFloatValue("ExposureTime", limits) == 0:
            self.exp_min, self.exp_max = limits.fMin, limits.fMax

        self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)        # 0: tắt tự động
        self.set_exposure(self.exposure_us)
        return True

    def _start(self):
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"[WARN] MV_CC_StartGrabbing thất bại: 0x{ret:x}", flush=True)
            return False
        self._grabbing = True
        self.thread = threading.Thread(target=self._grab_worker, daemon=True)
        self.thread.start()
        return True

    # ---------------------------------------------------------- phơi sáng
    def set_exposure(self, exposure_us):
        if self.cam is None:
            return False
        clamped = max(self.exp_min, min(self.exp_max, float(exposure_us)))
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", clamped)
        if ret != 0:
            print(f"[WARN] Không đặt được ExposureTime {clamped:.0f} µs: 0x{ret:x}", flush=True)
            return False
        self.exposure_us = clamped
        print(f"[Camera] ExposureTime: {clamped:.0f} µs ({clamped / 1000:.1f} ms)", flush=True)
        return True

    # ---------------------------------------------------------- luồng ảnh
    def _grab_worker(self):
        sdk, pixel = self.sdk, self.pixel
        st_out = sdk.MV_FRAME_OUT()
        memset(byref(st_out), 0, sizeof(st_out))
        bgr_buf, bgr_size = None, 0

        while not self.stop_event.is_set():
            # timeout phải dài hơn thời gian phơi sáng
            timeout_ms = int(max(2000, self.exposure_us / 1000.0 * 2 + 500))
            ret = self.cam.MV_CC_GetImageBuffer(st_out, timeout_ms)
            if ret != 0 or not st_out.pBufAddr:
                self.stop_event.wait(0.01)
                continue

            image = None
            try:
                finfo = st_out.stFrameInfo
                w, h = finfo.nWidth, finfo.nHeight
                if finfo.enPixelType == pixel.PixelType_Gvsp_Mono8:
                    mono = np.frombuffer(string_at(st_out.pBufAddr, finfo.nFrameLen),
                                         dtype=np.uint8).reshape((h, w))
                    image = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
                else:                                   # Bayer, RGB... -> BGR bằng SDK
                    needed = w * h * 3
                    if bgr_buf is None or bgr_size != needed:
                        bgr_buf, bgr_size = (c_ubyte * needed)(), needed
                    param = sdk.MV_CC_PIXEL_CONVERT_PARAM_EX()
                    memset(byref(param), 0, sizeof(param))
                    param.nWidth, param.nHeight = w, h
                    param.pSrcData = st_out.pBufAddr
                    param.nSrcDataLen = finfo.nFrameLen
                    param.enSrcPixelType = finfo.enPixelType
                    param.enDstPixelType = pixel.PixelType_Gvsp_BGR8_Packed
                    param.pDstBuffer = bgr_buf
                    param.nDstBufferSize = needed
                    if self.cam.MV_CC_ConvertPixelTypeEx(param) == 0:
                        image = np.frombuffer(string_at(bgr_buf, needed),
                                              dtype=np.uint8).reshape((h, w, 3))
            except Exception as e:
                print(f"[WARN] Lỗi giải mã frame Hikvision: {e}", flush=True)
            finally:
                self.cam.MV_CC_FreeImageBuffer(st_out)

            if image is not None:
                with self.lock:                          # mỗi frame là mảng mới
                    self.frame = image
                    self.frame_id += 1

    # ---------------------------------------------------------- giao diện chung
    def is_opened(self):
        return self._opened

    def read(self):
        with self.lock:
            return self.frame, self.frame_id

    def wait_for_frame(self, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.read()[0] is not None:
                return True
            time.sleep(0.02)
        return False

    def _release(self):
        if self.cam is not None:
            if self._grabbing:
                self.cam.MV_CC_StopGrabbing()
                self._grabbing = False
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
        if self._sdk_ready:
            self.sdk.MvCamera.MV_CC_Finalize()
            self._sdk_ready = False

    def stop(self):
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self._release()


def open_camera():
    """Tự động mở camera. Trả về (stream, nhãn). Hikvision trước, dự phòng SOURCE."""
    backend = CAMERA_BACKEND.strip().lower()

    if backend in ("auto", "hikvision", "hik"):
        print("[INFO] Đang tìm camera Hikvision...", flush=True)
        hik = HikCameraStream()
        if hik.is_opened():
            print(f"[INFO] Đã mở camera Hikvision: {hik.device_name}", flush=True)
            return hik, f"Hikvision {hik.device_name}"
        hik.stop()
        if backend != "auto":
            raise RuntimeError(
                "\nKhông mở được camera Hikvision.\n\n"
                "Kiểm tra:\n"
                "1. Camera đã cấp nguồn và nối cáp (GigE/USB).\n"
                "2. Đã cài MVS và có thư mục Python/MvImport (hoặc đặt biến HIK_MVIMPORT_PATH).\n"
                "3. Không có chương trình khác (MVS, camera_viewer.py) đang giữ camera.\n"
                "4. Muốn dùng camera khác, đặt CAMERA_BACKEND = \"ip\" hoặc \"auto\"."
            )
        print(f"[INFO] Không có camera Hikvision, chuyển sang SOURCE = {SOURCE}", flush=True)

    return CameraStream(SOURCE), f"SOURCE = {SOURCE}"


# ============================================================
# 3. LOAD MODEL / REFERENCE
# ============================================================

def find_model():
    for model_name in MODEL_CANDIDATES:
        if Path(model_name).exists():
            return model_name
    raise FileNotFoundError(
        "Không tìm thấy model YOLO.\n"
        f"Đã kiểm tra: {MODEL_CANDIDATES}\n"
        "Hãy đặt file .pt vào cùng thư mục với file này."
    )


def load_reference():
    path = Path(REFERENCE_JSON)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {REFERENCE_JSON}.\nHãy chạy machpcbgoc.py trước."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "components" not in data:
        raise ValueError(f"{REFERENCE_JSON} không có trường 'components'.")
    if len(data["components"]) == 0:
        raise ValueError(f"{REFERENCE_JSON} không có linh kiện nào.")

    for i, component in enumerate(data["components"], start=1):
        missing = {"class", "x", "y"} - set(component.keys())
        if missing:
            raise ValueError(f"Linh kiện {i} thiếu dữ liệu: {missing}")

    return data


def load_reference_image():
    image_path = Path(REFERENCE_IMAGE)
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh reference: {image_path.resolve()}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Không đọc được ảnh reference.")
    return image


# ============================================================
# 4. YOLO
# ============================================================

def adjust_contrast(image):
    return cv2.convertScaleAbs(image, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA)


def detect_components(model, frame):
    results = model.predict(
        source=frame,
        conf=CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        half=USE_CUDA,
        max_det=MAX_DET,
        agnostic_nms=True,
        verbose=False,
    )
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    # Chuyển cả lô sang CPU một lần (tránh đồng bộ GPU cho từng box).
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
    confs = boxes.conf.cpu().numpy().tolist()
    cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
    cx = ((xyxy[:, 0] + xyxy[:, 2]) / 2.0).tolist()
    cy = ((xyxy[:, 1] + xyxy[:, 3]) / 2.0).tolist()
    names = model.names

    return [
        {
            "id": i + 1,
            "class_id": cls_ids[i],
            "class": normalize_component_class(names[cls_ids[i]]),
            "confidence": confs[i],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx[i], "cy": cy[i],
        }
        for i, (x1, y1, x2, y2) in enumerate(xyxy.tolist())
    ]


# ============================================================
# 5. PHÁT HIỆN VÀ CẮT KHUNG PCB
#    lọc contour -> 4 góc -> đệm theo cạnh -> ổn định nhiều frame
# ============================================================

def order_points(points):
    """TL, TR, BR, BL theo chiều kim đồng hồ — đúng cả khi PCB xoay nghiêng."""
    pts = np.array(points, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -np.argmin(pts.sum(axis=1)), axis=0)


def _side_lengths(q):
    return [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]


def _quad_angles(q):
    out = []
    for i in range(4):
        a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        out.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    return out


def _valid_quad(q):
    """Chỉ chấp nhận tứ giác góc gần vuông, cạnh đối xấp xỉ nhau."""
    s = _side_lengths(q)
    if min(s) < BOARD_MIN_SIDE_PX:
        return False
    if max(s[0], s[2]) / min(s[0], s[2]) > BOARD_MAX_SIDE_RATIO:
        return False
    if max(s[1], s[3]) / min(s[1], s[3]) > BOARD_MAX_SIDE_RATIO:
        return False
    return all(BOARD_ANGLE_RANGE[0] <= a <= BOARD_ANGLE_RANGE[1]
               for a in _quad_angles(q))


def _quad_ok(q):
    """Tứ giác lồi, đúng chiều kim đồng hồ, không suy biến."""
    for i in range(4):
        e1 = q[(i + 1) % 4] - q[i]
        e2 = q[(i + 2) % 4] - q[(i + 1) % 4]
        if e1[0] * e2[1] - e1[1] * e2[0] <= 1e-6:
            return False
    return True


def _line_intersection(p1, d1, p2, d2):
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def _offset_quad(q, pad):
    """Đẩy từng CẠNH ra ngoài pad px theo pháp tuyến rồi giao hai đường kề nhau."""
    lines = []
    for i in range(4):
        p, d = q[i], q[(i + 1) % 4] - q[i]
        d = d / (np.linalg.norm(d) + 1e-9)
        n = np.array([d[1], -d[0]], dtype=np.float32)   # pháp tuyến hướng ra ngoài
        lines.append((p + n * pad, d))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        pt = _line_intersection(p1, d1, p2, d2)
        out.append(pt if pt is not None else p2)
    return np.array(out, dtype=np.float32)


def _compute_pad(q):
    return float(np.clip(BOARD_PAD_RATIO * min(_side_lengths(q)),
                         BOARD_PAD_MIN, BOARD_PAD_MAX))


def _corners_inside(q, shape):
    h, w = shape[:2]
    return bool(np.all((q[:, 0] >= 0) & (q[:, 0] <= w - 1)
                       & (q[:, 1] >= 0) & (q[:, 1] <= h - 1)))


def _check_inside(contour, quad, shape, m):
    """Trả về lý do nếu PCB chưa nằm trọn trong khung hình, ngược lại None."""
    h, w = shape[:2]
    if (np.any(quad[:, 0] < m) or np.any(quad[:, 1] < m)
            or np.any(quad[:, 0] > w - 1 - m) or np.any(quad[:, 1] > h - 1 - m)):
        return "Goc PCB sat/ngoai mep khung"
    x, y, bw, bh = cv2.boundingRect(contour)
    if x <= m or y <= m or x + bw >= w - m or y + bh >= h - m:
        return "Bounding box cham mep khung"
    p = contour.reshape(-1, 2)
    if np.any((p[:, 0] <= m) | (p[:, 1] <= m)
              | (p[:, 0] >= w - 1 - m) | (p[:, 1] >= h - 1 - m)):
        return "Contour cham mep anh"
    return None


def _hull_to_quad(hull, solidity, rectangularity):
    if (solidity >= BOARD_MINRECT_SOLIDITY
            and rectangularity >= BOARD_MINRECT_RECTANGULARITY):
        quad = order_points(cv2.boxPoints(cv2.minAreaRect(hull)))
        if _valid_quad(quad):
            return quad

    peri = cv2.arcLength(hull, True)
    for eps in BOARD_APPROX_EPS:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            q = order_points(approx)
            if _valid_quad(q):
                return q
    return None


def _touches_border(contour, shape, margin):
    h, w = shape[:2]
    points = contour.reshape(-1, 2)
    return bool(np.any(
        (points[:, 0] <= margin) | (points[:, 1] <= margin)
        | (points[:, 0] >= w - 1 - margin)
        | (points[:, 1] >= h - 1 - margin)
    ))


def _evaluate_contour(c, shape):
    frame_area = shape[0] * shape[1]
    area = cv2.contourArea(c)
    if area < BOARD_MIN_AREA_RATIO * frame_area:
        return "small"
    if area > BOARD_MAX_AREA_RATIO * frame_area:
        return "large"
    if _touches_border(c, shape, BOARD_BORDER_MARGIN):
        return "border"
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return "solid"
    solidity = area / hull_area
    if solidity < BOARD_MIN_SOLIDITY:
        return "solid"
    (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
    if rw * rh <= 0:
        return "rect"
    rectangularity = hull_area / (rw * rh)
    if rectangularity < BOARD_MIN_RECTANGULARITY:
        return "rect"
    if min(rw, rh) / max(rw, rh) < BOARD_MIN_ASPECT:
        return "aspect"
    quad = _hull_to_quad(hull, solidity, rectangularity)
    if quad is None:
        return "quad"
    if _check_inside(c, quad, shape, BOARD_BORDER_MARGIN):
        return "border"
    return area * solidity * rectangularity, quad


def _best_candidate(masks, shape):
    best, rejects = None, Counter()
    largest_border_area = 0.0
    for source, binary in masks:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]
        for contour in contours:
            result = _evaluate_contour(contour, shape)
            if isinstance(result, str):
                rejects[result] += 1
                if result == "border":
                    largest_border_area = max(
                        largest_border_area, cv2.contourArea(contour)
                    )
                continue
            if best is None or result[0] > best[0]:
                best = (result[0], contour, result[1], source)
    return best, rejects, largest_border_area


def _basis(shape):
    h, w = shape[:2]
    x = np.linspace(-1, 1, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    y = np.linspace(-1, 1, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    return np.stack([np.ones_like(x), x, y, x * y, x * x, y * y], axis=-1)


def _border_band(shape, fraction):
    h, w = shape[:2]
    band_h, band_w = max(3, int(h * fraction)), max(3, int(w * fraction))
    band = np.zeros((h, w), dtype=bool)
    band[:band_h] = band[-band_h:] = True
    band[:, :band_w] = band[:, -band_w:] = True
    return band


def _fit_background(gray, basis, band):
    """Mô hình hóa nền không đồng đều từ dải viền và loại điểm ngoại lai."""
    step = 4
    sampled_band = band[::step, ::step]
    features = basis[::step, ::step][sampled_band]
    values = gray[::step, ::step][sampled_band].astype(np.float32)
    if len(values) < 50:
        features = basis[band]
        values = gray[band].astype(np.float32)
    keep = np.ones(len(values), dtype=bool)
    sigma = 1.0
    for _ in range(3):
        coefficients = np.linalg.lstsq(features[keep], values[keep], rcond=None)[0]
        residual = values - features @ coefficients
        inlier_residual = residual[keep]
        sigma = float(
            1.4826 * np.median(
                np.abs(inlier_residual - np.median(inlier_residual))
            ) + 1e-3
        )
        updated_keep = np.abs(residual) < 3.0 * sigma
        if updated_keep.sum() < 50:
            break
        keep = updated_keep
    return basis @ coefficients, sigma


def _fill_ratio(contour, corners, shape):
    """Tỉ lệ ảnh cắt thuộc PCB. Dùng contour và góc THÔ của cùng một frame."""
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if min(w, h) < 1:
        return 0.0
    mask = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return float(cv2.warpPerspective(mask, M, (w, h)).mean() / 255.0)


def _check_output_size(frame_corners):
    """Kiểm tra kích thước crop ở độ phân giải gốc, không ở ảnh phân tích."""
    tl, tr, br, bl = frame_corners
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if min(w, h) < BOARD_MIN_OUT_PX:
        return f"Anh cat qua nho ({min(w, h)}px)"
    if max(w, h) / min(w, h) > BOARD_MAX_OUT_ASPECT:
        return "Ti le anh cat bat thuong"
    return None


class CornerTracker:
    """Xác nhận PCB qua nhiều frame liên tiếp và làm mượt tọa độ bốn góc."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.smoothed = None
        self.pending = None
        self.confirm = 0
        self.miss = 0
        self.outlier = 0

    @staticmethod
    def _align(ref, q):
        """Xoay vòng thứ tự góc để khớp ref (tránh nhảy góc khi PCB gần 45°)."""
        best = min(range(4), key=lambda k: np.abs(np.roll(q, -k, axis=0) - ref).sum())
        return np.roll(q, -best, axis=0)

    @staticmethod
    def _max_dist(a, b):
        return float(np.linalg.norm(a - b, axis=1).max())

    def update(self, q):
        if q is None:
            self.miss += 1
            # Chưa xác nhận xong: frame mất phát hiện làm gián đoạn -> đếm lại từ đầu
            if self.smoothed is None:
                self.pending = None
                self.confirm = 0
            if self.miss > BOARD_MAX_MISS:
                self.reset()
            return self.smoothed
        self.miss = 0
        if self.smoothed is None:                       # đang xác nhận
            if self.pending is not None:
                q = self._align(self.pending, q)
            if self.pending is not None and self._max_dist(self.pending, q) < BOARD_JUMP_PX:
                self.confirm += 1
            else:
                self.confirm = 1
            self.pending = q
            if self.confirm >= BOARD_CONFIRM_FRAMES:
                self.smoothed = q.copy()
            return self.smoothed
        q = self._align(self.smoothed, q)               # đã xác nhận
        if self._max_dist(self.smoothed, q) > BOARD_JUMP_PX:
            self.outlier += 1
            if self.outlier >= BOARD_CONFIRM_FRAMES:    # thay đổi kéo dài -> PCB mới
                self.smoothed, self.outlier = q.copy(), 0
        else:
            self.outlier = 0
            self.smoothed = ((1 - BOARD_SMOOTH_ALPHA) * self.smoothed
                             + BOARD_SMOOTH_ALPHA * q)
        return self.smoothed


class BoardDetector:
    """Chạy trên từng frame live. `ready` chỉ True khi frame HIỆN TẠI có PCB hợp lệ,
    nằm trọn trong khung và đã được xác nhận ổn định qua nhiều frame."""

    def __init__(self):
        self.tracker = CornerTracker()
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (BOARD_CLOSE_KERNEL, BOARD_CLOSE_KERNEL)
        )
        k = BOARD_SEG_CLOSE | 1
        self.seg_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.small_kernel = np.ones((3, 3), dtype=np.uint8)
        self._shape = None
        self._basis = None
        self._band = None

    def _prepare_background(self, shape):
        if self._shape != shape[:2]:
            self._shape = shape[:2]
            self._basis = _basis(shape)
            self._band = _border_band(shape, BOARD_BG_BAND)

    def _candidates(self, gray):
        """Kết hợp nhiều ngưỡng biên và phân đoạn để chống đổi sáng/nền vân."""
        self._prepare_background(gray.shape)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.abs(gx) + np.abs(gy)
        background_gradient = float(np.percentile(gradient[self._band], 75))
        adaptive_low = int(np.clip(
            0.5 * background_gradient, BOARD_EDGE_MIN_LO, 80
        ))
        canny_candidates = []
        all_edges = np.zeros_like(gray)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (BOARD_CANNY_KERNEL, BOARD_CANNY_KERNEL)
        )
        thresholds = (
            (adaptive_low, max(adaptive_low + 20, 2 * adaptive_low)),
            (adaptive_low, max(adaptive_low + 20, 3 * adaptive_low)),
            (int(1.5 * adaptive_low), max(adaptive_low + 20, 3 * adaptive_low)),
        )
        for low, high in thresholds:
            edge_map = cv2.Canny(blur, low, min(high, 220))
            edge_map = cv2.morphologyEx(
                edge_map, cv2.MORPH_CLOSE, kernel, iterations=BOARD_CLOSE_ITER
            )
            all_edges = cv2.bitwise_or(all_edges, edge_map)
            canny_candidates.append((f"canny_{low}_{high}", edge_map))

        background, sigma = _fit_background(blur, self._basis, self._band)
        difference = np.abs(blur.astype(np.float32) - background)
        threshold = max(BOARD_SEG_MIN_DIFF, BOARD_SEG_K * sigma)
        segmented = (difference > threshold).astype(np.uint8) * 255
        segmented = cv2.morphologyEx(
            segmented, cv2.MORPH_OPEN, self.small_kernel
        )
        segmented = cv2.morphologyEx(
            segmented, cv2.MORPH_CLOSE, self.seg_close
        )
        contours, _ = cv2.findContours(
            segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        solid = np.zeros_like(segmented)
        cv2.drawContours(solid, contours, -1, 255, cv2.FILLED)
        info = {
            "canny_thresholds": thresholds,
            "background_gradient": background_gradient,
            "background_sigma": sigma,
            "segmentation_threshold": threshold,
        }
        return canny_candidates + [("background", solid)], all_edges, solid, info

    def update(self, frame):
        frame_h, frame_w = frame.shape[:2]
        det_w = BOARD_FRAME_WIDTH
        det_h = max(1, int(frame_h * det_w / frame_w))
        small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        gray = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        shape = gray.shape
        masks, edges, solid, info = self._candidates(gray)
        candidate, rejects, largest_border_area = _best_candidate(masks, shape)
        if (candidate is not None
                and largest_border_area >= 2.0 * cv2.contourArea(candidate[1])):
            candidate = None

        state = {"ready": False, "status": "", "raw_quad": None, "corners": None,
                 "frame_corners": None, "det_size": (det_w, det_h), "edges": edges,
                 "mask": solid, "source": None, "rejects": rejects, "info": info}
        blocked = detected_now = False
        contour = raw_quad = None

        if candidate is None:
            if rejects["border"]:
                state["status"] = "PCB cham/vuot mep khung - khong quet"
                blocked = True
                self.tracker.reset()
            else:
                detail = ", ".join(
                    f"{key}:{count}" for key, count in sorted(rejects.items())
                )
                state["status"] = "Khong tim thay PCB hop le" + (
                    f" ({detail})" if detail else ""
                )
        else:
            _, contour, raw_quad, source = candidate
            detected_now = True
            state["raw_quad"] = raw_quad
            state["source"] = source

        stable = None if blocked else self.tracker.update(raw_quad)

        if stable is not None and detected_now:
            corners = _offset_quad(stable, _compute_pad(stable))          # góc ổn định
            raw_corners = _offset_quad(raw_quad, _compute_pad(raw_quad))  # góc thô, để kiểm tra
            scale = np.array([frame_w / det_w, frame_h / det_h], dtype=np.float32)
            if not (_quad_ok(corners) and _quad_ok(raw_corners)):
                state["status"] = "Goc sau khi dem khong hop le"
            elif not (_corners_inside(corners, shape)
                      and _corners_inside(raw_corners, shape)):
                state["status"] = "Phan dem vuot ra ngoai khung"
            else:
                err = _check_output_size(corners * scale)
                if err is None:
                    fill = _fill_ratio(contour, raw_corners, shape)
                    if fill < BOARD_MIN_FILL:
                        err = f"Anh cat chua nhieu nen ({fill:.0%} la PCB)"
                if err:
                    state["status"] = err
                else:
                    state["ready"] = True
                    state["status"] = "San sang quet"
                    state["corners"] = corners
                    state["frame_corners"] = (corners * scale).astype(np.float32)
        elif not state["status"]:
            state["status"] = f"Dang xac nhan PCB {self.tracker.confirm}/{BOARD_CONFIRM_FRAMES}"
        return state


def rectify_board(frame, corners):
    """Cắt PCB theo bốn góc (toạ độ khung hình gốc) và hiệu chỉnh phối cảnh.
    Trả về (ảnh cắt, ma trận khung hình -> ảnh cắt)."""
    corners = np.asarray(corners, dtype=np.float32)
    if (corners.shape != (4, 2) or not np.all(np.isfinite(corners))
            or not _quad_ok(corners) or not _corners_inside(corners, frame.shape)):
        return None, None

    top_left, top_right, bottom_right, bottom_left = corners
    width = int(max(np.linalg.norm(bottom_right - bottom_left),
                    np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right),
                     np.linalg.norm(top_left - bottom_left)))
    if width < 1 or height < 1:
        return None, None

    destination = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, destination)
    if not np.all(np.isfinite(transform)):
        return None, None
    board = cv2.warpPerspective(frame, transform, (width, height))
    if board.size == 0:
        return None, None

    if BOARD_AUTO_LANDSCAPE and height > width:          # luôn xuất ảnh nằm ngang
        board = cv2.rotate(board, cv2.ROTATE_90_CLOCKWISE)
        rotation = np.array([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
                            dtype=np.float64)
        transform = rotation @ transform                 # giữ đúng cho _paste_on_frame
    return board, transform


def draw_board_overlay(live, board, sx, sy):
    """Vẽ viền PCB và trạng thái phát hiện lên video trực tiếp."""
    scale = np.array([sx, sy], dtype=np.float32)
    if board["raw_quad"] is not None:
        cv2.polylines(live, [(board["raw_quad"] * scale).astype(np.int32)],
                      True, UI_COLORS["info"], 1, cv2.LINE_AA)
    if board["ready"]:
        cv2.polylines(live, [(board["corners"] * scale).astype(np.int32)],
                      True, UI_COLORS["pass"], 2, cv2.LINE_AA)
        text, color = "PCB: San sang - nhan s de quet", UI_COLORS["pass"]
    else:
        text, color = "PCB: " + board["status"], UI_COLORS["warning"]
    cv2.putText(live, text, (10, live.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)


def dump_board_debug(frame, board):
    """Lưu ảnh đầu vào và dữ liệu trung gian để chẩn đoán lỗi crop."""
    folder = PROJECT_DIR / BOARD_DEBUG_DIR / time.strftime("board_%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    images = {
        "frame.png": frame,
        "edges.png": board["edges"],
        "mask.png": board["mask"],
    }
    for filename, image in images.items():
        if not cv2.imwrite(str(folder / filename), image):
            raise OSError(f"Không lưu được dữ liệu debug: {folder / filename}")

    with (folder / "info.txt").open("w", encoding="utf-8") as debug_file:
        debug_file.write(
            f"frame_shape: {frame.shape}\n"
            f"status: {board['status']}\n"
            f"source: {board['source']}\n"
            f"rejects: {board['rejects']}\n"
        )
        for key, value in board["info"].items():
            formatted = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
            debug_file.write(f"{key}: {formatted}\n")
    return folder


# ============================================================
# 6. CĂN CHỈNH (ORB + AFFINE THEO LINH KIỆN)
# ============================================================

def is_valid_homography(H, ref_shape, frame_shape):
    """Kiểm tra H có cho ra hình dạng PCB hợp lý trong khung hình không."""
    rh, rw = ref_shape[:2]
    fh, fw = frame_shape[:2]

    if not np.all(np.isfinite(H)):
        return False
    if abs(np.linalg.det(H[:2, :2])) < 1e-6:
        return False

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return False

    corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2).astype(np.float32)

    if not np.all(np.isfinite(quad)) or not cv2.isContourConvex(quad):
        return False

    area = cv2.contourArea(quad)
    frame_area = float(fw * fh)
    return MIN_BOARD_AREA_RATIO * frame_area < area < MAX_BOARD_AREA_RATIO * frame_area


class OrbAligner:
    """Giữ sẵn ORB + matcher của ảnh reference để dùng lại ở mỗi lần quét."""

    def __init__(self, reference_image):
        self.ref_shape = reference_image.shape
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        gray = cv2.equalizeHist(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY))
        self.ref_kp, self.ref_desc = self.orb.detectAndCompute(gray, None)

        if self.ref_desc is None:
            raise RuntimeError("Không tạo được ORB descriptor cho ảnh reference.")
        if len(self.ref_kp) < MIN_GOOD_MATCHES:
            raise RuntimeError(
                f"Reference chỉ có {len(self.ref_kp)} keypoints. "
                "Không đủ để căn chỉnh PCB."
            )

    def compute_homography(self, frame):
        """Trả về H (khung hình -> ảnh gốc). None nếu không thấy PCB."""
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frame_kp, frame_desc = self.orb.detectAndCompute(gray, None)

        if frame_desc is None or frame_kp is None or len(frame_kp) < MIN_GOOD_MATCHES:
            return None

        try:
            matches = self.matcher.knnMatch(frame_desc, self.ref_desc, k=2)
        except cv2.error:
            return None

        good = [pair[0] for pair in matches
                if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance]
        if len(good) < MIN_GOOD_MATCHES:
            return None

        src_pts = np.float32([frame_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
        if H is None or mask is None:
            return None

        inliers = int(mask.sum())
        if inliers < MIN_INLIERS or inliers / len(good) < MIN_INLIER_RATIO:
            return None

        if not is_valid_homography(H, self.ref_shape, frame.shape):
            return None
        return H


def estimate_component_alignment(reference_components, detections, tolerance):
    """Căn chỉnh affine dựa trên các linh kiện chỉ xuất hiện đúng 1 lần mỗi loại."""
    reference_by_class = {}
    detections_by_class = {}

    for component in reference_components:
        reference_by_class.setdefault(
            normalize_component_class(component["class"]), []).append(component)
    for detection in detections:
        if detection["confidence"] >= CONF:
            detections_by_class.setdefault(
                normalize_component_class(detection["class"]), []).append(detection)

    source_points, reference_points = [], []
    for class_name, reference_items in reference_by_class.items():
        class_detections = detections_by_class.get(class_name, [])
        if len(reference_items) != 1 or len(class_detections) != 1:
            continue
        source_points.append([class_detections[0]["cx"], class_detections[0]["cy"]])
        reference_points.append(
            [float(reference_items[0]["x"]), float(reference_items[0]["y"])])

    if len(source_points) < 3:
        return None

    src = np.float32(source_points)
    dst = np.float32(reference_points)
    affine, inlier_mask = cv2.estimateAffine2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=tolerance,
        maxIters=5000,
        confidence=0.99,
        refineIters=20,
    )
    if affine is None or inlier_mask is None or int(inlier_mask.sum()) < 3:
        return None

    keep = inlier_mask.ravel() == 1
    projected = cv2.transform(src[keep].reshape(-1, 1, 2), affine).reshape(-1, 2)
    mean_error = float(np.linalg.norm(projected - dst[keep], axis=1).mean())
    if mean_error > tolerance:
        return None

    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = affine
    return homography


def attach_reference_box_sizes(reference_components, model, reference_image):
    reference_detections = detect_components(model, adjust_contrast(reference_image))
    if not reference_detections:
        return

    det_xy = np.array([[d["cx"], d["cy"]] for d in reference_detections])
    det_wh = np.array([[d["x2"] - d["x1"], d["y2"] - d["y1"]]
                       for d in reference_detections])
    max_distance = 0.08 * max(reference_image.shape[:2])

    for component in reference_components:
        point = np.array([float(component["x"]), float(component["y"])])
        distances = np.hypot(*(det_xy - point).T)
        nearest = int(np.argmin(distances))
        if distances[nearest] > max_distance:
            continue
        component["box_width"] = max(1.0, float(det_wh[nearest, 0]))
        component["box_height"] = max(1.0, float(det_wh[nearest, 1]))


# ============================================================
# 7. GHÉP LINH KIỆN GỐC <-> CAMERA
# ============================================================

def match_components(reference_components, detections, tolerance,
                     require_same_class=True):
    """Ghép tối đa (augmenting path), tính ma trận khoảng cách bằng numpy."""
    num_ref, num_det = len(reference_components), len(detections)
    if num_ref == 0 or num_det == 0:
        return [], set(), set()

    ref_xy = np.array([[float(c["x"]), float(c["y"])] for c in reference_components])
    det_xy = np.array([[d["ref_x"], d["ref_y"]] for d in detections])
    det_box = np.array([d["ref_bbox"] for d in detections])  # (D, 4)

    # Ma trận khoảng cách (R, D)
    dist = np.hypot(ref_xy[:, None, 0] - det_xy[None, :, 0],
                    ref_xy[:, None, 1] - det_xy[None, :, 1])

    # Tâm linh kiện gốc nằm trong bbox đã ánh xạ
    rx, ry = ref_xy[:, 0:1], ref_xy[:, 1:2]
    contains = ((det_box[None, :, 0] <= rx) & (rx <= det_box[None, :, 2])
                & (det_box[None, :, 1] <= ry) & (ry <= det_box[None, :, 3]))

    # Loại bbox quá lớn so với kích thước kỳ vọng (NaN -> so sánh False -> giữ nguyên)
    exp_w = np.array([c.get("box_width", np.nan) for c in reference_components],
                     dtype=np.float64)[:, None]
    exp_h = np.array([c.get("box_height", np.nan) for c in reference_components],
                     dtype=np.float64)[:, None]
    box_w = (det_box[:, 2] - det_box[:, 0])[None, :]
    box_h = (det_box[:, 3] - det_box[:, 1])[None, :]
    too_big = (box_w > 3.0 * exp_w) | (box_h > 3.0 * exp_h)
    contains &= ~too_big

    allowed = (dist <= tolerance) | contains
    if require_same_class:
        ref_cls = np.array([normalize_component_class(c["class"])
                            for c in reference_components])
        det_cls = np.array([normalize_component_class(d["class"]) for d in detections])
        allowed &= ref_cls[:, None] == det_cls[None, :]

    cost = np.where(contains, np.minimum(dist, tolerance), dist)

    candidates = []
    for r in range(num_ref):
        idx = np.flatnonzero(allowed[r])
        candidates.append(idx[np.argsort(cost[r, idx], kind="stable")].tolist())

    detection_to_reference = {}

    def assign(reference_index, visited):
        for det_index in candidates[reference_index]:
            if det_index in visited:
                continue
            visited.add(det_index)
            previous = detection_to_reference.get(det_index)
            if previous is None or assign(previous, visited):
                detection_to_reference[det_index] = reference_index
                return True
        return False

    for ref_index in sorted(range(num_ref), key=lambda i: (len(candidates[i]), i)):
        assign(ref_index, set())

    matches = [
        {"det_index": det_index, "ref_index": ref_index,
         "distance": float(cost[ref_index, det_index])}
        for det_index, ref_index in detection_to_reference.items()
    ]
    matches.sort(key=lambda m: m["ref_index"])
    return matches, set(detection_to_reference), set(detection_to_reference.values())


# ============================================================
# 8. ĐÁNH GIÁ MẠCH
# ============================================================

def evaluate_board_status(reference_components, detections, matches,
                          used_references, aligned):
    stats = {
        "matched": len(matches),
        "missing": 0,
        "extra": 0,
        "missing_components": [],
        "extra_components": [],
    }

    if aligned:
        matched_detections = {m["det_index"] for m in matches}
        stats["missing"] = len(reference_components) - len(used_references)
        stats["missing_components"] = [
            c["class"] for i, c in enumerate(reference_components)
            if i not in used_references
        ]
        stats["extra_components"] = [
            d["class"] for i, d in enumerate(detections)
            if i not in matched_detections and d["confidence"] >= EXTRA_MIN_CONF
        ]
        stats["extra"] = len(stats["extra_components"])
        if stats["missing"] > MISSING_ALLOWED:
            return STATUS_FAULTY, stats
        return STATUS_OK, stats

    return STATUS_UNALIGNED, stats


# ============================================================
# 9. VẼ KẾT QUẢ
# ============================================================

def draw_results(frame, detections, reference_components, matches,
                 used_references, aligned, H, status, stats, show_header=True):
    output = frame.copy()
    matched_detections = {m["det_index"] for m in matches}

    for det_index, det in enumerate(detections):
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])

        if not aligned:
            color, tag = UI_COLORS["info"], "DETECTED"
        elif det_index in matched_detections:
            color, tag = UI_COLORS["pass"], "OK"
        elif det["confidence"] >= EXTRA_MIN_CONF:
            color, tag = UI_COLORS["faulty"], "EXTRA"
        else:
            color, tag = UI_COLORS["warning"], "LOW"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, f'{det["class"]} {det["confidence"]:.2f} {tag}',
                    (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, color, 2, cv2.LINE_AA)

    if aligned and H is not None and stats["missing"] > 0:
        missing_idx = [i for i in range(len(reference_components))
                       if i not in used_references]
        try:
            H_inverse = np.linalg.inv(H)
            pts = np.float32([[float(reference_components[i]["x"]),
                               float(reference_components[i]["y"])]
                              for i in missing_idx]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(pts, H_inverse).reshape(-1, 2)
            h, w = output.shape[:2]
            for i, (px, py) in zip(missing_idx, projected):
                px, py = int(px), int(py)
                if not (0 <= px < w and 0 <= py < h):
                    continue
                color = UI_COLORS["faulty"]
                cv2.circle(output, (px, py), 10, color, 2)
                cv2.putText(output, f'MISSING {reference_components[i]["class"]}',
                            (px + 12, py), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 2, cv2.LINE_AA)
        except np.linalg.LinAlgError:
            pass

    if show_header:
        draw_header(output, status, stats, aligned)
    return output


def draw_header(output, status, stats, aligned):
    header_width = min(output.shape[1], 520)
    cv2.rectangle(output, (0, 0), (header_width - 1, 79),
                  UI_COLORS["background"], -1)
    cv2.rectangle(output, (0, 0), (header_width - 1, 79),
                  UI_COLORS["border"], 1)
    cv2.rectangle(output, (0, 0), (4, 79), STATUS_COLOR[status], -1)
    cv2.putText(output, STATUS_TEXT_IMAGE[status], (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, STATUS_COLOR[status], 3, cv2.LINE_AA)
    if aligned:
        info = (f'Position check: Matched: {stats["matched"]}  '
                f'Missing: {stats["missing"]}  Extra: {stats["extra"]}')
    else:
        info = f'Components detected: {stats["detected"]}  QA unavailable: alignment failed'
    cv2.putText(output, info, (10, 68), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, UI_COLORS["secondary"], 2, cv2.LINE_AA)

    details = []
    if stats["missing_components"]:
        details.append(f'Missing: {", ".join(stats["missing_components"])}')
    if stats["extra_components"]:
        details.append(f'Extra: {", ".join(stats["extra_components"])}')
    if details:
        details_width = min(output.shape[1], 900)
        cv2.rectangle(output, (0, 80), (details_width - 1, 112),
                      UI_COLORS["panel"], -1)
        cv2.rectangle(output, (0, 80), (details_width - 1, 112),
                      UI_COLORS["border"], 1)
        cv2.putText(output, " | ".join(details), (10, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    UI_COLORS["text"], 2, cv2.LINE_AA)

    return output


def draw_status_overlay(image, status):
    output = image.copy()
    banner_width = min(output.shape[1], 520)
    cv2.rectangle(output, (0, 0), (banner_width - 1, 48),
                  UI_COLORS["background"], -1)
    cv2.rectangle(output, (0, 0), (banner_width - 1, 48),
                  UI_COLORS["border"], 1)
    cv2.rectangle(output, (0, 0), (4, 48), STATUS_COLOR[status], -1)
    cv2.putText(output, STATUS_TEXT_IMAGE[status], (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, STATUS_COLOR[status], 3, cv2.LINE_AA)
    return output


def resize_for_display(image):
    h, w = image.shape[:2]
    scale = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


# ============================================================
# 10. MỘT LẦN QUÉT
# ============================================================

class PCBChecker:
    def __init__(self, model, reference_components, reference_image):
        self.model = model
        self.reference_components = reference_components
        ref_h, ref_w = reference_image.shape[:2]
        self.tolerance = MATCH_TOLERANCE_RATIO * max(ref_w, ref_h)

        attach_reference_box_sizes(reference_components, model, reference_image)
        self.aligner = OrbAligner(reference_image) if USE_ORB_ALIGNMENT else None

    @staticmethod
    def _paste_on_frame(frame, annotated, board_transform):
        """Ghép ảnh PCB đã vẽ kết quả trở lại đúng vị trí trên khung hình gốc."""
        h, w = frame.shape[:2]
        inverse = np.linalg.inv(board_transform)
        warped = cv2.warpPerspective(annotated, inverse, (w, h))
        mask = cv2.warpPerspective(
            np.full(annotated.shape[:2], 255, dtype=np.uint8), inverse, (w, h)
        ) > 127
        output = frame.copy()
        output[mask] = warped[mask]
        return output

    @staticmethod
    def _map_detections(detections, H):
        """Ánh xạ tâm và bbox của toàn bộ detection sang toạ độ reference (1 lần gọi)."""
        if not detections:
            return []

        centers = np.float32([[d["cx"], d["cy"]] for d in detections]).reshape(-1, 1, 2)
        ref_centers = cv2.perspectiveTransform(centers, H).reshape(-1, 2)

        box_corners = np.float32([
            [[d["x1"], d["y1"]], [d["x2"], d["y1"]],
             [d["x2"], d["y2"]], [d["x1"], d["y2"]]]
            for d in detections
        ]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(box_corners, H).reshape(-1, 4, 2)
        mins, maxs = mapped.min(axis=1), mapped.max(axis=1)

        transformed = []
        for i, det in enumerate(detections):
            d = det.copy()
            d["ref_x"], d["ref_y"] = float(ref_centers[i, 0]), float(ref_centers[i, 1])
            d["ref_bbox"] = (float(mins[i, 0]), float(mins[i, 1]),
                             float(maxs[i, 0]), float(maxs[i, 1]))
            transformed.append(d)
        return transformed

    def scan(self, frame, board_corners):
        board_crop, board_transform = rectify_board(frame, board_corners)
        if board_crop is None:
            raise RuntimeError("Không cắt được PCB từ bốn góc đã xác nhận.")
        detection_frame = adjust_contrast(board_crop)  # dùng chung cho YOLO và ORB
        detections = detect_components(self.model, detection_frame)

        H = estimate_component_alignment(
            self.reference_components, detections, self.tolerance)
        if H is None and self.aligner is not None:
            H = self.aligner.compute_homography(detection_frame)
        aligned = H is not None

        if aligned:
            shown = self._map_detections(detections, H)
            matches, _, used_references = match_components(
                self.reference_components, shown, self.tolerance,
                require_same_class=False,
            )
        else:
            shown, matches, used_references = detections, [], set()

        status, stats = evaluate_board_status(
            self.reference_components, shown, matches, used_references, aligned)
        stats["detected"] = len(detections)

        annotated = draw_results(
            board_crop, shown, self.reference_components, matches,
            used_references, aligned, H, status, stats,
            show_header=not SHOW_FULL_FRAME_RESULT,
        )
        if SHOW_FULL_FRAME_RESULT:
            annotated = self._paste_on_frame(frame, annotated, board_transform)
            draw_header(annotated, status, stats, aligned)
        return {"status": status, "annotated": annotated, "stats": stats}


# ============================================================
# 11. QUÉT Ở LUỒNG NỀN
# ============================================================

class Scanner:
    def __init__(self, checker):
        self.checker = checker
        self.lock = threading.Lock()
        self.busy = False
        self.result = None
        self.error = None

    def start(self, frame, board_corners):
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            self.error = None
        threading.Thread(target=self._run, args=(frame.copy(), board_corners.copy()),
                         daemon=True).start()
        return True

    def _run(self, frame, board_corners):
        result = None
        error = None
        try:
            result = self.checker.scan(frame, board_corners)
        except Exception as e:
            error = str(e)
            print(f"[ERROR] {e}", flush=True)
        finally:
            with self.lock:
                self.result = result
                self.error = error
                self.busy = False

    def pop_result(self):
        with self.lock:
            result, self.result = self.result, None
        return result

    def pop_error(self):
        with self.lock:
            error, self.error = self.error, None
        return error

    def is_busy(self):
        with self.lock:
            return self.busy


def stable_status(history):
    """Lấy trạng thái chiếm đa số trong các lần quét gần nhất."""
    if len(history) == 1:
        return history[-1]
    status, count = Counter(history).most_common(1)[0]
    return status if count >= 2 else history[-1]


# ============================================================
# 12. MAIN
# ============================================================

def main():
    os.system("")  # bật màu ANSI trên Windows terminal

    detect_only = DETECT_ONLY or "--detect-only" in sys.argv
    checker = None
    if detect_only:
        print("[INFO] --detect-only: chỉ xem phát hiện/crop PCB, không nạp YOLO.",
              flush=True)
    else:
        if YOLO is None:
            raise RuntimeError(
                f"Thiếu thư viện YOLO/torch: {_IMPORT_ERROR}\n"
                "Cài torch và ultralytics, hoặc chạy: python test.py --detect-only"
            )
        reference_components = load_reference()["components"]
        reference_counts = Counter(c["class"] for c in reference_components)
        print("[INFO] PCB checker v5 - checks detected component positions against reference.",
              flush=True)
        print(f"[INFO] Linh kiện mạch gốc: {dict(reference_counts)}", flush=True)

        reference_image = load_reference_image()
        model = YOLO(find_model())
        checker = PCBChecker(model, reference_components, reference_image)

        # Chạy thử YOLO 1 lần để lần quét đầu không bị khựng.
        model.predict(
            source=np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8),
            imgsz=IMGSZ, device=DEVICE, half=USE_CUDA, verbose=False,
        )

    stream, camera_label = open_camera()      # tự động mở camera khi chạy chương trình

    if not stream.is_opened():
        stream.stop()
        raise RuntimeError(
            "\nKhông mở được camera.\n"
            "Đã thử: camera Hikvision (không thấy) và "
            f"SOURCE = {SOURCE}\n\n"
            "Kiểm tra:\n"
            "1. Camera Hikvision: nguồn, cáp, MVS/MvImport, không bị chương trình khác giữ.\n"
            "2. IP Webcam: điện thoại và máy tính cùng WiFi, IP còn đúng, ứng dụng đang chạy.\n"
            "3. Thử mở URL trên trình duyệt.\n"
            "4. Nếu dùng webcam, SOURCE = 0."
        )

    if not stream.wait_for_frame(CAMERA_STARTUP_TIMEOUT_SECONDS):
        stream.stop()
        raise RuntimeError(
            "\nCamera đã mở nhưng không nhận được hình ảnh.\n"
            f"Camera: {camera_label}\n"
            "Kiểm tra cáp/nguồn và thời gian lấy sáng (HIK_EXPOSURE_US) "
            "hoặc IP/URL của IP Webcam."
        )

    has_exposure = hasattr(stream, "set_exposure")
    exposure_steps = {ord("+"): 10000.0, ord("="): 10000.0, ord("-"): -10000.0,
                      ord("_"): -10000.0, ord("]"): 50000.0, ord("["): -50000.0}
    if detect_only:
        hint_text = ("DETECT-ONLY | d: Debug | q: Thoat | +/-/[/]: Exp | r: Reset"
                     if has_exposure else "DETECT-ONLY | d: Debug | q: Thoat")
    else:
        hint_text = ("s: Quet | d: Debug | q: Thoat | +/-/[/]: Exposure | r: Reset"
                     if has_exposure else "s: Chup va nhan dien | d: Debug | q: Thoat")
    print(f"[INFO] Camera: {camera_label}", flush=True)

    scanner = Scanner(checker) if checker is not None else None
    board_detector = BoardDetector()
    crop_open = False
    board = None
    history = deque(maxlen=max(1, SMOOTHING_WINDOW))
    last_status = None
    scan_notice = None
    last_shown_id = -1

    try:
        while True:
            frame, frame_id = stream.read()

            if frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue

            # ---------- NHẬN KẾT QUẢ ----------
            result = scanner.pop_result() if scanner is not None else None
            if result is not None:
                history.append(result["status"])
                last_status = stable_status(history)
                scan_notice = f"Scan xong: {STATUS_TEXT_IMAGE[last_status]}"
                print(f"{ANSI[last_status]}{STATUS_TEXT_TERMINAL[last_status]}{ANSI_RESET}",
                      flush=True)

                if SHOW_RESULT_WINDOW:
                    annotated = draw_status_overlay(result["annotated"], last_status)
                    cv2.imshow(WINDOW_RESULT, resize_for_display(annotated))

            if scanner is not None:
                scan_error = scanner.pop_error()
                if scan_error:
                    scan_notice = f"Loi nhan dien: {scan_error}"

            # ---------- VIDEO TRỰC TIẾP ----------
            if frame_id != last_shown_id:
                last_shown_id = frame_id

                # Phát hiện PCB trên frame này; phím s dùng đúng kết quả này
                board = board_detector.update(frame)
                if SHOW_BOARD_EDGES or detect_only:
                    debug_view = np.hstack([board["edges"], board["mask"]])
                    cv2.imshow("Board Debug (edges | mask)",
                               resize_for_display(debug_view))

                live = resize_for_display(frame)
                if live is frame:
                    live = frame.copy()
                draw_board_overlay(live, board,
                                   live.shape[1] / board["det_size"][0],
                                   live.shape[0] / board["det_size"][1])
                if scanner is not None and scanner.is_busy():
                    scan_notice = "Dang cat PCB va nhan dien YOLO..."
                if scan_notice:
                    cv2.putText(live, scan_notice[:100], (10, 86),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                UI_COLORS["info"], 2, cv2.LINE_AA)

                cv2.rectangle(live, (0, 0), (live.shape[1], 58),
                              UI_COLORS["background"], -1)
                cv2.line(live, (0, 58), (live.shape[1], 58),
                         UI_COLORS["border"], 1, cv2.LINE_AA)
                cv2.rectangle(live, (0, 0), (4, 58), UI_COLORS["primary"], -1)
                if last_status is not None:
                    cv2.putText(live, f"Last: {STATUS_TEXT_IMAGE[last_status]}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                STATUS_COLOR[last_status], 2, cv2.LINE_AA)
                cv2.putText(live, hint_text, (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_COLORS["secondary"], 2,
                            cv2.LINE_AA)
                if has_exposure:
                    cv2.putText(live, f"Exp: {stream.exposure_us / 1000:.0f} ms",
                                (live.shape[1] - 170, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, UI_COLORS["info"], 2, cv2.LINE_AA)

                if detect_only or SHOW_CROP_PREVIEW:
                    if board["ready"]:
                        crop, _ = rectify_board(frame, board["frame_corners"])
                        if crop is not None:
                            cv2.imshow("PCB Crop", resize_for_display(crop))
                            crop_open = True
                    elif crop_open:
                        cv2.destroyWindow("PCB Crop")
                        crop_open = False

                cv2.imshow(WINDOW_LIVE, live)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(5) & 0xFF

            if key == ord("q"):
                break
            if has_exposure:                      # chỉnh thời gian lấy sáng (camera Hikvision)
                if key in exposure_steps:
                    stream.set_exposure(max(HIK_EXPOSURE_MIN_US,
                                            stream.exposure_us + exposure_steps[key]))
                elif key in (ord("r"), ord("R")):
                    stream.set_exposure(HIK_EXPOSURE_US)
            if key in (ord("d"), ord("D")) and board is not None:
                folder = dump_board_debug(frame, board)
                print(f"[INFO] Da luu du lieu debug phat hien PCB: {folder}", flush=True)
            if key in (ord("s"), ord("S")) and detect_only:
                print("[INFO] --detect-only: khong chay nhan dien YOLO.", flush=True)
            elif key in (ord("s"), ord("S")):
                if board is None or not board["ready"]:
                    reason = board["status"] if board else "chua co khung hinh"
                    print(f"[WARN] Khong quet: {reason}", flush=True)
                elif scanner is not None and scanner.start(
                    frame, board["frame_corners"].copy()
                ):
                    scan_notice = "Dang cat PCB va nhan dien YOLO..."
                    print("[INFO] Da chup anh, dang cat PCB va nhan dien...", flush=True)
                elif scanner is not None and scanner.is_busy():
                    print("[INFO] Dang co lan quet chay; vui long cho ket qua.", flush=True)
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[ERROR] {e}")
        cv2.destroyAllWindows()
        raise
import json
import os
import sys
import threading
import time
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

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
    else (PROJECT_DIR / "2.jpg")
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

# ---- Nhận diện ----
SMOOTHING_WINDOW = 1
CONTRAST_ALPHA = 1.15
CONTRAST_BETA = -10

# ---- YOLO (PHẢI GIỐNG machpcbgoc.py) ----
CONF = 0.25
IMGSZ = 1536
MAX_DET = 1000
USE_CUDA = torch.cuda.is_available()
DEVICE = 0 if USE_CUDA else "cpu"

# ---- So sánh vị trí ----
MATCH_TOLERANCE_RATIO = 0.045
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

# ---- Cắt khung PCB ----
BOARD_DETECTION_SIZE = (640, 240)
BOARD_CONTOUR_MIN_AREA = 20000
BOARD_PADDING_RATIO = 0.05

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
# ============================================================

def order_points(points):
    """Sắp xếp bốn góc: trên-trái, trên-phải, dưới-phải, dưới-trái."""
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = np.argmin(ordered.sum(axis=1))
    return np.roll(ordered, -start, axis=0)


_BOARD_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def find_board_corners(frame):
    """Tìm khung PCB trên ảnh thu nhỏ, trả về các góc theo kích thước ảnh gốc."""
    frame_height, frame_width = frame.shape[:2]
    detect_width, detect_height = BOARD_DETECTION_SIZE
    resized = cv2.resize(frame, BOARD_DETECTION_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _BOARD_KERNEL, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area = BOARD_CONTOUR_MIN_AREA
    best_corners = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= best_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        corners = order_points(approx.reshape(4, 2))
        edge_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        if edge_lengths.min() < 1:
            continue

        # Chừa viền nhỏ để giữ trọn mép PCB trong ảnh crop.
        scale = 1 + 2 * BOARD_PADDING_RATIO
        center = corners.mean(axis=0)
        corners = center + (corners - center) * scale
        corners[:, 0] = np.clip(corners[:, 0], 0, detect_width - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, detect_height - 1)
        corners[:, 0] *= frame_width / detect_width
        corners[:, 1] *= frame_height / detect_height

        best_area = area
        best_corners = corners

    return best_corners


def rectify_board(frame, corners):
    """Cắt PCB theo bốn góc và hiệu chỉnh phối cảnh."""
    top_left, top_right, bottom_right, bottom_left = corners
    width = int(max(
        np.linalg.norm(bottom_right - bottom_left),
        np.linalg.norm(top_right - top_left),
    ))
    height = int(max(
        np.linalg.norm(top_right - bottom_right),
        np.linalg.norm(top_left - bottom_left),
    ))
    if width < 1 or height < 1:
        return None, None

    destination = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(frame, transform, (width, height)), transform


_ROI_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
_ROI_OPEN_KERNEL = np.ones((5, 5), dtype=np.uint8)
_BLUE_LOW = np.array([85, 35, 20], dtype=np.uint8)
_BLUE_HIGH = np.array([145, 255, 255], dtype=np.uint8)


def find_board_roi(frame):
    """Tìm ROI PCB màu xanh để dùng dự phòng khi không tìm được bốn góc."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, _BLUE_LOW, _BLUE_HIGH)
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_CLOSE, _ROI_CLOSE_KERNEL, iterations=2
    )
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_OPEN, _ROI_OPEN_KERNEL, iterations=1
    )
    contours, _ = cv2.findContours(
        blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    frame_height, frame_width = frame.shape[:2]
    if (cv2.contourArea(contour) < frame_height * frame_width * 0.005
            or width < 80 or height < 80):
        return None

    pad_x = int(width * BOARD_PADDING_RATIO)
    pad_y = int(height * BOARD_PADDING_RATIO)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2 = min(frame_width, x + width + pad_x)
    y2 = min(frame_height, y + height + pad_y)
    return x1, y1, x2, y2


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

    def _crop_board(self, frame):
        corners = find_board_corners(frame)
        rectified, transform = (rectify_board(frame, corners)
                                if corners is not None else (None, None))
        if rectified is not None:
            return rectified, transform

        roi = find_board_roi(frame)
        if roi is None:
            raise RuntimeError("Không tìm thấy PCB trong ảnh vừa chụp; bỏ qua lần quét.")
        x1, y1, x2, y2 = roi
        translation = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], dtype=np.float64)
        return frame[y1:y2, x1:x2], translation

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

    def scan(self, frame):
        board_crop, board_transform = self._crop_board(frame)
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

    def start(self, frame):
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        # Frame không bị ghi đè (mỗi lần read là mảng mới) nên không cần copy.
        threading.Thread(target=self._run, args=(frame,), daemon=True).start()
        return True

    def _run(self, frame):
        result = None
        try:
            result = self.checker.scan(frame)
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
        finally:
            with self.lock:
                self.result = result
                self.busy = False

    def pop_result(self):
        with self.lock:
            result, self.result = self.result, None
        return result


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

    stream = CameraStream(SOURCE)

    if not stream.is_opened():
        stream.stop()
        raise RuntimeError(
            "\nKhông mở được camera.\n"
            f"SOURCE = {SOURCE}\n\n"
            "Kiểm tra:\n"
            "1. Điện thoại và máy tính cùng WiFi.\n"
            "2. IP điện thoại còn đúng.\n"
            "3. IP Webcam đang chạy.\n"
            "4. Thử mở URL trên trình duyệt.\n"
            "5. Nếu dùng webcam, SOURCE = 0."
        )

    if not stream.wait_for_frame(CAMERA_STARTUP_TIMEOUT_SECONDS):
        stream.stop()
        raise RuntimeError(
            "\nCamera đã mở nhưng không nhận được hình ảnh.\n"
            f"SOURCE = {SOURCE}\n"
            "Kiểm tra IP/URL, đảm bảo IP Webcam đang phát video và "
            "điện thoại với máy tính cùng mạng."
        )

    scanner = Scanner(checker)
    history = deque(maxlen=max(1, SMOOTHING_WINDOW))
    last_status = None
    last_shown_id = -1

    try:
        while True:
            frame, frame_id = stream.read()

            if frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue

            # ---------- NHẬN KẾT QUẢ ----------
            result = scanner.pop_result()
            if result is not None:
                history.append(result["status"])
                last_status = stable_status(history)
                print(f"{ANSI[last_status]}{STATUS_TEXT_TERMINAL[last_status]}{ANSI_RESET}",
                      flush=True)

                if SHOW_RESULT_WINDOW:
                    annotated = draw_status_overlay(result["annotated"], last_status)
                    cv2.imshow(WINDOW_RESULT, resize_for_display(annotated))

            # ---------- VIDEO TRỰC TIẾP ----------
            if frame_id != last_shown_id:
                last_shown_id = frame_id

                live = resize_for_display(frame)
                if live is frame:
                    live = frame.copy()

                cv2.rectangle(live, (0, 0), (live.shape[1], 58),
                              UI_COLORS["background"], -1)
                cv2.line(live, (0, 58), (live.shape[1], 58),
                         UI_COLORS["border"], 1, cv2.LINE_AA)
                cv2.rectangle(live, (0, 0), (4, 58), UI_COLORS["primary"], -1)
                if last_status is not None:
                    cv2.putText(live, f"Last: {STATUS_TEXT_IMAGE[last_status]}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                STATUS_COLOR[last_status], 2, cv2.LINE_AA)
                cv2.putText(live, "s: Chup va nhan dien | q: Thoat", (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_COLORS["secondary"], 2,
                            cv2.LINE_AA)

                cv2.imshow(WINDOW_LIVE, live)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(5) & 0xFF

            if key == ord("q"):
                break
            if key in (ord("s"), ord("S")):
                if scanner.start(frame):
                    print("[INFO] Da chup anh, dang cat PCB va nhan dien...", flush=True)
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
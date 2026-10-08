import json
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# 1. CẤU HÌNH
# ============================================================

MODEL_CANDIDATES = ["best.pt"]

REFERENCE_JSON = "reference_components.json"
REFERENCE_IMAGE = "2.jpg"

# Camera điện thoại Android IP Webcam (hoặc SOURCE = 0 cho webcam máy tính)
SOURCE_VALUE = os.environ.get(
    "PCB_CAMERA_SOURCE", "http://192.168.0.190:8080/video"
)
try:
    SOURCE = int(SOURCE_VALUE)
except ValueError:
    SOURCE = SOURCE_VALUE
CAMERA_STARTUP_TIMEOUT_SECONDS = 10

# ---- Nhận diện ----
SMOOTHING_WINDOW = 1            # phản hồi ngay theo kết quả quét mới nhất
CONTRAST_ALPHA = 1.15           # tăng tương phản nhẹ khi xử lý ảnh nhận diện
CONTRAST_BETA = -10

# ---- YOLO (PHẢI GIỐNG machpcbgoc.py) ----
CONF = 0.25
IMGSZ = 1536
MAX_DET = 1000
USE_CUDA = torch.cuda.is_available()
DEVICE = 0 if USE_CUDA else "cpu"

# ---- So sánh vị trí ----
MATCH_TOLERANCE_RATIO = 0.045   # sai số vị trí sau căn chỉnh
MISSING_ALLOWED = 0             # cho phép thiếu tối đa bao nhiêu linh kiện
EXTRA_ALLOWED = 2               # cho phép thừa tối đa bao nhiêu linh kiện
EXTRA_MIN_CONF = 0.50           # box thừa phải có conf >= giá trị này mới tính là thừa

# ---- ORB ----
USE_ORB_ALIGNMENT = True
ORB_FEATURES = 2000
RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 10
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.40
RANSAC_THRESHOLD = 5.0
MIN_BOARD_AREA_RATIO = 0.03     # PCB chiếm tối thiểu 3% khung hình
MAX_BOARD_AREA_RATIO = 2.00

# ---- Cắt khung PCB ----
BOARD_DETECTION_SIZE = (640, 320)
BOARD_CONTOUR_MIN_AREA = 20000
BOARD_PADDING_RATIO = 0.05

# ---- Hiển thị ----
WINDOW_LIVE = "PCB Camera"
WINDOW_RESULT = "PCB Scan Result"
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 480
SHOW_RESULT_WINDOW = True

# ---- Trạng thái ----
STATUS_FAULTY = "FAULTY"
STATUS_OK = "OK"

STATUS_TEXT_TERMINAL = {
    STATUS_FAULTY: "MẠCH BỊ LỖI",
    STATUS_OK: "MẠCH ĐẠT CHUẨN",
}
STATUS_TEXT_IMAGE = {
    STATUS_FAULTY: "PCB FAULTY",
    STATUS_OK: "PCB OK",
}
STATUS_COLOR = {                       # BGR
    STATUS_FAULTY: (0, 0, 255),
    STATUS_OK: (0, 255, 0),
}
ANSI = {
    STATUS_FAULTY: "\033[91m",
    STATUS_OK: "\033[92m",
}
ANSI_RESET = "\033[0m"

CLASS_ALIASES = {
    "potentiometer": "variable resistor",
    "variable_resistor": "variable resistor",
}


def normalize_component_class(class_name):
    normalized = " ".join(
        str(class_name).casefold().replace("-", " ").replace("_", " ").split()
    )
    return CLASS_ALIASES.get(normalized, normalized)


# ============================================================
# 2. ĐỌC CAMERA Ở LUỒNG RIÊNG
# ============================================================

class CameraStream:
    """Luôn đọc frame mới nhất trong nền."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True

        if self.cap.isOpened():
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def is_opened(self):
        return self.cap.isOpened()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self):
        with self.lock:
            return self.frame, self.frame_id

    def wait_for_frame(self, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            frame, _ = self.read()
            if frame is not None:
                return True
            time.sleep(0.02)
        return False

    def stop(self):
        self.running = False
        time.sleep(0.1)
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
# 4. ORB + HOMOGRAPHY
# ============================================================

def build_reference_orb(reference_image):
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None:
        raise RuntimeError("Không tạo được ORB descriptor cho ảnh reference.")
    if len(keypoints) < MIN_GOOD_MATCHES:
        raise RuntimeError(
            f"Reference chỉ có {len(keypoints)} keypoints. Không đủ để căn chỉnh PCB."
        )
    return orb, keypoints, descriptors


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

    if not np.all(np.isfinite(quad)):
        return False
    if not cv2.isContourConvex(quad):
        return False

    area = cv2.contourArea(quad)
    frame_area = float(fw * fh)
    return MIN_BOARD_AREA_RATIO * frame_area < area < MAX_BOARD_AREA_RATIO * frame_area


def compute_homography(ref_kp, ref_desc, ref_shape, frame, orb):
    """Trả về H (khung hình -> ảnh gốc). None nếu không thấy PCB."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    frame_kp, frame_desc = orb.detectAndCompute(gray, None)

    if frame_desc is None or frame_kp is None:
        return None
    if len(frame_kp) < MIN_GOOD_MATCHES:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = matcher.knnMatch(frame_desc, ref_desc, k=2)
    except cv2.error:
        return None

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_TEST * n.distance:
            good.append(m)

    if len(good) < MIN_GOOD_MATCHES:
        return None

    src_pts = np.float32([frame_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    if H is None or mask is None:
        return None

    inliers = int(mask.ravel().sum())
    if inliers < MIN_INLIERS or inliers / len(good) < MIN_INLIER_RATIO:
        return None

    if not is_valid_homography(H, ref_shape, frame.shape):
        return None

    return H


def transform_point_to_reference(x, y, H):
    point = np.array([[[float(x), float(y)]]], dtype=np.float32)
    rx, ry = cv2.perspectiveTransform(point, H)[0, 0]
    return float(rx), float(ry)


def estimate_component_alignment(reference_components, detections, tolerance):
    reference_by_class = {}
    detections_by_class = {}

    for component in reference_components:
        class_name = normalize_component_class(component["class"])
        reference_by_class.setdefault(class_name, []).append(component)
    for detection in detections:
        class_name = normalize_component_class(detection["class"])
        if detection["confidence"] >= CONF:
            detections_by_class.setdefault(class_name, []).append(detection)

    source_points = []
    reference_points = []
    for class_name, reference_items in reference_by_class.items():
        class_detections = detections_by_class.get(class_name, [])
        if len(reference_items) != 1 or len(class_detections) != 1:
            continue

        source_points.append(
            [class_detections[0]["cx"], class_detections[0]["cy"]]
        )
        reference_points.append(
            [float(reference_items[0]["x"]), float(reference_items[0]["y"])]
        )

    if len(source_points) < 3:
        return None

    affine, inlier_mask = cv2.estimateAffine2D(
        np.float32(source_points),
        np.float32(reference_points),
        method=cv2.RANSAC,
        ransacReprojThreshold=tolerance,
        maxIters=5000,
        confidence=0.99,
        refineIters=20,
    )
    if affine is None or inlier_mask is None:
        return None
    if int(inlier_mask.sum()) < 3:
        return None

    inlier_source = np.float32(source_points)[inlier_mask.ravel() == 1]
    inlier_reference = np.float32(reference_points)[inlier_mask.ravel() == 1]
    projected = cv2.transform(inlier_source.reshape(-1, 1, 2), affine).reshape(-1, 2)
    mean_error = float(np.linalg.norm(projected - inlier_reference, axis=1).mean())
    if mean_error > tolerance:
        return None

    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = affine
    return homography


def attach_reference_box_sizes(reference_components, model, reference_image):
    detection_image = cv2.convertScaleAbs(
        reference_image, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA
    )
    reference_detections = detect_components(model, detection_image)

    for component in reference_components:
        nearest = min(
            reference_detections,
            key=lambda detection: np.hypot(
                detection["cx"] - float(component["x"]),
                detection["cy"] - float(component["y"]),
            ),
            default=None,
        )
        if nearest is None:
            continue

        distance = np.hypot(
            nearest["cx"] - float(component["x"]),
            nearest["cy"] - float(component["y"]),
        )
        if distance > 0.08 * max(reference_image.shape[:2]):
            continue

        component["box_width"] = max(1.0, nearest["x2"] - nearest["x1"])
        component["box_height"] = max(1.0, nearest["y2"] - nearest["y1"])


# ============================================================
# 5. PHÁT HIỆN VÀ CẮT KHUNG PCB + YOLO
# ============================================================

def order_points(points):
    """Sắp xếp bốn góc theo thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái."""
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = np.argmin(ordered.sum(axis=1))
    return np.roll(ordered, -start, axis=0)


def find_board_corners(frame):
    """Tìm khung PCB trên ảnh thu nhỏ, trả về các góc theo kích thước ảnh gốc."""
    frame_height, frame_width = frame.shape[:2]
    detect_width, detect_height = BOARD_DETECTION_SIZE
    resized = cv2.resize(frame, BOARD_DETECTION_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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
        edge_lengths = np.linalg.norm(
            np.roll(corners, -1, axis=0) - corners, axis=1
        )
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
        return None

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(frame, transform, (width, height))


def find_board_roi(frame):
    """Tìm ROI PCB màu xanh để dùng dự phòng khi không tìm được bốn góc."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(
        hsv,
        np.array([85, 35, 20], dtype=np.uint8),
        np.array([145, 255, 255], dtype=np.uint8),
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2
    )
    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
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


def detect_components(model, frame, origin=(0, 0)):
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

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes
    detections = []

    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[idx].detach().cpu().numpy().tolist()
        x1 += origin[0]
        x2 += origin[0]
        y1 += origin[1]
        y2 += origin[1]
        cls_id = int(boxes.cls[idx].item())
        detections.append({
            "id": idx + 1,
            "class_id": cls_id,
            "class": normalize_component_class(model.names[cls_id]),
            "confidence": float(boxes.conf[idx].item()),
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "cx": float((x1 + x2) / 2.0),
            "cy": float((y1 + y2) / 2.0),
        })

    return detections


# ============================================================
# 6. GHÉP LINH KIỆN GỐC <-> CAMERA
# ============================================================

def match_components(reference_components, detections, tolerance,
                     require_same_class=True):
    candidates_by_reference = [[] for _ in reference_components]
    for ref_index, ref in enumerate(reference_components):
        for det_index, det in enumerate(detections):
            if (require_same_class
                    and normalize_component_class(det["class"])
                    != normalize_component_class(ref["class"])):
                continue

            distance = float(np.hypot(det["ref_x"] - float(ref["x"]),
                                      det["ref_y"] - float(ref["y"])))
            bbox = det.get("ref_bbox")
            contains_reference = (
                bbox is not None
                and bbox[0] <= float(ref["x"]) <= bbox[2]
                and bbox[1] <= float(ref["y"]) <= bbox[3]
            )
            if contains_reference:
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                expected_width = ref.get("box_width")
                expected_height = ref.get("box_height")
                if (expected_width is not None and expected_height is not None
                        and (bbox_width > 3.0 * expected_width
                             or bbox_height > 3.0 * expected_height)):
                    contains_reference = False
            if distance <= tolerance or contains_reference:
                candidates_by_reference[ref_index].append(
                    (min(distance, tolerance if contains_reference else distance),
                     det_index)
                )

        candidates_by_reference[ref_index].sort()

    # Augmenting paths avoid losing a valid match to an earlier greedy choice.
    detection_to_reference = {}

    def assign(reference_index, visited_detections):
        for distance, det_index in candidates_by_reference[reference_index]:
            if det_index in visited_detections:
                continue
            visited_detections.add(det_index)

            previous_reference = detection_to_reference.get(det_index)
            if (previous_reference is None
                    or assign(previous_reference, visited_detections)):
                detection_to_reference[det_index] = reference_index
                return True
        return False

    reference_order = sorted(
        range(len(reference_components)),
        key=lambda index: (len(candidates_by_reference[index]), index),
    )
    for ref_index in reference_order:
        assign(ref_index, set())

    matches = []
    for det_index, ref_index in detection_to_reference.items():
        distance = next(
            distance for distance, candidate_index
            in candidates_by_reference[ref_index]
            if candidate_index == det_index
        )
        matches.append({"det_index": det_index, "ref_index": ref_index,
                        "distance": distance})
    matches.sort(key=lambda match: match["ref_index"])

    return matches, set(detection_to_reference), set(detection_to_reference.values())


# ============================================================
# 7. ĐÁNH GIÁ MẠCH
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
        matched_detections = {match["det_index"] for match in matches}
        stats["missing"] = len(reference_components) - len(used_references)
        stats["missing_components"] = [
            component["class"]
            for index, component in enumerate(reference_components)
            if index not in used_references
        ]
        stats["extra_components"] = [
            detection["class"]
            for index, detection in enumerate(detections)
            if index not in matched_detections
            and detection["confidence"] >= EXTRA_MIN_CONF
        ]
        stats["extra"] = len(stats["extra_components"])
        if (stats["missing"] > MISSING_ALLOWED
                or stats["extra"] > EXTRA_ALLOWED):
            return STATUS_FAULTY, stats
        return STATUS_OK, stats

    reference_counts = Counter(
        normalize_component_class(component["class"])
        for component in reference_components
    )
    detection_confidences = {}
    for detection in detections:
        if detection["confidence"] >= CONF:
            class_name = normalize_component_class(detection["class"])
            detection_confidences.setdefault(class_name, []).append(
                detection["confidence"]
            )

    class_names = set(reference_counts) | set(detection_confidences)
    for class_name in class_names:
        confidences = sorted(
            detection_confidences.get(class_name, []), reverse=True
        )
        expected_count = reference_counts[class_name]
        stats["missing"] += max(expected_count - len(confidences), 0)
        stats["extra"] += sum(
            confidence >= EXTRA_MIN_CONF
            for confidence in confidences[expected_count:]
        )
    stats["matched"] = len(reference_components) - stats["missing"]
    stats["missing_components"] = ["count mismatch"] * stats["missing"]
    stats["extra_components"] = ["count mismatch"] * stats["extra"]

    if (stats["missing"] > MISSING_ALLOWED
            or stats["extra"] > EXTRA_ALLOWED):
        return STATUS_FAULTY, stats

    # Without geometric alignment, class counts cannot prove components are
    # installed at the reference locations.
    return STATUS_FAULTY, stats


# ============================================================
# 7. VẼ KẾT QUẢ
# ============================================================

def draw_results(frame, detections, reference_components, matches,
                 used_references, aligned, H, status, stats):
    output = frame.copy()
    matched_detections = {match["det_index"] for match in matches}

    for det_index, det in enumerate(detections):
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])

        if aligned and det_index in matched_detections:
            color = (0, 255, 0)
            label = f'{det["class"]} {det["confidence"]:.2f} OK'
        elif det["confidence"] >= EXTRA_MIN_CONF:
            color = (0, 0, 255)
            label = f'{det["class"]} {det["confidence"]:.2f} EXTRA'
        else:
            color = (0, 165, 255)
            label = f'{det["class"]} {det["confidence"]:.2f} LOW'

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    if aligned and H is not None and stats["missing"] > 0:
        try:
            H_inverse = np.linalg.inv(H)

            for ref_index, ref in enumerate(reference_components):
                if ref_index in used_references:
                    continue

                point = np.array([[[float(ref["x"]), float(ref["y"])]]],
                                 dtype=np.float32)
                px, py = cv2.perspectiveTransform(point, H_inverse)[0, 0]
                px, py = int(px), int(py)

                if px < 0 or px >= output.shape[1] or py < 0 or py >= output.shape[0]:
                    continue

                color = (255, 0, 255)
                cv2.circle(output, (px, py), 10, color, 2)
                cv2.putText(output, f'MISSING {ref["class"]}', (px + 12, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        except np.linalg.LinAlgError:
            pass

    cv2.rectangle(output, (0, 0), (520, 80), (25, 25, 25), -1)
    cv2.putText(output, STATUS_TEXT_IMAGE[status], (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, STATUS_COLOR[status], 3, cv2.LINE_AA)
    if aligned:
        info = (f'Position check: Matched: {stats["matched"]}  '
                f'Missing: {stats["missing"]}  Extra: {stats["extra"]}')
    else:
        info = (f'Alignment unavailable; count check: Matched: {stats["matched"]}  '
                f'Missing: {stats["missing"]}  Extra: {stats["extra"]}')
    cv2.putText(output, info, (10, 68), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    details = []
    if stats["missing_components"]:
        details.append(f'Missing: {", ".join(stats["missing_components"])}')
    if stats["extra_components"]:
        details.append(f'Extra: {", ".join(stats["extra_components"])}')
    if details:
        cv2.rectangle(output, (0, 80), (min(output.shape[1], 900), 112), (25, 25, 25), -1)
        cv2.putText(output, " | ".join(details), (10, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return output


def draw_status_overlay(image, status):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (520, 48), (25, 25, 25), -1)
    cv2.putText(output, STATUS_TEXT_IMAGE[status], (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, STATUS_COLOR[status], 3, cv2.LINE_AA)
    return output


def resize_for_display(image):
    h, w = image.shape[:2]
    scale = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_LINEAR)


# ============================================================
# 8. MỘT LẦN QUÉT
# ============================================================

def scan_board(frame, model, reference_components, ref_shape, orb, ref_kp, ref_desc):
    # Cắt và nắn thẳng PCB trước khi nhận diện linh kiện.
    corners = find_board_corners(frame)
    rectified = rectify_board(frame, corners) if corners is not None else None

    if rectified is not None:
        board_crop = rectified
    else:
        roi = find_board_roi(frame)
        if roi is None:
            raise RuntimeError("Không tìm thấy PCB trong ảnh vừa chụp; bỏ qua lần quét.")
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        board_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    detection_frame = cv2.convertScaleAbs(
        board_crop, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA
    )
    detections = detect_components(model, detection_frame)

    ref_h, ref_w = ref_shape[:2]
    tolerance = MATCH_TOLERANCE_RATIO * max(ref_w, ref_h)

    H = estimate_component_alignment(reference_components, detections, tolerance)
    if H is None and USE_ORB_ALIGNMENT:
        alignment_frame = cv2.convertScaleAbs(
            board_crop, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA
        )
        H = compute_homography(
            ref_kp, ref_desc, ref_shape, alignment_frame, orb
        )

    aligned = H is not None

    transformed = []
    if aligned:
        for det in detections:
            rx, ry = transform_point_to_reference(det["cx"], det["cy"], H)
            d = det.copy()
            d["ref_x"], d["ref_y"] = rx, ry
            bbox_corners = np.float32([[
                [det["x1"], det["y1"]],
                [det["x2"], det["y1"]],
                [det["x2"], det["y2"]],
                [det["x1"], det["y2"]],
            ]])
            mapped_corners = cv2.perspectiveTransform(
                bbox_corners, H
            ).reshape(-1, 2)
            d["ref_bbox"] = (
                float(mapped_corners[:, 0].min()),
                float(mapped_corners[:, 1].min()),
                float(mapped_corners[:, 0].max()),
                float(mapped_corners[:, 1].max()),
            )
            transformed.append(d)

        matches, _, used_references = match_components(
            reference_components, transformed, tolerance,
            require_same_class=False,
        )
    else:
        matches, used_references = [], set()

    shown_detections = transformed if aligned else detections

    status, stats = evaluate_board_status(
        reference_components, shown_detections, matches, used_references, aligned
    )

    annotated = draw_results(
        board_crop, shown_detections, reference_components, matches,
        used_references, aligned, H, status, stats,
    )

    return {"status": status, "annotated": annotated, "stats": stats}


# ============================================================
# 9. QUÉT Ở LUỒNG NỀN
# ============================================================

class Scanner:
    def __init__(self, model, reference_components, ref_shape, orb, ref_kp, ref_desc):
        self.args = (model, reference_components, ref_shape, orb, ref_kp, ref_desc)
        self.busy = False
        self.result = None

    def start(self, frame):
        if self.busy:
            return False
        self.busy = True
        threading.Thread(target=self._run, args=(frame.copy(),), daemon=True).start()
        return True

    def _run(self, frame):
        try:
            self.result = scan_board(frame, *self.args)
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
        finally:
            self.busy = False

    def pop_result(self):
        result = self.result
        self.result = None
        return result


def stable_status(history):
    """Lấy trạng thái chiếm đa số trong các lần quét gần nhất."""
    if len(history) == 1:
        return history[-1]
    status, count = Counter(history).most_common(1)[0]
    return status if count >= 2 else history[-1]


# ============================================================
# 10. MAIN
# ============================================================

def main():
    os.system("")  # bật màu ANSI trên Windows terminal

    reference_components = load_reference()["components"]
    reference_counts = Counter(
        component["class"] for component in reference_components
    )
    print(
        "[INFO] PCB checker v4 - checks detected component positions against reference.",
        flush=True,
    )
    print(f"[INFO] Linh kiện mạch gốc: {dict(reference_counts)}", flush=True)
    reference_image = load_reference_image()
    ref_shape = reference_image.shape
    model = YOLO(find_model())
    attach_reference_box_sizes(
        reference_components, model, reference_image
    )

    if USE_ORB_ALIGNMENT:
        orb, ref_kp, ref_desc = build_reference_orb(reference_image)
    else:
        orb = ref_kp = ref_desc = None

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

    scanner = Scanner(model, reference_components, ref_shape, orb, ref_kp, ref_desc)

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
                annotated = draw_status_overlay(result["annotated"], last_status)

                print(f"{ANSI[last_status]}{STATUS_TEXT_TERMINAL[last_status]}{ANSI_RESET}",
                      flush=True)

                if SHOW_RESULT_WINDOW:
                    cv2.imshow(WINDOW_RESULT, resize_for_display(annotated))

            # ---------- VIDEO TRỰC TIẾP ----------
            if frame_id != last_shown_id:
                last_shown_id = frame_id

                live = resize_for_display(frame)
                if live is frame:
                    live = frame.copy()

                cv2.rectangle(live, (0, 0), (live.shape[1], 58), (25, 25, 25), -1)

                if last_status is not None:
                    cv2.putText(live, f"Last: {STATUS_TEXT_IMAGE[last_status]}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                STATUS_COLOR[last_status], 2, cv2.LINE_AA)

                cv2.putText(live, "s: Chup va nhan dien | q: Thoat",
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)

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
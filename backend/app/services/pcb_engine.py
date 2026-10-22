"""YOLO detection, PCB alignment, and reference-based quality inspection."""

import json
import os
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ============================================================
# CẤU HÌNH MẶC ĐỊNH
# ============================================================
DEFAULT_MODEL_PATH = "best.pt"
DEFAULT_REF_JSON = "reference_components.json"
DEFAULT_REF_IMAGE = "2.jpg"

DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 1536
DEFAULT_TOLERANCE_RATIO = 0.045
DEFAULT_MISSING_ALLOWED = 0
DEFAULT_EXTRA_ALLOWED = 2
DEFAULT_EXTRA_MIN_CONF = 0.50

ORB_FEATURES = 2000
RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 10
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.40
RANSAC_THRESHOLD = 5.0
MIN_BOARD_AREA_RATIO = 0.03
MAX_BOARD_AREA_RATIO = 2.00

BOARD_DETECTION_SIZE = (640, 240)
BOARD_CONTOUR_MIN_AREA = 20000
BOARD_PADDING_RATIO = 0.05

STATUS_OK = "OK"
STATUS_FAULTY = "FAULTY"
STATUS_UNALIGNED = "UNALIGNED"

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


def adjust_contrast(image, alpha=1.15, beta=-10):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def order_points(points):
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = np.argmin(ordered.sum(axis=1))
    return np.roll(ordered, -start, axis=0)


_BOARD_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
_ROI_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
_ROI_OPEN_KERNEL = np.ones((5, 5), dtype=np.uint8)
_BLUE_LOW = np.array([85, 35, 20], dtype=np.uint8)
_BLUE_HIGH = np.array([145, 255, 255], dtype=np.uint8)


def find_board_corners(frame):
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


def find_board_roi(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, _BLUE_LOW, _BLUE_HIGH)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, _ROI_CLOSE_KERNEL, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, _ROI_OPEN_KERNEL, iterations=1)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    frame_height, frame_width = frame.shape[:2]
    if (cv2.contourArea(contour) < frame_height * frame_width * 0.005 or width < 80 or height < 80):
        return None

    pad_x = int(width * BOARD_PADDING_RATIO)
    pad_y = int(height * BOARD_PADDING_RATIO)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2 = min(frame_width, x + width + pad_x)
    y2 = min(frame_height, y + height + pad_y)
    return x1, y1, x2, y2


def is_valid_homography(H, ref_shape, frame_shape):
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
    def __init__(self, reference_image):
        self.ref_shape = reference_image.shape
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        gray = cv2.equalizeHist(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY))
        self.ref_kp, self.ref_desc = self.orb.detectAndCompute(gray, None)

        if self.ref_desc is None or len(self.ref_kp) < MIN_GOOD_MATCHES:
            self.valid = False
        else:
            self.valid = True

    def compute_homography(self, frame):
        if not self.valid:
            return None
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frame_kp, frame_desc = self.orb.detectAndCompute(gray, None)
        if frame_desc is None or frame_kp is None or len(frame_kp) < MIN_GOOD_MATCHES:
            return None
        try:
            matches = self.matcher.knnMatch(frame_desc, self.ref_desc, k=2)
        except cv2.error:
            return None
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance]
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


def estimate_component_alignment(reference_components, detections, tolerance, conf_thresh):
    reference_by_class = {}
    detections_by_class = {}

    for component in reference_components:
        reference_by_class.setdefault(normalize_component_class(component["class"]), []).append(component)
    for detection in detections:
        if detection["confidence"] >= conf_thresh:
            detections_by_class.setdefault(normalize_component_class(detection["class"]), []).append(detection)

    source_points, reference_points = [], []
    for class_name, reference_items in reference_by_class.items():
        class_detections = detections_by_class.get(class_name, [])
        if len(reference_items) != 1 or len(class_detections) != 1:
            continue
        source_points.append([class_detections[0]["cx"], class_detections[0]["cy"]])
        reference_points.append([float(reference_items[0]["x"]), float(reference_items[0]["y"])])

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


def match_components(reference_components, detections, tolerance, require_same_class=True):
    num_ref, num_det = len(reference_components), len(detections)
    if num_ref == 0 or num_det == 0:
        return [], set(), set()

    ref_xy = np.array([[float(c["x"]), float(c["y"])] for c in reference_components])
    det_xy = np.array([[d["ref_x"], d["ref_y"]] for d in detections])
    det_box = np.array([d["ref_bbox"] for d in detections])

    dist = np.hypot(ref_xy[:, None, 0] - det_xy[None, :, 0], ref_xy[:, None, 1] - det_xy[None, :, 1])
    rx, ry = ref_xy[:, 0:1], ref_xy[:, 1:2]
    contains = ((det_box[None, :, 0] <= rx) & (rx <= det_box[None, :, 2]) & (det_box[None, :, 1] <= ry) & (ry <= det_box[None, :, 3]))

    exp_w = np.array([c.get("box_width", np.nan) for c in reference_components], dtype=np.float64)[:, None]
    exp_h = np.array([c.get("box_height", np.nan) for c in reference_components], dtype=np.float64)[:, None]
    box_w = (det_box[:, 2] - det_box[:, 0])[None, :]
    box_h = (det_box[:, 3] - det_box[:, 1])[None, :]
    too_big = (box_w > 3.0 * exp_w) | (box_h > 3.0 * exp_h)
    contains &= ~too_big

    allowed = (dist <= tolerance) | contains
    if require_same_class:
        ref_cls = np.array([normalize_component_class(c["class"]) for c in reference_components])
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
        {
            "det_index": det_index,
            "ref_index": ref_index,
            "distance": float(cost[ref_index, det_index]),
        }
        for det_index, ref_index in detection_to_reference.items()
    ]
    matches.sort(key=lambda m: m["ref_index"])
    return matches, set(detection_to_reference), set(detection_to_reference.values())


def evaluate_board_status(reference_components, detections, matches, used_references, aligned,
                          missing_allowed=DEFAULT_MISSING_ALLOWED, extra_allowed=DEFAULT_EXTRA_ALLOWED,
                          extra_min_conf=DEFAULT_EXTRA_MIN_CONF, conf_thresh=DEFAULT_CONF):
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
            if i not in matched_detections and d["confidence"] >= extra_min_conf
        ]
        stats["extra"] = len(stats["extra_components"])
        if stats["missing"] > missing_allowed or stats["extra"] > extra_allowed:
            return STATUS_FAULTY, stats
        return STATUS_OK, stats

    return STATUS_UNALIGNED, stats


# ============================================================
# PCB PIPELINE CLASS
# ============================================================

class PCBPipeline:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, ref_json_path=DEFAULT_REF_JSON, ref_image_path=DEFAULT_REF_IMAGE):
        self.model_path = model_path
        self.ref_json_path = ref_json_path
        self.ref_image_path = ref_image_path

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.use_cuda = torch.cuda.is_available()
        self.model = None
        self.reference_data = None
        self.reference_components = []
        self.reference_image = None
        self.aligner = None

        self.load_model()
        self.load_reference()

    def load_model(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Không tìm thấy model YOLO tại: {self.model_path}")
        print(f"[PCBPipeline] Loading YOLO model: {self.model_path} on {self.device}")
        self.model = YOLO(self.model_path)

    def load_reference(self):
        json_path = Path(self.ref_json_path)
        img_path = Path(self.ref_image_path)

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                self.reference_data = json.load(f)
            self.reference_components = self.reference_data.get("components", [])
            print(f"[PCBPipeline] Loaded {len(self.reference_components)} reference components from {json_path}")
        else:
            self.reference_data = None
            self.reference_components = []
            print(f"[PCBPipeline] Warning: Reference JSON {json_path} not found.")

        if img_path.exists():
            self.reference_image = cv2.imread(str(img_path))
            if self.reference_image is not None and len(self.reference_components) > 0:
                self._attach_reference_box_sizes()
                try:
                    self.aligner = OrbAligner(self.reference_image)
                    print(f"[PCBPipeline] ORB Aligner initialized successfully.")
                except Exception as e:
                    print(f"[PCBPipeline] ORB Aligner warning: {e}")
                    self.aligner = None
        else:
            self.reference_image = None
            self.aligner = None

    def _attach_reference_box_sizes(self):
        if self.model is None or self.reference_image is None or not self.reference_components:
            return
        ref_dets = self.detect_components(adjust_contrast(self.reference_image), conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ)
        if not ref_dets:
            return
        det_xy = np.array([[d["cx"], d["cy"]] for d in ref_dets])
        det_wh = np.array([[d["x2"] - d["x1"], d["y2"] - d["y1"]] for d in ref_dets])
        max_dist = 0.08 * max(self.reference_image.shape[:2])

        for component in self.reference_components:
            point = np.array([float(component["x"]), float(component["y"])])
            distances = np.hypot(*(det_xy - point).T)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= max_dist:
                component["box_width"] = max(1.0, float(det_wh[nearest, 0]))
                component["box_height"] = max(1.0, float(det_wh[nearest, 1]))

    def detect_components(self, frame, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ):
        predict_kwargs = {
            "source": frame,
            "conf": conf,
            "imgsz": imgsz,
            "device": self.device,
            "max_det": 1000,
            "agnostic_nms": True,
            "verbose": False,
        }
        if self.use_cuda:
            predict_kwargs["half"] = True
        results = self.model.predict(**predict_kwargs)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
        confs = boxes.conf.cpu().numpy().tolist()
        cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        cx = ((xyxy[:, 0] + xyxy[:, 2]) / 2.0).tolist()
        cy = ((xyxy[:, 1] + xyxy[:, 3]) / 2.0).tolist()
        names = self.model.names

        detections = []
        for i, (x1, y1, x2, y2) in enumerate(xyxy.tolist()):
            cls_name = names[cls_ids[i]]
            detections.append({
                "id": i + 1,
                "class_id": cls_ids[i],
                "class": normalize_component_class(cls_name),
                "original_class": cls_name,
                "confidence": round(float(confs[i]), 4),
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
                "cx": round(float(cx[i]), 2),
                "cy": round(float(cy[i]), 2),
                "width": round(float(x2 - x1), 2),
                "height": round(float(y2 - y1), 2),
            })
        return detections

    def crop_board(self, frame):
        corners = find_board_corners(frame)
        if corners is not None:
            rectified, transform = rectify_board(frame, corners)
            if rectified is not None:
                return rectified, transform, "perspective"

        roi = find_board_roi(frame)
        if roi is not None:
            x1, y1, x2, y2 = roi
            translation = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], dtype=np.float64)
            return frame[y1:y2, x1:x2], translation, "roi"

        # Fallback to entire image
        h, w = frame.shape[:2]
        identity = np.eye(3, dtype=np.float64)
        return frame, identity, "full"

    def _map_detections(self, detections, H):
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
            d["ref_x"] = float(ref_centers[i, 0])
            d["ref_y"] = float(ref_centers[i, 1])
            d["ref_bbox"] = (float(mins[i, 0]), float(mins[i, 1]), float(maxs[i, 0]), float(maxs[i, 1]))
            transformed.append(d)
        return transformed

    def run_detection_only(self, frame, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ):
        start_time = time.time()
        detections = self.detect_components(frame, conf=conf, imgsz=imgsz)
        annotated = frame.copy()

        # Class counts
        counter = Counter(d["class"] for d in detections)

        for det in detections:
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(annotated, label, (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2, cv2.LINE_AA)

        proc_time = round((time.time() - start_time) * 1000, 2)
        return {
            "detections": detections,
            "total_count": len(detections),
            "class_counts": dict(counter),
            "annotated_image": annotated,
            "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
            "latency_ms": proc_time,
        }

    def inspect_pcb(self, frame, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ,
                    tolerance_ratio=DEFAULT_TOLERANCE_RATIO, extra_allowed=DEFAULT_EXTRA_ALLOWED,
                    missing_allowed=DEFAULT_MISSING_ALLOWED, auto_crop=True):
        start_time = time.time()

        if self.reference_image is None or not self.reference_components:
            raise RuntimeError("Chưa có dữ liệu mạch tham chiếu (reference). Vui lòng nạp hoặc tạo reference trước.")

        ref_h, ref_w = self.reference_image.shape[:2]
        tolerance = tolerance_ratio * max(ref_w, ref_h)

        crop_method = "full"
        if auto_crop:
            board_crop, board_transform, crop_method = self.crop_board(frame)
        else:
            board_crop = frame
            board_transform = np.eye(3, dtype=np.float64)

        det_frame = adjust_contrast(board_crop)
        raw_detections = self.detect_components(det_frame, conf=conf, imgsz=imgsz)

        # 1. Thử estimate affine qua các class độc nhất
        H = estimate_component_alignment(self.reference_components, raw_detections, tolerance, conf)
        align_method = "Affine (Components)" if H is not None else None

        # 2. Nếu thất bại, thử ORB feature matching
        if H is None and self.aligner is not None:
            H = self.aligner.compute_homography(det_frame)
            if H is not None:
                align_method = "Homography (ORB)"

        aligned = H is not None
        if not aligned:
            align_method = "Không căn chỉnh được"

        # Ghép linh kiện
        if aligned:
            shown_detections = self._map_detections(raw_detections, H)
            matches, _, used_references = match_components(
                self.reference_components, shown_detections, tolerance, require_same_class=False
            )
        else:
            shown_detections = raw_detections
            matches = []
            used_references = set()

        status, stats = evaluate_board_status(
            self.reference_components, shown_detections, matches, used_references, aligned,
            missing_allowed=missing_allowed, extra_allowed=extra_allowed, conf_thresh=conf
        )

        matched_detection_indices = {m["det_index"] for m in matches}
        match_map = {m["det_index"]: m["ref_index"] for m in matches}

        # Bổ sung trạng thái cho từng detection
        detailed_detections = []
        for i, det in enumerate(shown_detections):
            d_item = det.copy()
            if not aligned:
                d_item["status"] = "DETECTED"
                d_item["matched_ref_id"] = None
                d_item["matched_ref_class"] = None
            elif i in matched_detection_indices:
                d_item["status"] = "OK"
                ref_idx = match_map[i]
                d_item["matched_ref_id"] = self.reference_components[ref_idx].get("id", ref_idx + 1)
                d_item["matched_ref_class"] = self.reference_components[ref_idx]["class"]
            elif det["confidence"] >= DEFAULT_EXTRA_MIN_CONF:
                d_item["status"] = "EXTRA"
                d_item["matched_ref_id"] = None
                d_item["matched_ref_class"] = None
            else:
                d_item["status"] = "LOW_CONF"
                d_item["matched_ref_id"] = None
                d_item["matched_ref_class"] = None
            detailed_detections.append(d_item)

        # Chi tiết linh kiện thiếu
        missing_details = []
        if aligned and stats["missing"] > 0:
            for idx, c in enumerate(self.reference_components):
                if idx not in used_references:
                    missing_details.append({
                        "ref_id": c.get("id", idx + 1),
                        "class": c["class"],
                        "ref_x": float(c["x"]),
                        "ref_y": float(c["y"]),
                    })

        # Vẽ kết quả lên ảnh
        annotated = board_crop.copy()
        for i, det in enumerate(shown_detections):
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            det_status = detailed_detections[i]["status"]

            if det_status == "OK":
                color = (0, 230, 0)      # Xanh lá
                tag = "OK"
            elif det_status == "DETECTED":
                color = (0, 220, 255)
                tag = "DETECTED"
            elif det_status == "EXTRA":
                color = (0, 0, 255)      # Đỏ
                tag = "DIFF"
            else:
                color = (0, 165, 255)    # Cam (Low conf)
                tag = "LOW"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{det['class']} {det['confidence']:.2f} [{tag}]",
                        (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

        # Vẽ đánh dấu linh kiện thiếu
        if aligned and H is not None and missing_details:
            try:
                H_inv = np.linalg.inv(H)
                pts = np.float32([[m["ref_x"], m["ref_y"]] for m in missing_details]).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
                h_img, w_img = annotated.shape[:2]
                for m_item, (px, py) in zip(missing_details, projected):
                    px, py = int(px), int(py)
                    if 0 <= px < w_img and 0 <= py < h_img:
                        cv2.circle(annotated, (px, py), 12, (255, 0, 255), 2)
                        cv2.circle(annotated, (px, py), 3, (255, 0, 255), -1)
                        cv2.putText(annotated, f"MISSING: {m_item['class']}",
                                    (px + 14, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"[PCBPipeline] Draw missing points error: {e}")

        # Vẽ Header trạng thái
        status_color = (
            (0, 220, 0) if status == STATUS_OK else
            (0, 190, 255) if status == STATUS_UNALIGNED else
            (0, 0, 235)
        )
        status_label = {
            STATUS_OK: "MẠCH ĐẠT CHUẨN (OK)",
            STATUS_FAULTY: "MẠCH BỊ LỖI (FAULTY)",
            STATUS_UNALIGNED: "ĐÃ QUÉT - CHƯA ĐỐI CHIẾU",
        }[status]

        header_h = 75
        cv2.rectangle(annotated, (0, 0), (min(annotated.shape[1], 560), header_h), (20, 20, 25), -1)
        cv2.putText(annotated, status_label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)

        if aligned:
            summary_txt = f"Matched: {stats['matched']} | Missing: {stats['missing']} | Diff: {stats['extra']}"
        else:
            summary_txt = f"Detected: {len(raw_detections)} | QA: alignment unavailable"
        cv2.putText(annotated, summary_txt, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        proc_time = round((time.time() - start_time) * 1000, 2)

        return {
            "status": status,
            "status_vi": {
                STATUS_OK: "MẠCH ĐẠT CHUẨN",
                STATUS_FAULTY: "MẠCH BỊ LỖI",
                STATUS_UNALIGNED: "ĐÃ QUÉT - CHƯA ĐỐI CHIẾU",
            }[status],
            "aligned": aligned,
            "qa_available": aligned,
            "alignment_method": align_method,
            "crop_method": crop_method,
            "stats": {
                **stats,
                "reference_count": len(self.reference_components),
                "detected_count": len(raw_detections),
            },
            "class_counts": dict(Counter(d["class"] for d in raw_detections)),
            "detections": detailed_detections,
            "missing_components_details": missing_details,
            "annotated_image": annotated,
            "board_crop_image": board_crop,
            "latency_ms": proc_time,
        }

    def create_reference_from_image(self, image_np, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ):
        """Tạo bản đồ linh kiện tham chiếu mới từ ảnh (giống machpcbgoc.py)"""
        h, w = image_np.shape[:2]
        results = self.model.predict(
            source=image_np,
            conf=conf,
            imgsz=imgsz,
            device=self.device,
            agnostic_nms=True,
            verbose=False
        )
        if not results:
            raise RuntimeError("YOLO không trả về kết quả.")

        result = results[0]
        components = []
        annotated = image_np.copy()

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for idx in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().tolist()
                cls_id = int(boxes.cls[idx].item())
                c_conf = float(boxes.conf[idx].item())
                class_name = str(self.model.names[cls_id])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                component = {
                    "id": idx + 1,
                    "class_id": cls_id,
                    "class": class_name,
                    "confidence_reference": round(c_conf, 4),
                    "x": round(cx, 2),
                    "y": round(cy, 2),
                }
                components.append(component)

                label = f"{class_name} {c_conf:.2f}"
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated, label, (int(x1), max(int(y1) - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        if not components:
            raise RuntimeError(
                "Không nhận diện được linh kiện trong ảnh mạch chuẩn. "
                "Hãy dùng ảnh rõ nét hơn hoặc giảm ngưỡng confidence."
            )

        output_data = {
            "image": Path(self.ref_image_path).name,
            "image_width": w,
            "image_height": h,
            "components_count": len(components),
            "components": components,
        }

        # Lưu ảnh và json
        if not cv2.imwrite(self.ref_image_path, image_np):
            raise RuntimeError(f"Không lưu được ảnh mạch chuẩn: {self.ref_image_path}")
        detected_path = Path(self.ref_image_path).with_name("reference_detected.jpg")
        if not cv2.imwrite(str(detected_path), annotated):
            raise RuntimeError(f"Không lưu được ảnh linh kiện reference đã nhận diện: {detected_path}")
        with open(self.ref_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # Nạp lại reference
        self.load_reference()

        return {
            "components_count": len(components),
            "components": components,
            "annotated_image": annotated,
        }

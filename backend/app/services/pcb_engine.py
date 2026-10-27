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
DEFAULT_REF_IMAGE = "Image_20261019190958053.bmp"

DEFAULT_CONF = 0.35
DEFAULT_IMGSZ = 1536
DEFAULT_TOLERANCE_RATIO = 0.55
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
    "potentiometer": "inductor/potentiometer",
    "variable_resistor": "inductor/potentiometer",
    "variable resistor": "inductor/potentiometer",
    "inductor": "inductor/potentiometer",
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
_GREEN_LOW = np.array([35, 35, 20], dtype=np.uint8)
_GREEN_HIGH = np.array([85, 255, 255], dtype=np.uint8)
# Thêm ngưỡng cho mạch màu đen/tối
_DARK_VALUE_THRESH = 80
_DARK_SAT_LOW = 0
_DARK_SAT_HIGH = 255


def find_board_corners(frame, reference_aspect=None):
    frame_height, frame_width = frame.shape[:2]
    detect_width = BOARD_DETECTION_SIZE[0]
    scale = detect_width / frame_width
    detect_height = max(1, round(frame_height * scale))
    resized = cv2.resize(frame, (detect_width, detect_height))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    min_area = max(
        BOARD_CONTOUR_MIN_AREA,
        0.015 * detect_width * detect_height,
    )
    best_candidate = None
    for low, high in ((20, 60), (35, 100), (50, 150), (70, 180)):
        edges = cv2.Canny(blur, low, high)
        for kernel_size in (3, 5, 7):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (kernel_size, kernel_size)
            )
            closed = cv2.morphologyEx(
                edges, cv2.MORPH_CLOSE, kernel, iterations=2
            )
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in sorted(
                contours, key=cv2.contourArea, reverse=True
            )[:15]:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                # Bỏ qua nếu contour chiếm gần hết khung hình (viền băng tải / camera)
                if area > 0.75 * detect_width * detect_height:
                    continue
                if x <= 3 or y <= 3 or x + width >= detect_width - 3 or y + height >= detect_height - 3:
                    continue

                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area <= 0 or area / hull_area < 0.20:
                    continue
                rect = cv2.minAreaRect(hull)
                rect_width, rect_height = rect[1]
                rect_area = rect_width * rect_height
                if rect_area <= 0:
                    continue
                rectangularity = hull_area / rect_area
                aspect = max(rect_width, rect_height) / max(1, min(rect_width, rect_height))
                if rectangularity < 0.55:
                    continue
                if reference_aspect is not None:
                    if not 0.5 * reference_aspect <= aspect <= 2.0 * reference_aspect:
                        continue
                elif not 1.0 <= aspect <= 6.0:
                    continue

                score = area * rectangularity * (area / hull_area)
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, rect)

    if best_candidate is None:
        return None

    corners = order_points(cv2.boxPoints(best_candidate[1]))
    center = corners.mean(axis=0)
    corners = center + (corners - center) * (1 + 2 * BOARD_PADDING_RATIO)
    corners[:, 0] *= frame_width / detect_width
    corners[:, 1] *= frame_height / detect_height
    return corners


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
    frame_height, frame_width = frame.shape[:2]

    # Mask cho mạch xanh dương và xanh lá (board truyền thống)
    blue_mask = cv2.inRange(hsv, _BLUE_LOW, _BLUE_HIGH)
    green_mask = cv2.inRange(hsv, _GREEN_LOW, _GREEN_HIGH)
    color_mask = cv2.bitwise_or(blue_mask, green_mask)

    # Mask cho mạch đen/tối: vùng có value thấp (mạch PCB đen)
    dark_mask = cv2.inRange(
        hsv,
        np.array([0, _DARK_SAT_LOW, 0], dtype=np.uint8),
        np.array([180, _DARK_SAT_HIGH, _DARK_VALUE_THRESH], dtype=np.uint8),
    )

    # Mask cho vùng có texture/edge (linh kiện tạo cạnh rõ)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_dilated = cv2.dilate(edges, _ROI_CLOSE_KERNEL, iterations=3)

    # Kết hợp tất cả: board xanh HOẶC board đen HOẶC vùng có cạnh dày đặc
    mask = cv2.bitwise_or(color_mask, dark_mask)
    mask = cv2.bitwise_or(mask, edge_dilated)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ROI_CLOSE_KERNEL, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ROI_OPEN_KERNEL, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    if (cv2.contourArea(contour) < frame_height * frame_width * 0.005 or width < 60 or height < 60):
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


class BoardCropError(RuntimeError):
    """Raised when automatic PCB cropping cannot find a complete board."""


def find_reference_board_corners(frame, aligner):
    if aligner is None or not aligner.valid:
        return None

    homography = aligner.compute_homography(frame)
    if homography is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    reference_height, reference_width = aligner.ref_shape[:2]
    reference_corners = np.float32([
        [0, 0],
        [reference_width - 1, 0],
        [reference_width - 1, reference_height - 1],
        [0, reference_height - 1],
    ]).reshape(-1, 1, 2)

    try:
        corners = cv2.perspectiveTransform(
            reference_corners, np.linalg.inv(homography)
        ).reshape(-1, 2)
    except np.linalg.LinAlgError:
        return None

    if (not np.all(np.isfinite(corners))
            or not cv2.isContourConvex(corners.astype(np.float32))):
        return None
    if (np.any(corners[:, 0] < -2) or np.any(corners[:, 0] > frame_width + 1)
            or np.any(corners[:, 1] < -2) or np.any(corners[:, 1] > frame_height + 1)):
        return None
    corners[:, 0] = np.clip(corners[:, 0], 0, frame_width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, frame_height - 1)

    area_ratio = cv2.contourArea(corners.astype(np.float32)) / (
        frame_width * frame_height
    )
    if not 0.03 <= area_ratio <= 1.0:
        return None
    return order_points(corners)


def estimate_component_alignment(reference_components, detections, tolerance, conf_thresh, crop_shape=None, ref_shape=None):
    """
    Ước lượng ma trận Affine biến đổi tọa độ từ khung hình quét sang mạch chuẩn.
    Kết hợp:
    1. Căn chỉnh theo tỷ lệ kích thước (Scale & Aspect Ratio) theo góc 0° và 180°.
    2. Ghép cặp theo các loại linh kiện đặc trưng (IC, diode, capacitor, inductor...) qua RANSAC.
    """
    if not reference_components or not detections:
        return None

    valid_dets = [d for d in detections if d["confidence"] >= conf_thresh]
    if len(valid_dets) < 2:
        return None

    best_H = None
    best_inliers = 0

    # 1. Thử căn chỉnh bằng Scale trực tiếp từ crop -> reference (với 0° và 180°)
    if crop_shape is not None and ref_shape is not None:
        ch, cw = crop_shape[:2]
        rh, rw = ref_shape[:2]
        if cw > 0 and ch > 0 and rw > 0 and rh > 0:
            scale_candidates = [
                ("scale_0", np.array([[rw / cw, 0, 0], [0, rh / ch, 0], [0, 0, 1]], dtype=np.float64)),
                ("scale_180", np.array([[-rw / cw, 0, rw], [0, -rh / ch, rh], [0, 0, 1]], dtype=np.float64)),
            ]
            det_pts = np.float32([[d["cx"], d["cy"]] for d in valid_dets]).reshape(-1, 1, 2)
            for _, H_cand in scale_candidates:
                proj = cv2.perspectiveTransform(det_pts, H_cand).reshape(-1, 2)
                matched_pairs_src = []
                matched_pairs_dst = []
                used_ref = set()
                for di, (px, py) in enumerate(proj):
                    d_cls = normalize_component_class(valid_dets[di]["class"])
                    best_dist = float("inf")
                    best_ri = None
                    for ri, r in enumerate(reference_components):
                        if ri in used_ref:
                            continue
                        if normalize_component_class(r["class"]) == d_cls:
                            d_val = float(np.hypot(px - float(r["x"]), py - float(r["y"])))
                            if d_val <= tolerance and d_val < best_dist:
                                best_dist = d_val
                                best_ri = ri
                    if best_ri is not None:
                        used_ref.add(best_ri)
                        matched_pairs_src.append([valid_dets[di]["cx"], valid_dets[di]["cy"]])
                        matched_pairs_dst.append([float(reference_components[best_ri]["x"]), float(reference_components[best_ri]["y"])])

                if len(matched_pairs_src) >= 3 and len(matched_pairs_src) > best_inliers:
                    src_arr = np.float32(matched_pairs_src)
                    dst_arr = np.float32(matched_pairs_dst)
                    affine, inliers = cv2.estimateAffine2D(src_arr, dst_arr, method=cv2.RANSAC, ransacReprojThreshold=tolerance)
                    if affine is not None and inliers is not None and int(inliers.sum()) >= 3:
                        best_inliers = len(matched_pairs_src)
                        best_H = np.eye(3, dtype=np.float64)
                        best_H[:2, :] = affine
                    else:
                        best_inliers = len(matched_pairs_src)
                        best_H = H_cand

    # 2. RANSAC dựa trên các anchor linh kiện cùng class
    reference_by_class = {}
    detections_by_class = {}
    for c in reference_components:
        reference_by_class.setdefault(normalize_component_class(c["class"]), []).append(c)
    for d in valid_dets:
        detections_by_class.setdefault(normalize_component_class(d["class"]), []).append(d)

    common_classes = sorted(
        [cls for cls in reference_by_class if cls in detections_by_class],
        key=lambda c: len(reference_by_class[c]) * len(detections_by_class[c])
    )

    unique_pairs = []
    for cls in common_classes:
        if len(reference_by_class[cls]) == 1 and len(detections_by_class[cls]) == 1:
            unique_pairs.append((detections_by_class[cls][0], reference_by_class[cls][0]))

    if len(unique_pairs) >= 3:
        src = np.float32([[d["cx"], d["cy"]] for d, r in unique_pairs])
        dst = np.float32([[float(r["x"]), float(r["y"])] for d, r in unique_pairs])
        affine, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=tolerance)
        if affine is not None and inliers is not None and int(inliers.sum()) >= 3:
            H = np.eye(3, dtype=np.float64)
            H[:2, :] = affine
            return H

    # Nếu chỉ có 1 hoặc 2 unique pairs, thử kết hợp với các class có ít instance (như tụ hoặc cuộn cảm)
    if len(unique_pairs) in (1, 2) and best_inliers < 4:
        small_classes = [c for c in common_classes if 1 < len(reference_by_class[c]) * len(detections_by_class[c]) <= 6]
        for extra_cls in small_classes:
            r_items = reference_by_class[extra_cls]
            d_items = detections_by_class[extra_cls]
            for d_it in d_items:
                for r_it in r_items:
                    trial_src = [[d["cx"], d["cy"]] for d, _ in unique_pairs] + [[d_it["cx"], d_it["cy"]]]
                    trial_dst = [[float(r["x"]), float(r["y"])] for _, r in unique_pairs] + [[float(r_it["x"]), float(r_it["y"])]]
                    if len(trial_src) < 3:
                        continue
                    src_arr = np.float32(trial_src[:3])
                    dst_arr = np.float32(trial_dst[:3])
                    try:
                        M = cv2.getAffineTransform(src_arr, dst_arr)
                    except cv2.error:
                        continue
                    det_all = np.float32([[d["cx"], d["cy"]] for d in valid_dets]).reshape(-1, 1, 2)
                    proj = cv2.transform(det_all, M).reshape(-1, 2)
                    cur_inliers = 0
                    for di, (px, py) in enumerate(proj):
                        d_cls = normalize_component_class(valid_dets[di]["class"])
                        for r in reference_components:
                            if normalize_component_class(r["class"]) == d_cls:
                                if np.hypot(px - float(r["x"]), py - float(r["y"])) <= tolerance:
                                    cur_inliers += 1
                                    break
                    if cur_inliers > best_inliers:
                        best_inliers = cur_inliers
                        best_H = np.eye(3, dtype=np.float64)
                        best_H[:2, :] = M

    if best_H is not None and best_inliers >= 3:
        return best_H

    return None


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
        # Nếu không có ứng viên trong phạm vi tolerance, tìm các detection cùng class
        if len(idx) == 0 and require_same_class:
            same_cls_idx = np.flatnonzero(ref_cls[r] == det_cls)
            if len(same_cls_idx) > 0:
                idx = same_cls_idx
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
        stats["extra"] = 0
        stats["extra_components"] = []
        # Tiêu chuẩn: Lỗi CHỈ xác định khi thiếu linh kiện (stats["missing"] > missing_allowed)
        # Tuyệt đối không có trường hợp thừa linh kiện
        if stats["missing"] > missing_allowed:
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
        self.mono_cropper = None
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
        try:
            from pcb_crop_engine import PCBMonoCropper
            self.mono_cropper = PCBMonoCropper(model=self.model, conf_thresh=DEFAULT_CONF, inference_size=1024)
            print("[PCBPipeline] PCBMonoCropper initialized successfully.")
        except Exception as e:
            print(f"[PCBPipeline] PCBMonoCropper init warning: {e}")
            self.mono_cropper = None

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

        # Thu thập các hộp IC để lọc bỏ nhận diện giả trên chân/thân IC
        ic_boxes = []
        for i, (x1, y1, x2, y2) in enumerate(xyxy.tolist()):
            cls_name = names[cls_ids[i]]
            if normalize_component_class(cls_name) == "ic":
                ic_boxes.append((x1, y1, x2, y2))

        detections = []
        det_id = 1
        for i, (x1, y1, x2, y2) in enumerate(xyxy.tolist()):
            cls_name = names[cls_ids[i]]
            norm_cls = normalize_component_class(cls_name)
            center_x = float(cx[i])
            center_y = float(cy[i])

            # Lọc nhận diện giả (ví dụ: resistor nhận diện nhầm trên chân kim loại của IC)
            if norm_cls != "ic" and any(bx1 <= center_x <= bx2 and by1 <= center_y <= by2 for bx1, by1, bx2, by2 in ic_boxes):
                continue

            detections.append({
                "id": det_id,
                "class_id": cls_ids[i],
                "class": norm_cls,
                "original_class": cls_name,
                "confidence": round(float(confs[i]), 4),
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
                "cx": round(center_x, 2),
                "cy": round(center_y, 2),
                "width": round(float(x2 - x1), 2),
                "height": round(float(y2 - y1), 2),
            })
            det_id += 1
        return detections

    def crop_board(self, frame, fallback_full=False):
        # 1. PCBMonoCropper (Ưu tiên OBB AI theo cụm linh kiện + nắn thẳng)
        if self.mono_cropper is not None:
            try:
                cropped, info = self.mono_cropper.process(frame)
                if info.get("success") and cropped is not None and cropped.size > 0:
                    corners = info.get("corners")
                    if corners is not None:
                        c_area = cv2.contourArea(corners)
                        frame_area = frame.shape[0] * frame.shape[1]
                        if c_area >= 0.88 * frame_area:
                            # Frame đã là ảnh cắt sát bo mạch chuẩn, không cần warp lại làm mờ pixel
                            return frame, np.eye(3, dtype=np.float64), "mono_cropper"

                    transform = info.get("transform")
                    if transform is None and corners is not None:
                        cw, ch = cropped.shape[1], cropped.shape[0]
                        dst = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32)
                        transform = cv2.getPerspectiveTransform(corners, dst)
                    return cropped, transform, "mono_cropper"
            except Exception as e:
                print(f"[PCBPipeline] PCBMonoCropper error: {e}")

        # 2. Thử tìm góc qua homography với ảnh reference
        reference_corners = find_reference_board_corners(frame, self.aligner)
        if reference_corners is not None:
            rectified, transform = rectify_board(frame, reference_corners)
            if rectified is not None:
                return rectified, transform, "reference"

        # 3. Thử tìm contour bo mạch (đã lọc không lấy nhầm viền khung hình)
        reference_aspect = None
        if self.reference_image is not None:
            reference_height, reference_width = self.reference_image.shape[:2]
            reference_aspect = reference_width / reference_height
        corners = find_board_corners(frame, reference_aspect=reference_aspect)
        if corners is not None:
            rectified, transform = rectify_board(frame, corners)
            if rectified is not None:
                return rectified, transform, "perspective"

        # 4. Thử tìm ROI màu/texture
        roi = find_board_roi(frame)
        if roi is not None:
            x1, y1, x2, y2 = roi
            translation = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], dtype=np.float64)
            return frame[y1:y2, x1:x2], translation, "roi"

        # 5. Fallback toàn bộ khung hình
        print("[PCBPipeline] Không tách riêng được vùng board, dùng toàn bộ khung hình.")
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

    def run_detection_only(self, frame, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ, auto_crop=True):
        start_time = time.time()
        crop_method = "full"
        if auto_crop:
            board_crop, board_transform, crop_method = self.crop_board(frame)
        else:
            board_crop = frame
        detections = self.detect_components(board_crop, conf=conf, imgsz=imgsz)
        annotated = board_crop.copy()

        # Class counts
        counter = Counter(d["class"] for d in detections)

        for det in detections:
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            label = f"{det['class']} {det['confidence']:.2f} [OK]"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 0), 2)
            cv2.putText(annotated, label, (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 2, cv2.LINE_AA)

        proc_time = round((time.time() - start_time) * 1000, 2)
        return {
            "detections": detections,
            "total_count": len(detections),
            "class_counts": dict(counter),
            "annotated_image": annotated,
            "image_size": {"width": board_crop.shape[1], "height": board_crop.shape[0]},
            "crop_method": crop_method,
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

        if np.mean(board_crop) < 60 or np.std(board_crop) < 30:
            det_frame = adjust_contrast(board_crop)
        else:
            det_frame = board_crop
        raw_detections = self.detect_components(det_frame, conf=conf, imgsz=imgsz)

        # 1. Thử estimate affine qua các class độc nhất & tỷ lệ kích thước
        H = estimate_component_alignment(
            self.reference_components, raw_detections, tolerance, conf,
            crop_shape=board_crop.shape, ref_shape=self.reference_image.shape
        )
        align_method = "Affine (Components)" if H is not None else None

        # 2. Thử ORB feature matching (cả 0° và 180°) nếu affine chưa tìm được
        if H is None and self.aligner is not None:
            H = self.aligner.compute_homography(det_frame)
            if H is not None:
                align_method = "Homography (ORB)"
            else:
                det_frame_180 = cv2.rotate(det_frame, cv2.ROTATE_180)
                H_180 = self.aligner.compute_homography(det_frame_180)
                if H_180 is not None:
                    ch, cw = det_frame.shape[:2]
                    R180 = np.array([[-1, 0, cw - 1], [0, -1, ch - 1], [0, 0, 1]], dtype=np.float64)
                    H = H_180 @ R180
                    align_method = "Homography (ORB 180°)"

        aligned = H is not None

        # Ghép linh kiện: Chỉ cần đúng loại linh kiện (class) so với mạch gốc, không cần chuẩn xác tọa độ
        shown_detections = self._map_detections(raw_detections, H) if aligned else raw_detections

        matches = []
        used_references = set()
        for di, d in enumerate(shown_detections):
            d_cls = normalize_component_class(d["class"])
            for ri, r in enumerate(self.reference_components):
                if ri not in used_references and normalize_component_class(r["class"]) == d_cls:
                    used_references.add(ri)
                    matches.append({"det_index": di, "ref_index": ri, "distance": 0.0})
                    break

        status, stats = evaluate_board_status(
            self.reference_components, shown_detections, matches, used_references, aligned=True,
            missing_allowed=missing_allowed, extra_allowed=extra_allowed, conf_thresh=conf
        )
        matched_detection_indices = {m["det_index"] for m in matches}
        match_map = {m["det_index"]: m["ref_index"] for m in matches}

        # Bổ sung trạng thái cho từng detection: Cứ nhận diện được (detection) là OK
        detailed_detections = []
        for i, det in enumerate(shown_detections):
            d_item = det.copy()
            d_item["status"] = "OK"
            if i in matched_detection_indices:
                ref_idx = match_map[i]
                d_item["matched_ref_id"] = self.reference_components[ref_idx].get("id", ref_idx + 1)
                d_item["matched_ref_class"] = self.reference_components[ref_idx]["class"]
            else:
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

        # Vẽ kết quả lên ảnh: toàn bộ linh kiện nhận diện đều viền xanh lá OK
        annotated = board_crop.copy()
        for i, det in enumerate(shown_detections):
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            color = (0, 230, 0)      # Xanh lá: OK
            tag = "OK"

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
            STATUS_FAULTY: "MẠCH BỊ LỖI (THIẾU LINH KIỆN)",
            STATUS_UNALIGNED: "ĐÃ QUÉT - CHƯA ĐỐI CHIẾU",
        }[status]

        header_h = 75
        cv2.rectangle(annotated, (0, 0), (min(annotated.shape[1], 560), header_h), (20, 20, 25), -1)
        cv2.putText(annotated, status_label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)

        if aligned:
            summary_txt = f"Khớp: {stats['matched']}/{len(self.reference_components)} | Thiếu: {stats['missing']}"
        else:
            summary_txt = f"Detected: {len(raw_detections)} | QA: alignment unavailable"
        cv2.putText(annotated, summary_txt, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        proc_time = round((time.time() - start_time) * 1000, 2)

        return {
            "status": status,
            "status_vi": {
                STATUS_OK: "MẠCH ĐẠT CHUẨN",
                STATUS_FAULTY: "MẠCH BỊ LỖI (THIẾU LINH KIỆN)",
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
            # Thu thập các hộp IC để lọc bỏ nhận diện giả trên chân/thân IC
            ic_boxes = []
            for idx in range(len(boxes)):
                cls_id = int(boxes.cls[idx].item())
                if str(self.model.names[cls_id]).lower() == "ic":
                    ic_boxes.append(boxes.xyxy[idx].cpu().numpy().tolist())

            comp_idx = 1
            for idx in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().tolist()
                cls_id = int(boxes.cls[idx].item())
                c_conf = float(boxes.conf[idx].item())
                class_name = str(self.model.names[cls_id])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Bỏ nhận diện giả trên chân/thân IC
                if class_name.lower() != "ic" and any(
                    b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in ic_boxes
                ):
                    continue

                component = {
                    "id": comp_idx,
                    "class_id": cls_id,
                    "class": class_name,
                    "confidence_reference": round(c_conf, 4),
                    "x": round(cx, 2),
                    "y": round(cy, 2),
                }
                comp_idx += 1
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

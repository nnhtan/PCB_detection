"""YOLO detection, PCB alignment, and reference-based quality inspection."""

import json
import logging
import math
import os
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

logger = logging.getLogger(__name__)

# ============================================================
# CẤU HÌNH MẶC ĐỊNH
# ============================================================
DEFAULT_MODEL_PATH = "best.pt"
DEFAULT_REF_JSON = "reference_components.json"
DEFAULT_REF_IMAGE = "Image_20261019190958053.bmp"

DEFAULT_CONF = 0.35
DEFAULT_IMGSZ = 1536
DEFAULT_TOLERANCE_RATIO = 0.045
MAX_TOLERANCE_RATIO = 0.15
DEFAULT_MISSING_ALLOWED = 0
DEFAULT_EXTRA_ALLOWED = 2
DEFAULT_EXTRA_MIN_CONF = 0.50
DETECTION_OVERLAP_THRESHOLD = 0.50
DETECTION_MAX_AREA_RATIO = 4.0
IGNORED_DETECTION_CLASSES = frozenset({"buzzer"})
SAHI_SLICE_SIZE = 320
FAST_SAHI_SLICE_SIZE = 960
GPU_SAHI_BATCH_SIZE = 2
SAHI_OVERLAP_RATIO = 0.20
REFERENCE_ROI_MIN_SIZE = 96
REFERENCE_ROI_MAX_SIZE = 384
REFERENCE_ROI_SIZE_SCALE = 3.0
REFERENCE_ROI_INFERENCE_SCALE = 2.0
REFERENCE_ROI_CONFIDENCE = 0.01

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
STATUS_REVIEW = "REVIEW"
STATUS_UNALIGNED = "UNALIGNED"

CLASS_ALIASES = {
    "variable resistor": "potentiometer",
}

MODEL_CLASS_DISPLAY_NAMES = {
    "capacitor": "capacitor",
    "ic": "IC",
    "led": "LED",
    "resistor": "resistor",
}


def canonical_model_class_name(class_name):
    class_name = str(class_name).strip()
    normalized = " ".join(class_name.casefold().replace("_", " ").split())
    return MODEL_CLASS_DISPLAY_NAMES.get(normalized, class_name)


@lru_cache(maxsize=None)
def normalize_component_class(class_name):
    normalized = " ".join(
        str(class_name).casefold().replace("-", " ").replace("_", " ").split()
    )
    return CLASS_ALIASES.get(normalized, normalized)


def find_reference_mounting_holes(reference_image, reference_components):
    height, width = reference_image.shape[:2]
    minimum_radius = max(10, round(min(height, width) * 0.045))
    maximum_radius = max(minimum_radius + 1, round(min(height, width) * 0.12))
    gray = cv2.medianBlur(
        cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY), 5
    )
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, round(min(height, width) * 0.10)),
        param1=100,
        param2=40,
        minRadius=minimum_radius,
        maxRadius=maximum_radius,
    )
    if circles is None:
        return []

    holes = []
    for circle_x, circle_y, radius in circles[0]:
        circle_inside_image = (
            circle_x - radius >= 0
            and circle_y - radius >= 0
            and circle_x + radius < width
            and circle_y + radius < height
        )
        in_board_corner = (
            circle_x < 0.22 * width and circle_y < 0.20 * height
        ) or (
            circle_x > 0.78 * width and circle_y > 0.80 * height
        )
        near_component = any(
            math.hypot(
                circle_x - float(component["x"]),
                circle_y - float(component["y"]),
            )
            <= radius + 35
            for component in reference_components
        )
        if circle_inside_image and in_board_corner and not near_component:
            holes.append((float(circle_x), float(circle_y), float(radius)))
    return holes


def estimate_anchor_orientation(reference_components, detections, conf_thresh):
    reference_caps = [
        component for component in reference_components
        if normalize_component_class(component["class"]) == "capacitor"
    ]
    reference_ics = [
        component for component in reference_components
        if normalize_component_class(component["class"]) == "ic"
    ]
    if len(reference_caps) < 2 or not reference_ics:
        return None, False, None

    reference_caps.sort(key=lambda component: float(component["x"]))
    ref_left, ref_right = reference_caps[0], reference_caps[-1]
    ref_vector = np.array([
        float(ref_right["x"]) - float(ref_left["x"]),
        float(ref_right["y"]) - float(ref_left["y"]),
    ])
    ref_length = float(np.linalg.norm(ref_vector))
    if ref_length <= 0:
        return None, True, "Hai tụ mốc trong reference trùng vị trí."

    ref_unit = ref_vector / ref_length
    ref_ic = max(reference_ics, key=lambda component: float(component.get("confidence_reference", 0)))
    ref_ic_offset = np.array([
        float(ref_ic["x"]) - float(ref_left["x"]),
        float(ref_ic["y"]) - float(ref_left["y"]),
    ])
    target_projection = float(np.dot(ref_ic_offset, ref_unit) / ref_length)
    target_perpendicular = float(
        (ref_unit[0] * ref_ic_offset[1] - ref_unit[1] * ref_ic_offset[0])
        / ref_length
    )

    detected_caps = [
        detection for detection in detections
        if (normalize_component_class(detection["class"]) == "capacitor"
            and detection["confidence"] >= conf_thresh)
    ]
    detected_ics = [
        detection for detection in detections
        if (normalize_component_class(detection["class"]) == "ic"
            and detection["confidence"] >= conf_thresh)
    ]
    if len(detected_caps) < 2 or not detected_ics:
        return (
            None,
            True,
            "Cần phát hiện được ít nhất hai tụ và một IC để xác định hướng PCB.",
        )

    candidates = []
    for left_index, first_cap in enumerate(detected_caps):
        for second_cap in detected_caps[left_index + 1:]:
            for source_left, source_right in (
                (first_cap, second_cap),
                (second_cap, first_cap),
            ):
                source_vector = np.array([
                    source_right["cx"] - source_left["cx"],
                    source_right["cy"] - source_left["cy"],
                ])
                source_length = float(np.linalg.norm(source_vector))
                if source_length <= 0:
                    continue
                source_unit = source_vector / source_length
                for ic in detected_ics:
                    ic_offset = np.array([
                        ic["cx"] - source_left["cx"],
                        ic["cy"] - source_left["cy"],
                    ])
                    projection = float(np.dot(ic_offset, source_unit) / source_length)
                    perpendicular = float(
                        (source_unit[0] * ic_offset[1] - source_unit[1] * ic_offset[0])
                        / source_length
                    )
                    projection_error = abs(projection - target_projection)
                    perpendicular_error = abs(perpendicular - target_perpendicular)
                    score = math.hypot(projection_error, perpendicular_error)
                    if projection_error <= 0.20 and perpendicular_error <= 0.15:
                        candidates.append((
                            score,
                            -(source_left["confidence"] + source_right["confidence"] + ic["confidence"]),
                            source_left,
                            source_right,
                            ic,
                        ))

    if not candidates:
        return (
            None,
            True,
            "Không tìm thấy cặp tụ và IC có bố cục khớp với các mốc trong reference.",
        )

    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
    score, _, source_left, source_right, ic = candidates[0]
    source_angle = math.atan2(
        source_right["cy"] - source_left["cy"],
        source_right["cx"] - source_left["cx"],
    )
    reference_angle = math.atan2(ref_vector[1], ref_vector[0])
    rotation_degrees = math.degrees(source_angle - reference_angle)
    return {
        "left_cap": source_left,
        "right_cap": source_right,
        "ic": ic,
        "rotation_degrees": rotation_degrees,
        "score": score,
    }, True, None


def rotate_board_to_reference(frame, rotation_degrees):
    height, width = frame.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    transform = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    corners = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]).reshape(-1, 1, 2)
    rotated_corners = cv2.transform(corners, transform).reshape(-1, 2)
    min_x, min_y = np.floor(rotated_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(rotated_corners.max(axis=0)).astype(int)
    transform[:, 2] -= (min_x, min_y)
    output_size = (max_x - min_x + 1, max_y - min_y + 1)
    rotated = cv2.warpAffine(
        frame,
        transform,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    homogeneous_transform = np.eye(3, dtype=np.float64)
    homogeneous_transform[:2] = transform
    return rotated, homogeneous_transform


def crop_board_to_reference_bounds(frame, reference_components, anchor_orientation, ref_shape):
    reference_caps = [
        component for component in reference_components
        if normalize_component_class(component["class"]) == "capacitor"
    ]
    reference_caps.sort(key=lambda component: float(component["x"]))
    reference_ics = [
        component for component in reference_components
        if normalize_component_class(component["class"]) == "ic"
    ]
    if len(reference_caps) < 2 or not reference_ics:
        return frame, np.eye(3, dtype=np.float64), False

    ref_left, ref_right = reference_caps[0], reference_caps[-1]
    ref_ic = max(
        reference_ics,
        key=lambda component: float(component.get("confidence_reference", 0)),
    )
    source_points = np.float32([
        [anchor_orientation["left_cap"]["cx"], anchor_orientation["left_cap"]["cy"]],
        [anchor_orientation["right_cap"]["cx"], anchor_orientation["right_cap"]["cy"]],
        [anchor_orientation["ic"]["cx"], anchor_orientation["ic"]["cy"]],
    ])
    reference_points = np.float32([
        [ref_left["x"], ref_left["y"]],
        [ref_right["x"], ref_right["y"]],
        [ref_ic["x"], ref_ic["y"]],
    ])
    affine = cv2.getAffineTransform(source_points, reference_points)
    if (not np.all(np.isfinite(affine))
            or np.linalg.det(affine[:, :2]) <= 0):
        return frame, np.eye(3, dtype=np.float64), False

    ref_height, ref_width = ref_shape[:2]
    reference_corners = np.float32([
        [0, 0],
        [ref_width - 1, 0],
        [ref_width - 1, ref_height - 1],
        [0, ref_height - 1],
    ]).reshape(-1, 1, 2)
    source_corners = cv2.transform(
        reference_corners, cv2.invertAffineTransform(affine)
    ).reshape(-1, 2)
    frame_height, frame_width = frame.shape[:2]
    if (not np.all(np.isfinite(source_corners))
            or np.any(source_corners[:, 0] < 0)
            or np.any(source_corners[:, 0] >= frame_width)
            or np.any(source_corners[:, 1] < 0)
            or np.any(source_corners[:, 1] >= frame_height)):
        return frame, np.eye(3, dtype=np.float64), False

    cap_distance = float(np.linalg.norm(source_points[1] - source_points[0]))
    margin = max(4, int(round(0.10 * cap_distance)))
    x1 = max(0, int(np.floor(source_corners[:, 0].min())) - margin)
    y1 = max(0, int(np.floor(source_corners[:, 1].min())) - margin)
    x2 = min(frame_width, int(np.ceil(source_corners[:, 0].max())) + margin + 1)
    y2 = min(frame_height, int(np.ceil(source_corners[:, 1].max())) + margin + 1)
    if (x2 <= x1 or y2 <= y1
            or (x2 - x1) * (y2 - y1) >= 0.90 * frame_width * frame_height):
        return frame, np.eye(3, dtype=np.float64), False

    translation = np.array([
        [1, 0, -x1],
        [0, 1, -y1],
        [0, 0, 1],
    ], dtype=np.float64)
    return frame[y1:y2, x1:x2], translation, True


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
    )) + 1
    height = int(max(
        np.linalg.norm(top_right - bottom_right),
        np.linalg.norm(top_left - bottom_left),
    )) + 1
    if width < 1 or height < 1:
        return None, None

    destination = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(frame, transform, (width, height)), transform


def rectify_board_to_reference(frame, corners, reference_shape):
    reference_height, reference_width = reference_shape[:2]
    if reference_width < 1 or reference_height < 1:
        return None, None

    destination = np.float32([
        [0, 0],
        [reference_width - 1, 0],
        [reference_width - 1, reference_height - 1],
        [0, reference_height - 1],
    ])
    transform = cv2.getPerspectiveTransform(
        np.asarray(corners, dtype=np.float32), destination
    )
    if not np.all(np.isfinite(transform)):
        return None, None

    rectified = cv2.warpPerspective(
        frame,
        transform,
        (reference_width, reference_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return rectified, transform


def warp_board_to_reference(frame, transform, reference_shape):
    reference_height, reference_width = reference_shape[:2]
    transform = np.asarray(transform, dtype=np.float64)
    if (
        reference_width < 1
        or reference_height < 1
        or transform.shape != (3, 3)
        or not np.all(np.isfinite(transform))
    ):
        return None
    return cv2.warpPerspective(
        frame,
        transform,
        (reference_width, reference_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


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
        logger.debug("Homography rejected: contains non-finite values.")
        return False
    determinant = float(np.linalg.det(H[:2, :2]))
    if abs(determinant) < 1e-6:
        logger.debug("Homography rejected: affine determinant %.6g is too small.", determinant)
        return False
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError as exc:
        logger.debug("Homography rejected: matrix is not invertible: %s", exc)
        return False
    corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2).astype(np.float32)
    if not np.all(np.isfinite(quad)) or not cv2.isContourConvex(quad):
        logger.debug("Homography rejected: projected reference corners are non-finite or non-convex.")
        return False
    area = cv2.contourArea(quad)
    frame_area = float(fw * fh)
    area_ratio = area / frame_area
    valid = MIN_BOARD_AREA_RATIO < area_ratio < MAX_BOARD_AREA_RATIO
    if not valid:
        logger.debug(
            "Homography rejected: projected PCB area ratio %.4f outside (%.4f, %.4f).",
            area_ratio,
            MIN_BOARD_AREA_RATIO,
            MAX_BOARD_AREA_RATIO,
        )
    else:
        logger.debug(
            "Homography valid: determinant=%.6g projected_area_ratio=%.4f.",
            determinant,
            area_ratio,
        )
    return valid


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
            logger.info(
                "ORB alignment unavailable: reference has %d keypoints and descriptors=%s (minimum good matches=%d).",
                len(self.ref_kp),
                self.ref_desc is not None,
                MIN_GOOD_MATCHES,
            )
            return None
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frame_kp, frame_desc = self.orb.detectAndCompute(gray, None)
        frame_keypoint_count = len(frame_kp) if frame_kp is not None else 0
        logger.info(
            "ORB keypoints: reference=%d frame=%d descriptors_available=%s.",
            len(self.ref_kp),
            frame_keypoint_count,
            frame_desc is not None,
        )
        if frame_desc is None or frame_kp is None or frame_keypoint_count < MIN_GOOD_MATCHES:
            logger.info(
                "ORB alignment failed: frame keypoints/descriptors below minimum (%d).",
                MIN_GOOD_MATCHES,
            )
            return None
        try:
            matches = self.matcher.knnMatch(frame_desc, self.ref_desc, k=2)
        except cv2.error as exc:
            logger.warning("ORB matching failed: %s", exc)
            return None
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance]
        logger.info(
            "ORB matches: candidates=%d good=%d (ratio_test=%.2f).",
            len(matches),
            len(good),
            RATIO_TEST,
        )
        if len(good) < MIN_GOOD_MATCHES:
            logger.info(
                "ORB alignment failed: only %d good matches, need at least %d.",
                len(good),
                MIN_GOOD_MATCHES,
            )
            return None
        src_pts = np.float32([frame_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
        if H is None or mask is None:
            logger.info("ORB alignment failed: findHomography returned no matrix or inlier mask.")
            return None
        inliers = int(mask.sum())
        inlier_ratio = inliers / len(good)
        logger.info(
            "ORB RANSAC: inliers=%d/%d ratio=%.3f (minimum=%d, ratio=%.2f).",
            inliers,
            len(good),
            inlier_ratio,
            MIN_INLIERS,
            MIN_INLIER_RATIO,
        )
        if inliers < MIN_INLIERS or inlier_ratio < MIN_INLIER_RATIO:
            logger.info("ORB alignment failed: insufficient RANSAC inliers.")
            return None
        if not is_valid_homography(H, self.ref_shape, frame.shape):
            logger.info("ORB alignment failed: homography validation rejected the matrix.")
            return None
        logger.info("ORB alignment succeeded.")
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
    return corners


def estimate_component_alignment(reference_components, detections, tolerance, conf_thresh, crop_shape=None, ref_shape=None):
    """
    Ước lượng ma trận Affine biến đổi tọa độ từ khung hình quét sang mạch chuẩn.
    Kết hợp:
    1. Căn chỉnh theo tỷ lệ kích thước (Scale & Aspect Ratio) theo góc 0° và 180°.
    2. Ghép cặp theo các loại linh kiện đặc trưng (IC, diode, capacitor, inductor...) qua RANSAC.
    """
    if not reference_components or not detections:
        logger.info(
            "Component alignment failed: reference_count=%d detection_count=%d.",
            len(reference_components),
            len(detections),
        )
        return None

    valid_dets = [d for d in detections if d["confidence"] >= conf_thresh]
    reference_counts = Counter(
        normalize_component_class(c["class"]) for c in reference_components
    )
    detection_counts = Counter(
        normalize_component_class(d["class"]) for d in valid_dets
    )
    logger.info(
        "Component alignment counts: reference=%d detections=%d valid_detections=%d "
        "confidence_threshold=%.3f reference_classes=%s detection_classes=%s.",
        len(reference_components),
        len(detections),
        len(valid_dets),
        conf_thresh,
        dict(reference_counts),
        dict(detection_counts),
    )
    logger.info(
        "Anchor condition (2 capacitors + 1 IC): reference capacitors=%d, "
        "detected capacitors=%d, reference IC=%d, detected IC=%d.",
        reference_counts.get("capacitor", 0),
        detection_counts.get("capacitor", 0),
        reference_counts.get("ic", 0),
        detection_counts.get("ic", 0),
    )
    if len(valid_dets) < 2:
        logger.info(
            "Component alignment failed: %d detections meet confidence threshold; need at least 2.",
            len(valid_dets),
        )
        return None

    reference_by_class = {}
    detections_by_class = {}
    for component in reference_components:
        reference_by_class.setdefault(
            normalize_component_class(component["class"]), []
        ).append(component)
    for detection in valid_dets:
        detections_by_class.setdefault(
            normalize_component_class(detection["class"]), []
        ).append(detection)

    common_classes = sorted(
        reference_by_class.keys() & detections_by_class.keys()
    )
    unique_classes = [
        cls for cls in common_classes
        if len(reference_by_class[cls]) == 1 and len(detections_by_class[cls]) == 1
    ]
    logger.info(
        "Affine anchor classes: shared=%s unique_one_to_one=%s count=%d.",
        common_classes,
        unique_classes,
        len(unique_classes),
    )

    for cls in common_classes:
        ref_items = reference_by_class[cls]
        det_items = detections_by_class[cls]
        if len(ref_items) > 1 or len(det_items) > 1:
            ref_distances = [
                round(
                    float(np.hypot(
                        float(ref_items[i]["x"]) - float(ref_items[j]["x"]),
                        float(ref_items[i]["y"]) - float(ref_items[j]["y"]),
                    )),
                    2,
                )
                for i in range(len(ref_items))
                for j in range(i + 1, len(ref_items))
            ]
            det_distances = [
                round(
                    float(np.hypot(
                        det_items[i]["cx"] - det_items[j]["cx"],
                        det_items[i]["cy"] - det_items[j]["cy"],
                    )),
                    2,
                )
                for i in range(len(det_items))
                for j in range(i + 1, len(det_items))
            ]
            logger.info(
                "Repeated-class pairwise geometry: class=%s reference_pairs=%d "
                "reference_distances_px=%s detection_pairs=%d detection_distances_px=%s.",
                cls,
                len(ref_distances),
                ref_distances,
                len(det_distances),
                det_distances,
            )

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
            for candidate_name, H_cand in scale_candidates:
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

                logger.debug(
                    "Scale alignment candidate=%s matched_same_class_pairs=%d.",
                    candidate_name,
                    len(matched_pairs_src),
                )
                if len(matched_pairs_src) >= 3 and len(matched_pairs_src) > best_inliers:
                    src_arr = np.float32(matched_pairs_src)
                    dst_arr = np.float32(matched_pairs_dst)
                    affine, inliers = cv2.estimateAffine2D(src_arr, dst_arr, method=cv2.RANSAC, ransacReprojThreshold=tolerance)
                    if affine is not None and inliers is not None and int(inliers.sum()) >= 3:
                        best_inliers = len(matched_pairs_src)
                        best_H = np.eye(3, dtype=np.float64)
                        best_H[:2, :] = affine
                        logger.info(
                            "Scale alignment candidate=%s produced affine from %d pairs (%d RANSAC inliers).",
                            candidate_name,
                            len(matched_pairs_src),
                            int(inliers.sum()),
                        )
                    else:
                        best_inliers = len(matched_pairs_src)
                        best_H = H_cand
                        logger.info(
                            "Scale alignment candidate=%s retained scale transform from %d class-matched pairs; affine RANSAC was insufficient.",
                            candidate_name,
                            len(matched_pairs_src),
                        )

    # 2. RANSAC dựa trên các anchor linh kiện cùng class
    unique_pairs = []
    for cls in common_classes:
        if len(reference_by_class[cls]) == 1 and len(detections_by_class[cls]) == 1:
            unique_pairs.append((detections_by_class[cls][0], reference_by_class[cls][0]))

    if len(unique_pairs) >= 4:
        src = np.float32([[d["cx"], d["cy"]] for d, _ in unique_pairs])
        dst = np.float32([[float(r["x"]), float(r["y"])] for _, r in unique_pairs])
        reference_scale = max(ref_shape[:2]) if ref_shape is not None else 0
        reprojection_threshold = max(
            5.0,
            min(float(tolerance), 0.02 * reference_scale)
            if reference_scale
            else float(tolerance),
        )
        try:
            homography, inliers = cv2.findHomography(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=reprojection_threshold,
            )
        except cv2.error as exc:
            logger.warning("Component homography estimation failed: %s", exc)
            homography, inliers = None, None

        inlier_count = int(inliers.sum()) if inliers is not None else 0
        inlier_ratio = inlier_count / len(unique_pairs)
        geometry_valid = (
            ref_shape is None
            or crop_shape is None
            or (
                homography is not None
                and is_valid_homography(homography, ref_shape, crop_shape)
            )
        )
        if (
            homography is not None
            and inliers is not None
            and inlier_count >= 4
            and inlier_ratio >= 0.75
            and geometry_valid
        ):
            logger.info(
                "Unique-class component homography succeeded: anchors=%d "
                "inliers=%d ratio=%.3f reprojection_threshold=%.2f.",
                len(unique_pairs),
                inlier_count,
                inlier_ratio,
                reprojection_threshold,
            )
            return homography
        logger.info(
            "Unique-class component homography rejected: anchors=%d inliers=%d "
            "ratio=%.3f geometry_valid=%s.",
            len(unique_pairs),
            inlier_count,
            inlier_ratio,
            geometry_valid,
        )

    if len(unique_pairs) >= 3:
        src = np.float32([[d["cx"], d["cy"]] for d, r in unique_pairs])
        dst = np.float32([[float(r["x"]), float(r["y"])] for d, r in unique_pairs])
        affine, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=tolerance)
        if affine is not None and inliers is not None and int(inliers.sum()) >= 3:
            H = np.eye(3, dtype=np.float64)
            H[:2, :] = affine
            logger.info(
                "Unique-class affine alignment succeeded: anchors=%d inliers=%d.",
                len(unique_pairs),
                int(inliers.sum()),
            )
            return H
        logger.info(
            "Unique-class affine alignment failed: anchors=%d affine=%s inliers=%s.",
            len(unique_pairs),
            affine is not None,
            int(inliers.sum()) if inliers is not None else 0,
        )
    else:
        logger.info(
            "Unique-class affine alignment unavailable: found %d unique class pairs; need at least 3.",
            len(unique_pairs),
        )

    # Nếu chỉ có 1 hoặc 2 unique pairs, thử kết hợp với các class có ít instance (như tụ hoặc cuộn cảm)
    if len(unique_pairs) in (1, 2) and best_inliers < 4:
        small_classes = [c for c in common_classes if 1 < len(reference_by_class[c]) * len(detections_by_class[c]) <= 6]
        logger.info(
            "Repeated-class affine candidates: unique_anchors=%d repeated_classes=%s prior_best_pairs=%d.",
            len(unique_pairs),
            small_classes,
            best_inliers,
        )
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
                        logger.debug(
                            "Repeated-class candidate improved affine: class=%s score=%d.",
                            extra_cls,
                            cur_inliers,
                        )

    if best_H is not None and best_inliers >= 3:
        logger.info(
            "Component affine alignment succeeded: inlier_pairs=%d.",
            best_inliers,
        )
        return best_H

    logger.info(
        "Component affine alignment failed: best_inlier_pairs=%d (minimum=3).",
        best_inliers,
    )
    return None


def _hungarian_assignment(cost_matrix):
    """Return minimum-cost (row, column) pairs using the Hungarian algorithm."""
    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("Cost matrix must be two-dimensional.")
    if not np.all(np.isfinite(costs)):
        raise ValueError("Cost matrix must contain only finite values.")
    if costs.size == 0:
        return []

    transposed = costs.shape[0] > costs.shape[1]
    if transposed:
        costs = costs.T

    row_count, column_count = costs.shape
    row_potentials = np.zeros(row_count + 1, dtype=np.float64)
    column_potentials = np.zeros(column_count + 1, dtype=np.float64)
    assigned_row = np.zeros(column_count + 1, dtype=np.int64)
    previous_column = np.zeros(column_count + 1, dtype=np.int64)

    for row in range(1, row_count + 1):
        assigned_row[0] = row
        current_column = 0
        minimum_reduced_cost = np.full(column_count + 1, np.inf)
        visited_columns = np.zeros(column_count + 1, dtype=bool)

        while True:
            visited_columns[current_column] = True
            current_row = assigned_row[current_column]
            delta = np.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if visited_columns[column]:
                    continue
                reduced_cost = (
                    costs[current_row - 1, column - 1]
                    - row_potentials[current_row]
                    - column_potentials[column]
                )
                if reduced_cost < minimum_reduced_cost[column]:
                    minimum_reduced_cost[column] = reduced_cost
                    previous_column[column] = current_column
                if minimum_reduced_cost[column] < delta:
                    delta = minimum_reduced_cost[column]
                    next_column = column

            for column in range(column_count + 1):
                if visited_columns[column]:
                    row_potentials[assigned_row[column]] += delta
                    column_potentials[column] -= delta
                else:
                    minimum_reduced_cost[column] -= delta
            current_column = next_column
            if assigned_row[current_column] == 0:
                break

        while True:
            next_column = previous_column[current_column]
            assigned_row[current_column] = assigned_row[next_column]
            current_column = next_column
            if current_column == 0:
                break

    assignments = [
        (int(assigned_row[column] - 1), column - 1)
        for column in range(1, column_count + 1)
        if assigned_row[column] != 0
    ]
    if transposed:
        assignments = [(column, row) for row, column in assignments]
    return sorted(assignments)


def match_components(
    reference_components,
    detections,
    tolerance,
    require_same_class=True,
    allow_location_class_mismatch=False,
):
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
    valid_expected_size = np.isfinite(exp_w) & np.isfinite(exp_h) & (exp_w > 0) & (exp_h > 0)
    valid_detection_size = (box_w > 0) & (box_h > 0)
    has_size_data = valid_expected_size & valid_detection_size
    too_large = has_size_data & ((box_w > 3.0 * exp_w) | (box_h > 3.0 * exp_h))
    contains &= ~too_large

    base_tolerance = max(float(tolerance), 0.0)
    adaptive_tolerance = np.full((num_ref, 1), base_tolerance)
    if np.any(valid_expected_size):
        expected_diagonal = np.hypot(
            np.nan_to_num(exp_w, nan=0.0),
            np.nan_to_num(exp_h, nan=0.0),
        )
        size_based_tolerance = np.minimum(
            0.25 * expected_diagonal,
            max(base_tolerance, 24.0) * 1.5,
        )
        adaptive_tolerance = np.where(
            valid_expected_size,
            np.maximum(adaptive_tolerance, size_based_tolerance),
            adaptive_tolerance,
        )
    allowed = (dist <= adaptive_tolerance) & ~too_large
    ref_cls = np.array([normalize_component_class(c["class"]) for c in reference_components])
    det_cls = np.array([normalize_component_class(d["class"]) for d in detections])
    class_match = ref_cls[:, None] == det_cls[None, :]
    if require_same_class:
        allowed &= class_match

    location_class_mismatch = np.zeros((num_ref, num_det), dtype=bool)
    if require_same_class and allow_location_class_mismatch:
        mismatch_radius = np.minimum(
            np.maximum(
                24.0,
                0.25 * np.hypot(
                    np.nan_to_num(exp_w, nan=0.0),
                    np.nan_to_num(exp_h, nan=0.0),
                ),
            ),
            np.maximum(adaptive_tolerance, 24.0),
        )
        location_class_mismatch = (
            ~class_match
            & (dist <= mismatch_radius)
            & ~too_large
        )

    size_penalty = np.zeros((num_ref, num_det), dtype=np.float64)
    if np.any(has_size_data):
        safe_box_w = np.maximum(box_w, 1e-6)
        safe_box_h = np.maximum(box_h, 1e-6)
        width_ratio = np.maximum(safe_box_w / np.maximum(exp_w, 1e-6), 1e-6)
        height_ratio = np.maximum(safe_box_h / np.maximum(exp_h, 1e-6), 1e-6)
        size_penalty = np.where(
            has_size_data,
            np.abs(np.log(width_ratio)) + np.abs(np.log(height_ratio)),
            0.0,
        )

    distance_scale = np.maximum(adaptive_tolerance, 1.0)
    pair_cost = dist / distance_scale + 0.25 * size_penalty
    unmatched_cost = 1_000_000.0
    invalid_cost = unmatched_cost * (num_ref + num_det + 1)
    assignment_cost = np.full(
        (num_ref, num_det + num_ref), unmatched_cost, dtype=np.float64
    )
    assignment_cost[:, :num_det] = np.where(allowed, pair_cost, invalid_cost)

    assignments = _hungarian_assignment(assignment_cost)
    matches = []
    for ref_index, det_index in assignments:
        if det_index >= num_det or not allowed[ref_index, det_index]:
            continue
        matches.append({
            "det_index": det_index,
            "ref_index": ref_index,
            "distance": float(dist[ref_index, det_index]),
            "size_penalty": float(size_penalty[ref_index, det_index]),
            "class_match": bool(class_match[ref_index, det_index]),
        })

    if np.any(location_class_mismatch):
        used_ref_indices = {match["ref_index"] for match in matches}
        used_det_indices = {match["det_index"] for match in matches}
        remaining_ref_indices = [
            index for index in range(num_ref) if index not in used_ref_indices
        ]
        remaining_det_indices = [
            index for index in range(num_det) if index not in used_det_indices
        ]
        fallback_allowed = location_class_mismatch[
            np.ix_(remaining_ref_indices, remaining_det_indices)
        ]
        if fallback_allowed.size and np.any(fallback_allowed):
            fallback_cost = pair_cost[
                np.ix_(remaining_ref_indices, remaining_det_indices)
            ]
            fallback_assignment_cost = np.full(
                (
                    len(remaining_ref_indices),
                    len(remaining_det_indices) + len(remaining_ref_indices),
                ),
                unmatched_cost,
                dtype=np.float64,
            )
            fallback_assignment_cost[:, :len(remaining_det_indices)] = np.where(
                fallback_allowed,
                fallback_cost,
                invalid_cost,
            )
            for local_ref_index, local_det_index in _hungarian_assignment(
                fallback_assignment_cost
            ):
                if (
                    local_det_index >= len(remaining_det_indices)
                    or not fallback_allowed[local_ref_index, local_det_index]
                ):
                    continue
                ref_index = remaining_ref_indices[local_ref_index]
                det_index = remaining_det_indices[local_det_index]
                matches.append({
                    "det_index": det_index,
                    "ref_index": ref_index,
                    "distance": float(dist[ref_index, det_index]),
                    "size_penalty": float(size_penalty[ref_index, det_index]),
                    "class_match": False,
                })

    matches.sort(key=lambda match: match["ref_index"])
    matched_detections = {match["det_index"] for match in matches}
    used_references = {match["ref_index"] for match in matches}
    logger.info(
        "Component assignment: matches=%d references=%d detections=%d one_to_one=%s.",
        len(matches),
        num_ref,
        num_det,
        len(matched_detections) == len(matches) == len(used_references),
    )
    for match in matches:
        logger.info(
            "Component match: ref_index=%d det_index=%d center_distance=%.2fpx size_penalty=%.3f.",
            match["ref_index"],
            match["det_index"],
            match["distance"],
            match["size_penalty"],
        )
    return matches, matched_detections, used_references


def evaluate_board_status(reference_components, detections, matches, used_references, aligned,
                          missing_allowed=DEFAULT_MISSING_ALLOWED, extra_allowed=DEFAULT_EXTRA_ALLOWED,
                          extra_min_conf=DEFAULT_EXTRA_MIN_CONF, conf_thresh=DEFAULT_CONF):
    stats = {
        "matched": len(matches),
        "class_mismatch": sum(
            not match.get("class_match", True) for match in matches
        ),
        "missing": 0,
        "extra": 0,
        "missing_components": [],
        "extra_components": [],
    }

    if aligned:
        matched_detections = {m["det_index"] for m in matches}
        matched_reference_indices = {
            index for index in used_references if 0 <= index < len(reference_components)
        }
        stats["missing_components"] = [
            c["class"] for i, c in enumerate(reference_components)
            if i not in matched_reference_indices
        ]
        stats["missing"] = len(stats["missing_components"])
        stats["extra_detection_indices"] = [
            index
            for index, detection in enumerate(detections)
            if index not in matched_detections
            and float(detection.get("confidence", 0.0)) >= extra_min_conf
        ]
        stats["extra"] = len(stats["extra_detection_indices"])
        stats["extra_components"] = [
            detections[index]["class"]
            for index in stats["extra_detection_indices"]
        ]
        if (
            len({m["det_index"] for m in matches}) != len(matches)
            or len({m["ref_index"] for m in matches}) != len(matches)
        ):
            logger.error("Component matches violate one-to-one assignment.")
        logger.info(
            "Reference evaluation: matched=%d missing=%d missing_components=%s.",
            stats["matched"],
            stats["missing"],
            stats["missing_components"],
        )
        # A configured allowance must not hide a missing reference component.
        if stats["missing"] > 0:
            return STATUS_FAULTY, stats
        if stats["class_mismatch"] or stats["extra"] > extra_allowed:
            return STATUS_REVIEW, stats
        return STATUS_OK, stats

    return STATUS_UNALIGNED, stats


def evaluate_component_inventory(reference_components, detections):
    """Evaluate the total detected component count against the reference."""
    reference_count = len(reference_components)
    detected_count = len(detections)
    matched = min(reference_count, detected_count)
    missing = max(reference_count - detected_count, 0)
    extra = max(detected_count - reference_count, 0)
    stats = {
        "matched": matched,
        "class_mismatch": 0,
        "missing": missing,
        "extra": extra,
        "inventory_mismatch_classes": int(missing > 0 or extra > 0),
        "missing_components": ["Linh kiện"] * missing,
        "extra_components": ["Linh kiện"] * extra,
    }
    status = (
        STATUS_OK
        if missing == 0 and extra == 0
        else STATUS_FAULTY
    )
    comparison = [{
        "class": "Tổng số linh kiện",
        "reference_count": reference_count,
        "detected_count": detected_count,
        "matched": matched,
        "missing": missing,
        "extra": extra,
    }]
    return status, stats, comparison


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
        self.sahi_model = None
        self.mono_cropper = None
        self.reference_data = None
        self.reference_components = []
        self.reference_image = None
        self.aligner = None
        self.reference_mounting_holes = []

        self.load_model()
        self.load_reference()

    def load_model(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Không tìm thấy model YOLO tại: {self.model_path}")
        print(f"[PCBPipeline] Loading YOLO model: {self.model_path} on {self.device}")
        if self.use_cuda:
            print(f"[PCBPipeline] CUDA inference enabled: {torch.cuda.get_device_name(self.device)}")
        else:
            print("[PCBPipeline] CUDA unavailable; using CPU inference.")
        self.model = YOLO(self.model_path)
        if self.use_cuda:
            self.model.to(f"cuda:{self.device}")
        sahi_device = "cuda:0" if self.use_cuda else "cpu"
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model=self.model,
            confidence_threshold=DEFAULT_CONF,
            device=sahi_device,
            image_size=DEFAULT_IMGSZ,
        )
        print("[PCBPipeline] SAHI sliced inference initialized with the shared YOLO model.")
        try:
            from pcb_crop_engine import PCBMonoCropper
            self.mono_cropper = PCBMonoCropper(model=self.model, conf_thresh=DEFAULT_CONF, inference_size=1024)
            print("[PCBPipeline] PCBMonoCropper initialized successfully.")
        except Exception as e:
            print(f"[PCBPipeline] PCBMonoCropper init warning: {e}")
            self.mono_cropper = None

    def load_reference(self):
        self.reference_mounting_holes = []
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
                self.reference_mounting_holes = find_reference_mounting_holes(
                    self.reference_image, self.reference_components
                )
                logger.info(
                    "Reference mounting-hole regions detected: count=%d.",
                    len(self.reference_mounting_holes),
                )
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
            self.reference_mounting_holes = []

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

    def detect_components(
        self, frame, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ, fast_mode=False
    ):
        if self.sahi_model is None:
            raise RuntimeError("SAHI sliced inference is not initialized.")
        image_height, image_width = frame.shape[:2]
        slice_size = FAST_SAHI_SLICE_SIZE if fast_mode else SAHI_SLICE_SIZE
        self.sahi_model.image_size = imgsz
        full_frame_results = self.model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            device=self.device,
            augment=not fast_mode,
            agnostic_nms=False,
            verbose=False,
        )
        if not full_frame_results:
            raise RuntimeError("YOLO không trả về kết quả nhận diện toàn ảnh.")

        full_frame_detections = []
        full_frame_boxes = full_frame_results[0].boxes
        if full_frame_boxes is not None:
            for index in range(len(full_frame_boxes)):
                x1, y1, x2, y2 = (
                    float(value)
                    for value in full_frame_boxes.xyxy[index].cpu().tolist()
                )
                class_id = int(full_frame_boxes.cls[index].item())
                full_frame_detections.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": float(full_frame_boxes.conf[index].item()),
                    "class_id": class_id,
                    "class_name": self.model.names[class_id],
                })

        prediction = get_sliced_prediction(
            image=frame,
            detection_model=self.sahi_model,
            slice_height=min(slice_size, image_height),
            slice_width=min(slice_size, image_width),
            overlap_height_ratio=SAHI_OVERLAP_RATIO,
            overlap_width_ratio=SAHI_OVERLAP_RATIO,
            perform_standard_pred=False,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
            confidence_threshold=conf,
            verbose=0,
            progress_bar=False,
            batch_size=(
                GPU_SAHI_BATCH_SIZE
                if self.use_cuda and not fast_mode
                else 1
            ),
        )
        sahi_detections = []
        for prediction_object in prediction.object_prediction_list:
            x1, y1, x2, y2 = (
                float(value) for value in prediction_object.bbox.to_xyxy()
            )
            cls_id = int(prediction_object.category.id)
            cls_name = prediction_object.category.name
            sahi_detections.append({
                "box": (x1, y1, x2, y2),
                "confidence": float(prediction_object.score.value),
                "class_id": cls_id,
                "class_name": cls_name,
            })
        full_frame_detections = self._filter_unsupported_classes(
            full_frame_detections
        )
        sahi_detections = self._filter_unsupported_classes(sahi_detections)
        sahi_detections = self._merge_full_frame_and_sliced_detections(
            full_frame_detections, sahi_detections
        )
        sahi_detections = self._suppress_overlapping_detections(sahi_detections)
        if not sahi_detections:
            return []

        sahi_detections = self._suppress_ic_artifacts(sahi_detections)

        detections = []
        for detection in sahi_detections:
            x1, y1, x2, y2 = detection["box"]
            cls_id = detection["class_id"]
            cls_name = detection["class_name"]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            detections.append({
                "id": len(detections) + 1,
                "class_id": cls_id,
                "class": cls_name,
                "original_class": cls_name,
                "confidence": round(detection["confidence"], 4),
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
                "cx": round(center_x, 2),
                "cy": round(center_y, 2),
                "width": round(float(x2 - x1), 2),
                "height": round(float(y2 - y1), 2),
            })
        if fast_mode:
            logger.info(
                "Fast SAHI inference completed: image=%dx%d tile=%dx%d imgsz=%d detections=%d.",
                image_width,
                image_height,
                min(slice_size, image_width),
                min(slice_size, image_height),
                imgsz,
                len(detections),
            )
        else:
            logger.info(
                "SAHI YOLO inference completed: image=%dx%d tile=%dx%d detections=%d.",
                image_width,
                image_height,
                min(SAHI_SLICE_SIZE, image_width),
                min(SAHI_SLICE_SIZE, image_height),
                len(detections),
            )
        return detections

    @staticmethod
    def _are_duplicate_boxes(first_box, second_box):
        x1, y1, x2, y2 = first_box
        bx1, by1, bx2, by2 = second_box
        first_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        intersection = max(0.0, min(x2, bx2) - max(x1, bx1)) * max(
            0.0, min(y2, by2) - max(y1, by1)
        )
        smaller_area = min(first_area, second_area)
        larger_area = max(first_area, second_area)
        return (
            smaller_area > 0.0
            and intersection / smaller_area >= DETECTION_OVERLAP_THRESHOLD
            and larger_area / smaller_area <= DETECTION_MAX_AREA_RATIO
        )

    @staticmethod
    def _are_reference_local_duplicates(first_box, second_box, reference):
        x1, y1, x2, y2 = first_box
        bx1, by1, bx2, by2 = second_box
        first_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        smaller_area = min(first_area, second_area)
        if smaller_area <= 0.0:
            return False

        intersection = max(0.0, min(x2, bx2) - max(x1, bx1)) * max(
            0.0, min(y2, by2) - max(y1, by1)
        )
        if intersection / smaller_area < DETECTION_OVERLAP_THRESHOLD:
            return False

        reference_x = float(reference["x"])
        reference_y = float(reference["y"])
        if not (
            bx1 <= reference_x <= bx2
            and by1 <= reference_y <= by2
        ):
            return False

        expected_width = float(reference.get("box_width") or 0.0)
        expected_height = float(reference.get("box_height") or 0.0)
        if expected_width > 0.0 and expected_height > 0.0:
            center_radius = max(
                24.0,
                0.5 * math.hypot(expected_width, expected_height),
            )
        else:
            center_radius = max(
                24.0,
                0.5 * math.hypot(x2 - x1, y2 - y1),
            )
        candidate_x = (bx1 + bx2) / 2.0
        candidate_y = (by1 + by2) / 2.0
        return math.hypot(candidate_x - reference_x, candidate_y - reference_y) <= center_radius

    @staticmethod
    def _format_detection_annotation_label(detection):
        yolo_class = detection.get("yolo_class")
        if yolo_class and normalize_component_class(yolo_class) != normalize_component_class(
            detection["class"]
        ):
            return (
                f"{detection['class']} (YOLO: {yolo_class} "
                f"{detection['confidence']:.2f}) [REF]"
            )
        if (
            detection.get("class_match") is False
            and detection.get("matched_ref_class")
        ):
            return (
                f"{detection['matched_ref_class']} (YOLO: "
                f"{detection['class']} {detection['confidence']:.2f}) [REVIEW]"
            )
        return (
            f"{detection['class']} {detection['confidence']:.2f} "
            f"[{detection['status']}]"
        )

    @staticmethod
    def _apply_reference_class_authority(reference_components, detections, matches):
        for match in matches:
            if match["class_match"]:
                continue

            detection = detections[match["det_index"]]
            reference = reference_components[match["ref_index"]]
            yolo_class = detection.get("yolo_class", detection["class"])
            detection["yolo_class"] = yolo_class
            detection["class"] = reference["class"]
            detection["class_match"] = True
            detection["class_source"] = "reference"
            detection["yolo_class_match"] = False
            match["class_match"] = True
            match["yolo_class_match"] = False
            logger.info(
                "Reference class used as authoritative label: reference_id=%s "
                "reference_class=%s yolo_class=%s confidence=%.3f.",
                reference.get("id", match["ref_index"] + 1),
                reference["class"],
                yolo_class,
                detection["confidence"],
            )

    @staticmethod
    def _merge_full_frame_and_sliced_detections(full_frame_detections, sliced_detections):
        return PCBPipeline._suppress_overlapping_detections(
            list(full_frame_detections) + list(sliced_detections)
        )

    @staticmethod
    def _filter_unsupported_classes(detections):
        filtered_detections = []
        for detection in detections:
            class_name = normalize_component_class(detection["class_name"])
            if class_name in IGNORED_DETECTION_CLASSES:
                logger.info(
                    "Ignored unsupported detection class: class=%s confidence=%.3f.",
                    detection["class_name"],
                    detection["confidence"],
                )
                continue
            canonical_detection = dict(detection)
            canonical_detection["class_name"] = canonical_model_class_name(
                detection["class_name"]
            )
            filtered_detections.append(canonical_detection)
        return filtered_detections

    @staticmethod
    def _suppress_overlapping_detections(detections):
        """Suppress duplicate boxes only when they predict the same component class."""
        ranked_detections = sorted(
            enumerate(detections),
            key=lambda item: item[1]["confidence"],
            reverse=True,
        )
        kept_detections = []
        for original_index, detection in ranked_detections:
            duplicate_of = None
            for _, kept in kept_detections:
                detection_class = normalize_component_class(
                    detection.get("class_name", detection.get("class", ""))
                )
                kept_class = normalize_component_class(
                    kept.get("class_name", kept.get("class", ""))
                )
                if detection_class != kept_class:
                    continue
                if PCBPipeline._are_duplicate_boxes(
                    detection["box"], kept["box"]
                ):
                    duplicate_of = kept
                    break
            if duplicate_of is None:
                kept_detections.append((original_index, detection))
            else:
                logger.info(
                    "Suppressed overlapping detection: class=%s confidence=%.3f "
                    "overlaps class=%s confidence=%.3f.",
                    detection.get("class_name", detection.get("class", "unknown")),
                    detection["confidence"],
                    duplicate_of.get(
                        "class_name", duplicate_of.get("class", "unknown")
                    ),
                    duplicate_of["confidence"],
                )
        return [
            detection
            for _, detection in sorted(kept_detections, key=lambda item: item[0])
        ]

    @staticmethod
    def _suppress_class_conflicts_by_reference(
        reference_components, detections, tolerance
    ):
        """Prefer a reference-matching class when overlapping class hypotheses exist."""
        if not reference_components or not detections:
            return detections

        exact_matches, _, _ = match_components(
            reference_components,
            detections,
            tolerance,
            require_same_class=True,
        )
        suppressed_indices = set()
        exact_detection_indices = {
            match["det_index"] for match in exact_matches
        }
        for match in exact_matches:
            reference = reference_components[match["ref_index"]]
            matched_detection = detections[match["det_index"]]
            reference_class = normalize_component_class(reference["class"])
            matched_box = matched_detection.get("ref_bbox")
            if matched_box is None:
                continue
            for detection_index, detection in enumerate(detections):
                if detection_index in exact_detection_indices:
                    continue
                if normalize_component_class(detection["class"]) == reference_class:
                    continue
                candidate_box = detection.get("ref_bbox")
                if candidate_box is not None and (
                    PCBPipeline._are_duplicate_boxes(matched_box, candidate_box)
                    or PCBPipeline._are_reference_local_duplicates(
                        matched_box, candidate_box, reference
                    )
                ):
                    suppressed_indices.add(detection_index)

        location_matches, _, _ = match_components(
            reference_components,
            detections,
            tolerance,
            require_same_class=True,
            allow_location_class_mismatch=True,
        )
        for match in location_matches:
            if match["class_match"]:
                continue
            matched_index = match["det_index"]
            if matched_index in suppressed_indices:
                continue
            matched_detection = detections[matched_index]
            matched_box = matched_detection.get("ref_bbox")
            if matched_box is None:
                continue

            conflict_indices = [
                detection_index
                for detection_index, detection in enumerate(detections)
                if detection_index not in exact_detection_indices
                and detection_index not in suppressed_indices
                and normalize_component_class(detection["class"])
                != normalize_component_class(matched_detection["class"])
                and detection.get("ref_bbox") is not None
                and (
                    PCBPipeline._are_duplicate_boxes(
                        matched_box, detection["ref_bbox"]
                    )
                    or PCBPipeline._are_reference_local_duplicates(
                        matched_box,
                        detection["ref_bbox"],
                        reference_components[match["ref_index"]],
                    )
                )
            ]
            if not conflict_indices:
                continue

            winner_index = max(
                [matched_index, *conflict_indices],
                key=lambda index: detections[index]["confidence"],
            )
            suppressed_indices.update(
                index
                for index in [matched_index, *conflict_indices]
                if index != winner_index
            )

        if suppressed_indices:
            logger.info(
                "Suppressed overlapping class alternatives after reference match: "
                "count=%d.",
                len(suppressed_indices),
            )
        return [
            detection
            for index, detection in enumerate(detections)
            if index not in suppressed_indices
        ]

    @staticmethod
    def _suppress_ic_artifacts(detections):
        ic_boxes = [
            detection["box"]
            for detection in detections
            if normalize_component_class(detection["class_name"]) == "ic"
        ]
        filtered_detections = []
        for detection in detections:
            if normalize_component_class(detection["class_name"]) == "ic":
                filtered_detections.append(detection)
                continue
            x1, y1, x2, y2 = detection["box"]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            if any(
                ic_x1 <= center_x <= ic_x2 and ic_y1 <= center_y <= ic_y2
                for ic_x1, ic_y1, ic_x2, ic_y2 in ic_boxes
            ):
                logger.info(
                    "Suppressed detection centered inside IC box: class=%s "
                    "confidence=%.3f center=(%.1f, %.1f).",
                    detection["class_name"],
                    detection["confidence"],
                    center_x,
                    center_y,
                )
                continue
            filtered_detections.append(detection)
        return filtered_detections

    def crop_board(self, frame, fallback_full=False):
        # 1. Align the captured PCB directly to the reference frame.
        reference_corners = find_reference_board_corners(frame, self.aligner)
        if reference_corners is not None and self.reference_image is not None:
            rectified, transform = rectify_board_to_reference(
                frame, reference_corners, self.reference_image.shape
            )
            if rectified is not None:
                return rectified, transform, "reference_homography"

        # 2. PCBMonoCropper (Ưu tiên OBB AI theo cụm linh kiện + nắn thẳng)
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

        if not fallback_full:
            raise BoardCropError(
                "Không thể tự động xác định vùng PCB; đã dừng thay vì dùng toàn bộ khung hình."
            )

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
            logger.info(
                "Detection transformed to reference: class=%s center=(%.2f, %.2f) "
                "bbox=(%.2f, %.2f, %.2f, %.2f).",
                d["class"],
                d["ref_x"],
                d["ref_y"],
                *d["ref_bbox"],
            )
        return transformed

    def _filter_reference_roi_detections(
        self, detections, reference_indices, transform, tolerance
    ):
        accepted = []
        reference_indices = set(reference_indices)
        for detection in detections:
            reference_index = detection.get("_roi_ref_index")
            if reference_index not in reference_indices:
                continue
            mapped_detection = self._map_detections([detection], transform)[0]
            matches, _, _ = match_components(
                [self.reference_components[reference_index]],
                [mapped_detection],
                tolerance,
                require_same_class=True,
            )
            if matches:
                accepted.append(detection)
        return accepted

    def _detect_missing_reference_rois(
        self, frame, reference_indices, transform, imgsz
    ):
        if not reference_indices:
            return []

        try:
            inverse_transform = np.linalg.inv(transform)
        except np.linalg.LinAlgError as exc:
            logger.warning(
                "Reference ROI inference skipped: alignment transform is not invertible: %s.",
                exc,
            )
            return []

        frame_height, frame_width = frame.shape[:2]
        roi_detections = []
        for reference_index in reference_indices:
            component = self.reference_components[reference_index]
            expected_width = float(component.get("box_width") or 0.0)
            expected_height = float(component.get("box_height") or 0.0)
            roi_width = min(
                REFERENCE_ROI_MAX_SIZE,
                max(
                    REFERENCE_ROI_MIN_SIZE,
                    expected_width * REFERENCE_ROI_SIZE_SCALE,
                ),
            )
            roi_height = min(
                REFERENCE_ROI_MAX_SIZE,
                max(
                    REFERENCE_ROI_MIN_SIZE,
                    expected_height * REFERENCE_ROI_SIZE_SCALE,
                ),
            )
            center_x = float(component["x"])
            center_y = float(component["y"])
            reference_corners = np.float32([[
                [center_x - roi_width / 2, center_y - roi_height / 2],
                [center_x + roi_width / 2, center_y - roi_height / 2],
                [center_x + roi_width / 2, center_y + roi_height / 2],
                [center_x - roi_width / 2, center_y + roi_height / 2],
            ]])
            frame_corners = cv2.perspectiveTransform(
                reference_corners, inverse_transform
            ).reshape(-1, 2)
            if not np.all(np.isfinite(frame_corners)):
                logger.warning(
                    "Reference ROI skipped: non-finite projected corners for ref_index=%d.",
                    reference_index,
                )
                continue

            roi_x1 = int(np.floor(frame_corners[:, 0].min()))
            roi_y1 = int(np.floor(frame_corners[:, 1].min()))
            roi_x2 = int(np.ceil(frame_corners[:, 0].max())) + 1
            roi_y2 = int(np.ceil(frame_corners[:, 1].max())) + 1
            x1 = max(0, roi_x1)
            y1 = max(0, roi_y1)
            x2 = min(frame_width, roi_x2)
            y2 = min(frame_height, roi_y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                logger.info(
                    "Reference ROI skipped: projected region is outside the scan "
                    "for ref_index=%d.",
                    reference_index,
                )
                continue

            crop = frame[y1:y2, x1:x2]
            padding = (
                x1 - roi_x1,
                y1 - roi_y1,
                roi_x2 - x2,
                roi_y2 - y2,
            )
            if any(padding):
                crop = cv2.copyMakeBorder(
                    crop,
                    padding[1],
                    padding[3],
                    padding[0],
                    padding[2],
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            scaled_crop = cv2.resize(
                crop,
                None,
                fx=REFERENCE_ROI_INFERENCE_SCALE,
                fy=REFERENCE_ROI_INFERENCE_SCALE,
                interpolation=cv2.INTER_CUBIC,
            )
            results = self.model.predict(
                source=scaled_crop,
                conf=REFERENCE_ROI_CONFIDENCE,
                imgsz=max(imgsz, DEFAULT_IMGSZ),
                device=self.device,
                agnostic_nms=False,
                augment=True,
                verbose=False,
            )
            if not results:
                raise RuntimeError(
                    "YOLO không trả về kết quả nhận diện cho vùng reference."
                )
            roi_detection_count = len(roi_detections)
            boxes = results[0].boxes
            if boxes is not None:
                for detection_index in range(len(boxes)):
                    local_x1, local_y1, local_x2, local_y2 = (
                        float(value) / REFERENCE_ROI_INFERENCE_SCALE
                        for value in boxes.xyxy[detection_index].cpu().tolist()
                    )
                    class_id = int(boxes.cls[detection_index].item())
                    class_name = canonical_model_class_name(
                        self.model.names[class_id]
                    )
                    if normalize_component_class(class_name) in IGNORED_DETECTION_CLASSES:
                        continue
                    mapped_detection = {
                        "class_id": class_id,
                        "class": class_name,
                        "confidence": float(boxes.conf[detection_index].item()),
                        "x1": max(
                            0.0,
                            min(frame_width, local_x1 + roi_x1),
                        ),
                        "y1": max(
                            0.0,
                            min(frame_height, local_y1 + roi_y1),
                        ),
                        "x2": max(
                            0.0,
                            min(frame_width, local_x2 + roi_x1),
                        ),
                        "y2": max(
                            0.0,
                            min(frame_height, local_y2 + roi_y1),
                        ),
                        "id": -1,
                        "_roi_ref_index": reference_index,
                    }
                    if (
                        mapped_detection["x2"] <= mapped_detection["x1"]
                        or mapped_detection["y2"] <= mapped_detection["y1"]
                    ):
                        continue
                    mapped_detection["cx"] = (
                        mapped_detection["x1"] + mapped_detection["x2"]
                    ) / 2.0
                    mapped_detection["cy"] = (
                        mapped_detection["y1"] + mapped_detection["y2"]
                    ) / 2.0
                    mapped_detection["width"] = (
                        mapped_detection["x2"] - mapped_detection["x1"]
                    )
                    mapped_detection["height"] = (
                        mapped_detection["y2"] - mapped_detection["y1"]
                    )
                    roi_detections.append(mapped_detection)
            logger.info(
                "Reference ROI inference: ref_index=%d class=%s roi=(%d,%d,%d,%d) "
                "detections=%d imgsz=%d confidence=%.3f scale=%.1f.",
                reference_index,
                component["class"],
                x1,
                y1,
                x2,
                y2,
                len(roi_detections) - roi_detection_count,
                max(imgsz, DEFAULT_IMGSZ),
                REFERENCE_ROI_CONFIDENCE,
                REFERENCE_ROI_INFERENCE_SCALE,
            )

        if not roi_detections:
            return []

        return [
            {
                key: value
                for key, value in detection.items()
                if key != "box"
            }
            for detection in self._suppress_overlapping_detections(
                [
                    {
                        **detection,
                        "box": (
                            detection["x1"],
                            detection["y1"],
                            detection["x2"],
                            detection["y2"],
                        ),
                    }
                    for detection in roi_detections
                ]
            )
        ]

    @staticmethod
    def _suppress_reference_hole_detections(detections, hole_circles):
        if not hole_circles:
            return detections

        filtered_detections = []
        for detection in detections:
            center_x = detection.get("ref_x", detection["cx"])
            center_y = detection.get("ref_y", detection["cy"])
            bbox = detection.get(
                "ref_bbox",
                (
                    detection["x1"],
                    detection["y1"],
                    detection["x2"],
                    detection["y2"],
                ),
            )
            box_diagonal = math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
            matching_hole = next(
                (
                    (hole_x, hole_y, radius)
                    for hole_x, hole_y, radius in hole_circles
                    if math.hypot(center_x - hole_x, center_y - hole_y)
                    <= radius + 0.15 * box_diagonal
                ),
                None,
            )
            if matching_hole is not None:
                logger.info(
                    "Suppressed false detection over reference mounting hole: "
                    "class=%s center=(%.1f, %.1f) hole=(%.1f, %.1f, r=%.1f).",
                    detection["class"],
                    center_x,
                    center_y,
                    *matching_hole,
                )
                continue
            filtered_detections.append(detection)
        return filtered_detections

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
                    missing_allowed=DEFAULT_MISSING_ALLOWED, auto_crop=True,
                    fast_mode=False):
        start_time = time.time()

        if self.reference_image is None or not self.reference_components:
            raise RuntimeError("Chưa có dữ liệu mạch tham chiếu (reference). Vui lòng nạp hoặc tạo reference trước.")

        ref_h, ref_w = self.reference_image.shape[:2]
        tolerance = tolerance_ratio * max(ref_w, ref_h)
        logger.info(
            "PCB inspection started: frame=%dx%d reference=%dx%d reference_components=%d "
            "confidence=%.3f tolerance=%.2f auto_crop=%s.",
            frame.shape[1],
            frame.shape[0],
            ref_w,
            ref_h,
            len(self.reference_components),
            conf,
            tolerance,
            auto_crop,
        )

        crop_method = "full"
        if auto_crop:
            try:
                board_crop, board_transform, crop_method = self.crop_board(frame)
            except BoardCropError:
                logger.exception("PCB inspection stopped: automatic board crop failed.")
                raise
        else:
            board_crop = frame
            board_transform = np.eye(3, dtype=np.float64)
        logger.info(
            "PCB crop selected: method=%s frame=%dx%d crop=%dx%d transform_available=%s.",
            crop_method,
            frame.shape[1],
            frame.shape[0],
            board_crop.shape[1],
            board_crop.shape[0],
            board_transform is not None,
        )

        if np.mean(board_crop) < 60 or np.std(board_crop) < 30:
            det_frame = adjust_contrast(board_crop)
        else:
            det_frame = board_crop
        raw_detections = self.detect_components(
            det_frame, conf=conf, imgsz=imgsz, fast_mode=fast_mode
        )
        reference_counts = Counter(
            normalize_component_class(component["class"])
            for component in self.reference_components
        )
        detection_counts = Counter(
            normalize_component_class(detection["class"])
            for detection in raw_detections
        )
        logger.info(
            "PCB component counts: reference=%d %s; yolo=%d %s.",
            len(self.reference_components),
            dict(reference_counts),
            len(raw_detections),
            dict(detection_counts),
        )

        reference_rectified = crop_method == "reference_homography"
        if reference_rectified:
            anchor_orientation, anchor_required, anchor_error = None, False, None
        else:
            anchor_orientation, anchor_required, anchor_error = estimate_anchor_orientation(
                self.reference_components, raw_detections, conf
            )
        orientation_transform = (
            board_transform.copy()
            if crop_method == "reference_homography"
            else np.eye(3, dtype=np.float64)
        )
        pre_rotation_degrees = 0
        if anchor_required and anchor_orientation is None:
            for candidate_rotation in (90, -90, 180):
                candidate_crop, candidate_transform = rotate_board_to_reference(
                    board_crop, candidate_rotation
                )
                if np.mean(candidate_crop) < 60 or np.std(candidate_crop) < 30:
                    candidate_frame = adjust_contrast(candidate_crop)
                else:
                    candidate_frame = candidate_crop
                candidate_detections = self.detect_components(
                    candidate_frame, conf=conf, imgsz=imgsz, fast_mode=fast_mode
                )
                candidate_orientation, _, candidate_error = estimate_anchor_orientation(
                    self.reference_components, candidate_detections, conf
                )
                if candidate_orientation is not None:
                    board_crop = candidate_crop
                    det_frame = candidate_frame
                    raw_detections = candidate_detections
                    anchor_orientation = candidate_orientation
                    orientation_transform = candidate_transform
                    pre_rotation_degrees = candidate_rotation
                    anchor_error = None
                    break
                anchor_error = candidate_error or anchor_error

        if anchor_orientation is not None:
            tightened_crop, crop_transform, crop_refined = crop_board_to_reference_bounds(
                board_crop,
                self.reference_components,
                anchor_orientation,
                self.reference_image.shape,
            )
            if crop_refined:
                if np.mean(tightened_crop) < 60 or np.std(tightened_crop) < 30:
                    tightened_frame = adjust_contrast(tightened_crop)
                else:
                    tightened_frame = tightened_crop
                tightened_detections = self.detect_components(
                    tightened_frame, conf=conf, imgsz=imgsz, fast_mode=fast_mode
                )
                tightened_orientation, _, _ = estimate_anchor_orientation(
                    self.reference_components, tightened_detections, conf
                )
                if tightened_orientation is not None:
                    board_crop = tightened_crop
                    det_frame = tightened_frame
                    raw_detections = tightened_detections
                    anchor_orientation = tightened_orientation
                    orientation_transform = crop_transform @ orientation_transform
                    crop_method = f"{crop_method}+anchor_crop"
                    logger.info(
                        "PCB crop tightened from anchor-projected reference bounds: "
                        "crop=%dx%d detections=%d.",
                        board_crop.shape[1],
                        board_crop.shape[0],
                        len(raw_detections),
                    )

            board_crop, residual_orientation_transform = rotate_board_to_reference(
                board_crop, anchor_orientation["rotation_degrees"]
            )
            orientation_transform = (
                residual_orientation_transform @ orientation_transform
            )
            crop_method = f"{crop_method}+anchor_rotation"
            if np.mean(board_crop) < 60 or np.std(board_crop) < 30:
                det_frame = adjust_contrast(board_crop)
            else:
                det_frame = board_crop
            raw_detections = self.detect_components(
                det_frame, conf=conf, imgsz=imgsz, fast_mode=fast_mode
            )
            logger.info(
                "PCB orientation normalized by anchors: rotation=%.2f degrees "
                "score=%.4f left_cap=(%.1f, %.1f) right_cap=(%.1f, %.1f) "
                "ic=(%.1f, %.1f) detections_after_rotation=%d.",
                anchor_orientation["rotation_degrees"],
                anchor_orientation["score"],
                anchor_orientation["left_cap"]["cx"],
                anchor_orientation["left_cap"]["cy"],
                anchor_orientation["right_cap"]["cx"],
                anchor_orientation["right_cap"]["cy"],
                anchor_orientation["ic"]["cx"],
                anchor_orientation["ic"]["cy"],
                len(raw_detections),
            )
            if pre_rotation_degrees:
                logger.info(
                    "Anchor detection required a preliminary %d-degree rotation.",
                    pre_rotation_degrees,
                )
        elif anchor_required:
            anchor_error = (
                f"{anchor_error} Đã thử các hướng 0°, 90°, 180° và 270°."
            )
            logger.warning("PCB anchor orientation failed: %s", anchor_error)

        # 1. Căn chỉnh theo homography/affine từ các class độc nhất
        H = np.eye(3, dtype=np.float64) if reference_rectified else None
        align_method = "Homography (Reference)" if reference_rectified else None
        if reference_rectified:
            logger.info(
                "PCB is already rectified to reference coordinates; "
                "skipping component-based re-alignment."
            )
        elif not anchor_required or anchor_orientation is not None:
            H = estimate_component_alignment(
                self.reference_components, raw_detections, tolerance, conf,
                crop_shape=board_crop.shape, ref_shape=self.reference_image.shape
            )
        if H is not None and not reference_rectified:
            is_projective = not np.allclose(H[2], [0.0, 0.0, 1.0], atol=1e-6)
            align_method = (
                "Homography (Components)" if is_projective else "Affine (Components)"
            )
            logger.info(
                "Alignment succeeded using component %s alignment.",
                "homography" if is_projective else "affine",
            )
        else:
            logger.info("Component alignment did not produce a transform.")

        # Prefer validated projective alignment for both QA coordinates and crop rendering.
        if not reference_rectified and (not anchor_required or anchor_orientation is not None) and self.aligner is not None:
            logger.info("Trying ORB homography alignment at 0 degrees.")
            orb_H = self.aligner.compute_homography(det_frame)
            orb_align_method = "Homography (ORB)"
            if orb_H is None:
                det_frame_180 = cv2.rotate(det_frame, cv2.ROTATE_180)
                logger.info("Trying ORB homography alignment at 180 degrees.")
                H_180 = self.aligner.compute_homography(det_frame_180)
                if H_180 is not None:
                    ch, cw = det_frame.shape[:2]
                    R180 = np.array([[-1, 0, cw - 1], [0, -1, ch - 1], [0, 0, 1]], dtype=np.float64)
                    orb_H = H_180 @ R180
                    orb_align_method = "Homography (ORB 180°)"

            if orb_H is not None:
                H = orb_H
                align_method = orb_align_method
                logger.info("Projective alignment selected using %s.", align_method)
            elif H is not None:
                logger.info(
                    "ORB homography unavailable; keeping component-based alignment."
                )
            else:
                logger.info("ORB homography alignment failed at both 0 and 180 degrees.")
        elif not reference_rectified and (not anchor_required or anchor_orientation is not None):
            logger.warning("ORB homography alignment skipped: reference ORB aligner is unavailable.")

        aligned = H is not None
        alignment_error = anchor_error if anchor_required and anchor_orientation is None else None
        if not aligned:
            if alignment_error is None:
                alignment_error = (
                    "Không phát hiện linh kiện để căn chỉnh."
                    if not raw_detections
                    else "Không tìm được phép căn chỉnh bằng linh kiện hoặc đặc trưng ORB."
                )
            logger.warning(
                "PCB alignment failed: method=none detections=%d error=%s.",
                len(raw_detections),
                alignment_error,
            )
        else:
            logger.info(
                "PCB alignment complete: method=%s qa_available=%s.",
                align_method,
                aligned,
            )
            if anchor_required:
                align_method = f"Anchor rotation + {align_method}"

        # Match exact classes first; spatially confirmed class disagreements need review.
        shown_detections = self._map_detections(raw_detections, H) if aligned else raw_detections
        if aligned:
            shown_detections = self._suppress_reference_hole_detections(
                shown_detections, self.reference_mounting_holes
            )
            shown_detections = self._suppress_class_conflicts_by_reference(
                self.reference_components, shown_detections, tolerance
            )

        if aligned:
            matches, matched_detection_indices, used_references = match_components(
                self.reference_components,
                shown_detections,
                tolerance,
                require_same_class=True,
                allow_location_class_mismatch=True,
            )
            unmatched_reference_indices = [
                index
                for index in range(len(self.reference_components))
                if index not in used_references
            ]
            class_mismatch_matches = [
                match for match in matches if not match["class_match"]
            ]
            mismatch_reference_indices = {
                match["ref_index"] for match in class_mismatch_matches
            }
            roi_reference_indices = sorted(
                set(unmatched_reference_indices) | mismatch_reference_indices
            )
            roi_detections = self._detect_missing_reference_rois(
                det_frame,
                roi_reference_indices,
                H,
                imgsz,
            )
            if roi_detections:
                roi_replacements = []
                replaced_detection_indices = set()
                for mismatch in class_mismatch_matches:
                    ref_index = mismatch["ref_index"]
                    reference_component = self.reference_components[ref_index]
                    expected_class = normalize_component_class(
                        reference_component["class"]
                    )
                    candidates = [
                        detection
                        for detection in roi_detections
                        if detection.get("_roi_ref_index") == ref_index
                        and normalize_component_class(detection["class"])
                        == expected_class
                    ]
                    valid_candidates = []
                    for candidate in candidates:
                        mapped_candidate = self._map_detections(
                            [candidate], H
                        )[0]
                        candidate_matches, _, _ = match_components(
                            [reference_component],
                            [mapped_candidate],
                            tolerance,
                            require_same_class=True,
                        )
                        if candidate_matches:
                            valid_candidates.append(candidate)

                    if valid_candidates:
                        best_candidate = max(
                            valid_candidates,
                            key=lambda detection: detection["confidence"],
                        )
                        roi_replacements.append(best_candidate)
                        replaced_detection_indices.add(mismatch["det_index"])
                        logger.info(
                            "Replaced class-mismatched detection with reference-ROI "
                            "confirmation: ref_index=%d expected_class=%s "
                            "confidence=%.3f.",
                            ref_index,
                            reference_component["class"],
                            best_candidate["confidence"],
                        )

                unmatched_roi_detections = self._filter_reference_roi_detections(
                    roi_detections,
                    unmatched_reference_indices,
                    H,
                    tolerance,
                )
                roi_detections_to_merge = (
                    roi_replacements + unmatched_roi_detections
                )
                raw_detections = [
                    detection
                    for index, detection in enumerate(raw_detections)
                    if index not in replaced_detection_indices
                ]
                combined_detections = [
                    {
                        **detection,
                        "box": (
                            detection["x1"],
                            detection["y1"],
                            detection["x2"],
                            detection["y2"],
                        ),
                    }
                    for detection in raw_detections + roi_detections_to_merge
                ]
                raw_detections = [
                    {
                        key: value
                        for key, value in detection.items()
                        if key not in {"box", "_roi_ref_index"}
                    }
                    for detection in self._suppress_overlapping_detections(
                        combined_detections
                    )
                ]
                for detection_index, detection in enumerate(raw_detections, 1):
                    detection["id"] = detection_index
                shown_detections = self._map_detections(raw_detections, H)
                shown_detections = self._suppress_reference_hole_detections(
                    shown_detections, self.reference_mounting_holes
                )
                shown_detections = self._suppress_class_conflicts_by_reference(
                    self.reference_components, shown_detections, tolerance
                )
                matches, matched_detection_indices, used_references = match_components(
                    self.reference_components,
                    shown_detections,
                    tolerance,
                    require_same_class=True,
                    allow_location_class_mismatch=True,
                )
        else:
            matches, matched_detection_indices, used_references = [], set(), set()

        inventory_status, inventory_stats, inventory_comparison = (
            evaluate_component_inventory(
                self.reference_components, shown_detections
            )
        )
        if aligned:
            self._apply_reference_class_authority(
                self.reference_components, shown_detections, matches
            )
            _, position_stats = evaluate_board_status(
                self.reference_components, shown_detections, matches,
                used_references, aligned=True,
                missing_allowed=missing_allowed, extra_allowed=extra_allowed,
                conf_thresh=conf,
            )
            status = inventory_status
            stats = {**inventory_stats, "class_mismatch": position_stats["class_mismatch"]}
        else:
            status = inventory_status
            stats = inventory_stats
        match_map = {m["det_index"]: m["ref_index"] for m in matches}

        extra_detection_indices = set(
            position_stats.get("extra_detection_indices", []) if aligned else []
        )
        detailed_detections = []
        for i, det in enumerate(shown_detections):
            d_item = det.copy()
            if i in matched_detection_indices:
                ref_idx = match_map[i]
                match = next(
                    match for match in matches if match["det_index"] == i
                )
                d_item["status"] = (
                    "OK" if match["class_match"] else "REVIEW"
                )
                d_item["matched_ref_id"] = self.reference_components[ref_idx].get("id", ref_idx + 1)
                d_item["matched_ref_class"] = self.reference_components[ref_idx]["class"]
                d_item["match_distance"] = match["distance"]
                d_item["class_match"] = match["class_match"]
            else:
                d_item["status"] = (
                    "EXTRA" if i in extra_detection_indices
                    else "REVIEW" if aligned
                    else STATUS_UNALIGNED
                )
                d_item["matched_ref_id"] = None
                d_item["matched_ref_class"] = None
            detailed_detections.append(d_item)

        missing_details = []
        if aligned and position_stats["missing"] > 0:
            for idx, c in enumerate(self.reference_components):
                if idx not in used_references:
                    missing_details.append({
                        "ref_id": c.get("id", idx + 1),
                        "class": c["class"],
                        "ref_x": float(c["x"]),
                        "ref_y": float(c["y"]),
                    })
        extra_details = [
            {
                "detection_id": detailed_detections[index].get("id", index + 1),
                "class": detailed_detections[index]["class"],
                "confidence": detailed_detections[index]["confidence"],
                "x": detailed_detections[index]["cx"],
                "y": detailed_detections[index]["cy"],
                "ref_x": detailed_detections[index].get("ref_x"),
                "ref_y": detailed_detections[index].get("ref_y"),
                "reason": "Không ghép được với linh kiện tham chiếu.",
            }
            for index in (
                position_stats.get("extra_detection_indices", []) if aligned else []
            )
        ]
        matched_details = [
            {
                "reference_id": self.reference_components[match["ref_index"]].get(
                    "id", match["ref_index"] + 1
                ),
                "reference_class": self.reference_components[match["ref_index"]]["class"],
                "detection_id": detailed_detections[match["det_index"]].get(
                    "id", match["det_index"] + 1
                ),
                "detection_class": detailed_detections[match["det_index"]]["class"],
                "yolo_class": detailed_detections[match["det_index"]].get(
                    "yolo_class",
                    detailed_detections[match["det_index"]]["class"],
                ),
                "distance": match["distance"],
                "size_penalty": match["size_penalty"],
                "class_match": match["class_match"],
                "yolo_class_match": match.get(
                    "yolo_class_match", match["class_match"]
                ),
            }
            for match in matches
        ]
        class_mismatch_details = [
            {
                "reference_id": self.reference_components[match["ref_index"]].get(
                    "id", match["ref_index"] + 1
                ),
                "reference_class": self.reference_components[match["ref_index"]]["class"],
                "detection_id": detailed_detections[match["det_index"]].get(
                    "id", match["det_index"] + 1
                ),
                "detection_class": detailed_detections[match["det_index"]]["class"],
                "confidence": detailed_detections[match["det_index"]]["confidence"],
                "distance": match["distance"],
            }
            for match in matches
            if not match["class_match"]
        ]
        logger.info(
            "Inspection matching result: status=%s matches=%d missing=%d extra=%d "
            "missing_components=%s extra_components=%s.",
            status,
            len(matches),
            inventory_stats["missing"],
            inventory_stats["extra"],
            [component["class"] for component in missing_details],
            [component["class"] for component in extra_details],
        )

        display_crop = board_crop
        if aligned and not reference_rectified:
            display_crop = warp_board_to_reference(
                board_crop, H, self.reference_image.shape
            )
            if display_crop is not None:
                crop_method = f"{crop_method}+reference_warp"
            else:
                display_crop = board_crop
        display_rectified = aligned and (
            reference_rectified or display_crop is not board_crop
        )

        annotated = display_crop.copy()
        for i, det in enumerate(shown_detections):
            if detailed_detections[i]["status"] == "EXTRA":
                continue
            if display_rectified:
                x1, y1, x2, y2 = (
                    int(round(value))
                    for value in det["ref_bbox"]
                )
            else:
                x1, y1, x2, y2 = (
                    int(det["x1"]),
                    int(det["y1"]),
                    int(det["x2"]),
                    int(det["y2"]),
                )
            detection_status = detailed_detections[i]["status"]
            color, _ = {
                "OK": ((0, 230, 0), "OK"),
                "EXTRA": ((0, 0, 235), "EXTRA"),
                "REVIEW": ((0, 190, 255), "REVIEW"),
                STATUS_UNALIGNED: ((255, 190, 0), "UNALIGNED"),
            }[detection_status]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = self._format_detection_annotation_label(detailed_detections[i])
            cv2.putText(annotated, label,
                        (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

        # Vẽ đánh dấu linh kiện thiếu
        if aligned and H is not None and missing_details:
            try:
                pts = np.float32([[m["ref_x"], m["ref_y"]] for m in missing_details]).reshape(-1, 1, 2)
                if display_rectified:
                    projected = pts.reshape(-1, 2)
                else:
                    H_inv = np.linalg.inv(H)
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
            (0, 190, 255) if status == STATUS_REVIEW else
            (0, 0, 235)
        )
        status_label = {
            STATUS_OK: "MẠCH ĐẠT CHUẨN (OK)",
            STATUS_FAULTY: "MẠCH BỊ LỖI (SỐ LƯỢNG LINH KIỆN KHÔNG KHỚP)",
            STATUS_REVIEW: "CẦN KIỂM TRA LẠI (NHẬN DIỆN CHƯA CHẮC CHẮN)",
            STATUS_UNALIGNED: "ĐÃ ĐỐI CHIẾU SỐ LƯỢNG",
        }[status]

        header_h = 75
        cv2.rectangle(annotated, (0, 0), (min(annotated.shape[1], 560), header_h), (20, 20, 25), -1)
        cv2.putText(annotated, status_label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)

        if aligned:
            summary_txt = (
                f"Đủ loại: {stats['matched']}/{len(self.reference_components)} | "
                f"Thiếu: {stats['missing']} | Dư: {stats['extra']}"
            )
        else:
            summary_txt = (
                f"Đủ loại: {stats['matched']}/{len(self.reference_components)} "
                f"| Thiếu: {stats['missing']} | Dư: {stats['extra']}"
            )
        cv2.putText(annotated, summary_txt, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        proc_time = round((time.time() - start_time) * 1000, 2)

        return {
            "status": status,
            "status_vi": {
                STATUS_OK: "MẠCH ĐẠT CHUẨN",
                STATUS_FAULTY: "MẠCH BỊ LỖI (SỐ LƯỢNG LINH KIỆN KHÔNG KHỚP)",
                STATUS_REVIEW: "CẦN KIỂM TRA LẠI (NHẬN DIỆN CHƯA CHẮC CHẮN)",
                STATUS_UNALIGNED: "ĐÃ ĐỐI CHIẾU SỐ LƯỢNG",
            }[status],
            "aligned": aligned,
            "qa_available": aligned,
            "comparison_scope": "component_count",
            "comparison_note": (
                "Kết luận dựa trên số lượng linh kiện theo từng loại. "
                + (
                    "Ảnh chưa căn chỉnh được nên chưa thể xác nhận vị trí."
                    if not aligned
                    else "Căn chỉnh vị trí chỉ dùng để hiển thị chi tiết đối chiếu."
                )
            ),
            "alignment_method": align_method,
            "alignment_error": alignment_error,
            "alignment_transform": H.tolist() if aligned else None,
            "orientation_transform": orientation_transform.tolist(),
            "crop_method": crop_method,
            "stats": {
                **stats,
                "reference_count": len(self.reference_components),
                "detected_count": len(shown_detections),
            },
            "inventory_comparison": inventory_comparison,
            "class_counts": dict(Counter(d["class"] for d in shown_detections)),
            "detections": detailed_detections,
            "matched_components": matched_details,
            "class_mismatch_details": class_mismatch_details,
            "missing_components_details": missing_details,
            "extra_components_details": extra_details,
            "annotated_image": annotated,
            "board_crop_image": display_crop,
            "latency_ms": proc_time,
        }

    def create_reference_from_image(self, image_np, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ):
        """Create a reference map using the same postprocessed detector as inspection."""
        h, w = image_np.shape[:2]
        detections = self.detect_components(image_np, conf=conf, imgsz=imgsz)
        components = []
        annotated = image_np.copy()

        for component_id, detection in enumerate(detections, 1):
            x1, y1 = detection["x1"], detection["y1"]
            x2, y2 = detection["x2"], detection["y2"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            components.append({
                "id": component_id,
                "class_id": detection["class_id"],
                "class": class_name,
                "confidence_reference": round(confidence, 4),
                "x": round(detection["cx"], 2),
                "y": round(detection["cy"], 2),
            })

            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(
                annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )
            cv2.putText(
                annotated,
                label,
                (int(x1), max(int(y1) - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

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

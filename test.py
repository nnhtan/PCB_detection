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
SOURCE = "http://192.168.0.190:8080/video"

# ---- Chu kỳ quét ----
SCAN_INTERVAL_SECONDS = 5       # 5 giây quét 1 lần
RECENT_FRAMES = 5               # giữ 5 frame gần nhất, chọn frame nét nhất để quét
RECENT_FRAME_GAP = 0.12         # giây giữa 2 frame được lưu
SMOOTHING_WINDOW = 1            # phản hồi ngay theo kết quả quét mới nhất

# ---- YOLO (PHẢI GIỐNG machpcbgoc.py) ----
CONF = 0.25
IMGSZ = 960
MAX_DET = 1000
USE_CUDA = torch.cuda.is_available()
DEVICE = 0 if USE_CUDA else "cpu"

# ---- So sánh vị trí ----
MATCH_TOLERANCE_RATIO = 0.02    # sai số = 2% cạnh dài nhất của ảnh gốc
MISSING_ALLOWED = 0             # cho phép thiếu tối đa bao nhiêu linh kiện
EXTRA_ALLOWED = 2               # cho phép thừa tối đa bao nhiêu linh kiện
EXTRA_MIN_CONF = 0.50           # box thừa phải có conf >= giá trị này mới tính là thừa

# ---- ORB ----
USE_ORB_ALIGNMENT = True
MIN_DETECTION_RATIO = 0.30
ORB_FEATURES = 2000
RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 10
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.40
RANSAC_THRESHOLD = 5.0
MIN_BOARD_AREA_RATIO = 0.03     # PCB chiếm tối thiểu 3% khung hình
MAX_BOARD_AREA_RATIO = 2.00

# ---- Hiển thị ----
WINDOW_LIVE = "PCB Camera"
WINDOW_RESULT = "PCB Scan Result"
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 480
SHOW_RESULT_WINDOW = True
RESULT_IMAGE = "camera_result.jpg"

# ---- Trạng thái ----
STATUS_NO_PCB = "NO_PCB"
STATUS_FAULTY = "FAULTY"
STATUS_OK = "OK"

STATUS_TEXT_TERMINAL = {
    STATUS_NO_PCB: "KHÔNG CÓ MẠCH PCB",
    STATUS_FAULTY: "MẠCH BỊ LỖI",
    STATUS_OK: "MẠCH ĐẠT CHUẨN",
}
STATUS_TEXT_IMAGE = {
    STATUS_NO_PCB: "NO PCB",
    STATUS_FAULTY: "PCB FAULTY",
    STATUS_OK: "PCB OK",
}
STATUS_COLOR = {                       # BGR
    STATUS_NO_PCB: (0, 200, 255),
    STATUS_FAULTY: (0, 0, 255),
    STATUS_OK: (0, 255, 0),
}
ANSI = {
    STATUS_NO_PCB: "\033[93m",
    STATUS_FAULTY: "\033[91m",
    STATUS_OK: "\033[92m",
}
ANSI_RESET = "\033[0m"


# ============================================================
# 2. ĐỌC CAMERA Ở LUỒNG RIÊNG
# ============================================================

class CameraStream:
    """Luôn đọc frame mới nhất trong nền, giữ vài frame gần nhất."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.recent = deque(maxlen=RECENT_FRAMES)
        self._last_saved = 0.0
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
            now = time.monotonic()
            with self.lock:
                self.frame = frame
                self.frame_id += 1
                if now - self._last_saved >= RECENT_FRAME_GAP:
                    self.recent.append(frame)
                    self._last_saved = now

    def read(self):
        with self.lock:
            return self.frame, self.frame_id

    def get_recent(self):
        with self.lock:
            return list(self.recent)

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


# ============================================================
# 5. YOLO
# ============================================================

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

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes
    detections = []

    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[idx].detach().cpu().numpy().tolist()
        cls_id = int(boxes.cls[idx].item())
        detections.append({
            "id": idx + 1,
            "class_id": cls_id,
            "class": str(model.names[cls_id]),
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

def match_components(reference_components, detections, tolerance):
    candidates = []

    for det_index, det in enumerate(detections):
        for ref_index, ref in enumerate(reference_components):
            if det["class"] != ref["class"]:
                continue

            distance = float(np.hypot(det["ref_x"] - float(ref["x"]),
                                      det["ref_y"] - float(ref["y"])))
            if distance <= tolerance:
                candidates.append((distance, det_index, ref_index))

    candidates.sort(key=lambda c: c[0])

    used_det, used_ref, matches = set(), set(), []
    for distance, det_index, ref_index in candidates:
        if det_index in used_det or ref_index in used_ref:
            continue
        used_det.add(det_index)
        used_ref.add(ref_index)
        matches.append({"det_index": det_index, "ref_index": ref_index,
                        "distance": distance})

    return matches, used_det, used_ref


# ============================================================
# 7. ĐÁNH GIÁ 3 TRẠNG THÁI
# ============================================================

def evaluate_board_status(reference_components, detections, matches,
                          used_references, aligned):
    stats = {"matched": len(matches), "missing": 0, "extra": 0}

    if not aligned:
        reference_classes = {component["class"] for component in reference_components}
        recognized = sum(
            1 for detection in detections
            if detection["class"] in reference_classes
            and detection["confidence"] >= CONF
        )
        stats["detected"] = recognized
        minimum_detected = max(
            1, int(np.ceil(MIN_DETECTION_RATIO * len(reference_components)))
        )
        if recognized >= minimum_detected:
            return STATUS_FAULTY, stats
        return STATUS_NO_PCB, stats

    matched_det = {m["det_index"] for m in matches}
    missing = len(reference_components) - len(used_references)
    extra = sum(
        1 for i, d in enumerate(detections)
        if i not in matched_det and d["confidence"] >= EXTRA_MIN_CONF
    )

    stats["missing"] = missing
    stats["extra"] = extra

    if missing > MISSING_ALLOWED or extra > EXTRA_ALLOWED:
        return STATUS_FAULTY, stats

    return STATUS_OK, stats


# ============================================================
# 8. VẼ KẾT QUẢ
# ============================================================

def draw_results(frame, detections, reference_components, matches,
                 used_references, aligned, H, status, stats):
    output = frame.copy()
    match_map = {m["det_index"]: m for m in matches}

    for det_index, det in enumerate(detections):
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])

        if det_index in match_map:
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

    if aligned and H is not None:
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
        info = (f'Matched: {stats["matched"]}  Missing: {stats["missing"]}  '
                f'Extra: {stats["extra"]}')
    else:
        info = f'Alignment failed; PCB-class detections: {stats["detected"]}'
    cv2.putText(output, info, (10, 68), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)

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
# 9. MỘT LẦN QUÉT
# ============================================================

def frame_sharpness(frame):
    h, w = frame.shape[:2]
    scale = 640.0 / max(w, 1)
    small = cv2.resize(frame, (640, max(1, int(h * scale)))) if scale < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_sharpest(frames):
    return max(frames, key=frame_sharpness)


def scan_board(frame, model, reference_components, ref_shape, orb, ref_kp, ref_desc):
    detections = detect_components(model, frame)
    ref_h, ref_w = ref_shape[:2]
    tolerance = MATCH_TOLERANCE_RATIO * max(ref_w, ref_h)

    if USE_ORB_ALIGNMENT:
        H = compute_homography(ref_kp, ref_desc, ref_shape, frame, orb)
    else:
        H = np.eye(3, dtype=np.float64)

    aligned = H is not None

    transformed = []
    if aligned:
        for det in detections:
            rx, ry = transform_point_to_reference(det["cx"], det["cy"], H)
            d = det.copy()
            d["ref_x"], d["ref_y"] = rx, ry
            transformed.append(d)

        matches, _, used_references = match_components(
            reference_components, transformed, tolerance
        )
    else:
        matches, used_references = [], set()

    shown_detections = transformed if aligned else detections

    status, stats = evaluate_board_status(
        reference_components, shown_detections, matches, used_references, aligned
    )

    annotated = draw_results(
        frame, shown_detections, reference_components, matches,
        used_references, aligned, H, status, stats,
    )

    return {"status": status, "annotated": annotated, "stats": stats}


# ============================================================
# 10. QUÉT Ở LUỒNG NỀN
# ============================================================

class Scanner:
    def __init__(self, stream, model, reference_components, ref_shape,
                 orb, ref_kp, ref_desc):
        self.stream = stream
        self.args = (model, reference_components, ref_shape, orb, ref_kp, ref_desc)
        self.busy = False
        self.result = None

    def start(self, fallback_frame):
        if self.busy:
            return
        self.busy = True
        frames = self.stream.get_recent() or [fallback_frame]
        threading.Thread(target=self._run, args=(frames,), daemon=True).start()

    def _run(self, frames):
        try:
            frame = pick_sharpest(frames)
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
# 11. MAIN
# ============================================================

def main():
    os.system("")  # bật màu ANSI trên Windows terminal

    reference_components = load_reference()["components"]
    reference_image = load_reference_image()
    ref_shape = reference_image.shape
    model = YOLO(find_model())

    if USE_ORB_ALIGNMENT:
        orb, ref_kp, ref_desc = build_reference_orb(reference_image)
    else:
        orb = ref_kp = ref_desc = None

    # Chạy thử YOLO 1 lần để lần quét đầu không bị khựng
    model.predict(
        source=np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8),
        imgsz=IMGSZ, device=DEVICE, half=USE_CUDA, verbose=False,
    )

    stream = CameraStream(SOURCE)

    if not stream.is_opened():
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

    scanner = Scanner(stream, model, reference_components, ref_shape,
                      orb, ref_kp, ref_desc)

    history = deque(maxlen=max(1, SMOOTHING_WINDOW))
    next_scan_time = 0.0
    last_status = None
    last_annotated = None
    last_shown_id = -1

    try:
        while True:
            frame, frame_id = stream.read()

            if frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue

            # ---------- GIAO VIỆC QUÉT THEO CHU KỲ ----------
            if time.monotonic() >= next_scan_time and not scanner.busy:
                scanner.start(frame)
                next_scan_time = time.monotonic() + SCAN_INTERVAL_SECONDS

            # ---------- NHẬN KẾT QUẢ ----------
            result = scanner.pop_result()
            if result is not None:
                history.append(result["status"])
                last_status = stable_status(history)
                last_annotated = draw_status_overlay(result["annotated"], last_status)

                print(f"{ANSI[last_status]}{STATUS_TEXT_TERMINAL[last_status]}{ANSI_RESET}",
                      flush=True)

                if SHOW_RESULT_WINDOW:
                    cv2.imshow(WINDOW_RESULT, resize_for_display(last_annotated))

            # ---------- VIDEO TRỰC TIẾP ----------
            if frame_id != last_shown_id:
                last_shown_id = frame_id

                live = resize_for_display(frame)
                if live is frame:
                    live = frame.copy()

                remaining = max(0.0, next_scan_time - time.monotonic())

                cv2.rectangle(live, (0, 0), (live.shape[1], 36), (25, 25, 25), -1)

                if last_status is not None:
                    cv2.putText(live, f"Last: {STATUS_TEXT_IMAGE[last_status]}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                STATUS_COLOR[last_status], 2, cv2.LINE_AA)

                cv2.putText(live, f"Next scan: {remaining:.0f}s",
                            (max(live.shape[1] - 200, 10), 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(WINDOW_LIVE, live)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(5) & 0xFF

            if key == ord("q"):
                break
            if key == ord("n"):
                next_scan_time = 0.0
            if key == ord("s") and last_annotated is not None:
                cv2.imwrite(RESULT_IMAGE, last_annotated)
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
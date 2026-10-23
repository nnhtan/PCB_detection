import cv2
import numpy as np

URL = 'http://192.168.100.25:4747/video'
FRAME_WIDTH = 640           # chỉ cố định chiều rộng, chiều cao tính theo tỷ lệ gốc

# ---- Lọc contour (mục 1) ----
MIN_AREA_RATIO = 0.02       # nhỏ hơn 2% khung hình -> nhiễu
MAX_AREA_RATIO = 0.90       # lớn hơn 90% khung hình -> mặt bàn / nền
MIN_SOLIDITY = 0.90         # diện tích contour / diện tích bao lồi
MIN_RECTANGULARITY = 0.85   # diện tích bao lồi / diện tích hình chữ nhật nhỏ nhất
MIN_ASPECT = 0.25           # cạnh ngắn / cạnh dài của hình chữ nhật bao

# ---- Xác định 4 góc (mục 2) ----
APPROX_EPS = (0.01, 0.015, 0.02, 0.03, 0.04, 0.06)
ANGLE_RANGE = (70, 110)     # góc trong hợp lệ của tứ giác (độ)
MAX_SIDE_RATIO = 1.6        # tỉ lệ tối đa giữa hai cạnh đối diện
MIN_SIDE_PX = 30            # cạnh ngắn nhất cho phép (px)
MINRECT_SOLIDITY = 0.93     # chỉ dùng minAreaRect khi contour đủ tin cậy
MINRECT_RECTANGULARITY = 0.92

# ---- Khoảng đệm theo cạnh (mục 3) ----
PAD_RATIO = 0.02            # đệm = 2% cạnh ngắn của PCB
PAD_MIN = 3
PAD_MAX = 15

# ---- Kiểm tra biên khung hình (mục 5) ----
BORDER_MARGIN = 5

# ---- Ổn định giữa các frame (mục 6) ----
CONFIRM_FRAMES = 5          # số frame liên tiếp để xác nhận PCB
MAX_MISS = 5                # mất PCB quá số frame này thì reset
SMOOTH_ALPHA = 0.4          # hệ số làm mượt tọa độ (EMA)
JUMP_PX = 40                # góc nhảy quá số px này bị coi là bất thường

# ---- Kiểm tra ảnh đầu ra (mục 7) ----
MIN_OUT_SIZE = 80           # kích thước tối thiểu ảnh cắt (px)
MAX_OUT_ASPECT = 5.0
MIN_FILL = 0.75             # tỉ lệ ảnh cắt thuộc PCB (còn lại là nền/đệm)
AUTO_LANDSCAPE = True


# ================================================================ Hình học cơ bản
def order_points(pts):
    """TL, TR, BR, BL theo chiều kim đồng hồ — đúng cả khi PCB xoay nghiêng."""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -np.argmin(pts.sum(axis=1)), axis=0)


def side_lengths(q):
    return [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]


def quad_angles(q):
    out = []
    for i in range(4):
        a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        out.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    return out


def valid_quad(q):
    """Chỉ chấp nhận tứ giác lồi, góc gần vuông, cạnh đối xấp xỉ nhau."""
    s = side_lengths(q)
    if min(s) < MIN_SIDE_PX:
        return False
    if max(s[0], s[2]) / min(s[0], s[2]) > MAX_SIDE_RATIO:
        return False
    if max(s[1], s[3]) / min(s[1], s[3]) > MAX_SIDE_RATIO:
        return False
    return all(ANGLE_RANGE[0] <= a <= ANGLE_RANGE[1] for a in quad_angles(q))


def line_intersection(p1, d1, p2, d2):
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def offset_quad(q, pad):
    """Đẩy từng CẠNH ra ngoài pad px theo pháp tuyến rồi giao hai đường kề nhau."""
    lines = []
    for i in range(4):
        p, d = q[i], q[(i + 1) % 4] - q[i]
        d = d / (np.linalg.norm(d) + 1e-9)
        n = np.array([d[1], -d[0]], dtype="float32")   # pháp tuyến hướng ra ngoài
        lines.append((p + n * pad, d))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        pt = line_intersection(p1, d1, p2, d2)
        out.append(pt if pt is not None else p2)
    return np.array(out, dtype="float32")


def compute_pad(q):
    return float(np.clip(PAD_RATIO * min(side_lengths(q)), PAD_MIN, PAD_MAX))


# ================================================================ Tiền xử lý + tách biên (mục 4)
def resize_keep_ratio(img, width):
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(h * width / w)))


def auto_canny(img, sigma):
    v = np.median(img)
    lo, hi = int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v))
    return cv2.Canny(img, lo, hi), lo, hi


def odd(v):
    v = max(1, v)
    return v if v % 2 else v + 1


def process(frame, params):
    resized = resize_keep_ratio(frame, FRAME_WIDTH)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=max(1, params['clahe']) / 10.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges, lo, hi = auto_canny(blur, params['sigma'] / 100.0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (odd(params['kernel']),) * 2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=max(1, params['iter']))
    return resized, gray, edges, lo, hi


# ================================================================ Chọn contour + 4 góc (mục 1, 2)
def hull_to_quad(hull, solidity, rectangularity):
    peri = cv2.arcLength(hull, True)
    for eps in APPROX_EPS:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            q = order_points(approx)
            if valid_quad(q):
                return q
    # minAreaRect chỉ khi contour gần như là hình chữ nhật thật
    if solidity >= MINRECT_SOLIDITY and rectangularity >= MINRECT_RECTANGULARITY:
        q = order_points(cv2.boxPoints(cv2.minAreaRect(hull)))
        if valid_quad(q):
            return q
    return None


def evaluate_contour(c, frame_area):
    area = cv2.contourArea(c)
    if not (MIN_AREA_RATIO * frame_area <= area <= MAX_AREA_RATIO * frame_area):
        return None
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return None
    solidity = area / hull_area
    if solidity < MIN_SOLIDITY:
        return None
    (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
    if rw * rh <= 0:
        return None
    rectangularity = hull_area / (rw * rh)
    if rectangularity < MIN_RECTANGULARITY or min(rw, rh) / max(rw, rh) < MIN_ASPECT:
        return None
    quad = hull_to_quad(hull, solidity, rectangularity)
    if quad is None:
        return None
    return area * solidity * rectangularity, quad


def find_pcb(edges, shape):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = shape[0] * shape[1]
    best = None
    for c in contours:
        r = evaluate_contour(c, frame_area)
        if r and (best is None or r[0] > best[0]):
            best = (r[0], c, r[1])
    return (best[1], best[2]) if best else None


# ================================================================ Kiểm tra biên (mục 5)
def check_inside(contour, quad, shape, m):
    h, w = shape[:2]
    if np.any(quad[:, 0] < m) or np.any(quad[:, 1] < m) or \
       np.any(quad[:, 0] > w - 1 - m) or np.any(quad[:, 1] > h - 1 - m):
        return "Goc PCB sat/ngoai mep khung"
    x, y, bw, bh = cv2.boundingRect(contour)
    if x <= m or y <= m or x + bw >= w - m or y + bh >= h - m:
        return "Bounding box cham mep khung"
    p = contour.reshape(-1, 2)
    if np.any((p[:, 0] <= m) | (p[:, 1] <= m) | (p[:, 0] >= w - 1 - m) | (p[:, 1] >= h - 1 - m)):
        return "Contour cham mep anh"
    return None


def corners_inside(q, shape):
    h, w = shape[:2]
    return bool(np.all((q[:, 0] >= 0) & (q[:, 0] <= w - 1) & (q[:, 1] >= 0) & (q[:, 1] <= h - 1)))


# ================================================================ Ổn định giữa các frame (mục 6)
class CornerTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.smoothed = None
        self.pending = None
        self.confirm = 0
        self.miss = 0
        self.outlier = 0

    @staticmethod
    def align(ref, q):
        """Xoay vòng thứ tự góc để khớp ref (tránh nhảy góc khi PCB gần 45°)."""
        best = min(range(4), key=lambda k: np.abs(np.roll(q, -k, axis=0) - ref).sum())
        return np.roll(q, -best, axis=0)

    @staticmethod
    def max_dist(a, b):
        return float(np.linalg.norm(a - b, axis=1).max())

    def update(self, q):
        if q is None:
            self.miss += 1
            if self.miss > MAX_MISS:
                self.reset()
            return self.smoothed
        self.miss = 0
        if self.smoothed is None:                       # đang xác nhận
            if self.pending is not None:
                q = self.align(self.pending, q)
            if self.pending is not None and self.max_dist(self.pending, q) < JUMP_PX:
                self.confirm += 1
            else:
                self.confirm = 1
            self.pending = q
            if self.confirm >= CONFIRM_FRAMES:
                self.smoothed = q.copy()
            return self.smoothed
        q = self.align(self.smoothed, q)                # đã xác nhận
        if self.max_dist(self.smoothed, q) > JUMP_PX:
            self.outlier += 1
            if self.outlier >= CONFIRM_FRAMES:          # thay đổi kéo dài -> PCB mới
                self.smoothed, self.outlier = q.copy(), 0
        else:
            self.outlier = 0
            self.smoothed = (1 - SMOOTH_ALPHA) * self.smoothed + SMOOTH_ALPHA * q
        return self.smoothed


# ================================================================ Cắt + kiểm tra ảnh đầu ra (mục 7)
def warp_pcb(img, corners, contour):
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if min(w, h) < MIN_OUT_SIZE:
        return None, "Anh cat qua nho"
    if max(w, h) / min(w, h) > MAX_OUT_ASPECT:
        return None, "Ti le anh cat bat thuong"
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    if contour is not None:                              # kiểm tra có lấy nhầm nền không
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        fill = cv2.warpPerspective(mask, M, (w, h)).mean() / 255.0
        if fill < MIN_FILL:
            return None, f"Anh cat chua nhieu nen ({fill:.0%} la PCB)"
    out = cv2.warpPerspective(img, M, (w, h))
    if AUTO_LANDSCAPE and h > w:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    return out, None


# ================================================================ Vòng lặp chính
cv2.namedWindow('Tuning')
for name, init, mx in (('Canny sigma x100', 33, 100), ('CLAHE clip x10', 20, 80),
                       ('Close kernel', 5, 15), ('Close iter', 2, 5)):
    cv2.createTrackbar(name, 'Tuning', init, mx, lambda v: None)

cap = cv2.VideoCapture(URL)
tracker = CornerTracker()
crop_open = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được hình ảnh từ camera")
        break

    params = {
        'sigma': cv2.getTrackbarPos('Canny sigma x100', 'Tuning'),
        'clahe': cv2.getTrackbarPos('CLAHE clip x10', 'Tuning'),
        'kernel': cv2.getTrackbarPos('Close kernel', 'Tuning'),
        'iter': cv2.getTrackbarPos('Close iter', 'Tuning'),
    }
    resized, gray, edges, lo, hi = process(frame, params)
    result = resized.copy()

    status, ok = "", False
    contour = raw_quad = None
    blocked = False

    cand = find_pcb(edges, resized.shape)
    if cand is None:
        status = "Khong tim thay PCB hop le"
    else:
        contour, raw_quad = cand
        reason = check_inside(contour, raw_quad, resized.shape, BORDER_MARGIN)
        if reason:                                       # PCB chưa nằm trọn -> không quét
            status, blocked = reason + " - khong quet", True
            tracker.reset()
            raw_quad = None
        else:
            cv2.polylines(result, [raw_quad.astype(np.int32)], True, (0, 255, 255), 1)

    stable = None if blocked else tracker.update(raw_quad)

    pcb_crop = None
    if stable is not None:
        corners = offset_quad(stable, compute_pad(stable))
        if not corners_inside(corners, resized.shape):
            status = "Phan dem vuot ra ngoai khung"
        else:
            pcb_crop, err = warp_pcb(resized, corners, contour)
            if err:
                status = err
            else:
                ok, status = True, "OK"
                cv2.polylines(result, [corners.astype(np.int32)], True, (0, 255, 0), 2)
                for nm, (x, y) in zip(("TL", "TR", "BR", "BL"), corners):
                    cv2.circle(result, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.putText(result, f"{nm}({int(x)},{int(y)})", (int(x) + 6, int(y) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    elif not status:
        status = f"Dang xac nhan PCB {tracker.confirm}/{CONFIRM_FRAMES}"

    if blocked:
        cv2.rectangle(result, (0, 0), (resized.shape[1] - 1, resized.shape[0] - 1), (0, 0, 255), 2)
    cv2.putText(result, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 200, 0) if ok else (0, 0, 255), 2)
    cv2.putText(result, f"Canny {lo}/{hi}", (10, resized.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    '''cv2.imshow('Original', resized)
    cv2.imshow('Gray + CLAHE', gray)
    cv2.imshow('Edges', edges)'''
    cv2.imshow('Result', result)

    if pcb_crop is not None:                             # chỉ hiển thị, không lưu
        cv2.imshow('PCB Crop', pcb_crop)
        crop_open = True
    elif crop_open:                                      # mất PCB -> đóng cửa sổ ảnh cắt
        cv2.destroyWindow('PCB Crop')
        crop_open = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
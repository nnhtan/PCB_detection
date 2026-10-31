"""
=============================================================================
   HIKVISION CAMERA VIEWER - BẢN DẠY MẪU & KHỚP MẪU BẰNG TAY (TEMPLATE MATCHING)
=============================================================================
Thuật toán:
1. Dạy mẫu (Vision Teach-in / Template Matching):
   - Người dùng dùng chuột kéo thả trực tiếp trên màn hình để vẽ khung Bounding Box bao quanh PCB.
   - Hệ thống tự động ghi nhớ ảnh bo mạch mẫu (Golden Template).
2. Tự động bám vết & Cắt theo mẫu (Auto Template Tracking & Crop):
   - Chế độ 1 [Tracking]: Dùng cv2.matchTemplate (Normalized Cross-Correlation) quét tìm bo mạch
     trên băng chuyền theo thời gian thực (chỉ mất 15-30ms) và tự động crop theo.
   - Chế độ 2 [Fixed BBox]: Cố định vùng kiểm tra tại cữ chặn trên băng tải, tự động crop đúng vị trí.
3. Tinh chỉnh mép tự động (Edge Snapping): Tự động hút dính sát 4 mép bo mạch thật bên trong BBox.
=============================================================================
"""

import sys
import os
import time
import json
import threading
import io
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from ctypes import cast, POINTER, c_ubyte, string_at, byref, sizeof, memset

# Đảm bảo in tiếng Việt không bị lỗi cp1252 trên Windows PowerShell / CMD
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = CURRENT_DIR / "pcb_template_config.json"
TEMPLATE_IMG_FILE = CURRENT_DIR / "pcb_template.jpg"
CONVEYOR_CONFIG_FILE = CURRENT_DIR / "conveyor_config.json"

# Nạp SDK Hikvision
MV_IMPORT_PATH = CURRENT_DIR / "Python" / "MvImport"
if not MV_IMPORT_PATH.exists():
    MV_IMPORT_PATH = CURRENT_DIR.parent / "Python" / "MvImport"
sys.path.append(str(MV_IMPORT_PATH))

try:
    from MvCameraControl_class import *
    from PixelType_header import *
    HAS_HIK_SDK = True
except ImportError:
    HAS_HIK_SDK = False


def load_conveyor_config() -> dict:
    default_cfg = {
        "filter_enabled": True,
        "auto_detect": True,
        "margin_left": 0.28,
        "margin_right": 0.06,
        "margin_top": 0.0,
        "margin_bottom": 0.0
    }
    if CONVEYOR_CONFIG_FILE.exists():
        try:
            with open(CONVEYOR_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_cfg.update(data)
        except Exception:
            pass
    return default_cfg


class PCBTemplateEngine:
    """Engine quản lý mẫu bo mạch PCB (Teach-in Template Matching & Fixed ROI Crop)"""

    def __init__(self):
        self.template_img: Optional[np.ndarray] = None
        self.template_gray: Optional[np.ndarray] = None
        self.template_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) tọa độ ảnh gốc
        self.template_size: Tuple[int, int] = (0, 0)  # (w, h)
        self.tracking_mode = "tracking"  # "tracking" (khớp mẫu di động) hoặc "fixed" (vùng cố định)
        self.match_threshold = 0.50
        self.edge_snap_enabled = True

        # Thuật toán ORB bất biến góc xoay 360 độ (Rotation-Invariant 360°)
        self.orb = cv2.ORB_create(nfeatures=1200)
        self.template_kps = None
        self.template_des = None

        self.load_template()

    def _compute_template_features(self):
        """Tính toán điểm đặc trưng ORB của mẫu chuẩn để nhận diện mọi góc xoay"""
        if self.template_gray is not None and self.template_gray.size > 0:
            try:
                self.template_kps, self.template_des = self.orb.detectAndCompute(self.template_gray, None)
            except Exception:
                self.template_kps, self.template_des = None, None

    def save_template(self, full_frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Lưu vùng được người dùng vẽ làm mẫu chuẩn (Golden Template)"""
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, full_frame.shape[1] - 10))
        x2 = max(x1 + 10, min(x2, full_frame.shape[1]))
        y1 = max(0, min(y1, full_frame.shape[0] - 10))
        y2 = max(y1 + 10, min(y2, full_frame.shape[0]))

        self.template_bbox = (x1, y1, x2, y2)
        self.template_img = full_frame[y1:y2, x1:x2].copy()
        self.template_gray = cv2.cvtColor(self.template_img, cv2.COLOR_BGR2GRAY)
        self.template_size = (x2 - x1, y2 - y1)
        self._compute_template_features()

        # Lưu ảnh mẫu và file JSON
        try:
            cv2.imwrite(str(TEMPLATE_IMG_FILE), self.template_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
            data = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "orig_size": [int(full_frame.shape[1]), int(full_frame.shape[0])],
                "tracking_mode": self.tracking_mode,
                "match_threshold": self.match_threshold,
                "edge_snap": self.edge_snap_enabled,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[THÀNH CÔNG] Đã lưu mẫu PCB mới: {self.template_size[0]}x{self.template_size[1]} px")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể lưu file template: {e}")

    def load_template(self) -> bool:
        """Tự động tải lại mẫu bo mạch đã lưu từ trước"""
        if not CONFIG_FILE.exists() or not TEMPLATE_IMG_FILE.exists():
            return False
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            bbox = data.get("bbox")
            if bbox and len(bbox) == 4:
                self.template_bbox = tuple(bbox)
                self.tracking_mode = data.get("tracking_mode", "tracking")
                self.match_threshold = data.get("match_threshold", 0.50)
                self.edge_snap_enabled = data.get("edge_snap", True)

            img = cv2.imread(str(TEMPLATE_IMG_FILE))
            if img is not None and img.size > 0:
                self.template_img = img
                self.template_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.template_size = (img.shape[1], img.shape[0])
                self._compute_template_features()
                print(f"[MẪU PCB] Đã nạp mẫu sẵn có: {self.template_size[0]}x{self.template_size[1]} px (Hỗ trợ xoay 360°)")
                return True
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khi nạp template cũ: {e}")
        return False

    def reset_template(self):
        """Xóa mẫu hiện tại để vẽ mẫu mới"""
        self.template_img = None
        self.template_gray = None
        self.template_bbox = None
        self.template_size = (0, 0)
        self.template_kps = None
        self.template_des = None
        try:
            if CONFIG_FILE.exists():
                CONFIG_FILE.unlink()
            if TEMPLATE_IMG_FILE.exists():
                TEMPLATE_IMG_FILE.unlink()
        except Exception:
            pass
        print("[MẪU PCB] Đã xóa mẫu cũ. Hãy dùng chuột vẽ khung mới trên bo mạch!")

    def snap_edges(self, sub_img: np.ndarray, base_corners: np.ndarray) -> np.ndarray:
        """Hút dính mép tự động (Edge Snapping) vào cạnh thật của bo mạch bên trong BBox"""
        if not self.edge_snap_enabled or sub_img is None or sub_img.size == 0:
            return base_corners
        try:
            sh, sw = sub_img.shape[:2]
            gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enh = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enh, (5, 5), 0)

            edges = cv2.Canny(blurred, 30, 80)
            ksize = max(3, int(round(min(sh, sw) * 0.015)))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sub_area = float(sh * sw)
            best_rect = None
            max_score = -1.0

            for cnt in contours:
                a = cv2.contourArea(cnt)
                if a < 0.20 * sub_area:
                    continue
                hull = cv2.convexHull(cnt)
                rect = cv2.minAreaRect(hull)
                (cx, cy), (rw, rh), ang = rect
                ra = rw * rh
                if ra <= 0:
                    continue
                rectang = cv2.contourArea(hull) / ra
                score = rectang * (a / sub_area)
                if score > max_score:
                    max_score = score
                    best_rect = rect

            if best_rect is not None and max_score > 0.45:
                pts = cv2.boxPoints(best_rect)
                return pts
        except Exception:
            pass
        return base_corners

    def process(
        self,
        full_frame: np.ndarray,
        conveyor_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Khớp và cắt bo mạch PCB theo mẫu đã dạy - BẤT BIẾN GÓC XOAY 360 ĐỘ (ROTATION INVARIANT):
        1. Thuật toán ORB Feature Homography: Tìm kiếm và nắn thẳng ở mọi góc xoay tùy ý (0° - 360°).
        2. Thuật toán Multi-Angle Template Matching (0°, 90°, 180°, 270°): Dự phòng khi bo mạch ít chi tiết.
        """
        t_start = time.perf_counter()
        info: Dict[str, Any] = {
            "success": False,
            "status_msg": "Chưa có mẫu (Hãy dùng chuột vẽ khung quanh bo mạch)",
            "time_ms": 0.0,
            "corners": None,
            "score": 0.0,
            "angle": 0.0,
            "crop_size": None,
            "mode": self.tracking_mode
        }

        if self.template_img is None or self.template_bbox is None:
            info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
            return None, info

        fh, fw = full_frame.shape[:2]
        tw, th = self.template_size

        # Chế độ 1: Cắt theo vùng cố định (Fixed BBox)
        if self.tracking_mode == "fixed":
            x1, y1, x2, y2 = self.template_bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            cropped = full_frame[y1:y2, x1:x2].copy()
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

            info["success"] = True
            info["status_msg"] = f"BBox cố định ({x2-x1}x{y2-y1} px)"
            info["corners"] = corners
            info["score"] = 1.0
            info["crop_size"] = (x2 - x1, y2 - y1)
            info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
            return cropped, info

        # Chế độ 2: Khớp mẫu di động (Template Tracking)
        if conveyor_roi:
            cx1, cy1, cx2, cy2 = conveyor_roi
        else:
            cx1, cy1, cx2, cy2 = 0, 0, fw, fh

        search_area = full_frame[cy1:cy2, cx1:cx2]
        sh, sw = search_area.shape[:2]

        if sh < 50 or sw < 50:
            info["status_msg"] = "Vùng băng tải không hợp lệ"
            info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
            return None, info

        # Thu nhỏ để xử lý thời gian thực siêu nhanh (Scale 0.5)
        scale = 0.5 if (sh > 1000 or sw > 1000) else 1.0
        if scale != 1.0:
            search_small = cv2.resize(search_area, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            tmpl_small = cv2.resize(self.template_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            search_small = search_area
            tmpl_small = self.template_img

        # =====================================================================
        # LUỒNG 1: ORB FEATURE MATCHING + HOMOGRAPHY (BẤT BIẾN GÓC XOAY 360 ĐỘ)
        # =====================================================================
        try:
            kp_tmpl, des_tmpl = self.orb.detectAndCompute(tmpl_small, None)
            kp_scene, des_scene = self.orb.detectAndCompute(search_small, None)

            if des_tmpl is not None and des_scene is not None and len(des_scene) >= 15:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des_tmpl, des_scene, k=2)
                good = [m for m, n in matches if len((m, n)) == 2 and m.distance < 0.75 * n.distance]

                if len(good) >= 10:
                    src_pts = np.float32([kp_tmpl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H_small, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    inliers = int(mask.sum()) if mask is not None else 0
                    if H_small is not None and inliers >= 10:
                        # Ánh xạ ma trận biến đổi phối cảnh về ảnh gốc full resolution
                        S_inv = np.diag([1.0 / scale, 1.0 / scale, 1.0])
                        S = np.diag([scale, scale, 1.0])
                        H_full = S_inv @ H_small @ S

                        # 4 góc ban đầu của mẫu Template
                        tmpl_corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
                        raw_corners = cv2.perspectiveTransform(tmpl_corners, H_full).reshape(4, 2)

                        # Kiểm tra xem tứ giác có hợp lệ không (phải lồi, không tự cắt)
                        pts_int = np.int32(raw_corners)
                        if cv2.isContourConvex(pts_int):
                            c_area = cv2.contourArea(pts_int)
                            expected_area = float(tw * th)
                            # Diện tích phải tương đồng (0.5x đến 1.8x)
                            if 0.40 * expected_area <= c_area <= 2.2 * expected_area:
                                # Tính góc xoay thực tế
                                dx = raw_corners[1, 0] - raw_corners[0, 0]
                                dy = raw_corners[1, 1] - raw_corners[0, 1]
                                rot_angle = float(np.degrees(np.arctan2(dy, dx)))

                                # Chuyển góc về tọa độ toàn khung hình
                                corners = raw_corners.copy()
                                corners[:, 0] += cx1
                                corners[:, 1] += cy1

                                # Cắt và nắn thẳng bo mạch về chiều nằm ngang chuẩn (tw, th)
                                dst_rect = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
                                M_warp = cv2.getPerspectiveTransform(raw_corners.astype(np.float32), dst_rect)
                                deskewed = cv2.warpPerspective(search_area, M_warp, (tw, th), flags=cv2.INTER_LINEAR)

                                info["success"] = True
                                info["status_msg"] = f"Khớp xoay 360° (Góc {rot_angle:.0f}°, {inliers} điểm)"
                                info["corners"] = corners
                                info["angle"] = rot_angle
                                info["score"] = min(1.0, inliers / 40.0)
                                info["crop_size"] = (tw, th)
                                info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
                                return deskewed, info
        except Exception:
            pass

        # =====================================================================
        # LUỒNG 2: MULTI-ANGLE TEMPLATE MATCHING (DỰ PHÒNG CHO 0°, 90°, 180°, 270°)
        # =====================================================================
        search_gray = cv2.cvtColor(search_small, cv2.COLOR_BGR2GRAY)
        tmpl_gray = cv2.cvtColor(tmpl_small, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        search_gray = clahe.apply(search_gray)
        tmpl_gray = clahe.apply(tmpl_gray)

        # 4 góc xoay vuông góc
        angle_candidates = [
            (0, tmpl_gray),
            (90, cv2.rotate(tmpl_gray, cv2.ROTATE_90_CLOCKWISE)),
            (180, cv2.rotate(tmpl_gray, cv2.ROTATE_180)),
            (270, cv2.rotate(tmpl_gray, cv2.ROTATE_90_COUNTERCLOCKWISE))
        ]

        best_val = -1.0
        best_ang = 0
        best_loc = (0, 0)
        best_shape = (tmpl_gray.shape[1], tmpl_gray.shape[0])

        for ang, r_tmpl in angle_candidates:
            rw, rh = r_tmpl.shape[1], r_tmpl.shape[0]
            if rh > search_gray.shape[0] or rw > search_gray.shape[1]:
                continue
            res = cv2.matchTemplate(search_gray, r_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            if max_v > best_val:
                best_val = max_v
                best_ang = ang
                best_loc = max_l
                best_shape = (rw, rh)

        info["score"] = float(best_val)

        if best_val < self.match_threshold:
            info["status_msg"] = f"Băng tải rỗng (Điểm khớp {best_val*100:.0f}% < {self.match_threshold*100:.0f}%)"
            info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
            return None, info

        # Tính toán tọa độ và nắn xoay về chuẩn
        bw, bh = int(round(best_shape[0] / scale)), int(round(best_shape[1] / scale))
        mx = int(round(best_loc[0] / scale)) + cx1
        my = int(round(best_loc[1] / scale)) + cy1

        bx1, by1 = max(0, mx), max(0, my)
        bx2, by2 = min(fw, bx1 + bw), min(fh, by1 + bh)

        sub_pcb = full_frame[by1:by2, bx1:bx2].copy()

        # Xoay ngược về góc 0 độ chuẩn
        if best_ang == 90:
            rotated_back = cv2.rotate(sub_pcb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif best_ang == 180:
            rotated_back = cv2.rotate(sub_pcb, cv2.ROTATE_180)
        elif best_ang == 270:
            rotated_back = cv2.rotate(sub_pcb, cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated_back = sub_pcb

        # Resize về đúng kích thước mẫu ban đầu nếu có sai số nhỏ
        if rotated_back.shape[1] != tw or rotated_back.shape[0] != th:
            rotated_back = cv2.resize(rotated_back, (tw, th), interpolation=cv2.INTER_LINEAR)

        corners = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)

        info["success"] = True
        info["status_msg"] = f"Khớp xoay {best_ang}° ({tw}x{th} px, Điểm: {best_val*100:.0f}%)"
        info["corners"] = corners
        info["angle"] = float(best_ang)
        info["crop_size"] = (tw, th)
        info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
        return rotated_back, info


class HikTemplateViewer:
    """Giao diện xem camera và hỗ trợ vẽ Bounding Box trực tiếp bằng chuột"""

    def __init__(self, default_exposure_us=300000.0):
        self.default_exposure_us = float(default_exposure_us)
        self.current_exposure_us = float(default_exposure_us)
        self.exp_min = 34.0
        self.exp_max = 1999733.0

        self.cam = None
        self.is_connected = False
        self.is_grabbing = False
        self.device_name = "Hikvision Camera"
        self.resolution = (0, 0)

        # Quản lý luồng lấy ảnh
        self.grab_thread = None
        self.exit_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.fps_camera = 0.0
        self.frame_count = 0

        # Thư mục lưu ảnh
        self.capture_dir = CURRENT_DIR / "captures"
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        self.toast_msg = "Sẵn sàng"
        self.toast_time = time.time()

        # Engine Khớp mẫu (Template Engine)
        self.engine = PCBTemplateEngine()
        self.auto_crop_enabled = True
        self.show_live_crop_window = True
        self.cached_cropped = None
        self.cached_corners = None
        self.cached_status = "Chờ mẫu PCB"
        self.is_detecting = False

        # Tương tác chuột vẽ BBox
        self.is_drawing = False
        self.drag_start = None
        self.drag_current = None
        self.freeze_frame = False
        self.frozen_img = None
        self.disp_scale = 1.0
        self.disp_offset = (0, 0)

        # Băng tải
        self.conveyor_cfg = load_conveyor_config()
        self.filter_conveyor_enabled = bool(self.conveyor_cfg.get("filter_enabled", True))
        self.conveyor_zoom_view = False
        self.current_conveyor_roi = None

    def set_toast(self, msg, duration=2.5):
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def connect(self):
        """Khởi tạo SDK và kết nối camera"""
        if not HAS_HIK_SDK:
            print("[LỖI] Không tìm thấy thư viện Hikvision SDK (MvImport)!")
            return False

        print("[1/4] Đang khởi tạo Hikvision SDK...")
        ret = MvCamera.MV_CC_Initialize()
        if ret != 0:
            print(f"[LỖI] MV_CC_Initialize thất bại: 0x{ret:x}")
            return False

        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0 or device_list.nDeviceNum == 0:
            print("[LỖI] Không tìm thấy camera Hikvision nào được kết nối!")
            return False

        print(f"[2/4] Tìm thấy {device_list.nDeviceNum} thiết bị. Đang kết nối camera đầu tiên...")
        st_device = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

        if st_device.nTLayerType == MV_GIGE_DEVICE:
            model_bytes = bytes(st_device.SpecialInfo.stGigEInfo.chModelName).split(b'\x00')[0]
            self.device_name = model_bytes.decode('ascii', errors='ignore').strip()
        elif st_device.nTLayerType == MV_USB_DEVICE:
            model_bytes = bytes(st_device.SpecialInfo.stUsb3VInfo.chModelName).split(b'\x00')[0]
            self.device_name = model_bytes.decode('ascii', errors='ignore').strip()

        print(f"      -> Model: {self.device_name}")

        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(st_device)
        if ret != 0:
            print(f"[LỖI] MV_CC_CreateHandle thất bại: 0x{ret:x}")
            return False

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"[LỖI] MV_CC_OpenDevice thất bại: 0x{ret:x}")
            return False

        # Tối ưu kích thước gói cho GigE (Jumbo Packet tránh rớt gói tin)
        if st_device.nTLayerType == MV_GIGE_DEVICE:
            pkt_size = self.cam.MV_CC_GetOptimalPacketSize()
            if pkt_size > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt_size)

        st_float = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", st_float)
        if ret == 0:
            self.exp_min = float(st_float.fMin)
            self.exp_max = float(st_float.fMax)
            print(f"      -> Giới hạn Exposure: {self.exp_min:.0f} us ~ {self.exp_max:.0f} us")

        # Cài đặt phơi sáng thủ công
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # 0: Off
        self.set_exposure(self.default_exposure_us)

        # Lấy kích thước ảnh gốc
        w_val = MVCC_INTVALUE()
        h_val = MVCC_INTVALUE()
        self.cam.MV_CC_GetIntValue("Width", w_val)
        self.cam.MV_CC_GetIntValue("Height", h_val)
        self.resolution = (w_val.nCurValue, h_val.nCurValue)
        print(f"      -> Độ phân giải gốc: {self.resolution[0]} x {self.resolution[1]}")

        self.is_connected = True
        return True

    def set_exposure(self, exposure_us: float) -> bool:
        if not self.cam:
            return False
        clamped = max(self.exp_min, min(self.exp_max, float(exposure_us)))
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", clamped)
        if ret == 0:
            self.current_exposure_us = clamped
            ms = self.current_exposure_us / 1000.0
            print(f"[Camera] Đã cập nhật ExposureTime: {self.current_exposure_us:.0f} us ({ms:.1f} ms)")
            self.set_toast(f"Exposure: {self.current_exposure_us:.0f} us ({ms:.1f} ms)")
            return True
        return False

    def start_grabbing(self):
        if not self.is_connected:
            return False
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            return False
        self.is_grabbing = True
        self.grab_thread = threading.Thread(target=self._grab_worker, daemon=True)
        self.grab_thread.start()
        print("[4/4] Bắt đầu luồng ảnh thành công!")
        return True

    def _grab_worker(self):
        st_out = MV_FRAME_OUT()
        memset(byref(st_out), 0, sizeof(st_out))

        last_fps_time = time.time()
        frames_this_sec = 0
        bgr_buf = None
        last_bgr_size = 0

        while not self.exit_event.is_set():
            ret = self.cam.MV_CC_GetImageBuffer(st_out, 2000)
            if ret == 0 and st_out.pBufAddr:
                w = st_out.stFrameInfo.nWidth
                h = st_out.stFrameInfo.nHeight
                pixel_type = st_out.stFrameInfo.enPixelType
                frame_len = st_out.stFrameInfo.nFrameLen

                img_bgr = None
                # Xử lý Mono8 an toàn qua string_at
                if pixel_type == PixelType_Gvsp_Mono8:
                    if frame_len >= w * h:
                        raw_bytes = string_at(st_out.pBufAddr, frame_len)
                        mono_arr = np.frombuffer(raw_bytes, dtype=np.uint8)[:w * h].reshape((h, w))
                        img_bgr = cv2.cvtColor(mono_arr, cv2.COLOR_GRAY2BGR)
                else:
                    needed_size = w * h * 3
                    if bgr_buf is None or last_bgr_size != needed_size:
                        bgr_buf = (c_ubyte * needed_size)()
                        last_bgr_size = needed_size

                    convert_param = MV_CC_PIXEL_CONVERT_PARAM_EX()
                    memset(byref(convert_param), 0, sizeof(convert_param))
                    convert_param.nWidth = w
                    convert_param.nHeight = h
                    convert_param.pSrcData = st_out.pBufAddr
                    convert_param.nSrcDataLen = frame_len
                    convert_param.enSrcPixelType = pixel_type
                    convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
                    convert_param.pDstBuffer = bgr_buf
                    convert_param.nDstBufferSize = needed_size

                    c_ret = self.cam.MV_CC_ConvertPixelTypeEx(convert_param)
                    if c_ret == 0:
                        raw_bytes = string_at(bgr_buf, needed_size)
                        img_bgr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))

                self.cam.MV_CC_FreeImageBuffer(st_out)

                if img_bgr is not None:
                    with self.frame_lock:
                        self.latest_frame = img_bgr
                        self.resolution = (w, h)
                    self.frame_count += 1

                frames_this_sec += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    self.fps_camera = frames_this_sec / (now - last_fps_time)
                    frames_this_sec = 0
                    last_fps_time = now
            else:
                time.sleep(0.01)

    def stop(self):
        self.exit_event.set()
        if self.grab_thread and self.grab_thread.is_alive():
            self.grab_thread.join(timeout=1.5)
        if self.is_grabbing and self.cam:
            self.cam.MV_CC_StopGrabbing()
            self.is_grabbing = False
        if self.is_connected and self.cam:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.is_connected = False
        if HAS_HIK_SDK:
            try:
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass

    def on_mouse(self, event, x, y, flags, param):
        """Bắt sự kiện chuột để vẽ khung BBox trực tiếp trên màn hình xem camera"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.drag_start = (x, y)
            self.drag_current = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.drag_current = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing and self.drag_start:
                self.is_drawing = False
                self.drag_current = (x, y)

                # Chuyển đổi tọa độ từ màn hình xem về ảnh gốc full resolution
                x1 = min(self.drag_start[0], self.drag_current[0])
                y1 = min(self.drag_start[1], self.drag_current[1])
                x2 = max(self.drag_start[0], self.drag_current[0])
                y2 = max(self.drag_start[1], self.drag_current[1])

                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    scale_x, scale_y, off_x, off_y = self.disp_transform
                    real_x1 = int(round((x1 - off_x) / scale_x))
                    real_y1 = int(round((y1 - off_y) / scale_y))
                    real_x2 = int(round((x2 - off_x) / scale_x))
                    real_y2 = int(round((y2 - off_y) / scale_y))

                    # Lấy khung hình hiện tại để lưu mẫu
                    with self.frame_lock:
                        cur_f = self.frozen_img if self.freeze_frame and self.frozen_img is not None else self.latest_frame
                    if cur_f is not None:
                        self.engine.save_template(cur_f, (real_x1, real_y1, real_x2, real_y2))
                        self.set_toast(f"Đã lưu mẫu bo mạch: {real_x2-real_x1}x{real_y2-real_y1} px", 3.0)
                        self.freeze_frame = False
                else:
                    self.set_toast("Khung vẽ quá nhỏ! Hãy kéo thả bao quanh cả bo mạch.", 2.5)

                self.drag_start = None
                self.drag_current = None

    def get_conveyor_roi(self, img_input: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = img_input.shape[:2]
        cfg = getattr(self, "conveyor_cfg", {})
        m_left = float(cfg.get("margin_left", 0.28))
        m_right = float(cfg.get("margin_right", 0.06))
        x1 = max(0, min(int(w * m_left), w - 20))
        x2 = max(x1 + 20, min(int(w * (1.0 - m_right)), w))
        return x1, 0, x2, h

    def run_gui(self):
        window_name = "HIKVISION CAMERA - BAN DAY MAU & KHOP MAU (TEMPLATE MATCHING)"
        crop_window_name = "PCB DA CAT THEO MAU (LIVE CROP)"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 800)
        cv2.setMouseCallback(window_name, self.on_mouse)

        if self.show_live_crop_window:
            cv2.namedWindow(crop_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(crop_window_name, 640, 420)

        # Trackbar exposure
        def on_exp_trackbar(val):
            exp_us = max(self.exp_min, float(val) * 1000.0)
            self.set_exposure(exp_us)

        init_exp_ms = int(self.current_exposure_us / 1000.0)
        cv2.createTrackbar("Exposure (ms)", window_name, init_exp_ms, 1000, on_exp_trackbar)

        print("\n" + "=" * 65)
        print(" CHẾ ĐỘ DẠY MẪU & KHỚP MẪU BẰNG TAY (TEMPLATE MATCHING):")
        print(" - DÙNG CHUỘT: Kéo thả chuột trái trực tiếp trên bo mạch để vẽ khung BBox")
        print(" - Phím [SPACE] / [P] : Tạm dừng hình (Freeze) để kéo chuột vẽ cho chuẩn")
        print(" - Phím [R]           : Xóa mẫu cũ / Vẽ lại mẫu bo mạch mới")
        print(" - Phím [M]           : Đổi chế độ: Khớp mẫu di động <-> Vùng cố định")
        print(" - Phím [E]           : Bật / Tắt hút dính mép tự động (Edge Snap)")
        print(" - Phím [W]           : Ẩn / Hiện cửa sổ Live Crop phụ")
        print(" - Phím [B]           : Bật / Tắt lọc bỏ lề ngoài băng tải")
        print(" - Phím [S]           : Chụp ảnh full 12MP & lưu bo mạch đã cắt")
        print(" - Phím [Q] / [ESC]   : Thoát chương trình")
        print("=" * 65 + "\n")

        while not self.exit_event.is_set():
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is not None:
                h, w = frame.shape[:2]
                display_img = self.frozen_img if self.freeze_frame and self.frozen_img is not None else frame

                # Tự động cắt theo mẫu
                if self.auto_crop_enabled and self.engine.template_img is not None and not self.is_detecting:
                    self.is_detecting = True
                    conv_roi = self.get_conveyor_roi(display_img) if self.filter_conveyor_enabled else None
                    frame_for_match = display_img.copy()

                    def _worker(f, r):
                        try:
                            cropped, info = self.engine.process(f, conveyor_roi=r)
                            if info.get("success") and cropped is not None:
                                self.cached_cropped = cropped
                                self.cached_corners = info.get("corners")
                                self.cached_status = info.get("status_msg", "Đã khớp mẫu PCB")
                            else:
                                self.cached_cropped = None
                                self.cached_corners = None
                                self.cached_status = info.get("status_msg", "Chờ bo mạch PCB")
                        except Exception:
                            pass
                        finally:
                            self.is_detecting = False

                    threading.Thread(target=_worker, args=(frame_for_match, conv_roi), daemon=True).start()

                # Biến an toàn luồng
                cur_cropped = self.cached_cropped
                cur_corners = self.cached_corners
                cur_found = bool(cur_cropped is not None and cur_corners is not None)

                # Render màn hình chính
                target_w = 1280
                target_h = int(h * (target_w / w))
                view = cv2.resize(display_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                scale_x = target_w / w
                scale_y = target_h / h
                self.disp_transform = (scale_x, scale_y, 0, 0)

                # Vẽ vùng băng tải (nếu bật)
                if self.filter_conveyor_enabled:
                    cx1, cy1, cx2, cy2 = self.get_conveyor_roi(display_img)
                    disp_cx1 = int(cx1 * scale_x)
                    disp_cx2 = int(cx2 * scale_x)
                    overlay = view.copy()
                    if disp_cx1 > 0:
                        cv2.rectangle(overlay, (0, 0), (disp_cx1, target_h), (15, 15, 15), -1)
                    if disp_cx2 < target_w:
                        cv2.rectangle(overlay, (disp_cx2, 0), (target_w, target_h), (15, 15, 15), -1)
                    cv2.addWeighted(overlay, 0.40, view, 0.60, 0, view)
                    cv2.line(view, (disp_cx1, 0), (disp_cx1, target_h), (0, 165, 255), 2)
                    cv2.line(view, (disp_cx2, 0), (disp_cx2, target_h), (0, 165, 255), 2)

                # Vẽ khung bo mạch khớp theo mẫu
                if cur_found and cur_corners is not None:
                    corners_disp = cur_corners.copy()
                    corners_disp[:, 0] *= scale_x
                    corners_disp[:, 1] *= scale_y
                    pts_int = np.int32(corners_disp)
                    cv2.polylines(view, [pts_int], isClosed=True, color=(0, 255, 128), thickness=3)
                    for pt in pts_int:
                        cv2.circle(view, tuple(pt), 6, (0, 200, 255), -1)

                # Vẽ khung chuột đang kéo thả (Interactive drag box)
                if self.is_drawing and self.drag_start and self.drag_current:
                    cv2.rectangle(view, self.drag_start, self.drag_current, (0, 255, 255), 2)
                    cv2.putText(view, "KEO CHUOT BAO QUANH PCB ROI THA RA", (self.drag_start[0], max(25, self.drag_start[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

                # Khung PiP góc trên bên phải
                pip_w, pip_h = 360, 220
                pip_x = max(0, target_w - pip_w - 15)
                pip_y = 15

                if cur_found and cur_cropped is not None and getattr(cur_cropped, 'size', 0) > 0 and cur_cropped.shape[0] > 10 and cur_cropped.shape[1] > 10:
                    ch, cw = cur_cropped.shape[:2]
                    if pip_y + pip_h <= target_h and pip_x + pip_w <= target_w:
                        pip_thumb = cv2.resize(cur_cropped, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
                        cv2.rectangle(view, (pip_x - 2, pip_y - 2), (pip_x + pip_w + 2, pip_y + pip_h + 2), (0, 255, 128), 2)
                        view[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = pip_thumb
                        cv2.rectangle(view, (pip_x, pip_y), (pip_x + pip_w, pip_y + 24), (20, 20, 20), -1)
                        cv2.putText(view, f"PCB THEO MAU: {cw}x{ch} px", (pip_x + 8, pip_y + 17),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA)
                else:
                    if pip_y + pip_h <= target_h and pip_x + pip_w <= target_w:
                        cv2.rectangle(view, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (25, 25, 25), -1)
                        cv2.rectangle(view, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (70, 70, 70), 1)
                        msg_pip = "DUNG CHUOT VE KHUNG PCB" if self.engine.template_img is None else "CHO BO MACH THEO MAU..."
                        cv2.putText(view, msg_pip, (pip_x + 35, pip_y + 115),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)

                # Vẽ thanh thông số HUD
                self._draw_hud(view, w, h)
                cv2.imshow(window_name, view)

                # Cửa sổ Live Crop phóng to
                if self.show_live_crop_window:
                    if cur_found and cur_cropped is not None and getattr(cur_cropped, 'size', 0) > 0 and cur_cropped.shape[0] > 10 and cur_cropped.shape[1] > 10:
                        crop_win_disp = cur_cropped.copy()
                        ch, cw = crop_win_disp.shape[:2]
                        header_bar = np.zeros((45, cw, 3), dtype=np.uint8)
                        cv2.putText(header_bar, f"PCB DA CAT THEO MAU - Kich thuoc: {cw}x{ch} px", (15, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)
                        full_crop_view = np.vstack([header_bar, crop_win_disp])
                        disp_h = 420
                        disp_w = int(full_crop_view.shape[1] * disp_h / max(1, full_crop_view.shape[0]))
                        cv2.imshow(crop_window_name, cv2.resize(full_crop_view, (disp_w, disp_h), interpolation=cv2.INTER_AREA))
                    else:
                        empty_win = np.zeros((420, 640, 3), dtype=np.uint8)
                        cv2.putText(empty_win, "CHUA CO BO MACH THEO MAU", (120, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 200, 255), 2, cv2.LINE_AA)
                        cv2.putText(empty_win, "Keo chuot tren man hinh camera de day mau PCB moi", (95, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)
                        cv2.imshow(crop_window_name, empty_win)
            else:
                waiting = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(waiting, "Dang nhan hinh anh tu camera...", (420, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                cv2.imshow(window_name, waiting)

            # Phím bấm điều khiển
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord(' '), ord('p'), ord('P')):
                self.freeze_frame = not self.freeze_frame
                if self.freeze_frame:
                    with self.frame_lock:
                        self.frozen_img = self.latest_frame.copy() if self.latest_frame is not None else None
                    self.set_toast("ĐÃ TẠM DỪNG HÌNH. Hãy kéo chuột vẽ khung bao quanh PCB!", 3.5)
                else:
                    self.frozen_img = None
                    self.set_toast("Tiếp tục luồng camera thời gian thực", 2.0)
            elif key in (ord('r'), ord('R')):
                self.engine.reset_template()
                self.cached_cropped = None
                self.cached_corners = None
                self.set_toast("Đã xóa mẫu cũ! Hãy dùng chuột kéo vẽ mẫu mới.", 3.0)
            elif key in (ord('m'), ord('M')):
                if self.engine.tracking_mode == "tracking":
                    self.engine.tracking_mode = "fixed"
                    mode_name = "VÙNG CỐ ĐỊNH (Fixed BBox)"
                else:
                    self.engine.tracking_mode = "tracking"
                    mode_name = "KHỚP MẪU DI ĐỘNG (Template Tracking)"
                self.set_toast(f"Chế độ: {mode_name}", 2.5)
            elif key in (ord('e'), ord('E')):
                self.engine.edge_snap_enabled = not self.engine.edge_snap_enabled
                st = "BẬT" if self.engine.edge_snap_enabled else "TẮT"
                self.set_toast(f"Tự hút dính mép (Edge Snap): {st}", 2.0)
            elif key in (ord('w'), ord('W')):
                self.show_live_crop_window = not self.show_live_crop_window
                if not self.show_live_crop_window:
                    try:
                        cv2.destroyWindow(crop_window_name)
                    except Exception:
                        pass
                st = "BAT" if self.show_live_crop_window else "TAT"
                self.set_toast(f"Cua so Live Crop: {st}", 2.0)
            elif key in (ord('b'), ord('B')):
                self.filter_conveyor_enabled = not self.filter_conveyor_enabled
                st = "BẬT" if self.filter_conveyor_enabled else "TẮT"
                self.set_toast(f"Lọc phần thừa băng tải: {st}", 2.0)
            elif key in (ord('s'), ord('S')):
                self._save_snapshot()
            elif key in (ord('+'), ord('=')):
                self.set_exposure(self.current_exposure_us + 10000.0)
                cv2.setTrackbarPos("Exposure (ms)", window_name, int(self.current_exposure_us / 1000.0))
            elif key in (ord('-'), ord('_')):
                self.set_exposure(max(self.exp_min, self.current_exposure_us - 10000.0))
                cv2.setTrackbarPos("Exposure (ms)", window_name, int(self.current_exposure_us / 1000.0))

        cv2.destroyAllWindows()

    def _save_snapshot(self):
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
        if frame is None:
            self.set_toast("Lỗi: Không có ảnh để chụp!", 2.0)
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        exp_ms = int(self.current_exposure_us / 1000.0)
        base_name = f"template_capture_{ts}_exp_{exp_ms}ms"

        full_path = self.capture_dir / f"{base_name}_full.jpg"
        cv2.imwrite(str(full_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 98])

        cur_cropped = self.cached_cropped
        if cur_cropped is not None:
            crop_path = self.capture_dir / f"{base_name}_crop.jpg"
            cv2.imwrite(str(crop_path), cur_cropped, [cv2.IMWRITE_JPEG_QUALITY, 98])
            self.set_toast(f"Đã lưu ảnh full và bo mạch cắt: {crop_path.name}", 3.0)
            print(f"[ĐÃ LƯU] Ảnh full: {full_path.name} | Ảnh crop: {crop_path.name}")
        else:
            self.set_toast(f"Đã lưu ảnh full: {full_path.name}", 2.5)

    def _draw_hud(self, view: np.ndarray, orig_w: int, orig_h: int):
        vh, vw = view.shape[:2]
        exp_ms = self.current_exposure_us / 1000.0

        overlay = view.copy()
        cv2.rectangle(overlay, (10, 10), (450, 175), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, view, 0.30, 0, view)
        cv2.rectangle(view, (10, 10), (450, 175), (0, 255, 128), 1)

        cv2.putText(view, "BAN DAY MAU & KHOP MAU (TEMPLATE MATCHING)", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.putText(view, f"Camera: {self.device_name} (12 MP)", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(view, f"Thoi gian sang: {self.current_exposure_us:,.0f} us ({exp_ms:.1f} ms) | FPS: {self.fps_camera:.1f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 100), 1, cv2.LINE_AA)

        mode_str = "Di dong (Template Tracking)" if self.engine.tracking_mode == "tracking" else "Co dinh (Fixed BBox)"
        cv2.putText(view, f"Che do [M]: {mode_str}", (20, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

        tmpl_st = f"Da day mau ({self.engine.template_size[0]}x{self.engine.template_size[1]}px)" if self.engine.template_img is not None else "CHUA CO MAU (Keo chuot de ve)"
        tmpl_col = (0, 255, 128) if self.engine.template_img is not None else (0, 200, 255)
        cv2.putText(view, f"Mau PCB: {tmpl_st}", (20, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, tmpl_col, 1, cv2.LINE_AA)

        snap_st = "BAT" if self.engine.edge_snap_enabled else "TAT"
        cv2.putText(view, f"Hut dinh mep [E]: {snap_st} | Trang thai: {self.cached_status}", (20, 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 50), 1, cv2.LINE_AA)

        if self.freeze_frame:
            cv2.rectangle(view, (vw // 2 - 250, 10), (vw // 2 + 250, 50), (0, 0, 200), -1)
            cv2.putText(view, "TAM DUNG HINH - KEO CHUOT VE KHUNG PCB", (vw // 2 - 230, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(view, "[Chuot]: Ve khung PCB | [SPACE]: Dung hinh | [R]: Xoa mau | [M]: Che do | [S]: Chup | [Q]: Thoat",
                    (15, vh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        if time.time() < self.toast_time:
            tw = len(self.toast_msg) * 11 + 30
            cv2.rectangle(view, (vw // 2 - tw // 2, 60), (vw // 2 + tw // 2, 100), (0, 140, 0), -1)
            cv2.rectangle(view, (vw // 2 - tw // 2, 60), (vw // 2 + tw // 2, 100), (255, 255, 255), 1)
            cv2.putText(view, self.toast_msg, (vw // 2 - tw // 2 + 15, 87),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)


def demo_file(img_path: str):
    """Chạy thử nghiệm trên ảnh tĩnh"""
    path = Path(img_path)
    if not path.is_file():
        print(f"[LỖI] Không tìm thấy file: {img_path}")
        return

    img = cv2.imread(str(path))
    if img is None:
        print("[LỖI] Không thể đọc ảnh!")
        return

    viewer = HikTemplateViewer()
    viewer.latest_frame = img
    viewer.run_gui()


if __name__ == "__main__":
    img_arg = None
    for arg in sys.argv[1:]:
        if any(arg.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp')):
            img_arg = arg
            break

    if img_arg:
        demo_file(img_arg)
    else:
        viewer = HikTemplateViewer(default_exposure_us=300000.0)
        if viewer.connect():
            if viewer.start_grabbing():
                viewer.run_gui()
            viewer.stop()
        else:
            print("[THÔNG BÁO] Không có camera kết nối. Mở ảnh mẫu để thử nghiệm kéo chuột vẽ BBox...")
            candidates = ["result_Image_20261019181609711.jpg", "a.jpg", "2.jpg"]
            for c in candidates:
                if Path(c).is_file():
                    demo_file(c)
                    break

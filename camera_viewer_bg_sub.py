"""
=============================================================================
   HIKVISION CAMERA VIEWER - BẢN NHẬN DIỆN & CẮT PCB BẰNG PHƯƠNG PHÁP TRỪ NỀN
                     (BACKGROUND SUBTRACTION INDUSTRIAL AOI)
=============================================================================
Thuật toán Trừ nền (Background Subtraction):
1. Chụp / Ghi nhớ ảnh nền băng tải rỗng (Reference Background):
   - Khi băng tải chưa có bo mạch, bấm phím [SPACE] hoặc [K] để chụp lưu nền chuẩn.
   - Hệ thống tự động lưu vào file 'conveyor_background.jpg' và 'bg_sub_config.json'.
   - Lần sau mở máy tự động tải lại nền cũ, không cần chụp lại!
2. Phép trừ nền triệt tiêu ngoại cảnh (Absolute Difference & Thresholding):
   - Hiệu số cv2.absdiff(Khung_hình_hiện_tại, Nền_băng_tải).
   - Nền băng tải (dù màu đen, xám hay có ray kim loại) bị triệt tiêu 100% về 0.
   - Bo mạch PCB (vật thể mới xuất hiện) nổi bật hoàn toàn thành màu trắng!
3. Bất biến góc xoay 360° & Nắn thẳng tự động (Deskewing):
   - Trích xuất bao lồi Convex Hull và hình chữ nhật xoay minAreaRect.
   - Bắt dính bo mạch ở bất kỳ góc xoay nào (0° đến 360°, xoay ngang, dọc, chéo).
   - Tự động nắn phẳng phiu bo mạch về chiều nằm ngang chuẩn mực.
4. Hỗ trợ 2 chế độ:
   - Chế độ 1: Trừ nền tĩnh chuẩn (Static Absdiff - Cực nhanh, chỉ 15-30ms, chuẩn công nghiệp).
   - Chế độ 2: Trừ nền động tự thích ứng (Dynamic MOG2 - Bấm phím [M] để đổi).
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
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
BG_CONFIG_FILE = CURRENT_DIR / "bg_sub_config.json"
BG_IMG_FILE = CURRENT_DIR / "conveyor_background.jpg"
CONVEYOR_CONFIG_FILE = CURRENT_DIR / "conveyor_config.json"
COLOR_CONFIG_FILE = CURRENT_DIR / "pcb_color_config.json"

# Danh sách bộ màu mẫu công nghiệp chuẩn (Presets)
COLOR_PRESETS = [
    {
        "id": 1,
        "name": "Băng tải Đen + PCB Xanh lá (Chuẩn FR4)",
        "conveyor": [25, 25, 25],
        "conveyor_name": "Băng tải Đen",
        "pcb": [45, 135, 50],
        "pcb_name": "PCB Xanh lá",
        "badge": "🏭 Đen + 🟩 Xanh lá"
    },
    {
        "id": 2,
        "name": "Băng tải Đen + PCB Xanh dương (Blue PCB)",
        "conveyor": [25, 25, 25],
        "conveyor_name": "Băng tải Đen",
        "pcb": [140, 70, 30],
        "pcb_name": "PCB Xanh dương",
        "badge": "🏭 Đen + 🟦 Xanh dương"
    },
    {
        "id": 3,
        "name": "Băng tải Đen + PCB Đỏ (Red PCB)",
        "conveyor": [25, 25, 25],
        "conveyor_name": "Băng tải Đen",
        "pcb": [35, 40, 150],
        "pcb_name": "PCB Đỏ",
        "badge": "🏭 Đen + 🟥 Đỏ"
    },
    {
        "id": 4,
        "name": "Băng tải Đen + PCB Vàng (Yellow PCB)",
        "conveyor": [25, 25, 25],
        "conveyor_name": "Băng tải Đen",
        "pcb": [30, 150, 160],
        "pcb_name": "PCB Vàng",
        "badge": "🏭 Đen + 🟨 Vàng"
    },
    {
        "id": 5,
        "name": "Băng tải Xám + PCB Xanh lá",
        "conveyor": [90, 90, 95],
        "conveyor_name": "Băng tải Xám",
        "pcb": [45, 135, 50],
        "pcb_name": "PCB Xanh lá",
        "badge": "⚙️ Xám + 🟩 Xanh lá"
    },
    {
        "id": 6,
        "name": "Băng tải Xám + PCB Đen (Black PCB)",
        "conveyor": [90, 90, 95],
        "conveyor_name": "Băng tải Xám",
        "pcb": [25, 25, 25],
        "pcb_name": "PCB Đen",
        "badge": "⚙️ Xám + ⬛ Đen"
    }
]

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

import argparse
from camera_manager import (
    CameraSource,
    scan_all_camera_sources,
    show_camera_picker_dialog,
    terminal_camera_picker,
    load_camera_config,
    save_camera_config,
    parse_camera_source_arg,
    SuppressStderr,
    show_color_picker_dialog,
    bgr_to_hex,
    hex_to_bgr
)


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


class PCBBackgroundSubEngine:
    """Engine nhận diện và bóc tách bo mạch PCB bằng phương pháp trừ nền (Background Subtraction)"""

    def __init__(self):
        self.bg_img: Optional[np.ndarray] = None
        self.bg_gray: Optional[np.ndarray] = None
        self.bg_size: Tuple[int, int] = (0, 0)
        self.sub_mode = "static"  # "static" (trừ nền tĩnh absdiff) hoặc "dynamic" (MOG2)
        self.threshold_val = 30   # Ngưỡng phát hiện chênh lệch (15-60)
        self.min_area_ratio = 0.015
        self.max_area_ratio = 0.95

        # Cấu hình màu nền băng tải & màu bo mạch PCB
        self.color_mode_enabled = True
        self.conveyor_color = [25, 25, 25]  # BGR
        self.conveyor_name = "Băng tải Đen"
        self.pcb_color = [45, 135, 50]      # BGR
        self.pcb_name = "PCB Xanh lá"
        self.color_tolerance = 28
        self.use_otsu = False  # Chế độ tự động tìm ngưỡng Otsu cho camera mono

        # Bộ trừ nền động MOG2
        self.subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=25, detectShadows=False)

        self.load_background()
        self.load_color_config()

    def load_color_config(self) -> bool:
        """Tải cấu hình màu nền băng tải và màu bo mạch PCB đã lưu"""
        if COLOR_CONFIG_FILE.exists():
            try:
                with open(COLOR_CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.color_mode_enabled = bool(data.get("enabled", True))
                self.conveyor_color = list(data.get("conveyor_color", [25, 25, 25]))
                self.conveyor_name = str(data.get("conveyor_name", "Băng tải Đen"))
                self.pcb_color = list(data.get("pcb_color", [45, 135, 50]))
                self.pcb_name = str(data.get("pcb_name", "PCB Xanh lá"))
                self.color_tolerance = int(data.get("color_tolerance", 28))

                # Tự động tính ngưỡng tối ưu cho Camera Mono
                bg_gray = float(np.mean(self.conveyor_color))
                pcb_gray = float(np.mean(self.pcb_color))
                delta = abs(pcb_gray - bg_gray)
                if delta >= 4:
                    self.threshold_val = max(4, min(60, int(round(delta * 0.45))))
                    print(f"[Mono Engine] Nền={bg_gray:.0f}, PCB={pcb_gray:.0f} (Chênh lệch: {delta:.0f}) -> Tự động đặt ngưỡng = {self.threshold_val}")

                return True
            except Exception:
                pass
        return False

    def save_color_config(self) -> bool:
        """Lưu cấu hình màu nền băng tải và màu bo mạch PCB"""
        try:
            data = {
                "enabled": self.color_mode_enabled,
                "conveyor_color": self.conveyor_color,
                "conveyor_name": self.conveyor_name,
                "pcb_color": self.pcb_color,
                "pcb_name": self.pcb_name,
                "color_tolerance": self.color_tolerance,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(COLOR_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[Cảnh báo] Không thể lưu cấu hình màu: {e}")
            return False

    def apply_color_preset(self, preset_id: int) -> bool:
        """Áp dụng bộ màu mẫu công nghiệp chuẩn (1-6)"""
        for p in COLOR_PRESETS:
            if p["id"] == preset_id:
                self.conveyor_color = list(p["conveyor"])
                self.conveyor_name = p["conveyor_name"]
                self.pcb_color = list(p["pcb"])
                self.pcb_name = p["pcb_name"]
                self.color_mode_enabled = True
                self.save_color_config()
                print(f"[Màu sắc] Đã chọn bộ màu #{preset_id}: {self.conveyor_name} + {self.pcb_name}")
                return True
        return False

    def capture_background(self, full_frame: np.ndarray):
        """Lưu khung hình hiện tại làm ảnh nền băng tải rỗng chuẩn"""
        if full_frame is None or full_frame.size == 0:
            return
        self.bg_img = full_frame.copy()
        if len(full_frame.shape) == 3:
            self.bg_gray = cv2.cvtColor(self.bg_img, cv2.COLOR_BGR2GRAY)
        else:
            self.bg_gray = self.bg_img.copy()

        h, w = full_frame.shape[:2]
        self.bg_size = (w, h)

        try:
            cv2.imwrite(str(BG_IMG_FILE), self.bg_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
            data = {
                "size": [int(w), int(h)],
                "threshold": self.threshold_val,
                "sub_mode": self.sub_mode,
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(BG_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[THÀNH CÔNG] Đã chụp và lưu ảnh nền băng tải: {w}x{h} px")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể lưu file nền: {e}")

    def load_background(self) -> bool:
        """Tải lại ảnh nền băng tải đã lưu trước đó"""
        if not BG_IMG_FILE.exists():
            return False
        try:
            img = cv2.imread(str(BG_IMG_FILE))
            if img is not None and img.size > 0:
                self.bg_img = img
                if len(img.shape) == 3:
                    self.bg_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    self.bg_gray = img.copy()
                self.bg_size = (img.shape[1], img.shape[0])

                if BG_CONFIG_FILE.exists():
                    try:
                        with open(BG_CONFIG_FILE, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        self.threshold_val = cfg.get("threshold", 30)
                        self.sub_mode = cfg.get("sub_mode", "static")
                    except Exception:
                        pass

                print(f"[NỀN BĂNG TẢI] Đã nạp ảnh nền sẵn có: {self.bg_size[0]}x{self.bg_size[1]} px")
                return True
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khi nạp ảnh nền cũ: {e}")
        return False

    def reset_background(self):
        """Xóa ảnh nền hiện tại để chụp lại"""
        self.bg_img = None
        self.bg_gray = None
        self.bg_size = (0, 0)
        try:
            if BG_IMG_FILE.exists():
                BG_IMG_FILE.unlink()
            if BG_CONFIG_FILE.exists():
                BG_CONFIG_FILE.unlink()
        except Exception:
            pass
        print("[NỀN BĂNG TẢI] Đã xóa ảnh nền cũ. Hãy bấm [SPACE] hoặc [K] khi băng tải rỗng để chụp nền mới!")

    @staticmethod
    def order_corners(pts: np.ndarray) -> np.ndarray:
        """Sắp xếp 4 đỉnh theo thứ tự chuẩn: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]"""
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]       # Top-Left
        rect[2] = pts[np.argmax(s)]       # Bottom-Right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]    # Top-Right
        rect[3] = pts[np.argmax(diff)]    # Bottom-Left
        return rect

    def process(
        self,
        full_frame: np.ndarray,
        conveyor_roi: Optional[Tuple[int, int, int, int]] = None,
        return_mask: bool = False
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Nhận diện và cắt nắn thẳng bo mạch PCB bằng phương pháp trừ nền:
        - full_frame: Khung hình gốc từ camera (12MP)
        - conveyor_roi: Vùng giới hạn lòng băng tải (loại bỏ mép cơ khí bên ngoài)
        """
        t_start = time.perf_counter()
        info: Dict[str, Any] = {
            "success": False,
            "status_msg": "Chưa có ảnh nền (Bấm [SPACE] khi băng tải rỗng để chụp nền)",
            "time_ms": 0.0,
            "corners": None,
            "angle": 0.0,
            "crop_size": None,
            "mode": self.sub_mode,
            "fg_mask": None
        }

        if full_frame is None or full_frame.size == 0:
            info["status_msg"] = "LỖI: Khung hình rỗng"
            return None, info

        fh, fw = full_frame.shape[:2]

        if conveyor_roi:
            cx1, cy1, cx2, cy2 = conveyor_roi
        else:
            cx1, cy1, cx2, cy2 = 0, 0, fw, fh

        cur_roi_img = full_frame[cy1:cy2, cx1:cx2]
        rh, rw = cur_roi_img.shape[:2]

        if rh < 50 or rw < 50:
            info["status_msg"] = "Vùng băng tải không hợp lệ"
            return None, info

        total_area = float(rh * rw)

        # Tối ưu tốc độ: Xử lý trừ nền trên ảnh thu nhỏ scale 0.5 (nhanh gấp 4 lần, chỉ ~40-60ms)
        scale = 0.5 if (rh > 1000 or rw > 1000) else 1.0
        if scale != 1.0:
            cur_small = cv2.resize(cur_roi_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            cur_small = cur_roi_img

        sh, sw = cur_small.shape[:2]
        small_total_area = float(sh * sw)

        # Chế độ 0: Tự động phân ngưỡng Otsu Mono trên lòng băng tải (nếu bật use_otsu)
        if self.use_otsu:
            if len(cur_small.shape) == 3:
                cur_gray = cv2.cvtColor(cur_small, cv2.COLOR_BGR2GRAY)
            else:
                cur_gray = cur_small
            otsu_val, fg_mask = cv2.threshold(cur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(self.conveyor_color) > np.mean(self.pcb_color):
                fg_mask = cv2.bitwise_not(fg_mask)
            info["otsu_val"] = otsu_val

        # Chế độ 1: Trừ nền tĩnh (Static Reference Background) hoặc Trừ màu (Color Subtraction)
        elif self.sub_mode == "static":
            if self.bg_img is not None and self.bg_img.shape[:2] == full_frame.shape[:2]:
                # Đã có ảnh nền rỗng thực tế khớp độ phân giải
                bg_roi_img = self.bg_img[cy1:cy2, cx1:cx2]
                if scale != 1.0:
                    bg_small = cv2.resize(bg_roi_img, (sw, sh), interpolation=cv2.INTER_AREA)
                else:
                    bg_small = bg_roi_img

                # Phép trừ nền đa kênh tuyệt đối (Color Absdiff)
                diff_bgr = cv2.absdiff(cur_small, bg_small)
                diff_max = np.max(diff_bgr, axis=2)
                _, fg_mask = cv2.threshold(diff_max, self.threshold_val, 255, cv2.THRESH_BINARY)
            else:
                # CHƯA CÓ ẢNH NỀN RỖNG -> TỰ ĐỘNG DÙNG MÀU NỀN BĂNG TẢI & PCB ĐÃ CHỌN!
                bg_synthetic = np.full_like(cur_small, self.conveyor_color, dtype=np.uint8)
                diff_bgr = cv2.absdiff(cur_small, bg_synthetic)
                diff_max = np.max(diff_bgr, axis=2)
                _, fg_mask = cv2.threshold(diff_max, self.threshold_val, 255, cv2.THRESH_BINARY)

        else:
            # Chế độ 2: Trừ nền động MOG2
            if len(cur_small.shape) == 3:
                cur_gray = cv2.cvtColor(cur_small, cv2.COLOR_BGR2GRAY)
            else:
                cur_gray = cur_small
            fg_mask = self.subtractor_mog2.apply(cur_gray)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Xử lý hình thái học (Morphological Filtering)
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k_open)

        k_close_dim = max(7, int(round(min(sh, sw) * 0.018)))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close_dim, k_close_dim))
        clean_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

        if return_mask:
            info["fg_mask"] = clean_mask

        # Tìm các đường bao vật thể trên mặt nạ trừ nền
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area_ratio * small_total_area <= area <= self.max_area_ratio * small_total_area):
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)

            rect = cv2.minAreaRect(hull)
            (rcx, rcy), (rw_rect, rh_rect), angle = rect
            short_side = min(rw_rect, rh_rect)
            long_side = max(rw_rect, rh_rect)

            if short_side < 10:
                continue

            rect_area = short_side * long_side
            if rect_area <= 0:
                continue

            rectangularity = hull_area / rect_area
            if rectangularity < 0.45:
                continue

            score = (rectangularity ** 2) * (area / small_total_area)
            candidates.append({
                "contour": cnt,
                "hull": hull,
                "rect": rect,
                "area": area,
                "score": score,
                "angle": angle,
                "long_side": long_side,
                "short_side": short_side
            })

        if not candidates:
            info["status_msg"] = "Băng tải rỗng (Nền sạch, chưa có PCB)"
            info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
            return None, info

        # Chọn ứng viên có điểm số tốt nhất
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        best_hull = best["hull"]
        best_rect = best["rect"]

        # Tìm 4 đỉnh góc của bo mạch
        peri = cv2.arcLength(best_hull, True)
        approx = cv2.approxPolyDP(best_hull, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners_sub = approx.reshape(4, 2).astype(np.float32)
        else:
            corners_sub = cv2.boxPoints(best_rect).astype(np.float32)

        # Ánh xạ 4 góc về hệ tọa độ ảnh gốc full 12MP
        corners = (corners_sub / scale)
        corners[:, 0] += cx1
        corners[:, 1] += cy1

        # Sắp xếp 4 đỉnh theo thứ tự chuẩn
        ordered_corners = self.order_corners(corners)
        (tl, tr, br, bl) = ordered_corners

        # Tính kích thước thực của bo mạch
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        crop_w = int(round(max(width_top, width_bottom)))

        height_right = np.linalg.norm(tr - br)
        height_left = np.linalg.norm(tl - bl)
        crop_h = int(round(max(height_right, height_left)))

        if crop_w < 20 or crop_h < 20:
            info["status_msg"] = "LỖI: Kích thước bo mạch quá nhỏ"
            info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
            return None, info

        # Nắn thẳng bo mạch (Perspective Warp)
        dst_pts = np.array([
            [0, 0],
            [crop_w - 1, 0],
            [crop_w - 1, crop_h - 1],
            [0, crop_h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
        cropped_pcb = cv2.warpPerspective(full_frame, M, (crop_w, crop_h), flags=cv2.INTER_LINEAR)

        # Chuẩn hóa hướng nằm ngang (chiều rộng > chiều cao)
        if cropped_pcb.shape[0] > cropped_pcb.shape[1]:
            cropped_pcb = cv2.rotate(cropped_pcb, cv2.ROTATE_90_CLOCKWISE)
            crop_w, crop_h = crop_h, crop_w

        # Tính góc xoay thực tế của bo mạch trên băng tải (-180°..+180°)
        dx = tr[0] - tl[0]
        dy = tr[1] - tl[1]
        rot_angle = float(np.degrees(np.arctan2(dy, dx)))

        info["success"] = True
        mode_str = "Trừ nền" if (self.bg_img is not None and self.bg_img.shape[:2] == full_frame.shape[:2]) else f"Trừ màu ({self.conveyor_name})"
        info["status_msg"] = f"Phát hiện PCB ({mode_str}: {crop_w}x{crop_h}px, Góc {rot_angle:.0f}°)"
        info["corners"] = corners
        info["angle"] = rot_angle
        info["crop_size"] = (crop_w, crop_h)
        info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
        return cropped_pcb, info


class HikBgSubViewer:
    """Giao diện xem camera và điều khiển thuật toán trừ nền thời gian thực"""

    def __init__(self, default_exposure_us=300000.0):
        self.default_exposure_us = float(default_exposure_us)
        self.current_exposure_us = float(default_exposure_us)
        self.exp_min = 34.0
        self.exp_max = 1999733.0

        # Quản lý nguồn camera
        self.current_source: Optional[CameraSource] = None
        self.available_sources: List[CameraSource] = []
        self.show_camera_menu: bool = False

        self.cam = None  # Hikvision MvCamera
        self.cap = None  # OpenCV VideoCapture (Webcam)
        self.file_img = None  # Ảnh tĩnh mẫu

        self.is_connected = False
        self.is_grabbing = False
        self.device_name = "Chưa kết nối"
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

        # Engine Trừ nền (Background Subtraction Engine)
        self.engine = PCBBackgroundSubEngine()
        self.auto_crop_enabled = True
        self.show_live_crop_window = True
        self.show_debug_mask = False

        self.cached_cropped = None
        self.cached_corners = None
        self.cached_status = "Chờ nền băng tải"
        self.is_detecting = False

        # Cấu hình băng tải
        self.conveyor_cfg = load_conveyor_config()
        self.filter_conveyor_enabled = bool(self.conveyor_cfg.get("filter_enabled", True))
        self.conveyor_zoom_view = False

        # Chế độ hiệu chuẩn / chọn màu (Color Selection & Eyedropper)
        self.show_color_menu: bool = False
        self.eyedropper_mode: Optional[str] = None  # None, "conveyor", hoặc "pcb"
        self.mouse_display_pos: Tuple[int, int] = (0, 0)
        self.last_conveyor_click_pt: Optional[Tuple[int, int]] = None
        self.last_pcb_click_pt: Optional[Tuple[int, int]] = None
        self.view_w: int = 1280
        self.view_h: int = 800

    def set_toast(self, msg, duration=2.5):
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def connect(self, source: Optional[CameraSource] = None) -> bool:
        """Khởi tạo và kết nối nguồn camera (Hikvision, Webcam USB hoặc File)"""
        if source is not None:
            self.current_source = source

        if self.current_source is None:
            saved_cfg = load_camera_config()
            if saved_cfg:
                self.current_source = CameraSource.from_dict(saved_cfg)
            else:
                sources = scan_all_camera_sources()
                if sources:
                    self.current_source = sources[0]
                else:
                    print("[LỖI] Không tìm thấy bất kỳ nguồn camera nào!")
                    return False

        src_type = self.current_source.source_type
        print(f"[1/4] Đang kết nối nguồn: {self.current_source.badge} | {self.current_source.name}...")

        # =====================================================================
        # 1. Camera công nghiệp Hikvision (GigE / USB3 Vision)
        # =====================================================================
        if src_type == "hikvision":
            if not HAS_HIK_SDK:
                print("[LỖI] Không tìm thấy thư viện Hikvision SDK (MvImport)!")
                return False

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

            dev_idx = 0
            try:
                dev_idx = int(self.current_source.source_id)
            except Exception:
                dev_idx = 0
            if dev_idx >= device_list.nDeviceNum:
                dev_idx = 0

            st_device = cast(device_list.pDeviceInfo[dev_idx], POINTER(MV_CC_DEVICE_INFO)).contents

            if st_device.nTLayerType == MV_GIGE_DEVICE:
                model_bytes = bytes(st_device.SpecialInfo.stGigEInfo.chModelName).split(b'\x00')[0]
                self.device_name = model_bytes.decode('ascii', errors='ignore').strip() or f"Hikvision GigE #{dev_idx}"
            elif st_device.nTLayerType == MV_USB_DEVICE:
                model_bytes = bytes(st_device.SpecialInfo.stUsb3VInfo.chModelName).split(b'\x00')[0]
                self.device_name = model_bytes.decode('ascii', errors='ignore').strip() or f"Hikvision USB #{dev_idx}"

            self.cam = MvCamera()
            ret = self.cam.MV_CC_CreateHandle(st_device)
            if ret != 0:
                print(f"[LỖI] MV_CC_CreateHandle thất bại: 0x{ret:x}")
                return False

            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                print(f"[LỖI] MV_CC_OpenDevice thất bại: 0x{ret:x} (Camera có thể đang bị MVS hoặc app khác chiếm dụng)")
                return False

            if st_device.nTLayerType == MV_GIGE_DEVICE:
                pkt_size = self.cam.MV_CC_GetOptimalPacketSize()
                if pkt_size > 0:
                    self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt_size)

            st_float = MVCC_FLOATVALUE()
            ret = self.cam.MV_CC_GetFloatValue("ExposureTime", st_float)
            if ret == 0:
                self.exp_min = float(st_float.fMin)
                self.exp_max = float(st_float.fMax)

            self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            self.set_exposure(self.default_exposure_us)

            w_val = MVCC_INTVALUE()
            h_val = MVCC_INTVALUE()
            self.cam.MV_CC_GetIntValue("Width", w_val)
            self.cam.MV_CC_GetIntValue("Height", h_val)
            self.resolution = (w_val.nCurValue, h_val.nCurValue)
            print(f"      -> Model: {self.device_name} | Phân giải: {self.resolution[0]} x {self.resolution[1]}")
            self.is_connected = True
            return True

        # =====================================================================
        # 2. Camera USB / Webcam ngoài qua OpenCV DirectShow
        # =====================================================================
        elif src_type == "opencv":
            cam_idx = 0
            try:
                cam_idx = int(self.current_source.source_id)
            except Exception:
                cam_idx = 0

            with SuppressStderr():
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(cam_idx)

            if not cap.isOpened():
                print(f"[LỖI] Không thể mở Webcam USB #{cam_idx}!")
                return False

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Thử cấu hình độ phân giải cao
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                print(f"[LỖI] Webcam USB #{cam_idx} không nhận được hình ảnh!")
                return False

            self.cap = cap
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
            self.resolution = (w, h)
            self.device_name = self.current_source.name
            with self.frame_lock:
                self.latest_frame = frame
            self.is_connected = True
            print(f"      -> {self.device_name} kết nối thành công ({w} x {h} px)")
            return True

        # =====================================================================
        # 3. File ảnh / video tĩnh mẫu
        # =====================================================================
        elif src_type == "file":
            file_path = Path(self.current_source.source_id)
            if not file_path.is_file():
                print(f"[LỖI] Không tìm thấy file mẫu: {file_path}")
                return False

            img = cv2.imread(str(file_path))
            if img is None:
                print(f"[LỖI] Không thể đọc file ảnh: {file_path}")
                return False

            self.file_img = img
            h, w = img.shape[:2]
            self.resolution = (w, h)
            self.device_name = self.current_source.name
            with self.frame_lock:
                self.latest_frame = img.copy()
            self.is_connected = True
            print(f"      -> Đã nạp file mẫu: {file_path.name} ({w} x {h} px)")
            return True

        return False

    def set_exposure(self, exposure_us: float) -> bool:
        clamped = max(self.exp_min, min(self.exp_max, float(exposure_us)))
        self.current_exposure_us = clamped
        ms = self.current_exposure_us / 1000.0

        if self.cam:
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", clamped)
            if ret == 0:
                print(f"[Camera] Đã cập nhật ExposureTime: {self.current_exposure_us:.0f} us ({ms:.1f} ms)")
                self.set_toast(f"Exposure: {self.current_exposure_us:.0f} us ({ms:.1f} ms)")
                return True
        elif self.cap:
            try:
                exp_prop = int(np.log2(max(0.0001, ms / 1000.0)))
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_prop)
            except Exception:
                pass
            self.set_toast(f"Exposure Webcam: {ms:.1f} ms")
            return True
        else:
            self.set_toast(f"Exposure: {ms:.1f} ms")
            return True
        return False

    def start_grabbing(self):
        if not self.is_connected:
            return False

        if self.current_source and self.current_source.source_type == "hikvision" and self.cam:
            ret = self.cam.MV_CC_StartGrabbing()
            if ret != 0:
                print(f"[LỖI] MV_CC_StartGrabbing thất bại: 0x{ret:x}")
                return False

        self.exit_event.clear()
        self.is_grabbing = True
        self.grab_thread = threading.Thread(target=self._grab_worker, daemon=True)
        self.grab_thread.start()
        print("[4/4] Bắt đầu luồng ảnh thành công!")
        return True

    def _grab_worker(self):
        src_type = self.current_source.source_type if self.current_source else "hikvision"
        last_fps_time = time.time()
        frames_this_sec = 0

        # Luồng 1: Hikvision Camera
        if src_type == "hikvision":
            st_out = MV_FRAME_OUT()
            memset(byref(st_out), 0, sizeof(st_out))
            bgr_buf = None
            last_bgr_size = 0

            while not self.exit_event.is_set():
                if not self.cam:
                    break
                ret = self.cam.MV_CC_GetImageBuffer(st_out, 2000)
                if ret == 0 and st_out.pBufAddr:
                    w = st_out.stFrameInfo.nWidth
                    h = st_out.stFrameInfo.nHeight
                    pixel_type = st_out.stFrameInfo.enPixelType
                    frame_len = st_out.stFrameInfo.nFrameLen

                    img_bgr = None
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

        # Luồng 2: USB Webcam / OpenCV VideoCapture
        elif src_type == "opencv":
            while not self.exit_event.is_set():
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.02)
                    continue

                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.resolution = (frame.shape[1], frame.shape[0])
                    self.frame_count += 1
                    frames_this_sec += 1
                    now = time.time()
                    if now - last_fps_time >= 1.0:
                        self.fps_camera = frames_this_sec / (now - last_fps_time)
                        frames_this_sec = 0
                        last_fps_time = now
                else:
                    time.sleep(0.01)

        # Luồng 3: File ảnh / video tĩnh mẫu
        elif src_type == "file":
            if self.file_img is not None:
                h, w = self.file_img.shape[:2]
                with self.frame_lock:
                    self.latest_frame = self.file_img.copy()
                    self.resolution = (w, h)
                self.fps_camera = 30.0

            while not self.exit_event.is_set():
                time.sleep(0.033)
                self.frame_count += 1

    def stop_grabbing(self):
        self.exit_event.set()
        if self.grab_thread and self.grab_thread.is_alive():
            self.grab_thread.join(timeout=1.2)
        if self.is_grabbing and self.cam:
            try:
                self.cam.MV_CC_StopGrabbing()
            except Exception:
                pass
        self.is_grabbing = False
        self.exit_event.clear()

    def disconnect(self):
        self.stop_grabbing()
        if self.is_connected and self.cam:
            try:
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            self.cam = None

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.file_img = None
        self.is_connected = False

    def stop(self):
        self.disconnect()
        if HAS_HIK_SDK:
            try:
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass

    def switch_camera_source(self, new_source: CameraSource) -> bool:
        """Chuyển đổi nguồn camera an toàn ngay trong khi ứng dụng đang chạy"""
        if new_source is None:
            return False

        print(f"\n[Camera] Đang chuyển đổi sang nguồn: {new_source.badge} | {new_source.name}...")
        self.disconnect()

        with self.frame_lock:
            self.latest_frame = None

        self.cached_cropped = None
        self.cached_corners = None
        self.is_detecting = False

        ok = self.connect(new_source)
        if not ok:
            print(f"[LỖI] Không thể kết nối tới {new_source.name}!")
            self.set_toast(f"Lỗi kết nối {new_source.name}!", 3.0)
            return False

        self.start_grabbing()
        save_camera_config(new_source)

        # Kiểm tra xem ảnh nền cũ có khớp kích thước với camera mới không
        if self.engine.bg_img is not None and self.engine.bg_img.shape[:2] != (self.resolution[1], self.resolution[0]):
            self.set_toast("Đã đổi Camera! Bấm [SPACE] để chụp lưu nền băng tải mới.", 3.5)
        else:
            self.set_toast(f"Đã chuyển sang: {new_source.name}", 2.5)

        return True

    def _draw_camera_osd(self, view: np.ndarray):
        """Vẽ menu OSD chọn camera đè lên màn hình hiển thị trực quan"""
        vh, vw = view.shape[:2]
        menu_w = min(840, vw - 40)
        num_items = min(8, len(self.available_sources))
        menu_h = 135 + num_items * 36
        mx = (vw - menu_w) // 2
        my = (vh - menu_h) // 2

        # Lớp nền mờ tối
        overlay = view.copy()
        cv2.rectangle(overlay, (mx, my), (mx + menu_w, my + menu_h), (16, 16, 22), -1)
        cv2.addWeighted(overlay, 0.92, view, 0.08, 0, view)
        cv2.rectangle(view, (mx, my), (mx + menu_w, my + menu_h), (0, 230, 118), 2)

        # Tiêu đề
        cv2.putText(view, "QUET & CHON NGUON CAMERA HE THONG", (mx + 20, my + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 230, 118), 2, cv2.LINE_AA)
        cv2.putText(view, "Bam phim so [1-9] de ket noi ngay nguon tuong ung:", (mx + 20, my + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

        # Đường kẻ ngang
        cv2.line(view, (mx + 15, my + 72), (mx + menu_w - 15, my + 72), (60, 60, 75), 1)

        # Danh sách thiết bị
        item_y = my + 102
        for idx in range(num_items):
            src = self.available_sources[idx]
            is_active = src.matches(self.current_source)

            if is_active:
                cv2.rectangle(view, (mx + 15, item_y - 20), (mx + menu_w - 15, item_y + 8), (0, 95, 38), -1)
                cv2.rectangle(view, (mx + 15, item_y - 20), (mx + menu_w - 15, item_y + 8), (0, 230, 118), 1)
                txt_col = (255, 255, 255)
                mark_str = "<-- DANG KET NOI"
            else:
                txt_col = (220, 220, 220)
                mark_str = ""

            line_str = f"[{idx + 1}]  {src.badge:<16} |  {src.name:<24} |  {src.details}"
            cv2.putText(view, line_str, (mx + 25, item_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_col, 1, cv2.LINE_AA)
            if mark_str:
                cv2.putText(view, mark_str, (mx + menu_w - 195, item_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 128), 2, cv2.LINE_AA)

            item_y += 36

        # Footer hướng dẫn
        cv2.line(view, (mx + 15, my + menu_h - 42), (mx + menu_w - 15, my + menu_h - 42), (60, 60, 75), 1)
        cv2.putText(view, "[1-9]: Chon | [R]: Quet lai | [G]: Hop thoai GUI Tkinter | [C]/[ESC]: Dong",
                    (mx + 20, my + menu_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 50), 1, cv2.LINE_AA)

    @staticmethod
    def get_color_name(bgr: list) -> str:
        """Tự động phân tích và trả về tên màu trực quan từ mã BGR"""
        b, g, r = [int(v) for v in bgr[:3]]
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        diff = max_c - min_c

        if diff < 22:
            if max_c < 55:
                return f"Băng tải Đen ({b},{g},{r})"
            elif max_c > 195:
                return f"Trắng Sáng ({b},{g},{r})"
            else:
                return f"Xám ({b},{g},{r})"

        if g >= r and g >= b:
            if g > 1.25 * max(r, b):
                return f"PCB Xanh lá (FR4) ({b},{g},{r})"
            return f"Xanh lá ({b},{g},{r})"
        elif b >= r and b >= g:
            if b > 1.25 * max(r, g):
                return f"PCB Xanh dương ({b},{g},{r})"
            return f"Xanh dương ({b},{g},{r})"
        elif r >= g and r >= b:
            if g > 110 and b < 100:
                return f"PCB Vàng/Cam ({b},{g},{r})"
            return f"PCB Đỏ ({b},{g},{r})"
        return f"RGB({r},{g},{b})"

    def _sample_pixel_color(self, disp_x: int, disp_y: int) -> Optional[List[int]]:
        """Lấy mẫu màu BGR trung bình 9x9 xung quanh tọa độ click trên khung hình camera thực tế"""
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
        if frame is None or frame.size == 0:
            return None

        fh, fw = frame.shape[:2]
        vw = max(1, getattr(self, "view_w", 1280))
        vh = max(1, getattr(self, "view_h", 800))

        rx = int(disp_x * (fw / vw))
        ry = int(disp_y * (fh / vh))
        rx = max(0, min(fw - 1, rx))
        ry = max(0, min(fh - 1, ry))

        x0 = max(0, rx - 4)
        x1 = min(fw, rx + 5)
        y0 = max(0, ry - 4)
        y1 = min(fh, ry + 5)
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return None

        mean_bgr = np.mean(patch, axis=(0, 1)).astype(int).tolist()
        return [int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])]

    def on_mouse(self, event, x, y, flags, param):
        """Xử lý sự kiện chuột: di chuyển tâm ngắm và click hút màu trực tiếp"""
        self.mouse_display_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.eyedropper_mode == "conveyor":
                bgr = self._sample_pixel_color(x, y)
                if bgr is not None:
                    self.engine.conveyor_color = bgr
                    self.engine.conveyor_name = self.get_color_name(bgr)
                    self.last_conveyor_click_pt = (x, y)
                    self.eyedropper_mode = "pcb"
                    self.set_toast(f"Đã chọn Màu Băng Tải: {self.engine.conveyor_name}! Bước 2: Hãy click vào Bo Mạch PCB.", 3.5)
                    print(f"[Màu sắc] Đã chọn Màu Nền Băng Tải: BGR={bgr} ({self.engine.conveyor_name})")
            elif self.eyedropper_mode == "pcb":
                bgr = self._sample_pixel_color(x, y)
                if bgr is not None:
                    self.engine.pcb_color = bgr
                    self.engine.pcb_name = self.get_color_name(bgr)
                    self.last_pcb_click_pt = (x, y)

                    # Tự động tính ngưỡng tối ưu cho camera Mono / độ tương phản thấp
                    bg_gray = float(np.mean(self.engine.conveyor_color))
                    pcb_gray = float(np.mean(self.engine.pcb_color))
                    delta = abs(pcb_gray - bg_gray)
                    if delta >= 4:
                        optimal_thresh = max(4, min(60, int(round(delta * 0.45))))
                        self.engine.threshold_val = optimal_thresh
                        try:
                            cv2.setTrackbarPos("Nguong (Thresh)", "HIKVISION CAMERA - PHUONG PHAP TRU NEN (BACKGROUND SUBTRACTION)", optimal_thresh)
                        except Exception:
                            pass
                        print(f"[Mono Auto-Threshold] Nền={bg_gray:.0f}, PCB={pcb_gray:.0f} (Chênh lệch: {delta:.0f}) -> Tự động đặt Ngưỡng = {optimal_thresh}")

                    self.engine.save_color_config()
                    self.eyedropper_mode = None
                    self.show_color_menu = False
                    self.set_toast(f"HOÀN TẤT! Nền: {self.engine.conveyor_name} | PCB: {self.engine.pcb_name} | Ngưỡng: {self.engine.threshold_val}", 4.0)
                    print(f"[Màu sắc] Đã chọn Màu Bo Mạch PCB: BGR={bgr} ({self.engine.pcb_name}) -> Đã lưu cấu hình!")

    def _draw_eyedropper_overlay(self, view: np.ndarray):
        """Vẽ banner hướng dẫn hút màu, tâm ngắm crosshair và ô màu xem trước theo con trỏ chuột"""
        vh, vw = view.shape[:2]
        mx, my = self.mouse_display_pos
        mx = max(0, min(vw - 1, mx))
        my = max(0, min(vh - 1, my))

        hover_bgr = [int(c) for c in view[my, mx]]
        hover_rgb = [hover_bgr[2], hover_bgr[1], hover_bgr[0]]

        # 1. Banner trên cùng
        banner_h = 56
        banner_bg = view[:banner_h, :].copy()
        if self.eyedropper_mode == "conveyor":
            cv2.rectangle(banner_bg, (0, 0), (vw, banner_h), (20, 90, 180), -1)
            cv2.addWeighted(banner_bg, 0.88, view[:banner_h, :], 0.12, 0, view[:banner_h, :])
            cv2.line(view, (0, banner_h), (vw, banner_h), (0, 165, 255), 2)
            cv2.putText(view, "[BUOC 1/2]: HAY CLICK CHUOT VAO NEN BANG TAI TREN MAN HINH",
                        (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, "Phim [1-6]: Chon mau co san | [T]: Bang mau Tkinter | [P]/[ESC]: Huy",
                        (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 230, 150), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(banner_bg, (0, 0), (vw, banner_h), (25, 120, 45), -1)
            cv2.addWeighted(banner_bg, 0.88, view[:banner_h, :], 0.12, 0, view[:banner_h, :])
            cv2.line(view, (0, banner_h), (vw, banner_h), (0, 255, 128), 2)
            cv2.putText(view, "[BUOC 2/2]: TUYET VOI! BAY GIO HAY CLICK VAO THAN BO MACH PCB",
                        (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, f"Da luu Bang tai: {self.engine.conveyor_name} | Phim [1-6]: Mau co san | [ESC]: Huy",
                        (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA)

        # Ô xem trước màu con trỏ chuột ở góc banner
        bw_box = 210
        bx = vw - bw_box - 15
        cv2.rectangle(view, (bx, 8), (bx + 38, 46), hover_bgr, -1)
        cv2.rectangle(view, (bx, 8), (bx + 38, 46), (255, 255, 255), 1)
        cv2.putText(view, "Con tro:", (bx + 46, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        cv2.putText(view, f"RGB({hover_rgb[0]},{hover_rgb[1]},{hover_rgb[2]})",
                    (bx + 46, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

        # 2. Vòng tròn & Tâm ngắm Crosshair theo con trỏ chuột
        if my > banner_h + 10:
            target_col = (0, 200, 255) if self.eyedropper_mode == "conveyor" else (0, 255, 128)
            cv2.circle(view, (mx, my), 16, target_col, 2)
            cv2.circle(view, (mx, my), 2, (0, 0, 255), -1)
            cv2.line(view, (mx - 24, my), (mx - 8, my), target_col, 2)
            cv2.line(view, (mx + 8, my), (mx + 24, my), target_col, 2)
            cv2.line(view, (mx, my - 24), (mx, my - 8), target_col, 2)
            cv2.line(view, (mx, my + 8), (mx, my + 24), target_col, 2)

            px = mx + 20
            py = my - 20
            if px + 40 < vw and py - 10 > 0:
                cv2.rectangle(view, (px, py), (px + 24, py + 24), hover_bgr, -1)
                cv2.rectangle(view, (px, py), (px + 24, py + 24), (255, 255, 255), 1)

        # 3. Điểm đánh dấu đã click nền băng tải (nếu có)
        if self.last_conveyor_click_pt is not None and self.eyedropper_mode == "pcb":
            cx, cy = self.last_conveyor_click_pt
            if 0 <= cx < vw and 0 <= cy < vh:
                cv2.circle(view, (cx, cy), 8, (0, 255, 0), -1)
                cv2.circle(view, (cx, cy), 12, (255, 255, 255), 2)
                cv2.putText(view, "Nen Bang Tai", (cx + 14, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    def _draw_color_osd(self, view: np.ndarray):
        """Vẽ bảng OSD chọn màu nền băng tải & màu bo mạch PCB"""
        vh, vw = view.shape[:2]
        menu_w, menu_h = 560, 390
        mx = (vw - menu_w) // 2
        my = (vh - menu_h) // 2

        overlay = view.copy()
        cv2.rectangle(overlay, (mx, my), (mx + menu_w, my + menu_h), (20, 20, 26), -1)
        cv2.addWeighted(overlay, 0.90, view, 0.10, 0, view)
        cv2.rectangle(view, (mx, my), (mx + menu_w, my + menu_h), (0, 230, 118), 2)

        # Header
        cv2.rectangle(view, (mx, my), (mx + menu_w, my + 44), (32, 32, 42), -1)
        cv2.putText(view, "CHON MAU NEN BANG TAI & BO MACH PCB",
                    (mx + 20, my + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 128), 2, cv2.LINE_AA)

        # Trạng thái hiện tại
        cv2.putText(view, "Cau hinh mau dang su dung:", (mx + 20, my + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        # Swatch Nền băng tải
        cv2.rectangle(view, (mx + 25, my + 84), (mx + 55, my + 108), self.engine.conveyor_color, -1)
        cv2.rectangle(view, (mx + 25, my + 84), (mx + 55, my + 108), (200, 200, 200), 1)
        cv2.putText(view, f"Nen bang tai: {self.engine.conveyor_name} | BGR: {self.engine.conveyor_color}",
                    (mx + 65, my + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        # Swatch Bo mạch PCB
        cv2.rectangle(view, (mx + 25, my + 116), (mx + 55, my + 140), self.engine.pcb_color, -1)
        cv2.rectangle(view, (mx + 25, my + 116), (mx + 55, my + 140), (200, 200, 200), 1)
        cv2.putText(view, f"Than bo PCB:  {self.engine.pcb_name} | BGR: {self.engine.pcb_color}",
                    (mx + 65, my + 134), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        # Đường kẻ phân cách
        cv2.line(view, (mx + 20, my + 152), (mx + menu_w - 20, my + 152), (60, 60, 75), 1)

        # Danh sách 6 Presets
        cv2.putText(view, "Phim [1-6]: Chon nhanh bo mau cong nghiep:",
                    (mx + 20, my + 174), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)

        preset_y = my + 198
        for p in COLOR_PRESETS:
            pid = p["id"]
            badge = p.get("badge", p["name"])
            cv2.rectangle(view, (mx + 25, preset_y - 12), (mx + 45, preset_y + 4), p["conveyor"], -1)
            cv2.rectangle(view, (mx + 48, preset_y - 12), (mx + 68, preset_y + 4), p["pcb"], -1)
            cv2.putText(view, f"[{pid}] {badge}", (mx + 80, preset_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            preset_y += 24

        # Footer hướng dẫn
        cv2.line(view, (mx + 15, my + menu_h - 48), (mx + menu_w - 15, my + menu_h - 48), (60, 60, 75), 1)
        cv2.putText(view, "[Click]/[E]: Hut mau tren video | [T]: Bang mau Tkinter | [P]/[ESC]: Dong",
                    (mx + 20, my + menu_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 50), 1, cv2.LINE_AA)

    def open_color_dialog(self):
        """Mở hộp thoại bảng màu Tkinter GUI để chọn màu nền và PCB"""
        show_color_picker_dialog(
            self.engine.conveyor_color,
            self.engine.pcb_color,
            COLOR_PRESETS,
            on_apply_callback=self._on_color_dialog_applied
        )

    def _on_color_dialog_applied(self, conv_bgr: list, pcb_bgr: list):
        self.engine.conveyor_color = list(conv_bgr)
        self.engine.conveyor_name = self.get_color_name(conv_bgr)
        self.engine.pcb_color = list(pcb_bgr)
        self.engine.pcb_name = self.get_color_name(pcb_bgr)
        self.engine.save_color_config()
        self.show_color_menu = False
        self.eyedropper_mode = None
        self.set_toast(f"Đã cập nhật: {self.engine.conveyor_name} + {self.engine.pcb_name}", 3.0)

    def get_conveyor_roi(self, img_input: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = img_input.shape[:2]
        cfg = getattr(self, "conveyor_cfg", {})
        m_left = float(cfg.get("margin_left", 0.28))
        m_right = float(cfg.get("margin_right", 0.06))
        x1 = max(0, min(int(w * m_left), w - 20))
        x2 = max(x1 + 20, min(int(w * (1.0 - m_right)), w))
        return x1, 0, x2, h

    def run_gui(self):
        window_name = "HIKVISION CAMERA - PHUONG PHAP TRU NEN (BACKGROUND SUBTRACTION)"
        crop_window_name = "PCB DA CAT (TRU NEN LIVE CROP)"
        mask_window_name = "MAT NA TRU NEN (DEBUG MASK)"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 800)
        cv2.setMouseCallback(window_name, self.on_mouse)

        if self.show_live_crop_window:
            cv2.namedWindow(crop_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(crop_window_name, 640, 420)

        # Trackbar điều chỉnh Exposure
        def on_exp_trackbar(val):
            exp_us = max(self.exp_min, float(val) * 1000.0)
            self.set_exposure(exp_us)

        init_exp_ms = int(self.current_exposure_us / 1000.0)
        cv2.createTrackbar("Exposure (ms)", window_name, init_exp_ms, 1000, on_exp_trackbar)

        # Trackbar điều chỉnh ngưỡng trừ nền (Threshold)
        def on_thresh_trackbar(val):
            self.engine.threshold_val = max(5, int(val))

        cv2.createTrackbar("Nguong (Thresh)", window_name, self.engine.threshold_val, 100, on_thresh_trackbar)

        print("\n" + "=" * 65)
        print(" HƯỚNG DẪN ĐIỀU KHIỂN BẢN TRỪ NỀN (BACKGROUND SUBTRACTION):")
        print(" - Phím [P]          : CHỌN MÀU NỀN BĂNG TẢI & PCB (Hút trực tiếp / Mẫu 1-6 / Bảng màu)")
        print(" - Phím [O]          : Bật / Tắt TỰ ĐỘNG PHÂN NGƯỠNG OTSU CHO CAMERA MONO")
        print(" - Phím [C]          : QUÉT & ĐỔI NGUỒN CAMERA (Hikvision / USB Webcam / File)")
        print(" - Phím [SPACE] / [K]: CHỤP ẢNH NỀN BĂNG TẢI RỖNG (Khi chưa có bo mạch)")
        print(" - Phím [R]          : Xóa ảnh nền cũ / Chụp lại nền mới")
        print(" - Phím [M]          : Đổi chế độ: Trừ nền tĩnh (Absdiff) <-> Trừ nền động (MOG2)")
        print(" - Phím [D]          : Bật / Tắt cửa sổ xem mặt nạ nhị phân trừ nền (Debug Mask)")
        print(" - Phím [W]          : Ẩn / Hiện cửa sổ Live Crop phụ")
        print(" - Phím [B]          : Bật / Tắt lọc bỏ lề ngoài băng tải")
        print(" - Phím [S]          : Chụp ảnh full & lưu bo mạch đã cắt nắn thẳng")
        print(" - Phím [+] / [-]    : Tăng / Giảm thời gian phơi sáng")
        print(" - Phím [Q] / [ESC]  : Thoát chương trình")
        print("=" * 65 + "\n")

        latest_debug_mask = None

        while not self.exit_event.is_set():
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is not None:
                h, w = frame.shape[:2]

                # Nếu chưa có ảnh nền và là frame đầu tiên, hiển thị nhắc người dùng chụp nền
                if self.engine.bg_img is None and self.frame_count == 1:
                    self.set_toast("Bấm [SPACE] khi băng tải rỗng để lưu ảnh nền!", 4.0)

                # Nhận diện trừ nền bất đồng bộ
                if self.auto_crop_enabled and not self.is_detecting:
                    self.is_detecting = True
                    conv_roi = self.get_conveyor_roi(frame) if self.filter_conveyor_enabled else None
                    frame_for_sub = frame.copy()
                    need_mask = self.show_debug_mask

                    def _sub_worker(f, r, nm):
                        nonlocal latest_debug_mask
                        try:
                            cropped, info = self.engine.process(f, conveyor_roi=r, return_mask=nm)
                            if nm and info.get("fg_mask") is not None:
                                latest_debug_mask = info["fg_mask"].copy()

                            if info.get("success") and cropped is not None:
                                self.cached_cropped = cropped
                                self.cached_corners = info.get("corners")
                                self.cached_status = info.get("status_msg", "Đã bắt dính PCB")
                            else:
                                self.cached_cropped = None
                                self.cached_corners = None
                                self.cached_status = info.get("status_msg", "Băng tải rỗng")
                        except Exception:
                            pass
                        finally:
                            self.is_detecting = False

                    threading.Thread(target=_sub_worker, args=(frame_for_sub, conv_roi, need_mask), daemon=True).start()

                cur_cropped = self.cached_cropped
                cur_corners = self.cached_corners
                cur_found = bool(cur_cropped is not None and cur_corners is not None)

                # Render màn hình chính
                target_w = 1280
                target_h = int(h * (target_w / w))
                self.view_w = target_w
                self.view_h = target_h
                view = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                scale_x = target_w / w
                scale_y = target_h / h

                # Vẽ vùng băng tải (nếu bật)
                if self.filter_conveyor_enabled:
                    cx1, cy1, cx2, cy2 = self.get_conveyor_roi(frame)
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

                # Vẽ khung định vị bo mạch PCB tìm thấy bằng phương pháp trừ nền
                if cur_found and cur_corners is not None:
                    corners_disp = cur_corners.copy()
                    corners_disp[:, 0] *= scale_x
                    corners_disp[:, 1] *= scale_y
                    pts_int = np.int32(corners_disp)
                    cv2.polylines(view, [pts_int], isClosed=True, color=(0, 255, 128), thickness=3)
                    for pt in pts_int:
                        cv2.circle(view, tuple(pt), 6, (0, 200, 255), -1)

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
                        cv2.putText(view, f"PCB TRU NEN: {cw}x{ch} px", (pip_x + 8, pip_y + 17),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA)
                else:
                    if pip_y + pip_h <= target_h and pip_x + pip_w <= target_w:
                        cv2.rectangle(view, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (25, 25, 25), -1)
                        cv2.rectangle(view, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (70, 70, 70), 1)
                        msg_pip = "BAM [SPACE] CHUP NEN BANG TAI" if self.engine.bg_img is None else "BANG TAI RONG..."
                        cv2.putText(view, msg_pip, (pip_x + 25, pip_y + 115),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)

                # Vẽ HUD
                self._draw_hud(view, w, h)

                # Menu OSD Camera (nếu đang bật)
                if self.show_camera_menu:
                    self._draw_camera_osd(view)
                elif self.eyedropper_mode is not None:
                    self._draw_eyedropper_overlay(view)
                elif self.show_color_menu:
                    self._draw_color_osd(view)

                cv2.imshow(window_name, view)

                # Cửa sổ Live Crop phóng to bo mạch
                if self.show_live_crop_window:
                    if cur_found and cur_cropped is not None and getattr(cur_cropped, 'size', 0) > 0 and cur_cropped.shape[0] > 10 and cur_cropped.shape[1] > 10:
                        crop_win_disp = cur_cropped.copy()
                        ch, cw = crop_win_disp.shape[:2]
                        header_bar = np.zeros((45, cw, 3), dtype=np.uint8)
                        cv2.putText(header_bar, f"PCB DA CAT (TRU NEN) - Kich thuoc: {cw}x{ch} px", (15, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)
                        full_crop_view = np.vstack([header_bar, crop_win_disp])
                        disp_h = 420
                        disp_w = int(full_crop_view.shape[1] * disp_h / max(1, full_crop_view.shape[0]))
                        cv2.imshow(crop_window_name, cv2.resize(full_crop_view, (disp_w, disp_h), interpolation=cv2.INTER_AREA))
                    else:
                        empty_win = np.zeros((420, 640, 3), dtype=np.uint8)
                        cv2.putText(empty_win, "BANG TAI RONG", (200, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2, cv2.LINE_AA)
                        cv2.putText(empty_win, "Dang cho bo mach PCB tren bang tai...", (115, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)
                        cv2.imshow(crop_window_name, empty_win)

                # Cửa sổ Debug Mask (nếu bật)
                if self.show_debug_mask and latest_debug_mask is not None:
                    mask_disp = cv2.resize(latest_debug_mask, (640, int(640 * latest_debug_mask.shape[0] / latest_debug_mask.shape[1])))
                    cv2.imshow(mask_window_name, mask_disp)

            else:
                waiting = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(waiting, "Dang nhan hinh anh tu camera...", (420, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                cv2.imshow(window_name, waiting)

            # Phím bấm điều khiển
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                if self.show_camera_menu:
                    self.show_camera_menu = False
                elif self.show_color_menu or self.eyedropper_mode is not None:
                    self.show_color_menu = False
                    self.eyedropper_mode = None
                    self.set_toast("Đã tắt chế độ chọn màu", 1.5)
                else:
                    break
            elif key in (ord('p'), ord('P')):
                if self.show_color_menu or self.eyedropper_mode is not None:
                    self.show_color_menu = False
                    self.eyedropper_mode = None
                    self.set_toast("Đã tắt chế độ chọn màu", 1.5)
                else:
                    self.show_camera_menu = False
                    self.show_color_menu = False
                    self.eyedropper_mode = "conveyor"
                    self.set_toast("BƯỚC 1/2: HÃY CLICK CHUỘT VÀO NỀN BĂNG TẢI!", 4.0)
            elif key in (ord('c'), ord('C')):
                self.show_camera_menu = not self.show_camera_menu
                if self.show_camera_menu:
                    self.show_color_menu = False
                    self.eyedropper_mode = None
                    self.available_sources = scan_all_camera_sources()
            elif self.show_camera_menu:
                if ord('1') <= key <= ord('9'):
                    sel_idx = key - ord('1')
                    if 0 <= sel_idx < len(self.available_sources):
                        self.switch_camera_source(self.available_sources[sel_idx])
                        self.show_camera_menu = False
                elif key in (ord('r'), ord('R')):
                    self.available_sources = scan_all_camera_sources()
                    self.set_toast(f"Đã quét lại! Tìm thấy {len(self.available_sources)} nguồn.", 2.5)
                elif key in (ord('g'), ord('G')):
                    chosen = show_camera_picker_dialog(
                        self.available_sources,
                        current_source=self.current_source,
                        timeout_sec=0
                    )
                    if chosen:
                        self.switch_camera_source(chosen)
                    self.show_camera_menu = False
            elif self.show_color_menu or self.eyedropper_mode is not None:
                if ord('1') <= key <= ord('6'):
                    preset_idx = key - ord('0')
                    if self.engine.apply_color_preset(preset_idx):
                        self.set_toast(f"Đã chọn: {self.engine.conveyor_name} + {self.engine.pcb_name}", 3.0)
                        self.show_color_menu = False
                        self.eyedropper_mode = None
                elif key in (ord('e'), ord('E')):
                    self.eyedropper_mode = "conveyor"
                    self.set_toast("BƯỚC 1/2: HÃY CLICK CHUỘT VÀO NỀN BĂNG TẢI!", 3.5)
                elif key in (ord('t'), ord('T')):
                    self.open_color_dialog()
            elif key in (ord(' '), ord('k'), ord('K')):
                with self.frame_lock:
                    cur_f = self.latest_frame.copy() if self.latest_frame is not None else None
                if cur_f is not None:
                    self.engine.capture_background(cur_f)
                    self.set_toast("ĐÃ LƯU ẢNH NỀN BĂNG TẢI CHUẨN!", 3.0)
            elif key in (ord('r'), ord('R')):
                self.engine.reset_background()
                self.cached_cropped = None
                self.cached_corners = None
                self.set_toast("Đã xóa ảnh nền cũ. Bấm [SPACE] để chụp nền mới!", 3.0)
            elif key in (ord('m'), ord('M')):
                if self.engine.sub_mode == "static":
                    self.engine.sub_mode = "dynamic"
                    mode_name = "TRỪ NỀN ĐỘNG (MOG2)"
                else:
                    self.engine.sub_mode = "static"
                    mode_name = "TRỪ NỀN TĨNH (Absdiff Chuẩn)"
                self.set_toast(f"Chế độ: {mode_name}", 2.5)
            elif key in (ord('o'), ord('O')):
                self.engine.use_otsu = not self.engine.use_otsu
                st = "BẬT (Tự động thích ứng Otsu Mono)" if self.engine.use_otsu else f"TẮT (Dùng ngưỡng {self.engine.threshold_val})"
                self.set_toast(f"Tự động tính ngưỡng Otsu: {st}", 2.5)
            elif key in (ord('d'), ord('D')):
                self.show_debug_mask = not self.show_debug_mask
                if not self.show_debug_mask:
                    try:
                        cv2.destroyWindow(mask_window_name)
                    except Exception:
                        pass
                st = "BAT" if self.show_debug_mask else "TAT"
                self.set_toast(f"Xem mat na nhi phan (Debug Mask): {st}", 2.0)
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
        base_name = f"bgsub_capture_{ts}_exp_{exp_ms}ms"

        full_path = self.capture_dir / f"{base_name}_full.jpg"
        cv2.imwrite(str(full_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 98])

        cur_cropped = self.cached_cropped
        if cur_cropped is not None:
            crop_path = self.capture_dir / f"{base_name}_crop.jpg"
            cv2.imwrite(str(crop_path), cur_cropped, [cv2.IMWRITE_JPEG_QUALITY, 98])
            self.set_toast(f"Đã lưu ảnh full và bo mạch cắt: {crop_path.name}", 3.0)
            print(f"[ĐÃ LƯU] Ảnh full: {full_path.name} | Ảnh bo mạch: {crop_path.name}")
        else:
            self.set_toast(f"Đã lưu ảnh full: {full_path.name}", 2.5)

    def _draw_hud(self, view: np.ndarray, orig_w: int, orig_h: int):
        vh, vw = view.shape[:2]
        exp_ms = self.current_exposure_us / 1000.0

        overlay = view.copy()
        cv2.rectangle(overlay, (10, 10), (480, 195), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, view, 0.30, 0, view)
        cv2.rectangle(view, (10, 10), (480, 195), (0, 255, 128), 1)

        cv2.putText(view, "THUAT TOAN TRU NEN (BACKGROUND SUBTRACTION)", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 128), 2, cv2.LINE_AA)
        badge_str = self.current_source.badge if self.current_source else "Camera"
        cv2.putText(view, f"{badge_str}: {self.device_name} ({orig_w}x{orig_h} px)", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(view, f"Thoi gian sang: {self.current_exposure_us:,.0f} us ({exp_ms:.1f} ms) | FPS: {self.fps_camera:.1f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)

        if self.engine.use_otsu:
            mode_str = "Tu dong Otsu Mono [O]"
        elif self.engine.sub_mode == "static":
            mode_str = "Tru nen tinh (Absdiff Chuan)"
        else:
            mode_str = "Tru nen dong (MOG2)"
        cv2.putText(view, f"Che do [M/O]: {mode_str}", (20, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        bg_st = f"Da luu nen ({self.engine.bg_size[0]}x{self.engine.bg_size[1]}px)" if self.engine.bg_img is not None else "CHUA CO NEN (Bam [SPACE] de chup)"
        bg_col = (0, 255, 128) if self.engine.bg_img is not None else (0, 200, 255)
        cv2.putText(view, f"Anh nen: {bg_st}", (20, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bg_col, 1, cv2.LINE_AA)

        # Hiển thị màu nền băng tải & màu PCB với ô màu swatch trực quan
        cv2.putText(view, "Mau [P]: Nen", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(view, (120, 139), (138, 153), self.engine.conveyor_color, -1)
        cv2.rectangle(view, (120, 139), (138, 153), (200, 200, 200), 1)
        cv2.putText(view, "PCB", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(view, (185, 139), (203, 153), self.engine.pcb_color, -1)
        cv2.rectangle(view, (185, 139), (203, 153), (200, 200, 200), 1)
        cv2.putText(view, f"({self.engine.conveyor_name[:12]} + {self.engine.pcb_name[:12]})",
                    (215, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (180, 255, 180), 1, cv2.LINE_AA)

        cv2.putText(view, f"Nguong (Thresh): {self.engine.threshold_val} | {self.cached_status}", (20, 172),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 200, 50), 1, cv2.LINE_AA)

        cv2.putText(view, "[P]: Mau | [O]: Otsu Mono | [SPACE]: Chup Nen | [C]: Cam | [M]: Che do | [S]: Luu | [Q]: Thoat",
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

    file_src = CameraSource(
        source_type="file",
        source_id=str(path),
        name=f"[Ảnh mẫu] {path.name}",
        details=f"Kích thước: {img.shape[1]}x{img.shape[0]} px",
        resolution=(img.shape[1], img.shape[0]),
        badge="🖼️ Ảnh mẫu"
    )

    viewer = HikBgSubViewer()
    viewer.available_sources = scan_all_camera_sources()
    if viewer.connect(file_src):
        if viewer.start_grabbing():
            viewer.run_gui()
        viewer.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hikvision & USB Camera Viewer - Phương Pháp Trừ Nền")
    parser.add_argument("--scan", action="store_true", help="Quét và hiển thị toàn bộ camera khả dụng rồi thoát")
    parser.add_argument("--source", type=str, default=None, help="Chỉ định nguồn camera (ví dụ: 0, 1, hik:0, file:a.jpg)")
    parser.add_argument("--no-gui", action="store_true", help="Không mở hộp thoại GUI chọn camera, dùng Terminal")
    parser.add_argument("--auto", action="store_true", help="Tự động kết nối nguồn ưu tiên không cần hỏi")
    parser.add_argument("--exposure", type=float, default=300000.0, help="Thời gian lấy sáng mặc định (us)")
    args, unknown = parser.parse_known_args()

    # Kiểm tra nếu người dùng truyền file ảnh trực tiếp qua tham số dòng lệnh
    for u in unknown:
        if any(u.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.mp4')):
            args.source = f"file:{u}"
            break

    print("=" * 68)
    print("   HỆ THỐNG NHẬN DIỆN & CẮT PCB BẰNG PHƯƠNG PHÁP TRỪ NỀN")
    print("   (Industrial Background Subtraction & Multi-Camera Inspection)")
    print("=" * 68)

    # 1. Quét toàn bộ nguồn camera
    print("[1/3] Đang quét toàn bộ nguồn camera hệ thống...")
    available_sources = scan_all_camera_sources()
    print(f"[2/3] Tìm thấy {len(available_sources)} nguồn thiết bị khả dụng.")

    if args.scan:
        print("\nDanh sách camera quét được:")
        for idx, s in enumerate(available_sources, 1):
            print(f"  [{idx}] {s.badge:<18} | {s.name:<25} | {s.details}")
        sys.exit(0)

    selected_source = None

    # 2. Nếu người dùng chỉ định qua tham số dòng lệnh --source
    if args.source:
        selected_source = parse_camera_source_arg(args.source, available_sources)

    # 3. Nếu chưa có, kiểm tra nguồn đã lưu trong file cấu hình camera_source_config.json
    if selected_source is None:
        saved_cfg = load_camera_config()
        if saved_cfg:
            saved_src = CameraSource.from_dict(saved_cfg)
            for s in available_sources:
                if s.matches(saved_src):
                    selected_source = s
                    break

    # 4. Nếu có nhiều nguồn và người dùng không bật cờ --auto, mở giao diện chọn trực quan
    if (selected_source is None or not args.auto) and not args.source and len(available_sources) > 1:
        print("[3/3] Hiển thị bảng chọn nguồn camera...")
        if args.no_gui:
            selected_source = terminal_camera_picker(
                available_sources,
                current_source=selected_source,
                timeout_sec=5
            )
        else:
            selected_source = show_camera_picker_dialog(
                available_sources,
                current_source=selected_source,
                timeout_sec=5
            )

    # 5. Fallback nếu vẫn chưa chọn
    if selected_source is None and available_sources:
        selected_source = available_sources[0]

    if selected_source is None:
        print("[LỖI] Không tìm thấy bất kỳ nguồn camera hay file mẫu nào để khởi chạy!")
        sys.exit(1)

    print(f"\n[KẾT NỐI] Đang khởi động: {selected_source.badge} | {selected_source.name}")
    save_camera_config(selected_source)

    viewer = HikBgSubViewer(default_exposure_us=args.exposure)
    viewer.available_sources = available_sources
    if viewer.connect(selected_source):
        if viewer.start_grabbing():
            viewer.run_gui()
        viewer.stop()
    else:
        print("[LỖI] Không thể kết nối tới nguồn camera đã chọn!")
        sys.exit(1)

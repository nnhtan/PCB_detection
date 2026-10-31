# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH DEBUG THUẬT TOÁN CẮT & NẮN THẲNG BO MẠCH PCB (SMART DESKEW VISUAL DEBUGGER)
========================================================================================
Tính năng chuyên sâu phục vụ tinh chỉnh & kiểm tra thuật toán:
1. Màn hình 4 ô trực quan (2x2 Grid View):
   - Ô 1: Ảnh gốc + Khung nhận diện 4 góc OBB bám dính + Vùng đo độ sáng băng tải.
   - Ô 2: Ảnh nhị phân tách nền (Adaptive Binary Threshold Mask).
   - Ô 3: Mặt nạ đặc bo mạch sau khi lấp đầy tụ nhôm & lỗ khoan (Hole-Filled Mask).
   - Ô 4: Bo mạch đã được nắn thẳng 90° nằm ngang (Deskewed PCB).
2. Cửa sổ phóng to chất lượng cao độc lập cho bo mạch đã cắt (Zoom View).
3. Thanh trượt tương tác thời gian thực (Interactive OpenCV Trackbars):
   - Chọn chế độ: Tự động (0) / Băng tải Trắng (1) / Băng tải Đen (2).
   - Tinh chỉnh ngưỡng nhị phân (Threshold Factor).
   - Tinh chỉnh bán kính lấp đầy thân bo mạch (Morph Kernel Size).
   - Tinh chỉnh diện tích tối thiểu (Min Area Ratio).
   - Tinh chỉnh thời gian phơi sáng camera Hikvision (Exposure Time).
4. Hỗ trợ 2 nguồn linh hoạt:
   - Nguồn 1: Camera công nghiệp Hikvision GigE/USB (MV-CE120-10GM) trực tiếp.
   - Nguồn 2: Duyệt nhanh danh sách file ảnh (ảnh người dùng gửi, captures/, *.jpg, *.png, *.bmp).
5. Phím tắt tiện lợi:
   - [S] / [SPACE]: Lưu toàn bộ bộ ảnh debug (ảnh gốc, ảnh nhị phân, mặt nạ, PCB nắn thẳng).
   - [M]          : Đổi nhanh chế độ băng tải (Tự động <-> Sáng <-> Tối).
   - [N] / [P]    : Chuyển sang ảnh Tiếp theo / Trước đó (khi xem file).
   - [W]          : Bật / Tắt cửa sổ phụ phóng to bo mạch.
   - [R]          : Reset các thông số về mặc định.
   - [Q] / [ESC]  : Thoát chương trình.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Cấu hình UTF-8 cho console Windows
if sys.platform.startswith('win'):
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np

# Nạp SDK Hikvision nếu có
try:
    from camera_viewer import HikCameraViewer
    HAS_HIK_SDK = True
except Exception:
    HAS_HIK_SDK = False

# Nạp Cropper AI
try:
    from pcb_crop_engine import PCBMonoCropper
    HAS_CROPPER = True
except Exception:
    HAS_CROPPER = False


class PCBCropDebugger:
    """Công cụ Visual Debugger cho thuật toán cắt và nắn thẳng PCB"""

    def __init__(self, target_path: Optional[str] = None):
        self.camera_viewer = None
        self.using_camera = False
        self.image_files: List[Path] = []
        self.current_img_idx = 0
        self.current_frame: Optional[np.ndarray] = None

        # Khởi tạo Cropper Engine
        self.cropper = PCBMonoCropper(conf_thresh=0.30, inference_size=1024) if HAS_CROPPER else None

        # Thư mục lưu kết quả debug
        self.out_dir = Path("debug_crops")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Cài đặt tham số điều khiển
        self.mode_idx = 0          # 0: Auto, 1: Băng tải Trắng (Bright), 2: Băng tải Đen (Dark)
        self.th_factor_pct = 60    # 60% (với nền sáng) hoặc +35 (với nền tối)
        self.morph_kernel_val = 35 # Kernel size
        self.min_area_pct = 4      # 4%
        self.exposure_ms = 30      # 30 ms

        # Cửa sổ hiển thị
        self.main_window = "PCB CROP DEBUGGER - [1: Goc | 2: Nhi Phan | 3: Mat Na | 4: PCB Deskewed]"
        self.zoom_window = "PCB Deskewed (Zoom View)"
        self.show_zoom_window = True

        # Thu thập danh sách ảnh nếu có
        self._init_source(target_path)

    def _init_source(self, target_path: Optional[str]):
        """Khởi tạo nguồn hình ảnh (File hoặc Camera)"""
        if target_path and Path(target_path).exists():
            p = Path(target_path)
            if p.is_file():
                self.image_files = [p]
            elif p.is_dir():
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                    self.image_files.extend(list(p.glob(ext)))
        else:
            # Tìm các ảnh mẫu có sẵn trong thư mục
            sample_candidates = ["test_user_pcb.png", "Image_20261019190958053.bmp", "2.jpg"]
            for s in sample_candidates:
                if Path(s).is_file():
                    self.image_files.append(Path(s))

            captures_dir = Path("captures")
            if captures_dir.is_dir():
                self.image_files.extend(list(captures_dir.glob("*.jpg")))

        # Nếu không có file chỉ định, thử kết nối camera Hikvision
        if not target_path and HAS_HIK_SDK:
            try:
                print("[*] Đang kiểm tra kết nối Camera Hikvision...")
                viewer = HikCameraViewer(default_exposure_us=30000.0)
                if viewer.connect():
                    if viewer.start_stream():
                        self.camera_viewer = viewer
                        self.using_camera = True
                        print("[+] Đã kết nối Camera Hikvision thành công! Sử dụng luồng trực tiếp.")
            except Exception as e:
                print(f"[!] Không mở được Camera Hikvision ({e}), chuyển sang chế độ nạp file ảnh.")

        if not self.using_camera and not self.image_files:
            print("[CẢNH BÁO] Không tìm thấy ảnh nào và camera chưa cắm. Tạo ảnh mẫu giả lập để chạy thử...")
            dummy = np.full((720, 1280, 3), 235, dtype=np.uint8)
            cv2.rectangle(dummy, (300, 200), (980, 520), (50, 50, 50), -1)
            cv2.putText(dummy, "PCB SAMPLE", (450, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            dummy_path = Path("sample_dummy.png")
            cv2.imwrite(str(dummy_path), dummy)
            self.image_files.append(dummy_path)

    def get_current_frame(self) -> Tuple[Optional[np.ndarray], str]:
        """Lấy khung hình hiện tại từ camera hoặc file ảnh"""
        if self.using_camera and self.camera_viewer:
            with self.camera_viewer.frame_lock:
                frame = self.camera_viewer.latest_frame.copy() if self.camera_viewer.latest_frame is not None else None
            return frame, f"Hikvision Live ({self.camera_viewer.device_name})"
        elif self.image_files:
            p = self.image_files[self.current_img_idx % len(self.image_files)]
            img = cv2.imread(str(p))
            return img, f"File [{self.current_img_idx + 1}/{len(self.image_files)}]: {p.name}"
        return None, "Không có nguồn ảnh"

    def debug_algorithm(self, img_input: np.ndarray) -> Dict[str, Any]:
        """
        Thực thi từng bước trong thuật toán và xuất ra toàn bộ ảnh trung gian
        """
        orig_h, orig_w = img_input.shape[:2]
        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if len(img_input.shape) == 3 else img_input.copy()

        # Bước 1: Đo độ sáng nền băng tải
        pad_edge = max(10, int(min(orig_h, orig_w) * 0.025))
        edge_pixels = np.concatenate([
            gray[:pad_edge, :].ravel(),
            gray[-pad_edge:, :].ravel(),
            gray[:, :pad_edge].ravel(),
            gray[:, -pad_edge:].ravel()
        ])
        bg_median = float(np.median(edge_pixels))

        # Phân loại chế độ
        if self.mode_idx == 1:
            is_bright = True
            mode_name = "Thu Cong: Bang tai TRANG (Nen Sang)"
        elif self.mode_idx == 2:
            is_bright = False
            mode_name = "Thu Cong: Bang tai DEN (Nen Toi)"
        else:
            is_bright = bg_median > 115.0
            mode_name = f"Tu Dong: {'Bang tai TRANG' if is_bright else 'Bang tai DEN'}"

        # Bước 2: Phân đoạn ngưỡng nhị phân
        if is_bright:
            factor = self.th_factor_pct / 100.0
            th_val = int(np.clip(bg_median * factor, 30, 220))
            _, binary = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY_INV)
            th_info = f"BINARY_INV (Nguong: {th_val}, Factor: {factor:.2f})"
        else:
            offset = self.th_factor_pct - 30
            th_val = int(np.clip(bg_median + offset, 20, 220))
            _, binary = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)
            th_info = f"BINARY (Nguong: {th_val}, Offset: {offset:+d})"

        # Bước 3: Phép đóng hình thái học và lấp đầy thân bo mạch
        k_val = max(3, self.morph_kernel_val | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_val, k_val))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(closed)
        min_mask_area = orig_h * orig_w * (self.min_area_pct / 100.0)
        for c in cnts:
            if cv2.contourArea(c) > min_mask_area:
                cv2.drawContours(filled_mask, [c], -1, 255, -1)

        # Bước 4: Chấm điểm ứng viên bo mạch
        cnts_cand, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in cnts_cand:
            area = cv2.contourArea(c)
            if area < min_mask_area or area > (orig_h * orig_w * 0.90):
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), angle = rect
            short_s = min(rw, rh)
            long_s = max(rw, rh)
            if short_s < 15 or long_s < 30:
                continue
            ar = long_s / short_s
            if not (1.05 <= ar <= 5.5):
                continue
            rectangularity = area / (rw * rh)
            if rectangularity < 0.55:
                continue
            score = (rectangularity ** 2) * area
            candidates.append({
                "contour": c, "rect": rect, "area": area,
                "ar": ar, "angle": angle, "rectangularity": rectangularity,
                "score": score
            })

        best = None
        warped_pcb = None
        ordered_corners = None
        if candidates:
            candidates.sort(key=lambda item: item["score"], reverse=True)
            best = candidates[0]
            box_pts = cv2.boxPoints(best["rect"]).astype(np.float32)

            hull = cv2.convexHull(best["contour"])
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.025 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = approx.reshape(4, 2).astype(np.float32)
            else:
                corners = box_pts

            # Sắp xếp 4 góc
            rect_arr = np.zeros((4, 2), dtype=np.float32)
            s = corners.sum(axis=1)
            rect_arr[0] = corners[np.argmin(s)]
            rect_arr[2] = corners[np.argmax(s)]
            diff = np.diff(corners, axis=1)
            rect_arr[1] = corners[np.argmin(diff)]
            rect_arr[3] = corners[np.argmax(diff)]
            ordered_corners = rect_arr

            (tl, tr, br, bl) = ordered_corners
            max_w = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
            max_h = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))

            if max_w >= 20 and max_h >= 20:
                dst_pts = np.array([
                    [0, 0], [max_w - 1, 0],
                    [max_w - 1, max_h - 1], [0, max_h - 1]
                ], dtype=np.float32)
                M = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
                warped_pcb = cv2.warpPerspective(img_input, M, (max_w, max_h), flags=cv2.INTER_CUBIC)
                if warped_pcb.shape[0] > warped_pcb.shape[1]:
                    warped_pcb = cv2.rotate(warped_pcb, cv2.ROTATE_90_CLOCKWISE)

        # Tạo ảnh hiển thị cho 4 Panel
        panel1 = img_input.copy()
        # Vẽ viền đo nền băng tải
        cv2.rectangle(panel1, (0, 0), (orig_w, pad_edge), (0, 165, 255), 1)
        cv2.rectangle(panel1, (0, orig_h - pad_edge), (orig_w, orig_h), (0, 165, 255), 1)
        cv2.rectangle(panel1, (0, 0), (pad_edge, orig_h), (0, 165, 255), 1)
        cv2.rectangle(panel1, (orig_w - pad_edge, 0), (orig_w, orig_h), (0, 165, 255), 1)

        if ordered_corners is not None:
            pts_int = np.int32(ordered_corners)
            cv2.polylines(panel1, [pts_int], isClosed=True, color=(0, 240, 255), thickness=3)
            labels = ["TL", "TR", "BR", "BL"]
            colors = [(0, 255, 0), (0, 200, 255), (0, 100, 255), (255, 200, 0)]
            for i, pt in enumerate(pts_int):
                cv2.circle(panel1, tuple(pt), 7, colors[i], -1)
                cv2.putText(panel1, labels[i], (pt[0] + 10, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2, cv2.LINE_AA)

        panel2 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        panel3 = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)

        # Panel 4: PCB Deskewed
        if warped_pcb is not None:
            panel4 = warped_pcb.copy()
        else:
            panel4 = np.zeros((300, 600, 3), dtype=np.uint8)
            cv2.putText(panel4, "CHUA PHAT HIEN PCB", (140, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

        return {
            "panel1": panel1,
            "panel2": panel2,
            "panel3": panel3,
            "panel4": panel4,
            "warped_pcb": warped_pcb,
            "best": best,
            "bg_median": bg_median,
            "mode_name": mode_name,
            "th_info": th_info,
            "corners": ordered_corners
        }

    def create_grid_display(self, res: Dict[str, Any], title_info: str) -> np.ndarray:
        """Ghép 4 Panel thành một lưới 2x2 đẹp mắt (1280x960)"""
        target_w, target_h = 640, 440

        def prepare_cell(img, header_text, sub_text=""):
            cell = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            # Banner đầu đề
            overlay = cell.copy()
            cv2.rectangle(overlay, (0, 0), (target_w, 42), (20, 24, 33), -1)
            cv2.addWeighted(overlay, 0.85, cell, 0.15, 0, cell)
            cv2.putText(cell, header_text, (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 240, 255), 1, cv2.LINE_AA)
            if sub_text:
                cv2.putText(cell, sub_text, (target_w - 280, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 220), 1, cv2.LINE_AA)
            # Viền ngăn cách ô
            cv2.rectangle(cell, (0, 0), (target_w - 1, target_h - 1), (50, 60, 75), 1)
            return cell

        best = res.get("best")
        if best:
            c1_sub = f"Goc: {best['angle']:.1f} | AR: {best['ar']:.2f}"
            c4_sub = f"Kich thuoc: {res['warped_pcb'].shape[1]}x{res['warped_pcb'].shape[0]} px" if res.get("warped_pcb") is not None else ""
        else:
            c1_sub = "Dang cho PCB..."
            c4_sub = "Rong"

        c1 = prepare_cell(res["panel1"], "[1/4] ANH GOC & DINH VI 4 GOC (OBB)", c1_sub)
        c2 = prepare_cell(res["panel2"], "[2/4] NHI PHAN TACH NEN (BINARY MASK)", res.get("th_info", ""))
        c3 = prepare_cell(res["panel3"], "[3/4] MAT NA DAC BO MACH (HOLE-FILLED)", f"Kernel: {self.morph_kernel_val}px")
        c4 = prepare_cell(res["panel4"], "[4/4] BO MACH DA NAN THANG 90 (DESKEWED)", c4_sub)

        grid = np.vstack([np.hstack([c1, c2]), np.hstack([c3, c4])])

        # Thanh trạng thái dưới cùng (Bottom HUD Bar)
        hud_bar = np.zeros((65, target_w * 2, 3), dtype=np.uint8)
        cv2.rectangle(hud_bar, (0, 0), (target_w * 2, 65), (15, 18, 26), -1)
        cv2.line(hud_bar, (0, 0), (target_w * 2, 0), (0, 200, 255), 2)

        # Chữ thông tin HUD
        cv2.putText(hud_bar, f"Nguon: {title_info}", (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud_bar, f"Che do: {res['mode_name']} (Nen: {res['bg_median']:.1f})", (15, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 1, cv2.LINE_AA)

        controls_hint = "[S]: Luu bo anh | [M]: Doi che do | [N/P]: Anh tiep/truoc | [W]: Zoom View | [Q]: Thoat"
        cv2.putText(hud_bar, controls_hint, (target_w * 2 - 620, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 100), 1, cv2.LINE_AA)

        return np.vstack([grid, hud_bar])

    def save_debug_bundle(self, img_input: np.ndarray, res: Dict[str, Any]):
        """Lưu toàn bộ ảnh phân tích vào thư mục debug_crops"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = self.out_dir / f"crop_{timestamp}"

        cv2.imwrite(f"{prefix}_1_orig.jpg", img_input)
        cv2.imwrite(f"{prefix}_2_binary.jpg", res["panel2"])
        cv2.imwrite(f"{prefix}_3_filled_mask.jpg", res["panel3"])
        cv2.imwrite(f"{prefix}_debug_grid.jpg", self.create_grid_display(res, "SAVED"))

        if res.get("warped_pcb") is not None:
            cv2.imwrite(f"{prefix}_4_deskewed_pcb.jpg", res["warped_pcb"], [cv2.IMWRITE_JPEG_QUALITY, 98])
            print(f"[ĐÃ LƯU] Bộ ảnh debug hoàn tất: {prefix.name}_*.jpg (Kích thước PCB: {res['warped_pcb'].shape[1]}x{res['warped_pcb'].shape[0]} px)")
        else:
            print(f"[ĐÃ LƯU] Bộ ảnh debug (Chưa phát hiện PCB): {prefix.name}_*.jpg")

    def run(self):
        """Khởi chạy giao diện Visual Debugger"""
        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.main_window, 1280, 945)

        # Thiết lập các thanh trượt điều khiển (Trackbars)
        def on_trackbar_change(val):
            pass

        cv2.createTrackbar("Che Do (0:Auto, 1:Trang, 2:Den)", self.main_window, self.mode_idx, 2, on_trackbar_change)
        cv2.createTrackbar("Nguong (%)", self.main_window, self.th_factor_pct, 100, on_trackbar_change)
        cv2.createTrackbar("Kernel Morph", self.main_window, self.morph_kernel_val, 75, on_trackbar_change)
        cv2.createTrackbar("Dien Tich Min (%)", self.main_window, self.min_area_pct, 30, on_trackbar_change)

        if self.using_camera and self.camera_viewer:
            cv2.createTrackbar("Exposure (ms)", self.main_window, self.exposure_ms, 500, on_trackbar_change)

        print("\n" + "=" * 65)
        print("   PCB CROP VISUAL DEBUGGER ĐANG CHẠY")
        print("   - Nhấn [S] hoặc [SPACE] : Lưu toàn bộ bộ ảnh debug")
        print("   - Nhấn [M]             : Đổi chế độ băng tải")
        print("   - Nhấn [N] hoặc [P]     : Đổi sang ảnh tiếp theo / trước đó")
        print("   - Nhấn [W]             : Bật / Tắt cửa sổ zoom riêng")
        print("   - Nhấn [Q] hoặc [ESC]   : Thoát")
        print("=" * 65 + "\n")

        try:
            while True:
                # 1. Đọc giá trị từ Trackbars
                self.mode_idx = cv2.getTrackbarPos("Che Do (0:Auto, 1:Trang, 2:Den)", self.main_window)
                self.th_factor_pct = max(10, cv2.getTrackbarPos("Nguong (%)", self.main_window))
                self.morph_kernel_val = max(5, cv2.getTrackbarPos("Kernel Morph", self.main_window))
                self.min_area_pct = max(1, cv2.getTrackbarPos("Dien Tich Min (%)", self.main_window))

                # Cập nhật exposure camera nếu có
                if self.using_camera and self.camera_viewer:
                    exp_val = cv2.getTrackbarPos("Exposure (ms)", self.main_window)
                    if exp_val != self.exposure_ms and exp_val >= 1:
                        self.exposure_ms = exp_val
                        self.camera_viewer.set_exposure(float(self.exposure_ms * 1000.0))

                # 2. Lấy khung hình
                frame, title_info = self.get_current_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                # 3. Chạy thuật toán debug
                res = self.debug_algorithm(frame)

                # 4. Hiển thị màn hình chính 4 ô
                display_grid = self.create_grid_display(res, title_info)
                cv2.imshow(self.main_window, display_grid)

                # 5. Hiển thị cửa sổ Zoom bo mạch
                if self.show_zoom_window:
                    if res.get("warped_pcb") is not None:
                        zoom_img = res["warped_pcb"]
                        # Phóng to nếu bo mạch quá nhỏ để nhìn linh kiện
                        zh, zw = zoom_img.shape[:2]
                        if zw < 800:
                            scale = 800.0 / zw
                            zoom_disp = cv2.resize(zoom_img, (800, int(zh * scale)), interpolation=cv2.INTER_LINEAR)
                        else:
                            zoom_disp = zoom_img.copy()
                        cv2.imshow(self.zoom_window, zoom_disp)
                    else:
                        placeholder = np.zeros((360, 800, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "DANG CHO BO MACH PCB...", (180, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                        cv2.imshow(self.zoom_window, placeholder)

                # 6. Bắt phím điều khiển
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('s'), ord('S'), 32):
                    self.save_debug_bundle(frame, res)
                elif key in (ord('m'), ord('M')):
                    self.mode_idx = (self.mode_idx + 1) % 3
                    cv2.setTrackbarPos("Che Do (0:Auto, 1:Trang, 2:Den)", self.main_window, self.mode_idx)
                elif key in (ord('n'), ord('N')):
                    if self.image_files:
                        self.current_img_idx = (self.current_img_idx + 1) % len(self.image_files)
                elif key in (ord('p'), ord('P')):
                    if self.image_files:
                        self.current_img_idx = (self.current_img_idx - 1) % len(self.image_files)
                elif key in (ord('w'), ord('W')):
                    self.show_zoom_window = not self.show_zoom_window
                    if not self.show_zoom_window:
                        try:
                            cv2.destroyWindow(self.zoom_window)
                        except Exception:
                            pass
                elif key in (ord('r'), ord('R')):
                    self.mode_idx = 0
                    self.th_factor_pct = 60
                    self.morph_kernel_val = 35
                    self.min_area_pct = 4
                    cv2.setTrackbarPos("Che Do (0:Auto, 1:Trang, 2:Den)", self.main_window, 0)
                    cv2.setTrackbarPos("Nguong (%)", self.main_window, 60)
                    cv2.setTrackbarPos("Kernel Morph", self.main_window, 35)
                    cv2.setTrackbarPos("Dien Tich Min (%)", self.main_window, 4)

                # Kiểm tra nếu đóng cửa sổ bằng nút X
                if cv2.getWindowProperty(self.main_window, cv2.WND_PROP_VISIBLE) < 1:
                    break

        finally:
            if self.using_camera and self.camera_viewer:
                print("[DỌN DẸP] Đang ngắt kết nối Camera Hikvision...")
                self.camera_viewer.stop()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            print("[HOÀN TẤT] Debugger đã đóng an toàn.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    debugger = PCBCropDebugger(target)
    debugger.run()

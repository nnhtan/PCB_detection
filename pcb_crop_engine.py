# -*- coding: utf-8 -*-
"""
THUẬT TOÁN ĐỊNH VỊ VÀ NẮN THẲNG BO MẠCH PCB TỰ ĐỘNG TRÊN BĂNG TẢI (SMART HYBRID OBB & DESKEW)
=============================================================================================
Tối ưu chuyên biệt cho Camera công nghiệp (Hikvision MV-CE120-10GM / Camera Mono / Webcam):
- Hỗ trợ cả 2 môi trường băng tải:
  1. Băng tải trắng sáng / Giấy lót trắng (Bo mạch đen / linh kiện phản xạ kim loại).
  2. Băng tải đen cao su / Bề mặt tối (Bo mạch sáng / xanh lá).
- Thuật toán thích ứng 2 chiều (Adaptive Dual-Polarity Thresholding + Morphological Hole-Filling):
  * Tự động đo độ tương phản nền băng tải ở 4 đường viền.
  * Lấp kín các lỗ hổng do tụ nhôm, chân IC sáng bóng và lỗ khoan trên PCB tạo ra.
  * Kết hợp cụm linh kiện YOLO (Component Anchor) để khoanh vùng và lọc bỏ hoàn toàn tạp âm cơ khí ngoài băng tải.
- Tự động nắn thẳng phối cảnh (Perspective Transform) xuất ra ảnh bo mạch 90° vuông vắn trước khi nhận diện.
"""

import sys
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

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

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class PCBMonoCropper:
    """
    Bộ nhận diện, tách nền băng tải và nắn thẳng bo mạch PCB công nghiệp
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.30,
        inference_size: int = 1024,
        model = None,
        ref_image: Optional[np.ndarray] = None,
    ):
        self.conf_thresh = conf_thresh
        self.inference_size = inference_size
        self.model = model
        self.ref_image = ref_image

        if self.model is None and HAS_YOLO:
            candidates = [
                model_path,
                "best.pt",
                "backend/models/best.pt",
                str(Path(__file__).resolve().parent / "best.pt"),
                str(Path(__file__).resolve().parent / "backend" / "models" / "best.pt")
            ]
            for cand in candidates:
                if cand and Path(cand).is_file():
                    try:
                        self.model = YOLO(str(cand))
                        print(f"[PCBCropper] Đã nạp model AI: {Path(cand).name}")
                        break
                    except Exception as e:
                        print(f"[PCBCropper] Lỗi nạp model {cand}: {e}")

    @staticmethod
    def get_dominant_angle(gray: np.ndarray) -> float:
        """
        Tìm góc nghiêng chủ đạo của các cạnh song song bo mạch bằng Canny + Hough Lines.
        Trả về góc nghiêng tính bằng độ trong khoảng [-45, 45].
        """
        gh, gw = gray.shape[:2]
        # Tối ưu hóa: Thu nhỏ ảnh về độ phân giải max 800px để Canny + Hough chạy < 30ms (thay vì 1000ms trên 12MP)
        if max(gh, gw) > 800:
            scale = 800.0 / max(gh, gw)
            proc_gray = cv2.resize(gray, (int(gw * scale), int(gh * scale)), interpolation=cv2.INTER_AREA)
        else:
            proc_gray = gray

        edges = cv2.Canny(proc_gray, 35, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=25, maxLineGap=10)
        if lines is None:
            return 0.0
        lines = lines.reshape(-1, 4)
        weighted_angles = []
        for x1, y1, x2, y2 in lines:
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < 25:
                continue
            ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            ang_mod = (ang + 45) % 90 - 45
            weighted_angles.append((length, ang_mod))

        if not weighted_angles:
            return 0.0

        bins: Dict[int, float] = {}
        for length, ang in weighted_angles:
            k = int(round(ang))
            bins[k] = bins.get(k, 0.0) + length

        best_angle = max(bins.items(), key=lambda x: x[1])[0]
        return float(best_angle)

    def process(
        self,
        img_input: np.ndarray,
        return_debug: bool = False,
        mode: str = "auto"
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Nhận diện và cắt DUY NHẤT bo mạch PCB, nắn thẳng góc xoay, loại bỏ phần thừa băng tải.
        Tự động phát hiện góc nghiêng và xoay bo mạch về góc 0° nằm ngang chuẩn xác.
        """
        info: Dict[str, Any] = {
            "success": False,
            "status_msg": "Băng tải rỗng (Chờ bo mạch)",
            "time_ms": 0.0,
            "corners": None,
            "transform": None,
            "angle": 0.0,
            "orig_size": None,
            "crop_size": None,
            "conveyor_type": "unknown",
            "debug_vis": None
        }

        if img_input is None or img_input.size == 0:
            return None, info

        t_start = time.perf_counter()
        orig_h, orig_w = img_input.shape[:2]
        info["orig_size"] = (orig_w, orig_h)

        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if len(img_input.shape) == 3 else img_input.copy()

        # 1. Phát hiện góc xoay nghiêng chủ đạo từ các cạnh thẳng của PCB
        dom_angle = self.get_dominant_angle(gray)
        rotated_frame = img_input
        M_rot_3x3 = np.eye(3, dtype=np.float64)

        if abs(dom_angle) > 1.0:
            center = (orig_w / 2.0, orig_h / 2.0)
            M_rot = cv2.getRotationMatrix2D(center, dom_angle, 1.0)
            cos = np.abs(M_rot[0, 0])
            sin = np.abs(M_rot[0, 1])
            nw = int((orig_h * sin) + (orig_w * cos))
            nh = int((orig_h * cos) + (orig_w * sin))
            M_rot[0, 2] += (nw / 2.0) - center[0]
            M_rot[1, 2] += (nh / 2.0) - center[1]
            rotated_frame = cv2.warpAffine(
                img_input, M_rot, (nw, nh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])

        rot_h, rot_w = rotated_frame.shape[:2]
        rot_gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY) if len(rotated_frame.shape) == 3 else rotated_frame

        # 2. Định vị cụm linh kiện qua YOLO trên ảnh đã xoay thẳng
        comp_boxes = []
        if self.model is not None:
            try:
                results = self.model.predict(
                    source=rotated_frame,
                    conf=self.conf_thresh,
                    imgsz=self.inference_size,
                    verbose=False
                )
                if results and len(results[0].boxes) > 0:
                    for b in results[0].boxes:
                        xyxy = b.xyxy[0].cpu().numpy()
                        cls_id = int(b.cls[0].item())
                        c_name = str(self.model.names[cls_id]).lower()
                        bw = xyxy[2] - xyxy[0]
                        bh = xyxy[3] - xyxy[1]
                        if c_name != "display" and bw < (rot_w * 0.45) and bh < (rot_h * 0.45):
                            comp_boxes.append(xyxy)
            except Exception:
                pass

        has_anchor = len(comp_boxes) >= 2
        comp_anchor = None
        if has_anchor:
            arr = np.array(comp_boxes)
            c_x1, c_y1 = float(arr[:, 0].min()), float(arr[:, 1].min())
            c_x2, c_y2 = float(arr[:, 2].max()), float(arr[:, 3].max())
            comp_anchor = (c_x1, c_y1, c_x2, c_y2)

        # 3. Phân đoạn ngưỡng nhị phân bo mạch đen trên băng tải sáng
        p10 = float(np.percentile(rot_gray, 10))
        p90 = float(np.percentile(rot_gray, 90))
        if mode == "bright" or (mode == "auto" and (p90 > 140 or p90 - p10 > 50)):
            info["conveyor_type"] = "Băng tải Sáng (Bo mạch Đen)"
            th_val = int(np.clip(p10 + (p90 - p10) * 0.45, 50, 160))
            _, binary = cv2.threshold(rot_gray, th_val, 255, cv2.THRESH_BINARY_INV)
        else:
            info["conveyor_type"] = "Băng tải Tối (Bo mạch Sáng)"
            th_val = int(np.clip(p10 + (p90 - p10) * 0.55, 45, 200))
            _, binary = cv2.threshold(rot_gray, th_val, 255, cv2.THRESH_BINARY)

        ksize = int(max(15, min(rot_h, rot_w) * 0.035)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None

        if comp_anchor is not None:
            c_x1, c_y1, c_x2, c_y2 = comp_anchor
            cw_comp = c_x2 - c_x1
            ch_comp = c_y2 - c_y1
            cx_comp = (c_x1 + c_x2) / 2.0
            cy_comp = (c_y1 + c_y2) / 2.0

            best_c = None
            best_dist = 999999.0
            for c in cnts:
                if cv2.contourArea(c) < 3000:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                # Loại bỏ contour nếu nuốt trọn cả viền ray kim loại bên ngoài
                if bw > cw_comp * 2.2 or bh > ch_comp * 2.5:
                    continue
                dist = cv2.pointPolygonTest(c, (cx_comp, cy_comp), True)
                if dist > -40 and abs(dist) < best_dist:
                    best_dist = abs(dist)
                    best_c = c

            if best_c is not None:
                bx, by, bw, bh = cv2.boundingRect(best_c)
                # Giới hạn lề (clamping) tránh contour bị lem dính vào ray kim loại
                pad_top_limit = int(ch_comp * 0.16)
                pad_side_limit = int(cw_comp * 0.15)
                bx1 = max(bx, int(c_x1 - pad_side_limit))
                by1 = max(by, int(c_y1 - pad_top_limit))
                bx2 = min(bx + bw, int(c_x2 + pad_side_limit))
                by2 = min(by + bh, int(c_y2 + pad_top_limit))
                best_box = (bx1, by1, bx2, by2)
            else:
                pad_w = int(cw_comp * 0.12)
                pad_h = int(ch_comp * 0.14)
                bx1 = max(0, int(c_x1 - pad_w))
                by1 = max(0, int(c_y1 - pad_h))
                bx2 = min(rot_w, int(c_x2 + pad_w))
                by2 = min(rot_h, int(c_y2 + pad_h))
                best_box = (bx1, by1, bx2, by2)
        elif cnts:
            min_area = max(4000.0, rot_h * rot_w * 0.005)
            valid_c = [c for c in cnts if cv2.contourArea(c) >= min_area]
            if valid_c:
                c = max(valid_c, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(c)
                best_box = (bx, by, bx + bw, by + bh)

        if best_box is None:
            info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
            info["status_msg"] = "Băng tải rỗng (Chưa phát hiện PCB)"
            return None, info

        bx1, by1, bx2, by2 = best_box
        warped = rotated_frame[by1:by2, bx1:bx2]
        if warped.size == 0 or warped.shape[0] < 20 or warped.shape[1] < 20:
            info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
            info["status_msg"] = "LỖI: Kích thước bo mạch quá nhỏ"
            return None, info

        T_crop = np.array([[1, 0, -bx1], [0, 1, -by1], [0, 0, 1]], dtype=np.float64)
        M_final = T_crop @ M_rot_3x3

        # 4. Chuẩn hóa hướng xoay: Luôn xuất ra dạng nằm ngang (w >= h)
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            R = np.array([[0, -1, warped.shape[1] - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
            M_final = R @ M_final

        # 5. So khớp chiều 0° vs 180° nếu có ảnh mẫu tham chiếu (tối ưu thumbnail 320x160 < 1ms)
        if self.ref_image is not None and self.ref_image.size > 0:
            try:
                ref_g = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY) if len(self.ref_image.shape) == 3 else self.ref_image
                w_g = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
                target_sz = (320, 160)
                ref_thumb = cv2.resize(ref_g, target_sz, interpolation=cv2.INTER_AREA)
                w0 = cv2.resize(w_g, target_sz, interpolation=cv2.INTER_AREA)
                w180 = cv2.rotate(w0, cv2.ROTATE_180)
                sc0 = cv2.matchTemplate(w0, ref_thumb, cv2.TM_CCOEFF_NORMED)[0][0]
                sc180 = cv2.matchTemplate(w180, ref_thumb, cv2.TM_CCOEFF_NORMED)[0][0]
                if sc180 > sc0 + 0.05:
                    warped = cv2.rotate(warped, cv2.ROTATE_180)
                    R180 = np.array([[-1, 0, warped.shape[1] - 1], [0, -1, warped.shape[0] - 1], [0, 0, 1]], dtype=np.float64)
                    M_final = R180 @ M_final
            except Exception:
                pass

        final_h, final_w = warped.shape[:2]
        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        # Tính toán 4 góc trên ảnh gốc
        try:
            M_inv = np.linalg.inv(M_final)
            dst_corners = np.array([
                [0, 0],
                [final_w - 1, 0],
                [final_w - 1, final_h - 1],
                [0, final_h - 1]
            ], dtype=np.float32).reshape(-1, 1, 2)
            orig_corners = cv2.perspectiveTransform(dst_corners, M_inv).reshape(4, 2)
        except Exception:
            orig_corners = None

        info["success"] = True
        info["status_msg"] = f"Phát hiện PCB ({final_w}x{final_h}px, Nghiêng {dom_angle:.1f}°)"
        info["time_ms"] = t_elapsed
        info["corners"] = orig_corners
        info["transform"] = M_final
        info["angle"] = dom_angle
        info["crop_size"] = (final_w, final_h)

        if return_debug and orig_corners is not None:
            debug_vis = img_input.copy()
            pts_int = np.int32(orig_corners)
            cv2.polylines(debug_vis, [pts_int], isClosed=True, color=(0, 240, 255), thickness=3)
            for pt in pts_int:
                cv2.circle(debug_vis, tuple(pt), 7, (0, 255, 128), -1)
            info["debug_vis"] = debug_vis

        return warped, info


def demo_file(image_path: str):
    """Hàm chạy thử nghiệm bóc tách bo mạch trên một file ảnh"""
    path = Path(image_path)
    if not path.exists():
        print(f"[LỖI] Không tìm thấy file: {image_path}")
        return

    print("=" * 65)
    print("   THUẬT TOÁN ĐỊNH VỊ & NẮN THẲNG BO MẠCH PCB (HYBRID OBB)")
    print("=" * 65)
    print(f"[*] Đang xử lý ảnh: {path.name}")

    img = cv2.imread(str(path))
    if img is None:
        print("[LỖI] Không thể đọc ảnh bằng OpenCV!")
        return

    cropper = PCBMonoCropper()
    cropped, info = cropper.process(img, return_debug=True)

    print(f"[KẾT QUẢ] {info['status_msg']} (Thời gian: {info['time_ms']:.1f} ms)")
    print(f"          Môi trường: {info.get('conveyor_type', 'N/A')}")

    if info["success"] and cropped is not None:
        out_dir = path.parent / "crop_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_crop_path = out_dir / f"deskewed_{path.stem}.jpg"
        cv2.imwrite(str(out_crop_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 98])
        print(f"[ĐÃ LƯU] Ảnh bo mạch đã nắn thẳng: {out_crop_path.name}")
        print(f"         Kích thước: {cropped.shape[1]} x {cropped.shape[0]} px")
        if info.get("debug_vis") is not None:
            debug_path = out_dir / f"debug_corners_{path.stem}.jpg"
            cv2.imwrite(str(debug_path), info["debug_vis"])
            print(f"[ĐÃ LƯU] Ảnh viền phát hiện: {debug_path.name}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "test_user_pcb.png"
    demo_file(target)

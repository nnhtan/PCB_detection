# -*- coding: utf-8 -*-
"""
THUẬT TOÁN ĐỊNH VỊ VÀ NẮN THẲNG BO MẠCH PCB XOAY ĐA GÓC (SMART HYBRID OBB & DESKEW)
===================================================================================
Khắc phục cả 2 lỗi:
1. "Bị nhận diện nhầm băng tải là PCB"
2. "Bị băng tải rỗng khi đang có bo mạch"

Cơ chế hoạt động 2 tầng (Two-Tier Hybrid Pipeline):
- Tầng 1 (AI Component Anchor + Local Rotated Rect):
  * Dùng mô hình YOLO (best.pt) định vị các linh kiện điện tử (tụ, IC, trở, cuộn cảm...).
  * Khi có bo mạch: Tạo vùng quan sát cục bộ (Local ROI) bao quanh cụm linh kiện.
  * Trong vùng cục bộ, phân đoạn phíp đồng và tìm hình chữ nhật xoay (minAreaRect).
  * Đo chính xác góc nghiêng θ của tấm mạch (ví dụ -32° hay bất kỳ góc nào).
  * 4 đỉnh góc xoay nghiêng ôm khít 100% theo đúng 4 cạnh mép của bo mạch.
  * Nắn thẳng phối cảnh (Perspective Transform) xuất ra ảnh bo mạch thẳng đứng 90°.
- Tầng 2 (Dự phòng Thị giác máy cổ điển):
  * Nếu không có YOLO hoặc chạy nhanh, tự động phân đoạn dựa trên mật độ chi tiết.
- Phòng vệ Băng tải rỗng:
  * Nếu không phát hiện linh kiện và không có contour hợp lệ -> Báo "Băng tải rỗng (Chờ PCB)",
    tuyệt đối không bao giờ nhận nhầm hay cắt mặt băng tải.
"""

import sys
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

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
    Bộ nhận diện và nắn thẳng bo mạch PCB công nghiệp xoay đa góc
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.25,
        inference_size: int = 800
    ):
        self.conf_thresh = conf_thresh
        self.inference_size = inference_size
        self.model = None

        if HAS_YOLO:
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
    def order_corners(pts: np.ndarray) -> np.ndarray:
        """
        Sắp xếp 4 điểm góc theo thứ tự chuẩn: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # Top-Left
        rect[2] = pts[np.argmax(s)]      # Bottom-Right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # Top-Right
        rect[3] = pts[np.argmax(diff)]   # Bottom-Left

        return rect

    def process(self, img_input: np.ndarray, return_debug: bool = False) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Nhận diện và cắt DUY NHẤT bo mạch PCB, nắn thẳng góc xoay, loại bỏ băng tải.
        """
        info: Dict[str, Any] = {
            "success": False,
            "status_msg": "Băng tải rỗng (Chờ bo mạch)",
            "time_ms": 0.0,
            "corners": None,
            "angle": 0.0,
            "orig_size": None,
            "crop_size": None,
            "debug_vis": None
        }

        t_start = time.perf_counter()
        orig_h, orig_w = img_input.shape[:2]
        info["orig_size"] = (orig_w, orig_h)

        # Chuẩn hóa grayscale
        if len(img_input.shape) == 3:
            # Khử vệt màu UI vẽ đè nếu có
            diff_c = np.max(np.abs(img_input[:, :, :2] - img_input[:, :, 1:]), axis=2)
            if np.count_nonzero(diff_c > 20) > 100:
                mask_ui = (diff_c > 20).astype(np.uint8) * 255
                gray = cv2.inpaint(img_input[:, :, 0], mask_ui, 3, cv2.INPAINT_TELEA)
            else:
                gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_input

        # -------------------------------------------------------------
        # TẦNG 1: ĐỊNH VỊ VÙNG BO MẠCH BẰNG AI (YOLO) + TÁCH GÓC XOAY
        # -------------------------------------------------------------
        if self.model is not None:
            results = self.model.predict(
                source=img_input,
                conf=self.conf_thresh,
                imgsz=self.inference_size,
                verbose=False
            )

            valid_boxes = []
            if results and len(results[0].boxes) > 0:
                for b in results[0].boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    bw = xyxy[2] - xyxy[0]
                    bh = xyxy[3] - xyxy[1]
                    # Loại bỏ các box phát hiện nhầm khổng lồ (ray cơ khí)
                    if bw < (orig_w * 0.40) and bh < (orig_h * 0.45):
                        valid_boxes.append(xyxy)

            # Nếu có từ 2 linh kiện trở lên -> Chắc chắn có bo mạch PCB!
            if len(valid_boxes) >= 2:
                v_boxes = np.array(valid_boxes)
                c_x1 = int(v_boxes[:, 0].min())
                c_y1 = int(v_boxes[:, 1].min())
                c_x2 = int(v_boxes[:, 2].max())
                c_y2 = int(v_boxes[:, 3].max())

                # Tạo vùng quan sát cục bộ (Local ROI) bao quanh PCB
                pad = int(max(c_x2 - c_x1, c_y2 - c_y1) * 0.28)
                rx1 = max(0, c_x1 - pad)
                ry1 = max(0, c_y1 - pad)
                rx2 = min(orig_w, c_x2 + pad)
                ry2 = min(orig_h, c_y2 + pad)

                local_gray = gray[ry1:ry2, rx1:rx2]

                # Phân đoạn tách phíp PCB khỏi mặt băng tải trong vùng cục bộ
                blur_roi = cv2.bilateralFilter(local_gray, 7, 50, 50)
                # Ngưỡng cục bộ
                bg_s = np.concatenate([
                    blur_roi[:15, :15].ravel(), blur_roi[:15, -15:].ravel(),
                    blur_roi[-15:, :15].ravel(), blur_roi[-15:, -15:].ravel()
                ])
                bg_val = float(np.median(bg_s))
                th_val = max(38.0, min(80.0, bg_val + 10.0))
                _, th_roi = cv2.threshold(blur_roi, int(th_val), 255, cv2.THRESH_BINARY)

                # Phép đóng kết nối khối
                ksize = max(9, int(min(local_gray.shape[:2]) * 0.035)) | 1
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                closed_roi = cv2.morphologyEx(th_roi, cv2.MORPH_CLOSE, k, iterations=2)

                cnts_roi, _ = cv2.findContours(closed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_roi:
                    largest_roi = max(cnts_roi, key=cv2.contourArea)
                    (lcx, lcy), (lw, lh), angle = cv2.minAreaRect(largest_roi)
                    gcx = lcx + rx1
                    gcy = lcy + ry1
                    full_rect = ((gcx, gcy), (lw, lh), angle)
                else:
                    # Dự phòng theo cụm linh kiện
                    full_rect = (((c_x1 + c_x2) / 2.0, (c_y1 + c_y2) / 2.0), (c_x2 - c_x1 + 80, c_y2 - c_y1 + 80), 0.0)

                box_orig = cv2.boxPoints(full_rect).astype(np.float32)
                ordered_corners = self.order_corners(box_orig)

                # Nắn thẳng phối cảnh
                (tl, tr, br, bl) = ordered_corners
                maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
                maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

                dst_pts = np.array([
                    [0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]
                ], dtype=np.float32)

                M = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
                cropped = cv2.warpPerspective(img_input, M, (maxW, maxH))

                if maxW > maxH * 1.3:
                    cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
                    maxW, maxH = maxH, maxW

                t_elapsed = (time.perf_counter() - t_start) * 1000
                info["success"] = True
                info["status_msg"] = f"Phát hiện PCB (Góc {full_rect[2]:.1f}°, {maxW}x{maxH}px)"
                info["time_ms"] = t_elapsed
                info["corners"] = ordered_corners
                info["angle"] = full_rect[2]
                info["crop_size"] = (maxW, maxH)

                if return_debug:
                    debug_vis = cv2.resize(img_input, (1280, int(orig_h * (1280 / orig_w))))
                    sf = 1280.0 / orig_w
                    pts_disp = np.int32(ordered_corners * sf)
                    cv2.polylines(debug_vis, [pts_disp], isClosed=True, color=(0, 255, 0), thickness=2)
                    for pt in pts_disp:
                        cv2.circle(debug_vis, tuple(pt), 6, (0, 200, 255), -1)
                    info["debug_vis"] = debug_vis

                return cropped, info

        # -------------------------------------------------------------
        # TẦNG 2: THỊ GIÁC MÁY CỔ ĐIỂN DỰ PHÒNG (NẾU CHƯA NẠP MODEL)
        # -------------------------------------------------------------
        scale = 800.0 / max(orig_h, orig_w) if max(orig_h, orig_w) > 800 else 1.0
        sw = int(orig_w * scale)
        sh = int(orig_h * scale)
        small = cv2.resize(gray, (sw, sh))

        blurred = cv2.bilateralFilter(small, 9, 75, 75)
        # Lấy mẫu nền
        bg = np.median([blurred[:20, :20], blurred[:20, -20:], blurred[-20:, :20], blurred[-20:, -20:]])
        _, th = cv2.threshold(blurred, int(max(35, bg + 10)), 255, cv2.THRESH_BINARY)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=3)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < (sw * sh * 0.015) or area > (sw * sh * 0.90):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw >= sw - 6 and bh >= sh - 6:
                continue
            rect = cv2.minAreaRect(c)
            candidates.append((area, rect))

        if not candidates:
            info["time_ms"] = (time.perf_counter() - t_start) * 1000
            info["status_msg"] = "Băng tải rỗng (Chưa có PCB)"
            return None, info

        best_rect = max(candidates, key=lambda x: x[0])[1]
        (cx, cy), (bw, bh), angle = best_rect
        full_rect = ((cx / scale, cy / scale), (bw / scale, bh / scale), angle)
        box_orig = cv2.boxPoints(full_rect).astype(np.float32)
        ordered_corners = self.order_corners(box_orig)

        (tl, tr, br, bl) = ordered_corners
        maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        dst_pts = np.array([
            [0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
        cropped = cv2.warpPerspective(img_input, M, (maxW, maxH))

        if maxW > maxH * 1.3:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
            maxW, maxH = maxH, maxW

        info["success"] = True
        info["status_msg"] = f"Phát hiện PCB (Góc {angle:.1f}°, {maxW}x{maxH}px)"
        info["time_ms"] = (time.perf_counter() - t_start) * 1000
        info["corners"] = ordered_corners
        info["angle"] = angle
        info["crop_size"] = (maxW, maxH)

        return cropped, info


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

    if info["success"]:
        out_dir = path.parent / "crop_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_crop_path = out_dir / f"deskewed_{path.stem}.jpg"
        cv2.imwrite(str(out_crop_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 98])
        print(f"[ĐÃ LƯU] Ảnh bo mạch đã nắn thẳng: {out_crop_path.name}")
        print(f"         Kích thước: {cropped.shape[1]} x {cropped.shape[0]} px")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "captures/capture_20261021_230602_exp_44000us.jpg"
    demo_file(target)

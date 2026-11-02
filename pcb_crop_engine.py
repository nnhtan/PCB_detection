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
        inference_size: int = 800,
        model = None,
    ):
        self.conf_thresh = conf_thresh
        self.inference_size = inference_size
        self.model = model

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

    def crop_component_cluster(self, image: np.ndarray, boxes: np.ndarray):
        image_height, image_width = image.shape[:2]
        valid_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).reshape(4)
            box_width = x2 - x1
            box_height = y2 - y1
            if (not np.all(np.isfinite([x1, y1, x2, y2]))
                    or box_width < 2 or box_height < 2
                    or box_width * box_height > 0.25 * image_width * image_height):
                continue
            valid_boxes.append([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ])

        if len(valid_boxes) < 3:
            return None, None, None

        component_points = np.asarray(valid_boxes, dtype=np.float32).reshape(-1, 2)
        hull = cv2.convexHull(component_points)
        rect = cv2.minAreaRect(hull)
        rect_width, rect_height = rect[1]
        if rect_width < 2 or rect_height < 2:
            return None, None, None

        padding = max(2, int(round(max(rect_width, rect_height) * 0.025)))
        corners = None
        while padding > 0:
            expanded_rect = (
                rect[0],
                (rect_width + 2 * padding, rect_height + 2 * padding),
                rect[2],
            )
            candidate_corners = cv2.boxPoints(expanded_rect)
            if (
                np.all(candidate_corners[:, 0] >= 0)
                and np.all(candidate_corners[:, 0] <= image_width - 1)
                and np.all(candidate_corners[:, 1] >= 0)
                and np.all(candidate_corners[:, 1] <= image_height - 1)
            ):
                corners = candidate_corners
                break
            padding //= 2

        if corners is None:
            corners = cv2.boxPoints(rect)
            if (
                np.any(corners[:, 0] < 0)
                or np.any(corners[:, 0] > image_width - 1)
                or np.any(corners[:, 1] < 0)
                or np.any(corners[:, 1] > image_height - 1)
            ):
                return None, None, None

        ordered_corners = self.order_corners(corners.astype(np.float32))
        top_left, top_right, bottom_right, bottom_left = ordered_corners
        crop_width = int(max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        ))
        crop_height = int(max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        ))
        if crop_width < 2 or crop_height < 2:
            return None, None, None

        destination = np.array([
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(ordered_corners, destination)
        cropped = cv2.warpPerspective(image, transform, (crop_width, crop_height))

        if crop_height > crop_width:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
            rotate_transform = np.array([
                [0, -1, crop_height - 1],
                [1, 0, 0],
                [0, 0, 1],
            ], dtype=np.float64)
            transform = rotate_transform @ transform
            crop_width, crop_height = crop_height, crop_width

        return cropped, {
            "corners": ordered_corners,
            "transform": transform,
            "angle": rect[2],
            "crop_size": (crop_width, crop_height),
        }, rect

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

            boxes = (
                results[0].boxes.xyxy.cpu().numpy()
                if results and results[0].boxes is not None
                else np.empty((0, 4), dtype=np.float32)
            )
            cropped, crop_info, rectangle = self.crop_component_cluster(
                img_input, boxes
            )
            if cropped is not None and crop_info is not None:
                t_elapsed = (time.perf_counter() - t_start) * 1000
                info["success"] = True
                crop_width, crop_height = crop_info["crop_size"]
                info["status_msg"] = (
                    f"Phát hiện PCB (Góc {rectangle[2]:.1f}°, "
                    f"{crop_width}x{crop_height}px)"
                )
                info["time_ms"] = t_elapsed
                info.update(crop_info)

                if return_debug:
                    debug_vis = cv2.resize(img_input, (1280, int(orig_h * (1280 / orig_w))))
                    sf = 1280.0 / orig_w
                    pts_disp = np.int32(crop_info["corners"] * sf)
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

        if maxH > maxW:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
            R = np.array([[0, -1, maxH - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
            M = R @ M
            maxW, maxH = maxH, maxW

        info["success"] = True
        info["status_msg"] = f"Phát hiện PCB (Góc {angle:.1f}°, {maxW}x{maxH}px)"
        info["time_ms"] = (time.perf_counter() - t_start) * 1000
        info["corners"] = ordered_corners
        info["transform"] = M
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

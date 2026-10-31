# -*- coding: utf-8 -*-
"""
THUẬT TOÁN ĐỊNH VỊ VÀ BÓC TÁCH BO MẠCH PCB BẰNG MODEL YOLO OBB (pcb_crop.pt)
=============================================================================
Sử dụng mô hình YOLOv8-OBB (Oriented Bounding Box) đã huấn luyện riêng cho PCB:
- Trích xuất trực tiếp 4 góc nghiêng tự do (xyxyxyxy) của bo mạch từ mô hình AI.
- Tự động nắn thẳng phối cảnh (Perspective Transform) đưa PCB về hình chữ nhật chuẩn.
- Tự động chuẩn hóa xoay ngang 90° và đối chiếu chiều 0° / 180° với ảnh mẫu.
- Tích hợp bộ lọc cắt loại bỏ phần thừa ngoài băng tải (Conveyor ROI).
"""

import sys
import time
import json
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


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = CURRENT_DIR / "pcb_model_crop" / "pcb_crop.pt"
CONVEYOR_CONFIG_FILE = CURRENT_DIR / "conveyor_config.json"


def load_conveyor_config() -> Dict[str, Any]:
    """Đọc cấu hình lề và vùng băng tải từ file JSON"""
    default_cfg = {
        "filter_enabled": True,
        "auto_detect": True,
        "margin_left": 0.28,
        "margin_right": 0.06,
        "margin_top": 0.0,
        "margin_bottom": 0.0
    }
    if CONVEYOR_CONFIG_FILE.is_file():
        try:
            with open(CONVEYOR_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_cfg.update(data)
        except Exception:
            pass
    return default_cfg


def save_conveyor_config(cfg: Dict[str, Any]):
    """Lưu cấu hình vùng băng tải vào file JSON"""
    try:
        with open(CONVEYOR_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể lưu {CONVEYOR_CONFIG_FILE.name}: {e}")


class PCBYoloCropper:
    """
    Bộ bóc tách và nắn thẳng PCB bằng mô hình YOLO OBB (pcb_crop.pt):
    1. Dự đoán tọa độ 4 góc hộp bao xoay nghiêng (Oriented Bounding Box) qua YOLO.
    2. Sắp xếp 4 góc theo thứ tự [Trên-Trái, Trên-Phải, Dưới-Phải, Dưới-Trái].
    3. Biến đổi phối cảnh (Perspective Transform) xuất ảnh PCB phẳng.
    4. Tự động xoay ngang và căn chỉnh 180° theo ảnh tham chiếu.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = 0.05,
        imgsz: int = 640,
        reference_path: Optional[str] = None
    ):
        self.conf = conf
        self.imgsz = imgsz
        self.model = None
        self.ref_image = None
        self.conveyor_cfg = load_conveyor_config()
        self.opencv_fallback = None

        # Tìm kiếm file model
        candidates = [
            model_path,
            DEFAULT_MODEL_PATH,
            CURRENT_DIR / "pcb_crop.pt",
            Path("pcb_model_crop/pcb_crop.pt"),
            Path("pcb_crop.pt"),
        ]
        self.model_path = None
        for c in candidates:
            if c and Path(c).is_file():
                self.model_path = Path(c).resolve()
                break

        self._load_model()
        self.reference_path = reference_path
        self._load_reference()

    def _load_model(self):
        """Khởi tạo mô hình YOLO OBB"""
        if not HAS_YOLO:
            print("[CẢNH BÁO] Chưa cài đặt ultralytics. Hãy chạy: pip install ultralytics")
            return
        if not self.model_path:
            print(f"[CẢNH BÁO] Không tìm thấy file model: {DEFAULT_MODEL_PATH}")
            return
        try:
            print(f"[*] Đang tải mô hình YOLO OBB: {self.model_path.name}...")
            self.model = YOLO(str(self.model_path))
            print(f"[+] Tải model thành công: Task={getattr(self.model, 'task', 'obb')}")
        except Exception as e:
            print(f"[LỖI] Không thể nạp model {self.model_path}: {e}")
            self.model = None

    def _load_reference(self):
        """Nạp ảnh tham chiếu PCB nếu có để chuẩn hóa 0° / 180°"""
        candidates = [
            self.reference_path,
            CURRENT_DIR / "backend" / "storage" / "reference.jpg",
            CURRENT_DIR / "backend" / "storage" / "references" / "default" / "reference.jpg",
            CURRENT_DIR / "reference_detected.jpg",
            Path("backend/storage/reference.jpg"),
        ]
        for c in candidates:
            if c:
                p = Path(c)
                if p.is_file():
                    try:
                        img = cv2.imread(str(p))
                        if img is not None:
                            if img.shape[0] > img.shape[1]:
                                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                            self.ref_image = img
                            break
                    except Exception:
                        pass

    def get_conveyor_roi(self, img_input: np.ndarray) -> Tuple[int, int, int, int]:
        """Xác định vùng lòng băng tải loại bỏ các chi tiết thừa ngoài lề"""
        h, w = img_input.shape[:2]
        cfg = getattr(self, "conveyor_cfg", {})
        m_left = float(cfg.get("margin_left", 0.28))
        m_right = float(cfg.get("margin_right", 0.06))
        m_top = float(cfg.get("margin_top", 0.0))
        m_bottom = float(cfg.get("margin_bottom", 0.0))

        if cfg.get("auto_detect", True):
            try:
                gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if len(img_input.shape) == 3 else img_input
                sh = 300
                sw = max(10, int(w * sh / h))
                small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
                grad_x = np.abs(cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3))
                col_grad = np.mean(grad_x, axis=0)

                l_start, l_end = int(sw * 0.15), int(sw * 0.35)
                left_sub = col_grad[l_start:l_end]
                if len(left_sub) > 0 and np.max(left_sub) > np.mean(col_grad) * 1.15:
                    x1 = int((l_start + np.argmax(left_sub)) * w / sw)
                else:
                    x1 = int(w * m_left)

                r_start, r_end = int(sw * 0.85), int(sw * 0.98)
                right_sub = col_grad[r_start:r_end]
                if len(right_sub) > 0 and np.max(right_sub) > np.mean(col_grad) * 1.15:
                    x2 = int((r_start + np.argmax(right_sub)) * w / sw)
                else:
                    x2 = int(w * (1.0 - m_right))
            except Exception:
                x1 = int(w * m_left)
                x2 = int(w * (1.0 - m_right))
        else:
            x1 = int(w * m_left)
            x2 = int(w * (1.0 - m_right))

        y1 = int(h * m_top)
        y2 = int(h * (1.0 - m_bottom))
        x1 = max(0, min(x1, w - 20))
        x2 = max(x1 + 20, min(x2, w))
        y1 = max(0, min(y1, h - 20))
        y2 = max(y1 + 20, min(y2, h))
        return x1, y1, x2, y2

    def crop_conveyor(self, img_input: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Cắt và trả về duy nhất vùng băng tải, loại bỏ phần thừa ngoài biên"""
        if roi is None:
            roi = self.get_conveyor_roi(img_input)
        x1, y1, x2, y2 = roi
        return img_input[y1:y2, x1:x2].copy(), roi

    @staticmethod
    def order_corners(pts: np.ndarray) -> np.ndarray:
        """Sắp xếp 4 điểm góc theo thứ tự: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]"""
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
        img_input: np.ndarray,
        return_debug: bool = False,
        filter_conveyor: Optional[bool] = None,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Nhận diện và cắt nắn thẳng bo mạch PCB bằng mô hình YOLO OBB:
        - img_input: Ảnh đầu vào BGR
        - return_debug: Nếu True, trả về ảnh vẽ viền OBB và các góc
        - filter_conveyor: Bật/Tắt cắt lề ngoài băng tải
        - conf: Ngưỡng tự tin (mặc định self.conf = 0.20)
        - imgsz: Kích thước inference (mặc định self.imgsz = 1024)
        """
        t_start = time.perf_counter()
        info: Dict[str, Any] = {
            "success": False,
            "status_msg": "Băng tải rỗng (Chờ PCB)",
            "time_ms": 0.0,
            "corners": None,
            "confidence": 0.0,
            "angle": 0.0,
            "orig_size": None,
            "crop_size": None,
            "debug_vis": None,
            "corner_method": "YOLO_OBB",
            "conveyor_roi": None
        }

        if img_input is None or img_input.size == 0:
            info["status_msg"] = "LỖI: Ảnh đầu vào rỗng"
            return None, info

        if self.model is None:
            info["status_msg"] = "LỖI: Chưa nạp mô hình pcb_crop.pt"
            return None, info

        orig_h, orig_w = img_input.shape[:2]
        info["orig_size"] = (orig_w, orig_h)

        if filter_conveyor is None:
            filter_conveyor = bool(self.conveyor_cfg.get("filter_enabled", True))

        if filter_conveyor:
            cx1, cy1, cx2, cy2 = self.get_conveyor_roi(img_input)
            proc_img = img_input[cy1:cy2, cx1:cx2]
            shift_x, shift_y = cx1, cy1
        else:
            cx1, cy1, cx2, cy2 = 0, 0, orig_w, orig_h
            proc_img = img_input
            shift_x, shift_y = 0, 0

        info["conveyor_roi"] = (cx1, cy1, cx2, cy2)
        use_conf = conf if conf is not None else self.conf
        use_imgsz = imgsz if imgsz is not None else self.imgsz
        total_area = float(orig_h * orig_w)
        has_obb = False
        best_conf = 0.0
        raw_corners = None

        try:
            # 1. Dự đoán bằng YOLO OBB trực tiếp trên vùng làm việc proc_img để tăng tốc
            results = self.model.predict(
                source=proc_img,
                conf=use_conf,
                imgsz=use_imgsz,
                verbose=False
            )[0]
            if hasattr(results, 'obb') and results.obb is not None and len(results.obb) > 0:
                confs = results.obb.conf.cpu().numpy()
                boxes = results.obb.xyxyxyxy.cpu().numpy()
                best_idx = int(np.argmax(confs))
                best_conf = float(confs[best_idx])
                raw_corners = boxes[best_idx].astype(np.float32)
                # Chuyển đổi tọa độ từ proc_img về hệ tọa độ ảnh gốc
                raw_corners[:, 0] += shift_x
                raw_corners[:, 1] += shift_y
                has_obb = True
        except Exception:
            pass

        # Kiểm tra xem box OBB có phải đúng kích thước bo mạch PCB không
        # Nếu box bao trùm > 35% diện tích ảnh (tức là bao cả lòng băng tải) hoặc không tìm thấy OBB:
        use_fallback = False
        if not has_obb:
            use_fallback = True
        else:
            b_area = cv2.contourArea(np.int32(raw_corners))
            if b_area > 0.35 * total_area or b_area < 0.01 * total_area:
                use_fallback = True

        if use_fallback:
            if self.opencv_fallback is None:
                try:
                    from camera_viewer import PCBMonoCropper
                    self.opencv_fallback = PCBMonoCropper(reference_path=self.reference_path)
                except Exception:
                    pass
            if self.opencv_fallback is not None:
                crop_cv, info_cv = self.opencv_fallback.process(img_input, filter_conveyor=filter_conveyor)
                if info_cv.get("success") and crop_cv is not None:
                    info_cv["status_msg"] = f"PCB ({info_cv['crop_size'][0]}x{info_cv['crop_size'][1]}px, Conf: {best_conf*100:.0f}%)" if has_obb else f"PCB ({info_cv['crop_size'][0]}x{info_cv['crop_size'][1]}px)"
                    info_cv["confidence"] = best_conf if has_obb else 0.90
                    info_cv["time_ms"] = (time.perf_counter() - t_start) * 1000.0
                    return crop_cv, info_cv

        if not has_obb:
            info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
            info["status_msg"] = "Băng tải rỗng (Chưa phát hiện PCB)"
            return None, info

        # Nếu OBB đúng kích thước PCB
        corners = raw_corners.copy()
        corners[:, 0] = np.clip(corners[:, 0], 0, orig_w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, orig_h - 1)

        # 3. Sắp xếp 4 góc chuẩn hóa
        ordered_corners = self.order_corners(corners)
        (tl, tr, br, bl) = ordered_corners

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_w = int(round(max(width_top, width_bottom)))

        height_right = np.linalg.norm(tr - br)
        height_left = np.linalg.norm(tl - bl)
        max_h = int(round(max(height_right, height_left)))

        if max_w < 15 or max_h < 15:
            info["time_ms"] = (time.perf_counter() - t_start) * 1000.0
            info["status_msg"] = "LỖI: Kích thước PCB phát hiện quá nhỏ"
            return None, info

        # 4. Biến đổi phối cảnh (Perspective Transform)
        dst_pts = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
        warped = cv2.warpPerspective(img_input, M, (max_w, max_h), flags=cv2.INTER_CUBIC)

        # 5. Xoay chuẩn hóa nằm ngang 90°
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        # 6. Đối chiếu 0° / 180° với ảnh tham chiếu
        if self.ref_image is not None:
            try:
                ref_gray = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY) if len(self.ref_image.shape) == 3 else self.ref_image
                w_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped

                target_ref_size = (ref_gray.shape[1], ref_gray.shape[0])
                w0 = cv2.resize(w_gray, target_ref_size, interpolation=cv2.INTER_AREA)
                w180 = cv2.rotate(w0, cv2.ROTATE_180)

                score0 = cv2.matchTemplate(w0, ref_gray, cv2.TM_CCOEFF_NORMED)[0][0]
                score180 = cv2.matchTemplate(w180, ref_gray, cv2.TM_CCOEFF_NORMED)[0][0]

                if score180 > score0 + 0.05:
                    warped = cv2.rotate(warped, cv2.ROTATE_180)
            except Exception:
                pass

        crop_h, crop_w = warped.shape[:2]
        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        info["success"] = True
        info["status_msg"] = f"YOLO OBB PCB (Conf: {best_conf*100:.0f}%, {crop_w}x{crop_h}px)"
        info["time_ms"] = t_elapsed
        info["corners"] = ordered_corners
        info["confidence"] = best_conf
        info["crop_size"] = (crop_w, crop_h)

        if return_debug:
            debug_vis = img_input.copy()
            if filter_conveyor:
                cv2.rectangle(debug_vis, (cx1, cy1), (cx2, cy2), (255, 140, 0), 2)
            pts_int = np.int32(ordered_corners)
            cv2.polylines(debug_vis, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3)
            for pt in pts_int:
                cv2.circle(debug_vis, tuple(pt), 8, (0, 200, 255), -1)
            cv2.putText(
                debug_vis,
                f"YOLO OBB: {best_conf*100:.1f}%",
                (int(ordered_corners[0][0]), max(20, int(ordered_corners[0][1]) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )
            info["debug_vis"] = debug_vis

        return warped, info


def demo_file(image_path: str, conf: float = 0.20):
    """Kiểm tra mô hình YOLO OBB trên một file ảnh"""
    path = Path(image_path)
    if not path.exists():
        print(f"[LỖI] Không tìm thấy file: {image_path}")
        return

    print("=" * 65)
    print("   THỬ NGHIỆM ĐỊNH VỊ & CẮT PCB BẰNG MODEL YOLO OBB (pcb_crop.pt)")
    print("=" * 65)
    print(f"[*] Ảnh đầu vào: {path.name}")

    img = cv2.imread(str(path))
    if img is None:
        print("[LỖI] Không thể đọc ảnh bằng OpenCV!")
        return

    cropper = PCBYoloCropper(conf=conf)
    cropped, info = cropper.process(img, return_debug=True, filter_conveyor=True)

    print(f"[KẾT QUẢ] {info['status_msg']} (Thời gian: {info['time_ms']:.1f} ms)")

    out_dir = path.parent / "crop_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lưu ảnh lòng băng tải
    conveyor_clean, roi = cropper.crop_conveyor(img)
    conv_out = out_dir / f"conveyor_clean_{path.stem}.jpg"
    cv2.imwrite(str(conv_out), conveyor_clean, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[ĐÃ LƯU] Ảnh lòng băng tải (x={roi[0]}..{roi[2]}): {conv_out.name}")

    if info["success"] and cropped is not None:
        out_crop_path = out_dir / f"yolo_crop_{path.stem}.jpg"
        cv2.imwrite(str(out_crop_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 98])
        print(f"[ĐÃ LƯU] Ảnh PCB nắn thẳng: {out_crop_path.name}")
        print(f"         Kích thước: {cropped.shape[1]} x {cropped.shape[0]} px | Conf: {info['confidence']*100:.1f}%")
        if info.get("debug_vis") is not None:
            debug_path = out_dir / f"yolo_debug_{path.stem}.jpg"
            cv2.imwrite(str(debug_path), info["debug_vis"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[ĐÃ LƯU] Ảnh trực quan 4 góc: {debug_path.name}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "captures/capture_20261021_230612_exp_44000us.jpg"
    demo_file(target)

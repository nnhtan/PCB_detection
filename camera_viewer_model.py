# -*- coding: utf-8 -*-
"""
BẢN DÙNG THỬ BÓC TÁCH PCB BẰNG MODEL YOLO OBB (pcb_crop.pt)
=============================================================
- Kết nối Camera Công nghiệp Hikvision GigE / USB (MV-CE120-10GM...)
- Tích hợp mô hình AI chuyên dụng YOLO OBB (pcb_model_crop/pcb_crop.pt)
- Tự động nhận diện 4 góc nghiêng của PCB (Oriented Bounding Box)
- Tự động loại bỏ phần thừa ngoài băng tải (Conveyor Masking)
- Nắn phẳng bo mạch (Perspective Transform) và xoay chuẩn ngang 90°
- Hỗ trợ thanh kéo điều chỉnh thời gian lấy sáng (Exposure) và độ nhạy AI (Conf)
- Phím [S] chụp và lưu ảnh 12MP: ảnh gốc, ảnh lòng băng tải, ảnh PCB nắn thẳng
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from ctypes import *
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

CURRENT_DIR = Path(__file__).resolve().parent
MV_IMPORT_PATH = CURRENT_DIR / "Python" / "MvImport"
if MV_IMPORT_PATH.exists():
    sys.path.insert(0, str(MV_IMPORT_PATH))
    try:
        from MvCameraControl_class import *
        from CameraParams_const import *
        from CameraParams_header import *
        from PixelType_header import *
        HAS_HIK_SDK = True
    except ImportError:
        HAS_HIK_SDK = False
else:
    HAS_HIK_SDK = False

from pcb_crop_model_engine import PCBYoloCropper, load_conveyor_config, save_conveyor_config, demo_file


class HikModelViewer:
    """
    Ứng dụng Camera Viewer chạy thử nghiệm mô hình YOLO OBB (pcb_crop.pt)
    """

    def __init__(self, default_exposure_us: float = 300000.0, conf: float = 0.05):
        self.default_exposure_us = default_exposure_us
        self.current_exposure_us = default_exposure_us
        self.exp_min = 1000.0
        self.exp_max = 1000000.0
        self.conf = conf

        self.cam = None
        self.device_name = "Camera Hikvision"
        self.is_connected = False
        self.is_grabbing = False

        self.latest_frame = None
        self.latest_raw_frame = None
        self.frame_lock = threading.Lock()
        self.grab_thread = None
        self.exit_event = threading.Event()

        self.fps_camera = 0.0
        self.frame_count = 0

        self.capture_dir = CURRENT_DIR / "captures"
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        self.toast_msg = "Sẵn sàng"
        self.toast_time = time.time()

        # Engine AI YOLO OBB
        self.cropper = PCBYoloCropper(conf=self.conf, imgsz=640)
        self.auto_crop_enabled = True  # Mặc định bật sẵn nhận diện AI
        self.show_live_crop_window = True  # Mở cửa sổ hiển thị trực tiếp PCB đã cắt
        self.cached_cropped = None  # Ảnh PCB đã cắt nắn thẳng mới nhất
        self.pcb_found = False
        self.cached_corners = None
        self.cached_status = "Chờ bo mạch"
        self.is_detecting = False

        # Bộ lọc băng tải
        self.conveyor_cfg = load_conveyor_config()
        self.filter_conveyor_enabled = bool(self.conveyor_cfg.get("filter_enabled", True))
        self.conveyor_zoom_view = False
        self.current_conveyor_roi = None

    def set_toast(self, msg, duration=2.5):
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def connect(self):
        """Khởi tạo SDK và kết nối tới camera Hikvision"""
        if not HAS_HIK_SDK:
            print("[LỖI] Không tìm thấy thư viện MvCameraControl SDK!")
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

        if st_device.nTLayerType == MV_GIGE_DEVICE:
            n_packet_size = self.cam.MV_CC_GetOptimalPacketSize()
            if int(n_packet_size) > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", n_packet_size)

        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self._init_exposure_limits()
        self.set_exposure(self.default_exposure_us)

        self.is_connected = True
        print("[+] Kết nối camera thành công!")
        return True

    def _init_exposure_limits(self):
        st_float = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", st_float)
        if ret == 0:
            self.exp_min = float(st_float.fMin)
            self.exp_max = float(st_float.fMax)
            self.current_exposure_us = float(st_float.fCurValue)

    def set_exposure(self, exposure_us: float):
        if not self.cam:
            return False
        clamped_us = max(self.exp_min, min(self.exp_max, exposure_us))
        self.cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", clamped_us)
        if ret == 0:
            self.current_exposure_us = clamped_us
            return True
        return False

    def start_stream(self):
        print("[3/4] Bắt đầu luồng nhận ảnh...")
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"[LỖI] StartGrabbing thất bại: 0x{ret:x}")
            return False

        self.is_grabbing = True
        self.exit_event.clear()
        self.grab_thread = threading.Thread(target=self._grab_worker, daemon=True)
        self.grab_thread.start()
        print("[4/4] Luồng thu hình đã khởi chạy.")
        return True

    def _grab_worker(self):
        st_frame_info = MV_FRAME_OUT_INFO_EX()
        buf_size = 4096 * 3072 * 3
        p_data = (c_ubyte * buf_size)()

        t_prev = time.time()
        frames_in_second = 0

        while not self.exit_event.is_set():
            ret = self.cam.MV_CC_GetOneFrameTimeout(p_data, buf_size, st_frame_info, 1000)
            if ret == 0:
                raw_w = st_frame_info.nWidth
                raw_h = st_frame_info.nHeight
                en_type = st_frame_info.enPixelType

                img_bgr = None
                if en_type == PixelType_Gvsp_Mono8:
                    gray = np.frombuffer(p_data, dtype=np.uint8, count=raw_w * raw_h).reshape((raw_h, raw_w))
                    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                elif en_type in (PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerBG8):
                    bayer = np.frombuffer(p_data, dtype=np.uint8, count=raw_w * raw_h).reshape((raw_h, raw_w))
                    code = cv2.COLOR_BayerRG2BGR if en_type == PixelType_Gvsp_BayerRG8 else cv2.COLOR_BayerBG2BGR
                    img_bgr = cv2.cvtColor(bayer, code)
                elif en_type in (PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed):
                    arr = np.frombuffer(p_data, dtype=np.uint8, count=raw_w * raw_h * 3).reshape((raw_h, raw_w, 3))
                    img_bgr = arr if en_type == PixelType_Gvsp_BGR8_Packed else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                else:
                    gray = np.frombuffer(p_data, dtype=np.uint8, count=raw_w * raw_h).reshape((raw_h, raw_w))
                    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                if img_bgr is not None:
                    with self.frame_lock:
                        self.latest_raw_frame = img_bgr
                        self.latest_frame = img_bgr.copy()

                    frames_in_second += 1
                    t_now = time.time()
                    if t_now - t_prev >= 1.0:
                        self.fps_camera = frames_in_second / (t_now - t_prev)
                        frames_in_second = 0
                        t_prev = t_now
            else:
                time.sleep(0.01)

    def capture_image(self):
        """Chụp ảnh 12MP: ảnh gốc, ảnh lòng băng tải, ảnh PCB nắn thẳng bằng YOLO OBB"""
        with self.frame_lock:
            if self.latest_raw_frame is None:
                self.set_toast("LỖI: Chưa có frame để chụp!", 3.0)
                return None
            frame_to_save = self.latest_raw_frame.copy()

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_str = f"{int(self.current_exposure_us)}us"
        filename = f"capture_{now_str}_exp_{exp_str}.jpg"
        save_path = self.capture_dir / filename

        cv2.imwrite(str(save_path), frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 98])
        print(f"[ĐÃ LƯU] Ảnh gốc 12MP: {save_path.name}")

        # Lưu ảnh lòng băng tải
        if self.filter_conveyor_enabled and self.cropper:
            conveyor_img, conv_roi = self.cropper.crop_conveyor(frame_to_save)
            conv_filename = f"conveyor_{now_str}_exp_{exp_str}.jpg"
            conv_path = self.capture_dir / conv_filename
            cv2.imwrite(str(conv_path), conveyor_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
            print(f"[ĐÃ LƯU] Ảnh lòng băng tải: {conv_filename}")

        # Nhận diện và cắt PCB bằng YOLO OBB
        if self.cropper:
            cropped_pcb, crop_info = self.cropper.process(
                frame_to_save,
                filter_conveyor=self.filter_conveyor_enabled,
                conf=self.conf
            )
            if crop_info.get("success") and cropped_pcb is not None:
                crop_filename = f"pcb_yolo_{now_str}_exp_{exp_str}.jpg"
                crop_path = self.capture_dir / crop_filename
                cv2.imwrite(str(crop_path), cropped_pcb, [cv2.IMWRITE_JPEG_QUALITY, 98])
                print(f"[ĐÃ LƯU] Ảnh PCB YOLO nắn thẳng: {crop_filename} ({cropped_pcb.shape[1]}x{cropped_pcb.shape[0]}px, Conf: {crop_info.get('confidence', 0)*100:.1f}%)")
                self.set_toast("Đã lưu: Gốc + Băng tải + PCB (YOLO OBB)!", 3.5)
                return save_path

        self.set_toast(f"Đã lưu: {filename}", 3.0)
        return save_path

    def stop(self):
        print("[DỌN DẸP] Đang dừng camera...")
        self.exit_event.set()
        if self.grab_thread and self.grab_thread.is_alive():
            self.grab_thread.join(timeout=1.0)
        if self.cam:
            if self.is_grabbing:
                self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
        if HAS_HIK_SDK:
            MvCamera.MV_CC_Finalize()
        print("[HOÀN TẤT] Camera đã đóng an toàn.")

    def run_gui(self):
        if not self.connect():
            return
        if not self.start_stream():
            self.stop()
            return

        window_name = f"PCB YOLO OBB Viewer - [{self.device_name}]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 966)

        crop_window_name = "PCB Da Cat (Model Live Crop)"
        cv2.namedWindow(crop_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(crop_window_name, 640, 420)

        # Trackbar phơi sáng
        def on_trackbar_exp(val):
            if val < 1: val = 1
            self.set_exposure(float(val) * 1000.0)

        init_ms = int(self.default_exposure_us / 1000.0)
        max_ms = int(min(1000.0, self.exp_max / 1000.0))
        cv2.createTrackbar("Exposure (ms)", window_name, init_ms, max_ms, on_trackbar_exp)

        # Trackbar ngưỡng tự tin YOLO Conf (%)
        def on_trackbar_conf(val):
            self.conf = max(0.02, float(val) / 100.0)
            if self.cropper:
                self.cropper.conf = self.conf

        cv2.createTrackbar("YOLO Conf (%)", window_name, int(self.conf * 100), 90, on_trackbar_conf)

        print("\n" + "=" * 68)
        print(" BẢN DÙNG THỬ MODEL YOLO OBB (pcb_crop.pt):")
        print(" - Phím [C]          : Bật / Tắt nhận diện AI YOLO OBB")
        print(" - Phím [W]          : Bật / Tắt cửa sổ Live Crop phụ")
        print(" - Phím [B]          : Bật / Tắt cắt lọc phần thừa ngoài băng tải")
        print(" - Phím [V]          : Chuyển đổi góc nhìn (Toàn cảnh <-> Zoom Băng tải)")
        print(" - Phím [1] / [2]    : Tăng / Giảm lề trái băng tải (+2% / -2%)")
        print(" - Phím [3] / [4]    : Tăng / Giảm lề phải băng tải (+2% / -2%)")
        print(" - Phím [S] / [SPACE]: Chụp ảnh full 12MP kèm PCB nắn thẳng bằng YOLO")
        print(" - Phím [Q] / [ESC]  : Thoát chương trình")
        print("=" * 68 + "\n")

        try:
            while True:
                with self.frame_lock:
                    display_img = self.latest_frame.copy() if self.latest_frame is not None else None

                if display_img is not None:
                    h, w = display_img.shape[:2]

                    if self.cropper:
                        conv_roi = self.cropper.get_conveyor_roi(display_img)
                        self.current_conveyor_roi = conv_roi
                        cx1, cy1, cx2, cy2 = conv_roi
                    else:
                        conv_roi = (0, 0, w, h)
                        cx1, cy1, cx2, cy2 = conv_roi

                    # AI Inference bất đồng bộ
                    if self.auto_crop_enabled and self.cropper:
                        if not self.is_detecting:
                            self.is_detecting = True
                            frame_for_ai = display_img.copy()

                            def _ai_worker(f):
                                try:
                                    cropped_pcb, info = self.cropper.process(
                                        f,
                                        return_debug=False,
                                        filter_conveyor=self.filter_conveyor_enabled,
                                        conf=self.conf
                                    )
                                    if info.get("success") and info.get("corners") is not None and cropped_pcb is not None:
                                        self.pcb_found = True
                                        self.cached_corners = info["corners"]
                                        self.cached_cropped = cropped_pcb
                                        self.cached_status = info.get("status_msg", "Đã bắt dính PCB")
                                    else:
                                        self.pcb_found = False
                                        self.cached_corners = None
                                        self.cached_cropped = None
                                        self.cached_status = info.get("status_msg", "Băng tải rỗng")
                                except Exception:
                                    pass
                                finally:
                                    self.is_detecting = False

                            threading.Thread(target=_ai_worker, args=(frame_for_ai,), daemon=True).start()

                    # Lấy biến snapshot an toàn luồng tránh Race Condition
                    cur_cropped = self.cached_cropped
                    cur_corners = self.cached_corners
                    cur_found = bool(self.pcb_found and cur_cropped is not None and cur_corners is not None)

                    # Render giao diện
                    target_w = 1280
                    if self.conveyor_zoom_view and self.filter_conveyor_enabled:
                        conv_sub = display_img[cy1:cy2, cx1:cx2]
                        c_h, c_w = conv_sub.shape[:2]
                        target_h = int(c_h * (target_w / c_w))
                        view = cv2.resize(conv_sub, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                        if cur_found and cur_corners is not None:
                            corners_sub = cur_corners.copy()
                            corners_sub[:, 0] -= cx1
                            corners_sub[:, 1] -= cy1
                            scale_x = target_w / c_w
                            scale_y = target_h / c_h
                            corners_disp = corners_sub * np.array([scale_x, scale_y])
                            pts_int = np.int32(corners_disp)
                            cv2.polylines(view, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3)
                            for pt in pts_int:
                                cv2.circle(view, tuple(pt), 6, (0, 200, 255), -1)
                    else:
                        target_h = int(h * (target_w / w))
                        view = cv2.resize(display_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                        # Phủ mờ lề ngoài băng tải
                        if self.filter_conveyor_enabled:
                            disp_cx1 = int(cx1 * target_w / w)
                            disp_cx2 = int(cx2 * target_w / w)
                            overlay = view.copy()
                            if disp_cx1 > 0:
                                cv2.rectangle(overlay, (0, 0), (disp_cx1, target_h), (10, 10, 10), -1)
                            if disp_cx2 < target_w:
                                cv2.rectangle(overlay, (disp_cx2, 0), (target_w, target_h), (10, 10, 10), -1)
                            cv2.addWeighted(overlay, 0.40, view, 0.60, 0, view)
                            cv2.line(view, (disp_cx1, 0), (disp_cx1, target_h), (0, 165, 255), 2)
                            cv2.line(view, (disp_cx2, 0), (disp_cx2, target_h), (0, 165, 255), 2)

                        # Vẽ hộp OBB
                        if cur_found and cur_corners is not None:
                            scale_factor = target_w / w
                            corners_disp = cur_corners * scale_factor
                            pts_int = np.int32(corners_disp)
                            cv2.polylines(view, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3)
                            for pt in pts_int:
                                cv2.circle(view, tuple(pt), 6, (0, 200, 255), -1)

                    # Khung xem trước PCB đã cắt (Picture-in-Picture) ở góc trên bên phải
                    # Khung xem trước PCB đã cắt (Picture-in-Picture) ở góc trên bên phải
                    pip_w, pip_h = 360, 220
                    pip_x = max(0, target_w - pip_w - 15)
                    pip_y = 15

                    if (cur_found and cur_cropped is not None and 
                            getattr(cur_cropped, 'size', 0) > 0 and 
                            cur_cropped.shape[0] > 10 and cur_cropped.shape[1] > 10):
                        ch, cw = cur_cropped.shape[:2]
                        if pip_y + pip_h <= target_h and pip_x + pip_w <= target_w:
                            pip_thumb = cv2.resize(cur_cropped, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
                            cv2.rectangle(view, (pip_x - 2, pip_y - 2), (pip_x + pip_w + 2, pip_y + pip_h + 2), (0, 255, 128), 2)
                            view[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = pip_thumb
                            cv2.rectangle(view, (pip_x, pip_y), (pip_x + pip_w, pip_y + 24), (20, 20, 20), -1)
                            cv2.putText(view, f"PCB DA CAT (AI): {cw}x{ch} px", (pip_x + 8, pip_y + 17),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA)
                    else:
                        if pip_y + pip_h <= target_h and pip_x + pip_w <= target_w:
                            cv2.rectangle(view, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (25, 25, 25), -1)
                            cv2.rectangle(view, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (70, 70, 70), 1)
                            cv2.putText(view, "CHO BO MACH PCB...", (pip_x + 75, pip_y + 115),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1, cv2.LINE_AA)

                    # HUD
                    self._draw_hud(view, w, h)
                    cv2.imshow(window_name, view)

                    # Cửa sổ phụ phóng to bo mạch PCB đã cắt
                    if self.show_live_crop_window:
                        if (cur_found and cur_cropped is not None and 
                                getattr(cur_cropped, 'size', 0) > 0 and 
                                cur_cropped.shape[0] > 10 and cur_cropped.shape[1] > 10):
                            crop_win_disp = cur_cropped.copy()
                            ch, cw = crop_win_disp.shape[:2]
                            header_bar = np.zeros((45, cw, 3), dtype=np.uint8)
                            cv2.putText(header_bar, f"PCB DA CAT (YOLO AI) - Kich thuoc: {cw}x{ch} px", (15, 28),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)
                            full_crop_view = np.vstack([header_bar, crop_win_disp])
                            disp_h = 420
                            disp_w = int(full_crop_view.shape[1] * disp_h / max(1, full_crop_view.shape[0]))
                            cv2.imshow(crop_window_name, cv2.resize(full_crop_view, (disp_w, disp_h), interpolation=cv2.INTER_AREA))
                        else:
                            empty_win = np.zeros((420, 640, 3), dtype=np.uint8)
                            cv2.putText(empty_win, "BANG TAI RONG", (200, 190),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 200, 255), 2, cv2.LINE_AA)
                            cv2.putText(empty_win, "Dang cho bo mach PCB tren bang tai...", (115, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1, cv2.LINE_AA)
                            cv2.imshow(crop_window_name, empty_win)
                else:
                    waiting = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(waiting, "Dang ket noi camera Hikvision...", (400, 360),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.imshow(window_name, waiting)

                key = cv2.waitKey(20) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('w'), ord('W')):
                    self.show_live_crop_window = not self.show_live_crop_window
                    if not self.show_live_crop_window:
                        try:
                            cv2.destroyWindow(crop_window_name)
                        except Exception:
                            pass
                    state = "BAT" if self.show_live_crop_window else "TAT"
                    self.set_toast(f"Cua so Live Crop: {state}", 2.0)
                elif key in (ord('c'), ord('C')):
                    self.auto_crop_enabled = not self.auto_crop_enabled
                    state = "BAT" if self.auto_crop_enabled else "TAT"
                    self.set_toast(f"YOLO OBB: {state}", 2.0)
                elif key in (ord('b'), ord('B')):
                    self.filter_conveyor_enabled = not self.filter_conveyor_enabled
                    self.conveyor_cfg["filter_enabled"] = self.filter_conveyor_enabled
                    save_conveyor_config(self.conveyor_cfg)
                    state = "BAT" if self.filter_conveyor_enabled else "TAT"
                    self.set_toast(f"Cat le bang tai: {state}", 2.0)
                elif key in (ord('v'), ord('V')):
                    self.conveyor_zoom_view = not self.conveyor_zoom_view
                    name = "ZOOM BANG TAI" if self.conveyor_zoom_view else "TOAN CANH"
                    self.set_toast(f"Goc nhin: {name}", 2.0)
                elif key == ord('1'):
                    cur_l = self.conveyor_cfg.get("margin_left", 0.28)
                    self.conveyor_cfg["margin_left"] = round(min(0.45, cur_l + 0.02), 2)
                    self.conveyor_cfg["auto_detect"] = False
                    save_conveyor_config(self.conveyor_cfg)
                    self.set_toast(f"Le trai: {self.conveyor_cfg['margin_left']*100:.0f}%", 1.5)
                elif key == ord('2'):
                    cur_l = self.conveyor_cfg.get("margin_left", 0.28)
                    self.conveyor_cfg["margin_left"] = round(max(0.0, cur_l - 0.02), 2)
                    self.conveyor_cfg["auto_detect"] = False
                    save_conveyor_config(self.conveyor_cfg)
                    self.set_toast(f"Le trai: {self.conveyor_cfg['margin_left']*100:.0f}%", 1.5)
                elif key == ord('3'):
                    cur_r = self.conveyor_cfg.get("margin_right", 0.06)
                    self.conveyor_cfg["margin_right"] = round(min(0.35, cur_r + 0.02), 2)
                    self.conveyor_cfg["auto_detect"] = False
                    save_conveyor_config(self.conveyor_cfg)
                    self.set_toast(f"Le phai: {self.conveyor_cfg['margin_right']*100:.0f}%", 1.5)
                elif key == ord('4'):
                    cur_r = self.conveyor_cfg.get("margin_right", 0.06)
                    self.conveyor_cfg["margin_right"] = round(max(0.0, cur_r - 0.02), 2)
                    self.conveyor_cfg["auto_detect"] = False
                    save_conveyor_config(self.conveyor_cfg)
                    self.set_toast(f"Le phai: {self.conveyor_cfg['margin_right']*100:.0f}%", 1.5)
                elif key in (ord('+'), ord('=')):
                    self.set_exposure(self.current_exposure_us + 10000.0)
                    cv2.setTrackbarPos("Exposure (ms)", window_name, int(self.current_exposure_us / 1000.0))
                elif key in (ord('-'), ord('_')):
                    self.set_exposure(max(1000.0, self.current_exposure_us - 10000.0))
                    cv2.setTrackbarPos("Exposure (ms)", window_name, int(self.current_exposure_us / 1000.0))
                elif key in (ord('s'), ord('S'), 32):
                    self.capture_image()

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()
            self.stop()

    def _draw_hud(self, view, raw_w, raw_h):
        vh, vw = view.shape[:2]
        overlay = view.copy()
        cv2.rectangle(overlay, (10, 10), (490, 200), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, view, 0.3, 0, view)
        cv2.rectangle(view, (10, 10), (490, 200), (0, 255, 0), 1)

        exp_ms = self.current_exposure_us / 1000.0
        cv2.putText(view, f"MODEL: YOLO OBB (pcb_crop.pt)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(view, f"Camera: {self.device_name} (12 MP)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(view, f"Thoi gian sang: {self.current_exposure_us:,.0f} us ({exp_ms:.1f} ms)", (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(view, f"FPS: {self.fps_camera:.1f} | Nguong Conf: {self.conf*100:.0f}%", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 50), 1, cv2.LINE_AA)

        if not self.auto_crop_enabled:
            status_text = "TAT"
            status_color = (180, 180, 180)
        else:
            status_text = self.cached_status.upper()
            status_color = (0, 255, 0) if self.pcb_found else (0, 200, 255)

        cv2.putText(view, f"YOLO OBB [C]: {status_text}", (20, 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2 if self.pcb_found else 1, cv2.LINE_AA)

        conv_status = "BAT (Da loc phan thua)" if self.filter_conveyor_enabled else "TAT"
        view_mode = "ZOOM BANG TAI" if self.conveyor_zoom_view else "TOAN CANH"
        cv2.putText(view, f"Bang tai [B]: {conv_status} | [V]: {view_mode}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.putText(view, "[C]: YOLO Crop | [B]: Cat bang tai | [V]: Zoom | [1-4]: Le | [S]: Chup | [Q]: Thoat",
                    (15, vh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        if time.time() < self.toast_time:
            tw = len(self.toast_msg) * 12 + 40
            cv2.rectangle(view, (vw // 2 - tw // 2, 20), (vw // 2 + tw // 2, 65), (0, 150, 0), -1)
            cv2.rectangle(view, (vw // 2 - tw // 2, 20), (vw // 2 + tw // 2, 65), (255, 255, 255), 1)
            cv2.putText(view, self.toast_msg, (vw // 2 - tw // 2 + 15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    img_arg = None
    for arg in sys.argv[1:]:
        if any(arg.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp')):
            img_arg = arg
            break
    if img_arg:
        demo_file(img_arg)
    else:
        viewer = HikModelViewer(default_exposure_us=300000.0)
        viewer.run_gui()

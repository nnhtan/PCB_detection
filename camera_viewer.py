# -*- coding: utf-8 -*-
"""
Chương trình hiển thị và điều khiển Camera Công nghiệp Hikvision (Hikrobot)
- Tự động nhận diện camera GigE / USB (Model: MV-CE120-10GM...)
- Điều chỉnh thời gian lấy sáng (Exposure Time) linh hoạt, mặc định 300,000 µs (300 ms)
- Hỗ trợ thanh kéo (Trackbar) và phím tắt (+ / - / [ / ])
- Chụp ảnh full độ phân giải 12MP (4024x3036) lưu vào thư mục captures/
"""

import os
import sys
import time
import threading
from datetime import datetime
from ctypes import *
from pathlib import Path

# Cấu hình UTF-8 cho console Windows tránh lỗi UnicodeEncodeError
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

# Import module thuật toán cắt bo mạch PCB tối ưu
try:
    from pcb_crop_engine import PCBMonoCropper
except ImportError:
    PCBMonoCropper = None

# Đường dẫn đến module MvImport của Hikvision SDK
CURRENT_DIR = Path(__file__).resolve().parent
MV_IMPORT_PATH = CURRENT_DIR / "Python" / "MvImport"
if not MV_IMPORT_PATH.exists():
    MV_IMPORT_PATH = CURRENT_DIR.parent / "Python" / "MvImport"

sys.path.append(str(MV_IMPORT_PATH))

try:
    from MvCameraControl_class import *
    from PixelType_header import *
except ImportError as e:
    print(f"[LỖI] Không thể nạp Hikvision SDK (MvImport) từ: {MV_IMPORT_PATH}")
    print(f"Chi tiết: {e}")
    if __name__ == "__main__":
        sys.exit(1)
    raise ImportError(
        f"Không thể nạp Hikvision SDK từ {MV_IMPORT_PATH}: {e}"
    ) from e


class HikCameraViewer:
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
        self.pixel_type = 0

        # Quản lý luồng lấy ảnh
        self.grab_thread = None
        self.exit_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_raw_frame = None  # Ảnh gốc full phân giải cho chụp ảnh
        self.fps_camera = 0.0
        self.frame_count = 0

        # Thư mục lưu ảnh chụp
        self.capture_dir = CURRENT_DIR / "captures"
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        # Thông báo trạng thái hiển thị trên màn hình
        self.toast_msg = "Sẵn sàng"
        self.toast_time = time.time()

        # Bộ bóc tách bo mạch PCB
        self.cropper = PCBMonoCropper() if PCBMonoCropper else None
        self.auto_crop_enabled = False
        self.last_crop_info = None
        self.pcb_found = False
        self.cached_corners = None
        self.cached_status = "Chờ bo mạch"
        self.is_detecting = False

    def set_toast(self, msg, duration=2.5):
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def connect(self):
        """Khởi tạo SDK và kết nối tới camera đầu tiên tìm thấy"""
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

        # Lấy tên Model thiết bị
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

        # Tối ưu kích thước gói cho GigE
        if st_device.nTLayerType == MV_GIGE_DEVICE:
            pkt_size = self.cam.MV_CC_GetOptimalPacketSize()
            if pkt_size > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt_size)

        # Đọc thông số min/max của ExposureTime
        exp_val = MVCC_FLOATVALUE()
        if self.cam.MV_CC_GetFloatValue("ExposureTime", exp_val) == 0:
            self.exp_min = exp_val.fMin
            self.exp_max = exp_val.fMax
            print(f"      -> Giới hạn Exposure: {self.exp_min:.0f} µs ~ {self.exp_max:.0f} µs")

        # Cài đặt phơi sáng thủ công và gán giá trị mặc định 300,000 µs
        print(f"[3/4] Cài đặt thời gian lấy sáng mặc định: {self.default_exposure_us:.0f} µs...")
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # 0: Off
        self.set_exposure(self.default_exposure_us)

        # Lấy kích thước ảnh
        w_val = MVCC_INTVALUE()
        h_val = MVCC_INTVALUE()
        self.cam.MV_CC_GetIntValue("Width", w_val)
        self.cam.MV_CC_GetIntValue("Height", h_val)
        self.resolution = (w_val.nCurValue, h_val.nCurValue)
        print(f"      -> Độ phân giải gốc: {self.resolution[0]} x {self.resolution[1]}")

        self.is_connected = True
        return True

    def set_exposure(self, exposure_us):
        """Cập nhật thời gian phơi sáng (micro-giây)"""
        if not self.cam:
            return False
        # Giới hạn trong khoảng an toàn
        clamped = max(self.exp_min, min(self.exp_max, float(exposure_us)))
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", clamped)
        if ret == 0:
            self.current_exposure_us = clamped
            ms = self.current_exposure_us / 1000.0
            print(f"[Camera] Đã cập nhật ExposureTime: {self.current_exposure_us:.0f} µs ({ms:.1f} ms)")
            self.set_toast(f"Exposure: {self.current_exposure_us:.0f} us ({ms:.1f} ms)")
            return True
        else:
            print(f"[CẢNH BÁO] Không thể đặt ExposureTime ({clamped}): 0x{ret:x}")
            return False

    def start_stream(self):
        """Khởi động lấy luồng ảnh liên tục trong background thread"""
        if not self.is_connected:
            return False

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"[LỖI] MV_CC_StartGrabbing thất bại: 0x{ret:x}")
            return False

        self.is_grabbing = True
        self.exit_event.clear()
        self.grab_thread = threading.Thread(target=self._grab_worker, daemon=True)
        self.grab_thread.start()
        print("[4/4] Bắt đầu luồng ảnh thành công!")
        return True

    def _grab_worker(self):
        """Thread liên tục nhận frame từ camera và giải mã sang định dạng hiển thị"""
        st_out = MV_FRAME_OUT()
        memset(byref(st_out), 0, sizeof(st_out))

        last_fps_calc = time.time()
        frames_in_second = 0

        # Buffer chuyển đổi màu nếu cần
        bgr_buf = None
        last_bgr_size = 0

        while not self.exit_event.is_set():
            # Chờ lấy frame với timeout 2000ms (cho phơi sáng dài)
            ret = self.cam.MV_CC_GetImageBuffer(st_out, 2000)
            if ret == 0 and st_out.pBufAddr:
                w = st_out.stFrameInfo.nWidth
                h = st_out.stFrameInfo.nHeight
                pixel_type = st_out.stFrameInfo.enPixelType
                frame_len = st_out.stFrameInfo.nFrameLen

                img_bgr = None
                # Trường hợp ảnh Mono8 (như MV-CE120-10GM)
                if pixel_type == PixelType_Gvsp_Mono8:
                    raw_bytes = string_at(st_out.pBufAddr, frame_len)
                    mono_arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w))
                    img_bgr = cv2.cvtColor(mono_arr, cv2.COLOR_GRAY2BGR)
                else:
                    # Các chuẩn khác (Bayer, RGB...): dùng hàm convert của SDK
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

                # Giải phóng buffer camera
                self.cam.MV_CC_FreeImageBuffer(st_out)

                if img_bgr is not None:
                    with self.frame_lock:
                        self.latest_raw_frame = img_bgr
                        self.latest_frame = img_bgr.copy()
                        self.resolution = (w, h)
                    self.frame_count += 1
                    frames_in_second += 1

                # Tính FPS thực tế của Camera
                now = time.time()
                if now - last_fps_calc >= 1.0:
                    self.fps_camera = frames_in_second / (now - last_fps_calc)
                    frames_in_second = 0
                    last_fps_calc = now
            else:
                time.sleep(0.01)

    def capture_image(self):
        """Chụp và lưu ảnh gốc chất lượng cao full 12MP (và ảnh PCB đã cắt nắn thẳng nếu có)"""
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
        print(f"[ĐÃ LƯU] Ảnh gốc: {save_path.name} ({frame_to_save.shape[1]}x{frame_to_save.shape[0]})")

        # Nếu đang bật chế độ nhận diện PCB hoặc có cropper
        if self.cropper:
            cropped_pcb, crop_info = self.cropper.process(frame_to_save)
            if crop_info.get("success") and cropped_pcb is not None:
                crop_filename = f"pcb_cropped_{now_str}_exp_{exp_str}.jpg"
                crop_path = self.capture_dir / crop_filename
                cv2.imwrite(str(crop_path), cropped_pcb, [cv2.IMWRITE_JPEG_QUALITY, 98])
                print(f"[ĐÃ LƯU] Ảnh PCB nắn thẳng: {crop_filename} ({cropped_pcb.shape[1]}x{cropped_pcb.shape[0]})")
                self.set_toast(f"Đã lưu ảnh gốc & PCB nắn thẳng!", 3.5)
                return save_path

        self.set_toast(f"Đã lưu: {filename}", 3.5)
        return save_path

    def stop(self):
        """Dọn dẹp và ngắt kết nối camera"""
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

        MvCamera.MV_CC_Finalize()
        self.is_connected = False
        self.is_grabbing = False
        print("[HOÀN TẤT] Camera đã đóng an toàn.")

    def run_gui(self):
        """Vòng lặp hiển thị giao diện OpenCV với thanh trượt và phím tắt"""
        if not self.connect():
            return
        if not self.start_stream():
            self.stop()
            return

        window_name = f"Hikvision Camera - [{self.device_name}]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 966)

        # Callback khi kéo trackbar (đơn vị: ms)
        def on_trackbar(val):
            # val là mili-giây (1 ~ 1000)
            if val < 1:
                val = 1
            exposure_us = float(val) * 1000.0
            self.set_exposure(exposure_us)

        init_ms = int(self.default_exposure_us / 1000.0)
        max_ms = int(min(1000.0, self.exp_max / 1000.0))
        cv2.createTrackbar("Exposure (ms)", window_name, init_ms, max_ms, on_trackbar)

        print("\n" + "=" * 60)
        print(" HƯỚNG DẪN ĐIỀU KHIỂN:")
        print(" - Kéo thanh trượt 'Exposure (ms)' trên cửa sổ")
        print(" - Phím [+] hoặc [=] : Tăng +10 ms (+10,000 µs)")
        print(" - Phím [-] hoặc [_] : Giảm -10 ms (-10,000 µs)")
        print(" - Phím [ ] ]        : Tăng +50 ms (+50,000 µs)")
        print(" - Phím [ [ ]        : Giảm -50 ms (-50,000 µs)")
        print(" - Phím [R]          : Reset về mặc định 300,000 µs (300 ms)")
        print(" - Phím [C]          : Bật / Tắt nhận diện & cắt bo mạch PCB tự động")
        print(" - Phím [S] / [SPACE]: Chụp ảnh full 12MP (kèm ảnh PCB nắn thẳng)")
        print(" - Phím [Q] / [ESC]  : Thoát chương trình")
        print("=" * 60 + "\n")

        try:
            while True:
                with self.frame_lock:
                    if self.latest_frame is not None:
                        display_img = self.latest_frame.copy()
                    else:
                        display_img = None

                if display_img is not None:
                    # Thu nhỏ để hiển thị mượt mà trên màn hình
                    h, w = display_img.shape[:2]
                    target_w = 1280
                    target_h = int(h * (target_w / w))
                    view = cv2.resize(display_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                    # Nếu bật chế độ nhận diện bo mạch PCB (chạy AI bất đồng bộ để giữ UI 60 FPS)
                    if self.auto_crop_enabled and self.cropper:
                        # Kích hoạt luồng quét AI ngầm nếu không bận
                        if not self.is_detecting:
                            self.is_detecting = True
                            frame_for_ai = display_img.copy()

                            def _ai_worker(f):
                                try:
                                    _, info = self.cropper.process(f, return_debug=False)
                                    self.last_crop_info = info
                                    if info.get("success") and info.get("corners") is not None:
                                        self.pcb_found = True
                                        self.cached_corners = info["corners"]
                                        self.cached_status = info.get("status_msg", "Đã bắt dính PCB")
                                    else:
                                        self.pcb_found = False
                                        self.cached_corners = None
                                        self.cached_status = info.get("status_msg", "Băng tải rỗng")
                                except Exception as e:
                                    pass
                                finally:
                                    self.is_detecting = False

                            threading.Thread(target=_ai_worker, args=(frame_for_ai,), daemon=True).start()

                        # Vẽ khung định vị PCB lên màn hình xem thử (nếu tìm thấy)
                        if self.pcb_found and self.cached_corners is not None:
                            scale_factor = target_w / w
                            corners_disp = self.cached_corners * scale_factor
                            pts_int = np.int32(corners_disp)
                            # Vẽ viền bo mạch xanh neon bám dính
                            cv2.polylines(view, [pts_int], isClosed=True, color=(0, 255, 128), thickness=2)
                            for pt in pts_int:
                                cv2.circle(view, tuple(pt), 6, (0, 200, 255), -1)

                    # Vẽ thanh thông số (HUD)
                    self._draw_hud(view, w, h)
                    cv2.imshow(window_name, view)
                else:
                    # Chờ frame đầu tiên
                    waiting_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(waiting_screen, "Dang nhan hinh anh tu camera...", (400, 360),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.imshow(window_name, waiting_screen)

                # Bắt phím điều khiển
                key = cv2.waitKey(20) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # 27 là ESC
                    break
                elif key in (ord('c'), ord('C')):
                    self.auto_crop_enabled = not self.auto_crop_enabled
                    state_str = "BẬT" if self.auto_crop_enabled else "TẮT"
                    print(f"[PCB] Chế độ nhận diện bo mạch: {state_str}")
                    self.set_toast(f"Nhan dien PCB: {state_str}", 2.0)
                elif key in (ord('+'), ord('=')):
                    new_exp = self.current_exposure_us + 10000.0
                    self.set_exposure(new_exp)
                    cv2.setTrackbarPos("Exposure (ms)", window_name, int(self.current_exposure_us / 1000.0))
                elif key in (ord('-'), ord('_')):
                    new_exp = max(1000.0, self.current_exposure_us - 10000.0)
                    self.set_exposure(new_exp)
                    cv2.setTrackbarPos("Exposure (ms)", window_name, int(self.current_exposure_us / 1000.0))
                elif key == ord(']'):
                    new_exp = self.current_exposure_us + 50000.0
                    self.set_exposure(new_exp)
                    cv2.setTrackbarPos("Exposure (ms)", window_name, int(self.current_exposure_us / 1000.0))
                elif key == ord('['):
                    new_exp = max(1000.0, self.current_exposure_us - 50000.0)
                    self.set_exposure(new_exp)
                    cv2.setTrackbarPos("Exposure (ms)", window_name, int(self.current_exposure_us / 1000.0))
                elif key in (ord('r'), ord('R')):
                    self.set_exposure(self.default_exposure_us)
                    cv2.setTrackbarPos("Exposure (ms)", window_name, int(self.default_exposure_us / 1000.0))
                elif key in (ord('s'), ord('S'), 32):  # 32 là phím Space
                    self.capture_image()

                # Kiểm tra nếu người dùng đóng cửa sổ bằng nút X
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

        finally:
            cv2.destroyAllWindows()
            self.stop()

    def _draw_hud(self, view, raw_w, raw_h):
        """Vẽ bảng thông số kỹ thuật và hướng dẫn lên màn hình xem thử"""
        vh, vw = view.shape[:2]

        # Lớp nền mờ góc trên bên trái
        overlay = view.copy()
        cv2.rectangle(overlay, (10, 10), (450, 140), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, view, 0.3, 0, view)

        # Viền góc thông tin
        cv2.rectangle(view, (10, 10), (450, 165), (0, 200, 255), 1)

        # Chữ thông tin
        exp_ms = self.current_exposure_us / 1000.0
        cv2.putText(view, f"Camera: {self.device_name}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(view, f"Do phan giai: {raw_w} x {raw_h} (12 MP)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(view, f"Thoi gian sang: {self.current_exposure_us:,.0f} us ({exp_ms:.1f} ms)", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(view, f"FPS Camera: {self.fps_camera:.1f} fps", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 50), 1, cv2.LINE_AA)

        if not self.auto_crop_enabled:
            crop_status = "TAT"
            crop_color = (180, 180, 180)
        else:
            if self.pcb_found:
                crop_status = self.cached_status.upper()
                crop_color = (0, 255, 128)
            else:
                crop_status = self.cached_status.upper()
                crop_color = (0, 200, 255)

        cv2.putText(view, f"Auto-Crop PCB [C]: {crop_status}", (20, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, crop_color, 2 if self.auto_crop_enabled else 1, cv2.LINE_AA)

        # Hướng dẫn nhanh ở góc dưới
        cv2.putText(view, "[C]: Bat/Tat Crop | [+/-]: Chinh 10ms | [S]: Chup anh | [R]: Reset | [Q]: Thoat",
                    (15, vh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Thông báo Toast nếu có
        if time.time() < self.toast_time:
            tw = len(self.toast_msg) * 12 + 40
            cv2.rectangle(view, (vw // 2 - tw // 2, 20), (vw // 2 + tw // 2, 65), (0, 150, 0), -1)
            cv2.rectangle(view, (vw // 2 - tw // 2, 20), (vw // 2 + tw // 2, 65), (255, 255, 255), 1)
            cv2.putText(view, self.toast_msg, (vw // 2 - tw // 2 + 15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    viewer = HikCameraViewer(default_exposure_us=300000.0)
    viewer.run_gui()

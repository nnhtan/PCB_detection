"""
Module Quản lý & Quét Nguồn Camera Hệ Thống (Camera Manager)
Hỗ trợ:
1. Camera công nghiệp Hikvision (GigE & USB3 Vision qua MvCamera SDK)
2. Camera USB / Webcam / UVC ngoài qua OpenCV DirectShow
3. File ảnh / video mẫu thử nghiệm
4. Hộp thoại GUI chọn camera trực quan (Tkinter Dark Mode)
5. Menu chọn nhanh trên màn hình OSD và Terminal
6. Lưu trữ và ghi nhớ nguồn camera ưa thích (camera_source_config.json)
"""

import os
import sys
import io
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

# Đảm bảo in tiếng Việt không bị lỗi charmap trên console Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np

# Đường dẫn thư mục hiện tại và SDK Hikvision
CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = CURRENT_DIR / "camera_source_config.json"
MV_IMPORT_PATH = CURRENT_DIR / "Python" / "MvImport"
if not MV_IMPORT_PATH.exists():
    MV_IMPORT_PATH = CURRENT_DIR.parent / "Python" / "MvImport"

if MV_IMPORT_PATH.exists():
    sys.path.insert(0, str(MV_IMPORT_PATH))

try:
    from MvCameraControl_class import *
    from PixelType_header import *
    HAS_HIK_SDK = True
except Exception:
    HAS_HIK_SDK = False


class SuppressStderr:
    """Tắt stderr ở mức file descriptor C để ẩn các cảnh báo nội bộ của OpenCV/DirectShow"""
    def __enter__(self):
        try:
            sys.stderr.flush()
            self.null_fd = os.open("nul", os.O_WRONLY)
            self.orig_fd = os.dup(2)
            os.dup2(self.null_fd, 2)
            os.close(self.null_fd)
            self.active = True
        except Exception:
            self.active = False
        return self

    def __exit__(self, *args):
        if getattr(self, "active", False):
            try:
                os.dup2(self.orig_fd, 2)
                os.close(self.orig_fd)
            except Exception:
                pass


@dataclass
class CameraSource:
    source_type: str        # 'hikvision', 'opencv', 'file'
    source_id: Any          # int (0, 1) or str (file path)
    name: str               # Tên ngắn hiển thị
    details: str            # Chi tiết kỹ thuật (IP, S/N, Độ phân giải)
    resolution: Tuple[int, int] = (0, 0)
    badge: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "name": self.name,
            "details": self.details,
            "resolution": list(self.resolution),
            "badge": self.badge
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CameraSource":
        return cls(
            source_type=str(d.get("source_type", "opencv")),
            source_id=d.get("source_id", 0),
            name=str(d.get("name", "Camera")),
            details=str(d.get("details", "")),
            resolution=tuple(d.get("resolution", (0, 0))),
            badge=str(d.get("badge", ""))
        )

    def matches(self, other: Optional["CameraSource"]) -> bool:
        if other is None:
            return False
        return (self.source_type == other.source_type and 
                str(self.source_id) == str(other.source_id))


def load_camera_config() -> Optional[Dict[str, Any]]:
    """Tải cấu hình nguồn camera đã lưu gần nhất"""
    if CONFIG_FILE.is_file():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_camera_config(source: CameraSource) -> bool:
    """Lưu nguồn camera được chọn để tự động kết nối lần sau"""
    try:
        cfg = source.to_dict()
        cfg["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[Cảnh báo] Không thể lưu cấu hình camera: {e}")
        return False


def scan_hikvision_cameras() -> List[CameraSource]:
    """Quét toàn bộ camera công nghiệp Hikvision (GigE & USB3 Vision)"""
    sources = []
    if not HAS_HIK_SDK:
        return sources

    try:
        ret = MvCamera.MV_CC_Initialize()
        dev_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, dev_list)
        if ret == 0 and dev_list.nDeviceNum > 0:
            for i in range(dev_list.nDeviceNum):
                st_device = cast(dev_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                if st_device.nTLayerType == MV_GIGE_DEVICE:
                    gige = st_device.SpecialInfo.stGigEInfo
                    model = bytes(gige.chModelName).split(b"\x00")[0].decode("ascii", errors="ignore").strip() or f"Hikvision GigE #{i}"
                    serial = bytes(gige.chSerialNumber).split(b"\x00")[0].decode("ascii", errors="ignore").strip()
                    user_name = bytes(gige.chUserDefinedName).split(b"\x00")[0].decode("ascii", errors="ignore").strip()
                    nip1 = (gige.nCurrentIp & 0xff000000) >> 24
                    nip2 = (gige.nCurrentIp & 0x00ff0000) >> 16
                    nip3 = (gige.nCurrentIp & 0x0000ff00) >> 8
                    nip4 = gige.nCurrentIp & 0x000000ff
                    ip_str = f"{nip1}.{nip2}.{nip3}.{nip4}"
                    name = f"[Hikvision GigE #{i}] {model}"
                    details = f"IP: {ip_str} | S/N: {serial}"
                    if user_name:
                        details += f" ({user_name})"
                    sources.append(CameraSource(
                        source_type="hikvision",
                        source_id=i,
                        name=name,
                        details=details,
                        resolution=(4024, 3036),  # Mặc định dòng 12MP phổ biến
                        badge="🏭 Hikvision GigE"
                    ))
                elif st_device.nTLayerType == MV_USB_DEVICE:
                    usb = st_device.SpecialInfo.stUsb3VInfo
                    model = bytes(usb.chModelName).split(b"\x00")[0].decode("ascii", errors="ignore").strip() or f"Hikvision USB #{i}"
                    serial = bytes(usb.chSerialNumber).split(b"\x00")[0].decode("ascii", errors="ignore").strip()
                    user_name = bytes(usb.chUserDefinedName).split(b"\x00")[0].decode("ascii", errors="ignore").strip()
                    name = f"[Hikvision USB #{i}] {model}"
                    details = f"S/N: {serial}"
                    if user_name:
                        details += f" ({user_name})"
                    sources.append(CameraSource(
                        source_type="hikvision",
                        source_id=i,
                        name=name,
                        details=details,
                        badge="🏭 Hikvision USB"
                    ))
    except Exception as e:
        print(f"[Cảnh báo] Lỗi khi quét Hikvision camera: {e}")

    return sources


def scan_opencv_cameras(max_indices: int = 4) -> List[CameraSource]:
    """Quét camera USB / Webcam ngoài qua OpenCV DirectShow (đã ẩn cảnh báo C runtime)"""
    sources = []
    with SuppressStderr():
        for idx in range(max_indices):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
                    name = f"[Webcam USB #{idx}] Camera {idx}"
                    details = f"Độ phân giải: {w} x {h} px"
                    sources.append(CameraSource(
                        source_type="opencv",
                        source_id=idx,
                        name=name,
                        details=details,
                        resolution=(w, h),
                        badge="📷 Webcam USB"
                    ))
                cap.release()
    return sources


def scan_sample_files(base_dir: Optional[Path] = None) -> List[CameraSource]:
    """Tìm các file ảnh/video mẫu trong thư mục làm việc"""
    if base_dir is None:
        base_dir = CURRENT_DIR

    sources = []
    sample_names = ["result_Image_20261019181609711.jpg", "a.jpg", "2.jpg", "3.jpg", "boats.jpg"]
    for s_name in sample_names:
        p = base_dir / s_name
        if p.is_file():
            try:
                img = cv2.imread(str(p))
                if img is not None:
                    h, w = img.shape[:2]
                    sources.append(CameraSource(
                        source_type="file",
                        source_id=str(p),
                        name=f"[Ảnh mẫu] {p.name}",
                        details=f"Kích thước: {w} x {h} px ({p.stat().st_size // 1024} KB)",
                        resolution=(w, h),
                        badge="🖼️ Ảnh mẫu"
                    ))
            except Exception:
                pass
    return sources


def scan_all_camera_sources(base_dir: Optional[Path] = None, include_samples: bool = True) -> List[CameraSource]:
    """
    Quét toàn bộ nguồn camera:
    1. Hikvision Industrial Camera (GigE & USB3)
    2. USB Webcams
    3. Ảnh / Video mẫu
    """
    all_sources = []
    # 1. Hikvision Industrial Cameras
    all_sources.extend(scan_hikvision_cameras())
    # 2. USB Webcams
    all_sources.extend(scan_opencv_cameras(max_indices=5))
    # 3. File mẫu (nếu được bật)
    if include_samples:
        all_sources.extend(scan_sample_files(base_dir or CURRENT_DIR))

    return all_sources


def parse_camera_source_arg(arg: str, available_sources: Optional[List[CameraSource]] = None) -> Optional[CameraSource]:
    """Phân tích tham số dòng lệnh thành CameraSource"""
    if not arg:
        return None

    arg_lower = arg.lower().strip()

    # Kiểm tra theo prefix hik: / usb: / file:
    if arg_lower.startswith("hik:"):
        idx_str = arg_lower.split(":", 1)[1]
        idx = int(idx_str) if idx_str.isdigit() else 0
        return CameraSource("hikvision", idx, f"[Hikvision] Thiết bị #{idx}", f"Chỉ định dòng lệnh: {arg}", badge="🏭 Hikvision")
    elif arg_lower.startswith("usb:") or arg_lower.startswith("cam:"):
        idx_str = arg_lower.split(":", 1)[1]
        idx = int(idx_str) if idx_str.isdigit() else 0
        return CameraSource("opencv", idx, f"[Webcam USB] Camera {idx}", f"Chỉ định dòng lệnh: {arg}", badge="📷 Webcam USB")
    elif arg_lower.startswith("file:"):
        filepath = arg[5:].strip()
        p = Path(filepath)
        return CameraSource("file", str(p), f"[File] {p.name}", str(p), badge="🖼️ File")

    # Nếu truyền số nguyên thuần (ví dụ '0', '1', '2')
    if arg.isdigit():
        idx = int(arg)
        return CameraSource("opencv", idx, f"[Webcam USB] Camera {idx}", f"Cổng OpenCV #{idx}", badge="📷 Webcam USB")

    # Nếu truyền đường dẫn file ảnh hoặc video
    p = Path(arg)
    if p.is_file():
        return CameraSource("file", str(p), f"[File] {p.name}", str(p), badge="🖼️ File")

    # So khớp với danh sách có sẵn
    if available_sources:
        for s in available_sources:
            if arg_lower in s.name.lower() or arg_lower in s.details.lower():
                return s

    return None


def show_camera_picker_dialog(
    sources: List[CameraSource],
    current_source: Optional[CameraSource] = None,
    timeout_sec: int = 0
) -> Optional[CameraSource]:
    """
    Hiển thị hộp thoại GUI chọn nguồn camera hiện đại (Tkinter Dark Mode).
    Hỗ trợ đếm ngược tự động kết nối nếu timeout_sec > 0.
    """
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog
    except ImportError:
        print("[LỖI] Không thể nạp thư viện Tkinter. Chuyển sang chế độ chọn bằng Terminal.")
        return terminal_camera_picker(sources, current_source, timeout_sec)

    selected_source: Optional[CameraSource] = None
    countdown_active = (timeout_sec > 0)
    remaining_time = timeout_sec

    root = tk.Tk()
    root.title("Quét & Chọn Nguồn Camera - PCB Vision AI")
    root.geometry("740x510")
    root.minsize(680, 440)
    root.configure(bg="#18181f")

    # Căn giữa màn hình
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - 740) // 2
    y = (sh - 510) // 2
    root.geometry(f"+{x}+{y}")

    # Header
    header_frame = tk.Frame(root, bg="#22222c", pady=12, padx=16)
    header_frame.pack(fill="x")

    title_lbl = tk.Label(
        header_frame,
        text="📷 QUÉT & CHỌN NGUỒN CAMERA HỆ THỐNG",
        font=("Segoe UI", 13, "bold"),
        fg="#00e676",
        bg="#22222c"
    )
    title_lbl.pack(anchor="w")

    subtitle_lbl = tk.Label(
        header_frame,
        text="Hỗ trợ Camera công nghiệp Hikvision (GigE/USB), Webcam USB ngoài & File ảnh mẫu",
        font=("Segoe UI", 9),
        fg="#a0a0b5",
        bg="#22222c"
    )
    subtitle_lbl.pack(anchor="w", pady=(2, 0))

    # Thông báo đếm ngược (nếu có)
    countdown_var = tk.StringVar()
    timer_lbl = tk.Label(
        header_frame,
        textvariable=countdown_var,
        font=("Segoe UI", 9, "italic"),
        fg="#ffb74d",
        bg="#22222c"
    )
    if countdown_active:
        timer_lbl.pack(anchor="w", pady=(4, 0))

    # Bảng hiển thị danh sách thiết bị
    table_frame = tk.Frame(root, bg="#18181f", padx=16, pady=10)
    table_frame.pack(fill="both", expand=True)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "Treeview",
        background="#252530",
        foreground="#ffffff",
        fieldbackground="#252530",
        rowheight=32,
        font=("Segoe UI", 10)
    )
    style.configure(
        "Treeview.Heading",
        background="#2f2f3d",
        foreground="#00e676",
        font=("Segoe UI", 10, "bold")
    )
    style.map("Treeview", background=[("selected", "#007acc")], foreground=[("selected", "#ffffff")])

    cols = ("badge", "name", "details", "status")
    tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
    tree.heading("badge", text="Loại nguồn")
    tree.heading("name", text="Tên thiết bị / Nguồn")
    tree.heading("details", text="Thông số kỹ thuật")
    tree.heading("status", text="Trạng thái")

    tree.column("badge", width=140, anchor="center")
    tree.column("name", width=220, anchor="w")
    tree.column("details", width=240, anchor="w")
    tree.column("status", width=100, anchor="center")

    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def populate_tree():
        tree.delete(*tree.get_children())
        for idx, src in enumerate(sources):
            is_cur = src.matches(current_source)
            st_text = "🟢 Đang dùng" if is_cur else "Sẵn sàng"
            item_id = tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(src.badge, src.name, src.details, st_text)
            )
            if is_cur:
                tree.selection_set(item_id)
                tree.focus(item_id)

        # Nếu chưa chọn dòng nào, chọn dòng đầu tiên
        if not tree.selection() and sources:
            tree.selection_set("0")
            tree.focus("0")

    populate_tree()

    def cancel_countdown(event=None):
        nonlocal countdown_active
        countdown_active = False
        timer_lbl.pack_forget()

    tree.bind("<Button-1>", cancel_countdown)
    tree.bind("<Key>", cancel_countdown)

    def on_confirm():
        nonlocal selected_source
        sel = tree.selection()
        if sel:
            idx = int(sel[0])
            if 0 <= idx < len(sources):
                selected_source = sources[idx]
        root.destroy()

    def on_rescan():
        cancel_countdown()
        nonlocal sources
        sources = scan_all_camera_sources()
        populate_tree()

    def on_browse_file():
        cancel_countdown()
        filetypes = [
            ("Ảnh & Video", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi"),
            ("Tất cả tập tin", "*.*")
        ]
        chosen = filedialog.askopenfilename(title="Chọn ảnh hoặc video thử nghiệm", filetypes=filetypes)
        if chosen:
            p = Path(chosen)
            img = cv2.imread(str(p))
            res = (img.shape[1], img.shape[0]) if img is not None else (0, 0)
            custom_src = CameraSource(
                source_type="file",
                source_id=str(p),
                name=f"[File ngoài] {p.name}",
                details=f"Kích thước: {res[0]}x{res[1]} px",
                resolution=res,
                badge="🖼️ File ngoài"
            )
            sources.insert(0, custom_src)
            populate_tree()
            tree.selection_set("0")
            tree.focus("0")

    tree.bind("<Double-1>", lambda e: on_confirm())
    root.bind("<Return>", lambda e: on_confirm())
    root.bind("<Escape>", lambda e: root.destroy())

    # Khung nút bấm điều khiển
    btn_frame = tk.Frame(root, bg="#18181f", pady=12, padx=16)
    btn_frame.pack(fill="x")

    rescan_btn = tk.Button(
        btn_frame,
        text="🔄 Quét lại (F5)",
        font=("Segoe UI", 9),
        bg="#333342",
        fg="#ffffff",
        activebackground="#444455",
        activeforeground="#ffffff",
        padx=12,
        pady=5,
        bd=0,
        command=on_rescan
    )
    rescan_btn.pack(side="left", padx=(0, 8))
    root.bind("<F5>", lambda e: on_rescan())

    browse_btn = tk.Button(
        btn_frame,
        text="📁 Chọn File khác...",
        font=("Segoe UI", 9),
        bg="#333342",
        fg="#ffffff",
        activebackground="#444455",
        activeforeground="#ffffff",
        padx=12,
        pady=5,
        bd=0,
        command=on_browse_file
    )
    browse_btn.pack(side="left")

    close_btn = tk.Button(
        btn_frame,
        text="Hủy bỏ",
        font=("Segoe UI", 9),
        bg="#2d2d38",
        fg="#cfcfd8",
        padx=16,
        pady=5,
        bd=0,
        command=root.destroy
    )
    close_btn.pack(side="right", padx=(8, 0))

    confirm_btn = tk.Button(
        btn_frame,
        text="✅ Kết nối Nguồn này",
        font=("Segoe UI", 10, "bold"),
        bg="#00c853",
        fg="#ffffff",
        activebackground="#00e676",
        activeforeground="#000000",
        padx=16,
        pady=5,
        bd=0,
        command=on_confirm
    )
    confirm_btn.pack(side="right")

    # Bộ đếm ngược
    def tick():
        nonlocal remaining_time, countdown_active, selected_source
        if not countdown_active:
            return
        if remaining_time <= 0:
            sel = tree.selection()
            if sel:
                idx = int(sel[0])
                if 0 <= idx < len(sources):
                    selected_source = sources[idx]
            root.destroy()
            return

        def_name = ""
        sel = tree.selection()
        if sel:
            idx = int(sel[0])
            if 0 <= idx < len(sources):
                def_name = f": {sources[idx].name}"

        countdown_var.set(f"⏱️ Tự động kết nối sau {remaining_time} giây... (Nhấp chuột vào danh sách để dừng)")
        remaining_time -= 1
        root.after(1000, tick)

    if countdown_active:
        tick()

    root.mainloop()
    return selected_source


def terminal_camera_picker(
    sources: List[CameraSource],
    current_source: Optional[CameraSource] = None,
    timeout_sec: int = 0
) -> Optional[CameraSource]:
    """Hiển thị menu chọn nguồn camera trên Terminal"""
    if not sources:
        print("[CẢNH BÁO] Không tìm thấy nguồn camera nào.")
        return None

    print("\n" + "=" * 68)
    print("        DANH SÁCH NGUỒN CAMERA HỆ THỐNG ĐÃ QUÉT ĐƯỢC")
    print("=" * 68)

    default_idx = 0
    for i, s in enumerate(sources):
        is_cur = s.matches(current_source)
        mark = " (Đang kết nối)" if is_cur else ""
        if is_cur:
            default_idx = i
        print(f"  [{i + 1}] {s.badge:<18} | {s.name:<25} | {s.details}{mark}")

    print("-" * 68)
    prompt = f"Nhập số [1-{len(sources)}] (Mặc định: [{default_idx + 1}]): "

    try:
        ans = input(prompt).strip()
        if not ans:
            return sources[default_idx]
        choice = int(ans) - 1
        if 0 <= choice < len(sources):
            return sources[choice]
        else:
            print("[LỖI] Số chọn không hợp lệ, chọn nguồn mặc định.")
            return sources[default_idx]
    except Exception:
        return sources[default_idx]


def bgr_to_hex(bgr: List[int]) -> str:
    """Chuyển đổi BGR sang chuỗi màu HEX định dạng #RRGGBB"""
    b, g, r = [max(0, min(255, int(v))) for v in bgr[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_bgr(hex_str: str) -> List[int]:
    """Chuyển đổi chuỗi màu HEX sang mảng [B, G, R]"""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c * 2 for c in hex_str])
    if len(hex_str) != 6:
        return [0, 0, 0]
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return [b, g, r]


def show_color_picker_dialog(
    conveyor_bgr: List[int],
    pcb_bgr: List[int],
    presets: List[Dict[str, Any]],
    on_apply_callback=None
) -> Dict[str, Any]:
    """
    Hiển thị hộp thoại Tkinter Dark Theme để tùy chỉnh màu nền băng tải & màu bo mạch PCB
    """
    import tkinter as tk
    from tkinter import ttk, colorchooser

    root = tk.Tk()
    root.title("Hiệu Chuẩn Màu Nền Băng Tải & Bo Mạch PCB")
    root.geometry("560x540")
    root.configure(bg="#18181f")
    root.resizable(False, False)

    # Đưa lên trên cùng
    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)

    cur_conv = list(conveyor_bgr)
    cur_pcb = list(pcb_bgr)
    result = {"applied": False, "conveyor": cur_conv, "pcb": cur_pcb}

    # Header
    header = tk.Frame(root, bg="#22222c", padx=16, pady=12)
    header.pack(fill="x")
    title_lbl = tk.Label(
        header,
        text="🎨 HIỆU CHUẨN MÀU NỀN BĂNG TẢI & BO MẠCH PCB",
        font=("Segoe UI", 12, "bold"),
        fg="#00e676",
        bg="#22222c"
    )
    title_lbl.pack(anchor="w")
    sub_lbl = tk.Label(
        header,
        text="Dùng để loại trừ nền băng tải và nhận diện bóc tách PCB không cần ảnh rỗng",
        font=("Segoe UI", 9),
        fg="#a0a0b5",
        bg="#22222c"
    )
    sub_lbl.pack(anchor="w", pady=(2, 0))

    body = tk.Frame(root, bg="#18181f", padx=18, pady=12)
    body.pack(fill="both", expand=True)

    # 1. Băng tải
    conv_group = tk.LabelFrame(
        body,
        text=" 1. Màu Nền Băng Tải (Conveyor Belt Color) ",
        font=("Segoe UI", 10, "bold"),
        fg="#ffb74d",
        bg="#1f1f28",
        padx=12,
        pady=10
    )
    conv_group.pack(fill="x", pady=(0, 10))

    conv_swatch = tk.Label(
        conv_group,
        text="       ",
        bg=bgr_to_hex(cur_conv),
        relief="solid",
        bd=1,
        width=7,
        height=2
    )
    conv_swatch.pack(side="left", padx=(0, 12))

    conv_info_var = tk.StringVar(value=f"BGR: {cur_conv} | HEX: {bgr_to_hex(cur_conv)}")
    conv_info_lbl = tk.Label(
        conv_group,
        textvariable=conv_info_var,
        font=("Segoe UI", 9),
        fg="#ffffff",
        bg="#1f1f28"
    )
    conv_info_lbl.pack(side="left")

    def pick_conv_color():
        c = colorchooser.askcolor(color=bgr_to_hex(cur_conv), title="Chọn Màu Nền Băng Tải")
        if c and c[1]:
            cur_conv[:] = hex_to_bgr(c[1])
            conv_swatch.configure(bg=c[1])
            conv_info_var.set(f"BGR: {cur_conv} | HEX: {c[1].upper()}")

    conv_btn = tk.Button(
        conv_group,
        text="🎨 Chọn màu...",
        command=pick_conv_color,
        font=("Segoe UI", 9, "bold"),
        bg="#2f2f3d",
        fg="#ffffff",
        activebackground="#007acc",
        activeforeground="#ffffff",
        padx=10,
        pady=4
    )
    conv_btn.pack(side="right")

    # 2. PCB
    pcb_group = tk.LabelFrame(
        body,
        text=" 2. Màu Thân Bo Mạch PCB (PCB Substrate Color) ",
        font=("Segoe UI", 10, "bold"),
        fg="#69f0ae",
        bg="#1f1f28",
        padx=12,
        pady=10
    )
    pcb_group.pack(fill="x", pady=(0, 10))

    pcb_swatch = tk.Label(
        pcb_group,
        text="       ",
        bg=bgr_to_hex(cur_pcb),
        relief="solid",
        bd=1,
        width=7,
        height=2
    )
    pcb_swatch.pack(side="left", padx=(0, 12))

    pcb_info_var = tk.StringVar(value=f"BGR: {cur_pcb} | HEX: {bgr_to_hex(cur_pcb)}")
    pcb_info_lbl = tk.Label(
        pcb_group,
        textvariable=pcb_info_var,
        font=("Segoe UI", 9),
        fg="#ffffff",
        bg="#1f1f28"
    )
    pcb_info_lbl.pack(side="left")

    def pick_pcb_color():
        c = colorchooser.askcolor(color=bgr_to_hex(cur_pcb), title="Chọn Màu Bo Mạch PCB")
        if c and c[1]:
            cur_pcb[:] = hex_to_bgr(c[1])
            pcb_swatch.configure(bg=c[1])
            pcb_info_var.set(f"BGR: {cur_pcb} | HEX: {c[1].upper()}")

    pcb_btn = tk.Button(
        pcb_group,
        text="🎨 Chọn màu...",
        command=pick_pcb_color,
        font=("Segoe UI", 9, "bold"),
        bg="#2f2f3d",
        fg="#ffffff",
        activebackground="#007acc",
        activeforeground="#ffffff",
        padx=10,
        pady=4
    )
    pcb_btn.pack(side="right")

    # 3. Presets
    preset_group = tk.LabelFrame(
        body,
        text=" 3. Bộ Mẫu Chuẩn Công Nghiệp (Quick Presets) ",
        font=("Segoe UI", 10, "bold"),
        fg="#80d8ff",
        bg="#1f1f28",
        padx=12,
        pady=10
    )
    preset_group.pack(fill="x", pady=(0, 10))

    preset_frame = tk.Frame(preset_group, bg="#1f1f28")
    preset_frame.pack(fill="x")

    def apply_preset(p):
        cur_conv[:] = list(p["conveyor"])
        cur_pcb[:] = list(p["pcb"])
        conv_swatch.configure(bg=bgr_to_hex(cur_conv))
        conv_info_var.set(f"BGR: {cur_conv} | {p['conveyor_name']}")
        pcb_swatch.configure(bg=bgr_to_hex(cur_pcb))
        pcb_info_var.set(f"BGR: {cur_pcb} | {p['pcb_name']}")

    for idx, p in enumerate(presets):
        row = idx // 2
        col = idx % 2
        btn = tk.Button(
            preset_frame,
            text=f"[{p['id']}] {p.get('badge', p['name'])}",
            command=lambda pr=p: apply_preset(pr),
            font=("Segoe UI", 9),
            bg="#2d2d3a",
            fg="#ffffff",
            activebackground="#007acc",
            activeforeground="#ffffff",
            width=26,
            anchor="w",
            padx=8,
            pady=3
        )
        btn.grid(row=row, column=col, padx=4, pady=3, sticky="w")

    # Footer
    footer = tk.Frame(root, bg="#22222c", padx=16, pady=10)
    footer.pack(fill="x", side="bottom")

    def on_apply():
        result["applied"] = True
        result["conveyor"] = cur_conv
        result["pcb"] = cur_pcb
        if on_apply_callback:
            on_apply_callback(cur_conv, cur_pcb)
        root.destroy()

    apply_btn = tk.Button(
        footer,
        text="💾 LƯU & ÁP DỤNG NGAY",
        command=on_apply,
        font=("Segoe UI", 10, "bold"),
        bg="#00c853",
        fg="#ffffff",
        activebackground="#00e676",
        padx=14,
        pady=5
    )
    apply_btn.pack(side="right", padx=(8, 0))

    cancel_btn = tk.Button(
        footer,
        text="Đóng",
        command=root.destroy,
        font=("Segoe UI", 10),
        bg="#3d3d4d",
        fg="#ffffff",
        padx=12,
        pady=5
    )
    cancel_btn.pack(side="right")

    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    rw = root.winfo_width()
    rh = root.winfo_height()
    root.geometry(f"+{(sw-rw)//2}+{(sh-rh)//2}")

    root.mainloop()
    return result


if __name__ == "__main__":
    print("[1/2] Đang quét toàn bộ nguồn camera hệ thống...")
    found_sources = scan_all_camera_sources()
    print(f"[2/2] Tìm thấy {len(found_sources)} nguồn thiết bị.")

    # Kiểm tra cấu hình đã lưu
    saved_cfg = load_camera_config()
    current_src = CameraSource.from_dict(saved_cfg) if saved_cfg else None

    # Mở giao diện chọn
    chosen = show_camera_picker_dialog(found_sources, current_source=current_src, timeout_sec=0)
    if chosen:
        print(f"\n[KẾT QUẢ] Bạn đã chọn nguồn: {chosen.badge} | {chosen.name}")
        save_camera_config(chosen)
    else:
        print("\n[THÔNG BÁO] Đã hủy chọn camera.")

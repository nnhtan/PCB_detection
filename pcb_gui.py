"""
PCB Vision AI - Desktop GUI
Ứng dụng giao diện kéo thả file ảnh (.bmp, .jpg, .png) để nhận diện linh kiện PCB và kiểm định chất lượng (QA).
Hỗ trợ Zoom/Pan ảnh mượt mà, tùy chọn bật/tắt nhãn chữ (tránh che khuất bo mạch), và lọc linh kiện khi click bảng.
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize, QPointF, QRectF
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPixmap,
    QWheelEvent,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Từ điển dịch tên linh kiện sang Tiếng Việt
VIETNAMESE_NAMES = {
    "capacitor": "Tụ điện",
    "ic": "Vi mạch tích hợp (IC)",
    "led": "Đèn LED",
    "resistor": "Điện trở",
    "battery": "Pin / Nguồn",
    "buzzer": "Còi chíp (Buzzer)",
    "clock": "Thạch anh dao động",
    "connector": "Đầu cắm / Kết nối",
    "diode": "Đi-ốt (Diode)",
    "display": "Màn hình hiển thị",
    "fuse": "Cầu chì",
    "inductor": "Cuộn cảm",
    "potentiometer": "Biến trở xoay",
    "relay": "Rơ-le (Relay)",
    "switch": "Công tắc (Switch)",
    "transistor": "Bóng bán dẫn (Transistor)",
    "variable resistor": "Biến trở",
}

# Màu sắc BGR cho từng loại linh kiện (OpenCV dùng BGR)
CLASS_COLORS_BGR = {
    "resistor": (248, 189, 56),       # Sky blue
    "ic": (153, 72, 236),             # Pink/Purple
    "capacitor": (11, 158, 245),      # Amber/Gold
    "diode": (129, 185, 16),          # Emerald Green
    "connector": (246, 92, 139),      # Violet
    "transistor": (212, 182, 6),      # Cyan
    "potentiometer": (22, 115, 249),  # Orange
    "inductor": (22, 204, 132),       # Lime
    "led": (68, 68, 239),             # Red
    "default": (160, 140, 120),       # Slate
}

def get_vietnamese_name(name: str) -> str:
    key = str(name).strip().casefold().replace("-", " ").replace("_", " ")
    return VIETNAMESE_NAMES.get(key, name)

def get_class_bgr(name: str) -> tuple:
    key = str(name).strip().casefold()
    return CLASS_COLORS_BGR.get(key, CLASS_COLORS_BGR["default"])

def get_class_hex(name: str) -> str:
    b, g, r = get_class_bgr(name)
    return f"#{r:02x}{g:02x}{b:02x}"

def cv_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """Chuyển ảnh OpenCV BGR sang QPixmap của Qt."""
    if cv_img is None or cv_img.size == 0:
        return QPixmap()
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ============================================================
# HÀM TÙY BIẾN VẼ BOUNDING BOX & NHÃN (KHÔNG CHE KHUẤT MẠCH)
# ============================================================

def render_custom_annotations(
    raw_img: np.ndarray,
    detected_list: list,
    show_boxes: bool = True,
    show_labels: bool = False,       # Mặc định TẮT nhãn chữ để không che mạch
    show_conf: bool = False,         # Mặc định TẮT số confidence
    show_id_only: bool = False,      # Chỉ hiện số #1, #2...
    filter_class: str = None,        # Lọc hiển thị khi click vào bảng
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Vẽ bounding box và nhãn tinh tế lên ảnh:
    - Khi show_labels=False: Chỉ vẽ đường viền khung mảnh sắc nét, KHÔNG có chữ hay nền đặc,
      giúp quan sát 100% đường mạch và linh kiện.
    - Khi có filter_class: Làm nổi bật linh kiện được chọn, làm mờ các linh kiện còn lại.
    """
    if raw_img is None or raw_img.size == 0:
        return raw_img

    annotated = raw_img.copy()
    h, w = raw_img.shape[:2]

    # Tính toán cỡ chữ phù hợp với độ phân giải ảnh
    font_scale = max(0.4, min(0.7, h / 3000.0))

    for idx, d in enumerate(detected_list, 1):
        cls_name = d["class"]
        conf_val = d["confidence"]
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]

        is_selected = (filter_class is None) or (cls_name.casefold() == filter_class.casefold())

        # Nếu đang lọc theo loại linh kiện và linh kiện này không thuộc loại đó:
        if not is_selected:
            # Vẽ nét đứt/nét mờ mỏng màu xám để không làm rối mắt
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (70, 75, 85), 1)
            continue

        color = get_class_bgr(cls_name)
        thickness = line_thickness if is_selected else 1

        # 1. Vẽ khung viền (Bounding Box)
        if show_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # 2. Vẽ nhãn (chỉ khi người dùng bật show_labels)
        if show_labels:
            if show_id_only:
                text = f"#{idx}"
            elif show_conf:
                text = f"{cls_name} {conf_val:.2f}"
            else:
                text = f"{cls_name}"

            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

            # Vị trí đặt nhãn: nằm phía trên góc trái khung box
            tx = x1
            ty = max(y1 - 4, th + 4)

            # Vẽ thẻ nền mỏng bán trong suốt thay vì một khối màu che kín
            pad = 2
            cv2.rectangle(
                annotated,
                (tx - pad, ty - th - pad),
                (tx + tw + pad, ty + baseline),
                (18, 22, 30),
                cv2.FILLED
            )
            cv2.rectangle(
                annotated,
                (tx - pad, ty - th - pad),
                (tx + tw + pad, ty + baseline),
                color,
                1
            )
            cv2.putText(
                annotated,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return annotated


# ============================================================
# WORKER THREAD XỬ LÝ SUY LUẬN YOLO & QA
# ============================================================

class InferenceWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, image_path: str, model_path: str = "best.pt", conf: float = 0.25, imgsz: int = 1536):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz

    def run(self):
        try:
            self.progress.emit("Đang đọc file ảnh...")
            img_path = Path(self.image_path)
            if not img_path.exists():
                self.error.emit(f"Không tìm thấy file ảnh: {self.image_path}")
                return

            image = cv2.imread(str(img_path))
            if image is None:
                self.error.emit("OpenCV không thể đọc file ảnh này.")
                return

            h, w = image.shape[:2]

            self.progress.emit("Đang nạp model YOLO...")
            from ultralytics import YOLO
            import torch

            device = 0 if torch.cuda.is_available() else "cpu"
            model = YOLO(self.model_path)

            self.progress.emit("Đang suy luận phát hiện linh kiện...")
            start_time = time.time()
            results = model.predict(
                source=image,
                conf=self.conf,
                imgsz=self.imgsz,
                device=device,
                verbose=False,
            )
            latency = (time.time() - start_time) * 1000

            result = results[0]
            boxes = result.boxes
            total_count = len(boxes)

            detected_list = []
            class_counts = Counter()

            for idx, box in enumerate(boxes, 1):
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                conf_val = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                class_counts[cls_name] += 1
                detected_list.append({
                    "id": idx,
                    "class": cls_name,
                    "confidence": conf_val,
                    "bbox": xyxy
                })

            # Thử đối chiếu với mạch tham chiếu
            self.progress.emit("Đang kiểm định chất lượng đối chiếu mạch chuẩn...")
            qa_status = "CHỈ NHẬN DIỆN"
            missing_count = 0
            extra_count = 0
            qa_note = ""

            try:
                ref_json = "backend/storage/reference_components.json"
                ref_img = "backend/storage/reference.jpg"
                if not Path(ref_json).exists():
                    ref_json = "reference_components.json"
                if not Path(ref_img).exists():
                    ref_img = "Image_20261019190958053.bmp"

                if Path(ref_json).exists() and Path(ref_img).exists():
                    from backend.app.services.pcb_engine import PCBPipeline
                    pipeline = PCBPipeline(
                        model_path=self.model_path,
                        ref_json_path=ref_json,
                        ref_image_path=ref_img,
                    )
                    qa_res = pipeline.inspect_pcb(
                        frame=image,
                        conf=self.conf,
                        imgsz=self.imgsz,
                        auto_crop=True,
                    )
                    qa_status = qa_res.get("status", "UNALIGNED")
                    missing_count = len(qa_res.get("missing_components_details", []))
                    extra_count = qa_res.get("stats", {}).get("extra", 0)
                    align_method = qa_res.get("alignment_method", "None")
                    qa_note = f"Căn chỉnh: {align_method}"
            except Exception as e:
                qa_note = str(e)

            res_payload = {
                "image_path": str(img_path.resolve()),
                "filename": img_path.name,
                "width": w,
                "height": h,
                "latency_ms": round(latency, 1),
                "total_count": total_count,
                "class_counts": dict(class_counts),
                "detected_list": detected_list,
                "raw_image": image,
                "qa_status": qa_status,
                "missing_count": missing_count,
                "extra_count": extra_count,
                "qa_note": qa_note,
            }
            self.finished.emit(res_payload)

        except Exception as ex:
            self.error.emit(str(ex))


# ============================================================
# TRÌNH XEM ẢNH HỖ TRỢ ZOOM, PAN & KÉO THẢ (QGraphicsView)
# ============================================================

class ZoomableImageDropArea(QGraphicsView):
    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item = None
        self.current_pixmap = None

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        self.show_placeholder()

    def show_placeholder(self):
        self.scene.clear()
        self.pixmap_item = None
        self.current_pixmap = None

        text_item = self.scene.addText(
            "📥  KÉO & THẢ FILE ẢNH VÀO ĐÂY\n\n"
            "• Hỗ trợ định dạng: .BMP, .JPG, .JPEG, .PNG\n"
            "• Lăn chuột để Phóng to / Thu nhỏ chi tiết linh kiện\n"
            "• Giữ chuột trái kéo để di chuyển vùng soi\n"
            "• Bấm nút 'Chọn File Ảnh' để mở từ máy tính"
        )
        text_item.setDefaultTextColor(QColor("#64748b"))
        font = QFont("Segoe UI", 13, QFont.Bold)
        text_item.setFont(font)
        self.setSceneRect(text_item.boundingRect())

        self.setStyleSheet("""
            QGraphicsView {
                background-color: #111622;
                border: 2px dashed #334155;
                border-radius: 12px;
            }
        """)

    def set_display_pixmap(self, pixmap: QPixmap, fit_view: bool = False):
        if pixmap is None or pixmap.isNull():
            return

        self.current_pixmap = pixmap
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

        self.setStyleSheet("""
            QGraphicsView {
                background-color: #0b0f17;
                border: 1px solid #1e293b;
                border-radius: 12px;
            }
        """)

        if fit_view or self.transform().isIdentity():
            self.fit_in_view()

    def fit_in_view(self):
        """Thu nhỏ ảnh để vừa khít toàn bộ khung nhìn."""
        if self.pixmap_item:
            self.resetTransform()
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def set_actual_size(self):
        """Hiển thị ảnh ở kích thước thật 100%."""
        if self.pixmap_item:
            self.resetTransform()

    def zoom(self, factor: float):
        """Thu phóng theo tỉ lệ."""
        self.scale(factor, factor)

    def wheelEvent(self, event: QWheelEvent):
        """Lăn chuột để phóng to / thu nhỏ."""
        if self.pixmap_item is not None:
            zoom_in = 1.15
            zoom_out = 1.0 / zoom_in
            if event.angleDelta().y() > 0:
                self.zoom(zoom_in)
            else:
                self.zoom(zoom_out)
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Nhấp đúp chuột để reset vừa khung nhìn."""
        if self.pixmap_item is not None:
            self.fit_in_view()
        else:
            super().mouseDoubleClickEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                ext = Path(file_path).suffix.lower()
                if ext in [".bmp", ".jpg", ".jpeg", ".png"]:
                    event.acceptProposedAction()
                    self.setStyleSheet("""
                        QGraphicsView {
                            background-color: #1e293b;
                            border: 2px dashed #00d2ff;
                            border-radius: 12px;
                        }
                    """)
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        if not self.current_pixmap:
            self.show_placeholder()
        else:
            self.setStyleSheet("""
                QGraphicsView {
                    background-color: #0b0f17;
                    border: 1px solid #1e293b;
                    border-radius: 12px;
                }
            """)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.fileDropped.emit(file_path)
        event.acceptProposedAction()


# ============================================================
# CỬA SỔ CHÍNH (MAIN WINDOW)
# ============================================================

class PCBMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB Vision AI - Kiểm Tra & Nhận Diện Bảng Mạch In")
        self.resize(1360, 880)
        self.setMinimumSize(1000, 700)

        self.current_image_path = None
        self.current_result_data = None
        self.current_filtered_class = None
        self.showing_annotated = True
        self.cached_annotated_cv = None
        self.worker = None

        self.setup_ui()
        self.apply_theme()

    def setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ====================================================
        # BÊN TRÁI: KHU VỰC ẢNH & KÉO THẢ & ZOOM
        # ====================================================
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # 1. Toolbar chính
        toolbar_frame = QFrame()
        toolbar_frame.setObjectName("toolbarFrame")
        tb_layout = QHBoxLayout(toolbar_frame)
        tb_layout.setContentsMargins(10, 8, 10, 8)
        tb_layout.setSpacing(8)

        self.btn_select_file = QPushButton("📂 Chọn File Ảnh")
        self.btn_select_file.setObjectName("actionBtn")
        self.btn_select_file.clicked.connect(self.choose_file_dialog)
        tb_layout.addWidget(self.btn_select_file)

        self.btn_toggle_view = QPushButton("🔄 Xem Ảnh Gốc")
        self.btn_toggle_view.setObjectName("toggleBtn")
        self.btn_toggle_view.setEnabled(False)
        self.btn_toggle_view.clicked.connect(self.toggle_image_view)
        tb_layout.addWidget(self.btn_toggle_view)

        # Nút Zoom
        self.btn_zoom_in = QPushButton("➕")
        self.btn_zoom_in.setToolTip("Phóng to ảnh (hoặc dùng con lăn chuột)")
        self.btn_zoom_in.clicked.connect(lambda: self.drop_area.zoom(1.2))
        tb_layout.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("➖")
        self.btn_zoom_out.setToolTip("Thu nhỏ ảnh (hoặc dùng con lăn chuột)")
        self.btn_zoom_out.clicked.connect(lambda: self.drop_area.zoom(1.0 / 1.2))
        tb_layout.addWidget(self.btn_zoom_out)

        self.btn_fit_view = QPushButton("🔄 Vừa Khung")
        self.btn_fit_view.setToolTip("Thu vừa khít khung nhìn (hoặc nhấp đúp chuột)")
        self.btn_fit_view.clicked.connect(lambda: self.drop_area.fit_in_view())
        tb_layout.addWidget(self.btn_fit_view)

        self.btn_save_result = QPushButton("💾 Lưu Ảnh")
        self.btn_save_result.setObjectName("successBtn")
        self.btn_save_result.setEnabled(False)
        self.btn_save_result.clicked.connect(self.save_annotated_image)
        tb_layout.addWidget(self.btn_save_result)

        tb_layout.addStretch()
        left_layout.addWidget(toolbar_frame)

        # 2. Thanh cài đặt nhãn & viền hiển thị (GIẢI PHÁP CHỐNG CHE MẠCH)
        annotation_bar = QFrame()
        annotation_bar.setObjectName("annotationBar")
        ab_layout = QHBoxLayout(annotation_bar)
        ab_layout.setContentsMargins(12, 6, 12, 6)
        ab_layout.setSpacing(14)

        lbl_opt_title = QLabel("👁️ Hiển Thị:")
        lbl_opt_title.setStyleSheet("color: #38bdf8; font-weight: bold;")
        ab_layout.addWidget(lbl_opt_title)

        # CHECKBOX HIỆN NHÃN (MẶC ĐỊNH TẮT ĐỂ KHÔNG BỊ CHE KHUẤT BO MẠCH)
        self.chk_show_labels = QCheckBox("🏷️ Hiện nhãn chữ")
        self.chk_show_labels.setChecked(False)  # MẶC ĐỊNH TẮT
        self.chk_show_labels.setToolTip("Bật/Tắt chữ tên linh kiện để quan sát rõ toàn bộ bo mạch")
        self.chk_show_labels.stateChanged.connect(self.update_annotation_render)
        ab_layout.addWidget(self.chk_show_labels)

        self.chk_id_only = QCheckBox("🔢 Chỉ hiện số #ID")
        self.chk_id_only.setChecked(False)
        self.chk_id_only.setToolTip("Chỉ hiển thị số thứ tự nhỏ gọn (#1, #2...) thay vì tên dài")
        self.chk_id_only.stateChanged.connect(self.update_annotation_render)
        ab_layout.addWidget(self.chk_id_only)

        self.chk_show_conf = QCheckBox("📊 Hiện Confidence")
        self.chk_show_conf.setChecked(False)
        self.chk_show_conf.stateChanged.connect(self.update_annotation_render)
        ab_layout.addWidget(self.chk_show_conf)

        ab_layout.addWidget(QLabel("Độ dày nét:"))
        self.combo_thickness = QComboBox()
        self.combo_thickness.addItems(["1 px (Mảnh nhất)", "2 px (Chuẩn)", "3 px (Đậm)"])
        self.combo_thickness.setCurrentIndex(1)
        self.combo_thickness.currentIndexChanged.connect(self.update_annotation_render)
        ab_layout.addWidget(self.combo_thickness)

        ab_layout.addStretch()
        left_layout.addWidget(annotation_bar)

        # 3. Khung hiển thị / Kéo thả có Zoom/Pan
        self.drop_area = ZoomableImageDropArea()
        self.drop_area.fileDropped.connect(self.process_image_file)
        left_layout.addWidget(self.drop_area, 1)

        # 4. Thanh tiến trình
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.progress_bar)

        splitter.addWidget(left_container)

        # ====================================================
        # BÊN PHẢI: BẢNG THỐNG KÊ & KẾT QUẢ KIỂM ĐỊNH
        # ====================================================
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 1. Thẻ trạng thái kết quả (Status Card)
        self.status_card = QFrame()
        self.status_card.setObjectName("statusCard")
        sc_layout = QVBoxLayout(self.status_card)
        sc_layout.setContentsMargins(14, 12, 14, 12)
        sc_layout.setSpacing(6)

        self.lbl_main_status = QLabel("CHƯA CÓ DỮ LIỆU")
        self.lbl_main_status.setObjectName("statusTitle")
        sc_layout.addWidget(self.lbl_main_status)

        self.lbl_file_details = QLabel("Kéo thả file ảnh (.bmp, .jpg, .png) vào cửa sổ để bắt đầu.")
        self.lbl_file_details.setObjectName("statusSub")
        self.lbl_file_details.setWordWrap(True)
        sc_layout.addWidget(self.lbl_file_details)

        self.lbl_metrics = QLabel("")
        self.lbl_metrics.setObjectName("statusMetrics")
        sc_layout.addWidget(self.lbl_metrics)

        right_layout.addWidget(self.status_card)

        # 2. Thanh chỉnh Confidence
        params_card = QFrame()
        params_card.setObjectName("paramsCard")
        pc_layout = QHBoxLayout(params_card)
        pc_layout.setContentsMargins(12, 8, 12, 8)
        pc_layout.setSpacing(10)

        lbl_conf = QLabel("Độ tin cậy:")
        lbl_conf.setStyleSheet("color: #94a3b8; font-weight: bold;")
        pc_layout.addWidget(lbl_conf)

        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(25)
        self.slider_conf.valueChanged.connect(self.on_conf_changed)
        pc_layout.addWidget(self.slider_conf)

        self.lbl_conf_val = QLabel("0.25")
        self.lbl_conf_val.setStyleSheet("color: #38bdf8; font-weight: bold; min-width: 32px;")
        pc_layout.addWidget(self.lbl_conf_val)

        self.btn_re_scan = QPushButton("⚡ Quét Lại")
        self.btn_re_scan.setObjectName("actionBtn")
        self.btn_re_scan.setEnabled(False)
        self.btn_re_scan.clicked.connect(self.re_scan_image)
        pc_layout.addWidget(self.btn_re_scan)

        right_layout.addWidget(params_card)

        # 3. Tiêu đề bảng + Thông tin lọc
        table_header_layout = QHBoxLayout()
        lbl_table_title = QLabel("📊 BẢNG THỐNG KÊ LINH KIỆN")
        lbl_table_title.setStyleSheet("color: #f1f5f9; font-weight: bold; font-size: 14px;")
        table_header_layout.addWidget(lbl_table_title)

        self.btn_clear_filter = QPushButton("❌ Hiện Tất Cả")
        self.btn_clear_filter.setVisible(False)
        self.btn_clear_filter.setObjectName("toolBtn")
        self.btn_clear_filter.clicked.connect(self.clear_class_filter)
        table_header_layout.addWidget(self.btn_clear_filter)

        table_header_layout.addStretch()
        right_layout.addLayout(table_header_layout)

        self.lbl_filter_hint = QLabel("💡 Mẹo: Bấm vào dòng linh kiện để làm nổi bật riêng loại đó trên ảnh.")
        self.lbl_filter_hint.setStyleSheet("color: #64748b; font-size: 11px; font-style: italic;")
        right_layout.addWidget(self.lbl_filter_hint)

        # 4. Bảng thống kê
        self.table_stats = QTableWidget(0, 5)
        self.table_stats.setHorizontalHeaderLabels([
            "Loại Linh Kiện",
            "Tên Tiếng Việt",
            "Số Lượng",
            "Tỉ Lệ %",
            "Độ Tin Cậy TB"
        ])
        self.table_stats.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_stats.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_stats.verticalHeader().setVisible(False)
        self.table_stats.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_stats.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_stats.itemClicked.connect(self.on_table_item_clicked)
        right_layout.addWidget(self.table_stats, 1)

        # 5. Hàng nút chức năng bên dưới
        footer_layout = QHBoxLayout()
        self.btn_set_reference = QPushButton("📌 Đặt Làm Mạch Mẫu Chuẩn (Reference)")
        self.btn_set_reference.setObjectName("toolBtn")
        self.btn_set_reference.setEnabled(False)
        self.btn_set_reference.clicked.connect(self.set_as_reference)
        footer_layout.addWidget(self.btn_set_reference)

        right_layout.addLayout(footer_layout)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0f17;
            }
            QWidget {
                color: #e2e8f0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            #toolbarFrame, #annotationBar, #statusCard, #paramsCard {
                background-color: #151c28;
                border: 1px solid #222d3d;
                border-radius: 8px;
            }
            QPushButton {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #273549;
                border-color: #475569;
            }
            QPushButton:disabled {
                background-color: #121824;
                color: #475569;
                border-color: #1e293b;
            }
            #actionBtn {
                background-color: #0284c7;
                color: #ffffff;
                border: none;
                font-weight: bold;
            }
            #actionBtn:hover {
                background-color: #0369a1;
            }
            #successBtn {
                background-color: #059669;
                color: #ffffff;
                border: none;
                font-weight: bold;
            }
            #successBtn:hover {
                background-color: #047857;
            }
            #toggleBtn {
                background-color: #334155;
                color: #38bdf8;
                border: 1px solid #475569;
            }
            QCheckBox {
                color: #cbd5e1;
                font-size: 12px;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #475569;
                background-color: #1e293b;
            }
            QCheckBox::indicator:checked {
                background-color: #0284c7;
                border-color: #38bdf8;
            }
            QComboBox {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 3px 8px;
            }
            #statusTitle {
                font-size: 16px;
                font-weight: bold;
                color: #38bdf8;
            }
            #statusSub {
                color: #94a3b8;
                font-size: 12px;
            }
            #statusMetrics {
                color: #cbd5e1;
                font-size: 12px;
                margin-top: 3px;
            }
            QTableWidget {
                background-color: #121824;
                border: 1px solid #222d3d;
                border-radius: 8px;
                gridline-color: #1e293b;
                color: #f1f5f9;
            }
            QHeaderView::section {
                background-color: #182232;
                color: #94a3b8;
                font-weight: bold;
                padding: 6px;
                border: none;
                border-bottom: 2px solid #222d3d;
            }
            QTableWidget::item:selected {
                background-color: #1e3a5f;
                color: #38bdf8;
                font-weight: bold;
            }
            QProgressBar {
                background-color: #151c28;
                border: 1px solid #222d3d;
                border-radius: 6px;
                height: 20px;
                color: #38bdf8;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #0284c7;
                border-radius: 5px;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #222d3d;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #0284c7;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #38bdf8;
                border: 2px solid #0284c7;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
        """)

    def on_conf_changed(self, val):
        self.lbl_conf_val.setText(f"{val / 100:.2f}")

    def choose_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file ảnh bo mạch PCB",
            "",
            "Image Files (*.bmp *.jpg *.jpeg *.png)"
        )
        if file_path:
            self.process_image_file(file_path)

    def process_image_file(self, file_path: str):
        self.current_image_path = file_path
        self.current_filtered_class = None
        self.btn_clear_filter.setVisible(False)
        conf = self.slider_conf.value() / 100.0

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.lbl_main_status.setText("⏳ ĐANG PHÂN TÍCH...")
        self.lbl_main_status.setStyleSheet("color: #f59e0b;")
        self.lbl_file_details.setText(f"File: {Path(file_path).name}")

        self.btn_select_file.setEnabled(False)
        self.btn_re_scan.setEnabled(False)

        self.worker = InferenceWorker(
            image_path=file_path,
            model_path="best.pt",
            conf=conf,
            imgsz=1536
        )
        self.worker.progress.connect(lambda msg: self.progress_bar.setFormat(msg))
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.start()

    def re_scan_image(self):
        if self.current_image_path:
            self.process_image_file(self.current_image_path)

    def on_inference_finished(self, data: dict):
        self.current_result_data = data
        self.progress_bar.setVisible(False)
        self.btn_select_file.setEnabled(True)
        self.btn_re_scan.setEnabled(True)
        self.btn_toggle_view.setEnabled(True)
        self.btn_save_result.setEnabled(True)
        self.btn_set_reference.setEnabled(True)

        self.showing_annotated = True
        self.btn_toggle_view.setText("🔄 Xem Ảnh Gốc")

        # Cập nhật Card trạng thái
        total = data["total_count"]
        qa_status = data["qa_status"]
        status_color = "#38bdf8"

        if qa_status == "OK":
            status_text = f"✅ ĐẠT CHUẨN (OK) - {total} LINH KIỆN"
            status_color = "#10b981"
        elif qa_status == "FAULTY":
            missing = data["missing_count"]
            status_text = f"❌ LỖI (FAULTY) - Thiếu {missing} linh kiện"
            status_color = "#ef4444"
        elif qa_status == "UNALIGNED":
            status_text = f"⚠️ ĐÃ TÌM THẤY {total} LINH KIỆN (CHƯA CĂN CHỈNH)"
            status_color = "#f59e0b"
        else:
            status_text = f"🔍 ĐÃ PHÁT HIỆN: {total} LINH KIỆN"

        self.lbl_main_status.setText(status_text)
        self.lbl_main_status.setStyleSheet(f"color: {status_color}; font-size: 16px; font-weight: bold;")

        self.lbl_file_details.setText(
            f"File: {data['filename']}  |  Kích thước: {data['width']} x {data['height']} px"
        )
        self.lbl_metrics.setText(
            f"⏱️ Thời gian suy luận YOLO: {data['latency_ms']} ms  |  Model: best.pt  |  Conf: {self.lbl_conf_val.text()}"
        )

        # Cập nhật Bảng thống kê
        class_counts = data["class_counts"]
        detected_list = data["detected_list"]

        conf_sums = Counter()
        for d in detected_list:
            conf_sums[d["class"]] += d["confidence"]

        self.table_stats.setRowCount(len(class_counts))
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

        for row, (cls_name, count) in enumerate(sorted_classes):
            vn_name = get_vietnamese_name(cls_name)
            pct = (count / total * 100) if total > 0 else 0
            avg_conf = (conf_sums[cls_name] / count) if count > 0 else 0

            # Cột 1: Tên gốc + Chấm màu
            item_cls = QTableWidgetItem(cls_name)
            item_cls.setForeground(QColor(get_class_hex(cls_name)))
            item_cls.setFont(QFont("Segoe UI", 10, QFont.Bold))
            self.table_stats.setItem(row, 0, item_cls)

            # Cột 2: Tên tiếng Việt
            item_vn = QTableWidgetItem(vn_name)
            self.table_stats.setItem(row, 1, item_vn)

            # Cột 3: Số lượng
            item_cnt = QTableWidgetItem(str(count))
            item_cnt.setTextAlignment(Qt.AlignCenter)
            item_cnt.setFont(QFont("Segoe UI", 10, QFont.Bold))
            self.table_stats.setItem(row, 2, item_cnt)

            # Cột 4: Tỉ lệ
            item_pct = QTableWidgetItem(f"{pct:.1f}%")
            item_pct.setTextAlignment(Qt.AlignCenter)
            self.table_stats.setItem(row, 3, item_pct)

            # Cột 5: Độ tin cậy trung bình
            item_conf = QTableWidgetItem(f"{avg_conf:.2f}")
            item_conf.setTextAlignment(Qt.AlignCenter)
            self.table_stats.setItem(row, 4, item_conf)

        # Vẽ ảnh hiển thị
        self.update_annotation_render(fit_view=True)

    def on_table_item_clicked(self, item):
        """Khi click vào dòng trong bảng, chỉ làm nổi bật loại linh kiện đó!"""
        row = item.row()
        cls_item = self.table_stats.item(row, 0)
        if cls_item:
            clicked_class = cls_item.text()
            if self.current_filtered_class == clicked_class:
                # Click lại lần 2 -> bỏ lọc
                self.clear_class_filter()
            else:
                self.current_filtered_class = clicked_class
                self.btn_clear_filter.setText(f"❌ Bỏ Lọc ({clicked_class})")
                self.btn_clear_filter.setVisible(True)
                self.update_annotation_render(fit_view=False)

    def clear_class_filter(self):
        """Bỏ lọc linh kiện, hiển thị lại tất cả."""
        self.current_filtered_class = None
        self.btn_clear_filter.setVisible(False)
        self.table_stats.clearSelection()
        self.update_annotation_render(fit_view=False)

    def update_annotation_render(self, fit_view=False):
        """Vẽ lại ảnh bounding box tức thời theo các tùy chọn mà không cần chạy lại YOLO."""
        if not self.current_result_data:
            return

        if not self.showing_annotated:
            # Nếu đang ở chế độ xem ảnh gốc
            pixmap = cv_to_qpixmap(self.current_result_data["raw_image"])
            self.drop_area.set_display_pixmap(pixmap, fit_view=fit_view)
            return

        # Đọc tùy chọn từ thanh công cụ
        show_labels = self.chk_show_labels.isChecked()
        show_conf = self.chk_show_conf.isChecked()
        show_id_only = self.chk_id_only.isChecked()
        thickness_map = [1, 2, 3]
        line_thickness = thickness_map[self.combo_thickness.currentIndex()]

        # Vẽ ảnh với nhãn tinh chỉnh
        annotated_cv = render_custom_annotations(
            raw_img=self.current_result_data["raw_image"],
            detected_list=self.current_result_data["detected_list"],
            show_boxes=True,
            show_labels=show_labels,
            show_conf=show_conf,
            show_id_only=show_id_only,
            filter_class=self.current_filtered_class,
            line_thickness=line_thickness,
        )

        self.cached_annotated_cv = annotated_cv
        pixmap = cv_to_qpixmap(annotated_cv)
        self.drop_area.set_display_pixmap(pixmap, fit_view=fit_view)

    def on_inference_error(self, err_msg: str):
        self.progress_bar.setVisible(False)
        self.btn_select_file.setEnabled(True)
        self.btn_re_scan.setEnabled(True)
        self.lbl_main_status.setText("❌ LỖI XỬ LÝ")
        self.lbl_main_status.setStyleSheet("color: #ef4444;")
        self.lbl_file_details.setText(err_msg)
        QMessageBox.critical(self, "Lỗi Nhận Diện", err_msg)

    def toggle_image_view(self):
        if not self.current_result_data:
            return
        if self.showing_annotated:
            # Chuyển sang ảnh gốc
            self.showing_annotated = False
            self.btn_toggle_view.setText("🔄 Xem Ảnh Đánh Dấu")
            pixmap = cv_to_qpixmap(self.current_result_data["raw_image"])
            self.drop_area.set_display_pixmap(pixmap, fit_view=False)
        else:
            # Chuyển sang ảnh đánh dấu
            self.showing_annotated = True
            self.btn_toggle_view.setText("🔄 Xem Ảnh Gốc")
            self.update_annotation_render(fit_view=False)

    def save_annotated_image(self):
        target_img = (
            self.cached_annotated_cv
            if self.showing_annotated and self.cached_annotated_cv is not None
            else self.current_result_data["raw_image"]
        )
        if target_img is None:
            return

        default_name = f"result_{Path(self.current_image_path).stem}.jpg"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu ảnh kết quả nhận diện",
            default_name,
            "JPEG Image (*.jpg);;PNG Image (*.png)"
        )
        if save_path:
            cv2.imwrite(save_path, target_img)
            QMessageBox.information(self, "Thành Công", f"Đã lưu ảnh kết quả tại:\n{save_path}")

    def set_as_reference(self):
        if not self.current_result_data:
            return
        reply = QMessageBox.question(
            self,
            "Xác Nhận Đặt Mạch Chuẩn",
            f"Bạn có chắc muốn đặt ảnh '{self.current_result_data['filename']}' làm Mạch Mẫu Chuẩn (Reference)?\n"
            f"Tất cả các bo mạch sau sẽ được đối chiếu căn chỉnh theo mạch này.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                from backend.app.services.pcb_engine import PCBPipeline
                pipeline = PCBPipeline()
                pipeline.create_reference_from_image(self.current_result_data["raw_image"])
                QMessageBox.information(
                    self,
                    "Hoàn Tất",
                    "Đã lưu thành công bo mạch này làm Mạch Chuẩn mới!\n"
                    "Bây giờ khi kiểm tra các bo mạch cùng loại, hệ thống sẽ tự động so sánh phát hiện thiếu linh kiện."
                )
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu mạch chuẩn: {e}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PCB Vision AI")
    window = PCBMainWindow()
    window.show()

    # Nếu truyền đường dẫn file ảnh qua tham số dòng lệnh
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        window.process_image_file(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import threading
import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from pcb_engine import (
    DEFAULT_CONF,
    DEFAULT_EXTRA_ALLOWED,
    DEFAULT_IMGSZ,
    DEFAULT_MISSING_ALLOWED,
    DEFAULT_TOLERANCE_RATIO,
    PCBPipeline,
)

# Khởi tạo Flask App
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static",
)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # Cho phép tối đa 32MB

# Khởi tạo PCB Pipeline
print("[INFO] Khởi tạo PCBPipeline...")
pipeline = PCBPipeline()


def np_image_to_base64(image_np, quality=90):
    """Chuyển mảng OpenCV (BGR) thành Data URI base64 JPEG"""
    if image_np is None:
        return None
    success, buffer = cv2.imencode(
        ".jpg", image_np, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        return None
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def read_image_from_request():
    """Đọc ảnh từ file upload (multipart/form-data) hoặc base64 trong JSON body"""
    # 1. Kiểm tra File trong multipart upload
    if "image" in request.files:
        file = request.files["image"]
        if file.filename != "":
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                return image, file.filename
    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                return image, file.filename

    # 2. Kiểm tra Base64 trong JSON
    if request.is_json:
        data = request.get_json(silent=True) or {}
        b64_data = data.get("image") or data.get("image_base64")
        if b64_data:
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(b64_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if image is not None:
                    return image, "base64_upload.jpg"
            except Exception as e:
                raise ValueError(f"Không thể giải mã dữ liệu ảnh base64: {str(e)}")

    # 3. Kiểm tra Base64 trong Form data
    if "image_base64" in request.form:
        b64_data = request.form["image_base64"]
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is not None:
            return image, "form_base64.jpg"

    return None, None


# ============================================================
# CÁC ROUTE TRANG GIAO DIỆN
# ============================================================

@app.route("/")
def index():
    """Giao diện web người dùng chính"""
    return render_template("index.html")


@app.route("/reference_detected.jpg")
def serve_ref_detected():
    return send_from_directory(".", "reference_detected.jpg")


@app.route("/2.jpg")
def serve_ref_sample():
    return send_from_directory(".", "2.jpg")


# ============================================================
# CÁC API ENDPOINTS
# ============================================================

@app.route("/api/info", methods=["GET"])
def api_info():
    """Lấy thông tin hệ thống, mô hình YOLO và trạng thái reference"""
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    class_names = []
    if pipeline.model is not None and hasattr(pipeline.model, "names"):
        class_names = [pipeline.model.names[k] for k in sorted(pipeline.model.names.keys())]

    return jsonify({
        "status": "success",
        "model_name": Path(pipeline.model_path).name,
        "framework": "Ultralytics YOLOv8 / PyTorch + OpenCV",
        "device": str(pipeline.device),
        "device_name": device_name,
        "is_cuda": pipeline.use_cuda,
        "total_classes": len(class_names),
        "classes": class_names,
        "reference": {
            "loaded": len(pipeline.reference_components) > 0,
            "components_count": len(pipeline.reference_components),
            "image_name": Path(pipeline.ref_image_path).name,
            "orb_aligned_ready": pipeline.aligner is not None and pipeline.aligner.valid,
        },
        "default_parameters": {
            "conf": DEFAULT_CONF,
            "imgsz": DEFAULT_IMGSZ,
            "tolerance_ratio": DEFAULT_TOLERANCE_RATIO,
            "extra_allowed": DEFAULT_EXTRA_ALLOWED,
            "missing_allowed": DEFAULT_MISSING_ALLOWED,
        }
    })


@app.route("/api/reference", methods=["GET"])
def api_get_reference():
    """Lấy chi tiết mạch tham chiếu và danh sách linh kiện chuẩn"""
    ref_b64 = None
    if pipeline.reference_image is not None:
        ref_b64 = np_image_to_base64(pipeline.reference_image)

    detected_ref_b64 = None
    ref_det_path = Path("reference_detected.jpg")
    if ref_det_path.exists():
        ref_det_img = cv2.imread(str(ref_det_path))
        if ref_det_img is not None:
            detected_ref_b64 = np_image_to_base64(ref_det_img)

    return jsonify({
        "status": "success",
        "reference_data": pipeline.reference_data,
        "components_count": len(pipeline.reference_components),
        "components": pipeline.reference_components,
        "reference_image_base64": ref_b64,
        "detected_image_base64": detected_ref_b64,
    })


@app.route("/api/reference/update", methods=["POST"])
def api_update_reference():
    """Tải lên ảnh PCB mẫu mới và tự động tạo lại reference map"""
    try:
        image, filename = read_image_from_request()
        if image is None:
            return jsonify({"status": "error", "message": "Vui lòng chọn hoặc gửi file ảnh PCB mẫu mới."}), 400

        conf = float(request.form.get("conf") or (request.is_json and request.json.get("conf")) or DEFAULT_CONF)
        imgsz = int(request.form.get("imgsz") or (request.is_json and request.json.get("imgsz")) or DEFAULT_IMGSZ)

        result = pipeline.create_reference_from_image(image, conf=conf, imgsz=imgsz)
        annotated_b64 = np_image_to_base64(result["annotated_image"])

        return jsonify({
            "status": "success",
            "message": f"Đã tạo thành công reference map với {result['components_count']} linh kiện.",
            "components_count": result["components_count"],
            "components": result["components"],
            "annotated_image": annotated_b64,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sample", methods=["GET"])
def api_sample_image():
    """Lấy ảnh mẫu 2.jpg để test nhanh trên giao diện"""
    sample_path = Path("2.jpg")
    if not sample_path.exists():
        return jsonify({"status": "error", "message": "Không tìm thấy file ảnh 2.jpg"}), 404

    img = cv2.imread(str(sample_path))
    if img is None:
        return jsonify({"status": "error", "message": "Lỗi khi đọc file 2.jpg"}), 500

    return jsonify({
        "status": "success",
        "filename": "2.jpg",
        "image_base64": np_image_to_base64(img),
        "width": img.shape[1],
        "height": img.shape[0],
    })


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """API Nhận diện linh kiện YOLO thuần túy trên ảnh gửi lên"""
    try:
        image, filename = read_image_from_request()
        if image is None:
            return jsonify({"status": "error", "message": "Không tìm thấy ảnh tải lên."}), 400

        # Tham số cấu hình
        data = request.get_json(silent=True) or {} if request.is_json else request.form
        conf = float(data.get("conf", DEFAULT_CONF))
        imgsz = int(data.get("imgsz", DEFAULT_IMGSZ))

        result = pipeline.run_detection_only(image, conf=conf, imgsz=imgsz)
        annotated_b64 = np_image_to_base64(result["annotated_image"])

        return jsonify({
            "status": "success",
            "filename": filename,
            "total_count": result["total_count"],
            "class_counts": result["class_counts"],
            "detections": result["detections"],
            "image_size": result["image_size"],
            "latency_ms": result["latency_ms"],
            "annotated_image": annotated_b64,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    """API Kiểm tra lỗi PCB toàn diện (So khớp với PCB mẫu tham chiếu)"""
    try:
        image, filename = read_image_from_request()
        if image is None:
            return jsonify({"status": "error", "message": "Không tìm thấy ảnh tải lên để kiểm tra."}), 400

        # Tham số cấu hình
        data = request.get_json(silent=True) or {} if request.is_json else request.form
        conf = float(data.get("conf", DEFAULT_CONF))
        imgsz = int(data.get("imgsz", DEFAULT_IMGSZ))
        tolerance_ratio = float(data.get("tolerance_ratio", DEFAULT_TOLERANCE_RATIO))
        extra_allowed = int(data.get("extra_allowed", DEFAULT_EXTRA_ALLOWED))
        missing_allowed = int(data.get("missing_allowed", DEFAULT_MISSING_ALLOWED))

        auto_crop_val = data.get("auto_crop", "true")
        if isinstance(auto_crop_val, str):
            auto_crop = auto_crop_val.lower() in ("true", "1", "yes")
        else:
            auto_crop = bool(auto_crop_val)

        result = pipeline.inspect_pcb(
            frame=image,
            conf=conf,
            imgsz=imgsz,
            tolerance_ratio=tolerance_ratio,
            extra_allowed=extra_allowed,
            missing_allowed=missing_allowed,
            auto_crop=auto_crop,
        )

        annotated_b64 = np_image_to_base64(result["annotated_image"])
        crop_b64 = np_image_to_base64(result["board_crop_image"])

        return jsonify({
            "status": "success",
            "verdict": result["status"],               # "OK" hoặc "FAULTY"
            "verdict_vi": result["status_vi"],         # "MẠCH ĐẠT CHUẨN" hoặc "MẠCH BỊ LỖI"
            "is_ok": result["status"] == "OK",
            "aligned": result["aligned"],
            "alignment_method": result["alignment_method"],
            "crop_method": result["crop_method"],
            "stats": result["stats"],
            "detections": result["detections"],
            "missing_components_details": result["missing_components_details"],
            "latency_ms": result["latency_ms"],
            "annotated_image": annotated_b64,
            "board_crop_image": crop_b64,
            "filename": filename,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# QUẢN LÝ CAMERA STREAM TRỰC TIẾP
# ============================================================

class CameraManager:
    def __init__(self):
        self.source = os.environ.get("PCB_CAMERA_SOURCE", "http://192.168.0.190:8080/video")
        try:
            self.source = int(self.source)
        except (ValueError, TypeError):
            pass
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.fps = 0.0
        self.running = False
        self.thread = None
        self.is_connected = False
        self.last_error = ""

    def connect(self, source=None):
        if source is not None:
            try:
                self.source = int(source)
            except (ValueError, TypeError):
                self.source = str(source).strip()

        self.disconnect()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        # Đợi tối đa 2.5s để xác nhận luồng camera
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            with self.lock:
                if self.is_connected and self.frame is not None:
                    return True, "Kết nối camera thành công"
                if not self.running and self.last_error:
                    return False, self.last_error
            time.sleep(0.1)

        with self.lock:
            if self.is_connected:
                return True, "Camera đã kết nối, đang nạp hình..."
            return False, self.last_error or f"Hết thời gian chờ kết nối tới {self.source}"

    def disconnect(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.is_connected = False
        return True, "Đã ngắt kết nối camera"

    def _worker(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            with self.lock:
                self.last_error = f"Lỗi mở camera: {str(e)}"
                self.is_connected = False
                self.running = False
            return

        if not cap.isOpened():
            with self.lock:
                self.last_error = f"Không thể mở nguồn camera: {self.source}"
                self.is_connected = False
                self.running = False
            return

        with self.lock:
            self.cap = cap
            self.is_connected = True
            self.last_error = ""

        last_time = time.monotonic()
        frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                frame_count += 1
                now = time.monotonic()
                if now - last_time >= 1.0:
                    with self.lock:
                        self.fps = round(frame_count / (now - last_time), 1)
                    frame_count = 0
                    last_time = now
            else:
                time.sleep(0.04)
                if not self.running:
                    break
                try:
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(self.source)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    with self.lock:
                        self.cap = cap
                except Exception:
                    pass

        try:
            cap.release()
        except Exception:
            pass
        with self.lock:
            self.is_connected = False

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.frame_id, self.fps
            return None, 0, 0.0

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "connected": self.is_connected,
                "source": str(self.source),
                "frame_id": self.frame_id,
                "fps": self.fps,
                "last_error": self.last_error,
            }


camera_mgr = CameraManager()


@app.route("/api/camera/connect", methods=["POST"])
def api_camera_connect():
    """Bắt đầu kết nối nguồn Camera backend (IP camera hoặc webcam)"""
    data = request.get_json(silent=True) or {} if request.is_json else request.form
    source = data.get("source", "http://192.168.0.190:8080/video")
    ok, msg = camera_mgr.connect(source)
    return jsonify({
        "status": "success" if ok else "error",
        "message": msg,
        "camera": camera_mgr.get_status(),
    }), (200 if ok else 400)


@app.route("/api/camera/disconnect", methods=["POST"])
def api_camera_disconnect():
    """Ngắt kết nối luồng camera backend"""
    ok, msg = camera_mgr.disconnect()
    return jsonify({"status": "success", "message": msg, "camera": camera_mgr.get_status()})


@app.route("/api/camera/status", methods=["GET"])
def api_camera_status():
    """Kiểm tra trạng thái kết nối và FPS camera"""
    return jsonify({"status": "success", "camera": camera_mgr.get_status()})


@app.route("/api/camera/stream")
def api_camera_stream():
    """Luồng video MJPEG thời gian thực phát thẳng lên trình duyệt"""
    def generate():
        while True:
            frame, frame_id, fps = camera_mgr.get_frame()
            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "CAMERA CHUA KET NOI", (140, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 100, 110), 2, cv2.LINE_AA)
                cv2.putText(blank, "Chon nguon va nhan 'Ket Noi Camera'", (130, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 70, 80), 1, cv2.LINE_AA)
                success, buffer = cv2.imencode(".jpg", blank)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                time.sleep(0.4)
                continue

            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                time.sleep(0.03)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera/scan", methods=["POST"])
def api_camera_scan():
    """Chụp frame mới nhất của Camera backend và quét kiểm tra lỗi ngay lập tức"""
    frame, frame_id, fps = camera_mgr.get_frame()
    if frame is None:
        return jsonify({
            "status": "error",
            "message": "Camera backend chưa kết nối hoặc chưa nhận được khung hình. Hãy nhấn 'Kết Nối Camera' trước."
        }), 400

    data = request.get_json(silent=True) or {} if request.is_json else request.form
    conf = float(data.get("conf", DEFAULT_CONF))
    imgsz = int(data.get("imgsz", DEFAULT_IMGSZ))
    tolerance_ratio = float(data.get("tolerance_ratio", DEFAULT_TOLERANCE_RATIO))

    result = pipeline.inspect_pcb(frame, conf=conf, imgsz=imgsz, tolerance_ratio=tolerance_ratio)
    annotated_b64 = np_image_to_base64(result["annotated_image"])
    crop_b64 = np_image_to_base64(result["board_crop_image"])

    return jsonify({
        "status": "success",
        "verdict": result["status"],
        "verdict_vi": result["status_vi"],
        "is_ok": result["status"] == "OK",
        "aligned": result["aligned"],
        "alignment_method": result["alignment_method"],
        "crop_method": result["crop_method"],
        "stats": result["stats"],
        "detections": result["detections"],
        "missing_components_details": result["missing_components_details"],
        "latency_ms": result["latency_ms"],
        "annotated_image": annotated_b64,
        "board_crop_image": crop_b64,
        "frame_id": frame_id,
        "camera_fps": fps,
    })


@app.route("/api/camera/snapshot", methods=["POST"])
def api_camera_snapshot():
    """Chụp frame tĩnh và chạy kiểm tra trực tiếp"""
    try:
        data = request.get_json(silent=True) or {} if request.is_json else request.form
        source = data.get("source", "http://192.168.0.190:8080/video")
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return jsonify({
                "status": "error",
                "message": f"Không kết nối được với camera nguồn: {source}. Hãy kiểm tra IP webcam hoặc webcam máy tính."
            }), 400

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return jsonify({"status": "error", "message": "Không nhận được frame từ camera."}), 500

        conf = float(data.get("conf", DEFAULT_CONF))
        imgsz = int(data.get("imgsz", DEFAULT_IMGSZ))
        tolerance_ratio = float(data.get("tolerance_ratio", DEFAULT_TOLERANCE_RATIO))

        result = pipeline.inspect_pcb(frame, conf=conf, imgsz=imgsz, tolerance_ratio=tolerance_ratio)
        annotated_b64 = np_image_to_base64(result["annotated_image"])

        return jsonify({
            "status": "success",
            "verdict": result["status"],
            "verdict_vi": result["status_vi"],
            "is_ok": result["status"] == "OK",
            "aligned": result["aligned"],
            "alignment_method": result["alignment_method"],
            "stats": result["stats"],
            "latency_ms": result["latency_ms"],
            "annotated_image": annotated_b64,
            "raw_image": np_image_to_base64(frame),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n" + "=" * 60)
    print(f"🚀 PCB Detection & Inspection Server khởi động thành công!")
    print(f"📡 Web UI & API URL : http://localhost:{port}")
    print(f"⚙️  Thiết bị PyTorch : {pipeline.device}")
    print(f"📦 Mô hình YOLO     : {pipeline.model_path}")
    print(f"🎯 Mạch tham chiếu  : {len(pipeline.reference_components)} linh kiện")
    print(f"=" * 60 + "\n")
    app.run(host=host, port=port, debug=False)

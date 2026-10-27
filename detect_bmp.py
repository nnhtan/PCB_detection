import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
import cv2

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def run_detection(
    image_path="Image_20261019174732857.bmp",
    model_path="best.pt",
    conf=0.25,
    imgsz=1536,
    output_image_path="result_detected.jpg",
):
    print("=" * 65)
    print("      PCB VISION AI - KIỂM TRA NHẬN DIỆN ẢNH TĨNH (BMP)")
    print("=" * 65)

    # 1. Kiểm tra file ảnh và model
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"[LỖI] Không tìm thấy file ảnh: {image_path}")
        return

    mod_file = Path(model_path)
    if not mod_file.exists():
        alt_model = Path("backend/models/best.pt")
        if alt_model.exists():
            mod_file = alt_model
        else:
            print(f"[LỖI] Không tìm thấy file model: {model_path}")
            return

    print(f"[*] Ảnh đầu vào: {img_file.name}")
    print(f"[*] Đường dẫn đầy đủ: {img_file.resolve()}")
    print(f"[*] Model sử dụng: {mod_file.name}")

    # 2. Đọc ảnh
    image = cv2.imread(str(img_file))
    if image is None:
        print("[LỖI] Không thể đọc ảnh bằng OpenCV.")
        return

    h, w = image.shape[:2]
    print(f"[*] Kích thước ảnh: {w} x {h} px")

    # 3. Load YOLO Model
    print("\n[1] Đang nạp model YOLO...")
    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    dev_str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"    Thiết bị tính toán: {dev_str}")

    model = YOLO(str(mod_file))

    # 4. Chạy nhận diện tĩnh
    print(f"\n[2] Đang chạy nhận diện linh kiện (conf={conf}, imgsz={imgsz})...")
    start_time = time.time()
    results = model.predict(
        source=image,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    inference_time = (time.time() - start_time) * 1000
    print(f"    Thời gian suy luận: {inference_time:.1f} ms")

    result = results[0]
    boxes = result.boxes
    num_detections = len(boxes)
    print(f"\n=> TỔNG SỐ LINH KIỆN TÌM THẤY: {num_detections}")

    if num_detections == 0:
        print("[CẢNH BÁO] Không phát hiện được linh kiện nào.")
        return

    # Thống kê chi tiết
    class_names = model.names
    detected_classes = []
    detection_details = []

    # Thu thập các hộp IC để lọc bỏ nhận diện giả trên chân/thân IC
    ic_boxes = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_name = class_names.get(cls_id, f"class_{cls_id}").lower()
        if cls_name == "ic":
            ic_boxes.append(box.xyxy[0].cpu().numpy().tolist())

    valid_boxes = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        confidence = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        cx = (xyxy[0] + xyxy[2]) / 2.0
        cy = (xyxy[1] + xyxy[3]) / 2.0

        # Lọc bỏ nhận diện giả trên chân IC
        if cls_name.lower() != "ic" and any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in ic_boxes):
            continue

        valid_boxes.append(box)
        detected_classes.append(cls_name)
        detection_details.append({
            "class": cls_name,
            "confidence": round(confidence, 4),
            "bbox": [round(x, 1) for x in xyxy]
        })

    print(f"\n=> SỐ LINH KIỆN HỢP LỆ (ĐÃ LỌC BỎ NHẬN DIỆN GIẢ TRÊN IC): {len(valid_boxes)}")

    counter = Counter(detected_classes)
    print("\n--- BẢNG THỐNG KÊ CHI TIẾT THEO LOẠI LINH KIỆN ---")
    print(f"{'STT':<5} | {'Loại Linh Kiện':<25} | {'Số Lượng':<10}")
    print("-" * 46)
    for idx, (name, count) in enumerate(counter.most_common(), 1):
        print(f"{idx:<5} | {name:<25} | {count:<10}")
    print("-" * 46)

    # 5. Vẽ Bounding Boxes cho các linh kiện hợp lệ
    annotated_image = image.copy()
    for d in detection_details:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        label = f"{d['class']} {d['confidence']:.2f}"
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(output_image_path, annotated_image)
    print(f"\n[3] Đã lưu ảnh kết quả nhận diện (đã lọc sạch chân IC):")
    print(f"    -> {Path(output_image_path).resolve()}")

    # 6. Kiểm tra đối chiếu với mạch chuẩn (QA inspection)
    print("\n" + "=" * 65)
    print("      KIỂM ĐỊNH CHẤT LƯỢNG MẠCH (QA INSPECTION)")
    print("=" * 65)

    ref_json = "backend/storage/reference_components.json"
    ref_img = "backend/storage/reference.jpg"
    if not Path(ref_json).exists():
        ref_json = "reference_components.json"
    if not Path(ref_img).exists():
        ref_img = "Image_20261019190958053.bmp"

    try:
        from backend.app.services.pcb_engine import PCBPipeline
        pipeline = PCBPipeline(
            model_path=str(mod_file),
            ref_json_path=ref_json,
            ref_image_path=ref_img,
        )

        qa_res = pipeline.inspect_pcb(
            frame=image,
            conf=conf,
            imgsz=imgsz,
            auto_crop=True,
        )

        status = qa_res.get("status")
        align_method = qa_res.get("alignment_method")
        missing = qa_res.get("missing_components_details", [])
        matched = qa_res.get("stats", {}).get("matched", 0)
        ref_count = qa_res.get("stats", {}).get("reference_count", 0)

        print(f"[*] Phương pháp căn chỉnh: {align_method}")
        print(f"[*] Trạng thái kiểm định: {status}")
        print(f"[*] Số linh kiện khớp chuẩn: {matched}/{ref_count}")
        print(f"[*] Số linh kiện thiếu: {len(missing)}")

        if status == "OK":
            print("\n=> KẾT LUẬN: [ OK ] - Mạch đạt chuẩn so với mạch tham chiếu!")
        elif status == "FAULTY":
            print("\n=> KẾT LUẬN: [ FAULTY ] - Mạch bị lỗi: Thiếu linh kiện!")
            if missing:
                print("   Các linh kiện bị thiếu cần bổ sung:")
                for m in missing[:15]:
                    m_id = m.get('id') or m.get('ref_id')
                    m_x = m.get('x') or m.get('ref_x')
                    m_y = m.get('y') or m.get('ref_y')
                    print(f"    - ID {m_id}: {m.get('class')} tại tọa độ chuẩn ({m_x}, {m_y})")
        else:
            print("\n=> KẾT LUẬN: [ UNALIGNED ] - Chưa căn chỉnh được góc mạch với mạch tham chiếu hiện tại.")

        if "annotated_image" in qa_res and qa_res["annotated_image"] is not None:
            qa_output = "result_inspection.jpg"
            cv2.imwrite(qa_output, qa_res["annotated_image"])
            print(f"[*] Đã lưu ảnh kiểm định QA: {Path(qa_output).resolve()}")

    except Exception as e:
        print(f"[*] Thông tin QA: {e}")

    print("\n" + "=" * 65)
    print("HOÀN TẤT NHẬN DIỆN!")
    print("=" * 65)

if __name__ == "__main__":
    img_name = sys.argv[1] if len(sys.argv) > 1 else "Image_20261019174732857.bmp"
    run_detection(image_path=img_name)

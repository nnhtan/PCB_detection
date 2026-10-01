import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# 1. CẤU HÌNH
# ============================================================

MODEL_PATH = "best.pt"
REFERENCE_IMAGE = "2.jpg"

OUTPUT_JSON = "reference_components.json"
OUTPUT_IMAGE = "reference_detected.jpg"

CONF = 0.25
IMGSZ = 960
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Nếu model của bạn dùng tên class khác, không cần sửa phần này.
# Tên linh kiện sẽ lấy trực tiếp từ model.names.


# ============================================================
# 2. HÀM TẠO COMPONENT MAP
# ============================================================

def create_reference_map():
    image_path = Path(REFERENCE_IMAGE)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ảnh tham chiếu: {image_path.resolve()}"
        )

    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {model_path.resolve()}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError("Không đọc được ảnh PCB gốc.")

    h, w = image.shape[:2]

    print(f"[INFO] Reference image: {w} x {h}")
    print(f"[INFO] Loading model: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=image,
        conf=CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        agnostic_nms=True,
        verbose=False
    )

    if not results:
        raise RuntimeError("YOLO không trả về kết quả.")

    result = results[0]

    components = []
    annotated = image.copy()

    if result.boxes is None or len(result.boxes) == 0:
        print("[WARNING] Không phát hiện linh kiện nào.")
    else:
        boxes = result.boxes

        for idx in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().tolist()

            cls_id = int(boxes.cls[idx].item())
            conf = float(boxes.conf[idx].item())

            class_name = str(model.names[cls_id])

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            component = {
                "id": idx + 1,
                "class_id": cls_id,
                "class": class_name,
                "confidence_reference": round(conf, 4),
                "x": round(cx, 2),
                "y": round(cy, 2),
            }
            components.append(component)

            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                label,
                (int(x1), max(int(y1) - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    output_data = {
        "image": image_path.name,
        "image_width": w,
        "image_height": h,
        "components_count": len(components),
        "components": components,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=2)

    if not cv2.imwrite(OUTPUT_IMAGE, annotated):
        raise RuntimeError(f"Không lưu được ảnh kết quả: {OUTPUT_IMAGE}")

    print(f"[INFO] Detected components: {len(components)}")
    print(f"[INFO] Saved component map: {OUTPUT_JSON}")
    print(f"[INFO] Saved annotated image: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    create_reference_map()
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# 1. CẤU HÌNH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_STORAGE = PROJECT_DIR / "backend" / "storage"
MODEL_PATH = (
    PROJECT_DIR / "backend" / "models" / "best.pt"
    if (PROJECT_DIR / "backend" / "models" / "best.pt").is_file()
    else PROJECT_DIR / "best.pt"
)
REFERENCE_IMAGE = PROJECT_DIR / "Image_20261019190958053.bmp"

REFERENCE_OUTPUT_IMAGE = BACKEND_STORAGE / "reference.jpg"
OUTPUT_JSON = BACKEND_STORAGE / "reference_components.json"
OUTPUT_IMAGE = BACKEND_STORAGE / "reference_detected.jpg"

CONF = 0.35
IMGSZ = 1536
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

        # Thu thập các hộp IC để lọc bỏ nhận diện giả trên chân/thân IC
        ic_boxes = []
        for idx in range(len(boxes)):
            cls_id = int(boxes.cls[idx].item())
            if str(model.names[cls_id]).lower() == "ic":
                ic_boxes.append(boxes.xyxy[idx].cpu().numpy().tolist())

        comp_idx = 1
        for idx in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().tolist()

            cls_id = int(boxes.cls[idx].item())
            conf = float(boxes.conf[idx].item())

            class_name = str(model.names[cls_id])

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Loại bỏ nhận diện giả (ví dụ: resistor nhận diện nhầm trên chân kim loại của IC)
            if class_name.lower() != "ic" and any(
                b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in ic_boxes
            ):
                print(f"[FILTER] Bỏ nhận diện giả trên chân IC: {class_name} tại ({cx:.1f}, {cy:.1f})")
                continue

            component = {
                "id": comp_idx,
                "class_id": cls_id,
                "class": class_name,
                "confidence_reference": round(conf, 4),
                "x": round(cx, 2),
                "y": round(cy, 2),
            }
            comp_idx += 1
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

    if not components:
        raise RuntimeError(
            "Không phát hiện được linh kiện; không lưu reference map rỗng."
        )

    output_data = {
        "image": REFERENCE_OUTPUT_IMAGE.name,
        "image_width": w,
        "image_height": h,
        "components_count": len(components),
        "components": components,
    }

    BACKEND_STORAGE.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(REFERENCE_OUTPUT_IMAGE), image):
        raise RuntimeError(f"Không lưu được ảnh reference: {REFERENCE_OUTPUT_IMAGE}")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=2)

    # Đồng bộ ra thư mục gốc
    root_json = PROJECT_DIR / "reference_components.json"
    with open(root_json, "w", encoding="utf-8") as root_file:
        json.dump(output_data, root_file, ensure_ascii=False, indent=2)

    if not cv2.imwrite(str(OUTPUT_IMAGE), annotated):
        raise RuntimeError(f"Không lưu được ảnh kết quả: {OUTPUT_IMAGE}")
    
    root_detected = PROJECT_DIR / "reference_detected.jpg"
    cv2.imwrite(str(root_detected), annotated)

    print(f"[INFO] Detected components: {len(components)}")
    print(f"[INFO] Saved component map: {OUTPUT_JSON} & {root_json}")
    print(f"[INFO] Saved annotated image: {OUTPUT_IMAGE} & {root_detected}")


if __name__ == "__main__":
    create_reference_map()
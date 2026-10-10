# PCB Vision AI

Ứng dụng kiểm tra chất lượng PCB bằng YOLO và OpenCV, gồm backend FastAPI và giao diện React. Người dùng quản lý riêng ảnh mạch chuẩn, tải ảnh PCB cần quét, xem nhận diện linh kiện và đối chiếu kết quả khi hệ thống căn chỉnh được mạch.

## Chức năng

- Tạo reference map từ ảnh PCB chuẩn; lưu ảnh, tọa độ và lớp linh kiện.
- Quét ảnh riêng bằng YOLO hoặc so sánh ảnh PCB với mạch tham chiếu.
- Hiển thị linh kiện khớp, thiếu, thừa và trạng thái chưa căn chỉnh.
- Kết nối webcam/camera IP, xem luồng MJPEG và quét frame mới nhất.
- Giao diện React responsive; backend cung cấp Swagger UI tại `/docs`.

## Cấu trúc

```text
backend/
  app/
    api/v1/             FastAPI routes cho QA, detect, camera, reference
    core/               Cấu hình hệ thống
    schemas/            Pydantic schemas
    services/           YOLO/QA pipeline và quản lý camera
    utils/              Tiện ích ảnh và metrics
  models/               Đặt model YOLO tại đây
  storage/              Reference map, ảnh reference và cache
  main.py
frontend/
  src/
    components/         Giao diện layout, inspector, camera, reference
    hooks/              Hooks gọi inspection và camera
    pages/              Các trang ứng dụng
    services/           API client
    store/              Trạng thái Zustand
  package.json
  tailwind.config.js
```

Các file CLI cũ `machpcbgoc.py` và `test.py` vẫn được giữ để tạo reference hoặc quét camera từ cửa sổ desktop.

## Yêu cầu

- Python 3.10+
- Node.js 18+
- Weights YOLO: `backend\models\best.pt` (có thể cấu hình bằng `PCB_MODEL_PATH`)

Trong bản repo hiện tại, backend tự dùng `best.pt` và `2.jpg` ở thư mục gốc nếu chưa có model/ảnh tương ứng trong `backend\models` hoặc `backend\storage`. Tải model của bạn vào `backend\models\best.pt` cho bố cục triển khai mới. Ảnh/reference đang có trong repo được giữ nguyên; ảnh mạch mới tải lên được ghi vào `backend\storage`.

## Chạy phát triển

Mở hai terminal từ thư mục gốc repo.

Terminal 1 — cài và chạy backend:

```powershell
python -m pip install -r backend\requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 — chạy frontend:

```powershell
cd frontend
npm install
npm run dev
```

Mở địa chỉ Vite hiển thị trong terminal (mặc định `http://localhost:5173`). FastAPI docs ở `http://localhost:8000/docs`; health check ở `http://localhost:8000/health`.

Để build frontend và phục vụ cả ứng dụng từ FastAPI:

```powershell
cd frontend
npm run build
cd ..
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Sau khi build, mở `http://localhost:8000`.

## Quy trình kiểm tra PCB

1. Mở trang **Mạch tham chiếu**, tải ảnh PCB chuẩn và bấm **Lưu mạch chuẩn**. Backend chạy YOLO, ghi reference map và ảnh đã đánh dấu.
2. Vào **Kiểm tra PCB**, tải riêng ảnh PCB cần quét.
3. Điều chỉnh confidence/dung sai nếu cần và bấm **Bắt đầu kiểm tra PCB**.
4. Kết quả `OK`/`FAULTY` chỉ được kết luận khi ảnh scan căn chỉnh được với reference. Nếu không căn chỉnh được, hệ thống vẫn trả về detections và trạng thái `UNALIGNED`, không kết luận thiếu/thừa.

Trang **Nhận diện YOLO** chạy detector độc lập, không cần mạch chuẩn.

## API chính

Các API có sẵn tại `/api/...`; cùng endpoint được expose tương thích dưới prefix `/api/v1/...`.

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/api/info` | Thông tin model, thiết bị và reference |
| `GET` | `/api/reference` | Dữ liệu và ảnh mạch chuẩn |
| `POST` | `/api/reference/update` | Tải ảnh chuẩn và tạo reference map |
| `GET` | `/api/sample` | Lấy ảnh mẫu PCB |
| `POST` | `/api/detect` | Nhận diện YOLO trên một ảnh |
| `POST` | `/api/inspect` | Quét và đối chiếu ảnh với reference |
| `GET` | `/api/camera/status` | Trạng thái camera |
| `POST` | `/api/camera/connect` | Kết nối webcam hoặc camera IP |
| `POST` | `/api/camera/disconnect` | Ngắt kết nối |
| `GET` | `/api/camera/stream` | Video MJPEG |
| `POST` | `/api/camera/scan` | Quét frame mới nhất |
| `POST` | `/api/camera/snapshot` | Chụp và kiểm tra nguồn camera trực tiếp |

Các endpoint upload nhận `multipart/form-data`, tên trường ảnh là `image`. Giới hạn dung lượng ảnh là 32 MB.

## Cấu hình

| Biến môi trường | Mặc định | Mục đích |
|---|---|---|
| `PCB_MODEL_PATH` | `backend\models\best.pt` (fallback `best.pt` ở repo root) | Đường dẫn weights YOLO |
| `PCB_REFERENCE_PATH` | `backend\storage\reference_components.json` | Dữ liệu tọa độ mạch chuẩn |
| `PCB_REFERENCE_IMAGE_PATH` | `backend\storage\reference.jpg` | Ảnh chuẩn đầu ra/đọc lúc khởi động |
| `PCB_SAMPLE_IMAGE_PATH` | `backend\storage\sample.jpg` (fallback `2.jpg` ở repo root) | Ảnh dùng thử |
| `PCB_CAMERA_SOURCE` | `0` | Webcam index hoặc URL stream camera |

## Smoke test API

Khởi chạy backend, sau đó ở terminal khác từ thư mục gốc:

```powershell
python test_api.py
```

Hoặc đặt `PCB_API_URL` nếu API không chạy ở `http://127.0.0.1:8000`.

## Camera desktop

`test.py` chạy luồng camera bằng cửa sổ OpenCV. Mặc định dùng camera IP cũ của dự án; đổi nguồn bằng PowerShell:

```powershell
$env:PCB_CAMERA_SOURCE = "0"
python test.py
```

Nhấn `s` để quét frame hoặc `q` để thoát.

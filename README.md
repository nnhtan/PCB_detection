# PCB Vision AI

Ứng dụng kiểm tra chất lượng bảng mạch in (PCB) bằng YOLO và OpenCV. Dự án gồm backend FastAPI xử lý ảnh/camera và frontend React cho thao tác quét, xem kết quả, quản lý mạch chuẩn.

## Mục lục

- [Chức năng](#chức-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu](#yêu-cầu)
- [Cài đặt và chạy trên Windows](#cài-đặt-và-chạy-trên-windows)
- [Hướng dẫn sử dụng giao diện](#hướng-dẫn-sử-dụng-giao-diện)
- [API](#api)
- [Cấu hình](#cấu-hình)
- [Kiểm tra API](#kiểm-tra-api)
- [Chạy chương trình camera desktop](#chạy-chương-trình-camera-desktop)
- [Xử lý sự cố](#xử-lý-sự-cố)

## Chức năng

- **Danh mục mạch tham chiếu:** lưu nhiều ảnh PCB chuẩn với tên riêng, chọn lại mạch cần dùng bất cứ lúc nào; lựa chọn hiện tại được giữ sau khi backend khởi động lại.
- **Kiểm tra PCB:** tải ảnh PCB cần kiểm tra riêng, căn chỉnh với mạch chuẩn, so sánh vị trí và phân loại linh kiện thiếu/thừa.
- **Nhận diện YOLO:** phát hiện và thống kê linh kiện trong ảnh độc lập, không yêu cầu reference.
- **Camera trực tiếp:** kết nối webcam hoặc camera IP, xem MJPEG, quét từng frame hoặc đặt quét tự động mỗi 3, 5, hoặc 10 giây.
- **Kết quả trực quan:** ảnh đánh dấu detections, số lượng linh kiện, kết quả đối chiếu và thời gian xử lý.
- **API docs:** FastAPI Swagger UI tại `/docs`.

Việc kết luận `OK` hoặc `FAULTY` cần căn chỉnh được ảnh với mạch chuẩn. Nếu ảnh đã nhận diện được linh kiện nhưng không căn chỉnh được, kết quả có trạng thái `UNALIGNED`; hệ thống không giả kết luận linh kiện thiếu/thừa.

## Cấu trúc dự án

```text
PCB_detection/
├── backend/
│   ├── app/
│   │   ├── api/v1/              # API system, inspect, detect, camera, reference
│   │   ├── core/                # Cấu hình và logger
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── services/            # Pipeline YOLO/QA, căn chỉnh và camera
│   │   └── utils/               # Tiện ích ảnh và metrics
│   ├── models/                  # Đặt model best.pt tại đây
│   ├── storage/                 # Reference map và ảnh PCB
│   ├── main.py                 # Entrypoint FastAPI
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/          # Layout, inspector, camera, reference
│   │   ├── hooks/               # Hooks dùng API và trạng thái camera
│   │   ├── pages/               # Các trang QA, YOLO, camera, reference, docs
│   │   ├── services/             # API client
│   │   └── store/                # Trạng thái Zustand
│   ├── package.json
│   └── vite.config.js
├── best.pt                      # Model hiện có, dùng làm fallback
├── 2.jpg                        # Ảnh PCB mẫu/reference hiện có
├── machpcbgoc.py                # Tạo reference bằng script desktop
├── test.py                      # Kiểm tra PCB bằng camera desktop
└── test_api.py                  # Smoke test API
```

Pipeline chính nằm trong `backend/app/services/pcb_engine.py`. File này được các API FastAPI sử dụng trực tiếp.

## Yêu cầu

- Windows 10/11 (các lệnh bên dưới dùng PowerShell).
- Python 3.10 trở lên.
- Node.js 18 trở lên, gồm `node` và `npm`.
- File trọng số YOLO `best.pt`.
- Camera webcam hoặc URL stream nếu cần dùng tính năng camera.

Kiểm tra Python và Node.js:

```powershell
python --version
node --version
npm.cmd --version
```

Trong PowerShell, dùng `npm.cmd` nếu Execution Policy chặn `npm.ps1`.

## Cài đặt và chạy trên Windows

Mở thư mục repo `D:\PCB_detection` trong VS Code. Mở terminal bằng **Ctrl + `** và tạo hai terminal riêng, vì backend và frontend cần chạy đồng thời.

### 1. Cài dependencies backend

Chạy một lần từ thư mục gốc:

```powershell
cd D:\PCB_detection
python -m pip install -r backend\requirements.txt
```

Lệnh cài PyTorch có thể khác nhau tùy máy CPU/GPU. Nếu cài `torch` gặp lỗi, hãy chọn đúng lệnh cài từ trang hướng dẫn cài PyTorch cho hệ điều hành và CUDA của máy.

### 2. Cài dependencies frontend

Chạy một lần:

```powershell
cd D:\PCB_detection\frontend
npm.cmd install
```

### 3. Khởi chạy backend

Trong terminal thứ nhất:

```powershell
cd D:\PCB_detection
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Giữ terminal này chạy. Khi model được tải xong, backend phục vụ API ở `http://127.0.0.1:8000`.

### 4. Khởi chạy frontend ở chế độ phát triển

Trong terminal thứ hai:

```powershell
cd D:\PCB_detection\frontend
npm.cmd run dev
```

Mở URL Vite hiện trong terminal, mặc định là `http://localhost:5173`. Frontend chuyển tiếp các request `/api`, `/health` và `/docs` tới backend cổng 8000. Giữ cả hai terminal chạy khi sử dụng.

### Chạy giao diện đã build qua FastAPI

Nếu muốn mở giao diện tại cổng 8000 thay vì 5173, build frontend:

```powershell
cd D:\PCB_detection\frontend
npm.cmd run build
```

Sau đó chạy backend như trên và mở `http://127.0.0.1:8000`. Mỗi khi sửa frontend, chạy lại `npm.cmd run build` để cập nhật thư mục `frontend\dist`.

### Dừng dịch vụ

Chọn terminal của từng dịch vụ rồi nhấn **Ctrl + C**.

## Hướng dẫn sử dụng giao diện

### Nạp mạch chuẩn

1. Mở mục **Mạch tham chiếu**.
2. Chọn ảnh PCB mẫu đạt chuẩn, rõ nét và thấy toàn bộ bo mạch; nhập tên dễ nhận biết, ví dụ `PCB nguồn phiên bản A`.
3. Điều chỉnh ngưỡng tin cậy YOLO nếu cần.
4. Chọn **Lưu mạch chuẩn** và chờ hệ thống nhận diện. Mạch mới được thêm vào danh mục và tự động chọn làm mạch đang dùng.
5. Để chuyển sang một mạch đã lưu, chọn **Chọn mạch** ở dòng tương ứng. Mạch đang hoạt động được đánh dấu **Đang dùng**.
6. Kiểm tra ảnh YOLO đánh dấu và danh sách linh kiện. Ảnh không nhận diện được linh kiện sẽ không được lưu.

Mỗi mạch chuẩn được lưu riêng trong `backend\storage\references\<id>\`. Danh sách và mạch đang chọn được ghi trong `backend\storage\reference_catalog.json`. Mạch chuẩn cũ sẵn có được tự động đưa vào danh mục lần đầu backend khởi động với tên lấy từ dữ liệu reference ban đầu.

### Kiểm tra ảnh PCB

1. Mở **Kiểm tra PCB**.
2. Tải ảnh PCB cần quét. Đây là ảnh riêng, không ghi đè ảnh scan vào thao tác nạp reference.
3. Chỉnh confidence, dung sai vị trí hoặc tự động tìm vùng PCB nếu cần.
4. Chọn **Bắt đầu kiểm tra PCB**.
5. Xem kết quả và danh sách detections. `OK`/`FAULTY` chỉ được đưa ra khi căn chỉnh thành công; `UNALIGNED` nghĩa là đã nhận diện nhưng chưa so sánh được vị trí với reference.

### Nhận diện không đối chiếu

Trong **Nhận diện YOLO**, tải một ảnh rồi chọn **Nhận diện linh kiện**. Trang này chỉ chạy detector, không cần mạch chuẩn.

### Camera trực tiếp

1. Nhập `0` để dùng webcam mặc định trên máy chạy backend, hoặc nhập URL stream camera IP.
2. Chọn **Kết nối camera**.
3. Chọn **Quét khung hình** để quét thủ công. Kết quả xuất hiện bên cạnh stream.
4. Chọn chu kỳ **3 giây**, **5 giây** hoặc **10 giây**, sau đó chọn **Tự động quét**. Chọn **Dừng tự động** để dừng; ngắt camera cũng sẽ dừng chế độ này.

Camera index như `0` trỏ tới máy đang chạy backend, không phải máy đang mở trình duyệt. Để dùng camera IP, cả backend và thiết bị camera phải truy cập được URL stream.

## API

Các endpoint có tại `/api/...`. Alias tương thích cũng được cung cấp tại `/api/v1/...`.

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/info` | Model, thiết bị, lớp YOLO, reference và tham số mặc định |
| `GET` | `/api/reference` | Reference map cùng ảnh PCB chuẩn và ảnh đã đánh dấu |
| `POST` | `/api/reference/update` | Tạo/lưu reference từ ảnh upload |
| `POST` | `/api/reference/select` | Kích hoạt một mạch đã lưu (`reference_id`) |
| `GET` | `/api/sample` | Trả ảnh mẫu dưới dạng data URI |
| `POST` | `/api/detect` | YOLO detection độc lập |
| `POST` | `/api/inspect` | Detection, căn chỉnh và QA theo reference |
| `GET` | `/api/camera/status` | Trạng thái camera hiện tại |
| `POST` | `/api/camera/connect` | Kết nối camera |
| `POST` | `/api/camera/disconnect` | Ngắt camera |
| `GET` | `/api/camera/stream` | Luồng MJPEG |
| `POST` | `/api/camera/scan` | Quét frame mới nhất |
| `POST` | `/api/camera/snapshot` | Mở nguồn camera, chụp một frame và kiểm tra |

Swagger UI: `http://127.0.0.1:8000/docs`. ReDoc: `http://127.0.0.1:8000/redoc`.

### Upload ảnh và tham số

Các endpoint upload ảnh nhận `multipart/form-data`, trường file bắt buộc tên là `image`. Giới hạn upload là 32 MB.

- `/api/detect`: `image`, `conf`, `imgsz`.
- `/api/reference/update`: `image`, `name` (tùy chọn), `conf`, `imgsz`. Nếu bỏ `name`, tên file upload được dùng.
- `/api/reference/select`: `reference_id` (ID lấy từ danh sách `catalog` của `/api/reference`).
- `/api/inspect`: `image`, `conf`, `imgsz`, `tolerance_ratio`, `extra_allowed`, `missing_allowed`, `auto_crop`.
- `/api/camera/connect`: `source`.
- `/api/camera/scan`: `conf`, `imgsz`, `tolerance_ratio`.
- `/api/camera/snapshot`: `source`, `conf`, `imgsz`, `tolerance_ratio`.

Thông số kiểm tra mặc định trong pipeline: confidence `0.25`, ảnh YOLO `1536`, dung sai vị trí `0.045`, số linh kiện thiếu cho phép `0`, số linh kiện thừa cho phép `2`. Có thể xem thông số hiện hành tại `/api/info`.

Ví dụ PowerShell dùng `curl.exe` để nhận diện ảnh:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/detect" `
  -F "image=@D:\PCB_detection\2.jpg" `
  -F "conf=0.25"
```

## Cấu hình

Backend đọc các biến môi trường sau khi khởi động:

| Biến | Giá trị mặc định | Mục đích |
|---|---|---|
| `PCB_MODEL_PATH` | `backend\models\best.pt`; nếu không có thì dùng `best.pt` ở repo root | File trọng số YOLO |
| `PCB_REFERENCE_PATH` | `backend\storage\reference_components.json` | JSON reference ban đầu dùng khi khởi tạo danh mục lần đầu |
| `PCB_REFERENCE_IMAGE_PATH` | `backend\storage\reference.jpg` | Ảnh reference ban đầu dùng khi khởi tạo danh mục lần đầu |
| `PCB_SAMPLE_IMAGE_PATH` | `backend\storage\sample.jpg`; nếu không có thì dùng `2.jpg` ở repo root | Ảnh mẫu của API |
| `PCB_CAMERA_SOURCE` | `0` | Webcam index hoặc URL camera mặc định cho API |

Ví dụ cấu hình camera IP cho backend trong PowerShell:

```powershell
$env:PCB_CAMERA_SOURCE = "http://192.168.1.20:8080/video"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Đặt model mới tại `backend\models\best.pt`, hoặc chỉ định đường dẫn đầy đủ:

```powershell
$env:PCB_MODEL_PATH = "D:\models\pcb-best.pt"
uvicorn backend.main:app --reload
```

Mỗi reference được tạo trên giao diện có thư mục ảnh/JSON riêng dưới `backend\storage\references`. Các biến `PCB_REFERENCE_PATH` và `PCB_REFERENCE_IMAGE_PATH` chỉ định dữ liệu reference ban đầu để nhập vào danh mục nếu danh mục chưa tồn tại. Không xóa `reference_catalog.json` hoặc thư mục `references` nếu muốn giữ danh mục mạch đã lưu.

## Kiểm tra API

Khởi động backend trước, sau đó mở terminal khác tại repo root:

```powershell
cd D:\PCB_detection
python test_api.py
```

Smoke test kiểm tra health, thông tin model, reference, ảnh mẫu, camera status, detection và inspection nếu reference đã được nạp. Để kiểm tra server ở địa chỉ khác:

```powershell
$env:PCB_API_URL = "http://127.0.0.1:8000"
python test_api.py
```

Inference sẽ dùng model thật và có thể mất thời gian tùy phần cứng; smoke test này không chỉ kiểm tra HTTP.

## Chạy chương trình camera desktop

`test.py` chạy cửa sổ OpenCV độc lập, không phải frontend web. Chương trình dùng model trong `backend\models\best.pt` nếu có, nếu không sẽ fallback về model ở thư mục gốc. Reference JSON được đọc từ `backend\storage\reference_components.json`.

```powershell
cd D:\PCB_detection
$env:PCB_CAMERA_SOURCE = "0"
python test.py
```

Phím điều khiển phụ thuộc vào chương trình desktop; nhấn `s` để quét frame và `q` để thoát.

`machpcbgoc.py` tạo reference map bằng YOLO từ ảnh `2.jpg` và lưu dữ liệu/ảnh đã đánh dấu trong `backend\storage`:

```powershell
cd D:\PCB_detection
python machpcbgoc.py
```

Giao diện web cũng có thể tạo hoặc thay reference trực tiếp, không cần chạy script trên.

## Xử lý sự cố

### PowerShell báo `npm` không nhận diện hoặc chặn `npm.ps1`

Xác nhận Node.js đã cài:

```powershell
node --version
npm.cmd --version
```

Dùng `npm.cmd` thay cho `npm`, ví dụ:

```powershell
cd D:\PCB_detection\frontend
npm.cmd install
npm.cmd run dev
```

### Trang cổng 8000 báo frontend chưa build

Đây là địa chỉ backend. Chọn một trong hai cách:

- Chạy `npm.cmd run dev` trong `frontend` và mở `http://localhost:5173`; hoặc
- Chạy `npm.cmd run build` trong `frontend`, rồi tải lại `http://127.0.0.1:8000`.

### Frontend mở được nhưng API không phản hồi

- Kiểm tra terminal backend vẫn chạy và xem log lỗi.
- Mở `http://127.0.0.1:8000/health`; kết quả bình thường là `{"status":"ok"}`.
- Chạy frontend qua `http://localhost:5173` khi backend ở cổng 8000; Vite dev server đã cấu hình proxy API.

### Backend báo không tìm thấy model

Đặt file `.pt` tại `backend\models\best.pt` hoặc cấu hình `PCB_MODEL_PATH`. Model YOLO hợp lệ là bắt buộc để khởi tạo ứng dụng.

### Chưa có mạch tham chiếu

Inspection yêu cầu reference có ít nhất một linh kiện. Mở **Mạch tham chiếu**, tải ảnh rõ nét rồi tạo reference map trước.

### Camera không kết nối

- Kiểm tra webcam đang được hệ điều hành nhận diện và chưa bị ứng dụng khác giữ độc quyền.
- Thử `0` hoặc `1` với webcam; với IP camera, xác nhận URL stream mở được từ máy chạy backend.
- Camera index là index trên máy chạy backend.

### Ảnh upload quá lớn hoặc không đọc được

API chấp nhận ảnh tối đa 32 MB. Dùng file PNG/JPEG hợp lệ và tránh ảnh hỏng hoặc định dạng không được OpenCV giải mã.

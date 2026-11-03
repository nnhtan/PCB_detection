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
- **Camera trực tiếp:** kết nối webcam hoặc camera IP, lưu frame làm mạch chuẩn mới, quét từng frame hoặc đặt quét tự động.
- **Kết quả trực quan:** ảnh đánh dấu detections, số lượng linh kiện, kết quả đối chiếu và thời gian xử lý.
- **API docs:** FastAPI Swagger UI tại `/docs`.

Khi đã căn chỉnh với mạch chuẩn, kết luận dựa trên đối chiếu vị trí từng linh kiện; tổng số detections được hiển thị riêng, và detection dư có độ tin cậy thấp không tự làm mạch bị lỗi. Nếu ảnh đã nhận diện được linh kiện nhưng không căn chỉnh được, kết quả có trạng thái `UNALIGNED`; hệ thống không giả kết luận linh kiện thiếu/thừa.

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

Nếu backend chạy ở cổng khác (ví dụ `8001`), đặt cùng địa chỉ trước khi chạy frontend:

```powershell
$env:PCB_API_URL = "http://127.0.0.1:8001"
npm.cmd run dev
```

### Kết nối camera Hikvision trên giao diện web

Backend phải chạy trên chính máy tính đã kết nối camera Hikvision. Cài Hikvision MVS để có SDK `MvCameraControl.dll`. Backend tự kết nối nguồn `hikvision` khi khởi động; khi mở web, giao diện đọc trạng thái kết nối này và không mở kết nối trùng. Mặc định camera dùng IP `169.254.128.194` và thời gian phơi sáng `20 ms`; có thể đổi IP bằng biến môi trường `PCB_CAMERA_SOURCE` trước khi chạy backend. Nếu camera không được phát hiện, cần kiểm tra lại cáp LAN, địa chỉ IP của thiết bị, subnet và driver MVS. Khi quét, backend cắt/nắn PCB; nếu reference có hai tụ và một IC làm mốc, hệ thống dùng các mốc này để xoay crop đúng chiều rồi chạy YOLO lại trước khi đối chiếu. Phép xoay giữ toàn bộ vùng crop; nếu thiếu mốc hoặc không xác định được chiều, kết quả là `UNALIGNED` và không chạy QA. Giao diện chỉ hiển thị kết quả phân tích. Nếu thuật toán không tìm được bo mạch, yêu cầu bị từ chối thay vì phân tích toàn khung hình. Thanh **Exposure time** trên trang Camera trực tiếp thay đổi thời gian phơi sáng theo ms trong giới hạn mà thiết bị báo về. Không chạy đồng thời chương trình xem camera desktop. Webcam máy tính vẫn dùng chỉ số như `0`; camera Hikvision qua LAN nên dùng `hikvision` hoặc IP camera trực tiếp.

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
3. Chọn **Quét thông thường** để dùng luồng hiện tại, hoặc **Căn chỉnh mạch nghiêng** để định vị PCB bằng đặc trưng ảnh, xoay/cắt về khung reference rồi kiểm tra. Chế độ căn chỉnh mạch nghiêng sẽ báo lỗi nếu không khớp được ảnh với reference.
4. Chỉnh confidence và dung sai vị trí nếu cần. Tự động cắt vùng PCB chỉ áp dụng cho chế độ thông thường.
5. Chọn nút bắt đầu kiểm tra tương ứng với chế độ đã chọn.
6. Xem ảnh PCB đã căn chỉnh và kết quả/danh sách detections. Khi căn chỉnh thành công, `OK`/`FAULTY`/`REVIEW` được quyết định theo đối chiếu vị trí; tổng số detections vẫn được báo riêng. `UNALIGNED` nghĩa là đã nhận diện nhưng chưa so sánh được vị trí với reference.

Khi đã căn chỉnh và detection sai lớp nằm trong bán kính vị trí theo kích thước linh kiện tham chiếu, detection được ghép với reference dù hộp không chứa chính xác tâm reference; lớp reference được dùng làm nhãn chính và QA không tính đó là mismatch. Lớp YOLO cùng confidence gốc vẫn được giữ trong `yolo_class` và hiển thị trong bảng/nhãn ảnh để đối chiếu. Vì reference là nguồn quyết định lớp trong chế độ này, kết quả không xác nhận độc lập linh kiện lắp thực tế có đúng loại hay không; cần kiểm tra ảnh nếu YOLO bất đồng. Khi nhiều lớp dự đoán cùng một vùng, chúng được giữ lại cho đối chiếu; trong kiểm tra đã căn chỉnh, nếu một ứng viên khớp lớp reference tại đúng vị trí thì các ứng viên lớp khác chồng lấp tại cùng linh kiện bị loại kể cả khi hộp có kích thước chênh lệch lớn. Các detection khớp chính xác với reference của linh kiện lân cận luôn được giữ. Nhận diện YOLO độc lập vẫn giữ các lớp chồng lấp khác nhau để tránh xóa mất dự đoán thay thế.

### Nhận diện không đối chiếu

Trong **Nhận diện YOLO**, tải một ảnh rồi chọn **Nhận diện linh kiện**. Trang này chỉ chạy detector, không cần mạch chuẩn.

Pipeline kết hợp YOLO chạy trên toàn ảnh với dự đoán các lát SAHI để giữ khả năng bắt linh kiện nhỏ. Chỉ các hộp chồng lấp cùng lớp mới bị gộp theo độ tin cậy; lớp khác nhau được giữ lại vì có thể là ứng viên phân loại thay thế. Detector loại các dự đoán có tâm nằm bên trong hộp IC để giảm nhận nhầm chân IC thành linh kiện. Tạo reference cũng dùng cùng pipeline này để danh sách chuẩn và lượt kiểm tra nhất quán. Các biến thể viết hoa/thường của cùng tên lớp được chuẩn hóa; `inductor` và `potentiometer` vẫn là hai loại riêng biệt. Pipeline cũng bỏ lớp `buzzer`.

Với linh kiện reference bị thiếu hoặc sai lớp, pipeline chạy thêm một lượt YOLO trên vùng tham chiếu đã phóng lớn, dùng ngưỡng `0.01` để không bỏ ứng viên lớp đúng có confidence thấp. Kết quả bổ sung chỉ được ghép khi đúng lớp reference và đúng vị trí trong dung sai; ứng viên không đạt điều kiện vẫn được báo thiếu hoặc cần kiểm tra.

### Camera trực tiếp

1. Nhập `0` để dùng webcam mặc định trên máy chạy backend, hoặc nhập URL stream camera IP.
2. Chọn **Kết nối camera**.
3. Chọn **Quét khung hình** để xử lý frame và hiển thị kết quả đối chiếu với reference đang dùng. Quét camera không tạo hoặc lưu mạch chuẩn.
4. Chọn chu kỳ quét rồi chọn **Tự động quét**. Chọn **Dừng tự động** để dừng; ngắt camera cũng sẽ dừng chế độ này.

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
| `POST` | `/api/inspect/tilted` | Căn chỉnh nghiêng theo reference, cắt PCB rồi chạy QA |
| `GET` | `/api/camera/status` | Trạng thái camera hiện tại |
| `POST` | `/api/camera/connect` | Kết nối camera |
| `POST` | `/api/camera/disconnect` | Ngắt camera |
| `GET` | `/api/camera/stream` | Luồng MJPEG |
| `POST` | `/api/camera/scan` | Quét frame mới nhất |
| `POST` | `/api/camera/snapshot` | Mở nguồn camera, chụp một frame và kiểm tra |

Swagger UI: `http://127.0.0.1:8000/docs`. ReDoc: `http://127.0.0.1:8000/redoc`.

### Upload ảnh và tham số

Các endpoint upload ảnh nhận `multipart/form-data`, trường file bắt buộc tên là `image`. Giới hạn upload là 32 MB.

- `/api/detect`: `image`, `conf`, `imgsz`, `auto_crop`.
- `/api/reference/update`: `image`, `name` (tùy chọn), `conf`, `imgsz`. Nếu bỏ `name`, tên file upload được dùng.
- `/api/reference/select`: `reference_id` (ID lấy từ danh sách `catalog` của `/api/reference`).
- `/api/inspect`: `image`, `conf`, `imgsz`, `tolerance_ratio`, `extra_allowed`, `missing_allowed`, `auto_crop`.
- `/api/inspect/tilted`: `image`, `conf`, `imgsz`, `tolerance_ratio`, `extra_allowed`, `missing_allowed`. Trả HTTP 422 nếu không căn chỉnh được ảnh scan với reference.
- `/api/camera/connect`: `source`.
- `/api/camera/scan`: `conf`, `imgsz` (mặc định `1024`), `tolerance_ratio`, `auto_crop`, `fast_mode`, `scan_mode` (`standard` hoặc `tilted_reference`).
- `/api/camera/snapshot`: `source`, `conf`, `imgsz` (mặc định `1024`), `tolerance_ratio`, `auto_crop`, `scan_mode` (`standard` hoặc `tilted_reference`).
- `/api/conveyor/result`: JSON `{"is_ok": true}` hoặc `{"is_ok": false}` để gửi kết quả đang hiển thị trên web đến ESP.

Khi `auto_crop` được bật (mặc định), `/api/camera/scan` và `/api/inspect` trước tiên dùng ORB để tìm phép chiếu từ frame camera sang mạch tham chiếu; nếu thành công, OpenCV cắt/nắn PCB trực tiếp về đúng kích thước và chiều của ảnh reference. Chỉ khi không căn được theo reference, pipeline mới thử `PCBMonoCropper`, contour và ROI. Với các đường fallback, ảnh crop hiển thị cũng được nắn theo phép căn chỉnh cuối cùng; homography ORB hợp lệ được ưu tiên hơn affine để giảm nghiêng phối cảnh, đồng thời hộp detection và dấu linh kiện thiếu được vẽ trong cùng hệ tọa độ reference. Sau khi chuẩn hóa PCB, pipeline dùng SAHI overlap 20% để chia ảnh thành tile và phát hiện linh kiện; quét camera mặc định dùng chế độ nhanh tile 960 px để giảm độ trễ, có thể chuyển sang `fast_mode=false` để quét chi tiết với tile 320 px và ưu tiên linh kiện nhỏ; cả hai chế độ giữ `imgsz=1536`. Chế độ chi tiết bật test-time augmentation (TTA) cho lượt YOLO toàn ảnh và gom 2 tile/lượt trên GPU CUDA; chế độ nhanh tắt TTA và giữ batch 1. Benchmark trên ảnh mẫu cho thấy chế độ nhanh hoàn tất quét và ghép đủ 11/11 linh kiện trong khoảng 0,6 giây, nhanh hơn khoảng 3,2 lần so với chế độ chi tiết; kết quả còn tùy ảnh và phần cứng. Thử tăng batch lên 2/4 hoặc bật FP16 trên NVIDIA T600 không cải thiện thời gian ở ảnh mẫu, nên không bật các tùy chọn đó mặc định. Các hộp cùng lớp chồng ít nhất 50% diện tích hộp nhỏ hơn được gộp nếu chênh lệch diện tích không quá 4 lần. Hộp khác lớp không bị loại chỉ vì chồng lấp; khi căn chỉnh với reference, dự đoán cùng lớp tại đúng vị trí được ưu tiên, còn khi không có ứng viên cùng lớp thì dự đoán confidence cao nhất được giữ với trạng thái `REVIEW`. Phản hồi có `alignment_transform`, `orientation_transform` và `crop_method` để kiểm tra phép biến đổi và cách cắt đã dùng.

`/api/inspect/tilted` là luồng riêng cho ảnh PCB đặt nghiêng: ưu tiên dò contour bo mạch bằng cạnh ảnh và vùng tối, chuẩn hóa cạnh dài theo hướng reference rồi xoay/nắn và cắt về khung reference; ORB homography chỉ là phương án dự phòng. Cách dò vùng tối chỉ bật trong luồng nghiêng; Auto-Crop của chế độ tiêu chuẩn giữ nguyên cách tìm contour cũ. Nếu không tìm thấy PCB hoặc không căn được ảnh, pipeline dừng trước khi nhận diện thay vì phân loại nền. Luồng này không dùng fallback auto-crop, đồng thời giữ nguyên YOLO và tiền xử lý hiện hành.

Trong trang **Camera trực tiếp**, chế độ **Quét nhanh** tiêu chuẩn dùng `imgsz=1024` và lát SAHI 960 px; **Quét chi tiết** dùng `imgsz=1536` và lát 320 px để ưu tiên linh kiện nhỏ. Khi **Căn theo chuẩn** bật, ảnh suy luận dùng `imgsz=1536` ngay cả ở chế độ lát nhanh để tránh dự đoán dư đã quan sát được ở `imgsz=1024`; chế độ lát nhanh vẫn dùng SAHI 960 px. Camera cũng chọn frame có PCB rõ biên và xa mép nhất trong 12 frame gần nhất, dò contour từ cạnh ảnh hoặc vùng tối, dùng ORB homography nếu không tìm được contour và bỏ qua Auto-Crop thường để tránh crop lại ảnh đã căn. Nếu không thấy PCB trong lịch sử frame hoặc không căn được, API trả HTTP 422 và frame đó không được nhận diện/cộng vào bộ đếm. Tắt công tắc để quay lại quy trình camera tiêu chuẩn.

Thông số kiểm tra mặc định trong pipeline: confidence `0.35`, ảnh YOLO `1536`, dung sai vị trí `0.045` (4.5% cạnh lớn nhất của ảnh reference, tối đa 15%), số linh kiện thiếu cho phép `0`, số linh kiện thừa cho phép `2`. Việc ghép linh kiện luôn phải thỏa dung sai khoảng cách tâm; hộp detection bao trùm điểm reference không đủ để được tính là khớp. Có thể xem thông số hiện hành tại `/api/info`.

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
| `PCB_CONVEYOR_ENABLED` | `false` | Bật gửi verdict kiểm định qua Serial cho ESP |
| `PCB_SERIAL_PORT` | `COM3` | Cổng Serial của ESP32 trên máy hiện tại; đổi nếu Windows gán COM khác |
| `PCB_SERIAL_BAUDRATE` | `115200` | Tốc độ Serial |
| `PCB_SERIAL_TIMEOUT` | `5` | Thời gian chờ ESP xác nhận, tính bằng giây |

Khi bật `PCB_CONVEYOR_ENABLED`, giữ trang Camera mở và kết nối camera để nhận sự kiện `EVENT:SENSOR1` từ ESP qua Serial; mỗi cạnh phát hiện mới ở cảm biến 1 sẽ chờ 1,5 giây để băng tải dừng hẳn rồi tự kích hoạt một lượt quét camera. Sau khi quét, giao diện gửi verdict `OK` hoặc `FAULTY` tới `POST /api/conveyor/result` nếu có kết quả, rồi gửi `RESUME` tới `POST /api/conveyor/resume` để băng tải chạy tiếp. Trạng thái chưa căn chỉnh không được coi là mạch lỗi và không gửi verdict. Khi mạch tới cảm biến 2, ESP chạy servo 1 nếu mạch đạt; nếu mạch lỗi thì servo 1 và servo 2 cùng chạy. Băng tải tiếp tục sau khi các servo hoàn tất. Nếu cảm biến 2 phát hiện mạch trước khi có verdict, ESP dừng và chờ kết quả từ web. Gửi Serial kết thúc bằng newline.

Đóng PlatformIO Serial Monitor trước khi chạy backend vì hai chương trình không thể cùng chiếm một cổng COM. Ví dụ bật tích hợp trong PowerShell:

```powershell
$env:PCB_CONVEYOR_ENABLED = "true"
$env:PCB_SERIAL_PORT = "COM3"
$env:PCB_SERIAL_BAUDRATE = "115200"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

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

`test.py` chạy cửa sổ OpenCV độc lập, không phải frontend web. Chương trình dùng model trong `backend\models\best.pt` nếu có, nếu không sẽ fallback về model ở thư mục gốc. Reference JSON được đọc từ `backend\storage\reference_components.json`. `test1.py` là lối chạy tương thích, dùng chung mã nguồn và cách cắt với `test.py`.

Trước khi nhận diện, chương trình kết hợp nhiều ngưỡng Canny thích nghi với phân đoạn nền; nền được ước lượng từ các điểm mẫu ở viền ảnh để giảm tải xử lý. Với contour gần chữ nhật, bộ cắt ưu tiên `minAreaRect` để tránh các chi tiết linh kiện làm lệch góc. Nếu phát hiện contour lớn chạm mép ảnh cùng một contour nhỏ bên trong, chương trình từ chối crop để tránh nhận nhầm một phần linh kiện thành toàn bộ PCB; cần đặt trọn bo mạch trong khung camera. Góc PCB được làm mượt và xác nhận qua 5 frame; kích thước crop được kiểm tra ở độ phân giải gốc. Khi camera Hikvision xử lý ảnh, YOLO chạy ở luồng nền để cửa sổ live vẫn cập nhật và báo trạng thái đang quét/lỗi. Chỉ crop hợp lệ mới được chuyển sang nhận diện.

```powershell
cd D:\PCB_detection
$env:PCB_CAMERA_SOURCE = "0"
python test.py
```

Để căn chỉnh camera và kiểm tra khung cắt trước khi nạp YOLO, chạy `python test.py --detect-only` (hoặc `python test1.py --detect-only`). Cửa sổ `PCB Crop` luôn hiển thị crop đầu vào; chế độ `--detect-only` bổ sung cửa sổ debug biên/mask nhưng không nạp YOLO. Nhấn `d` để lưu frame, mask, biên và thông số vào `debug\`. Sau khi crop đã đúng, chạy bình thường; chờ trạng thái PCB sẵn sàng rồi nhấn `s` để quét, `q` để thoát. Với camera Hikvision, thời gian phơi sáng mặc định là 20 ms; có thể đổi bằng PowerShell trước khi chạy, ví dụ `$env:PCB_HIK_EXPOSURE_US = "50000"` để dùng 50 ms. Tăng phơi sáng nếu ảnh quá tối; giảm nếu PCB di chuyển bị nhòe. Webcam/IP camera không bị ảnh hưởng bởi thiết lập này.

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

import { BookOpenText, ExternalLink } from "lucide-react";
import PageHeader from "../components/common/PageHeader.jsx";

const endpoints = [
  ["GET", "/health", "Kiểm tra dịch vụ API"],
  ["GET", "/api/info", "Thông tin model, thiết bị và trạng thái reference"],
  ["GET", "/api/reference", "Danh mục, ảnh và linh kiện mạch tham chiếu"],
  ["POST", "/api/reference/update", "Tải ảnh mạch chuẩn, tạo reference map"],
  ["POST", "/api/reference/select", "Chọn mạch tham chiếu đã lưu"],
  ["DELETE", "/api/reference/{id}", "Xóa một mạch tham chiếu không sử dụng"],
  ["DELETE", "/api/reference/unused/all", "Xóa tất cả mạch tham chiếu chưa sử dụng"],
  ["POST", "/api/inspect", "Quét PCB và đối chiếu với reference"],
  ["POST", "/api/detect", "Chạy YOLO nhận diện độc lập"],
  ["GET", "/api/camera/status", "Trạng thái camera"],
  ["POST", "/api/camera/connect", "Kết nối webcam hoặc camera IP"],
  ["GET", "/api/camera/stream", "Luồng hình MJPEG"],
  ["POST", "/api/camera/scan", "Quét frame mới nhất từ camera"],
];

export default function DocsPage() {
  return (
    <>
      <PageHeader
        eyebrow="DEVELOPER"
        title="API & hướng dẫn"
        description="Các bước khởi chạy ứng dụng và endpoint FastAPI sẵn có."
        action={
          <a className="button button-light button-small" href="/docs" target="_blank" rel="noreferrer">
            <ExternalLink size={15} /> Swagger UI
          </a>
        }
      />
      <div className="docs-grid">
        <section className="panel-card docs-card">
          <div className="section-title">
            <div>
              <h2><BookOpenText size={19} /> Khởi chạy hệ thống</h2>
              <p>Chạy API và giao diện trong môi trường phát triển</p>
            </div>
          </div>
          <div className="guide-step">
            <span>01</span>
            <div>
              <strong>Cài backend</strong>
              <code>python -m pip install -r backend/requirements.txt</code>
            </div>
          </div>
          <div className="guide-step">
            <span>02</span>
            <div>
              <strong>Khởi chạy FastAPI</strong>
              <code>uvicorn backend.main:app --reload</code>
            </div>
          </div>
          <div className="guide-step">
            <span>03</span>
            <div>
              <strong>Cài và chạy React</strong>
              <code>cd frontend<br />npm.cmd install<br />npm.cmd run dev</code>
            </div>
          </div>
          <div className="guide-note">
            Model mặc định: <code>backend/models/best.pt</code>. Có thể đổi đường dẫn bằng biến môi trường <code>PCB_MODEL_PATH</code>.
          </div>
        </section>
        <section className="panel-card docs-card">
          <div className="section-title">
            <div>
              <h2>Endpoint API</h2>
              <p>JSON response · tải ảnh qua multipart/form-data</p>
            </div>
          </div>
          <div className="endpoint-list">
            {endpoints.map(([method, path, desc]) => (
              <div className="endpoint" key={`${method}${path}`}>
                <span className={`method ${method.toLowerCase()}`}>{method}</span>
                <code>{path}</code>
                <small>{desc}</small>
              </div>
            ))}
          </div>
        </section>
      </div>
    </>
  );
}

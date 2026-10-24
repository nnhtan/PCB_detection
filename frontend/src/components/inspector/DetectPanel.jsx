import { useState } from "react";
import { Crop, LoaderCircle, ScanSearch } from "lucide-react";
import PageHeader from "../common/PageHeader.jsx";
import UploadCard from "../common/UploadCard.jsx";
import InspectionResults from "./InspectionResults.jsx";
import { useInspect } from "../../hooks/useInspect.js";

export default function DetectPanel() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [autoCrop, setAutoCrop] = useState(true);
  const { result, error, busy, inspect } = useInspect("/api/detect");
  const pick = (selected) => {
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };
  return (
    <>
      <PageHeader eyebrow="OBJECT DETECTION" title="Nhận diện linh kiện" description="Phát hiện và thống kê linh kiện trên bo mạch PCB bằng mô hình YOLO." />
      <div className="panel-card">
        <div className="section-title"><div><h2><ScanSearch size={19} /> Quét ảnh bằng YOLO</h2><p>Chức năng nhận diện không cần mạch chuẩn</p></div></div>
        <div className="detect-grid">
          <UploadCard title="Tải ảnh PCB" description="PNG, JPG hoặc JPEG · tối đa 32 MB" onSelect={pick} preview={preview} />
          <div className="detect-side">
            <div className="helper-box"><strong>Nhận diện thuần</strong><p>YOLO xác định vị trí và loại linh kiện. Để kiểm tra thiếu/thừa, hãy dùng trang Kiểm tra PCB và nạp mạch chuẩn.</p></div>
            <div
              className={`toggle-crop-card ${autoCrop ? "active" : ""}`}
              onClick={() => setAutoCrop((prev) => !prev)}
              role="button"
              tabIndex={0}
              title="Bật/Tắt tự động cắt bo mạch trước khi nhận diện"
            >
              <div className="toggle-crop-card-info">
                <Crop size={16} color={autoCrop ? "var(--purple)" : "#7b7c8f"} />
                <span>Tự động cắt bo mạch (Auto-crop)</span>
              </div>
              <span className="toggle-crop-card-badge">{autoCrop ? "BẬT" : "TẮT"}</span>
            </div>
            <button className="button button-primary button-wide" disabled={!file || busy} onClick={() => inspect(file, { auto_crop: autoCrop })}>{busy ? <><LoaderCircle className="spin" size={17} /> Đang nhận diện...</> : <><ScanSearch size={17} /> Nhận diện linh kiện</>}</button>
          </div>
        </div>
        {error && <div className="alert alert-error">{error}</div>}
      </div>
      <InspectionResults result={result} />
    </>
  );
}

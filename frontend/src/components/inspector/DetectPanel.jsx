import { useState } from "react";
import { LoaderCircle, ScanSearch } from "lucide-react";
import PageHeader from "../common/PageHeader.jsx";
import UploadCard from "../common/UploadCard.jsx";
import InspectionResults from "./InspectionResults.jsx";
import { useInspect } from "../../hooks/useInspect.js";

export default function DetectPanel() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const { result, error, busy, inspect } = useInspect("/api/detect");
  const pick = (selected) => {
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };
  return (
    <>
      <PageHeader eyebrow="OBJECT DETECTION" title="Nhận diện linh kiện" description="Tự động cắt vùng bo mạch trước khi chạy YOLO để phát hiện và thống kê linh kiện." />
      <div className="panel-card">
        <div className="section-title"><div><h2><ScanSearch size={19} /> Quét ảnh bằng YOLO</h2><p>Chức năng nhận diện không cần mạch chuẩn</p></div></div>
        <div className="detect-grid">
          <UploadCard title="Tải ảnh PCB" description="PNG, JPG hoặc JPEG · tối đa 32 MB" onSelect={pick} preview={preview} />
          <div className="detect-side">
            <div className="helper-box"><strong>Nhận diện thuần</strong><p>YOLO xác định vị trí và loại linh kiện. Để kiểm tra thiếu/thừa, hãy dùng trang Kiểm tra PCB và nạp mạch chuẩn.</p></div>
            <button className="button button-primary button-wide" disabled={!file || busy} onClick={() => inspect(file)}>{busy ? <><LoaderCircle className="spin" size={17} /> Đang nhận diện...</> : <><ScanSearch size={17} /> Nhận diện linh kiện</>}</button>
          </div>
        </div>
        {error && <div className="alert alert-error">{error}</div>}
      </div>
      <InspectionResults result={result} />
    </>
  );
}

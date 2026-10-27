import { useState } from "react";
import { ArrowRight, Crop, ImagePlus, LoaderCircle, ScanLine, Sparkles } from "lucide-react";
import PageHeader from "../common/PageHeader.jsx";
import UploadCard from "../common/UploadCard.jsx";
import InspectionResults from "./InspectionResults.jsx";
import { useInspect } from "../../hooks/useInspect.js";
import { api } from "../../services/api.js";
import { useAppStore } from "../../store/appStore.js";

export default function InspectPanel() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [conf, setConf] = useState(0.35);
  const [tolerance, setTolerance] = useState(0.55);
  const [autoCrop, setAutoCrop] = useState(true);
  const reference = useAppStore((state) => state.reference);
  const { result, error, busy, inspect } = useInspect();
  const pick = (selected) => {
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };
  const useSample = async () => {
    try {
      const payload = await api.get("/api/sample");
      const response = await fetch(payload.image);
      const blob = await response.blob();
      pick(new File([blob], "sample-pcb.jpg", { type: "image/jpeg" }));
    } catch (requestError) {
      window.alert(requestError.message);
    }
  };

  return (
    <>
      <PageHeader eyebrow="QUALITY ASSURANCE" title="Kiểm tra mạch PCB" description="Tải riêng ảnh mạch chuẩn và ảnh cần quét để đối chiếu vị trí linh kiện." action={<span className={`reference-chip ${reference?.components_count ? "ready" : ""}`}><span />{reference?.components_count ? `${reference.components_count} linh kiện chuẩn` : "Chưa có mạch chuẩn"}</span>} />
      <div className="workflow-strip">
        <div className="workflow-step complete"><span>01</span><div><strong>Mạch chuẩn</strong><small>{reference?.components_count ? "Đã nạp lên hệ thống" : "Nạp tại mục Mạch tham chiếu"}</small></div></div>
        <ArrowRight size={17} className="workflow-arrow" />
        <div className="workflow-step active"><span>02</span><div><strong>Ảnh quét</strong><small>Tải ảnh PCB cần kiểm tra</small></div></div>
        <ArrowRight size={17} className="workflow-arrow" />
        <div className="workflow-step"><span>03</span><div><strong>Đối chiếu QA</strong><small>Kiểm tra thiếu linh kiện</small></div></div>
      </div>
      <div className="panel-card inspect-card">
        <div className="section-title"><div><h2><ScanLine size={19} /> Ảnh mạch cần quét</h2><p>Ảnh chụp rõ nét, nhìn thấy toàn bộ vùng linh kiện trên PCB</p></div><button className="button button-light button-small" onClick={useSample}><ImagePlus size={15} /> Dùng ảnh mẫu</button></div>
        <div className="inspect-grid">
          <UploadCard title="Tải ảnh PCB cần kiểm tra" description="PNG, JPG hoặc JPEG · tối đa 32 MB" onSelect={pick} preview={preview} />
          <div className="inspect-options">
            <div className="option-heading"><Sparkles size={16} /> Tham số kiểm tra</div>
            <label className="field-label">Độ tin cậy YOLO <strong>{Math.round(conf * 100)}%</strong></label>
            <input type="range" min="0.1" max="0.9" step="0.05" value={conf} onChange={(event) => setConf(Number(event.target.value))} />
            <label className="field-label">Dung sai độ lệch vị trí <strong>{Math.round(tolerance * 100)}%</strong></label>
            <input type="range" min="0.05" max="0.80" step="0.05" value={tolerance} onChange={(event) => setTolerance(Number(event.target.value))} />
            <div
              className={`toggle-crop-card ${autoCrop ? "active" : ""}`}
              onClick={() => setAutoCrop((prev) => !prev)}
              role="button"
              tabIndex={0}
              title="Bật/Tắt tự động tìm và cắt bo mạch trước khi kiểm tra"
            >
              <div className="toggle-crop-card-info">
                <Crop size={16} color={autoCrop ? "var(--purple)" : "#7b7c8f"} />
                <span>Tự động cắt bo mạch (Auto-crop)</span>
              </div>
              <span className="toggle-crop-card-badge">{autoCrop ? "BẬT" : "TẮT"}</span>
            </div>
            <div className="reference-note"><span className={reference?.components_count ? "text-success" : "text-warning"}>{reference?.components_count ? "●" : "●"}</span> Mạch chuẩn đang có <strong>{reference?.components_count || 0}</strong> linh kiện tham chiếu</div>
            <button className="button button-primary button-wide" disabled={!file || busy || !reference?.components_count} onClick={() => inspect(file, { conf, tolerance_ratio: tolerance, auto_crop: autoCrop })}>
              {busy ? <><LoaderCircle className="spin" size={17} /> Đang phân tích ảnh...</> : <><ScanLine size={17} /> Bắt đầu kiểm tra PCB</>}
            </button>
            {!reference?.components_count && <small className="form-help">Hãy tải ảnh mạch chuẩn tại trang “Mạch tham chiếu” trước khi kiểm tra.</small>}
          </div>
        </div>
        {error && <div className="alert alert-error">{error}</div>}
      </div>
      <InspectionResults result={result} />
    </>
  );
}

import { useEffect, useState } from "react";
import { Check, CircuitBoard, LoaderCircle, RefreshCw, UploadCloud } from "lucide-react";
import PageHeader from "../common/PageHeader.jsx";
import UploadCard from "../common/UploadCard.jsx";
import { api, imageForm } from "../../services/api.js";
import { useAppStore } from "../../store/appStore.js";

export default function ReferencePanel() {
  const reference = useAppStore((state) => state.reference);
  const loadReference = useAppStore((state) => state.loadReference);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [conf, setConf] = useState(0.25);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [detectedPreview, setDetectedPreview] = useState("");
  const [referenceName, setReferenceName] = useState("");

  useEffect(() => {
    setDetectedPreview(reference?.detected_image_base64 || "");
  }, [reference]);

  const pick = (selected) => {
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setReferenceName(selected.name.replace(/\.[^.]+$/, ""));
    setError("");
    setMessage("");
  };
  const save = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const result = await api.postForm(
        "/api/reference/update",
        imageForm(file, { conf, name: referenceName }),
      );
      await loadReference();
      setMessage(result.message);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };
  const selectReference = async (referenceId) => {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const form = new FormData();
      form.append("reference_id", referenceId);
      const result = await api.postForm("/api/reference/select", form);
      await loadReference();
      setMessage(result.message);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <PageHeader eyebrow="REFERENCE BOARD" title="Mạch tham chiếu" description="Nạp ảnh mạch chuẩn riêng để hệ thống lưu bản đồ vị trí linh kiện dùng cho các lượt kiểm tra." action={<button className="button button-light button-small" onClick={() => loadReference().catch((loadError) => setError(loadError.message))}><RefreshCw size={15} /> Làm mới</button>} />
      <div className="reference-layout">
        <div className="panel-card reference-upload">
          <div className="section-title"><div><h2><UploadCloud size={19} /> Tạo mạch chuẩn</h2><p>YOLO tự động nhận diện linh kiện và tạo reference map</p></div></div>
          <UploadCard title="Tải ảnh mạch chuẩn" description="Chọn PCB mẫu đạt chuẩn để lưu lại" onSelect={pick} preview={preview} />
          <div className="reference-controls">
            <label className="field-label" htmlFor="reference-name">Tên mạch tham chiếu</label>
            <input id="reference-name" className="reference-name-input" value={referenceName} onChange={(event) => setReferenceName(event.target.value)} placeholder="Ví dụ: PCB nguồn phiên bản A" />
            <label className="field-label">Độ tin cậy YOLO <strong>{Math.round(conf * 100)}%</strong></label>
            <input type="range" min="0.1" max="0.9" step="0.05" value={conf} onChange={(event) => setConf(Number(event.target.value))} />
            <button className="button button-primary button-wide" disabled={!file || busy} onClick={save}>{busy ? <><LoaderCircle className="spin" size={17} /> Đang tạo bản đồ...</> : <><CircuitBoard size={17} /> Lưu mạch chuẩn</>}</button>
          </div>
          {error && <div className="alert alert-error">{error}</div>}
          {message && <div className="alert alert-success"><Check size={16} />{message}</div>}
        </div>
        <div className="panel-card reference-summary">
          <div className="section-title"><div><h2><CircuitBoard size={19} /> Danh mục mạch tham chiếu</h2><p>Chọn mạch đã lưu để dùng cho các lượt kiểm tra PCB</p></div><span className={`badge ${reference?.components_count ? "badge-green" : "badge-gray"}`}>{reference?.components_count ? "ĐÃ SẴN SÀNG" : "CHƯA NẠP"}</span></div>
          {reference?.catalog?.length > 0 ? (
            <div className="reference-catalog">
              {reference.catalog.map((item) => {
                const active = item.id === reference.active_reference_id;
                return (
                  <div className={`reference-catalog-item ${active ? "active" : ""}`} key={item.id}>
                    <div className="reference-catalog-info">
                      <strong>{item.name}</strong>
                      <span>{item.components_count} linh kiện · {new Date(item.created_at).toLocaleDateString("vi-VN")}</span>
                    </div>
                    <button className={`button ${active ? "catalog-selected" : "button-light"} button-small`} disabled={busy || active} onClick={() => selectReference(item.id)}>
                      {busy && !active ? <LoaderCircle className="spin" size={14} /> : active ? <Check size={14} /> : <CircuitBoard size={14} />}
                      {active ? "Đang dùng" : "Chọn mạch"}
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="reference-catalog-empty">Chưa có mạch nào được lưu. Tải ảnh PCB chuẩn để tạo danh mục đầu tiên.</div>
          )}
          <div className="reference-current-title"><strong>Mạch đang dùng: {reference?.catalog?.find((item) => item.id === reference.active_reference_id)?.name || "Chưa chọn"}</strong><span>{reference?.components_count || 0} linh kiện</span></div>
          <div className="reference-preview-grid">
            <div><small>ẢNH MẠCH GỐC</small>{reference?.reference_image_base64 ? <img src={reference.reference_image_base64} alt="Ảnh PCB tham chiếu" /> : <div className="empty-image">Chưa có ảnh tham chiếu</div>}</div>
            <div><small>YOLO ĐÃ ĐÁNH DẤU</small>{detectedPreview ? <img src={detectedPreview} alt="Linh kiện trong mạch chuẩn" /> : <div className="empty-image">Ảnh đánh dấu sẽ hiển thị tại đây</div>}</div>
          </div>
          <div className="reference-count"><strong>{reference?.components_count || 0}</strong><span>linh kiện trong reference map</span></div>
          {reference?.components?.length > 0 && <div className="table-scroll reference-table"><table><thead><tr><th>#</th><th>Lớp linh kiện</th><th>Tọa độ tham chiếu</th></tr></thead><tbody>{reference.components.map((item, i) => <tr key={item.id || i}><td>{item.id || i + 1}</td><td><span className="class-pill">{item.class || item.label}</span></td><td className="muted">{Math.round(item.x)}, {Math.round(item.y)}</td></tr>)}</tbody></table></div>}
        </div>
      </div>
    </>
  );
}

import { CheckCircle2, CircleAlert, Clock3, Crosshair, ScanSearch, XCircle } from "lucide-react";
import StatCard from "../common/StatCard.jsx";

const statusMeta = {
  OK: { text: "MẠCH ĐẠT CHUẨN", className: "good", icon: CheckCircle2 },
  FAULTY: { text: "PHÁT HIỆN SAI KHÁC", className: "bad", icon: XCircle },
  UNALIGNED: { text: "ĐÃ QUÉT · CHƯA ĐỐI CHIẾU", className: "pending", icon: CircleAlert },
};

const cropLabels = {
  perspective: "PCB đã cắt · YOLO · OpenCV",
  roi: "PCB đã cắt · YOLO · OpenCV",
  full: "Ảnh gốc · YOLO · OpenCV",
};

export default function InspectionResults({ result }) {
  if (!result) return null;
  const verdict = result.verdict || result.status;
  const meta = statusMeta[verdict] || statusMeta.UNALIGNED;
  const StatusIcon = meta.icon;
  const stats = result.stats || {};
  return (
    <section className="results-section">
      <div className={`verdict-banner ${meta.className}`}>
        <StatusIcon size={22} />
        <div><strong>{result.verdict_vi || meta.text}</strong><span>{result.alignment_method || "Kết quả nhận diện YOLO"}</span></div>
        {!result.qa_available && <span className="verdict-note">Ảnh đã quét nhưng chưa thể căn chỉnh để so sánh vị trí</span>}
      </div>
      <div className="stats-grid">
        <StatCard label="Linh kiện phát hiện" value={stats.detected_count ?? result.total_count ?? result.detections?.length ?? 0} icon={ScanSearch} />
        <StatCard label="Khớp mạch chuẩn" value={stats.matched ?? "—"} icon={Crosshair} tone="green" />
        <StatCard label="Linh kiện thiếu" value={stats.missing ?? "—"} icon={CircleAlert} tone="orange" />
        <StatCard label="Sai khác / thời gian" value={result.latency_ms ? `${result.latency_ms} ms` : (stats.extra ?? 0)} icon={Clock3} tone="blue" />
      </div>
      {result.annotated_image && (
        <div className="result-image-card">
          <div className="section-title"><div><h3>Kết quả phân tích</h3><p>Các vùng nhận diện được đánh dấu trên ảnh</p></div><span className="image-tag">{cropLabels[result.crop_method] || "YOLO · OpenCV"}</span></div>
          <img className="result-image" src={result.annotated_image} alt="Ảnh PCB được chú thích kết quả" />
        </div>
      )}
      {result.detections?.length > 0 && (
        <div className="table-card">
          <div className="section-title"><div><h3>Danh sách linh kiện</h3><p>{result.detections.length} đối tượng được nhận diện</p></div></div>
          <div className="table-scroll"><table><thead><tr><th>#</th><th>Lớp linh kiện</th><th>Độ tin cậy</th><th>Trạng thái</th><th>Tọa độ tâm</th></tr></thead>
            <tbody>{result.detections.map((item, index) => <tr key={`${item.id || index}-${item.class}`}><td className="muted">{item.id || index + 1}</td><td><span className="class-pill">{item.class || item.label}</span></td><td>{Math.round((item.confidence || 0) * 100)}%</td><td><span className={`badge ${item.status === "OK" ? "badge-green" : item.status === "EXTRA" ? "badge-red" : "badge-gray"}`}>{item.status || "DETECTED"}</span></td><td className="muted">{Math.round(item.cx ?? item.x ?? 0)}, {Math.round(item.cy ?? item.y ?? 0)}</td></tr>)}</tbody>
          </table></div>
        </div>
      )}
      {result.missing_components_details?.length > 0 && (
        <div className="missing-list"><strong>Linh kiện thiếu cần kiểm tra</strong><span>{result.missing_components_details.map((item) => item.class).join(" · ")}</span></div>
      )}
    </section>
  );
}

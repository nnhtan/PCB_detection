import { CheckCircle2, CircleAlert, Clock3, Crosshair, ScanSearch, XCircle } from "lucide-react";
import StatCard from "../common/StatCard.jsx";

const statusMeta = {
  OK: { text: "MẠCH ĐẠT CHUẨN", className: "good", icon: CheckCircle2 },
  FAULTY: { text: "MẠCH BỊ LỖI (THIẾU LINH KIỆN)", className: "bad", icon: XCircle },
  UNALIGNED: { text: "ĐÃ QUÉT · CHƯA ĐỐI CHIẾU", className: "pending", icon: CircleAlert },
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
        <StatCard label="Thời gian xử lý" value={result.latency_ms ? `${result.latency_ms} ms` : "—"} icon={Clock3} tone="blue" />
      </div>
      {result.annotated_image && (
        <div className="result-image-card">
          <div className="section-title">
            <div>
              <h3>Kết quả phân tích bo mạch</h3>
              <p>
                {result.crop_method === "mono_cropper"
                  ? "Bo mạch đã được AI phát hiện, tự động cắt tách và nắn thẳng 90°"
                  : result.crop_method === "full"
                  ? "Đang dùng toàn bộ khung hình (Chưa tách được bo mạch riêng)"
                  : `Cắt theo phương pháp: ${result.crop_method}`}
              </p>
            </div>
            <span className={`image-tag ${result.crop_method === "mono_cropper" ? "tag-crop-success" : ""}`}>
              {result.crop_method === "mono_cropper" ? "✓ ĐÃ CẮT & NẮN THẲNG" : `CROP · ${result.crop_method || "OpenCV"}`}
            </span>
          </div>

          {result.crop_method === "full" && (
            <div className="alert alert-warning" style={{ margin: "0.5rem 0 1rem 0", fontSize: "0.85rem" }}>
              ⚠️ Thuật toán chưa bóc tách riêng được bo mạch nên đang dùng toàn bộ khung hình.
              Gợi ý: Điều chỉnh thanh trượt <strong>Thời gian lấy sáng (Exposure)</strong> để bo mạch rõ nét hơn, hoặc bật <strong>Vùng chọn tự do (ROI)</strong>.
            </div>
          )}

          <div className="result-image-wrapper">
            <img className="result-image" src={result.annotated_image} alt="Ảnh PCB được chú thích kết quả" />
          </div>

          {result.board_crop_image && result.board_crop_image !== result.annotated_image && (
            <details style={{ marginTop: "0.75rem", fontSize: "0.85rem", color: "var(--text-muted)" }}>
              <summary style={{ cursor: "pointer", fontWeight: 600, color: "var(--purple)" }}>
                🔍 Xem ảnh bo mạch gốc đã nắn thẳng (Chưa vẽ bounding box)
              </summary>
              <div style={{ marginTop: "0.5rem" }}>
                <img
                  className="result-image"
                  src={result.board_crop_image}
                  alt="Ảnh bo mạch đã nắn thẳng"
                  style={{ maxHeight: "350px", objectFit: "contain" }}
                />
              </div>
            </details>
          )}
        </div>
      )}
      {result.detections?.length > 0 && (
        <div className="table-card">
          <div className="section-title"><div><h3>Danh sách linh kiện</h3><p>{result.detections.length} đối tượng được nhận diện</p></div></div>
          <div className="table-scroll"><table><thead><tr><th>#</th><th>Lớp linh kiện</th><th>Độ tin cậy</th><th>Trạng thái</th><th>Tọa độ tâm</th></tr></thead>
            <tbody>{result.detections.map((item, index) => <tr key={`${item.id || index}-${item.class}`}><td className="muted">{item.id || index + 1}</td><td><span className="class-pill">{item.class || item.label}</span></td><td>{Math.round((item.confidence || 0) * 100)}%</td><td><span className={`badge ${item.status === "OK" ? "badge-green" : "badge-gray"}`}>{item.status || "DETECTED"}</span></td><td className="muted">{Math.round(item.cx ?? item.x ?? 0)}, {Math.round(item.cy ?? item.y ?? 0)}</td></tr>)}</tbody>
          </table></div>
        </div>
      )}
      {result.missing_components_details?.length > 0 && (
        <div className="missing-list"><strong>Linh kiện thiếu cần kiểm tra</strong><span>{result.missing_components_details.map((item) => item.class).join(" · ")}</span></div>
      )}
    </section>
  );
}

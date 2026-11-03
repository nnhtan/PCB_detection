import { CheckCircle2, CircleAlert, Clock3, Crosshair, ScanSearch, SearchCheck, XCircle } from "lucide-react";
import StatCard from "../common/StatCard.jsx";

const statusMeta = {
  OK: { text: "MẠCH ĐẠT CHUẨN", className: "good", icon: CheckCircle2 },
  FAULTY: { text: "MẠCH BỊ LỖI (SỐ LƯỢNG LINH KIỆN KHÔNG KHỚP)", className: "bad", icon: XCircle },
  REVIEW: { text: "CẦN KIỂM TRA LẠI (NHẬN DIỆN CHƯA CHẮC CHẮN)", className: "review", icon: SearchCheck },
  UNALIGNED: { text: "ĐÃ ĐỐI CHIẾU SỐ LƯỢNG · CHƯA CĂN CHỈNH VỊ TRÍ", className: "pending", icon: CircleAlert },
};

export default function InspectionResults({ result }) {
  if (!result) return null;
  const verdict = result.verdict || result.status;
  const meta = statusMeta[verdict] || statusMeta.UNALIGNED;
  const StatusIcon = meta.icon;
  const stats = result.stats || {};
  const comparedWithReference =
    result.qa_available === true || result.comparison_scope === "reference_position";
  const visibleDetections = (result.detections || []).filter(
    (item) => item.status !== "EXTRA",
  );
  return (
    <section className="results-section">
      <div className={`verdict-banner ${meta.className}`}>
        <StatusIcon size={22} />
        <div><strong>{result.verdict_vi || meta.text}</strong><span>{result.alignment_method || "Kết quả nhận diện YOLO"}</span></div>
        {result.comparison_note && <span className="verdict-note">{result.comparison_note}</span>}
      </div>
      <div className="stats-grid">
        <StatCard label="Linh kiện phát hiện" value={comparedWithReference ? stats.matched ?? visibleDetections.length : stats.detected_count ?? visibleDetections.length} icon={ScanSearch} />
        <StatCard label="Số linh kiện khớp" value={stats.matched ?? "—"} icon={Crosshair} tone="green" />
        {!comparedWithReference && <StatCard label="Linh kiện dư" value={stats.extra ?? 0} icon={CircleAlert} tone="orange" />}
        <StatCard label="Linh kiện thiếu" value={stats.missing ?? "—"} icon={CircleAlert} tone="orange" />
        <StatCard label="Thời gian xử lý" value={result.latency_ms ? `${result.latency_ms} ms` : "—"} icon={Clock3} tone="blue" />
      </div>
      {result.inventory_comparison?.length > 0 && (
        <div className="matching-list">
          <strong>Tổng số detections (thống kê toàn bộ)</strong>
          <span>{result.inventory_comparison.map((item) => `${item.class}: quét ${item.detected_count}/${item.reference_count}${item.missing ? ` · thiếu ${item.missing}` : ""}${item.extra ? ` · dư ${item.extra}` : ""}`).join(" · ")}</span>
        </div>
      )}
      {result.board_crop_image && (
        <div className="result-image-card">
          <div className="section-title"><div><h3>Ảnh PCB sau căn chỉnh/cắt</h3><p>Bo mạch sau khi tách vùng và căn về hướng mạch chuẩn</p></div><span className="image-tag">CROP · {result.crop_method || "không xác định"}</span></div>
          <img className="result-image" src={result.board_crop_image} alt="Ảnh PCB sau khi căn chỉnh và cắt" />
        </div>
      )}
      {result.annotated_image && (
        <div className="result-image-card">
          <div className="section-title"><div><h3>Kết quả phân tích</h3><p>Các vùng nhận diện được đánh dấu trên ảnh</p></div><span className="image-tag">{result.crop_method ? `CROP · ${result.crop_method}` : "YOLO · OpenCV"}</span></div>
          <img className="result-image" src={result.annotated_image} alt="Ảnh PCB được chú thích kết quả" />
        </div>
      )}
      {visibleDetections.length > 0 && (
        <div className="table-card">
          <div className="section-title"><div><h3>Danh sách linh kiện</h3><p>{visibleDetections.length} đối tượng được nhận diện</p></div></div>
          <div className="table-scroll"><table><thead><tr><th>#</th><th>Lớp linh kiện</th><th>Độ tin cậy</th><th>Đối chiếu</th><th>Trạng thái</th><th>Tọa độ tâm</th></tr></thead>
            <tbody>{visibleDetections.map((item, index) => {
              const modelClass = item.class || item.label;
              const classMismatch = item.class_match === false && item.matched_ref_class;
              const referenceClass = classMismatch ? item.matched_ref_class : modelClass;
              const yoloClass = item.yolo_class || (classMismatch ? modelClass : null);
              const yoloDisagrees = yoloClass && yoloClass !== referenceClass;
              return (
                <tr key={`${item.id || index}-${modelClass}`}>
                  <td className="muted">{item.id || index + 1}</td>
                  <td>
                    <span className="class-pill">{referenceClass}</span>
                    {yoloDisagrees && <span className="muted"> (tham chiếu) · YOLO: {yoloClass} ({Math.round((item.confidence || 0) * 100)}%)</span>}
                  </td>
                  <td>{Math.round((item.confidence || 0) * 100)}%</td>
                  <td>{item.matched_ref_id ? `Reference #${item.matched_ref_id}` : "—"}</td>
                  <td><span className={`badge ${item.status === "OK" ? "badge-green" : item.status === "EXTRA" ? "badge-red" : item.status === "REVIEW" ? "badge-review" : "badge-gray"}`}>{item.status || "DETECTED"}</span></td>
                  <td className="muted">{Math.round(item.cx ?? item.x ?? 0)}, {Math.round(item.cy ?? item.y ?? 0)}</td>
                </tr>
              );
            })}</tbody>
          </table></div>
        </div>
      )}
      {result.matched_components?.length > 0 && (
        <div className="matching-list"><strong>Linh kiện khớp vị trí</strong><span>{result.matched_components.map((item) => `Ref #${item.reference_id} ${item.reference_class} ↔ Detection #${item.detection_id} ${item.detection_class}${item.class_match ? "" : " (cần xác nhận loại)"} (${item.distance.toFixed(1)} px)`).join(" · ")}</span></div>
      )}
      {result.class_mismatch_details?.length > 0 && (
        <div className="review-list"><strong>Linh kiện nhận diện chưa chắc loại</strong><span>{result.class_mismatch_details.map((item) => `Ref #${item.reference_id}: ${item.reference_class} · YOLO dự đoán ${item.detection_class} (${Math.round(item.confidence * 100)}%)`).join(" · ")}</span></div>
      )}
      {result.missing_components_details?.length > 0 && (
        <div className="missing-list"><strong>Linh kiện thiếu cần kiểm tra</strong><span>{result.missing_components_details.map((item) => `Ref #${item.ref_id} ${item.class}`).join(" · ")}</span></div>
      )}
    </section>
  );
}

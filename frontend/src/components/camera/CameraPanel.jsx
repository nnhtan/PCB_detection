import { useState } from "react";
import { Camera, CircleStop, LoaderCircle, ScanLine, Video } from "lucide-react";
import PageHeader from "../common/PageHeader.jsx";
import InspectionResults from "../inspector/InspectionResults.jsx";
import { useCamera } from "../../hooks/useCamera.js";
import { api } from "../../services/api.js";

export default function CameraPanel() {
  const { status, error: cameraError, busy: cameraBusy, connect, disconnect } = useCamera();
  const [source, setSource] = useState("0");
  const [scanBusy, setScanBusy] = useState(false);
  const [scanError, setScanError] = useState("");
  const [result, setResult] = useState(null);
  const scan = async () => {
    setScanBusy(true);
    setScanError("");
    try {
      const response = await api.postForm("/api/camera/scan", new FormData());
      setResult(response);
    } catch (requestError) {
      setScanError(requestError.message);
    } finally {
      setScanBusy(false);
    }
  };
  return (
    <>
      <PageHeader eyebrow="LIVE INSPECTION" title="Camera trực tiếp" description="Kết nối webcam hoặc camera IP để xem stream và quét PCB ngay tại chỗ." />
      <div className="panel-card camera-panel">
        <div className="camera-toolbar">
          <label className="camera-source"><span>Nguồn camera</span><input value={source} onChange={(event) => setSource(event.target.value)} placeholder="0 hoặc URL camera IP" /></label>
          <div className="camera-actions">{status?.connected ? <button className="button button-danger" disabled={cameraBusy} onClick={disconnect}><CircleStop size={16} /> Ngắt kết nối</button> : <button className="button button-primary" disabled={cameraBusy || !source.trim()} onClick={() => connect(source)}>{cameraBusy ? <LoaderCircle className="spin" size={16} /> : <Video size={16} />} Kết nối camera</button>}<button className="button button-light" disabled={!status?.connected || scanBusy} onClick={scan}>{scanBusy ? <LoaderCircle className="spin" size={16} /> : <ScanLine size={16} />} Quét khung hình</button></div>
        </div>
        {(cameraError || scanError) && <div className="alert alert-error">{cameraError || scanError}</div>}
        <div className="camera-workspace">
          <section className="camera-view">
            <img src={api.cameraStream} alt="Luồng camera trực tiếp" />
            <div className="camera-live-pill"><span className={`status-dot ${status?.connected ? "online" : ""}`} />{status?.connected ? "LIVE" : "CHỜ CAMERA"}</div>
            {status?.connected && <div className="camera-fps">{status.fps || 0} FPS · FRAME #{status.frame_id}</div>}
            {!status?.connected && <div className="camera-overlay"><div className="camera-placeholder"><Camera size={30} /><strong>Chưa kết nối camera</strong><span>Nhập chỉ số webcam (0) hoặc URL camera IP phía trên</span></div></div>}
          </section>
          <section className="camera-result-panel">
            <div className="section-title"><div><h2><ScanLine size={18} /> Kết quả & đánh giá</h2><p>{result ? `Khung hình #${result.frame_id || status?.frame_id || "—"}` : "Quét khung hình để xem đánh giá PCB"}</p></div></div>
            {result ? <InspectionResults result={result} /> : (
              <div className="camera-result-empty">
                <span><ScanLine size={21} /></span>
                <strong>Chưa có kết quả quét</strong>
                <p>Kết nối camera và nhấn “Quét khung hình”. Kết quả nhận diện, đối chiếu và đánh giá sẽ hiển thị tại đây.</p>
              </div>
            )}
          </section>
        </div>
      </div>
    </>
  );
}

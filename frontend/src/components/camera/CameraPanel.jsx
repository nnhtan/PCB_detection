import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CheckCircle2, CircleStop, LoaderCircle, Pause, Play, RotateCcw, ScanLine, Video, Workflow, XCircle } from "lucide-react";
import PageHeader from "../common/PageHeader.jsx";
import InspectionResults from "../inspector/InspectionResults.jsx";
import { useCamera } from "../../hooks/useCamera.js";
import { api } from "../../services/api.js";

export default function CameraPanel() {
  const { status, error: cameraError, busy: cameraBusy, connect, disconnect } = useCamera();
  const [source, setSource] = useState("hikvision");
  const [scanBusy, setScanBusy] = useState(false);
  const [scanError, setScanError] = useState("");
  const [result, setResult] = useState(null);
  const [autoScan, setAutoScan] = useState(false);
  const [autoScanPaused, setAutoScanPaused] = useState(false);
  const [scanInterval, setScanInterval] = useState(3);
  const [autoScanCounts, setAutoScanCounts] = useState({ ok: 0, faulty: 0 });
  const scanInProgress = useRef(false);
  const countSession = useRef(0);

  const scan = useCallback(async (countAutomaticResult = false) => {
    if (scanInProgress.current) return;
    const scanCountSession = countSession.current;
    scanInProgress.current = true;
    setScanBusy(true);
    setScanError("");
    try {
      const response = await api.postForm("/api/camera/scan", new FormData());
      setResult(response);
      if (countAutomaticResult && scanCountSession === countSession.current) {
        const verdict = response.verdict || response.status;
        if (verdict === "OK") {
          setAutoScanCounts((counts) => ({ ...counts, ok: counts.ok + 1 }));
        } else if (verdict === "FAULTY") {
          setAutoScanCounts((counts) => ({
            ...counts,
            faulty: counts.faulty + 1,
          }));
        }
      }
    } catch (requestError) {
      setScanError(requestError.message);
    } finally {
      scanInProgress.current = false;
      setScanBusy(false);
    }
  }, []);

  useEffect(() => {
    if (!autoScan || autoScanPaused || !status?.connected) return undefined;
    const timer = window.setInterval(() => scan(true), scanInterval * 1000);
    return () => window.clearInterval(timer);
  }, [autoScan, autoScanPaused, scan, scanInterval, status?.connected]);

  const toggleAutoScan = () => {
    if (!autoScan) {
      countSession.current += 1;
      setAutoScanCounts({ ok: 0, faulty: 0 });
    }
    setAutoScanPaused(false);
    setAutoScan((enabled) => !enabled);
  };

  const restartAutoScanCounts = () => {
    countSession.current += 1;
    setAutoScanCounts({ ok: 0, faulty: 0 });
  };

  const stopCamera = async () => {
    setAutoScan(false);
    setAutoScanPaused(false);
    await disconnect();
  };

  return (
    <>
      <PageHeader eyebrow="LIVE INSPECTION" title="Camera trực tiếp" description="Kết nối camera Hikvision, webcam hoặc camera IP để xem stream và quét PCB ngay tại chỗ." />
      <div className="panel-card camera-panel">
        <div className="camera-toolbar">
          <label className="camera-source"><span>Nguồn camera</span><input value={source} onChange={(event) => setSource(event.target.value)} placeholder="hikvision, 0 hoặc URL camera IP" /></label>
          <div className="camera-actions">
            {status?.connected ? <button className="button button-danger" disabled={cameraBusy} onClick={stopCamera}><CircleStop size={16} /> Ngắt kết nối</button> : <button className="button button-primary" disabled={cameraBusy || !source.trim()} onClick={() => connect(source)}>{cameraBusy ? <LoaderCircle className="spin" size={16} /> : <Video size={16} />} Kết nối camera</button>}
            <button className="button button-light" disabled={!status?.connected || scanBusy} onClick={scan}>{scanBusy ? <LoaderCircle className="spin" size={16} /> : <ScanLine size={16} />} Quét khung hình</button>
            <div className="auto-scan-controls">
              <label className="auto-scan-interval" htmlFor="camera-scan-interval">Mỗi</label>
              <select id="camera-scan-interval" value={scanInterval} onChange={(event) => setScanInterval(Number(event.target.value))} aria-label="Chu kỳ quét tự động">
                <option value={3}>3 giây</option>
                <option value={5}>5 giây</option>
                <option value={10}>10 giây</option>
              </select>
              <button className={`button auto-scan-button ${autoScan ? "active" : ""}`} disabled={!status?.connected} aria-pressed={autoScan} onClick={toggleAutoScan}>
                <Workflow size={15} /> {autoScan ? "Dừng tự động" : "Tự động quét"}
              </button>
            </div>
          </div>
        </div>
        {(cameraError || status?.last_error || scanError) && <div className="alert alert-error">{cameraError || status?.last_error || scanError}</div>}
        <div className="camera-workspace">
          <section className="camera-view">
            <img src={api.cameraStream} alt="Luồng camera trực tiếp" />
            <div className="camera-live-pill"><span className={`status-dot ${status?.connected ? "online" : ""}`} />{status?.connected ? "LIVE" : "CHỜ CAMERA"}</div>
            {status?.connected && <div className="camera-fps">{status.fps || 0} FPS · FRAME #{status.frame_id}</div>}
            {!status?.connected && <div className="camera-overlay"><div className="camera-placeholder"><Camera size={30} /><strong>Chưa kết nối camera</strong><span>Nhập “hikvision”, chỉ số webcam (0) hoặc URL camera IP phía trên</span></div></div>}
          </section>
          <section className="camera-result-panel">
            <div className="section-title"><div><h2><ScanLine size={18} /> Kết quả & đánh giá</h2><p>{autoScanPaused ? "Đã tạm dừng quét tự động" : autoScan ? `Đang tự động quét mỗi ${scanInterval} giây` : result ? `Khung hình #${result.frame_id || status?.frame_id || "—"}` : "Quét khung hình để xem đánh giá PCB"}</p></div></div>
            <div className="auto-scan-totals" aria-live="polite">
              <div className="auto-scan-totals-heading">
                <strong>Số mạch đã quét</strong>
                <div className="auto-scan-counter-actions">
                  <button className={`button button-small ${autoScanPaused ? "auto-scan-resume" : "auto-scan-pause"}`} disabled={!autoScan} onClick={() => setAutoScanPaused((paused) => !paused)} title={autoScanPaused ? "Tiếp tục quét tự động" : "Tạm dừng quét tự động"} aria-label={autoScanPaused ? "Tiếp tục quét tự động" : "Tạm dừng quét tự động"}>
                    {autoScanPaused ? <Play size={14} /> : <Pause size={14} />} {autoScanPaused ? "Tiếp tục" : "Tạm dừng"}
                  </button>
                  <button className="button button-light button-small" onClick={restartAutoScanCounts} title="Đặt lại bộ đếm về 0" aria-label="Đặt lại bộ đếm mạch về 0">
                    <RotateCcw size={14} /> Bắt đầu lại
                  </button>
                </div>
              </div>
              <div className="auto-scan-total ok">
                <CheckCircle2 size={17} />
                <span>Mạch đạt chuẩn</span>
                <strong>{autoScanCounts.ok}</strong>
              </div>
              <div className="auto-scan-total faulty">
                <XCircle size={17} />
                <span>Mạch lỗi</span>
                <strong>{autoScanCounts.faulty}</strong>
              </div>
              <p>Bộ đếm tính theo từng lượt quét tự động có kết quả OK/FAULTY; UNALIGNED không tính.</p>
            </div>
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

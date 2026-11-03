import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CheckCircle2, CircleStop, Crop, LoaderCircle, Pause, Play, RotateCcw, RotateCw, ScanLine, Video, Workflow, XCircle, Zap } from "lucide-react";
import PageHeader from "../common/PageHeader.jsx";
import InspectionResults from "../inspector/InspectionResults.jsx";
import { useCamera } from "../../hooks/useCamera.js";
import { api } from "../../services/api.js";

export default function CameraPanel() {
  const { status, error: cameraError, busy: cameraBusy, connect, disconnect } = useCamera();
  const [source, setSource] = useState("169.254.128.194");
  const [scanBusy, setScanBusy] = useState(false);
  const [scanError, setScanError] = useState("");
  const [exposureUs, setExposureUs] = useState(20000);
  const [exposureBusy, setExposureBusy] = useState(false);
  const [exposureError, setExposureError] = useState("");
  const [result, setResult] = useState(null);
  const [autoCrop, setAutoCrop] = useState(true);
  const [referenceAlign, setReferenceAlign] = useState(true);
  const [fastMode, setFastMode] = useState(true);
  const [autoScan, setAutoScan] = useState(false);
  const [autoScanPaused, setAutoScanPaused] = useState(false);
  const [scanInterval, setScanInterval] = useState(0.5);
  const [autoScanCounts, setAutoScanCounts] = useState({ ok: 0, faulty: 0 });
  const scanInProgress = useRef(false);
  const countSession = useRef(0);

  useEffect(() => {
    if (status?.exposure_us != null) {
      setExposureUs(status.exposure_us);
    }
  }, [status?.source, status?.connected, status?.exposure_us]);

  const scan = useCallback(async (
    countResult = false,
    resumeConveyorAfterScan = false,
  ) => {
    if (scanInProgress.current) return;
    const scanCountSession = countSession.current;
    scanInProgress.current = true;
    setScanBusy(true);
    setScanError("");
    try {
      const form = new FormData();
      form.append("auto_crop", String(autoCrop));
      form.append("fast_mode", String(fastMode));
      form.append("imgsz", referenceAlign || !fastMode ? "1536" : "1024");
      form.append("scan_mode", referenceAlign ? "tilted_reference" : "standard");
      const response = await api.postForm("/api/camera/scan", form);
      setResult(response);
      const conveyorErrors = [];
      if (typeof response.is_ok === "boolean") {
        try {
          await api.sendConveyorResult(response.is_ok);
        } catch (sendError) {
          conveyorErrors.push(
            `Kết quả đã hiển thị nhưng không gửi được đến ESP: ${sendError.message}`,
          );
        }
      }
      if (resumeConveyorAfterScan) {
        try {
          await api.resumeConveyor();
        } catch (resumeError) {
          conveyorErrors.push(
            `Không gửi được lệnh chạy tiếp đến ESP: ${resumeError.message}`,
          );
        }
      }
      if (conveyorErrors.length > 0) setScanError(conveyorErrors.join(" "));
      if (countResult && scanCountSession === countSession.current) {
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
      setResult(null);
      setScanError(requestError.message);
    } finally {
      scanInProgress.current = false;
      setScanBusy(false);
    }
  }, [autoCrop, fastMode, referenceAlign]);
  const scanRef = useRef(scan);
  scanRef.current = scan;

  useEffect(() => {
    if (!status?.connected) return undefined;
    let cancelled = false;
    let timer;

    const waitForSensor = async () => {
      let retryDelay = 0;
      try {
        const response = await api.waitForConveyorEvent();
        if (!response.enabled) return;
        if (response.event === "SENSOR1") {
          await new Promise((resolve) => window.setTimeout(resolve, 1500));
          while (!cancelled && scanInProgress.current) {
            await new Promise((resolve) => window.setTimeout(resolve, 100));
          }
          if (!cancelled) await scanRef.current(true, true);
        }
      } catch (requestError) {
        retryDelay = 1000;
        if (!cancelled) {
          setScanError(
            `Không nhận được lệnh quét từ ESP: ${requestError.message}`,
          );
        }
      }

      if (!cancelled) timer = window.setTimeout(waitForSensor, retryDelay);
    };

    waitForSensor();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [status?.connected]);

  useEffect(() => {
    if (!autoScan || autoScanPaused || !status?.connected) return undefined;
    let cancelled = false;
    let timer;
    const scanOnSchedule = async () => {
      const startedAt = performance.now();
      await scan(true);
      if (!cancelled) {
        const remaining = scanInterval === 0
          ? 0
          : scanInterval * 1000 - (performance.now() - startedAt);
        timer = window.setTimeout(scanOnSchedule, Math.max(0, remaining));
      }
    };
    scanOnSchedule();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
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

  const applyExposure = async () => {
    setExposureBusy(true);
    setExposureError("");
    try {
      const form = new FormData();
      form.append("exposure_us", String(exposureUs));
      const response = await api.postForm("/api/camera/exposure", form);
      setExposureUs(response.exposure_us);
    } catch (requestError) {
      setExposureError(requestError.message);
    } finally {
      setExposureBusy(false);
    }
  };

  return (
    <>
      <PageHeader eyebrow="LIVE INSPECTION" title="Camera trực tiếp" description="Quét khung hình để xử lý PCB và hiển thị kết quả đối chiếu với mạch tham chiếu đang dùng." />
      <div className="panel-card camera-panel">
        <div className="camera-toolbar">
          <label className="camera-source"><span>Nguồn camera</span><input value={source} onChange={(event) => setSource(event.target.value)} placeholder="169.254.128.194, hikvision hoặc 0" /></label>
          <div className="camera-actions">
            {status?.connected ? <button className="button button-danger" disabled={cameraBusy} onClick={stopCamera}><CircleStop size={16} /> Ngắt kết nối</button> : <button className="button button-primary" disabled={cameraBusy || !source.trim()} onClick={() => connect(source)}>{cameraBusy ? <LoaderCircle className="spin" size={16} /> : <Video size={16} />} Kết nối camera</button>}
            <button className="button button-light" disabled={!status?.connected || scanBusy} onClick={scan}>{scanBusy ? <LoaderCircle className="spin" size={16} /> : <ScanLine size={16} />} Quét khung hình</button>
            <button
              type="button"
              className={`toggle-crop-button ${referenceAlign ? "active" : ""}`}
              disabled={scanBusy}
              onClick={() => {
                setReferenceAlign((enabled) => !enabled);
                setResult(null);
                setScanError("");
              }}
              title={referenceAlign ? "Ưu tiên biên PCB, dùng ORB làm dự phòng để căn frame theo mạch chuẩn" : "Dùng quy trình quét tiêu chuẩn; có thể kém ổn định với PCB nghiêng"}
              aria-pressed={referenceAlign}
            >
              <RotateCw size={15} />
              <span>Căn theo chuẩn</span>
              <span className="crop-indicator" />
              <strong>{referenceAlign ? "BẬT" : "TẮT"}</strong>
            </button>
            <button
              type="button"
              className={`toggle-crop-button ${autoCrop && !referenceAlign ? "active" : ""}`}
              disabled={referenceAlign}
              onClick={() => setAutoCrop((prev) => !prev)}
              title={referenceAlign ? "Đang căn chỉnh theo mạch chuẩn; Auto-Crop thường không áp dụng" : autoCrop ? "Tự động nhận diện & cắt nắn bo mạch: ĐANG BẬT" : "Tự động cắt bo mạch: ĐANG TẮT (quét toàn khung hình)"}
            >
              <Crop size={15} />
              <span>Auto-Crop</span>
              <span className="crop-indicator" />
              <strong>{autoCrop ? "BẬT" : "TẮT"}</strong>
            </button>
            <button
              type="button"
              className={`toggle-crop-button ${fastMode ? "" : "active"}`}
              onClick={() => setFastMode((enabled) => !enabled)}
              title={referenceAlign ? (fastMode ? "Căn theo chuẩn: ảnh YOLO 1536 px, SAHI lát nhanh 960 px" : "Căn theo chuẩn chi tiết: ảnh YOLO 1536 px, SAHI lát 320 px") : fastMode ? "Quét nhanh: ảnh YOLO 1024 px, SAHI lát 960 px" : "Quét chi tiết: ảnh YOLO 1536 px, SAHI lát 320 px, ưu tiên linh kiện nhỏ"}
            >
              <Zap size={15} />
              <span>{fastMode ? "Quét nhanh" : "Quét chi tiết"}</span>
              <span className="crop-indicator" />
              <strong>BẬT</strong>
            </button>
            <div className="auto-scan-controls">
              <label className="auto-scan-interval" htmlFor="camera-scan-interval">Mỗi</label>
              <select id="camera-scan-interval" value={scanInterval} onChange={(event) => setScanInterval(Number(event.target.value))} aria-label="Chu kỳ quét tự động">
                <option value={0}>Nhanh nhất</option>
                <option value={0.5}>0,5 giây</option>
                <option value={1}>1 giây</option>
                <option value={2}>2 giây</option>
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
        {status?.connected && status.exposure_min_us != null && (
          <div className="camera-exposure">
            <label className="camera-exposure-label" htmlFor="camera-exposure">
              <span>Thời gian lấy sáng</span>
              <strong>{(exposureUs / 1000).toFixed(1)} ms</strong>
            </label>
            <input
              id="camera-exposure"
              type="range"
              min={status.exposure_min_us}
              max={status.exposure_max_us}
              step={Math.max(1, Math.round((status.exposure_max_us - status.exposure_min_us) / 1000))}
              value={exposureUs}
              onChange={(event) => setExposureUs(Number(event.target.value))}
              disabled={exposureBusy}
              aria-label="Thời gian lấy sáng camera"
            />
            <button className="button button-light button-small" onClick={applyExposure} disabled={exposureBusy || exposureUs === status.exposure_us}>
              {exposureBusy ? <LoaderCircle className="spin" size={14} /> : null} Áp dụng
            </button>
            <span className="camera-exposure-hint">Khoảng hỗ trợ: {(status.exposure_min_us / 1000).toFixed(1)}–{(status.exposure_max_us / 1000).toFixed(1)} ms</span>
          </div>
        )}
        {(cameraError || status?.last_error || scanError) && <div className="alert alert-error">{cameraError || status?.last_error || scanError}</div>}
        {exposureError && <div className="alert alert-error">{exposureError}</div>}
        <div className="camera-workspace">
          <section className="camera-view">
            <img src={api.cameraStream} alt="Luồng camera trực tiếp" />
            <div className="camera-live-pill"><span className={`status-dot ${status?.connected ? "online" : ""}`} />{status?.connected ? "LIVE" : "CHỜ CAMERA"}</div>
            {status?.connected && <div className="camera-fps">{status.fps || 0} FPS · FRAME #{status.frame_id}</div>}
            {!status?.connected && <div className="camera-overlay"><div className="camera-placeholder"><Camera size={30} /><strong>Chưa kết nối camera</strong><span>Nhập IP camera (169.254.128.194), hikvision hoặc 0 phía trên</span></div></div>}
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
              <p>{referenceAlign ? "Ưu tiên tìm biên PCB để căn frame; chỉ dùng ORB khi không tìm được biên. Frame không có PCB hoặc không căn được sẽ không được nhận diện hay cộng vào bộ đếm." : "Bộ đếm chỉ tính kết quả OK/FAULTY; nếu chưa căn chỉnh được thì hệ thống vẫn đối chiếu số lượng nhưng chưa kết luận vị trí."}</p>
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

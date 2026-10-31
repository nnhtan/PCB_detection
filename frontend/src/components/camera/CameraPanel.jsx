import { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera,
  CheckCircle2,
  CircleStop,
  Crop,
  Eye,
  LoaderCircle,
  Maximize2,
  Minimize2,
  Pause,
  Play,
  RotateCcw,
  ScanLine,
  Sliders,
  Sun,
  Video,
  Workflow,
  XCircle,
  Zap,
} from "lucide-react";
import PageHeader from "../common/PageHeader.jsx";
import InspectionResults from "../inspector/InspectionResults.jsx";
import { useCamera } from "../../hooks/useCamera.js";
import { api } from "../../services/api.js";

export default function CameraPanel() {
  const { status, error: cameraError, busy: cameraBusy, connect, disconnect } = useCamera();
  const [source, setSource] = useState("hikvision");
  const [scanBusy, setScanBusy] = useState(false);
  const [scanError, setScanError] = useState("");
  const [exposureUs, setExposureUs] = useState(300000);
  const [exposureBusy, setExposureBusy] = useState(false);
  const [exposureError, setExposureError] = useState("");
  const [result, setResult] = useState(null);
  const [autoCrop, setAutoCrop] = useState(true);
  const [autoScan, setAutoScan] = useState(false);
  const [autoScanPaused, setAutoScanPaused] = useState(false);
  const [scanInterval, setScanInterval] = useState(3);
  const [realtimeScan, setRealtimeScan] = useState(false);
  const [realtimeFps, setRealtimeFps] = useState(0);
  const [realtimeLatency, setRealtimeLatency] = useState(0);
  const [autoScanCounts, setAutoScanCounts] = useState({ ok: 0, faulty: 0 });
  const scanInProgress = useRef(false);
  const countSession = useRef(0);
  const conveyorState = useRef("WAITING"); // "WAITING" (chờ mạch vào) | "LOCKED" (mạch đã chốt, đang trôi trên băng)
  const [conveyorMsg, setConveyorMsg] = useState("Đang chờ bo mạch...");

  // --- Quản lý Vùng Chọn Tự Do (ROI Crop) ---
  const [showRoiPanel, setShowRoiPanel] = useState(false);
  const [isDrawingRoi, setIsDrawingRoi] = useState(false);
  const [roi, setRoi] = useState({
    enabled: false,
    x1: 0.0,
    y1: 0.0,
    x2: 1.0,
    y2: 1.0,
    stream_crop: true,
  });
  const [roiBusy, setRoiBusy] = useState(false);
  const [roiMessage, setRoiMessage] = useState("");
  const [drawStart, setDrawStart] = useState(null);
  const [drawBox, setDrawBox] = useState(null);
  const imgRef = useRef(null);

  // Đồng bộ giá trị Exposure và ROI từ Camera Status
  useEffect(() => {
    if (status?.exposure_us != null) {
      setExposureUs(status.exposure_us);
    }
    if (status?.roi) {
      setRoi(status.roi);
    }
  }, [status?.exposure_us, status?.roi]);

  const scan = useCallback(async (countAutomaticResult = false) => {
    if (scanInProgress.current) return;
    const scanCountSession = countSession.current;
    scanInProgress.current = true;
    setScanBusy(true);
    setScanError("");
    try {
      const form = new FormData();
      form.append("auto_crop", String(autoCrop));
      form.append("apply_roi", String(roi.enabled));
      const response = await api.postForm("/api/camera/scan", form);

      const verdict = response.verdict || response.status;
      const isBoardPresent = Boolean(response.board_present) && verdict !== "NO_BOARD";
      const isStationary = response.is_stationary !== false;

      if (!isBoardPresent) {
        // Băng tải rỗng: Cập nhật hình ảnh trung tính, mở khóa sẵn sàng đón bo mạch tiếp theo
        setResult(response);
        conveyorState.current = "WAITING";
        setConveyorMsg("⚪ Băng tải trống · Đang chờ bo mạch...");
      } else {
        // Có bo mạch xuất hiện trong khung hình:
        if (conveyorState.current === "WAITING" || conveyorState.current === "MOVING") {
          if (!isStationary) {
            // Bo mạch vẫn đang trôi về phía cảm biến: hiển thị cập nhật hình nhưng CHƯA CHỐT
            setResult(response);
            conveyorState.current = "MOVING";
            setConveyorMsg(`🟡 Bo mạch đang vào (${response.motion_score || 0}) · Chờ cảm biến dừng...`);
          } else {
            // BO MẠCH ĐÃ DỪNG HẲN DƯỚI CẢM BIẾN: Chốt quét 1 lần và đếm +1
            setResult(response);
            if (countAutomaticResult && scanCountSession === countSession.current) {
              if (verdict === "OK") {
                setAutoScanCounts((counts) => ({ ...counts, ok: counts.ok + 1 }));
              } else if (verdict === "FAULTY") {
                setAutoScanCounts((counts) => ({
                  ...counts,
                  faulty: counts.faulty + 1,
                }));
              }
            }
            // Khóa lại: Bo mạch này đã chốt lúc dừng, đợi bo mạch rời băng mới đón mạch tiếp theo
            conveyorState.current = "LOCKED";
            setConveyorMsg(`🟢 Đã chốt lúc dừng (${verdict === "OK" ? "OK" : "LỖI"}) · Đợi mạch rời băng...`);
          }
        } else {
          // conveyorState === "LOCKED": Mạch hiện tại đang đứng yên hoặc bắt đầu trôi ra ngoài
          // Giữ nguyên kết quả, không cộng đếm lại, không quét trùng lặp
        }
      }
    } catch (requestError) {
      setScanError(requestError.message);
    } finally {
      scanInProgress.current = false;
      setScanBusy(false);
    }
  }, [autoCrop, roi.enabled]);

  useEffect(() => {
    if (!autoScan || autoScanPaused || !status?.connected) return undefined;
    const timer = window.setInterval(() => scan(true), scanInterval * 1000);
    return () => window.clearInterval(timer);
  }, [autoScan, autoScanPaused, scan, scanInterval, status?.connected]);

  // --- Chế độ Quét Realtime (Liên tục không độ trễ) ---
  useEffect(() => {
    if (!realtimeScan || !status?.connected) return undefined;
    let cancelled = false;

    const runRealtimeLoop = async () => {
      while (!cancelled) {
        if (!scanInProgress.current) {
          const t0 = performance.now();
          await scan(true);
          const elapsed = performance.now() - t0;
          if (!cancelled) {
            setRealtimeLatency(Math.round(elapsed));
            setRealtimeFps(Number((1000 / Math.max(elapsed, 1)).toFixed(1)));
          }
        }
        await new Promise((r) => setTimeout(r, 40));
      }
    };

    runRealtimeLoop();
    return () => {
      cancelled = true;
    };
  }, [realtimeScan, scan, status?.connected]);

  const toggleRealtimeScan = () => {
    if (!realtimeScan) {
      setAutoScan(false);
      countSession.current += 1;
      setAutoScanCounts({ ok: 0, faulty: 0 });
    }
    setRealtimeScan((prev) => !prev);
  };

  const toggleAutoScan = () => {
    if (!autoScan) {
      setRealtimeScan(false);
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
    setRealtimeScan(false);
    setAutoScan(false);
    setAutoScanPaused(false);
    await disconnect();
  };

  // --- Hàm điều chỉnh Exposure ---
  const applyExposureVal = async (us) => {
    const minVal = status?.exposure_min_us ?? 34;
    const maxVal = status?.exposure_max_us ?? 2000000;
    const clamped = Math.max(minVal, Math.min(maxVal, Math.round(us)));
    setExposureUs(clamped);
    setExposureBusy(true);
    setExposureError("");
    try {
      const form = new FormData();
      form.append("exposure_us", String(clamped));
      const response = await api.postForm("/api/camera/exposure", form);
      setExposureUs(response.exposure_us);
    } catch (requestError) {
      setExposureError(requestError.message);
    } finally {
      setExposureBusy(false);
    }
  };

  // --- Hàm cập nhật cấu hình ROI lên Backend ---
  const updateRoi = async (newRoi) => {
    setRoiBusy(true);
    setRoiMessage("");
    try {
      const form = new FormData();
      form.append("enabled", String(newRoi.enabled));
      form.append("x1", String(newRoi.x1));
      form.append("y1", String(newRoi.y1));
      form.append("x2", String(newRoi.x2));
      form.append("y2", String(newRoi.y2));
      form.append("stream_crop", String(newRoi.stream_crop ?? true));
      const res = await api.postForm("/api/camera/roi", form);
      setRoi(res.roi);
      setRoiMessage(newRoi.enabled ? "Đã áp dụng cắt vùng ROI!" : "Đã đặt lại về toàn khung hình.");
      setTimeout(() => setRoiMessage(""), 3000);
    } catch (err) {
      setRoiMessage(`Lỗi ROI: ${err.message}`);
    } finally {
      setRoiBusy(false);
    }
  };

  // --- Xử lý sự kiện kéo chuột vẽ vùng Crop trực tiếp trên ảnh ---
  const handleLayerMouseDown = (e) => {
    if (!isDrawingRoi || !imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    setDrawStart({ x, y });
    setDrawBox({ x1: x, y1: y, x2: x, y2: y });
  };

  const handleLayerMouseMove = (e) => {
    if (!isDrawingRoi || !drawStart || !imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    setDrawBox({
      x1: Math.min(drawStart.x, x),
      y1: Math.min(drawStart.y, y),
      x2: Math.max(drawStart.x, x),
      y2: Math.max(drawStart.y, y),
    });
  };

  const handleLayerMouseUp = async () => {
    if (!isDrawingRoi || !drawBox) return;
    const w = drawBox.x2 - drawBox.x1;
    const h = drawBox.y2 - drawBox.y1;
    if (w > 0.02 && h > 0.02) {
      await updateRoi({
        ...roi,
        enabled: true,
        x1: Number(drawBox.x1.toFixed(3)),
        y1: Number(drawBox.y1.toFixed(3)),
        x2: Number(drawBox.x2.toFixed(3)),
        y2: Number(drawBox.y2.toFixed(3)),
      });
    }
    setDrawStart(null);
    setDrawBox(null);
    setIsDrawingRoi(false);
  };

  const minExp = status?.exposure_min_us ?? 34;
  const maxExp = status?.exposure_max_us ?? 2000000;
  const expStep = Math.max(100, Math.round((maxExp - minExp) / 2000));
  const isHikvision = status?.source === "hikvision" || status?.source === "hikrobot" || status?.exposure_us != null;

  return (
    <>
      <PageHeader
        eyebrow="LIVE INSPECTION"
        title="Camera trực tiếp"
        description="Kết nối camera Hikvision hoặc Webcam, tinh chỉnh phơi sáng, cắt bỏ phần thừa ngoài và quét PCB tự động."
      />
      <div className="panel-card camera-panel">
        {/* THANH CÔNG CỤ CHÍNH */}
        <div className="camera-toolbar">
          <label className="camera-source">
            <span>Nguồn camera</span>
            <input
              value={source}
              onChange={(event) => setSource(event.target.value)}
              placeholder="hikvision, 0 hoặc URL camera IP"
            />
          </label>
          <div className="camera-actions">
            {status?.connected ? (
              <button className="button button-danger" disabled={cameraBusy} onClick={stopCamera}>
                <CircleStop size={16} /> Ngắt kết nối
              </button>
            ) : (
              <button
                className="button button-primary"
                disabled={cameraBusy || !source.trim()}
                onClick={() => connect(source)}
              >
                {cameraBusy ? <LoaderCircle className="spin" size={16} /> : <Video size={16} />} Kết nối camera
              </button>
            )}

            <button
              className="button button-light"
              disabled={!status?.connected || scanBusy || realtimeScan}
              onClick={() => scan(false)}
            >
              {scanBusy ? <LoaderCircle className="spin" size={16} /> : <ScanLine size={16} />} Quét khung hình
            </button>

            {/* Nút Quét Realtime (Liên tục) */}
            <button
              type="button"
              className={`button ${realtimeScan ? "button-primary active-pulse" : "button-light"}`}
              disabled={!status?.connected}
              onClick={toggleRealtimeScan}
              title="Quét liên tục không độ trễ theo luồng camera (Real-time)"
            >
              <Zap size={15} className={realtimeScan ? "text-amber-300" : ""} />
              <span>{realtimeScan ? "Dừng Realtime" : "⚡ Quét Realtime"}</span>
            </button>

            {/* Nút bật/tắt Bảng Crop Vùng Tự Do (ROI) */}
            <button
              type="button"
              className={`button ${roi.enabled ? "button-primary" : "button-light"}`}
              onClick={() => setShowRoiPanel((prev) => !prev)}
              title="Cắt bỏ phần thừa xung quanh (băng tải, nền ngoài)"
            >
              <Crop size={15} />
              <span>Cắt vùng tự do</span>
              {roi.enabled && <span className="crop-indicator" />}
            </button>

            {/* Nút Auto-Crop thuật toán */}
            <button
              type="button"
              className={`toggle-crop-button ${autoCrop ? "active" : ""}`}
              onClick={() => setAutoCrop((prev) => !prev)}
              title={
                autoCrop
                  ? "Tự động nhận diện & cắt nắn bo mạch: ĐANG BẬT"
                  : "Tự động cắt bo mạch: ĐANG TẮT (quét toàn khung hình)"
              }
            >
              <Maximize2 size={14} />
              <span>Auto-Crop PCB</span>
              <span className="crop-indicator" />
              <strong>{autoCrop ? "BẬT" : "TẮT"}</strong>
            </button>

            {/* Điều khiển quét tự động định kỳ */}
            <div className="auto-scan-controls">
              <label className="auto-scan-interval" htmlFor="camera-scan-interval">Mỗi</label>
              <select
                id="camera-scan-interval"
                value={scanInterval}
                onChange={(event) => setScanInterval(Number(event.target.value))}
                aria-label="Chu kỳ quét tự động"
              >
                <option value={2}>2 giây</option>
                <option value={3}>3 giây</option>
                <option value={5}>5 giây</option>
                <option value={10}>10 giây</option>
              </select>
              <button
                className={`button auto-scan-button ${autoScan ? "active" : ""}`}
                disabled={!status?.connected}
                aria-pressed={autoScan}
                onClick={toggleAutoScan}
              >
                <Workflow size={15} /> {autoScan ? "Dừng tự động" : "Tự động quét"}
              </button>
            </div>
          </div>
        </div>

        {/* THANH ĐIỀU CHỈNH THỜI GIAN LẤY SÁNG (EXPOSURE TIME SLIDER) */}
        {(isHikvision || status?.connected) && (
          <div className="camera-exposure">
            <div className="camera-exposure-header">
              <div className="camera-exposure-label">
                <Sun size={16} className="text-amber-500" />
                <span>Thời gian lấy sáng (Exposure Time):</span>
                <span className="camera-exposure-value">
                  {(exposureUs / 1000).toFixed(1)} ms
                </span>
                <span className="text-xs text-gray-400">
                  ({Math.round(exposureUs).toLocaleString()} µs)
                </span>
              </div>
              <div className="camera-exposure-presets">
                <span className="text-xs text-gray-500 font-semibold mr-1">Mốc nhanh:</span>
                {[15000, 30000, 50000, 100000, 300000].map((presetUs) => (
                  <button
                    key={presetUs}
                    type="button"
                    className={`camera-preset-btn ${Math.abs(exposureUs - presetUs) < 1000 ? "active" : ""}`}
                    onClick={() => applyExposureVal(presetUs)}
                    disabled={exposureBusy || !status?.connected}
                  >
                    {presetUs / 1000} ms
                  </button>
                ))}
              </div>
            </div>

            <div className="camera-exposure-controls">
              <div className="camera-exposure-slider-wrap">
                <button
                  type="button"
                  className="camera-step-btn"
                  title="Giảm 50 ms"
                  disabled={exposureBusy || !status?.connected}
                  onClick={() => applyExposureVal(exposureUs - 50000)}
                >
                  -50ms
                </button>
                <button
                  type="button"
                  className="camera-step-btn"
                  title="Giảm 10 ms"
                  disabled={exposureBusy || !status?.connected}
                  onClick={() => applyExposureVal(exposureUs - 10000)}
                >
                  -10ms
                </button>
                <input
                  id="camera-exposure"
                  type="range"
                  min={minExp}
                  max={maxExp}
                  step={expStep}
                  value={exposureUs}
                  onChange={(event) => setExposureUs(Number(event.target.value))}
                  onMouseUp={() => applyExposureVal(exposureUs)}
                  onTouchEnd={() => applyExposureVal(exposureUs)}
                  disabled={exposureBusy || !status?.connected}
                  aria-label="Thời gian lấy sáng camera"
                />
                <button
                  type="button"
                  className="camera-step-btn"
                  title="Tăng 10 ms"
                  disabled={exposureBusy || !status?.connected}
                  onClick={() => applyExposureVal(exposureUs + 10000)}
                >
                  +10ms
                </button>
                <button
                  type="button"
                  className="camera-step-btn"
                  title="Tăng 50 ms"
                  disabled={exposureBusy || !status?.connected}
                  onClick={() => applyExposureVal(exposureUs + 50000)}
                >
                  +50ms
                </button>
              </div>

              <button
                className="button button-light button-small"
                onClick={() => applyExposureVal(exposureUs)}
                disabled={exposureBusy || !status?.connected}
              >
                {exposureBusy ? <LoaderCircle className="spin" size={14} /> : null} Áp dụng
              </button>
            </div>
            <span className="camera-exposure-hint">
              Khoảng hỗ trợ: {(minExp / 1000).toFixed(1)} ms – {(maxExp / 1000).toFixed(1)} ms
            </span>
          </div>
        )}

        {/* BẢNG ĐIỀU KHIỂN CROP VÙNG TỰ DO (ROI PANEL) */}
        {showRoiPanel && (
          <div className="camera-roi-panel">
            <div className="camera-roi-header">
              <div className="camera-roi-title">
                <Crop size={16} />
                <span>Cắt Vùng Tự Do (Loại bỏ nền thừa ngoài bo mạch)</span>
                <span className={`camera-roi-status-pill ${roi.enabled ? "active" : "inactive"}`}>
                  {roi.enabled ? "ĐANG CẮT ROI" : "TOÀN CẢNH"}
                </span>
              </div>

              <div className="camera-roi-actions">
                <button
                  type="button"
                  className={`button button-small ${isDrawingRoi ? "button-danger" : "button-primary"}`}
                  onClick={() => {
                    setIsDrawingRoi((prev) => !prev);
                    if (!isDrawingRoi && roi.stream_crop) {
                      // Tạm chuyển sang xem toàn cảnh để người dùng dễ kéo chuột vẽ
                      updateRoi({ ...roi, stream_crop: false });
                    }
                  }}
                  title="Click chuột lên ảnh camera rồi kéo thả để chọn khung"
                >
                  {isDrawingRoi ? "Hủy vẽ chuột" : "✏️ Kéo chuột vẽ vùng"}
                </button>

                <button
                  type="button"
                  className={`button button-small ${roi.enabled ? "button-light" : "button-primary"}`}
                  onClick={() => updateRoi({ ...roi, enabled: !roi.enabled })}
                  disabled={roiBusy}
                >
                  {roi.enabled ? "Tắt cắt ROI" : "Bật cắt ROI"}
                </button>

                <button
                  type="button"
                  className="button button-light button-small"
                  onClick={() => updateRoi({ ...roi, stream_crop: !roi.stream_crop })}
                  title={roi.stream_crop ? "Đang phóng to vùng crop. Nhấn để xem toàn cảnh kèm khung." : "Đang xem toàn cảnh. Nhấn để phóng to vùng crop."}
                >
                  {roi.stream_crop ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
                  <span>{roi.stream_crop ? "Chế độ: Zoom vùng Crop" : "Chế độ: Toàn cảnh + Khung"}</span>
                </button>

                {/* Các Preset cắt viền nhanh */}
                <button
                  type="button"
                  className="camera-preset-btn"
                  onClick={() => updateRoi({ ...roi, enabled: true, x1: 0.10, y1: 0.08, x2: 0.90, y2: 0.92 })}
                >
                  Cắt viền 10%
                </button>
                <button
                  type="button"
                  className="camera-preset-btn"
                  onClick={() => updateRoi({ ...roi, enabled: true, x1: 0.20, y1: 0.15, x2: 0.80, y2: 0.85 })}
                >
                  Cắt viền 20%
                </button>
                <button
                  type="button"
                  className="camera-preset-btn"
                  onClick={() => updateRoi({ enabled: false, x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0, stream_crop: true })}
                >
                  Reset 100%
                </button>
              </div>
            </div>

            {/* Chi tiết chỉnh tọa độ từng cạnh (Lề trái, phải, trên, dưới) */}
            <div className="camera-roi-details">
              <div className="camera-roi-input-group">
                <label>Trái (X1): {Math.round(roi.x1 * 100)}%</label>
                <input
                  type="range"
                  min="0"
                  max="0.45"
                  step="0.01"
                  value={roi.x1}
                  onChange={(e) => setRoi((prev) => ({ ...prev, x1: parseFloat(e.target.value) }))}
                  onMouseUp={() => updateRoi({ ...roi, enabled: true })}
                />
              </div>

              <div className="camera-roi-input-group">
                <label>Phải (X2): {Math.round(roi.x2 * 100)}%</label>
                <input
                  type="range"
                  min="0.55"
                  max="1.0"
                  step="0.01"
                  value={roi.x2}
                  onChange={(e) => setRoi((prev) => ({ ...prev, x2: parseFloat(e.target.value) }))}
                  onMouseUp={() => updateRoi({ ...roi, enabled: true })}
                />
              </div>

              <div className="camera-roi-input-group">
                <label>Trên (Y1): {Math.round(roi.y1 * 100)}%</label>
                <input
                  type="range"
                  min="0"
                  max="0.45"
                  step="0.01"
                  value={roi.y1}
                  onChange={(e) => setRoi((prev) => ({ ...prev, y1: parseFloat(e.target.value) }))}
                  onMouseUp={() => updateRoi({ ...roi, enabled: true })}
                />
              </div>

              <div className="camera-roi-input-group">
                <label>Dưới (Y2): {Math.round(roi.y2 * 100)}%</label>
                <input
                  type="range"
                  min="0.55"
                  max="1.0"
                  step="0.01"
                  value={roi.y2}
                  onChange={(e) => setRoi((prev) => ({ ...prev, y2: parseFloat(e.target.value) }))}
                  onMouseUp={() => updateRoi({ ...roi, enabled: true })}
                />
              </div>
            </div>
            {roiMessage && <span className="text-xs text-blue-600 font-semibold">{roiMessage}</span>}
          </div>
        )}

        {/* THÔNG BÁO LỖI NẾU CÓ */}
        {(cameraError || status?.last_error || scanError) && (
          <div className="alert alert-error">{cameraError || status?.last_error || scanError}</div>
        )}
        {exposureError && <div className="alert alert-error">{exposureError}</div>}

        {/* KHÔNG GIAN LÀM VIỆC CAMERA & KẾT QUẢ */}
        <div className="camera-workspace">
          <section className="camera-view">
            <img
              ref={imgRef}
              src={api.cameraStream}
              alt="Luồng camera trực tiếp"
              draggable={false}
            />

            {/* Lớp phủ tương tác kéo chuột chọn ROI */}
            {isDrawingRoi && (
              <div
                className="roi-drawing-layer"
                onMouseDown={handleLayerMouseDown}
                onMouseMove={handleLayerMouseMove}
                onMouseUp={handleLayerMouseUp}
              >
                {drawBox && (
                  <div
                    className="roi-active-box"
                    style={{
                      left: `${drawBox.x1 * 100}%`,
                      top: `${drawBox.y1 * 100}%`,
                      width: `${(drawBox.x2 - drawBox.x1) * 100}%`,
                      height: `${(drawBox.y2 - drawBox.y1) * 100}%`,
                    }}
                  >
                    <div className="roi-box-label">
                      Vùng chọn: {Math.round((drawBox.x2 - drawBox.x1) * 100)}% x {Math.round((drawBox.y2 - drawBox.y1) * 100)}%
                    </div>
                  </div>
                )}
                <div className="roi-guide-banner">
                  👉 Nhấn giữ chuột trái và kéo để vẽ hình chữ nhật bao quanh bo mạch
                </div>
              </div>
            )}

            <div className="camera-live-pill">
              <span className={`status-dot ${status?.connected ? "online" : ""}`} />
              {status?.connected ? "LIVE" : "CHỜ CAMERA"}
              {roi.enabled && <span className="text-cyan-400 font-bold ml-1">· ROI CROP</span>}
            </div>

            {realtimeScan && (
              <div className={`camera-realtime-badge ${conveyorState.current === "LOCKED" ? "locked" : "waiting"}`}>
                <span className={`realtime-live-dot ${conveyorState.current === "LOCKED" ? "locked" : "waiting"}`} />
                <span>{conveyorMsg} ({realtimeFps} FPS)</span>
              </div>
            )}

            {status?.connected && (
              <div className="camera-fps">
                {status.fps || 0} FPS · FRAME #{status.frame_id}
              </div>
            )}

            {!status?.connected && (
              <div className="camera-overlay">
                <div className="camera-placeholder">
                  <Camera size={30} />
                  <strong>Chưa kết nối camera</strong>
                  <span>Nhập “hikvision”, chỉ số webcam (0) hoặc URL camera IP phía trên</span>
                </div>
              </div>
            )}
          </section>

          {/* KHUNG KẾT QUẢ QUÉT */}
          <section className="camera-result-panel">
            <div className="section-title">
              <div>
                <h2><ScanLine size={18} /> Kết quả & đánh giá</h2>
                <p>
                  {autoScanPaused
                    ? "Đã tạm dừng quét tự động"
                    : autoScan
                    ? `Đang tự động quét mỗi ${scanInterval} giây`
                    : result
                    ? `Khung hình #${result.frame_id || status?.frame_id || "—"}`
                    : "Quét khung hình để xem đánh giá PCB"}
                </p>
              </div>
            </div>

            <div className="auto-scan-totals" aria-live="polite">
              <div className="auto-scan-totals-heading">
                <strong>Số mạch đã quét</strong>
                <div className="auto-scan-counter-actions">
                  <button
                    className={`button button-small ${autoScanPaused ? "auto-scan-resume" : "auto-scan-pause"}`}
                    disabled={!autoScan}
                    onClick={() => setAutoScanPaused((paused) => !paused)}
                    title={autoScanPaused ? "Tiếp tục quét tự động" : "Tạm dừng quét tự động"}
                    aria-label={autoScanPaused ? "Tiếp tục quét tự động" : "Tạm dừng quét tự động"}
                  >
                    {autoScanPaused ? <Play size={14} /> : <Pause size={14} />} {autoScanPaused ? "Tiếp tục" : "Tạm dừng"}
                  </button>
                  <button
                    className="button button-light button-small"
                    onClick={restartAutoScanCounts}
                    title="Đặt lại bộ đếm về 0"
                    aria-label="Đặt lại bộ đếm mạch về 0"
                  >
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

            {result ? (
              <InspectionResults result={result} />
            ) : (
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

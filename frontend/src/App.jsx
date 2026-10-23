import { useEffect, useRef } from "react";
import { AlertTriangle, Bell, ChevronRight, Cpu } from "lucide-react";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import Sidebar from "./components/layout/Sidebar.jsx";
import QAPage from "./pages/QAPage.jsx";
import DetectPage from "./pages/DetectPage.jsx";
import CameraPage from "./pages/CameraPage.jsx";
import ReferencePage from "./pages/ReferencePage.jsx";
import DocsPage from "./pages/DocsPage.jsx";
import { useAppStore } from "./store/appStore.js";
import { useCamera } from "./hooks/useCamera.js";

const titles = {
  "/": "Kiểm tra PCB",
  "/detect": "Nhận diện YOLO",
  "/camera": "Camera trực tiếp",
  "/reference": "Mạch tham chiếu",
  "/guide": "API & hướng dẫn",
};

export default function App() {
  const location = useLocation();
  const { status: cameraStatus, connect: connectCamera } = useCamera();
  const cameraConnectAttempted = useRef(false);
  const loadSystem = useAppStore((state) => state.loadSystem);
  const error = useAppStore((state) => state.error);
  const info = useAppStore((state) => state.info);
  useEffect(() => { loadSystem(); }, [loadSystem]);
  useEffect(() => {
    if (!cameraStatus || cameraConnectAttempted.current) return;
    cameraConnectAttempted.current = true;
    if (!cameraStatus.connected || cameraStatus.source?.toLowerCase() !== "hikvision") {
      connectCamera("hikvision");
    }
  }, [cameraStatus, connectCamera]);

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumbs"><span>Workspace</span><ChevronRight size={14} /><strong>{titles[location.pathname] || "PCB Vision AI"}</strong></div>
          <div className="topbar-right"><span className="topbar-device"><Cpu size={15} /> {info?.device_name || "Đang kết nối API"}</span><button className="icon-button" aria-label="Thông báo trạng thái"><Bell size={17} /></button><div className="avatar">PV</div></div>
        </header>
        <div className="page-content">
          {error && <div className="alert alert-warning"><AlertTriangle size={16} />{error} <Link to="/guide">Xem hướng dẫn API</Link></div>}
          <Routes>
            <Route path="/" element={<QAPage />} />
            <Route path="/detect" element={<DetectPage />} />
            <Route path="/camera" element={<CameraPage />} />
            <Route path="/reference" element={<ReferencePage />} />
            <Route path="/guide" element={<DocsPage />} />
            <Route path="*" element={<QAPage />} />
          </Routes>
          <footer className="page-footer"><span>PCB VISION AI</span><span>AI-powered visual quality inspection</span><span>© 2025</span></footer>
        </div>
      </main>
    </div>
  );
}

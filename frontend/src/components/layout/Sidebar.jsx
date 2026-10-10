import {
  Activity,
  BookOpenText,
  Camera,
  CircuitBoard,
  ScanLine,
  ScanSearch,
  Settings2,
} from "lucide-react";
import { NavLink } from "react-router-dom";
import { useAppStore } from "../../store/appStore.js";

const links = [
  { to: "/", label: "Kiểm tra PCB", icon: ScanSearch, end: true },
  { to: "/detect", label: "Nhận diện YOLO", icon: ScanLine },
  { to: "/camera", label: "Camera trực tiếp", icon: Camera },
  { to: "/reference", label: "Mạch tham chiếu", icon: CircuitBoard },
  { to: "/guide", label: "API & hướng dẫn", icon: BookOpenText },
];

export default function Sidebar() {
  const info = useAppStore((state) => state.info);
  const loaded = info?.reference?.loaded;
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark"><CircuitBoard size={22} /></div>
        <div><strong>PCB Vision</strong><span>AI INSPECTION</span></div>
      </div>
      <div className="nav-label">WORKSPACE</div>
      <nav className="nav-list">
        {links.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
            <Icon size={18} strokeWidth={1.8} /><span>{label}</span>
            {to === "/" && <span className="nav-shortcut">QA</span>}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-spacer" />
      <div className="sidebar-status">
        <div className="sidebar-status-top"><Activity size={15} /><span>HỆ THỐNG</span><span className={`status-dot ${info ? "online" : ""}`} /></div>
        <div className="sidebar-status-row"><span>Model</span><strong>{info?.model_name || "Đang tải..."}</strong></div>
        <div className="sidebar-status-row"><span>Thiết bị</span><strong>{info?.device_name || "—"}</strong></div>
        <div className="sidebar-status-row"><span>Mạch chuẩn</span><strong className={loaded ? "text-success" : "text-warning"}>{loaded ? "Đã nạp" : "Chưa nạp"}</strong></div>
      </div>
      <div className="sidebar-footer"><Settings2 size={15} /> PCB Vision AI <span>v1.0</span></div>
    </aside>
  );
}

"""
PCB Vision AI - Unified Launcher
Script khởi chạy toàn bộ hệ thống (Backend FastAPI + Frontend React) chỉ bằng một lệnh duy nhất.
Tự động mở trình duyệt và dọn dẹp tiến trình an toàn khi tắt (Ctrl + C).
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_PORT = 5173

# Biến lưu trữ tiến trình con
processes = []


def print_banner():
    banner = r"""
============================================================
       PCB VISION AI - KHỞI ĐỘNG HỆ THỐNG TỰ ĐỘNG
============================================================
"""
    print(banner)


def kill_process_tree(pid: int):
    """Tắt toàn bộ cây tiến trình con trên Windows"""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def kill_existing_on_port(port: int):
    """Giải phóng cổng nếu đang có tiến trình cũ chiếm dụng"""
    if sys.platform != "win32":
        return
    try:
        cmd = f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
        )
        pids = [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]
        for pid_str in set(pids):
            pid = int(pid_str)
            if pid != os.getpid():
                print(f"[*] Đang giải phóng cổng {port} (PID: {pid})...")
                kill_process_tree(pid)
                time.sleep(0.5)
    except Exception:
        pass


def cleanup():
    """Tắt toàn bộ tiến trình khi thoát"""
    print("\n[+] Đang dừng toàn bộ dịch vụ...")
    for p in processes:
        if p.poll() is None:
            kill_process_tree(p.pid)
    print("[+] Toàn bộ tiến trình đã được đóng an toàn. Tạm biệt!\n")


def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)


def wait_for_url(url: str, timeout: float = 30.0, step: float = 0.5) -> bool:
    """Đợi endpoint phản hồi HTTP 200"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PCBLauncher/1.0"})
            with urllib.request.urlopen(req, timeout=1.5) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(step)
    return False


def main():
    print_banner()

    # Đăng ký bắt tín hiệu Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 1. Kiểm tra và giải phóng cổng 8000 & 5173 nếu có tiến trình cũ
    print("[1/4] Kiểm tra tài nguyên và cổng mạng...")
    kill_existing_on_port(BACKEND_PORT)
    kill_existing_on_port(FRONTEND_PORT)

    # 2. Khởi chạy Backend FastAPI
    print(f"[2/4] Đang khởi động Backend FastAPI (Port {BACKEND_PORT})...")
    python_exe = sys.executable
    backend_cmd = [
        python_exe,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(BACKEND_PORT),
        "--reload",
    ]

    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(ROOT_DIR),
        shell=False,
    )
    processes.append(backend_proc)

    # 3. Khởi chạy Frontend React (Vite)
    frontend_proc = None
    frontend_url = f"http://localhost:{FRONTEND_PORT}"
    backend_url = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

    has_frontend = (FRONTEND_DIR / "package.json").is_file()
    if has_frontend:
        print(f"[3/4] Đang khởi động Frontend Vite Dev Server (Port {FRONTEND_PORT})...")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        try:
            frontend_proc = subprocess.Popen(
                [npm_cmd, "run", "dev"],
                cwd=str(FRONTEND_DIR),
                shell=(sys.platform == "win32"),
            )
            processes.append(frontend_proc)
        except Exception as e:
            print(f"[!] Không thể khởi chạy Vite dev server ({e}), sẽ dùng frontend tích hợp qua FastAPI.")

    # 4. Chờ dịch vụ sẵn sàng
    print("[4/4] Đang chờ hệ thống nạp model YOLO và sẵn sàng...")
    backend_ready = wait_for_url(f"{backend_url}/health", timeout=35.0)

    if not backend_ready:
        print("[!] Backend phản hồi chậm hoặc có lỗi khi nạp model. Vui lòng kiểm tra log bên trên.")
    else:
        print("[OK] Backend đã sẵn sàng!")

    # Xác định URL mở trên trình duyệt
    target_url = backend_url
    if frontend_proc and frontend_proc.poll() is None:
        if wait_for_url(frontend_url, timeout=10.0):
            target_url = frontend_url
            print(f"[OK] Frontend đã sẵn sàng tại: {target_url}")

    print(f"\n============================================================")
    print(f"  >>> HỆ THỐNG ĐÃ KHỞI CHẠY THÀNH CÔNG! <<<")
    print(f"  * Web App:    {target_url}")
    print(f"  * API Docs:   {backend_url}/docs")
    print(f"  * Trạng thái: Nhấn Ctrl + C để dừng toàn bộ hệ thống")
    print(f"============================================================\n")

    # Tự động mở trình duyệt
    try:
        webbrowser.open(target_url)
    except Exception:
        pass

    # Giữ tiến trình chạy chính
    try:
        while True:
            # Kiểm tra xem có tiến trình nào bị sập bất ngờ không
            if backend_proc.poll() is not None:
                print("\n[!] Backend đã dừng đột ngột.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()

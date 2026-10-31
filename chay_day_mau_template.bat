@echo off
chcp 65001 >nul
title Hikvision Camera Viewer - Bản Dạy Mẫu & Khớp Mẫu (Template Matching)
cd /d "%~dp0"
echo =====================================================================
echo  HIKVISION CAMERA - BẢN DẠY MẪU & KHỚP MẪU BẰNG TAY (TEMPLATE MATCHING)
echo  Kéo chuột trực tiếp trên màn hình camera để vẽ khung BBox quanh bo mạch PCB
echo =====================================================================
python camera_viewer_template.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LỖI] Đã xảy ra lỗi khi chạy chương trình.
    pause
)

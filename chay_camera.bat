@echo off
chcp 65001 >nul
title Hikvision Camera Viewer - Industrial Camera
cd /d "%~dp0"
echo [Camera] Đang khởi động chương trình xem camera Hikvision...
python camera_viewer.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LỖI] Đã xảy ra lỗi khi chạy camera.
    pause
)

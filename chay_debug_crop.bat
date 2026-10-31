@echo off
chcp 65001 >nul
title PCB Vision AI - Visual Crop Debugger
cd /d "%~dp0"
echo [PCB Vision AI] Đang khởi động công cụ Debug thuật toán cắt ảnh...
python debug_crop.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LỖI] Đã xảy ra lỗi khi chạy công cụ debug.
    pause
)

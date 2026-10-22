@echo off
chcp 65001 >nul
title PCB Vision AI - Drag & Drop GUI
cd /d "%~dp0"
echo [PCB Vision AI] Đang khởi động giao diện...
python pcb_gui.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LỖI] Đã xảy ra lỗi khi khởi chạy ứng dụng GUI.
    pause
)

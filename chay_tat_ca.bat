@echo off
chcp 65001 >nul
title PCB Vision AI - Khởi động hệ thống tự động
cd /d "%~dp0"
echo [PCB Vision AI] Đang khởi động toàn bộ hệ thống...
python run.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LỖI] Đã xảy ra lỗi khi chạy hệ thống.
    pause
)

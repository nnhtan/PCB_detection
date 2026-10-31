@echo off
chcp 65001 >nul
title Hikvision Camera Viewer - Phương Pháp Trừ Nền (Background Subtraction)
cd /d "%~dp0"
echo =====================================================================
echo  PCB INSPECTION - BẢN NHẬN DIỆN & CẮT PCB BẰNG PHƯƠNG PHÁP TRỪ NỀN
echo  - Bấm phím [C]    : Quét & Đổi nguồn Camera (Hikvision / Webcam / File)
echo  - Bấm phím [SPACE]: Chụp lưu ảnh nền băng tải rỗng (Reference Background)
echo =====================================================================
python camera_viewer_bg_sub.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LỖI] Đã xảy ra lỗi khi chạy chương trình.
    pause
)

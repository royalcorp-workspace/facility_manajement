@echo off
setlocal
cd /d "%~dp0\.."

if not exist .venv\Scripts\python.exe (
    echo [ERROR] Virtual environment .venv tidak ditemukan!
    echo Jalankan instalasi environment terlebih dahulu.
    pause
    exit /b 1
)

set PYTHONUTF8=1
.\.venv\Scripts\python.exe tools\roi_calibrator.py %*

if errorlevel 1 (
    echo [ERROR] Terjadi kesalahan saat menjalankan ROI Calibrator.
    pause
)

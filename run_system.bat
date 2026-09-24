@echo off
setlocal
cd /d "%~dp0"

title Facility Management — Production Launcher (Port 8070)

echo ============================================================
echo   FACILITY MANAGEMENT — PRODUCTION LAUNCHER
echo ============================================================
echo.

:: 1. Validasi Interpreter Python .venv
if not exist .venv\Scripts\python.exe (
    echo [ERROR] Virtual environment .venv tidak ditemukan!
    echo Pastikan environment Python sudah diinisialisasi di direktori ini.
    echo.
    pause
    exit /b 1
)

set PYTHONUTF8=1

:: 2. Eksekusi Schema Audit Gate
echo [STEP 1/3] Menjalankan Audit Skema Konfigurasi...
.\.venv\Scripts\python.exe tools\audit_schema.py
if errorlevel 1 (
    echo.
    echo [ERROR] Schema audit gagal! Perbaiki konfigurasi kamera sebelum melanjutkan.
    echo.
    pause
    exit /b 1
)

:: 3. Otomatis Membuka Browser ke Web Dashboard
echo [STEP 2/3] Membuka Browser Web Dashboard (http://127.0.0.1:8070)...
start "" http://127.0.0.1:8070

:: 4. Meluncurkan Master Orchestrator System
echo [STEP 3/3] Meluncurkan Master Orchestrator System...
echo.
.\.venv\Scripts\python.exe main.py --host 0.0.0.0 --port 8070 %*

if errorlevel 1 (
    echo.
    echo [ERROR] Sistem berhenti dengan error code: %errorlevel%
    pause
)

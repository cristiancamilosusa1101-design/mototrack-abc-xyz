@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8501' -TimeoutSec 2 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }"
if %ERRORLEVEL% EQU 0 (
    start "" "http://127.0.0.1:8501"
    exit /b 0
)

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --server.address 127.0.0.1 --server.port 8501 --server.headless false
) else (
    python -m streamlit run "%~dp0app.py" --server.address 127.0.0.1 --server.port 8501 --server.headless false
)

if errorlevel 1 (
    echo.
    echo No se pudo iniciar. Instale las dependencias con:
    echo python -m pip install -r requirements.txt
    pause
)
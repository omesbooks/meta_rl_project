@echo off
REM ตัวช่วยรันโปรแกรม (สร้าง venv ครั้งแรก + ติดตั้ง + รัน)
cd /d "%~dp0"

if not exist ".venv" (
    echo [setup] creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [setup] installing dependencies... (ใช้เวลาสักครู่)
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

python app.py
pause

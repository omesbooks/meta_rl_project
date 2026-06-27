@echo off
setlocal
cd /d "%~dp0"

set "PY=%CD%\.venv\Scripts\python.exe"
set "NEED_SETUP=0"

if not exist "%PY%" (
    echo [setup] .venv not found. Creating virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        python -m venv .venv
    )
    set "NEED_SETUP=1"
)

if not exist "%PY%" (
    echo [error] Cannot find or create .venv\Scripts\python.exe
    echo Please install Python 3, then run this file again.
    pause
    exit /b 1
)

if exist "requirements.txt" (
    echo [setup] Checking requirements...
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [warn] Requirements install failed. Trying to launch with current environment...
    )
)

if not exist "rl_app.py" (
    echo [error] rl_app.py not found in this folder:
    echo %CD%
    pause
    exit /b 1
)

echo [run] Starting RL Trading Studio...
"%PY%" rl_app.py

if errorlevel 1 (
    echo.
    echo [error] App closed with an error.
    pause
)

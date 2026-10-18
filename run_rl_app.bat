@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem RL Trading Studio launcher.
rem   run_rl_app.bat              web GUI (rl_web.py, native window)  [default]
rem   run_rl_app.bat --classic    classic CustomTkinter app (rl_app.py)
rem   run_rl_app.bat --smoke      self-test: hidden window, every page, exit code 0 = OK
rem Other arguments (e.g. --debug) are passed to the app.

set "PY=%CD%\.venv\Scripts\python.exe"
set "APP=rl_web.py"
set "ARGS="
set "NOPAUSE=0"

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--classic" (
    set "APP=rl_app.py"
) else (
    if /i "%~1"=="--smoke" set "NOPAUSE=1"
    set "ARGS=!ARGS! %1"
)
shift
goto parse
:parsed

if not exist "%PY%" (
    echo [setup] .venv not found. Creating virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        python -m venv .venv
    )
)

if not exist "%PY%" (
    echo [error] Cannot find or create .venv\Scripts\python.exe
    echo Please install Python 3, then run this file again.
    if "%NOPAUSE%"=="0" pause
    exit /b 1
)

if exist "requirements.txt" (
    echo [setup] Checking requirements...
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [warn] Requirements install failed. Trying to launch with current environment...
    )
)

if not exist "%APP%" (
    echo [error] %APP% not found in this folder:
    echo %CD%
    if "%NOPAUSE%"=="0" pause
    exit /b 1
)

echo [run] Starting RL Trading Studio (%APP%)...
"%PY%" %APP%%ARGS%
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [error] App closed with an error ^(exit %RC%^).
    if "%NOPAUSE%"=="0" pause
)
exit /b %RC%

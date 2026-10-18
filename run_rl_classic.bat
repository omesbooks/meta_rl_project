@echo off
rem The classic CustomTkinter app (rl_app.py).
call "%~dp0run_rl_app.bat" --classic %*
exit /b %ERRORLEVEL%

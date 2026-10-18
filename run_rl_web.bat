@echo off
rem Same as run_rl_app.bat (the web GUI is now the default). Kept for old shortcuts.
call "%~dp0run_rl_app.bat" %*
exit /b %ERRORLEVEL%

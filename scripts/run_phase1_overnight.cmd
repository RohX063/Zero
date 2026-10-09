@echo off
setlocal
cd /d "%~dp0.."

where py >nul 2>&1
if %errorlevel%==0 (
    py -3 "%~dp0run_phase1_overnight.py"
    exit /b %errorlevel%
)

where python >nul 2>&1
if %errorlevel%==0 (
    python "%~dp0run_phase1_overnight.py"
    exit /b %errorlevel%
)

echo ERROR: Python launcher not found. Install Python 3 and ensure py.exe or python.exe is on PATH.
exit /b 2

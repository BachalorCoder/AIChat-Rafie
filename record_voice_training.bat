@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo.
echo Starting Rafie voice training recorder...
echo.

"%PYTHON_EXE%" -m localagent.voice_training

echo.
pause
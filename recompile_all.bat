@echo off
setlocal

cd /d "%~dp0"

echo.
echo Recompiling Rafie Python files...
echo Project folder: %CD%
echo.

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python:
"%PYTHON_EXE%" --version
echo.

echo Checking config.json...
"%PYTHON_EXE%" -m json.tool config.json >nul

if errorlevel 1 (
    echo.
    echo config.json is broken. Fix the JSON error shown above.
    echo.
    pause
    exit /b 1
)

echo config.json is valid.
echo.

echo Checking and compiling all Python files...
echo Skipping .venv and __pycache__ folders.
echo.

"%PYTHON_EXE%" -m compileall -f -q -x ".*[\\/]\.venv[\\/].*|.*[\\/]__pycache__[\\/].*" .

if errorlevel 1 (
    echo.
    echo Compile failed. Look above for the file and line number that broke.
    echo.
    pause
    exit /b 1
)

echo.
echo All Python files compiled successfully.
echo config.json is valid.
echo No syntax errors found.
echo.

pause
exit /b 0
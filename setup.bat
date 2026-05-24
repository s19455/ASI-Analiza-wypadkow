@echo off
REM Skrypt setup dla Windows - automates environment creation and testing

echo.
echo ========================================================================
echo ASI - Crash Severity Prediction - Setup Script (Windows)
echo ========================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+ first.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists. Skipping.
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Could not upgrade pip (may be locked). Continuing...
)

pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.txt.
    pause
    exit /b 1
)

pip install -e ".[dev]"
if errorlevel 1 (
    echo ERROR: Failed to install package with dev extras.
    pause
    exit /b 1
)

echo [4/4] Running tests...
pytest -q
if errorlevel 1 (
    echo WARNING: Some tests failed. Check output above.
) else (
    echo All tests passed!
)

echo.
echo ========================================================================
echo Setup complete!
echo ========================================================================
echo.
echo Next steps:
echo 1. API (Docker): docker-compose up --build
echo 2. Full project: kedro run
echo 3. API only: uvicorn crash_kedro.api.app:app --reload
echo.
echo For more details, see docs/SETUP.md
echo.
pause



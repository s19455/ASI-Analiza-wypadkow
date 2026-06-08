@echo off
REM Setup srodowiska dla Windows

python --version >nul 2>&1
if errorlevel 1 (
    echo Brak Pythona. Zainstaluj Python 3.10+.
    pause
    exit /b 1
)

if exist .venv (
    echo Srodowisko .venv juz istnieje - pomijam.
) else (
    echo Tworze srodowisko .venv...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Instaluje zaleznosci...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"

echo Uruchamiam testy...
pytest -q

echo.
echo Gotowe. Co dalej:
echo   kedro run
echo   uvicorn crash_kedro.api.app:app --reload
echo   docker-compose up --build
pause

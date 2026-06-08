#!/bin/bash
# Setup srodowiska dla Linux/macOS

if ! command -v python3 &> /dev/null; then
    echo "Brak Pythona. Zainstaluj Python 3.10+."
    exit 1
fi

echo "Python: $(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"

if [ -d ".venv" ]; then
    echo "Srodowisko .venv juz istnieje - pomijam."
else
    echo "Tworze srodowisko .venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Instaluje zaleznosci..."
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"

echo "Uruchamiam testy..."
pytest -q

echo ""
echo "Gotowe. Co dalej:"
echo "  source .venv/bin/activate"
echo "  kedro run"
echo "  uvicorn crash_kedro.api.app:app --reload"
echo "  docker-compose up --build"

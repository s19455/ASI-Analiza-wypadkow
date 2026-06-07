#!/bin/bash

# Skrypt setup dla Linux/macOS - automates environment creation and testing

echo ""
echo "========================================================================"
echo "ASI - Crash Severity Prediction - Setup Script (Linux/macOS)"
echo "========================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found. Please install Python 3.10+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

echo ""
echo "[1/4] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping."
else
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        exit 1
    fi
fi

echo ""
echo "[2/4] Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment."
    exit 1
fi

echo ""
echo "[3/4] Installing dependencies..."
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "WARNING: Could not upgrade pip (may be locked). Continuing..."
fi

pip install -r requirements.txt
pip install -e ".[dev]"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install package with dev extras."
    exit 1
fi

echo ""
echo "[4/4] Running tests..."
pytest -q
if [ $? -ne 0 ]; then
    echo "WARNING: Some tests failed. Check output above."
else
    echo "All tests passed!"
fi

echo ""
echo "========================================================================"
echo "Setup complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Activate venv: source .venv/bin/activate"
echo "2. API (Docker): docker-compose up --build"
echo "3. Full project: kedro run"
echo "4. API only: uvicorn crash_kedro.api.app:app --reload"
echo ""
echo "For more details, see docs/SETUP.md"
echo ""


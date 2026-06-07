# Instrukcje instalacji i uruchomienia

## Wymagania systemowe

- **Python:** 3.10 lub wyżej
- **Git:** do klonowania repozytorium
- **Docker** (opcjonalnie): do uruchomienia API w kontenerze
- **RAM:** co najmniej 4 GB
- **Dysk:** co najmniej 2 GB na dane i modele

## Opcja 1: Uruchomienie API w Dockerze (zalecane dla produkcji)

### Przygotowanie
Wymagany: `docker` i `docker-compose`

### Instalacja
```bash
# Klonowanie repozytorium
git clone <URL-repo>
cd ASI-Analiza-wypadkow

# Budowanie i uruchamianie
docker-compose up --build
```

### Użycie
- **Swagger (dokumentacja interaktywna):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health
- **POST /predict** — predykcja dla jednego rekordu
- **GET /predictions/recent** — ostatnie predykcje z logów

Parametry zmienne można ustawić w `docker-compose.yml`, np. `MODEL_PATH`.

---

## Opcja 2: Uruchomienie pełnego projektu (dla development)

### Przygotowanie środowiska - automatycznie (zalecane)

**Linux/macOS:**
```bash
# Klonowanie repozytorium
git clone <URL-repo>
cd ASI-Analiza-wypadkow

# Uruchomienie skryptu setup (automatyczne)
bash setup.sh
```

**Windows:**
```batch
# Klonowanie repozytorium
git clone <URL-repo>
cd ASI-Analiza-wypadkow

# Uruchomienie skryptu setup (automatyczne)
setup.bat
```

Skrypty `setup.sh` (Linux/macOS) i `setup.bat` (Windows) automatycznie:
1. Tworzą wirtualne środowisko (`.venv`)
2. Aktywują je
3. Instalują zależności z `requirements.txt` oraz `pyproject.toml[dev]`
4. Uruchamiają testy (`pytest -q`)

### Przygotowanie środowiska - ręcznie

Jeśli wolisz kontrolę ręczną:

```bash
# Klonowanie repozytorium
git clone <URL-repo>
cd ASI-Analiza-wypadkow

# Tworzenie wirtualnego środowiska
python -m venv .venv

# Aktywacja (Windows)
.venv\Scripts\activate

# Aktywacja (Linux/Mac)
source .venv/bin/activate
```

### Instalacja zależności
```bash
# Dla pracy nad pełnym projektem (z dev toolami)
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Uruchomienie pipeline'ów

#### Cały pipeline (data + model)
```bash
kedro run
```

#### Wybrany pipeline
```bash
# Tylko przygotowanie danych
kedro run --pipeline=data_processing

# Tylko trening bazowy
kedro run --pipeline=modeling

# Tuning hiperparametrów (Grid Search, Random Search, Optuna)
kedro run --pipeline=tuning

# Selekcja cech
kedro run --pipeline=feature_selection

# AutoML (AutoGluon)
kedro run --pipeline=autogluon

# AutoML (TPOT inne podejście)
kedro run --pipeline=automl
```

### Śledzenie eksperymentów (MLflow)
```bash
# Uruchomienie UI MLflow w przeglądarce
mlflow ui

# Dostępne pod http://localhost:5000
```

### Uruchomienie API
```bash
# Tryb development (hot reload)
uvicorn crash_kedro.api.app:app --reload

# Tryb produkcyjny
uvicorn crash_kedro.api.app:app --host 0.0.0.0 --port 8000
```

### Skrypty pomocnicze
```bash
# Demo — predykcje na zbiorze testowym
python scripts/demo.py

# Monitoring driftu danych
python scripts/run_drift_monitoring.py

# Walidacja wytrenowanego modelu
python scripts/validate_model.py

# Weryfikacja enkoderów
python scripts/verify_encoders.py
```

### Testy i analiza kodu
```bash
# Uruchomienie wszystkich testów
pytest -q

# Pokrycie kodów
pytest --cov=src/crash_kedro

# Linting (ruff)
ruff check src tests

# Analiza jakości (pylint) — główne moduły
pylint src/crash_kedro/api/app.py src/crash_kedro/utils/encoders.py src/crash_kedro/__main__.py

# Formatowanie kodu (opcjonalnie)
ruff format src tests
```

---

## Opcja 3: Tylko API (minimalne zależności)

Jeśli chcesz pracować **wyłącznie z API**, używając predefiniowanego modelu:

```bash
# Instalacja
pip install -r requirements-api.txt

# Uruchomienie
uvicorn crash_kedro.api.app:app --reload
```

Wymagane są pliki modeli w `data/06_models/` (np. `model.pkl`, `encoders.pkl`).

> Jeśli pracujesz na Pythonie 3.13, pełny zestaw ML może nie instalować się bezpośrednio
> z powodu braku zgodnych buildów dla części pakietów treningowych. W takim przypadku
> użyj `requirements-api.txt` dla backendu + Streamlit, a pełne środowisko ML uruchamiaj
> najlepiej na Pythonie 3.10–3.12.

---

## Rozwiązywanie problemów

### Problem: Brak modelu przy uruchomieniu API
**Objawy:** `health()` zwraca `"model_loaded": false`

**Rozwiązanie:**
1. Upewnij się, żeśród jest plik `data/06_models/model.pkl` lub `tuned_model.pkl`
2. Uruchom pełny pipeline: `kedro run`
3. Lub ustaw ścieżkę środowiską: `export MODEL_PATH=/path/to/model.pkl`

### Problem: Brak enkoderów kategorycznych
**Objawy:** predykcje zwracają kody `-1` dla wszystkich kategorii

**Rozwiązanie:**
1. Upewnij się, że jest `data/06_models/encoders.pkl`
2. Uruchom pipeline danych: `kedro run --pipeline=data_processing`

### Problem: Brak danych wejściowych
**Objawy:** `Plik nie znaleziony data/01_raw/crash_data.csv`

**Rozwiązanie:**
1. Pobierz dane ze źródła (Montgomery County, Maryland 2015-2024)
2. Umieść w `data/01_raw/crash_data.csv`
3. Uruchom pipeline

### Problem: Konflikt portów
Jeśli port `8000` jest zajęty:
```bash
uvicorn crash_kedro.api.app:app --port 8001
```

---

## Zmienne środowiskowe

| Zmienna | Opis | Domyślnie |
|---------|------|----------|
| `MODEL_PATH` | Ścieżka do pliku modelu | Automatyczne wyszukiwanie w `data/06_models/` |
| `PYTHONPATH` | Ścieżka do kodu źródłowego | `src` (w Dockerze automatyczne) |

Przykład:
```bash
export MODEL_PATH=/app/data/06_models/tuned_model.pkl
uvicorn crash_kedro.api.app:app
```

---

## Weryfikacja instalacji

Aby sprawdzić, że wszystko działa poprawnie:

```bash
# Testy
pytest -q

# Linting
ruff check src tests
pylint src/crash_kedro/api/app.py src/crash_kedro/utils/encoders.py

# Demo
python scripts/demo.py
```

Jeśli wszystkie komendy się powodzą, instalacja jest poprawna.

---

## Kolejne kroki

- Przeczytaj [ARCHITECTURE.md](ARCHITECTURE.md), aby zrozumieć strukturę projektu
- Przeczytaj [API.md](API.md), aby poznać endpointy
- Przeczytaj [DATA_DICTIONARY.md](DATA_DICTIONARY.md), aby zrozumieć dane wejściowe


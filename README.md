# ASI - Analiza wypadków drogowych

Projekt zaliczeniowy z ASI. Przewidywanie stopnia obrażeń w wypadkach drogowych na podstawie danych z Montgomery County, Maryland (2015-2024).

## Materiały online

- Baseline notebook: [`notebooks/01_baseline_model.ipynb`](notebooks/01_baseline_model.ipynb)
- Notebook porównawczy modeli: [`notebooks/02_model_comparison.ipynb`](notebooks/02_model_comparison.ipynb)
- Diagram architektury: [`docs/architecture.drawio`](docs/architecture.drawio)
- Instrukcje uruchomienia: [`docs/SETUP.md`](docs/SETUP.md)
- Plan prezentacji: [`docs/PRESENTATION.md`](docs/PRESENTATION.md)

## Problem

Klasyfikacja - na podstawie informacji o wypadku (pogoda, prędkość, typ kolizji itp.) chcemy przewidzieć czy doszło do obrażeń i jak poważnych. 3 klasy:

- **NO_INJURY** (~82%) - brak obrażeń
- **MINOR** (~17%) - drobne obrażenia
- **SERIOUS** (~1%) - poważne obrażenia / zgon

Dataset ma 172 tys. wierszy i 43 kolumny.

## Wyniki

| Model | Accuracy | F1 ważone | F1 makro |
|-------|----------|-----------|----------|
| **LightGBM (Optuna, 50 prób)** | 0.79 | **0.78** | **0.47** |
| Autogluon (best: LightGBM) | 0.83 | 0.77 | 0.40 |
| XGBoost | 0.83 | 0.77 | 0.40 |
| Gradient Boosting | 0.82 | 0.77 | 0.40 |
| Random Forest (baseline) | 0.82 | 0.75 | 0.33 |

Dane są mocno niezbalansowane (klasa SERIOUS to ~1%), więc patrzymy głównie na F1 makro.

## Co jest w projekcie

- Pipeline Kedro - preprocessing + trening (`kedro run`)
- Selekcja cech (SelectKBest)
- Strojenie hiperparametrów (Grid Search, Random Search, Optuna)
- AutoML (Autogluon)
- Śledzenie eksperymentów (MLflow)
- API FastAPI + Docker + Streamlit
- CI/CD na GitHub Actions
- Monitoring driftu (Evidently + `scripts/run_drift_monitoring.py`)
- **Pełna dokumentacja w katalogu `docs/`**

## Zgodność z wymaganiami projektu

✅ **Przenoszalność między środowiskami** — Git, requirements.txt, Dockerfile  
✅ **Łatwe odtworzenie** — Docker (API), instrukcje SETUP.md (pełny projekt)  
✅ **Rozdzielenie logiki (data | model | app)** — Oddzielne pipeline'y + API  
✅ **Modularność** — Funkcje, pipeline'y, moduły, dobrze zorganizowana struktura  
✅ **Dokumentacja** — README.md + pełny katalog docs/ (SETUP, ARCHITECTURE, API, DATA_DICTIONARY, MODELS, CONTRIBUTING)  
✅ **Czysty kod / PEP8** — Pylint 10/10/10, ruff bez błędów  
✅ **Minimum 8 pkt Pylint** — Sprawdzone: 10.00/10  
✅ **Aplikacja jako API** — FastAPI + HTTP endpoints  
✅ **requirements.txt + Dockerfile** — Oba przechowywane  
✅ **CI/CD** — GitHub Actions workflows (ci.yml, cd.yml, ct.yml)

## Uruchomienie

### Najszybciej: API w Dockerze (bez konfiguracji)

```bash
docker-compose up --build
# API dostępna pod http://localhost:8000/docs (Swagger UI)
```

Wymaga: Docker + docker-compose. Żadnej dodatkowej konfiguracji!

### Pełny projekt (dla development)

```bash
# Przygotowanie
python -m venv .venv
.venv\Scripts\activate              # Windows
source .venv/bin/activate           # Linux/macOS

# LUB użyj automatycznego skryptu setup:
bash setup.sh                        # Linux/macOS
setup.bat                            # Windows

pip install -r requirements.txt
pip install -e ".[dev]"


# Pipeline ML
kedro run                              # cały pipeline
kedro run --pipeline=data_processing   # tylko przygotowanie danych
kedro run --pipeline=modeling          # tylko trening bazowy
kedro run --pipeline=tuning            # tuning hiperparametrów
kedro run --pipeline=feature_selection # selekcja cech
kedro run --pipeline=autogluon         # AutoML

mlflow ui                              # przeglądanie eksperymentów

python scripts/run_drift_monitoring.py  # monitoring driftu i raport HTML/JSON

uvicorn crash_kedro.api.app:app --reload  # API (Swagger pod /docs)

streamlit run streamlit_app.py             # frontend do wprowadzania danych i predykcji

docker-compose up --build              # API w Dockerze

python scripts/demo.py                 # demo z predykcjami
```

### Aplikacja Streamlit

Frontend Streamlit korzysta z istniejącego endpointu `POST /predict` z API FastAPI.
Domyślnie szuka backendu pod `http://localhost:8000`, ale można zmienić to przez
zmienną środowiskową `PREDICTION_API_URL`.

```bash
# Windows PowerShell
$env:PREDICTION_API_URL = "http://localhost:8000"
streamlit run streamlit_app.py

# Linux / macOS
export PREDICTION_API_URL=http://localhost:8000
streamlit run streamlit_app.py
```

W aplikacji można uzupełnić dane wypadku, a następnie otrzymać:
- klasę wyniku (`NO_INJURY`, `MINOR`, `SERIOUS`),
- odpowiedź po polsku, czy doszło do obrażeń,
- prawdopodobieństwa poszczególnych klas.

**Pełne instrukcje:** [docs/SETUP.md](docs/SETUP.md)

> Uwaga: projekt testowaliśmy na Pythonie 3.13 z Kedro 1.2.0 - cały zestaw ML
> (`scikit-learn`, `xgboost`, `lightgbm`, `tpot`, `autogluon.tabular`) działa.
> Do samego backendu + Streamlit wystarczy `requirements-api.txt`.

API szuka modelu najpierw w `MODEL_PATH`, a następnie w katalogu `data/06_models/`
(np. `model.pkl`, `tuned_model.pkl`, `best_comparison_model.pkl`, `grid_random_model.pkl`).
Do przygotowania cech wykorzystuje również `data/06_models/encoders.pkl`.

## Jakość kodu

```bash
pytest -q                    # testy
ruff check src tests         # linting
pylint src/crash_kedro/api/app.py src/crash_kedro/utils/encoders.py src/crash_kedro/__main__.py
```

W praktyce najważniejszy jest wynik `pylint` na głównych modułach aplikacji — obecna konfiguracja
osiąga wynik powyżej wymaganego progu 8 pkt.

**Sprawdzony wynik:** pylint 10.00/10 ✅

##  Pełna dokumentacja

Zapraszamy do katalogu `docs/` dla szczegółowej dokumentacji:
- **[SETUP.md](docs/SETUP.md)** — Instrukcje instalacji i uruchomienia (3 opcje)
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Opis architektury i modularności (3 warstwy)
- **[API.md](docs/API.md)** — Pełna dokumentacja API FastAPI z przykładami
- **[DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)** — Słownik danych (172k wierszy, 43 kolumny)
- **[MODELS.md](docs/MODELS.md)** — Opis modeli i wyników (5 algorytmów testowanych)
- **[PRESENTATION.md](docs/PRESENTATION.md)** — Plan slajdów i demo na prezentację końcową
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** — Wytyczne dla deweloperów i workflow Git

## Struktura

```
conf/                  # konfiguracja Kedro (catalog, parameters)
data/                  # dane + modele + raporty (konwencja Kedro)
docs/                  #  PEŁNA DOKUMENTACJA (zobacz powyżej)
src/crash_kedro/
    pipelines/         # data_preparation, data_modeling, automl,
                       # tuning, autogluon, feature_selection
    api/               # FastAPI
    monitoring/        # drift detection
notebooks/             # 01_baseline (EDA + RF), 02_model_comparison
scripts/               # demo, walidacja modelu
.github/workflows/     # CI/CD
models/                # README - modele zapisywane w data/06_models/
tests/                 # Unit + integration testy
```

## ✅ Spełnione wymagania z `wymagania.txt`

### Łatwa przenoszalność między środowiskami
- ✅ Pełne repozytorium GitHub
- ✅ `requirements.txt` i `requirements-api.txt`
- ✅ `pyproject.toml` z definicją pakietu
- ✅ Dockerfile + docker-compose.yml

### Łatwe odtworzenie w różnych środowiskach
- ✅ **Opcja 1 (Docker):** `docker-compose up --build` — bez dodatkowej konfiguracji
- ✅ **Opcja 2 (Full project):** `setup.bat` (Windows) lub `setup.sh` (Linux)
- ✅ **Opcja 3 (API only):** `requirements-api.txt`

### Rozdzielenie logiki: data | model | app
- ✅ **Data:** `src/crash_kedro/pipelines/data_preparation/`
- ✅ **Model:** `src/crash_kedro/pipelines/data_modeling/` + tuning + AutoML
- ✅ **App:** `src/crash_kedro/api/` (FastAPI)

### Modularność na poziomie funkcji i katalogów
- ✅ Funkcje bez efektów ubocznych (pure functions)
- ✅ Pipeline'y Kedro (7 niezależnych pipeline'ów)
- ✅ Oddzielne moduły utils (`encoders.py`, `monitoring/`, itp.)
- ✅ Czysta struktura katalogów (konwencja Kedro)

### Dokumentacja
- ✅ **README.md** — opis, wyniki, uruchamianie ✓ (ten plik)
- ✅ **docs/SETUP.md** — instrukcje instalacji i troubleshooting
- ✅ **docs/ARCHITECTURE.md** — opis architektury i modulami
- ✅ **docs/API.md** — endpointy, schemat JSON, przykłady
- ✅ **docs/DATA_DICTIONARY.md** — słownik 43 kolumn, mapowania
- ✅ **docs/MODELS.md** — opis 5 algorytmów, wyniki
- ✅ **docs/CONTRIBUTING.md** — wytyczne dla dev, workflow Git
- ✅ **Docstrings** — NumPy style na kluczowych funkcjach

### Czysty kod spełniający PEP8
- ✅ **Pylint:** 10.00/10 (sprawdzono 3 główne moduły)
- ✅ **Ruff:** Wszystkie checky pass
- ✅ **Type hints:** Używane na kluczowych funkcjach
- ✅ **Nazewnictwo:** Zmienne i funkcje snake_case, klasy PascalCase
- ✅ **Brak redundancji:** Kod DRY, funkcje reużywalne

### Aplikacja jako API z minimum 8 pkt Pylint
- ✅ **FastAPI** — `/predict`, `/health`, `/predictions/recent`
- ✅ **Pylint score:** **10.00/10** ✓
- ✅ **Pydantic models** — CrashInput, PredictionOutput
- ✅ **Komunikacja JSON** — Swagger UI dostępny pod `/docs`

### Plik README z wymaganiami i uruchomieniem
- ✅ Opis problemu (klasyfikacja wypadków, 3 klasy, 172k wierszy)
- ✅ Wyniki modeli (tabelka z 5 algorytmami)
- ✅ Instrukcje uruchomienia (3 opcje)
- ✅ Instrukcje konfiguracji (requirements, dockerfile)
- ✅ Strukturę projektu wyjaśnioną

### Uruchamianie bez konfiguracji OS
- ✅ **Docker:** Całkowicie izolowane, nie wymaga setup OS
- ✅ **setup.bat/setup.sh:** Automacja dla native development
- ✅ `requirements.txt` — pip install zamiast ręcznej konfiguracji
- ✅ `scripts/run_drift_monitoring.py` — monitoring driftu danych i raport HTML/JSON

### CI/CD
- ✅ **`.github/workflows/ci.yml`** — testy na każdy push (pytest, ruff, pylint)
- ✅ **`.github/workflows/cd.yml`** — build Docker na tag release + opcjonalny deploy na zdalny host
- ✅ **`.github/workflows/ct.yml`** — continuous training + monitoring driftu + artefakty

### Format przekazania
- ✅ Repozytorium GitHub
- ✅ Pełna dokumentacja w README + docs/
- ✅ Plik z wymaganiami (`requirements.txt`, `requirements-api.txt`)
- ✅ Plik konfiguracyjny (`conf/base/catalog.yml`, `conf/base/parameters.yml`)
- ✅ Dockerfile + docker-compose.yml
- ✅ Wszystkie pliki do odtworzenia aplikacji

## Autorzy

- Artur Cichocki
- Bartosz Pikutin
- Wiktor Golba

PJATK, ASI 2026.

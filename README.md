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
- Dokumentacja w katalogu `docs/`

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

Dodatkowo aplikacja zawiera szybkie scenariusze testowe, które automatycznie
wypełniają formularz przykładowymi danymi, np. typowy bezpieczny wypadek,
noc z mokrą nawierzchnią albo potencjalnie poważne zdarzenie.

W panelu bocznym dostępna jest również historia ostatnich predykcji pobierana
z endpointu `GET /predictions/recent`, co ułatwia szybkie porównywanie
wykonanych zapytań.

**Pełne instrukcje:** [docs/SETUP.md](docs/SETUP.md)

> Uwaga: CI i Docker używają Pythona 3.12. Cały zestaw ML
> (`scikit-learn`, `xgboost`, `lightgbm`, `autogluon.tabular`) instaluje się z `requirements.txt`.
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

Lintujemy `ruff` i `pylint` (głównie moduły API). Staramy się trzymać próg 8 pkt z pylinta.

## Dokumentacja

Więcej szczegółów w katalogu `docs/`:
- [SETUP.md](docs/SETUP.md) — instalacja i uruchomienie
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — architektura i podział na warstwy
- [API.md](docs/API.md) — dokumentacja API
- [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) — słownik danych
- [MODELS.md](docs/MODELS.md) — opis modeli i wyników
- [PRESENTATION.md](docs/PRESENTATION.md) — plan prezentacji
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) — jak dodać nowy pipeline

## Struktura

```
conf/                  # konfiguracja Kedro (catalog, parameters)
data/                  # dane + modele + raporty (konwencja Kedro)
docs/                  # dokumentacja (zobacz wyżej)
src/crash_kedro/
    pipelines/         # data_preparation, data_modeling, automl,
                       # tuning, autogluon, feature_selection
    api/               # FastAPI
    ui/                # frontend Streamlit
    monitoring/        # drift detection
notebooks/             # 01_baseline (EDA + RF), 02_model_comparison
scripts/               # demo, monitoring driftu, walidacja modelu
.github/workflows/     # CI/CD/CT
models/                # README - modele zapisywane w data/06_models/
tests/                 # testy
```

## Jak projekt realizuje wymagania

- **Baseline (notebook)** — `notebooks/01_baseline_model.ipynb`: EDA, preprocessing, trening RF, ewaluacja.
- **Pipeline ML (Kedro)** — `data_preparation` + `data_modeling`, dodatkowo `tuning`, `automl`, `autogluon`, `feature_selection`.
- **Śledzenie eksperymentów** — MLflow (logowanie w węzłach treningu, tuningu, automl, autogluon, feature_selection).
- **AutoML** — pipeline `autogluon` (AutoGluon) + własne porównanie modeli w `automl`.
- **Inżynieria cech** — cechy czasowe i binarne w `data_preparation`, selekcja cech (SelectKBest) w `feature_selection`.
- **Strojenie hiperparametrów** — Grid Search, Random Search i Optuna w pipeline `tuning`.
- **Produkcja** — API FastAPI (`/predict`, `/health`, `/predictions/recent`) + frontend Streamlit, uruchamiane lokalnie lub w Dockerze.
- **Monitoring** — logowanie predykcji do `logs/predictions.jsonl` + wykrywanie driftu (Evidently) w `scripts/run_drift_monitoring.py`.
- **MLOps (CI/CD/CT)** — GitHub Actions: testy + linting (`ci.yml`), build/deploy Dockera (`cd.yml`), ponowny trening na harmonogramie (`ct.yml`).
- **Dokumentacja** — README + katalog `docs/` + diagram architektury (`docs/architecture.drawio`).

## Autorzy

- Artur Cichocki
- Bartosz Pikutin
- Wiktor Golba

PJATK, ASI 2026.

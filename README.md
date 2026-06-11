# ASI - Analiza wypadków drogowych

Projekt zaliczeniowy z ASI. Przewidywanie stopnia obrażeń w wypadkach drogowych na podstawie danych z Montgomery County, Maryland (2015-2024).

## Materiały

- Baseline notebook: [`notebooks/01_baseline_model.ipynb`](notebooks/01_baseline_model.ipynb)
- Notebook porównawczy modeli: [`notebooks/02_model_comparison.ipynb`](notebooks/02_model_comparison.ipynb)
- Diagram architektury: [`docs/architecture.drawio`](docs/architecture.drawio) (podgląd: `docs/architecture.png`)
- Prezentacja: `docs/prezentacja.pptx`

## Problem

Klasyfikacja - na podstawie informacji o wypadku (pogoda, prędkość, typ kolizji itp.) chcemy przewidzieć czy doszło do obrażeń i jak poważnych. 3 klasy:

- **NO_INJURY** (~82%) - brak obrażeń
- **MINOR** (~17%) - drobne obrażenia
- **SERIOUS** (~1%) - poważne obrażenia / zgon

Dataset ma 172 tys. wierszy i 43 kolumny.

## Dane

Surowe dane to `data/01_raw/crash_data.csv` (Montgomery County Open Data). Najważniejsze kolumny:
pogoda (Weather), oświetlenie (Light), typ kolizji (Collision Type), stan nawierzchni,
kontrola ruchu, dane pojazdu (typ, rok, uszkodzenia) oraz target `Injury Severity`.

W preprocessing:
- wyrzucamy kolumny nieprzydatne / z dużą liczbą braków (Report Number, Location itp.),
- braki uzupełniamy modą (tekst) i medianą (liczby),
- z daty robimy cechy: godzina, dzień tygodnia, miesiąc, rok, plus `is_night`, `is_bad_weather`,
  `is_wet_surface`, `vehicle_age`,
- target z 5 wartości mapujemy na 3 klasy (NO_INJURY / MINOR / SERIOUS),
- kolumny tekstowe kodujemy liczbami (nieznane wartości dostają -1).

## Wyniki

| Model | Accuracy | F1 ważone | F1 makro |
|-------|----------|-----------|----------|
| **LightGBM (Optuna, 50 prób)** | 0.79 | **0.78** | **0.47** |
| Autogluon (best: LightGBM) | 0.83 | 0.77 | 0.40 |
| XGBoost | 0.83 | 0.77 | 0.40 |
| Gradient Boosting | 0.82 | 0.77 | 0.40 |
| Random Forest (baseline) | 0.82 | 0.75 | 0.33 |

Dane są mocno niezbalansowane (klasa SERIOUS to ~1%), więc patrzymy głównie na F1 makro -
samo accuracy zawyża wynik, bo model lubi strzelać NO_INJURY.

## Architektura

Projekt jest podzielony na trzy części: dane, modelowanie i aplikacja. Spina to Kedro
(pipeline'y) oraz FastAPI (serwowanie modelu). Przepływ:

```
CSV (data/01_raw)
  -> Kedro data_preparation  -> crash_features.parquet (data/03_primary)
  -> data_modeling / tuning / automl / autogluon -> model.pkl (data/06_models)
  -> FastAPI /predict -> predictions.jsonl (logs/)
crash_features -> monitoring driftu (Evidently) -> raport HTML (data/08_reporting)
```

Pipeline'y Kedro (`src/crash_kedro/pipelines/`):
- `data_preparation` - czyszczenie, inżynieria cech, enkodowanie
- `data_modeling` - trening baseline (Random Forest) + ewaluacja
- `hyperparameter_tuning` - Grid Search, Random Search, Optuna
- `automl` - porównanie kilku modeli sklearn
- `autogluon` - AutoML (AutoGluon)
- `feature_selection` - SelectKBest

Wyniki wszystkich pipeline'ów trafiają do MLflow (`mlflow ui`).

## Uruchomienie

### Najszybciej: API w Dockerze

```bash
docker-compose up --build
# API pod http://localhost:8000/docs (Swagger)
```

### Pełny projekt

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
source .venv/bin/activate           # Linux/macOS

pip install -r requirements.txt
pip install -e ".[dev]"

# albo skrypt:
bash setup.sh                       # Linux/macOS
setup.bat                           # Windows
```

Pipeline ML:
```bash
kedro run                              # caly pipeline (data + model)
kedro run --pipeline=data_processing   # tylko przygotowanie danych
kedro run --pipeline=modeling          # tylko trening bazowy
kedro run --pipeline=tuning            # Grid / Random / Optuna
kedro run --pipeline=feature_selection # selekcja cech
kedro run --pipeline=autogluon         # AutoGluon
kedro run --pipeline=automl            # porownanie modeli
```

API, MLflow, frontend:
```bash
uvicorn crash_kedro.api.app:app --reload   # API (Swagger pod /docs)
mlflow ui                                   # eksperymenty (http://localhost:5000)
streamlit run streamlit_app.py              # frontend do predykcji
python scripts/demo.py                      # demo na zbiorze testowym
python scripts/run_drift_monitoring.py      # monitoring driftu
```

Frontend Streamlit łączy się z API (domyślnie `http://localhost:8000`, można zmienić
zmienną `PREDICTION_API_URL`). Najpierw trzeba uruchomić backend FastAPI.

W panelu bocznym aplikacji są szybkie scenariusze testowe (np. "noc i mokra nawierzchnia"),
które jednym kliknięciem wypełniają formularz, oraz historia ostatnich predykcji
pobierana z endpointu `GET /predictions/recent`.

> Uwaga: CI i Docker używają Pythona 3.12. Pełny zestaw ML (`scikit-learn`, `xgboost`,
> `lightgbm`, `autogluon.tabular`) instaluje się z `requirements.txt`. Do samego API +
> Streamlit wystarczy `requirements-api.txt`.

API szuka modelu w `MODEL_PATH`, a potem w `data/06_models/` (`model.pkl`, `tuned_model.pkl`,
`best_comparison_model.pkl`). Do kodowania cech używa `data/06_models/encoders.pkl`.

Endpointy API: `POST /predict`, `GET /health`, `GET /predictions/recent`.

## Testy i linting

```bash
pytest -q
ruff check src tests
pylint src/crash_kedro/api/app.py src/crash_kedro/utils/encoders.py
```

## CI/CD

GitHub Actions (`.github/workflows/`):
- `ci.yml` - testy + linting na każdy push
- `cd.yml` - build i push obrazu Dockera na tag `v*`
- `ct.yml` - ponowny trening + monitoring driftu na harmonogramie

## Struktura

```
conf/                  # konfiguracja Kedro (catalog, parameters)
data/                  # dane + modele + raporty (konwencja Kedro)
docs/                  # diagram architektury, prezentacja
src/crash_kedro/
    pipelines/         # data_preparation, data_modeling, automl,
                       # tuning, autogluon, feature_selection
    api/               # FastAPI
    ui/                # frontend Streamlit
    monitoring/        # drift detection
notebooks/             # 01_baseline (EDA + RF), 02_model_comparison
scripts/               # demo, monitoring driftu, walidacja modelu
tests/
.github/workflows/     # CI/CD/CT
```

## Autorzy

- Artur Cichocki
- Bartosz Pikutin
- Wiktor Golba

PJATK, ASI 2026.

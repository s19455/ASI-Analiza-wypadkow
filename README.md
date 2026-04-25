# ASI - Analiza wypadków drogowych

Projekt zaliczeniowy z ASI. Przewidywanie stopnia obrażeń w wypadkach drogowych na podstawie danych z Montgomery County, Maryland (2015-2024).

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
- API FastAPI + Docker
- CI/CD na GitHub Actions
- Monitoring driftu (Evidently)

## Uruchomienie

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

kedro run                              # cały pipeline
kedro run --pipeline=tuning            # tuning hiperparametrów
kedro run --pipeline=feature_selection # selekcja cech
kedro run --pipeline=autogluon         # AutoML

mlflow ui                              # przeglądanie eksperymentów

uvicorn src.crash_kedro.api.app:app    # API (Swagger pod /docs)

docker-compose up --build              # API w Dockerze

python scripts/demo.py                 # demo z predykcjami
```

## Struktura

```
conf/                  # konfiguracja Kedro (catalog, parameters)
data/                  # dane + modele + raporty (konwencja Kedro)
src/crash_kedro/
    pipelines/         # data_preparation, data_modeling, automl,
                       # tuning, autogluon, feature_selection
    api/               # FastAPI
    monitoring/        # drift detection
notebooks/             # 01_baseline (EDA + RF), 02_model_comparison
docs/                  # diagram, słownik danych, prezentacja
scripts/               # demo, walidacja modelu
.github/workflows/     # CI/CD
models/                # README - modele zapisywane w data/06_models/
```

## Autorzy

- Artur Cichocki
- Bartosz Pikutin
- Wiktor Golba

PJATK, ASI 2026.

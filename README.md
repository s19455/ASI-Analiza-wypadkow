# ASI - Analiza wypadków drogowych

Projekt zaliczeniowy z ASI. Przewidywanie stopnia obrażeń w wypadkach drogowych (dane z Montgomery County, Maryland, 2015-2024).

## O czym jest projekt

Na podstawie informacji o wypadku (pogoda, oświetlenie, prędkość, typ kolizji itd.) model przewiduje czy doszło do obrażeń i jak poważnych. Używamy 3 klas:
- **NO_INJURY** (82%) - brak obrażeń
- **MINOR** (17%) - drobne obrażenia
- **SERIOUS** (1%) - poważne obrażenia / zgon

Dataset ma 172 tys. rekordów i 43 kolumny.

## Co zrobiliśmy

1. **EDA + baseline** - eksploracja danych w notebooku, Random Forest jako model bazowy
2. **Pipeline Kedro** - przetwarzanie danych i trening modelu jako pipeline (kedro run)
3. **Selekcja cech** - SelectKBest z mutual_info_classif
4. **Porównanie modeli** - RF, Gradient Boosting, XGBoost, LightGBM
5. **Strojenie hiperparametrów** - Grid Search, Random Search i Optuna (Bayesian)
6. **AutoML** - Autogluon (auto-stacking, auto-ensembling)
7. **MLflow** - śledzenie eksperymentów (parametry, metryki, modele)
8. **API** - FastAPI z `/predict` (real-time) i `/predict_batch` (batch)
9. **Docker** - konteneryzacja API
10. **CI/CD** - GitHub Actions (lint, testy, budowanie dockera, continuous training)
11. **Monitoring** - logowanie predykcji, latency stats, wykrywanie driftu (Evidently)

## Wyniki

| Model | Dokładność | F1 ważone | F1 makro |
|-------|-----------|-----------|----------|
| **Optuna LightGBM** | 0.78 | **0.78** | **0.46** |
| XGBoost | 0.83 | 0.77 | 0.40 |
| Gradient Boosting | 0.82 | 0.77 | 0.40 |
| Random Forest | 0.82 | 0.75 | 0.33 |

Najlepszy model to LightGBM po tuningu Optuną. F1 makro wzrosło z 0.33 do 0.46 w porównaniu z baseline.

## Jak uruchomić

```bash
# instalacja
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# pipeline
kedro run                                  # cały pipeline (preprocessing + trening)
kedro run --pipeline=feature_selection     # selekcja cech (SelectKBest)
kedro run --pipeline=tuning                # Grid + Random + Bayesian Search
kedro run --pipeline=autogluon             # AutoML z Autogluonem

# mlflow ui
mlflow ui
# http://localhost:5000

# api
uvicorn src.crash_kedro.api.app:app --reload
# http://localhost:8000/docs

# docker
docker-compose up --build

# demo
python scripts/demo.py
```

## Struktura

```
├── conf/                    # konfiguracja kedro
├── data/01_raw/             # surowe dane (crash_data.csv)
├── src/crash_kedro/
│   ├── pipelines/           # data_preparation, data_modeling, automl, tuning
│   ├── api/                 # FastAPI
│   └── monitoring/          # wykrywanie driftu
├── notebooks/               # EDA, porównanie modeli
├── docs/                    # dokumentacja, diagram
├── .github/workflows/       # CI/CD
├── Dockerfile
└── scripts/                 # demo, walidacja modelu
```

## Optymalizacja produkcji

Wybralismy **architekturę real-time** z opcjonalnym batch endpointem:

- `/predict` — pojedyncza predykcja (real-time, ~10-50ms latency)
- `/predict_batch` — wiele predykcji w jednym requeście (efektywniejsze przy duzych wolumenach)
- `/predictions/stats` — statystyki latency: avg, p50, p95, p99, max

Latency mierzymy przy każdej predykcji i logujemy do `logs/predictions.jsonl`.
Model w pamieci (cache w `_model_cache`) - wczytywany tylko raz przy starcie.

Mozliwe przyszle optymalizacje: ONNX export, kwantyzacja, async batching.

## Technologie

Kedro, scikit-learn, XGBoost, LightGBM, Optuna, Autogluon, MLflow, FastAPI, Docker, GitHub Actions, Evidently

## Autorzy

- Artur Cichocki
- Bartosz Pikutin
- Wiktor Golba

Projekt na zaliczenie z ASI, PJATK.

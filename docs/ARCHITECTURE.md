# Architektura projektu

## Przegląd

Projekt składa się z trzech niezależnych, ale zintegrowanych warstw:

### Wizualizacja architektury

Diagram można edytować w `docs/architecture.drawio`, a poniższy Mermaid pokazuje główny przepływ danych:

```mermaid
flowchart LR
    A[CSV raw data\n data/01_raw] --> B[Kedro data_preparation]
    B --> C[crash_features.parquet\n data/03_primary]
    C --> D[data_modeling / tuning / AutoML]
    D --> E[Wytrenowany model\n data/06_models]
    E --> F[FastAPI /predict]
    F --> G[predictions.jsonl\n logs/]
    C --> H[Monitoring driftu\n scripts/run_drift_monitoring.py]
    H --> I[drift_report.html\n data/08_reporting]
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    API (FastAPI)                                 │
│                  crash_kedro.api.app                             │
│              [/predict, /health, /predictions]                  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
        ┌───────────▼────────────┐  ┌────────▼──────────────┐
        │ Modele (joblib/pickle) │  │  Utils (encoders)      │
        │ data/06_models/        │  │ crash_kedro.utils      │
        └───────────┬────────────┘  └────────┬──────────────┘
                    │                        │
        ┌───────────▼────────────────────────▼────────────┐
        │        Pipeline ML (Kedro)                      │
        │                                                  │
        │  1. Data Preparation   (data_preparation)       │
        │     - czyszczenie danych                        │
        │     - inżynieria cech                           │
        │     - enkodowanie kategorii                     │
        │                                                  │
        │  2. Modeling           (data_modeling)          │
        │     - trening (Random Forest baseline)          │
        │     - ewaluacja metryk                          │
        │                                                  │
        │  3. Tuning             (hyperparameter_tuning)  │
        │     - Grid Search                               │
        │     - Random Search                             │
        │     - Optuna                                    │
        │                                                  │
        │  4. AutoML             (automl, autogluon)      │
        │     - TPOT / AutoGluon                          │
        │                                                  │
        │  5. Feature Selection  (feature_selection)      │
        │     - SelectKBest                               │
        │                                                  │
        │  6. Monitoring         (monitoring.drift_detector)
        │     - Evidently                                 │
        └───────────┬────────────────────────────────────┘
                    │
        ┌───────────▼────────────┐
        │  Data (konwencja Kedro)│
        │  data/                 │
        │  ├── 01_raw/           │
        │  ├── 02_intermediate/  │
        │  ├── 03_primary/       │
        │  ├── 04_feature/       │
        │  ├── 06_models/        │
        │  └── 08_reporting/     │
        └────────────────────────┘
```

---

## Warstwy aplikacji

### 1. Data Layer (Warstwa danych)

**Odpowiedzialność:** przygotowanie, czyszczenie, transformacja danych

**Katalog:** `src/crash_kedro/pipelines/data_preparation/`

**Komponenty:**
- `nodes.py` — funkcje przetwarzające: drop_unnecessary_columns, clean_missing_values, engineer_features, map_target, encode_features
- `pipeline.py` — definicja Kedro pipeline

**Proces:**
1. Wczytanie surowych danych CSV (`data/01_raw/`)
2. Usunięcie niepotrzebnych kolumn
3. Imputacja braków
4. Inżynieria cech (daty, warunki pogody, wiek pojazdu)
5. Zmapowanie klasy docelowej (NO_INJURY, MINOR, SERIOUS)
6. Enkodowanie zmiennych kategorycznych
7. Zapis przetworzonych danych (`data/03_primary/crash_features.parquet`)

---

### 2. Model Layer (Warstwa modelowania)

**Odpowiedzialność:** trening, tuning, ewaluacja modeli

**Katalogi:**
- `src/crash_kedro/pipelines/data_modeling/` — trening baseline
- `src/crash_kedro/pipelines/hyperparameter_tuning/` — tuning
- `src/crash_kedro/pipelines/automl/` — AutoML (TPOT)
- `src/crash_kedro/pipelines/autogluon/` — AutoML (AutoGluon)
- `src/crash_kedro/pipelines/feature_selection/` — selekcja cech

**Algorytmy:**
- **Baseline:** Random Forest
- **Boosting:** LightGBM, XGBoost, Gradient Boosting
- **AutoML:** AutoGluon (automatyczne wybieranie i tunning)
- **Hyperparameter Tuning:** Grid Search, Random Search, Optuna

**Wyjścia:**
- Wytrenowane modele (`data/06_models/*.pkl`)
- Metryki (`data/08_reporting/*.json`)
- Encodery kategoryczne (`data/06_models/encoders.pkl`)

---

### 3. Application Layer (Warstwa aplikacji)

**Odpowiedzialność:** serwowanie modelu, obsługa żądań, logowanie predykcji

**Katalog:** `src/crash_kedro/api/`

**Komponenty:**
- `app.py` — aplikacja FastAPI z endpointami:
  - `POST /predict` — predykcja dla jednego rekordu
  - `GET /health` — status aplikacji i załadowanego modelu
  - `GET /predictions/recent` — ostatnie predykcje z logów

**Funkcje:**
- Ładowanie modelu z wielu miejsc (zmienna `MODEL_PATH`, katalog `data/06_models/`)
- Transformacja wejścia przy użyciu enkoderów
- Predykcja
- Logowanie do JSONL

---

## Modularność i rozdzielenie odpowiedzialności

### Poziom pipeline'ów (Kedro)
Każdy pipeline zajmuje się jednym aspektem:
```python
# src/crash_kedro/pipeline_registry.py
pipelines = {
    "__default__": data_preparation + data_modeling,
    "data_processing": data_preparation,
    "modeling": data_modeling,
    "tuning": data_preparation + tuning,
    "automl": data_preparation + automl,
    ...
}
```

Dzięki temu można:
- Uruchamiać tylko przygotowanie danych
- Trenować model bez przetwarza danych (cache)
- Testować różne algorytmy niezależnie

### Poziom funkcji (Nodes)
Każdy `node` w pipeline'u to czysta funkcja:
```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Funkcja bez efektów ubocznych."""
    # transformacje...
    return df_modified
```

To umożliwia:
- Testowanie (unit tests)
- Debugowanie
- Ponowne użycie w innym kontekście

### Poziom modułów (Utils)
Narzędzia są wydzielone:
```
src/crash_kedro/utils/
├── encoders.py      # Enkodowanie kategoryczne
├── validators.py    # Walidacja (jeśli będzie)
└── metrics.py       # Metryki (jeśli będzie)
```

---

## Przepływ danych

```
[CSV Data] 
    ↓
[data_preparation pipeline]
    ├── drop_unnecessary_columns
    ├── clean_missing_values
    ├── engineer_features
    ├── map_target
    └── encode_features
    ↓
[crash_features.parquet] 
    ↓
[data_modeling pipeline OR tuning OR automl]
    ├── split_data
    ├── train_model
    └── evaluate_model
    ↓
[model.pkl, tuned_model.pkl, automl_model.pkl]
    ↓
[API] 
    ├── GET /health → sprawdza model, encoders
    └── POST /predict → encode input → predict → zwróć JSON
    ↓
[predictions.jsonl] ← log wszystkich predykcji
```

---

## Konfiguracja (Kedro)

Konfiguracja jest wydzielona w `conf/base/`:

**`conf/base/parameters.yml`**
```yaml
test_size: 0.2
random_state: 42
target_column: "Injury Severity"
severity_mapping:
  NO_INJURY: ["NO APPARENT INJURY"]
  MINOR: ["POSSIBLE INJURY", "SUSPECTED MINOR INJURY"]
  SERIOUS: ["SUSPECTED SERIOUS INJURY", "FATAL INJURY"]
columns_to_drop:
  - "Report Number"
  - "Local Case Number"
  ...
```

**`conf/base/catalog.yml`**
```yaml
crash_raw:
  type: pandas.CSVDataset
  filepath: data/01_raw/crash_data.csv

crash_features:
  type: pandas.ParquetDataset
  filepath: data/03_primary/crash_features.parquet

model:
  type: pickle.PickleDataset
  filepath: data/06_models/model.pkl
...
```

To umożliwia:
- Zmianę ścieżek bez zmian kodu
- Zmianę parametrów treningu bez zmian kodu
- Łatwe eksportowanie konfiguracji

---

## Testowanie

```
tests/
├── conftest.py          # Fixtury pytest
├── test_run.py          # Testy pipeline'ów
├── api/
│   └── test_app.py      # Testy API (endpoints, encoding)
├── pipelines/
│   ├── data_preparation/
│   │   └── test_nodes.py
│   └── data_modeling/
│       └── (test_nodes.py)
└── utils/
    └── test_encoders.py # Testy enkoderów
```

Każdy komponent jest testowany:
- Unit testy dla funkcji
- Integracyjne testy pipeline'ów
- API testy dla endpointów

---

## CI/CD

Repozytorium ma GitHub Actions workflows:

**`.github/workflows/ci.yml`** — testy na każdy push
**`.github/workflows/cd.yml`** — budowanie Dockera na tag
**`.github/workflows/ct.yml`** — continuous training (cotygodniowo)

---

## Monitorowanie

**`src/crash_kedro/monitoring/drift_detector.py`**

**`scripts/run_drift_monitoring.py`**

Detect data/model drift za pomocą Evidently
- Monitorowanie zmian w rozkładach danych
- Alerting przy spadku jakości
- Generowanie raportu HTML i podsumowania JSON

---

## Zależności między warstwami

```
┌─────────────────┐
│  API            │ ← wczytuje model, encoders
├─────────────────┤
├─ Depends on:   │
│  - data/06_models/model.pkl
│  - data/06_models/encoders.pkl
│  - crash_kedro.utils.encoders
└─────────────────┘
        ↑
┌─────────────────┐
│  Model Layer    │ ← trenuje, tunuje
├─────────────────┤
├─ Produces:     │
│  - model.pkl
│  - encoders.pkl
└─────────────────┘
        ↑
┌─────────────────┐
│  Data Layer     │ ← przygotowuje
├─────────────────┤
├─ Produces:     │
│  - crash_features.parquet
│  - encoders.pkl
└─────────────────┘
        ↑
┌─────────────────┐
│  Raw Data       │ (CSV)
└─────────────────┘
```

---

## Rozszerzalność

Projekt jest zaprojektowany tak, aby łatwo dodawać nowe komponenty:

### Dodanie nowego pipeline'u
```python
# src/crash_kedro/pipelines/my_new_feature/pipeline.py
from kedro.pipeline import Pipeline, node

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(func=my_func, inputs="x", outputs="y", name="my_node")
    ])

# src/crash_kedro/pipeline_registry.py — dodaj do register_pipelines()
pipelines["my_feature"] = dp() + create_pipeline()
```

### Dodanie nowego endpointu API
```python
# src/crash_kedro/api/app.py
@app.post("/my_endpoint")
def my_endpoint(data: MyInput) -> MyOutput:
    ...
```

### Dodanie nowego modelu
Zmień `nodes.py` w `data_modeling`:
```python
from sklearn.ensemble import GradientBoostingClassifier

def train_model(X_train, y_train):
    model = GradientBoostingClassifier(...)
    model.fit(X_train, y_train)
    return model
```

---

## Konwencje nazewnictwa

| Element | Konwencja | Przykład |
|---------|-----------|---------|
| Pipeline | `snake_case` | `data_modeling`, `hyperparameter_tuning` |
| Funkcje (nodes) | `snake_case` | `engineer_features`, `train_model` |
| Zmienne w Kedro | `snake_case` | `crash_features`, `X_train` |
| Klasy | `PascalCase` | `CrashInput`, `PredictionOutput` |
| Konstanki | `UPPER_CASE` | `UNKNOWN_CATEGORY_CODE`, `MODEL_CANDIDATES` |

---

## Podsumowanie

Projekt przestrzega:
- **Separation of Concerns** — dane, model, aplikacja rozdzielone
- **DRY principle** — krótki kod bez redundancji
- **Testability** — komponenty są testowalne
- **Modularity** — pipeline'y, funkcje, moduły niezależne
- **Maintainability** — jasne nazwy, dokumentacja, typowanie


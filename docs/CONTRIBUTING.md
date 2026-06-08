# Wytyczne dla deweloperów

Krótki przewodnik, jak rozwijać projekt.

## Środowisko

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Dodanie nowego pipeline'u

Struktura:
```
src/crash_kedro/pipelines/my_pipeline/
├── __init__.py
├── nodes.py      # funkcje przetwarzające
└── pipeline.py   # definicja pipeline'u
```

`pipeline.py`:
```python
from kedro.pipeline import Pipeline, node
from .nodes import my_node

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(func=my_node, inputs="crash_features", outputs="my_output", name="my_node"),
    ])
```

Rejestracja w `src/crash_kedro/pipeline_registry.py`:
```python
try:
    from crash_kedro.pipelines.my_pipeline import create_pipeline as my_pipeline
    pipelines["my_feature"] = dp() + my_pipeline()
except ImportError:
    logger.warning("Pipeline 'my_feature' pominiety - brakuje zaleznosci.")
```

## Konwencje

| Element | Konwencja | Przykład |
|---------|-----------|----------|
| zmienne, funkcje, pipeline'y | `snake_case` | `engineer_features`, `data_processing` |
| klasy | `PascalCase` | `CrashInput` |
| stałe | `UPPER_CASE` | `UNKNOWN_CATEGORY_CODE` |

- Type hints na funkcjach.
- Logging zamiast `print()` (`logger = logging.getLogger(__name__)`).
- Długość linii: 88 znaków (`ruff`).

## Testy i lint przed commitem

```bash
pytest -q
ruff check src tests
ruff format src tests           # auto-format
pylint src/crash_kedro/api/app.py src/crash_kedro/utils/encoders.py
```

## Praca z gitem

```bash
git checkout -b feat/nazwa
git add .
git commit -m "opis zmiany"
git push origin feat/nazwa
```

Przed wypchnięciem upewnij się, że `pytest` i `ruff check` przechodzą.

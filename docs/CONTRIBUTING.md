# Wytyczne dla deweloperów (Contributing)

## Przygotowanie środowiska

### 1. Fork i clone repository
```bash
git clone https://github.com/YOUR_USERNAME/ASI-Analiza-wypadkow.git
cd ASI-Analiza-wypadkow
```

### 2. Wirtualne środowisko
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Instalacja dev dependenciesypi
```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## Struktura projektu — gdzie co dodawać

### Dodanie nowego pipeline'u

```
src/crash_kedro/pipelines/my_pipeline/
├── __init__.py
├── nodes.py      # ← tutaj funkcje
├── pipeline.py   # ← tutaj definicja pipeline'u
└── README.md     # ← opcjonalnie
```

**Przykład:**

```python
# nodes.py
def my_node(input_data: pd.DataFrame) -> pd.DataFrame:
    """Proces jednej liczby czynności.
    
    Parameters
    ----------
    input_data : pd.DataFrame
        Dane wejściowe.
    
    Returns
    -------
    pd.DataFrame
        Dane transformowane.
    """
    # Transformacje...
    return result

# pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import my_node

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=my_node,
            inputs="input_data",
            outputs="output_data",
            name="my_node_name"
        )
    ])
```

Puis dodaj do `src/crash_kedro/pipeline_registry.py`:
```python
try:
    from crash_kedro.pipelines.my_pipeline import create_pipeline as my_pipeline
    pipelines["my_feature"] = dp() + my_pipeline()
except ImportError:
    pass  # Pipeline jest opcjonalny
```

### Dodanie nowego endpointu API

```python
# src/crash_kedro/api/app.py

@app.post("/my_endpoint")
def my_endpoint_func(data: MyInputModel) -> MyOutputModel:
    """Opisz, co robi endpoint.
    
    Args:
        data: Wejście (zwalidowane automatycznie)
    
    Returns:
        Odpowiedź w JSON
    """
    result = process(data)
    return MyOutputModel(**result)
```

### Dodanie testu

```python
# tests/pipelines/my_pipeline/test_nodes.py
import pandas as pd
from crash_kedro.pipelines.my_pipeline.nodes import my_node

def test_my_node():
    """Test funkcji my_node."""
    input_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    
    result = my_node(input_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
```

---

## Konwencje kodowania

### Nazewnictwo

| Element | Konwencja | Przykład |
|---------|-----------|---------|
| Zmienne | `snake_case` | `df_cleaned`, `n_estimators` |
| Funkcje | `snake_case` | `engineer_features()` |
| Klasy | `PascalCase` | `CrashInput`, `RandomForest` |
| Stałe | `UPPER_CASE` | `UNKNOWN_CATEGORY_CODE`, `DEFAULT_SEED` |
| Pliki | `snake_case.py` | `data_preparation.py` |
| Pipeline'y | `snake_case` | `data_processing`, `hyperparameter_tuning` |

### Dokumentacja (Docstrings)

Używaj **NumPy style** docstrings:

```python
def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Oblicz metryki jakości predykcji.
    
    Parameters
    ----------
    predictions : np.ndarray
        Przewidywane wartości (shape: n_samples,)
    targets : np.ndarray
        Wartości rzeczywiste (shape: n_samples,)
    
    Returns
    -------
    dict
        Słownik zawierający {accuracy, precision, recall, f1}
    
    Raises
    ------
    ValueError
        Jeśli shape predictions != targets
    
    Examples
    --------
    >>> y_pred = np.array([0, 1, 1, 0])
    >>> y_true = np.array([0, 1, 0, 0])
    >>> metrics = calculate_metrics(y_pred, y_true)
    >>> metrics['f1']
    0.666...
    """
    if predictions.shape != targets.shape:
        raise ValueError("Shape mismatch")
    
    # Implementacja...
    return metrics
```

### Type hints

Zawsze używaj typów:

```python
# ❌ Źle
def process(data):
    return data.fillna(0)

# ✅ Dobrze
def process(data: pd.DataFrame) -> pd.DataFrame:
    return data.fillna(0)
```

### Długość linii

PEP8: **max 88 znaków** (configure w `pyproject.toml`)

```bash
ruff format src tests  # Auto-format
```

---

## Commit messages

Format:
```
[TYPE] DESCRIPTION

BODY (opcjonalnie)
```

**TYPES:**
- `feat:` — nowa funkcjonalność
- `fix:` — naprawienie błędu
- `docs:` — dokumentacja
- `test:` — testy
- `refactor:` — zmiana kodu bez zmian funkcjonalności
- `chore:` — konfiguracja, dependencjes
- `ci:` — CI/CD

**Przykłady:**
```
feat: add Optuna hyperparameter tuning pipeline

Implement Optuna-based hyperparameter optimization for LightGBM
with F1 macro as optimization metric.

fix: correct UNKNOWN_CATEGORY_CODE in transform_with_encoders

docs: add API.md documentation

test: add test_build_prediction_frame

ci: add Github Actions workflows
```

---

## Pull Requesty

1. **Utwórz branch** z sensowną nazwą:
   ```bash
   git checkout -b feat/my-feature
   git checkout -b fix/bug-description
   ```

2. **Commituj często:**
   ```bash
   git add ...
   git commit -m "[type] description"
   ```

3. **Push do fork'u:**
   ```bash
   git push origin feat/my-feature
   ```

4. **Pull Request na GitHub:**
   - Tytuł: `[TYPE] Description`
   - Opis: co zrobione, dlaczego, jak testować
   - Link issuesToy jeśli istnieją

5. **Cierpliwie czekaj na review** 😊

---

## Testowanie przed submitem

```bash
# Uruchom ALL testy
pytest -q

# Coverage
pytest --cov=src/crash_kedro --cov-report=html

# Linting
ruff check src tests

# Format
ruff format src tests

# Type checking (opcjonalnie, jeśli masz mypy)
mypy src

# Pylint na key modules
pylint src/crash_kedro/api/app.py

# Demo script
python scripts/demo.py
```

**Checklist przed PR:**
- [ ] Wszystkie testy pass
- [ ] `ruff check` bez błędów
- [ ] Docstrings na nowych funkcjach
- [ ] Type hints na nowych funkcjach
- [ ] Brak lęgących `print()` — użyj loggingu
- [ ] Git history jest czysty (rebase jeśli potrzeba)

---

## Logging

Używaj `logging` zamiast `print()`:

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.info("Starting process...")
    logger.debug(f"Variable x = {x}")
    logger.warning("Potential issue")
    logger.error("Something went wrong", exc_info=True)
```

Konfiguracja: `conf/logging.yml` (Kedro setup)

---

## Zmienne środowiskowe

Jeśli dodajesz nowe zmienne env, dokumentuj je:

```python
import os

MODEL_PATH = os.getenv("MODEL_PATH", "data/06_models/model.pkl")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
```

I dodaj do dokumentacji `docs/SETUP.md`.

---

## Git workflow dla Kedro

### Katalogi ignorowane (`.gitignore`)
```
.venv/
__pycache__/
.pytest_cache/
.coverage
data/01_raw/*  # (opcjonalnie, jeśli duże)
.idea/
.vscode/
*.pkl
*.parquet
mlruns/
```

### Branche
- `main` — production ready
- `develop` — integracyjny (jeśli used)
- `feat/xxx` — feature branches
- `fix/xxx` — bug fix branches

### Merge strategy
Preferuj **squash** dla feature branches, aby main miał czysty history.

---

## CI/CD — Github Actions

Workflows są w `.github/workflows/`:
- **ci.yml** — testy na każdy push
- **cd.yml** — build Docker na tag
- **ct.yml** — continuous training (weekly)

Jeśli zmienisz dependencjes:
1. Updatuj `pyproject.toml` lub `requirements.txt`
2. CI się odpal automatycznie
3. Jeśli fail → fix i push again

---

## Lokalny workflow

### Zanim zakommitujesz
```bash
# 1. Testuj
pytest -q

# 2. Lintuj
ruff check src tests

# 3. Formatuj
ruff format src tests

# 4. Commituj
git add .
git commit -m "[feat] describe your change"

# 5. Push
git push origin branch-name
```

### W case problemu
```bash
# Edytujesz coś co już commited
git add .
git commit --amend  # Dodaj do ostatniego commita

# Chcesz cofnąć ostatni commit
git reset --soft HEAD~1  # Keep changes
git reset --hard HEAD~1  # Discard changes

# Chcesz zrobić feature branch z main
git fetch origin
git checkout -b feat/new origin/main
```

---

## Kod review — czego szukamy

### Funkcjonalność
- [ ] Kod robi to, co ma robić
- [ ] Handlej edge cases (empty DataFrame, None, itp.)
- [ ] Brak redundancji

### Testy
- [ ] Test cases coverują happy path + edge cases
- [ ] Testy failują bez fix
- [ ] Coverage jest OK (~70% minimum)

### Dokumentacja
- [ ] Docstrings są jasne
- [ ] Type hints są prawidłowe
- [ ] Brak zastarzeałych komentarzy

### Style
- [ ] Zgodne z PEP8 / project style
- [ ] Sensowne nazwy zmiennych
- [ ] Brak duplicated code

### Performance
- [ ] Nie ma obvious bottlenecków
- [ ] Dane structures są appropriate
- [ ] Brak memory leaks

---

## Raportowanie błędów

Jeśli znajdziesz bug:
1. Otwórz **Issue** na GitHub
2. Tytuł: krótki opis
3. Body:
   ```
   ## Opis
   [Co się stało]
   
   ## Kroki repro
   1. ...
   2. ...
   
   ## Oczekiwane
   [Co miało się stać]
   
   ## Aktualne
   [Co się stało]
   
   ## Environment
   - OS: Windows/Linux/Mac
   - Python: 3.10/3.11/3.12
   - Branch: main/feature
   ```

---

## Pytania / Pomoc

- **Dokumentacja projektu:** `/docs`
- **Kod:** czytaj existing code + docstrings
- **ML questions:** czytaj `docs/MODELS.md`
- **API questions:** czytaj `docs/API.md`

---

## Podsumowanie workflow'u

```
Fork → Clone → Branch → Code → Test → Commit → Push → PR → Review → Merge
```

1. Utwórz branch
2. Kod + testy
3. `ruff`, `pytest` → pass
4. Commit + push
5. PR z opisem
6. Wait for review
7. Address feedback
8. Merge! 🎉

---

## Dodatkowe zasoby

- [PEP 8](https://www.python.org/dev/peps/pep-0008/) — Python style guide
- [Kedro docs](https://docs.kedro.org/) — Kedro documentation
- [FastAPI docs](https://fastapi.tiangolo.com/) — FastAPI tutorials
- [Sklearn docs](https://scikit-learn.org/) — Machine learning
- [Git docs](https://git-scm.com/book/en/v2) — Git learning

Powodzenia! 🚀


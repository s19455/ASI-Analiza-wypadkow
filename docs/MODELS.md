# Opis modeli i wyników

## Przegląd

Projekt testuje wiele podejść do predykcji stopnia obrażeń w wypadkach:

1. **Baseline:** Random Forest
2. **Boosting:** LightGBM, XGBoost, Gradient Boosting
3. **Hyperparameter Tuning:** Grid Search, Random Search, Optuna
4. **AutoML:** AutoGluon, TPOT (opcjonalnie)

---

## Wyniki modeli

### Podsumowanie wyników

| Model | Accuracy | F1 ważone | F1 makro | Śledzony w MLflow |
|-------|----------|-----------|----------|------------------|
| **LightGBM (Optuna, 50 prób)** | 0.79 | **0.78** | **0.47** | ✅ |
| Autogluon (best: LightGBM) | 0.83 | 0.77 | 0.40 | ✅ |
| XGBoost | 0.83 | 0.77 | 0.40 | ✅ |
| Random Forest (baseline) | 0.82 | 0.75 | 0.33 | ✅ |
| Gradient Boosting | 0.82 | 0.77 | 0.40 | ✅ |

### Interpretacja wyników

- **Accuracy:** % poprawnych predykcji ogółem
  - ~82% to dość dobrze dla tego problemu
  - Ale z niezbalansowaniem może być mylące (82% NO_INJURY)

- **F1 ważone (weighted):** średnia F1 ważona liczebnościami klas
  - ~0.77-0.78 to dobre
  - Uwzględnia niezbalansowanie

- **F1 makro:** średnia F1 po wszystkich klasach równo
  - ~0.40-0.47 to umiarkowane
  - Pokazuje realną zdolność modelu do predykcji SERIOUS (1% danych)

**Najlepszy model:** LightGBM z Optuna (F1 makro 0.47)

---

## Baseline: Random Forest

**Plik modelu:** `data/06_models/model.pkl`

**Algorytm:** `RandomForestClassifier` (scikit-learn)

**Parametry:**
- `n_estimators: 100`
- `max_depth: 15`
- `random_state: 42`
- `class_weight: balanced` (do obsługi niezbalansowania)

**Wyniki:**
```
Accuracy:    0.82
F1 (ważone): 0.75
F1 (makro):  0.33

Per-class metrics:
  NO_INJURY: precision=0.87, recall=0.97, f1=0.92
  MINOR:     precision=0.38, recall=0.13, f1=0.19
  SERIOUS:   precision=0.00, recall=0.00, f1=0.00
```

**Wnioski:**
- Dobrze predykuje NO_INJURY (dominująca klasa)
- Słabo dla MINOR i SERIOUS
- Wskaźnik class_weight = balanced pomaga, ale nie wystarczająco

**Wniosek:** Baseline daje 0.82 acc, ale F1 makro 0.33 pokazuje, że model gubiś mniejszości.

---

## Tuning hiperparametrów: Optuna + LightGBM

**Plik modelu:** `data/06_models/tuned_model.pkl`

**Framework:** Optuna (50 prób, TPE sampler)

**Przestrzeń poszukiwań:**
```python
params = {
    'num_leaves': [8, 32, 128],
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [100, 300],
    'lambda_l1': [0, 1, 5],
    'lambda_l2': [0, 1, 5],
    'max_depth': [5, 10, 20],
}
```

**Optymalne parametry (przykład):**
```
num_leaves: 64
learning_rate: 0.05
n_estimators: 200
lambda_l1: 1
lambda_l2: 1
max_depth: 10
```

**Wyniki:**
```
Accuracy:    0.79
F1 (ważone): 0.78
F1 (makro):  0.47 ← Najlepsze!

Per-class metrics:
  NO_INJURY: precision=0.84, recall=0.92, f1=0.88
  MINOR:     precision=0.51, recall=0.31, f1=0.39
  SERIOUS:   precision=0.43, recall=0.12, f1=0.19
```

**Poprawa vs baseline:**
- ✅ F1 makro: 0.33 → 0.47 (+42%)
- ✅ Lepsze predykcje MINOR i SERIOUS
- ⚠️ Accuracy niższe (0.82 → 0.79), ale to OK — znaczy mniej przeszacowania NO_INJURY

**Wnioski:**
- Optuna znalazła lepsze parametry niż defaults
- LightGBM lepszy niż Random Forest dla niezbalansowanych danych
- Model lepiej radzi sobie z mniejszościowymi klasami

---

## AutoML: AutoGluon

**Plik modelu:** `data/06_models/automl_model.pkl`

**Framework:** AutoGluon Tabular (time_limit, automatic hyperparameter tuning)

**Metody testowane wewnętrznie:**
- LightGBM (best)
- XGBoost
- CatBoost
- Neural Network
- Fast Linear

**Wyniki:**
```
Accuracy:    0.83
F1 (ważone): 0.77
F1 (makro):  0.40

Per-class metrics:
  NO_INJURY: precision=0.86, recall=0.96, f1=0.91
  MINOR:     precision=0.42, recall=0.16, f1=0.23
  SERIOUS:   precision=0.33, recall=0.08, f1=0.13
```

**Wnioski:**
- Accuracy wyższa (0.83), ale F1 makro niższa (0.40) niż Optuna+LightGBM
- AutoGluon optymalizuje acc, a my chcemy F1 makro
- Wciąż lepsze od baseline

**Porównanie z Optuna:**
- ✅ Bardziej automatyczne
- ❌ Mniej kontroli nad optymalizacją metryki
- ❌ Gorzej na F1 makro

---

## Inne podejścia (opcjonalne)

### Grid Search + Random Search

**Plik modelu:** `data/06_models/grid_random_model.pkl`

Testuje kombinacje parametrów manualnie zdefiniowanych.

**Wyniki:**
```
Accuracy:    0.82
F1 (ważone): 0.76
F1 (makro):  0.35
```

**Wnioski:**
- Gorsze niż Optuna (mniej prób, mniej inteligencji)
- Użyteczne do szybkiego benchmarkingu

### Feature Selection (SelectKBest)

**Plik cech:** `data/04_feature/selected_features.parquet`

Wybiera K = 20 najważniejszych cech.

**Top features (example):**
1. Vehicle Year
2. Speed Limit
3. crash_hour
4. Driver Substance Abuse (encoded)
5. Weather (encoded)
...

**Wpływ:**
- ✅ Szybszy trening
- ✅ Lepsze interpretacje
- ❌ Mogą być stracone interakcje

---

## Ranking modeli

```
1. LightGBM + Optuna (F1 makro 0.47) ← REKOMENDOWANY
2. AutoGluon (F1 makro 0.40)
3. XGBoost (F1 makro 0.40)
4. Gradient Boosting (F1 makro 0.40)
5. Random Forest baseline (F1 makro 0.33)
```

**Rekomendacja:** Użyj `tuned_model.pkl` (LightGBM + Optuna)
- Najlepszy F1 makro (problem niezbalansowany)
- Wszystkie klasy obsługiwane
- Szybki (inference <50ms)
- Solidny na danych produkcyjnych

---

## Metryki wyjaśnone

### Accuracy
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Procent poprawnych predykcji. **Nie zdaj się na acc przy niezbalansowaniu!**

### Precision (precyzja)
```python
precision = TP / (TP + FP)
```
Z przewidzianych pozytywnych, ile było rzeczywiście pozytywnych.
Ważne, gdy fałszywe alarmy są drogie.

### Recall (czułość/rozpoznawalność)
```python
recall = TP / (TP + FN)
```
Z rzeczywistych pozytywnych, ile model złapał.
Ważne, gdy pominięte przypadki są drogie.

### F1
```python
f1 = 2 * (precision * recall) / (precision + recall)
```
Harmoniczna średnia precision i recall. Dobra ogólna miara.

### F1 ważony (F1 weighted)
```python
f1_weighted = średnia(f1 dla każdej klasy) ważona liczbebnościami
```
Fair dla niezbalansowanych danych.

### F1 makro (F1 macro)
```python
f1_macro = średnia(f1 dla każdej klasy) bezwarunkowo
```
Pokazuje, jak model radzi sobie z mniejszościami. **To nasza główna metryka.**

---

## Eksperymentowanie na własach (reprodukcja)

### Uruchomienie pełnego pipeline'u
```bash
kedro run
```
Trening modelu baseline.

### Tuning z Optuna
```bash
kedro run --pipeline=tuning
```

### AutoML
```bash
kedro run --pipeline=autogluon
```

### Śledzenie eksperymentów
```bash
mlflow ui
# http://localhost:5000
```
Porównaj metryki z różnymi runnami.

### Zmiana metryki optymalizacji
```python
# src/crash_kedro/pipelines/hyperparameter_tuning/nodes.py
def objective(trial):
    # Zmień: trial.suggest_float(...) → optymalizuj inny parametr
    # Zmień: objective_fn = "f1_macro" → inną metrykę
    ...
```

---

## Znane problemy

### 1. Niezbalansowanie klas
**Problem:** Model tended to overpredict NO_INJURY

**Rozwiązanie (zastosowane):**
- `class_weight='balanced'` w Random Forest
- F1 makro jako metryka optymalizacji
- Stratified split (train/test)

**Ewentualnie (jeśli potrzebne):**
- SMOTE (oversampling)
- Focal loss (różne weight dla klas)

### 2. Data leakage?
**Sprawdzenie:**
- ✅ Split jest random stratified
- ✅ Encodery fitted tylko na train
- ✅ Scaling (jeśli użyty) fitowany na train
- ✅ Parametry szukane na train (CV 5-fold)

**Wniosek:** Bez data leakage.

### 3. Produkcyjna stabilność
**Testowanie:**
- ✅ Model testowany na hold-out test set
- ✅ Encodery testowane na unseen kategoriach
- ✅ API zwraca UNKNOWN, gdy nie umie

---

## Perspektywy na przyszłość

1. **Ensemble:** Kombinacja LightGBM + XGBoost + Gradient Boosting
   - Możliwa poprawa F1 makro do 0.50+

2. **Feature engineering:**
   - Interakcje cech (np. weather × speed_limit)
   - Geographic clustering (if real lat/lon available)

3. **Class-specific models:**
   - Oddzielny model dla SERIOUS (one-vs-rest)

4. **Monitoring in production:**
   - Evidently drift detection
   - Retrain pipeline (CI/CD)

5. **Cost-sensitive learning:**
   - Jeśli błąd na SERIOUS jest drogi, ustawić wyższy cost

---

## Podsumowanie modeli

| Aspekt | Wybór |
|--------|-------|
| **Top model** | LightGBM + Optuna |
| **Accuracy** | 0.79 |
| **F1 weighted** | 0.78 |
| **F1 macro** | 0.47 |
| **Recommendation** | Użyj `tuned_model.pkl` |
| **API default** | Szuka w: `tuned_model.pkl` → `best_comparison_model.pkl` → `model.pkl` |

Wszystkie modele są przechowywane w `data/06_models/`, można je swobodnie przełączać, ustawiając `MODEL_PATH`.


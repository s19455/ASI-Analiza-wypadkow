# Opis modeli i wyników

## Testowane podejścia

1. **Baseline:** Random Forest (`data_modeling`)
2. **Porównanie modeli:** Random Forest, Gradient Boosting, XGBoost, LightGBM (`tuning`, `automl`)
3. **Strojenie hiperparametrów:** Grid Search, Random Search, Optuna (`tuning`)
4. **AutoML:** AutoGluon (`autogluon`)
5. **Selekcja cech:** SelectKBest (`feature_selection`)

Wszystkie wyniki są logowane do MLflow (`mlflow ui`).

## Wyniki

| Model | Accuracy | F1 ważone | F1 makro |
|-------|----------|-----------|----------|
| **LightGBM (Optuna, 50 prób)** | 0.79 | **0.78** | **0.47** |
| Autogluon (best: LightGBM) | 0.83 | 0.77 | 0.40 |
| XGBoost | 0.83 | 0.77 | 0.40 |
| Gradient Boosting | 0.82 | 0.77 | 0.40 |
| Random Forest (baseline) | 0.82 | 0.75 | 0.33 |

Dane są mocno niezbalansowane (klasa SERIOUS to ~1%), więc głównie patrzymy na **F1 makro** — accuracy zawyża wynik, bo model często zgaduje dominującą klasę NO_INJURY.

## Baseline: Random Forest

- Plik: `data/06_models/model.pkl`
- `RandomForestClassifier` (scikit-learn), `n_estimators=100`, `class_weight="balanced"`, `random_state=42`
- Accuracy 0.82, F1 ważone 0.75, F1 makro 0.33

Dobrze przewiduje NO_INJURY, słabo radzi sobie z MINOR i SERIOUS — stąd niskie F1 makro.

## Strojenie: Optuna + LightGBM

- Plik: `data/06_models/tuned_model.pkl`
- Optuna, 50 prób, optymalizacja pod F1 makro
- Accuracy 0.79, F1 ważone 0.78, **F1 makro 0.47**

To nasz najlepszy model. Accuracy jest niższe niż w baseline, ale F1 makro wyraźnie lepsze — model rzadziej przewiduje wszystko jako NO_INJURY.

Pipeline `tuning` uruchamia też Grid Search i Random Search dla porównania (wyniki w MLflow, eksperyment `crash-severity-tuning`).

## AutoML: AutoGluon

- Pipeline `autogluon`, `time_limit=600`, preset `medium_quality`
- Accuracy 0.83, F1 ważone 0.77, F1 makro 0.40
- Najlepszy model w leaderboardzie: LightGBM

AutoGluon optymalizuje pod accuracy, więc ma wyższe acc, ale niższe F1 makro niż nasz tuning pod Optuna.

## Selekcja cech (SelectKBest)

Pipeline `feature_selection` wybiera 20 najważniejszych cech (`mutual_info_classif`) i porównuje model na pełnym zestawie cech vs. zredukowanym. Wyniki w `data/08_reporting/feature_selection_metrics.json`.

## Reprodukcja

```bash
kedro run                      # baseline
kedro run --pipeline=tuning    # Grid + Random + Optuna
kedro run --pipeline=automl    # porównanie modeli
kedro run --pipeline=autogluon # AutoGluon
mlflow ui                      # porównanie eksperymentów
```

## Uwagi

- **Niezbalansowanie klas** — używamy `class_weight="balanced"`, stratified split i F1 makro jako głównej metryki.
- **Nieznane kategorie** w inferencji są kodowane jako `-1` (encoder nie widział ich w treningu).
- API domyślnie szuka modelu w kolejności: `MODEL_PATH` → `model.pkl` → `tuned_model.pkl` → `best_comparison_model.pkl`.

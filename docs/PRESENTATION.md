# Prezentacja końcowa — plan slajdów

## Cel
Krótka prezentacja 10–15 minut pokazująca problem, pipeline ML, wyniki, wdrożenie i demo.

## Proponowana struktura
1. **Problem biznesowy**
   - Predykcja stopnia obrażeń w wypadkach drogowych.
   - Niezbalansowane klasy i sens używania F1 macro.

2. **Dane**
   - Źródło danych, liczba rekordów i kolumn.
   - Najważniejsze zmienne wejściowe.

3. **Baseline**
   - Notebook `notebooks/01_baseline_model.ipynb`.
   - EDA, preprocessing, model bazowy, ewaluacja.

4. **Pipeline ML**
   - Kedro: `data_preparation` → `data_modeling`.
   - Dodatkowe pipeline’y: `tuning`, `feature_selection`, `automl`, `autogluon`.

5. **Udoskonalanie modelu**
   - MLflow.
   - Grid Search / Random Search / Optuna.
   - SelectKBest.
   - AutoGluon.

6. **Wdrożenie**
   - FastAPI (`/predict`, `/health`, `/predictions/recent`).
   - Docker i opcjonalny deployment z GitHub Actions.

7. **Monitoring**
   - Logowanie predykcji do `logs/predictions.jsonl`.
   - Drift report generowany przez `scripts/run_drift_monitoring.py`.

8. **Wyniki i wnioski**
   - Porównanie modeli.
   - Finalny wybór modelu.
   - Najważniejsze ograniczenia i dalsze kroki.

## Demo
W prezentacji warto pokazać co najmniej jedno z poniższych:
- wywołanie endpointu `/predict` w Swagger UI,
- uruchomienie `python scripts/demo.py`,
- widok `streamlit run streamlit_app.py`,
- skrócony raport z `mlflow ui`.

## Materiały pomocnicze
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/API.md`
- `docs/MODELS.md`
- `docs/SETUP.md`


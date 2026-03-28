# Modele

Zgodnie z konwencja Kedro, wytrenowane modele zapisujemy w `data/06_models/`.

Lista modeli:
- `data/06_models/model.pkl` - Random Forest baseline
- `data/06_models/best_comparison_model.pkl` - najlepszy z porownania (RF/GB/XGB/LGBM)
- `data/06_models/grid_random_model.pkl` - najlepszy z Grid Search / Random Search
- `data/06_models/tuned_model.pkl` - LightGBM po tuningu Optuna
- `data/06_models/autogluon_predictor.pkl` - predictor Autogluona
- `data/06_models/autogluon/` - katalog z artefaktami Autogluona

Modele sa zapisywane przez Kedro (`PickleDataset` w `conf/base/catalog.yml`).

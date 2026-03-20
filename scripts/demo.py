"""
Demo predykcji stopnia obrazen w wypadkach drogowych.
Uruchomienie: python scripts/demo.py
"""

import pickle
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main():
    print("=" * 70)
    print("  DEMO: Predykcja stopnia obrazen w wypadkach drogowych")
    print("  Dane: Montgomery County, MD (2015-2024)")
    print("=" * 70)

    # 1. Wczytanie modelu
    models_dir = Path("data/06_models")
    model_path = models_dir / "tuned_model.pkl"
    if not model_path.exists():
        model_path = models_dir / "model.pkl"

    print(f"\n[1] Wczytywanie modelu: {model_path.name}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"    Typ modelu: {type(model).__name__}")

    # 2. Wczytanie metryk
    print("\n[2] Wyniki modeli:")
    reports_dir = Path("data/08_reporting")

    for name, filename in [
        ("Model bazowy (RF)", "metrics.json"),
        ("Porownanie modeli", "comparison_metrics.json"),
        ("Optuna-tuned LightGBM", "tuning_metrics.json"),
    ]:
        fpath = reports_dir / filename
        if fpath.exists():
            with open(fpath) as f:
                m = json.load(f)
            if "f1_weighted" in m:
                print(f"    {name}: F1w={m['f1_weighted']:.4f}, acc={m['accuracy']:.4f}")
            elif "optuna_f1_weighted" in m:
                print(f"    {name}: F1w={m['optuna_f1_weighted']:.4f}, acc={m['optuna_accuracy']:.4f}")
            elif "best_model_name" in m:
                best = m["best_model_name"]
                print(f"    {name}: najlepszy={best}, F1w={m[best]['f1_weighted']:.4f}")

    # 3. Predykcje na prawdziwych danych
    features_path = Path("data/03_primary/crash_features.parquet")
    if not features_path.exists():
        print("\n[!] Brak przetworzonych danych. Uruchom najpierw: kedro run")
        return

    print("\n[3] Predykcje na zbiorze testowym:")
    print("-" * 70)

    df = pd.read_parquet(features_path)
    X = df.drop("Severity_Group", axis=1)
    y = df["Severity_Group"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    # 4. Przykladowe predykcje na losowych wierszach
    print("\n[4] Przykladowe predykcje (5 losowych wypadkow):")
    print("-" * 70)

    sample_idx = np.random.RandomState(42).choice(len(X_test), 5, replace=False)
    X_sample = X_test.iloc[sample_idx]
    y_sample = y_test.iloc[sample_idx]

    sample_preds = model.predict(X_sample)

    etykiety = {
        "NO_INJURY": "Brak obrazen",
        "MINOR": "Drobne obrazenia",
        "SERIOUS": "Powazne obrazenia",
        0: "Drobne obrazenia",
        1: "Brak obrazen",
        2: "Powazne obrazenia",
    }

    for i, (idx, pred, true) in enumerate(zip(sample_idx, sample_preds, y_sample)):
        pred_name = etykiety.get(pred, str(pred))
        true_name = etykiety.get(true, str(true))
        status = "OK" if str(pred) == str(true) else "MISS"
        print(f"  [{i+1}] Predykcja: {pred_name:20s} | Rzeczywiste: {true_name:20s} | {status}")

    # 5. Informacja o API
    print("\n" + "=" * 70)
    print("  Aby uruchomic API:")
    print("    uvicorn src.crash_kedro.api.app:app --reload")
    print("    Swagger UI: http://localhost:8000/docs")
    print("=" * 70)


if __name__ == "__main__":
    main()

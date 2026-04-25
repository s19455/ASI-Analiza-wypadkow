"""Demo - wczytuje wytrenowany model i pokazuje wyniki na zbiorze testowym."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def main():
    print("Demo - predykcja stopnia obrazen w wypadkach\n")

    models_dir = Path("data/06_models")

    # bierzemy najlepszy dostepny model
    model_path = models_dir / "tuned_model.pkl"
    if not model_path.exists():
        model_path = models_dir / "model.pkl"

    print(f"Model: {model_path.name}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # wyniki z poprzednich runow
    print("\nWyniki:")
    reports_dir = Path("data/08_reporting")
    for filename in ["metrics.json", "tuning_metrics.json", "comparison_metrics.json"]:
        fpath = reports_dir / filename
        if fpath.exists():
            with open(fpath) as f:
                m = json.load(f)
            print(f"  {filename}: {json.dumps(m, indent=None)[:120]}...")

    # predykcje na test set
    features_path = Path("data/03_primary/crash_features.parquet")
    if not features_path.exists():
        print("\nBrak przetworzonych danych - uruchom najpierw `kedro run`")
        return

    df = pd.read_parquet(features_path)
    X = df.drop("Severity_Group", axis=1)
    y = df["Severity_Group"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preds = model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, preds))

    # przykladowe 5 predykcji
    print("Przykladowe predykcje:")
    idx = np.random.RandomState(42).choice(len(X_test), 5, replace=False)
    sample_preds = model.predict(X_test.iloc[idx])
    sample_y = y_test.iloc[idx]

    for i in range(5):
        pred = sample_preds[i]
        true = sample_y.iloc[i]
        ok = "OK" if str(pred) == str(true) else "BLAD"
        print(f"  pred={pred:12s} true={true:12s} -> {ok}")


if __name__ == "__main__":
    main()

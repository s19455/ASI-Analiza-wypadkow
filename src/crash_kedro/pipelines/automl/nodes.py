"""AutoML pipeline nodes - model comparison with multiple algorithms."""

import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_automl(df: pd.DataFrame, parameters: dict):
    """Automated model selection - compares multiple algorithms."""
    X = df.drop("Severity_Group", axis=1)
    y = df["Severity_Group"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=y,
    )

    candidates = {
        "RandomForest_200": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "ExtraTrees_200": ExtraTreesClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, random_state=42, algorithm="SAMME",
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
    }

    best_score = 0
    best_model = None
    best_name = ""
    all_results = {}

    for name, model in candidates.items():
        print(f"\nTraining {name}...")  # noqa: T201
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_w = f1_score(y_test, preds, average="weighted")
        f1_m = f1_score(y_test, preds, average="macro")

        all_results[name] = {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m}
        print(f"  F1 weighted: {f1_w:.4f}, F1 macro: {f1_m:.4f}")  # noqa: T201

        if f1_w > best_score:
            best_score = f1_w
            best_model = model
            best_name = name

    print(f"\n{'='*50}")  # noqa: T201
    print(f"Best AutoML model: {best_name} (F1w={best_score:.4f})")  # noqa: T201
    print(classification_report(y_test, best_model.predict(X_test)))  # noqa: T201

    metrics = {
        "automl_accuracy": all_results[best_name]["accuracy"],
        "automl_f1_weighted": all_results[best_name]["f1_weighted"],
        "automl_f1_macro": all_results[best_name]["f1_macro"],
        "best_pipeline": best_name,
        "all_results": all_results,
    }

    return best_model, metrics

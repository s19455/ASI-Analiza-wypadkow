"""
Pipeline modelowania danych - trening i ewaluacja modelu.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, parameters: dict):
    X = df.drop("Severity_Group", axis=1)
    y = df["Severity_Group"]

    return train_test_split(
        X,
        y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=y,
    )


def train_model(X_train, y_train):
    params = {
        "n_estimators": 100,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Logowanie do MLflow
    try:
        import mlflow  # noqa: PLC0415
        import mlflow.sklearn  # noqa: PLC0415

        mlflow.set_experiment("crash-severity-baseline")
        with mlflow.start_run(run_name="random_forest_baseline"):
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.sklearn.log_model(model, name="model")
    except Exception as e:
        print(f"[MLflow] Pominieto logowanie: {e}")  # noqa: T201

    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_weighted = f1_score(y_test, preds, average="weighted")
    f1_macro = f1_score(y_test, preds, average="macro")

    report = classification_report(y_test, preds)
    print(report)  # noqa: T201

    metrics = {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
    }

    # Logowanie metryk do MLflow
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_experiment("crash-severity-baseline")
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics(metrics)
    except Exception as e:
        print(f"[MLflow] Pominieto logowanie metryk: {e}")  # noqa: T201

    return metrics

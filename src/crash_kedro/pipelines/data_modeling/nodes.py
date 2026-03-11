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
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_weighted = f1_score(y_test, preds, average="weighted")
    f1_macro = f1_score(y_test, preds, average="macro")

    report = classification_report(y_test, preds)
    print(report)  # noqa: T201

    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
    }

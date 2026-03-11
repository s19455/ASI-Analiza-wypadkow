"""Hyperparameter tuning nodes with Grid Search, Random Search, and Bayesian Optimization."""

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def compare_models(df: pd.DataFrame, parameters: dict):
    X = df.drop("Severity_Group", axis=1)
    y = df["Severity_Group"]

    # XGBoost wymaga numerycznych labeli
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=y_encoded,
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="mlogloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, n_jobs=-1, class_weight="balanced", verbose=-1,
        ),
    }

    results = {}
    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_w = f1_score(y_test, preds, average="weighted")
        f1_m = f1_score(y_test, preds, average="macro")

        results[name] = {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m}
        print(f"\n{'='*50}")  # noqa: T201
        print(f"{name}")  # noqa: T201
        print(f"{'='*50}")  # noqa: T201
        print(classification_report(y_test, preds))  # noqa: T201

        if f1_w > best_score:
            best_score = f1_w
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} (F1 weighted: {best_score:.4f})")  # noqa: T201
    results["best_model_name"] = best_name

    return best_model, results


def bayesian_tuning(df: pd.DataFrame, parameters: dict):
    X = df.drop("Severity_Group", axis=1)
    y = df["Severity_Group"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=y,
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = LGBMClassifier(
            **params, random_state=42, n_jobs=-1,
            class_weight="balanced", verbose=-1,
        )
        score = cross_val_score(
            model, X_train, y_train, cv=3, scoring="f1_weighted", n_jobs=-1
        )
        return score.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\nBest trial F1 weighted: {study.best_value:.4f}")  # noqa: T201
    print(f"Best params: {study.best_params}")  # noqa: T201

    best_model = LGBMClassifier(
        **study.best_params, random_state=42, n_jobs=-1,
        class_weight="balanced", verbose=-1,
    )
    best_model.fit(X_train, y_train)

    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_w = f1_score(y_test, preds, average="weighted")
    f1_m = f1_score(y_test, preds, average="macro")

    print("\nOptuna-tuned LightGBM:")  # noqa: T201
    print(classification_report(y_test, preds))  # noqa: T201

    metrics = {
        "optuna_accuracy": acc,
        "optuna_f1_weighted": f1_w,
        "optuna_f1_macro": f1_m,
        "best_params": str(study.best_params),
        "n_trials": len(study.trials),
    }

    return best_model, metrics

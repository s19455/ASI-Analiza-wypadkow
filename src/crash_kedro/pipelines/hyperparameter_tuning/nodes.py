"""Hyperparameter tuning nodes - Grid Search, Random Search i Bayesian Optimization (Optuna)."""

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def _log_to_mlflow(experiment, run_name, params, metrics):
    """Helper - logowanie do MLflow z obsluga bledow."""
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
    except Exception as e:
        print(f"[MLflow] Pominieto: {e}")  # noqa: T201


def compare_models(df: pd.DataFrame, parameters: dict):
    """Porownanie kilku modeli (RF, GB, XGB, LGBM) na tych samych danych."""
    X = df.drop("Severity_Group", axis=1)
    y = df["Severity_Group"]

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

        _log_to_mlflow(
            "crash-severity-comparison",
            run_name=name,
            params={"model_type": name},
            metrics={"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m},
        )

        if f1_w > best_score:
            best_score = f1_w
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} (F1 weighted: {best_score:.4f})")  # noqa: T201
    results["best_model_name"] = best_name

    return best_model, results


def grid_random_search(df: pd.DataFrame, parameters: dict):
    """Grid Search i Random Search na Random Forest - dla porownania z Bayesian."""
    X = df.drop("Severity_Group", axis=1)
    y = df["Severity_Group"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=y,
    )

    # Grid Search - maly grid (3*3 = 9 kombinacji)
    grid_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
    }
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        grid_params,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    grid_preds = grid.predict(X_test)
    grid_metrics = {
        "grid_f1_weighted": f1_score(y_test, grid_preds, average="weighted"),
        "grid_f1_macro": f1_score(y_test, grid_preds, average="macro"),
        "grid_accuracy": accuracy_score(y_test, grid_preds),
    }
    print(f"\n[Grid Search] Best params: {grid.best_params_}")  # noqa: T201
    print(f"[Grid Search] F1 ważone: {grid_metrics['grid_f1_weighted']:.4f}")  # noqa: T201

    _log_to_mlflow(
        "crash-severity-tuning",
        run_name="grid_search",
        params={**grid.best_params_, "search_type": "GridSearchCV"},
        metrics=grid_metrics,
    )

    # Random Search - wiekszy parameter space, 15 losowych iteracji
    random_param_dist = {
        "n_estimators": [100, 150, 200, 250, 300, 400, 500],
        "max_depth": [5, 10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
    }
    random_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        random_param_dist,
        n_iter=15,
        cv=3,
        scoring="f1_weighted",
        random_state=42,
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)
    rs_preds = random_search.predict(X_test)
    rs_metrics = {
        "random_f1_weighted": f1_score(y_test, rs_preds, average="weighted"),
        "random_f1_macro": f1_score(y_test, rs_preds, average="macro"),
        "random_accuracy": accuracy_score(y_test, rs_preds),
    }
    print(f"\n[Random Search] Best params: {random_search.best_params_}")  # noqa: T201
    print(f"[Random Search] F1 ważone: {rs_metrics['random_f1_weighted']:.4f}")  # noqa: T201

    _log_to_mlflow(
        "crash-severity-tuning",
        run_name="random_search",
        params={**random_search.best_params_, "search_type": "RandomizedSearchCV"},
        metrics=rs_metrics,
    )

    # Wybor lepszego modelu
    if rs_metrics["random_f1_weighted"] > grid_metrics["grid_f1_weighted"]:
        best = random_search.best_estimator_
        best_name = "random_search"
    else:
        best = grid.best_estimator_
        best_name = "grid_search"

    combined = {
        **grid_metrics,
        **rs_metrics,
        "grid_best_params": str(grid.best_params_),
        "random_best_params": str(random_search.best_params_),
        "winner": best_name,
    }
    return best, combined


def bayesian_tuning(df: pd.DataFrame, parameters: dict):
    """Bayesian Optimization (Optuna) na LightGBM."""
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

    _log_to_mlflow(
        "crash-severity-tuning",
        run_name="optuna_bayesian",
        params={**study.best_params, "search_type": "Optuna_Bayesian", "n_trials": len(study.trials)},
        metrics={"f1_weighted": f1_w, "f1_macro": f1_m, "accuracy": acc},
    )

    return best_model, metrics

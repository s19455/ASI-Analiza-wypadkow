"""Hyperparameter tuning nodes - Grid Search, Random Search i Bayesian Optimization (Optuna)."""

import logging
from importlib import import_module

import optuna
import numpy as np
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
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

ADASYN = None
SMOTE = None
RandomUnderSampler = None
ImbPipeline = None
FunctionSampler = None
BalancedRandomForestClassifier = None

try:  # pragma: no cover - optional dependency
    _imblearn = import_module("imblearn")
    _imblearn_ensemble = import_module("imblearn.ensemble")
    _imblearn_pipeline = import_module("imblearn.pipeline")
    _imblearn_over = import_module("imblearn.over_sampling")
    _imblearn_under = import_module("imblearn.under_sampling")
except ImportError:
    pass
else:
    FunctionSampler = _imblearn.FunctionSampler
    BalancedRandomForestClassifier = _imblearn_ensemble.BalancedRandomForestClassifier
    ADASYN = _imblearn_over.ADASYN
    SMOTE = _imblearn_over.SMOTE
    RandomUnderSampler = _imblearn_under.RandomUnderSampler
    ImbPipeline = _imblearn_pipeline.Pipeline


ALLOWED_METRICS = {"accuracy", "f1_weighted", "f1_macro"}


def _resolve_metric(parameters: dict, key: str, default: str = "f1_macro") -> str:
    metric = str(parameters.get(key, default)).lower()
    return metric if metric in ALLOWED_METRICS else default


def _resolve_class_weight(y_train: pd.Series, parameters: dict):
    """Zwraca wagi klas dla modeli obsługujących class_weight."""
    strategy = str(parameters.get("class_weight_strategy", "balanced_plus")).lower()

    if strategy in {"none", "off", "disabled"}:
        return None

    if strategy in {"balanced", "balanced_subsample"}:
        return strategy

    classes = pd.Index(pd.unique(y_train)).to_numpy()
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    multipliers = parameters.get("class_weight_multipliers", {}) or {}

    return {
        cls: float(weight) * float(multipliers.get(cls, 1.0))
        for cls, weight in zip(classes, weights)
    }


def _build_sampler(y_train: pd.Series, parameters: dict):
    """Buduje sampler do balansowania klas na poziomie danych."""
    if ImbPipeline is None or FunctionSampler is None:
        return None

    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    strategy = str(parameters.get("sampling_strategy", "none")).lower()
    if strategy in {"none", "off", "disabled"}:
        return None

    counts = y_train.value_counts()
    if counts.empty:
        return None

    max_count = int(counts.max())
    min_count = int(counts.min())
    ratio = float(parameters.get("sampling_ratio", 0.2))

    if strategy in {"undersample", "random_undersample", "randomundersample"}:
        target_cap = max(min_count, int(max_count * ratio))

        def _fold_safe_undersample(X, y):
            y_series = pd.Series(y)
            fold_counts = y_series.value_counts()
            if fold_counts.empty or fold_counts.size < 2:
                return X, y

            majority_class = fold_counts.idxmax()
            majority_count = int(fold_counts[majority_class])
            fold_min_count = int(fold_counts.min())
            fold_target = max(fold_min_count, int(majority_count * ratio))
            fold_target = min(fold_target, majority_count)

            if fold_target >= majority_count:
                return X, y

            majority_index = y_series[y_series == majority_class].sample(
                n=fold_target,
                random_state=parameters.get("random_state", 42),
            ).index
            keep_index = y_series[y_series != majority_class].index.union(majority_index)

            if hasattr(X, "loc"):
                X_res = X.loc[keep_index]
            else:
                X_res = X[keep_index]
            y_res = y_series.loc[keep_index]
            if isinstance(y, pd.Series):
                y_res = y_res.astype(y.dtype)
            return X_res, y_res

        if max_count <= 1 or target_cap <= 1:
            return None

        return FunctionSampler(func=_fold_safe_undersample, validate=False)

    if strategy == "smote":
        minority_cap = max(2, min_count)
        if minority_cap < 2 or SMOTE is None:
            return None
        k_neighbors = min(int(parameters.get("smote_k_neighbors", 5)), minority_cap - 1)
        if k_neighbors < 1:
            return None
        target_counts = {
            cls: max(int(count), target_cap)
            for cls, count in counts.items()
            if int(count) < target_cap
        }
        if not target_counts:
            return None
        return SMOTE(
            sampling_strategy=target_counts,
            random_state=parameters.get("random_state", 42),
            k_neighbors=k_neighbors,
        )

    if strategy == "adasyn":
        if ADASYN is None or min_count < 2:
            return None
        n_neighbors = min(int(parameters.get("adasyn_n_neighbors", 5)), min_count - 1)
        if n_neighbors < 1:
            return None
        target_counts = {
            cls: max(int(count), target_cap)
            for cls, count in counts.items()
            if int(count) < target_cap
        }
        if not target_counts:
            return None
        return ADASYN(
            sampling_strategy=target_counts,
            random_state=parameters.get("random_state", 42),
            n_neighbors=n_neighbors,
        )

    logger.warning("Nieznana strategia balansowania '%s' - pomijam sampler.", strategy)
    return None


def _wrap_with_sampler(model, sampler):
    if sampler is None or ImbPipeline is None:
        return model
    return ImbPipeline([("sampler", sampler), ("model", model)])


def _prefix_model_params(params: dict, sampler) -> dict:
    if sampler is None or ImbPipeline is None:
        return params
    return {f"model__{key}": value for key, value in params.items()}


def _log_to_mlflow(experiment, run_name, params, metrics):
    """Helper - logowanie do MLflow z obsluga bledow."""
    try:
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
    except Exception as e:
        print(f"[MLflow] Pominieto: {e}")


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

    class_weight = _resolve_class_weight(y, parameters)
    if isinstance(class_weight, dict):
        class_weight = {int(le.transform([cls])[0]): weight for cls, weight in class_weight.items()}
    sampler = _build_sampler(y_train, parameters)
    selection_metric = _resolve_metric(parameters, "selection_metric", "f1_macro")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight=class_weight, random_state=42, n_jobs=-1
        ),
        **(
            {
                "BalancedRandomForest": BalancedRandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                )
            }
            if BalancedRandomForestClassifier is not None
            else {}
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
            random_state=42, n_jobs=-1, class_weight=class_weight, verbose=-1,
        ),
    }

    results = {}
    best_score = -1.0
    best_model = None
    best_name = ""

    for name, model in models.items():
        fitted_model = _wrap_with_sampler(model, sampler)
        fitted_model.fit(X_train, y_train)
        preds = fitted_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_w = f1_score(y_test, preds, average="weighted")
        f1_m = f1_score(y_test, preds, average="macro")

        results[name] = {
            "accuracy": acc,
            "f1_weighted": f1_w,
            "f1_macro": f1_m,
            "selection_metric": selection_metric,
        }
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        print(classification_report(y_test, preds, zero_division=0))

        _log_to_mlflow(
            "crash-severity-comparison",
            run_name=name,
            params={"model_type": name, "selection_metric": selection_metric},
            metrics={"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m},
        )

        score_for_selection = {
            "accuracy": acc,
            "f1_weighted": f1_w,
            "f1_macro": f1_m,
        }[selection_metric]

        if score_for_selection > best_score:
            best_score = score_for_selection
            best_model = fitted_model
            best_name = name

    print(f"\nBest model: {best_name} ({selection_metric}: {best_score:.4f})")
    results["best_model_name"] = best_name
    results["selection_metric"] = selection_metric

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

    selection_metric = _resolve_metric(parameters, "selection_metric", "f1_macro")
    class_weight = _resolve_class_weight(y_train, parameters)
    sampler = _build_sampler(y_train, parameters)
    base_estimator = RandomForestClassifier(
        class_weight=class_weight, random_state=42, n_jobs=-1
    )
    estimator = _wrap_with_sampler(base_estimator, sampler)
    grid_params = _prefix_model_params(
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
        },
        sampler,
    )

    # Grid Search - maly grid (3*3 = 9 kombinacji), zeby trening sie nie ciagnal
    grid = GridSearchCV(
        estimator,
        grid_params,
        cv=3,
        scoring=selection_metric,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    grid_preds = grid.predict(X_test)
    grid_metrics = {
        "grid_f1_weighted": f1_score(y_test, grid_preds, average="weighted"),
        "grid_f1_macro": f1_score(y_test, grid_preds, average="macro"),
        "grid_accuracy": accuracy_score(y_test, grid_preds),
    }
    print(f"\n[Grid Search] Best params: {grid.best_params_}")
    print(f"[Grid Search] F1 ważone: {grid_metrics['grid_f1_weighted']:.4f}")

    _log_to_mlflow(
        "crash-severity-tuning",
        run_name="grid_search",
        params={**grid.best_params_, "search_type": "GridSearchCV"},
        metrics=grid_metrics,
    )

    # Random Search - wiekszy parameter space, 15 losowych iteracji
    random_param_dist = _prefix_model_params(
        {
            "n_estimators": [100, 150, 200, 250, 300, 400, 500],
            "max_depth": [5, 10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
        },
        sampler,
    )
    random_search = RandomizedSearchCV(
        estimator,
        random_param_dist,
        n_iter=15,
        cv=3,
        scoring=selection_metric,
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
    print(f"\n[Random Search] Best params: {random_search.best_params_}")
    print(f"[Random Search] F1 ważone: {rs_metrics['random_f1_weighted']:.4f}")

    _log_to_mlflow(
        "crash-severity-tuning",
        run_name="random_search",
        params={**random_search.best_params_, "search_type": "RandomizedSearchCV"},
        metrics=rs_metrics,
    )

    # Wybor lepszego modelu
    if rs_metrics[f"random_{selection_metric}"] > grid_metrics[f"grid_{selection_metric}"]:
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
        "selection_metric": selection_metric,
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

    class_weight = _resolve_class_weight(y_train, parameters)
    sampler = _build_sampler(y_train, parameters)
    optimization_metric = _resolve_metric(parameters, "optimization_metric", "f1_macro")
    cv_folds = int(parameters.get("cv_folds", 3))
    n_trials = int(parameters.get("n_trials", 50))
    show_progress_bar = bool(parameters.get("show_progress_bar", True))

    def objective(trial):
        trial_params = dict(parameters)
        trial_params["class_weight_strategy"] = "balanced_plus"
        trial_params["class_weight_multipliers"] = {
            "NO_INJURY": 1.0,
            "MINOR": trial.suggest_float("minor_weight_multiplier", 1.0, 3.0),
            "SERIOUS": trial.suggest_float(
                "serious_weight_multiplier", 2.0, 10.0, log=True
            ),
        }
        trial_params["sampling_strategy"] = trial.suggest_categorical(
            "sampling_strategy", ["none", "undersample"]
        )
        trial_params["sampling_ratio"] = trial.suggest_float(
            "sampling_ratio", 0.10, 0.40, step=0.05
        )

        trial_class_weight = _resolve_class_weight(y_train, trial_params)
        trial_sampler = _build_sampler(y_train, trial_params)
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
            **params,
            random_state=42,
            n_jobs=-1,
            class_weight=trial_class_weight,
            verbose=-1,
        )
        estimator = _wrap_with_sampler(model, trial_sampler)
        score = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv_folds,
            scoring=optimization_metric,
            n_jobs=-1,
        )
        return score.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    print(f"\nBest trial {optimization_metric}: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_trial_params = dict(parameters)
    best_trial_params["class_weight_strategy"] = "balanced_plus"
    best_trial_params["class_weight_multipliers"] = {
        "NO_INJURY": 1.0,
        "MINOR": study.best_params.get("minor_weight_multiplier", 1.5),
        "SERIOUS": study.best_params.get("serious_weight_multiplier", 3.0),
    }
    best_trial_params["sampling_strategy"] = study.best_params.get(
        "sampling_strategy", parameters.get("sampling_strategy", "none")
    )
    best_trial_params["sampling_ratio"] = study.best_params.get(
        "sampling_ratio", parameters.get("sampling_ratio", 0.2)
    )

    best_class_weight = _resolve_class_weight(y_train, best_trial_params)
    best_sampler = _build_sampler(y_train, best_trial_params)

    best_model = LGBMClassifier(
        **study.best_params,
        random_state=42,
        n_jobs=-1,
        class_weight=best_class_weight,
        verbose=-1,
    )
    best_model = _wrap_with_sampler(best_model, best_sampler)
    best_model.fit(X_train, y_train)

    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_w = f1_score(y_test, preds, average="weighted")
    f1_m = f1_score(y_test, preds, average="macro")

    print("\nOptuna-tuned LightGBM:")
    print(classification_report(y_test, preds, zero_division=0))

    metrics = {
        "optuna_accuracy": acc,
        "optuna_f1_weighted": f1_w,
        "optuna_f1_macro": f1_m,
        "best_params": str(study.best_params),
        "n_trials": len(study.trials),
        "selection_metric": optimization_metric,
        "sampling_strategy": str(best_trial_params.get("sampling_strategy", "none")),
    }

    _log_to_mlflow(
        "crash-severity-tuning",
        run_name="optuna_bayesian",
        params={
            **study.best_params,
            "search_type": "Optuna_Bayesian",
            "n_trials": len(study.trials),
            "selection_metric": optimization_metric,
            "sampling_strategy": str(best_trial_params.get("sampling_strategy", "none")),
        },
        metrics={"f1_weighted": f1_w, "f1_macro": f1_m, "accuracy": acc},
    )

    return best_model, metrics

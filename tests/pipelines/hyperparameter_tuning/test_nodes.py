"""Testy dla pipeline'u strojenia hiperparametrow."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from src.crash_kedro.pipelines.hyperparameter_tuning.nodes import bayesian_tuning, _build_sampler  # noqa: E402


def _sample_df() -> pd.DataFrame:
    rows = []
    labels = ["NO_INJURY"] * 12 + ["MINOR"] * 6 + ["SERIOUS"] * 3
    for i, label in enumerate(labels):
        rows.append(
            {
                "feat_a": i,
                "feat_b": i % 4,
                "feat_c": (i * 3) % 7,
                "Severity_Group": label,
            }
        )
    return pd.DataFrame(rows)


def test_bayesian_tuning_returns_metrics_for_macro_optimization():
    """bayesian_tuning powinien zwracac model i metryki dla F1 macro."""
    df = _sample_df()
    params = {
        "test_size": 0.25,
        "random_state": 42,
        "optimization_metric": "f1_macro",
        "selection_metric": "f1_macro",
        "class_weight_strategy": "balanced_plus",
        "class_weight_multipliers": {
            "NO_INJURY": 1.0,
            "MINOR": 1.5,
            "SERIOUS": 3.0,
        },
        "sampling_strategy": "undersample",
        "sampling_ratio": 0.25,
        "cv_folds": 2,
        "n_trials": 1,
        "show_progress_bar": False,
    }

    model, metrics = bayesian_tuning(df, params)

    assert hasattr(model, "predict")
    assert metrics["selection_metric"] == "f1_macro"
    assert metrics["n_trials"] == 1
    assert 0.0 <= metrics["optuna_f1_macro"] <= 1.0
    assert 0.0 <= metrics["optuna_f1_weighted"] <= 1.0
    assert 0.0 <= metrics["optuna_accuracy"] <= 1.0


def test_build_sampler_accepts_numpy_encoded_labels():
    """_build_sampler should handle numpy.ndarray labels after encoding."""
    y_train = np.array([0, 0, 0, 1, 1, 2])

    sampler = _build_sampler(
        y_train,
        {
            "sampling_strategy": "undersample",
            "sampling_ratio": 0.5,
            "random_state": 42,
        },
    )

    assert sampler is None or hasattr(sampler, "fit") or hasattr(sampler, "func")



"""Testy wezlow pipeline'u modelowania."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from crash_kedro.pipelines.data_modeling.nodes import (  # noqa: E402
    evaluate_model,
    split_data,
    train_model,
)


def _sample_df() -> pd.DataFrame:
    """Maly, zbalansowany zbior do testow."""
    rows = []
    for i in range(30):
        group = ["NO_INJURY", "MINOR", "SERIOUS"][i % 3]
        rows.append({"feat_a": i, "feat_b": i % 5, "Severity_Group": group})
    return pd.DataFrame(rows)


def test_split_data_respects_test_size():
    """split_data powinien dzielic dane wg test_size."""
    df = _sample_df()
    params = {"test_size": 0.2, "random_state": 42}

    X_train, X_test, y_train, y_test = split_data(df, params)

    assert len(X_test) == 6  # 20% z 30
    assert len(X_train) == 24
    assert "Severity_Group" not in X_train.columns


def test_train_and_evaluate_returns_metrics():
    """Trening + ewaluacja powinny zwrocic sensowne metryki."""
    df = _sample_df()
    params = {"test_size": 0.2, "random_state": 42}
    X_train, X_test, y_train, y_test = split_data(df, params)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics) == {"accuracy", "f1_weighted", "f1_macro"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0

"""Pipeline AutoML z Autogluonem."""

from kedro.pipeline import Pipeline, node

from .nodes import run_autogluon


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_autogluon,
                inputs=["crash_features", "parameters"],
                outputs=["autogluon_predictor", "autogluon_metrics"],
                name="run_autogluon_node",
            ),
        ]
    )

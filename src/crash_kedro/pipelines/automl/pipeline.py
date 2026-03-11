"""AutoML pipeline definition."""

from kedro.pipeline import Pipeline, node

from .nodes import run_automl


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_automl,
                inputs=["crash_features", "parameters"],
                outputs=["automl_model", "automl_metrics"],
                name="run_automl_node",
            ),
        ]
    )

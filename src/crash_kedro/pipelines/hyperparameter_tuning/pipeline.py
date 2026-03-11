"""Hyperparameter tuning pipeline definition."""

from kedro.pipeline import Pipeline, node

from .nodes import bayesian_tuning, compare_models


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=compare_models,
                inputs=["crash_features", "parameters"],
                outputs=["best_comparison_model", "comparison_metrics"],
                name="compare_models_node",
            ),
            node(
                func=bayesian_tuning,
                inputs=["crash_features", "parameters"],
                outputs=["tuned_model", "tuning_metrics"],
                name="bayesian_tuning_node",
            ),
        ]
    )

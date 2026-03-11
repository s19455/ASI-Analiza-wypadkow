"""
Pipeline modelowania danych.
"""

from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["crash_features", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ]
    )

"""
Pipeline przygotowania danych.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    clean_missing_values,
    drop_unnecessary_columns,
    encode_features,
    engineer_features,
    map_target,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=drop_unnecessary_columns,
                inputs=["crash_raw", "parameters"],
                outputs="columns_dropped",
                name="drop_unnecessary_columns_node",
            ),
            node(
                func=clean_missing_values,
                inputs="columns_dropped",
                outputs="cleaned",
                name="clean_missing_values_node",
            ),
            node(
                func=engineer_features,
                inputs="cleaned",
                outputs="engineered",
                name="engineer_features_node",
            ),
            node(
                func=map_target,
                inputs=["engineered", "parameters"],
                outputs="target_mapped",
                name="map_target_node",
            ),
            node(
                func=encode_features,
                inputs="target_mapped",
                outputs="crash_features",
                name="encode_features_node",
            ),
        ]
    )

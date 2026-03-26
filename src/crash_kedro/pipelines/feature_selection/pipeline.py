"""Pipeline selekcji cech."""

from kedro.pipeline import Pipeline, node

from .nodes import select_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=select_features,
                inputs=["crash_features", "parameters"],
                outputs=["selected_features", "feature_selection_metrics"],
                name="select_features_node",
            ),
        ]
    )

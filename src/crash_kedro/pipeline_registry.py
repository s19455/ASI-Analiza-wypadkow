"""Project pipelines."""

from crash_kedro.pipelines.data_modeling import create_pipeline as mp
from crash_kedro.pipelines.data_preparation import create_pipeline as dp


def register_pipelines():
    pipelines = {
        "__default__": dp() + mp(),
        "data_processing": dp(),
        "modeling": mp(),
    }

    try:
        from crash_kedro.pipelines.automl import (  # noqa: PLC0415
            create_pipeline as automl,
        )

        pipelines["automl"] = dp() + automl()
    except ImportError:
        pass

    try:
        from crash_kedro.pipelines.hyperparameter_tuning import (  # noqa: PLC0415
            create_pipeline as tuning,
        )

        pipelines["tuning"] = dp() + tuning()
    except ImportError:
        pass

    try:
        from crash_kedro.pipelines.autogluon import (  # noqa: PLC0415
            create_pipeline as autogluon,
        )

        pipelines["autogluon"] = dp() + autogluon()
    except ImportError:
        pass

    try:
        from crash_kedro.pipelines.feature_selection import (  # noqa: PLC0415
            create_pipeline as fs,
        )

        pipelines["feature_selection"] = dp() + fs()
    except ImportError:
        pass

    return pipelines

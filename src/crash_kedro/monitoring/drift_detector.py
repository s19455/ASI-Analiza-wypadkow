"""Data drift detection using Evidently."""

import pandas as pd


def detect_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        report.save_html("data/08_reporting/drift_report.html")

        result = report.as_dict()
        return {
            "drift_detected": result["metrics"][0]["result"]["dataset_drift"],
            "drift_share": result["metrics"][0]["result"]["drift_share"],
            "report_path": "data/08_reporting/drift_report.html",
        }
    except ImportError:
        return {
            "error": "evidently not installed. Run: pip install evidently",
            "drift_detected": None,
        }

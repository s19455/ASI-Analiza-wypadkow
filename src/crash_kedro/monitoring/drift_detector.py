"""Wykrywanie driftu danych przy uzyciu Evidently."""

import pandas as pd


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_path: str = "data/08_reporting/drift_report.html",
) -> dict:
    """Porownuje dane referencyjne z biezacymi i zapisuje raport HTML."""
    try:
        # nowsze evidently (0.5+) ma legacy API pod evidently.legacy.*
        try:
            from evidently.legacy.metric_preset import DataDriftPreset
            from evidently.legacy.report import Report
        except ImportError:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        report.save_html(report_path)

        result = report.as_dict()
        return {
            "drift_detected": result["metrics"][0]["result"]["dataset_drift"],
            "drift_share": result["metrics"][0]["result"]["drift_share"],
            "report_path": report_path,
        }
    except ImportError:
        return {
            "error": "evidently not installed. Run: pip install evidently",
            "drift_detected": None,
        }

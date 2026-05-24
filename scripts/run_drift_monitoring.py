"""CLI helper for running drift monitoring with Evidently.

The script can either compare two explicitly provided datasets or, by default,
derive a reference/current split from the project feature dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from crash_kedro.monitoring.drift_detector import detect_drift

DEFAULT_REFERENCE = PROJECT_ROOT / "data" / "03_primary" / "crash_features.parquet"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "08_reporting"
DEFAULT_REPORT_JSON = DEFAULT_REPORT_DIR / "drift_summary.json"
DEFAULT_REPORT_HTML = DEFAULT_REPORT_DIR / "drift_report.html"


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe from CSV or Parquet based on file suffix."""

    if not path.exists():
        raise FileNotFoundError(f"Plik nie istnieje: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Nieobsługiwany format pliku: {path.suffix}")


def _derive_reference_current(
    df: pd.DataFrame,
    min_rows: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split one dataframe into reference and current slices.

    Preference is given to a time-based split using ``crash_year``. If that is
    not possible, the function falls back to a deterministic 80/20 split.
    """

    if "crash_year" in df.columns:
        years = sorted(y for y in df["crash_year"].dropna().unique())
        if len(years) >= 2:
            latest_year = years[-1]
            reference = df[df["crash_year"] < latest_year].copy()
            current = df[df["crash_year"] >= latest_year].copy()
            if len(reference) >= min_rows and len(current) >= min_rows:
                return reference, current, f"year_split(<{latest_year} / >= {latest_year})"

    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if len(shuffled) < 2:
        raise ValueError("Zbyt mało wierszy do wykonania monitoringu driftu.")

    split_idx = max(int(len(shuffled) * 0.8), 1)
    if split_idx >= len(shuffled):
        split_idx = len(shuffled) - 1

    reference = shuffled.iloc[:split_idx].copy()
    current = shuffled.iloc[split_idx:].copy()
    if len(current) == 0:
        current = shuffled.iloc[-1:].copy()
        reference = shuffled.iloc[:-1].copy()

    return reference, current, "random_split(80/20)"


def _align_columns(reference: pd.DataFrame, current: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = [column for column in reference.columns if column in current.columns]
    if not common:
        raise ValueError("Reference i current nie mają wspólnych kolumn do porównania.")

    return reference.loc[:, common].copy(), current.loc[:, common].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drift monitoring and save a report.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Plik referencyjny. Domyślnie data/03_primary/crash_features.parquet.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=None,
        help="Opcjonalny plik z bieżącymi danymi. Jeśli niepodany, skrypt sam wykona split.",
    )
    parser.add_argument(
        "--report-html",
        type=Path,
        default=DEFAULT_REPORT_HTML,
        help="Ścieżka do HTML z raportem Evidently.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REPORT_JSON,
        help="Ścieżka do JSON z podsumowaniem monitoringu.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_html.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)

    reference_df = _load_dataframe(args.reference)
    if args.current is not None:
        current_df = _load_dataframe(args.current)
        split_strategy = f"explicit({args.reference.name} vs {args.current.name})"
    else:
        reference_df, current_df, split_strategy = _derive_reference_current(reference_df)

    reference_df, current_df = _align_columns(reference_df, current_df)

    summary = detect_drift(reference_df, current_df, report_path=str(args.report_html))
    summary["split_strategy"] = split_strategy
    summary["reference_rows"] = int(len(reference_df))
    summary["current_rows"] = int(len(current_df))
    summary["reference_columns"] = list(reference_df.columns)
    summary["current_columns"] = list(current_df.columns)

    if not args.report_html.exists():
        args.report_html.write_text(
            "<html><body><h1>Drift monitoring</h1><pre>"
            + json.dumps(summary, indent=2, ensure_ascii=False)
            + "</pre></body></html>",
            encoding="utf-8",
        )

    args.report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Monitoring driftu zakończony.")
    print(f"- Strategia podziału: {split_strategy}")
    print(f"- Raport HTML: {args.report_html}")
    print(f"- Podsumowanie JSON: {args.report_json}")
    if "drift_detected" in summary:
        print(f"- Drift detected: {summary['drift_detected']}")
    if "drift_share" in summary:
        print(f"- Drift share: {summary['drift_share']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())




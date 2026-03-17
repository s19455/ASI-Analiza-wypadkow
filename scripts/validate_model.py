"""Model validation script - quality gate for CI/CD."""

import json
import sys
from pathlib import Path


def validate():
    metrics_path = Path("data/08_reporting/metrics.json")
    if not metrics_path.exists():
        print("ERROR: metrics.json not found")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    f1_weighted = metrics.get("f1_weighted", 0)
    min_threshold = 0.70

    print(f"Model F1 weighted: {f1_weighted:.4f}")
    print(f"Minimum threshold: {min_threshold:.4f}")

    if f1_weighted < min_threshold:
        print(f"FAIL: F1 weighted {f1_weighted:.4f} < {min_threshold:.4f}")
        sys.exit(1)

    print("PASS: Model meets quality threshold")


if __name__ == "__main__":
    validate()

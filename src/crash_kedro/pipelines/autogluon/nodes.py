"""Pipeline AutoML z Autogluonem - automatyczny trening ensembli."""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


def run_autogluon(df: pd.DataFrame, parameters: dict):
    """Trenowanie modelu autogluon na danych - autostacking i auto-ensembling."""
    from autogluon.tabular import TabularPredictor

    train, test = train_test_split(
        df,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=df["Severity_Group"],
    )

    predictor = TabularPredictor(
        label="Severity_Group",
        eval_metric="f1_weighted",
        path="data/06_models/autogluon/",
    ).fit(
        train_data=train,
        time_limit=600,
        presets="medium_quality",
        verbosity=2,
    )

    y_test = test["Severity_Group"]
    preds = predictor.predict(test.drop("Severity_Group", axis=1))

    acc = accuracy_score(y_test, preds)
    f1_w = f1_score(y_test, preds, average="weighted")
    f1_m = f1_score(y_test, preds, average="macro")

    print("\n" + "=" * 60)
    print("AUTOGLUON - wyniki na zbiorze testowym")
    print("=" * 60)
    print(classification_report(y_test, preds))

    leaderboard = predictor.leaderboard(test, silent=True)

    metrics = {
        "autogluon_accuracy": acc,
        "autogluon_f1_weighted": f1_w,
        "autogluon_f1_macro": f1_m,
        "best_model": str(leaderboard.iloc[0]["model"]) if len(leaderboard) > 0 else "unknown",
        "models_trained": len(leaderboard),
        "leaderboard_top": leaderboard.head(10).to_dict(orient="records"),
    }

    return predictor, metrics

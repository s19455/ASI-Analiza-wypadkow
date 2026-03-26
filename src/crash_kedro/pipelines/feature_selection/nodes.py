"""Selekcja cech - SelectKBest + porownanie modelu na pelnym vs. zredukowanym zbiorze cech."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def select_features(df: pd.DataFrame, parameters: dict):
    """Wybor top-k cech metoda mutual_info_classif (informacja wzajemna)."""
    X = df.drop("Severity_Group", axis=1)
    y = df["Severity_Group"]

    # liczba cech do wybrania (default: 20 z ~39)
    k = min(20, X.shape[1])

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X, y)

    selected_cols = X.columns[selector.get_support()].tolist()
    scores = dict(zip(X.columns, selector.scores_, strict=False))
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nWybranych cech: {len(selected_cols)} / {X.shape[1]}")  # noqa: T201
    print("\nTop 10 cech wedlug informacji wzajemnej:")  # noqa: T201
    for feat, score in sorted_scores[:10]:
        print(f"  {feat:35s} = {score:.4f}")  # noqa: T201

    # Trening Random Forest na pelnych vs. zredukowanych cechach
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=y,
    )

    rf_full = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_full.fit(X_train, y_train)
    full_preds = rf_full.predict(X_test)
    full_f1 = f1_score(y_test, full_preds, average="weighted")
    full_acc = accuracy_score(y_test, full_preds)

    rf_sel = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_sel.fit(X_train[selected_cols], y_train)
    sel_preds = rf_sel.predict(X_test[selected_cols])
    sel_f1 = f1_score(y_test, sel_preds, average="weighted")
    sel_acc = accuracy_score(y_test, sel_preds)

    print("\n=== Porownanie cech ===")  # noqa: T201
    print(f"Pelne {X.shape[1]} cech : F1w = {full_f1:.4f}, acc = {full_acc:.4f}")  # noqa: T201
    print(f"Zredukowane {k} cech : F1w = {sel_f1:.4f}, acc = {sel_acc:.4f}")  # noqa: T201

    # Logowanie do MLflow
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_experiment("crash-severity-feature-selection")
        with mlflow.start_run(run_name="select_kbest_mutual_info"):
            mlflow.log_params({"k": k, "method": "mutual_info_classif", "n_total_features": X.shape[1]})
            mlflow.log_metrics({
                "full_f1_weighted": full_f1,
                "full_accuracy": full_acc,
                "selected_f1_weighted": sel_f1,
                "selected_accuracy": sel_acc,
            })
    except Exception as e:
        print(f"[MLflow] Pominieto: {e}")  # noqa: T201

    selected_df = df[selected_cols + ["Severity_Group"]].copy()

    metrics = {
        "n_features_total": X.shape[1],
        "n_features_selected": k,
        "selected_features": selected_cols,
        "top_10_scores": [{"feature": f, "score": float(s)} for f, s in sorted_scores[:10]],
        "full_features_f1_weighted": float(full_f1),
        "selected_features_f1_weighted": float(sel_f1),
        "full_features_accuracy": float(full_acc),
        "selected_features_accuracy": float(sel_acc),
    }

    return selected_df, metrics

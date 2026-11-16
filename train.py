"""Trains baseline model v1 and registers it in MLflow as the champion."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import registry
from make_data import TARGET, TRAIN

MODEL_PARAMS = {
    "n_estimators": 200, "max_depth": 8, "class_weight": "balanced", "random_state": 42,
}


def build_model():
    return RandomForestClassifier(**MODEL_PARAMS, n_jobs=-1)


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4),
    }


def register_v1(model, X_train, metrics, params=None, root="."):
    """Registers a fitted model as v1, the first champion."""
    return registry.register_version(
        model,
        X_train,
        {
            "trigger": "initial training",
            "train_rows": int(len(X_train)),
            "evaluated_on": "original test split",
            "metrics": metrics,
        },
        params={"model_type": type(model).__name__, **(params or {}), "train_rows": len(X_train)},
        promote=True,
        root=root,
        run_name="v1 initial training",
    )


if __name__ == "__main__":
    df = pd.read_csv(TRAIN)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model().fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    registry.wipe()  # a fresh v1 means any old runs and versions are stale
    register_v1(model, X_train, metrics, MODEL_PARAMS)
    print("Registered model v1 in MLflow:", metrics)

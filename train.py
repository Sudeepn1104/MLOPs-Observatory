"""Trains baseline model v1 and saves model + reference data."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from registry import reset_registry

TARGET = "default"

df = pd.read_csv("data/loan.csv")
X, y = df.drop(columns=[TARGET]), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "auc": round(roc_auc_score(y_test, proba), 4),
}

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model_v1.joblib")
X_train.to_csv("models/reference_data.csv", index=False)  # drift baseline
Path("models/model_v1_metrics.json").write_text(json.dumps(metrics, indent=2))
reset_registry()  # a fresh v1 means any old v2, v3... are stale
print("Saved model v1:", metrics)

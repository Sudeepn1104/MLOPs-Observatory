"""Simulates incoming loan batches with growing drift and logs monitoring results."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from drift import check_drift
from make_data import make_loans

TARGET = "default"
N_BATCHES = 10
BATCH_SIZE = 500

model = joblib.load("models/model_v1.joblib")
ref = pd.read_csv("models/reference_data.csv")

Path("logs").mkdir(exist_ok=True)
rows, feature_rows = [], []

for i in range(N_BATCHES):
    # drift grows with each batch: incomes fall, borrowers get older
    batch = make_loans(BATCH_SIZE, income_shift=-0.06 * i, age_shift=1.5 * i)
    X, y = batch.drop(columns=[TARGET]), batch[TARGET]

    report = check_drift(ref, X)
    report.insert(0, "batch", i)
    feature_rows.append(report)

    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    drifted = report.loc[report["status"] == "red", "feature"].tolist()
    rows.append(
        {
            "batch": i,
            "accuracy": round(accuracy_score(y, pred), 4),
            "auc": round(roc_auc_score(y, proba), 4),
            "mean_pred": round(float(proba.mean()), 4),
            "max_psi": report["psi"].max(),
            "red_features": ",".join(drifted),
        }
    )

log = pd.DataFrame(rows)
log.to_csv("logs/batch_log.csv", index=False)
pd.concat(feature_rows).to_csv("logs/feature_drift.csv", index=False)
print(log.to_string(index=False))

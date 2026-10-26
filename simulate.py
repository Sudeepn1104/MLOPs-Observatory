"""Simulates incoming loan batches with growing drift and logs monitoring results.

Input drift starts at batch 0 (incomes fall, borrowers get older).
Concept drift (the default rule changes) starts at CONCEPT_START and ramps up,
so accuracy and AUC visibly drop in the later batches.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from drift import check_drift
from make_data import make_loans

TARGET = "default"
N_BATCHES = 10
BATCH_SIZE = 500
CONCEPT_START = 4  # first batch where the default rule starts changing


def concept_level(i):
    """0 before CONCEPT_START, then ramps linearly up to 1 at the last batch."""
    if i < CONCEPT_START:
        return 0.0
    return (i - CONCEPT_START + 1) / (N_BATCHES - CONCEPT_START)


if __name__ == "__main__":
    model = joblib.load("models/model_v1.joblib")
    ref = pd.read_csv("models/reference_data.csv")

    Path("logs").mkdir(exist_ok=True)
    rows, feature_rows = [], []

    for i in range(N_BATCHES):
        c = concept_level(i)
        batch = make_loans(
            BATCH_SIZE, income_shift=-0.06 * i, age_shift=1.5 * i, concept=c
        )
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
                "concept": round(c, 3),
                "accuracy": round(accuracy_score(y, pred), 4),
                "auc": round(roc_auc_score(y, proba), 4),
                "f1": round(f1_score(y, pred, zero_division=0), 4),
                "recall": round(recall_score(y, pred, zero_division=0), 4),
                "default_rate": round(float(y.mean()), 4),
                "mean_pred": round(float(proba.mean()), 4),
                "max_psi": report["psi"].max(),
                "red_features": ",".join(drifted),
            }
        )

    log = pd.DataFrame(rows)
    log.to_csv("logs/batch_log.csv", index=False)
    pd.concat(feature_rows).to_csv("logs/feature_drift.csv", index=False)
    print(log.to_string(index=False))

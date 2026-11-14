"""Simulates incoming loan batches with growing drift and logs monitoring results.

Batches are real clients from data/stream_pool.csv (never used for training).
Input drift starts at batch 0 (lower credit limits, older clients).
Concept drift (the default rule changes) starts at CONCEPT_START and ramps up,
so accuracy and AUC visibly drop in the later batches.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from drift import check_drift
from make_data import TARGET, make_batch

N_BATCHES = 10
BATCH_SIZE = 500
CONCEPT_START = 4  # first batch where the default rule starts changing
SIM_SEED = 7  # separate from the training seed so batches never replay training draws


def concept_level(i):
    """0 before CONCEPT_START, then ramps linearly up to 1 at the last batch."""
    if i < CONCEPT_START:
        return 0.0
    return (i - CONCEPT_START + 1) / (N_BATCHES - CONCEPT_START)


def run(model_path="models/model_v1.joblib", ref_path="models/reference_data.csv",
        out_dir="logs", seed=SIM_SEED):
    """Scores N_BATCHES simulated batches and writes the three monitoring logs."""
    model = joblib.load(model_path)
    ref = pd.read_csv(ref_path)
    rng = np.random.default_rng(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, feature_rows, all_batches = [], [], []

    for i in range(N_BATCHES):
        c = concept_level(i)
        batch = make_batch(
            BATCH_SIZE, limit_tilt=-0.12 * i, age_tilt=0.12 * i, concept=c, rng=rng
        )
        X, y = batch.drop(columns=[TARGET]), batch[TARGET]
        all_batches.append(batch.assign(batch=i))  # labeled data, used for retraining

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
    log.to_csv(out / "batch_log.csv", index=False)
    pd.concat(feature_rows).to_csv(out / "feature_drift.csv", index=False)
    pd.concat(all_batches).to_csv(out / "batches.csv", index=False)
    return log


if __name__ == "__main__":
    print(run().to_string(index=False))

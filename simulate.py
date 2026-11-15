"""Simulates incoming credit batches with growing drift and logs monitoring results.

Batches are real clients from data/stream_pool.csv (never used for training).
Input drift starts at batch 0 (lower credit limits, older clients).
Concept drift (the default rule changes) starts at CONCEPT_START and ramps up,
so accuracy and AUC visibly drop in the later batches. After the last
scenario batch the world stays drifted: later batches (streamed from the
dashboard) keep the batch-9 conditions.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

import registry
from drift import check_drift
from make_data import TARGET, make_batch

N_BATCHES = 10
BATCH_SIZE = 500
CONCEPT_START = 4  # first batch where the default rule starts changing
SIM_SEED = 7  # separate from the split seed so batches never replay training draws


def concept_level(i):
    """0 before CONCEPT_START, then ramps linearly up to 1 at the last batch."""
    if i < CONCEPT_START:
        return 0.0
    return min(1.0, (i - CONCEPT_START + 1) / (N_BATCHES - CONCEPT_START))


def scenario(i):
    """Drift settings for batch i: (limit_tilt, age_tilt, concept)."""
    k = min(i, N_BATCHES - 1)
    return -0.12 * k, 0.12 * k, concept_level(k)


def draw_batch(i, rng=None):
    """The labelled batch number i of the scenario.

    With no rng, each batch has its own fixed random stream, so batch 12 is the
    same batch no matter who streams it or in which session.
    """
    limit_tilt, age_tilt, concept = scenario(i)
    rng = np.random.default_rng([SIM_SEED, i]) if rng is None else rng
    batch = make_batch(BATCH_SIZE, limit_tilt=limit_tilt, age_tilt=age_tilt, concept=concept, rng=rng)
    return batch, concept


def score_batch(model, ref, batch, i, version, concept):
    """Drift report and performance of one model on one labelled batch.

    Returns (log_row, feature_rows): one summary row for batch_log.csv and one
    row per feature for feature_drift.csv.
    """
    X, y = batch.drop(columns=[TARGET]), batch[TARGET]
    report = check_drift(ref, X)
    report.insert(0, "batch", i)

    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    drifted = report.loc[report["status"] == "red", "feature"].tolist()
    row = {
        "batch": i,
        "model": version,
        "concept": round(concept, 3),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "default_rate": round(float(y.mean()), 4),
        "mean_pred": round(float(proba.mean()), 4),
        "max_psi": float(report["psi"].max()),
        "red_features": ",".join(drifted),
    }
    return row, report


def run(out_dir="logs", seed=SIM_SEED):
    """Scores N_BATCHES simulated batches with v1 and writes the three monitoring logs."""
    v1 = registry.load_registry()["models"][0]
    model = registry.load_model(v1)
    ref = registry.load_reference(v1)
    rng = np.random.default_rng(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, feature_rows, all_batches = [], [], []

    for i in range(N_BATCHES):
        batch, concept = draw_batch(i, rng)
        all_batches.append(batch.assign(batch=i))  # labeled data, used for retraining
        row, report = score_batch(model, ref, batch, i, "v1", concept)
        rows.append(row)
        feature_rows.append(report)

    log = pd.DataFrame(rows)
    log.to_csv(out / "batch_log.csv", index=False)
    pd.concat(feature_rows).to_csv(out / "feature_drift.csv", index=False)
    pd.concat(all_batches).to_csv(out / "batches.csv", index=False)
    return log


if __name__ == "__main__":
    print(run().to_string(index=False))

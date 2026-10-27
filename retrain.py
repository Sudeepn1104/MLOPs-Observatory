"""Retraining: train a challenger on recent labeled batches, compare, register.

Champion/challenger rule: the new model is promoted only if its AUC on the
same recent holdout is at least as good as the current active model's.
"""
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from registry import get_active, load_registry, save_registry

TARGET = "default"
RECENT_BATCHES = 4  # how many of the latest batches to learn from

# retrain rule
PSI_RED = 0.2
AUC_DROP = 0.05


def retrain_policy(batch_log, baseline_auc):
    """Returns a list of reasons to retrain (empty list = no retrain needed).

    Rule 1: max PSI of the latest batch is above 0.2 (input drift).
    Rule 2: mean AUC of the last 2 batches fell more than 0.05 below the
            baseline AUC (performance drop, catches concept drift).
    """
    reasons = []
    latest = batch_log.iloc[-1]
    if latest["max_psi"] > PSI_RED:
        reasons.append(f"max PSI {latest['max_psi']:.2f} is above {PSI_RED}")
    recent_auc = batch_log["auc"].tail(2).mean()
    if baseline_auc - recent_auc > AUC_DROP:
        reasons.append(
            f"AUC fell from {baseline_auc:.2f} to {recent_auc:.2f} (limit {AUC_DROP} drop)"
        )
    return reasons


def _metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y, proba), 4),
    }


def retrain(trigger="manual"):
    """Trains a challenger, compares it with the active model, updates the registry.

    Returns the new registry entry (includes the champion-vs-challenger comparison).
    """
    reg = load_registry()
    champion = get_active(reg)
    version = f"v{len(reg['models']) + 1}"

    data = pd.read_csv("logs/batches.csv")
    recent = data[data["batch"] > data["batch"].max() - RECENT_BATCHES]
    X, y = recent.drop(columns=[TARGET, "batch"]), recent[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    challenger = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    challenger.fit(X_tr, y_tr)

    champ_model = joblib.load(champion["file"])
    champ_metrics = _metrics(champ_model, X_te, y_te)
    new_metrics = _metrics(challenger, X_te, y_te)
    promoted = new_metrics["auc"] >= champ_metrics["auc"]

    model_file = f"models/model_{version}.joblib"
    ref_file = f"models/reference_{version}.csv"
    joblib.dump(challenger, model_file)
    X_tr.to_csv(ref_file, index=False)  # new drift baseline for this version

    entry = {
        "version": version,
        "status": "active" if promoted else "rejected",
        "file": model_file,
        "reference": ref_file,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "trigger": trigger,
        "train_rows": int(len(X_tr)),
        "evaluated_on": f"recent holdout ({len(X_te)} rows, last {RECENT_BATCHES} batches)",
        "metrics": new_metrics,
        "compared_with": champion["version"],
        "champion_metrics_on_holdout": champ_metrics,
        "promoted": bool(promoted),
    }
    if promoted:
        champion["status"] = "archived"
        reg["active"] = version
    reg["models"].append(entry)
    save_registry(reg)
    return entry

"""Retraining: train a challenger on recent labeled batches, compare, register.

Champion/challenger rule: the new model is promoted only if its AUC on the
same recent holdout is at least as good as the current active model's.
The retrain rule itself (when to retrain) lives in monitor.py.
"""
from datetime import datetime

import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from make_data import TARGET
from monitor import load_batches
from registry import get_active, load_registry, models_dir, save_registry
from train import build_model

RECENT_BATCHES = 4  # how many of the latest batches to learn from


def _metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4),
    }


def retrain(trigger="manual", root="."):
    """Trains a challenger, compares it with the active model, updates the registry.

    root is the workspace (see registry.py). Returns the new registry entry,
    including the champion-vs-challenger comparison.
    """
    reg = load_registry(root)
    champion = get_active(reg)
    version = f"v{len(reg['models']) + 1}"

    data = load_batches(root)
    last = int(data["batch"].max())
    recent = data[data["batch"] > last - RECENT_BATCHES]
    X, y = recent.drop(columns=[TARGET, "batch"]), recent[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    challenger = build_model().fit(X_tr, y_tr)

    champ_model = joblib.load(champion["file"])
    champ_metrics = _metrics(champ_model, X_te, y_te)
    new_metrics = _metrics(challenger, X_te, y_te)
    promoted = new_metrics["auc"] >= champ_metrics["auc"]

    folder = models_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    model_file = str(folder / f"model_{version}.joblib")
    ref_file = str(folder / f"reference_{version}.csv")
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
        "data_until_batch": last,
        "evaluated_on": f"recent holdout ({len(X_te)} rows, batches {last - RECENT_BATCHES + 1}-{last})",
        "metrics": new_metrics,
        "compared_with": champion["version"],
        "champion_metrics_on_holdout": champ_metrics,
        "promoted": bool(promoted),
    }
    if promoted:
        champion["status"] = "archived"
        reg["active"] = version
    reg["models"].append(entry)
    save_registry(reg, root)
    return entry


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train a challenger on recent batches and compare.")
    p.add_argument("--trigger", default="manual (cli)")
    p.add_argument("--root", default=".")
    args = p.parse_args()
    e = retrain(trigger=args.trigger, root=args.root)
    verdict = "promoted to active" if e["promoted"] else "rejected (not better)"
    print(
        f"{e['version']} {verdict}: AUC {e['metrics']['auc']:.3f} vs "
        f"{e['compared_with']} {e['champion_metrics_on_holdout']['auc']:.3f} on {e['evaluated_on']}"
    )

"""Retraining: train a challenger on recent labeled batches, compare, register.

Champion/challenger rule: the new model is promoted only if its AUC on the
same recent holdout is at least as good as the current active model's.
The retrain rule itself (when to retrain) lives in monitor.py.
"""

from sklearn.model_selection import train_test_split

import registry
from make_data import TARGET
from monitor import load_batches
from train import MODEL_PARAMS, build_model, evaluate

RECENT_BATCHES = 4  # how many of the latest batches to learn from


def retrain(trigger="manual", root="."):
    """Trains a challenger, compares it with the champion, and registers it in MLflow.

    root is the workspace (see registry.py). Returns the new registry entry,
    including the champion-vs-challenger comparison.
    """
    reg = registry.load_registry(root)
    champion = registry.get_active(reg)

    data = load_batches(root)
    last = int(data["batch"].max())
    recent = data[data["batch"] > last - RECENT_BATCHES]
    X, y = recent.drop(columns=[TARGET, "batch"]), recent[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    challenger = build_model().fit(X_tr, y_tr)

    champ_metrics = evaluate(registry.load_model(champion), X_te, y_te)
    new_metrics = evaluate(challenger, X_te, y_te)
    promoted = new_metrics["auc"] >= champ_metrics["auc"]

    # one MLflow run per retrain; the champion alias moves only if promoted
    return registry.register_version(
        challenger,
        X_tr,
        {
            "trigger": trigger,
            "train_rows": int(len(X_tr)),
            "data_until_batch": last,
            "evaluated_on": f"recent holdout ({len(X_te)} rows, batches {last - RECENT_BATCHES + 1}-{last})",
            "metrics": new_metrics,
            "compared_with": champion["version"],
            "champion_metrics_on_holdout": champ_metrics,
            "promoted": bool(promoted),
        },
        params={
            "model_type": type(challenger).__name__, **MODEL_PARAMS,
            "train_rows": len(X_tr), "holdout_rows": len(X_te),
            "batches": f"{last - RECENT_BATCHES + 1}-{last}", "compared_with": champion["version"],
        },
        promote=bool(promoted),
        root=root,
        run_name=f"challenger vs {champion['version']}",
    )


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

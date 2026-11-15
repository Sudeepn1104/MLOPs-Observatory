"""Live monitoring of the active model.

History: logs/*.csv hold the 10 scenario batches scored by v1 (simulate.py).
Stream:  root/logs/stream_*.csv hold batches streamed later from the
         dashboard, each scored by whichever model was active at the time.

The retrain rule judges the active model only on batches it scored itself,
against its own baseline AUC. Right after a promotion the new model has no
batches yet, so there is nothing to judge until the next batch arrives.
"""
from pathlib import Path

import pandas as pd

import registry
from registry import get_active
from simulate import draw_batch, score_batch

HISTORY = Path("logs")
STREAM_FILES = {
    "batch_log": "stream_batch_log.csv",
    "feature_drift": "stream_feature_drift.csv",
    "batches": "stream_batches.csv",
}

# retrain rule
PSI_RED = 0.2
AUC_DROP = 0.05
RULE_WINDOW = 2  # batches averaged for the AUC check


def _stream_path(root, kind):
    return Path(root) / "logs" / STREAM_FILES[kind]


def _read(kind, root):
    hist = pd.read_csv(HISTORY / f"{kind}.csv")
    if kind == "batch_log" and "model" not in hist.columns:
        hist.insert(1, "model", "v1")  # logs written before the model column existed
    stream = _stream_path(root, kind)
    if stream.exists():
        hist = pd.concat([hist, pd.read_csv(stream)], ignore_index=True)
    return hist


def load_logs(root="."):
    """(batch_log, feature_log): history plus this workspace's streamed batches."""
    return _read("batch_log", root), _read("feature_drift", root)


def load_batches(root="."):
    """All labelled batches (history plus stream), with a `batch` column."""
    return _read("batches", root)


def _append(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def stream_next(reg, root="."):
    """Draws the next batch, scores it with the active model and logs it.

    Returns the new batch_log row.
    """
    log, _ = load_logs(root)
    i = int(log["batch"].max()) + 1
    batch, concept = draw_batch(i)
    active = get_active(reg)
    model = registry.load_model(active)
    ref = registry.load_reference(active)
    row, report = score_batch(model, ref, batch, i, active["version"], concept)

    _append(pd.DataFrame([row]), _stream_path(root, "batch_log"))
    _append(report, _stream_path(root, "feature_drift"))
    _append(batch.assign(batch=i), _stream_path(root, "batches"))
    return row


def reset_stream(root="."):
    for kind in STREAM_FILES:
        _stream_path(root, kind).unlink(missing_ok=True)


def evaluate(batch_log, active):
    """Applies the retrain rule to the active model.

    Returns (reasons, own): reasons is a list of rule violations (empty = healthy)
    and own is the part of batch_log scored by the active model.

    Rule 1: max PSI of its latest batch is above PSI_RED (input drift).
    Rule 2: its mean AUC over the last RULE_WINDOW batches fell more than
            AUC_DROP below its own baseline AUC (performance drop, catches
            concept drift).
    """
    own = batch_log[batch_log["model"] == active["version"]]
    if own.empty:
        return [], own
    reasons = []
    latest = own.iloc[-1]
    if latest["max_psi"] > PSI_RED:
        reasons.append(f"max PSI {latest['max_psi']:.2f} is above {PSI_RED}")
    baseline = active["metrics"]["auc"]
    recent = own["auc"].tail(RULE_WINDOW).mean()
    if baseline - recent > AUC_DROP:
        reasons.append(
            f"{active['version']} AUC fell from {baseline:.2f} to {recent:.2f} "
            f"(limit {AUC_DROP} drop)"
        )
    return reasons, own


def can_auto_retrain(reg, batch_log):
    """Auto-retrain at most once per new batch.

    Without this, a rejected challenger would leave the same failing model
    active and auto mode would retrain again on every page refresh.
    """
    latest = int(batch_log["batch"].max())
    last_retrain = max((m.get("data_until_batch", -1) for m in reg["models"]), default=-1)
    return latest > last_retrain


if __name__ == "__main__":
    import argparse

    from registry import load_registry

    p = argparse.ArgumentParser(description="Stream new batches to the active model.")
    p.add_argument("command", choices=["stream"])
    p.add_argument("n", type=int, nargs="?", default=1)
    p.add_argument("--root", default=".")
    args = p.parse_args()
    reg = load_registry(args.root)
    for _ in range(args.n):
        row = stream_next(reg, args.root)
        print(
            f"batch {row['batch']} scored by {row['model']}: AUC {row['auc']:.3f}, "
            f"max PSI {row['max_psi']:.2f}"
        )

"""Evidently drift report: an interactive HTML report of how a batch drifted.

    pip install -r requirements-evidently.txt
    python drift_report.py                      # latest batch vs the champion's training data
    python drift_report.py --batch 9 --log-mlflow

Compares one monitoring batch with the training data of the model in
production (the MLflow champion) using Evidently's DataDriftPreset, and writes:

- an HTML report with per-feature distributions and test results
- a JSON summary (per-feature score, method, drift decision)
- with --log-mlflow, the HTML attached to the champion's MLflow run

Runs in its own environment (requirements-evidently.txt) because Evidently
needs pandas < 3 and is heavy. The app doesn't import this file; its drift
check (drift.py) reproduces the same tests with scipy, and
tests/test_evidently_agreement.py checks both give the same numbers.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# set before Evidently is imported: no usage data sent anywhere
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("EVIDENTLY_DISABLE_TELEMETRY", "1")

import pandas as pd  # noqa: E402

import registry  # noqa: E402
from make_data import CATEGORICAL, NUMERICAL, TARGET  # noqa: E402
from monitor import load_batches  # noqa: E402


def evidently_drift(reference, current):
    """Runs Evidently's DataDriftPreset. Returns (snapshot, per-feature DataFrame)."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    rows = []
    for m in snapshot.dict()["metrics"]:
        name = str(m.get("metric_id") or m.get("metric_name", ""))
        if not name.startswith("ValueDrift(column="):
            continue
        params = dict(p.split("=", 1) for p in name[len("ValueDrift("):-1].split(","))
        score = float(m["value"])
        threshold = float(params["threshold"])
        rows.append({
            "feature": params["column"],
            "method": params["method"],
            "drift_score": round(score, 6),
            "threshold": threshold,
            "drift_detected": bool(score >= threshold),
        })
    return snapshot, pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", type=int, help="batch number (default: the latest)")
    ap.add_argument("--root", default=".", help="workspace (default: project)")
    ap.add_argument("--out", type=Path, default=Path("reports/drift_report.html"))
    ap.add_argument("--json", type=Path, default=Path("reports/drift_summary.json"))
    ap.add_argument("--log-mlflow", action="store_true", help="attach the HTML to the champion's MLflow run")
    args = ap.parse_args(argv)

    champion = registry.get_active(registry.load_registry(args.root))
    reference = registry.load_reference(champion)
    batches = load_batches(args.root)
    batch = int(batches["batch"].max()) if args.batch is None else args.batch
    current = batches[batches["batch"] == batch].drop(columns=[TARGET, "batch"])
    if current.empty:
        print(f"batch {batch} not found", file=sys.stderr)
        return 2

    snapshot, table = evidently_drift(reference, current)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.out))
    summary = {
        "batch": batch,
        "model": champion["version"],
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drifted_features": sorted(table.loc[table["drift_detected"], "feature"]),
        "share_drifted": round(float(table["drift_detected"].mean()), 4),
        "features": table.to_dict(orient="records"),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2))

    if args.log_mlflow and champion.get("run_id"):
        registry.client(args.root).log_artifact(
            champion["run_id"], str(args.out), f"drift_reports/batch_{batch}"
        )

    print(f"Batch {batch} vs {champion['version']} training data: "
          f"{len(summary['drifted_features'])} of {len(table)} features drifted "
          f"({', '.join(summary['drifted_features']) or 'none'})")
    print(f"Report: {args.out}  Summary: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

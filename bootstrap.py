"""Builds data, model v1 and monitoring logs if they are missing.

Used on first start (for example on Streamlit Community Cloud, where the
generated files may not be in the repo). Safe to run locally too.
"""
import subprocess
import sys
from pathlib import Path

NEEDED = [
    "data/train.csv",
    "data/stream_pool.csv",
    "models/mlflow.db",  # the MLflow store holding model v1
    "logs/batch_log.csv",
    "logs/feature_drift.csv",
    "logs/batches.csv",
]


def missing_files():
    return [p for p in NEEDED if not Path(p).exists()]


def build_all():
    for script in ["make_data.py", "train.py", "simulate.py"]:
        subprocess.run([sys.executable, script], check=True, capture_output=True)

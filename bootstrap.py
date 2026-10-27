"""Builds data, model v1 and monitoring logs if they are missing.

Used on first start (for example on Streamlit Community Cloud, where the
generated files may not be in the repo). Safe to run locally too.
"""
import subprocess
import sys
from pathlib import Path

NEEDED = [
    "models/model_v1.joblib",
    "models/reference_data.csv",
    "models/model_v1_metrics.json",
    "logs/batch_log.csv",
    "logs/feature_drift.csv",
    "logs/batches.csv",
]


def missing_files():
    return [p for p in NEEDED if not Path(p).exists()]


def build_all():
    for script in ["make_data.py", "train.py", "simulate.py"]:
        subprocess.run([sys.executable, script], check=True, capture_output=True)

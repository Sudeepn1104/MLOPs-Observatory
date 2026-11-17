"""Builds data, model v1 and monitoring logs if they are missing or out of date.

Used on first start (for example on Streamlit Community Cloud, where the
generated files are not in the repo). Safe to run locally too.

A host can also keep generated files across a code update. If the stored
model was built by older code, running the new code against it fails, so
`stale_store()` detects that and the app rebuilds.
"""
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

STORE = "models/mlflow.db"  # the MLflow store holding model v1
NEEDED = [
    "data/train.csv",
    "data/stream_pool.csv",
    STORE,
    "logs/batch_log.csv",
    "logs/feature_drift.csv",
    "logs/batches.csv",
]


def missing_files():
    return [p for p in NEEDED if not Path(p).exists()]


def stored_model_format(store=STORE):
    """The model_format stamp of the registered v1, or None if there isn't one.

    Reads the SQLite file directly and closes it again. Going through MLflow
    here would cache a connection to a database that the rebuild then replaces.
    """
    if not Path(store).exists():
        return None
    try:
        with closing(sqlite3.connect(f"file:{Path(store).resolve()}?mode=ro", uri=True)) as con:
            row = con.execute(
                "SELECT value FROM model_version_tags "
                "WHERE name = 'credit-default' AND version = 1 AND key = 'model_format'"
            ).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def stale_store():
    """True if a store exists but its v1 was built by a different version of the model code."""
    from model import MODEL_FORMAT

    return Path(STORE).exists() and stored_model_format() != MODEL_FORMAT


def needs_build():
    """Why a build is needed ("" if none): files are missing, or the stored model is out of date."""
    missing = missing_files()
    if missing:
        return f"missing {', '.join(missing)}"
    if stale_store():
        return "the stored model was built by an older version of the code"
    return ""


def build_all():
    # --no-tune: the search already ran offline and its result is model.DEFAULT_PARAMS;
    # a hosted first start should be quick and light, not a hyperparameter search
    for cmd in (["make_data.py"], ["train.py", "--no-tune"], ["simulate.py"]):
        subprocess.run([sys.executable, *cmd], check=True, capture_output=True)

"""Shared fixtures: every test runs in its own temp copy of the project files.

Scripts use relative paths (data/, models/, logs/), so each test chdirs into a
fresh directory and builds a small v1 model there. Nothing in the repo is touched.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OBSERVATORY_SANDBOX", "0")  # tests opt in to sandbox mode explicitly
    for var in ("OBSERVATORY_ENV", "API_KEY", "MAX_REQUEST_BYTES"):
        monkeypatch.delenv(var, raising=False)  # tests opt in to production settings explicitly
    return tmp_path


@pytest.fixture(scope="session")
def empty_mlflow_db(tmp_path_factory):
    """A migrated but empty MLflow database, built once.

    Creating one runs every schema migration (over a second). It holds no
    paths, so each test can start from a copy instead.
    """
    import registry

    root = tmp_path_factory.mktemp("mlflow-template")
    registry.store_dir(root).mkdir()
    registry.client(root).search_registered_models()
    return registry.db_path(root)


@pytest.fixture
def trained(workdir, empty_mlflow_db):
    """A small but real v1: data split, model, reference, metrics and simulated logs."""
    import shutil

    import make_data
    import simulate
    import train

    Path("data").mkdir()
    shutil.copy(ROOT / "data" / "credit_default_raw.csv.gz", "data/")
    make_data.prepare()
    df = pd.read_csv(make_data.TRAIN)
    X, y = df.drop(columns=[make_data.TARGET]), df[make_data.TARGET]
    Path("models").mkdir()
    shutil.copy(empty_mlflow_db, "models/mlflow.db")
    model = RandomForestClassifier(
        n_estimators=30, max_depth=6, class_weight="balanced", random_state=0
    ).fit(X, y)
    train.register_v1(model, X, {"accuracy": 0.77, "f1": 0.5, "auc": 0.76}, {"n_estimators": 30})
    simulate.run()
    return workdir

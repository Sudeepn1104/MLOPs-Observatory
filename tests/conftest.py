"""Shared fixtures: every test runs in its own temp copy of the project files.

Scripts use relative paths (data/, models/, logs/), so each test chdirs into a
fresh directory and builds a small v1 model there. Nothing in the repo is touched.
"""
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import registry

    monkeypatch.setattr(registry, "REGISTRY", Path("models/registry.json"))
    return tmp_path


@pytest.fixture
def trained(workdir):
    """A tiny but real v1: data, model, reference, metrics and simulated logs."""
    import json

    import numpy as np

    import make_data
    import simulate

    rng = np.random.default_rng(0)
    df = make_data.make_loans(3000, rng=rng)
    Path("data").mkdir()
    df.to_csv("data/loan.csv", index=False)
    X, y = df.drop(columns=["default"]), df["default"]
    Path("models").mkdir()
    model = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=0).fit(X, y)
    joblib.dump(model, "models/model_v1.joblib")
    X.to_csv("models/reference_data.csv", index=False)
    Path("models/model_v1_metrics.json").write_text(
        json.dumps({"accuracy": 0.7, "f1": 0.4, "auc": 0.7})
    )
    simulate.run()
    return workdir

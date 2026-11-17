"""Shared fixtures: every test runs in its own temp copy of the project files.

Scripts use relative paths (data/, models/, logs/), so each test chdirs into a
fresh directory and builds a small v1 model there. Nothing in the repo is touched.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def source_mtimes():
    """What app.py's reload guard records: the modification time of each project module."""
    return {p.stem: p.stat().st_mtime_ns for p in ROOT.glob("*.py") if p.name != "app.py"}


@pytest.fixture(autouse=True)
def sources_unchanged():
    """Tells app.py's reload guard that no source changed, so it keeps the modules
    the tests imported (and patched). Tests of the guard itself override this."""
    import types

    seen = types.ModuleType("_observatory_sources")
    seen.mtimes = source_mtimes()
    sys.modules["_observatory_sources"] = seen
    yield
    sys.modules.pop("_observatory_sources", None)


@pytest.fixture
def restore_modules():
    """For tests that make app.py reload project modules: put the originals back
    afterwards, so other tests keep patching the modules that are actually in use."""
    names = list(source_mtimes())
    before = {n: sys.modules[n] for n in names if n in sys.modules}
    yield
    for n in names:
        sys.modules.pop(n, None)
    sys.modules.update(before)


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


@pytest.fixture(scope="session")
def fitted_v1(tmp_path_factory):
    """Model v1, fitted once per test session (fitting the tuned threshold takes seconds)."""
    import make_data
    import model as mdl

    root = tmp_path_factory.mktemp("fit")
    make_data.prepare(ROOT / make_data.RAW, root / "train.csv", root / "pool.csv")
    df = pd.read_csv(root / "train.csv")
    X, y = df.drop(columns=[make_data.TARGET]), df[make_data.TARGET]
    params = {**mdl.DEFAULT_PARAMS, "max_iter": 60}  # small, so tests stay fast
    fitted = mdl.build_model(params).fit(X, y)
    return fitted, params, mdl.evaluate(fitted, X, y)


@pytest.fixture
def trained(workdir, empty_mlflow_db, fitted_v1):
    """A small but real v1: data split, model, reference, metrics and simulated logs."""
    import shutil

    import make_data
    import simulate
    import train

    Path("data").mkdir()
    shutil.copy(ROOT / "data" / "credit_default_raw.csv.gz", "data/")
    make_data.prepare()
    X = pd.read_csv(make_data.TRAIN).drop(columns=[make_data.TARGET])
    Path("models").mkdir()
    shutil.copy(empty_mlflow_db, "models/mlflow.db")
    fitted, params, metrics = fitted_v1
    train.register_v1(fitted, X, metrics, params)
    simulate.run()
    return workdir

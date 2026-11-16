from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

import registry
from retrain import retrain


def mlflow_client(root="."):
    uri = registry.tracking_uri(root)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def test_v1_is_registered_as_champion(trained):
    reg = registry.load_registry()
    assert reg["active"] == "v1" and len(reg["models"]) == 1
    v1 = registry.get_active(reg)
    assert v1["status"] == "active" and v1["trigger"] == "initial training"
    assert 0.5 < v1["metrics"]["auc"] <= 1
    # the same facts, read straight from MLflow
    c = mlflow_client()
    champion = c.get_model_version_by_alias(registry.MODEL_NAME, registry.ALIAS)
    assert champion.version in (1, "1") and champion.run_id == v1["run_id"]
    run = c.get_run(v1["run_id"])
    assert run.data.metrics["auc"] == v1["metrics"]["auc"]
    assert run.data.metrics["threshold"] == v1["metrics"]["threshold"]
    assert run.data.params["model_type"] == "HistGradientBoostingClassifier"
    assert v1["model_params"]["max_iter"] == 60  # stored, so retraining can reuse them
    assert run.info.status == "FINISHED"


def test_model_and_reference_load_from_mlflow(trained):
    import pandas as pd

    import make_data

    v1 = registry.get_active(registry.load_registry())
    model, ref = registry.load_model(v1), registry.load_reference(v1)
    assert list(ref.columns) == make_data.FEATURE_NAMES and len(ref) == 20_000
    assert model.predict_proba(ref.head(5)).shape == (5, 2)
    assert isinstance(ref, pd.DataFrame)


def test_models_are_saved_as_skops_not_pickle(trained):
    """skops files can't run code on load; only sklearn/numpy/scipy types are trusted."""
    v1 = registry.get_active(registry.load_registry())
    files = {p.name for p in Path(registry._local(v1["model_uri"])).iterdir()}
    assert "model.skops" in files
    assert not {f for f in files if f.endswith((".pkl", ".joblib", ".pickle"))}

    class Evil:  # a type from outside the trusted packages
        pass

    with pytest.raises(ValueError, match="outside"):
        registry._trusted_types(Evil())


def test_retrain_logs_a_run_and_promotes(trained):
    entry = retrain(trigger="test")
    assert entry["version"] == "v2" and entry["compared_with"] == "v1"
    # under heavy concept drift the challenger should win
    assert entry["promoted"] and entry["status"] == "active"
    reg = registry.load_registry()
    assert reg["active"] == "v2"
    assert [m["status"] for m in reg["models"]] == ["archived", "active"]
    assert registry.champion_key() == ("v2", entry["run_id"])

    run = mlflow_client().get_run(entry["run_id"])
    assert run.data.metrics["auc"] == entry["metrics"]["auc"]
    assert run.data.metrics["champion_auc"] == entry["champion_metrics_on_holdout"]["auc"]
    assert run.data.tags["model_version"] == "v2"


def test_rejected_challenger_is_recorded_but_not_promoted(trained, monkeypatch):
    import model as mdl

    monkeypatch.setattr(  # make the challenger lose
        mdl, "evaluate",
        lambda model, X, y: {"accuracy": 0.5, "f1": 0.1, "auc": 0.9 if hasattr(model, "_champ") else 0.5},
    )
    real_load = registry.load_model

    def load_marked(entry):
        m = real_load(entry)
        m._champ = True
        return m

    monkeypatch.setattr(registry, "load_model", load_marked)
    entry = retrain(trigger="test")
    assert not entry["promoted"] and entry["status"] == "rejected"
    reg = registry.load_registry()
    assert reg["active"] == "v1" and [m["status"] for m in reg["models"]] == ["active", "rejected"]


def test_reset_goes_back_to_v1_and_numbering_restarts(trained):
    v1_run = registry.get_active(registry.load_registry())["run_id"]
    retrain(trigger="one")
    retrain(trigger="two")
    assert len(registry.load_registry()["models"]) == 3

    registry.reset_registry()
    reg = registry.load_registry()
    assert reg["active"] == "v1" and len(reg["models"]) == 1
    assert reg["models"][0]["status"] == "active" and reg["models"][0]["run_id"] == v1_run
    registry.load_model(reg["models"][0])  # v1's files survived
    # only v1's artifacts are left on disk
    assert len(list((registry.store_dir() / "mlartifacts").iterdir())) == 1
    # the next model is v2 again, not v4, and it is a different run
    again = retrain(trigger="after reset")
    assert again["version"] == "v2"


def test_workspace_gets_its_own_registry_linked_to_v1(trained, tmp_path_factory):
    other = tmp_path_factory.mktemp("visitor")
    seeded = registry.load_registry(other)
    project_v1 = registry.load_registry()["models"][0]
    assert seeded["active"] == "v1"
    assert seeded["models"][0]["model_uri"] == project_v1["model_uri"]  # a link, not a copy
    assert seeded["models"][0]["run_id"] == project_v1["run_id"]

    entry = retrain(trigger="test", root=other)
    assert registry.load_registry(other)["active"] == "v2"
    assert str(other.resolve()) in entry["model_uri"]  # the new model lives in the workspace
    # the project registry is still at v1
    assert registry.load_registry()["active"] == "v1" and len(registry.load_registry()["models"]) == 1

    registry.reset_registry(other)
    registry.load_model(registry.load_registry()["models"][0])  # project v1 files untouched


def test_missing_registry_gives_a_clear_error(workdir):
    with pytest.raises(FileNotFoundError, match="train.py"):
        registry.load_registry()

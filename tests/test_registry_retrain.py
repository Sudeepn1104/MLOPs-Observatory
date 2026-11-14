import json
from pathlib import Path

import registry
from retrain import retrain


def test_registry_bootstraps_v1(trained):
    reg = registry.load_registry()
    assert reg["active"] == "v1"
    assert registry.get_active(reg)["file"] == "models/model_v1.joblib"


def test_save_is_atomic_and_leaves_no_temp_files(trained):
    reg = registry.load_registry()
    reg["note"] = "x"
    registry.save_registry(reg)
    assert json.loads(Path("models/registry.json").read_text())["note"] == "x"
    assert not list(Path("models").glob(".registry-*"))


def test_retrain_promotes_and_reset_cleans_up(trained):
    entry = retrain(trigger="test")
    assert entry["version"] == "v2"
    assert entry["compared_with"] == "v1"
    reg = registry.load_registry()
    # under heavy concept drift the challenger should win
    assert entry["promoted"] and reg["active"] == "v2"
    assert Path("models/model_v2.joblib").exists()

    registry.reset_registry()
    reg = registry.load_registry()
    assert reg["active"] == "v1" and len(reg["models"]) == 1
    assert not Path("models/model_v2.joblib").exists()
    assert Path("models/model_v1.joblib").exists()


def test_workspace_root_keeps_project_registry_untouched(trained, tmp_path_factory):
    other = tmp_path_factory.mktemp("visitor")
    entry = retrain(trigger="test", root=other)
    assert Path(entry["file"]).parent == other / "models"
    assert registry.load_registry(other)["active"] == "v2"
    # the project-level registry is still at v1
    assert registry.load_registry()["active"] == "v1"
    assert not Path("models/model_v2.joblib").exists()

import json
from pathlib import Path

import pandas as pd

import registry
from retrain import retrain, retrain_policy


def test_policy_fires_on_psi_or_auc():
    ok = pd.DataFrame({"max_psi": [0.05, 0.05], "auc": [0.70, 0.70]})
    assert retrain_policy(ok, baseline_auc=0.70) == []
    psi_bad = ok.assign(max_psi=[0.05, 0.3])
    assert "PSI" in retrain_policy(psi_bad, 0.70)[0]
    auc_bad = ok.assign(auc=[0.6, 0.6])
    assert "AUC" in retrain_policy(auc_bad, 0.70)[0]


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

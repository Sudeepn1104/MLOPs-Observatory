"""Tiny model registry: a JSON file that tracks every model version.

Every function takes a workspace `root`. The default "." is the project
directory, which is what the scripts and a local dashboard use. The live demo
gives each visitor their own root (see workspace.py), so one visitor's
retrains never change what another visitor sees.

Model v1 and its reference data always live in the project's models/ folder
(they are read-only). Later versions are written under root/models/.
"""
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

V1_FILE = "models/model_v1.joblib"
V1_REFERENCE = "models/reference_data.csv"
V1_METRICS = "models/model_v1_metrics.json"


def registry_path(root="."):
    return Path(root) / "models" / "registry.json"


def models_dir(root="."):
    return Path(root) / "models"


def save_registry(reg, root="."):
    """Writes atomically: a crash or a second writer never leaves half a file."""
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".registry-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(reg, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_registry(root="."):
    """Loads the registry. Creates it from the v1 training output if missing."""
    path = registry_path(root)
    if path.exists():
        return json.loads(path.read_text())
    metrics = json.loads(Path(V1_METRICS).read_text())
    reg = {
        "active": "v1",
        "models": [
            {
                "version": "v1",
                "status": "active",
                "file": V1_FILE,
                "reference": V1_REFERENCE,
                "trained_at": datetime.fromtimestamp(Path(V1_FILE).stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
                "trigger": "initial training",
                "train_rows": None,
                "evaluated_on": "original test split",
                "metrics": metrics,
            }
        ],
    }
    save_registry(reg, root)
    return reg


def get_active(reg):
    return next(m for m in reg["models"] if m["version"] == reg["active"])


def reset_registry(root="."):
    """Back to v1 only (handy for re-running the demo). Never deletes v1."""
    registry_path(root).unlink(missing_ok=True)
    folder = models_dir(root)
    if not folder.exists():
        return
    for p in folder.glob("model_v*.joblib"):
        m = re.fullmatch(r"model_v(\d+)\.joblib", p.name)
        if m and int(m.group(1)) > 1:
            p.unlink()
    for p in folder.glob("reference_v*.csv"):
        p.unlink()

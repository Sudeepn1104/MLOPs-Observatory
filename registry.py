"""Tiny model registry: a JSON file that tracks every model version."""
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

REGISTRY = Path("models/registry.json")


def save_registry(reg):
    """Writes atomically: a crash or a second writer never leaves half a file."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=REGISTRY.parent, prefix=".registry-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(reg, f, indent=2)
        os.replace(tmp, REGISTRY)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_registry():
    """Loads the registry. Creates it from the v1 training output if missing."""
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    v1_file = Path("models/model_v1.joblib")
    metrics = json.loads(Path("models/model_v1_metrics.json").read_text())
    reg = {
        "active": "v1",
        "models": [
            {
                "version": "v1",
                "status": "active",
                "file": str(v1_file).replace("\\", "/"),
                "reference": "models/reference_data.csv",
                "trained_at": datetime.fromtimestamp(v1_file.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
                "trigger": "initial training",
                "train_rows": None,
                "evaluated_on": "original test split",
                "metrics": metrics,
            }
        ],
    }
    save_registry(reg)
    return reg


def get_active(reg):
    return next(m for m in reg["models"] if m["version"] == reg["active"])


def reset_registry():
    """Back to v1 only (handy for re-running the demo)."""
    REGISTRY.unlink(missing_ok=True)
    for p in Path("models").glob("model_v*.joblib"):
        m = re.fullmatch(r"model_v(\d+)\.joblib", p.name)
        if m and int(m.group(1)) > 1:
            p.unlink()
    for p in Path("models").glob("reference_v*.csv"):
        p.unlink()

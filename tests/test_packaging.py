"""Checks that packaging files agree with each other."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_mlflow_ui_image_matches_locked_client_version():
    """The MLflow server refuses a database written by a different MLflow version."""
    lock = (ROOT / "requirements-lock.txt").read_text()
    client = re.search(r"^mlflow-skinny==(\S+)", lock, re.M).group(1)
    server = re.search(r"^ARG MLFLOW_VERSION=(\S+)", (ROOT / "Dockerfile.mlflow").read_text(), re.M).group(1)
    assert server == client


def test_both_images_use_the_same_pinned_base():
    bases = {
        re.search(r"^FROM (\S+)", (ROOT / f).read_text(), re.M).group(1)
        for f in ("Dockerfile", "Dockerfile.mlflow")
    }
    assert len(bases) == 1 and "@sha256:" in bases.pop()

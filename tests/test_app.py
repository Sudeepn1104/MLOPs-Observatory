"""Smoke test: the whole dashboard script runs without an exception."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders(trained):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Tiny MLOps Observatory"


def test_retrain_button_promotes(trained):
    at = AppTest.from_file(APP, default_timeout=60).run()
    next(b for b in at.button if b.label == "Retrain now").click().run()
    assert not at.exception
    assert "v2" in [m.value for m in at.metric]

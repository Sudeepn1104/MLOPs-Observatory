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


def test_stream_button_adds_a_batch(trained):
    at = AppTest.from_file(APP, default_timeout=60).run()
    next(b for b in at.button if b.label == "Stream next batch").click().run()
    assert not at.exception
    import monitor

    assert monitor.load_logs()[0]["batch"].max() == 10


def test_sandbox_isolates_visitors(trained, monkeypatch, tmp_path_factory):
    """One visitor's retrain must not change what another visitor sees."""
    import workspace

    monkeypatch.setenv("OBSERVATORY_SANDBOX", "1")
    monkeypatch.setattr(workspace, "SANDBOX_BASE", tmp_path_factory.mktemp("sessions"))

    alice = AppTest.from_file(APP, default_timeout=60).run()
    bob = AppTest.from_file(APP, default_timeout=60).run()
    next(b for b in alice.button if b.label == "Retrain now").click().run()
    bob.run()
    assert not alice.exception and not bob.exception
    assert alice.metric[0].value == "v2"
    assert bob.metric[0].value == "v1"
    assert not Path("models/model_v2.joblib").exists()  # project dir untouched

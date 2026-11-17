"""Smoke test: the whole dashboard script runs without an exception."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders(trained):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Tiny MLOps Observatory"
    subheaders = [h.value for h in at.subheader]
    assert "Explain one client" in subheaders
    assert "Fairness across age groups" in subheaders
    assert "Calibration: can the probabilities be trusted?" in subheaders


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
    import registry

    assert registry.load_registry()["active"] == "v1"  # project registry untouched


def test_changed_source_files_are_reloaded_without_a_restart(trained, restore_modules):
    """Regression: the host updated the files under a running server, and the new app.py
    failed with "cannot import name 'needs_build' from 'bootstrap'" (the old module was in memory)."""
    import sys

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    sys.modules["drift"].marker = "loaded before the update"

    at.run()  # nothing changed on disk: the module in memory is kept
    assert getattr(sys.modules["drift"], "marker", None) == "loaded before the update"

    sys.modules["_observatory_sources"].mtimes["drift"] -= 1  # as if drift.py changed on disk
    at.run()
    assert not at.exception
    assert not hasattr(sys.modules["drift"], "marker")  # a fresh copy was imported


def test_first_run_in_a_process_with_old_modules_reloads_them(trained, restore_modules):
    """The fix must also heal a server that was already running the old code."""
    import sys

    AppTest.from_file(APP, default_timeout=60).run()
    sys.modules["bootstrap"].marker = "old module"
    del sys.modules["_observatory_sources"]  # a process that never ran the guard
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert not hasattr(sys.modules["bootstrap"], "marker")

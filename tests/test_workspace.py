import os
import time

import workspace


def test_sandbox_flag():
    assert workspace.sandbox_enabled({"OBSERVATORY_SANDBOX": "1"}, "/home/me/app")
    assert not workspace.sandbox_enabled({"OBSERVATORY_SANDBOX": "0"}, "/mount/src/app")
    # unset: on only on Streamlit Community Cloud
    assert workspace.sandbox_enabled({}, "/mount/src/mlops-observatory")
    assert not workspace.sandbox_enabled({}, "/home/me/app")


def test_session_root_is_private_and_stable(monkeypatch, tmp_path):
    monkeypatch.setenv("OBSERVATORY_SANDBOX", "1")
    monkeypatch.setattr(workspace, "SANDBOX_BASE", tmp_path)
    a, b = {}, {}
    ra, rb = workspace.session_root(a), workspace.session_root(b)
    assert ra != rb and ra.parent == rb.parent == tmp_path
    assert workspace.session_root(a) == ra  # same session keeps its workspace


def test_local_mode_uses_project_dir(monkeypatch):
    monkeypatch.setenv("OBSERVATORY_SANDBOX", "0")
    assert str(workspace.session_root({})) == "."


def test_cleanup_removes_only_stale_workspaces(tmp_path):
    old, fresh = tmp_path / "session-old", tmp_path / "session-new"
    old.mkdir()
    fresh.mkdir()
    long_ago = time.time() - 2 * workspace.MAX_AGE_SECONDS
    os.utime(old, (long_ago, long_ago))
    workspace.cleanup_old_sessions(tmp_path)
    assert not old.exists() and fresh.exists()

"""Backups: a restore brings back the registry (aliases included) and the prediction log."""
import tarfile
from datetime import datetime

import pytest

import backup
import registry
from retrain import retrain


def test_backup_and_restore_round_trip(trained, tmp_path):
    import sqlite3
    from pathlib import Path

    Path("logs").mkdir(exist_ok=True)
    with sqlite3.connect("logs/predictions.db") as db:
        db.execute("CREATE TABLE IF NOT EXISTS t (x)")
        db.execute("INSERT INTO t VALUES (42)")
    archive = backup.create()
    assert archive.exists() and archive.name.startswith("observatory-")

    retrain(trigger="after backup", shadow=True)  # change the state
    assert registry.get_challenger() is not None
    backup.restore(archive)
    registry.reconnect()
    reg = registry.load_registry()
    assert [m["version"] for m in reg["models"]] == ["v1"] and reg["active"] == "v1"
    assert registry.get_challenger() is None
    assert registry.load_model(reg["models"][0]) is not None  # artifacts came back too
    with sqlite3.connect("logs/predictions.db") as db:
        assert db.execute("SELECT x FROM t").fetchall() == [(42,)]


def test_keep_prunes_old_backups(trained):
    for day in range(1, 5):
        backup.create(keep=2, now=datetime(2026, 1, day))
    assert sorted(p.name for p in (trained / "backups").iterdir()) == [
        "observatory-20260103-000000.tar.gz", "observatory-20260104-000000.tar.gz",
    ]


def test_restore_refuses_a_foreign_archive(trained, tmp_path):
    bad = tmp_path / "bad.tar.gz"
    (tmp_path / "evil.txt").write_text("x")
    with tarfile.open(bad, "w:gz") as tar:
        tar.add(tmp_path / "evil.txt", arcname="../evil.txt")
    with pytest.raises(ValueError, match="not an observatory backup"):
        backup.restore(bad)

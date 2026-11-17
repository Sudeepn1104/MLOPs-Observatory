import sqlite3
from contextlib import closing

import bootstrap
import model as mdl
import registry


def test_fresh_store_is_not_stale(trained):
    assert bootstrap.stored_model_format() == mdl.MODEL_FORMAT
    assert not bootstrap.stale_store()
    assert bootstrap.needs_build() == ""
    assert registry.load_registry()["models"][0]["model_format"] == mdl.MODEL_FORMAT


def test_store_from_older_code_is_stale(trained, monkeypatch):
    """Regression: the host kept the old RandomForest store and the new code crashed on it."""
    monkeypatch.setattr(mdl, "MODEL_FORMAT", "some-newer-format")
    assert bootstrap.stale_store()
    assert "older version" in bootstrap.needs_build()


def test_store_without_a_stamp_is_stale(trained):
    with closing(sqlite3.connect(bootstrap.STORE)) as con, con:
        con.execute("DELETE FROM model_version_tags WHERE key = 'model_format'")
    assert bootstrap.stored_model_format() is None
    assert bootstrap.stale_store()


def test_no_store_means_missing_not_stale(workdir):
    assert not bootstrap.stale_store()
    assert bootstrap.needs_build().startswith("missing")


def test_retrained_versions_carry_the_stamp(trained):
    from retrain import retrain

    assert retrain(trigger="test")["model_format"] == mdl.MODEL_FORMAT


def test_reconnect_makes_a_running_process_see_a_replaced_database(trained, empty_mlflow_db):
    """Regression: after a rebuild the app kept reading the deleted database file."""
    import shutil

    import shap  # noqa: F401 - its lazy plotting module once made reconnect() raise

    assert len(registry.load_registry()["models"]) == 1  # opens a connection
    registry.db_path().unlink()
    shutil.copy(empty_mlflow_db, registry.db_path())  # another process replaced the file
    assert registry.exists()  # still reading the old, deleted file
    registry.reconnect()
    assert not registry.exists()  # now sees the new (empty) database

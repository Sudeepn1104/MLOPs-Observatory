"""drift.py must give the same answers as Evidently itself.

Runs only where Evidently is installed (pip install -r requirements-evidently.txt);
CI has a separate job for it, because Evidently needs pandas < 3.
"""
import numpy as np
import pytest

evidently = pytest.importorskip("evidently")


def test_scores_and_decisions_match_evidently(trained):
    import drift_report
    import registry
    from drift import check_drift
    from monitor import load_batches

    reference = registry.load_reference(registry.get_active(registry.load_registry()))
    batches = load_batches()
    for b in (0, 4, 9):  # no drift, partial drift, heavy drift
        current = batches[batches["batch"] == b].drop(columns=["default", "batch"])
        _, theirs = drift_report.evidently_drift(reference, current)
        ours = check_drift(reference, current).set_index("feature")
        theirs = theirs.set_index("feature").loc[ours.index]
        assert np.allclose(theirs["drift_score"], ours["drift_score"], atol=1e-4), b
        assert (theirs["drift_detected"] == ours["drift_detected"]).all(), b


def test_report_writes_html_json_and_logs_to_mlflow(trained, tmp_path):
    import json

    import drift_report
    import registry

    html, summary = tmp_path / "r.html", tmp_path / "r.json"
    args = ["--batch", "9", "--out", str(html), "--json", str(summary), "--log-mlflow"]
    assert drift_report.main(args) == 0
    assert html.stat().st_size > 100_000
    s = json.loads(summary.read_text())
    assert s["batch"] == 9 and s["share_drifted"] > 0.5 and len(s["features"]) == 10
    champion = registry.get_active(registry.load_registry())
    logged = [a.path for a in registry.client().list_artifacts(champion["run_id"], "drift_reports/batch_9")]
    assert logged == ["drift_reports/batch_9/r.html"]


def test_unknown_batch_fails_cleanly(trained, tmp_path):
    import drift_report

    assert drift_report.main(["--batch", "999", "--out", str(tmp_path / "x.html")]) == 2

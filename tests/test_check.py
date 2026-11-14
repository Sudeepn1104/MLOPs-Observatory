import json

import check
import monitor
from retrain import retrain


def test_drifted_v1_alerts_with_exit_code_1(trained, tmp_path):
    out_json, out_md = tmp_path / "r.json", tmp_path / "r.md"
    code = check.main(["--json", str(out_json), "--markdown", str(out_md)])
    assert code == 1
    report = json.loads(out_json.read_text())
    assert report["status"] == "alert" and report["model"] == "v1"
    assert any("PSI" in r for r in report["reasons"])
    md = out_md.read_text()
    assert "ALERT" in md and "Why the retrain rule fired" in md


def test_waiting_then_healthy_after_retrain(trained):
    retrain(trigger="test")
    assert check.build_report()["status"] == "waiting"
    assert check.main([]) == 0
    from registry import load_registry

    reg = load_registry()
    for _ in range(2):
        monitor.stream_next(reg)
    report = check.build_report()
    assert report["status"] == "healthy" and report["batches_scored_by_model"] == 2
    assert "No action needed" in check.to_markdown(report)


def test_live_traffic_drift_adds_a_reason(trained, monkeypatch):
    retrain(trigger="test")
    monkeypatch.setattr(
        check, "fetch_live_drift",
        lambda url: {"status": "red", "max_psi": 0.9, "worst_feature": "age", "rows": 500},
    )
    report = check.build_report(api_url="http://api:8000")
    assert report["status"] == "alert"
    assert "live traffic" in report["reasons"][0]
    assert "Live traffic" in check.to_markdown(report)


def test_broken_check_exits_2(workdir):
    # no model, no logs: the check must fail loudly, not report healthy
    assert check.main([]) == 2

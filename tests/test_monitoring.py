"""The Prometheus/Grafana setup: dashboard, scrape config, metrics and traffic generator."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "monitoring" / "grafana" / "dashboards" / "observatory.json"
sys.path.insert(0, str(ROOT / "monitoring"))


def exported_metric_names():
    """Every series name the API's Prometheus registry exports."""
    import serve

    names = set()
    for collector in serve.Metrics().registry.collect():
        for sample_suffix in ("", "_total", "_bucket", "_sum", "_count", "_created"):
            names.add(collector.name + sample_suffix)
    return names


def test_committed_dashboard_matches_its_generator():
    import build_dashboard

    assert json.loads(DASHBOARD.read_text()) == build_dashboard.build(), (
        "run python monitoring/build_dashboard.py and commit the result"
    )


def test_every_dashboard_query_uses_a_metric_the_api_exports():
    """A renamed metric would otherwise leave a silently empty panel."""
    exported = exported_metric_names()
    exprs = [t["expr"] for p in json.loads(DASHBOARD.read_text())["panels"] for t in p["targets"]]
    used = {m for e in exprs for m in re.findall(r"observatory_[a-z_]+", e)}
    assert used, "no queries found"
    assert used <= exported, f"unknown metrics: {sorted(used - exported)}"


def test_dashboard_panels_are_unique_and_laid_out():
    panels = json.loads(DASHBOARD.read_text())["panels"]
    assert len({p["id"] for p in panels}) == len(panels)
    assert all(p["datasource"]["uid"] == "prometheus" for p in panels)
    assert all(0 <= p["gridPos"]["x"] and p["gridPos"]["x"] + p["gridPos"]["w"] <= 24 for p in panels)


def test_prometheus_scrapes_metrics_with_a_bearer_key():
    cfg = yaml.safe_load((ROOT / "monitoring" / "prometheus.yml").read_text())
    job = cfg["scrape_configs"][0]
    assert job["metrics_path"] == "/metrics" and job["static_configs"][0]["targets"] == ["api:8000"]
    assert job["authorization"]["type"] == "Bearer"
    provisioned = yaml.safe_load(
        (ROOT / "monitoring/grafana/provisioning/datasources/prometheus.yml").read_text()
    )
    assert provisioned["datasources"][0]["uid"] == "prometheus"


KEY = "monitor-key-0123456789abcdef"


@pytest.fixture
def prod(trained, monkeypatch):
    import serve

    monkeypatch.setenv("OBSERVATORY_ENV", "production")
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setattr(serve, "DRIFT_REFRESH_SECONDS", 0)  # refresh on every scrape
    with TestClient(serve.app) as client:
        yield client


def clients(n, seed=0, **drift):
    import make_data

    b = make_data.make_batch(n, rng=np.random.default_rng(seed), **drift)
    return {"clients": b.drop(columns="default").to_dict(orient="records")}


def test_metrics_accept_a_bearer_token_like_prometheus_sends(prod):
    assert prod.get("/metrics").status_code == 401
    assert prod.get("/metrics", headers={"Authorization": "Bearer wrong-key-0123456789"}).status_code == 401
    assert prod.get("/metrics", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_scrape_publishes_live_drift_and_risk_distribution(prod):
    h = {"X-API-Key": KEY}
    text = prod.get("/metrics", headers=h).text
    assert 'observatory_live_drift_window_rows{model_version="v1"} 0.0' in text  # not enough traffic yet
    assert "observatory_live_drift_score{" not in text

    prod.post("/predict", json=clients(300, seed=1, age_tilt=1.0, limit_tilt=-1.0), headers=h)
    text = prod.get("/metrics", headers=h).text
    scores = dict(re.findall(r'observatory_live_drift_score\{[^}]*feature="(\w+)"[^}]*\} ([0-9.e+-]+)', text))
    assert len(scores) == 10 and float(scores["age"]) > 0.1
    drifted = re.search(r'observatory_live_drift_features_drifted\{model_version="v1"\} ([0-9.]+)', text)
    assert drifted and float(drifted.group(1)) >= 1
    assert 'observatory_predicted_probability_count{model_version="v1"} 300.0' in text


def test_loadgen_ramps_drift_and_builds_valid_requests(trained):
    import loadgen
    import serve

    assert loadgen.drift_level(0, 600) == 0
    assert loadgen.drift_level(300, 600) == pytest.approx(4.5)
    assert loadgen.drift_level(10_000, 600) == 9
    body = loadgen.make_request(5, 4.5, np.random.default_rng(0))
    serve.PredictRequest.model_validate(body)  # the API would accept it
    assert len(body["clients"]) == 5

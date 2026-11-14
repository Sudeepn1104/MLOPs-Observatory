"""The API as deployed: OBSERVATORY_ENV=production with an API key."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import make_data
import serve

KEY = "test-key-0123456789abcdef"
OLD_KEY = "old-key-0123456789abcdef"


def body(n=2):
    b = make_data.make_batch(n, rng=np.random.default_rng(0)).drop(columns="default")
    return {"clients": b.to_dict(orient="records")}


@pytest.fixture
def prod(trained, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_ENV", "production")
    monkeypatch.setenv("API_KEY", f"{KEY},{OLD_KEY}")
    monkeypatch.setenv("GIT_SHA", "abc1234")
    with TestClient(serve.app) as client:
        yield client


def test_production_refuses_to_start_without_a_key(trained, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_ENV", "production")
    with pytest.raises(RuntimeError, match="requires API_KEY"):
        with TestClient(serve.app):
            pass


def test_short_keys_are_rejected():
    with pytest.raises(RuntimeError, match="16 characters"):
        serve.Settings.from_env({"API_KEY": "short"})


def test_probes_stay_open(prod):
    assert prod.get("/health").status_code == 200
    assert prod.get("/ready").json()["model_version"] == "v1"
    v = prod.get("/version").json()
    assert v == {"version": serve.APP_VERSION, "git_sha": "abc1234", "model_version": "v1"}
    assert prod.get("/docs").status_code == 200


@pytest.mark.parametrize("path", ["/model", "/monitoring/drift", "/metrics"])
def test_protected_reads_need_a_key(prod, path):
    assert prod.get(path).status_code == 401
    assert prod.get(path, headers={"X-API-Key": "wrong-key-0123456789"}).status_code == 401
    assert prod.get(path, headers={"X-API-Key": KEY}).status_code == 200


def test_predict_needs_a_key_and_old_key_still_works_during_rotation(prod):
    assert prod.post("/predict", json=body()).status_code == 401
    for key in (KEY, OLD_KEY):
        r = prod.post("/predict", json=body(), headers={"X-API-Key": key})
        assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_request_id_is_echoed_or_generated(prod):
    r = prod.get("/health", headers={"X-Request-ID": "trace-42"})
    assert r.headers["X-Request-ID"] == "trace-42"
    assert len(prod.get("/health").headers["X-Request-ID"]) == 32


def test_one_request_id_across_header_body_and_prediction_log(prod):
    import sqlite3

    r = prod.post("/predict", json=body(2), headers={"X-API-Key": KEY, "X-Request-ID": "trace-99"})
    assert r.headers["X-Request-ID"] == "trace-99" == r.json()["request_id"]
    con = sqlite3.connect("logs/predictions.db")
    try:
        stored = con.execute("SELECT COUNT(*) FROM predictions WHERE request_id='trace-99'").fetchone()
    finally:
        con.close()
    assert stored == (2,)


def test_oversized_body_is_rejected_before_parsing(trained, monkeypatch):
    monkeypatch.setenv("MAX_REQUEST_BYTES", "2000")
    with TestClient(serve.app) as client:
        r = client.post("/predict", json=body(50))
    assert r.status_code == 413


def test_metrics_count_requests_and_predictions(prod):
    h = {"X-API-Key": KEY}
    prod.post("/predict", json=body(3), headers=h)
    prod.post("/predict", json=body(3))  # rejected: no key
    text = prod.get("/metrics", headers=h).text
    assert 'observatory_predictions_total{model_version="v1"} 3.0' in text
    assert 'observatory_active_model_info{version="v1"} 1.0' in text
    assert 'route="/predict",status="200"' in text
    assert 'route="/predict",status="401"' in text


def test_production_logs_are_json_with_request_ids(prod, caplog):
    import logging

    handler = logging.getLogger("observatory").handlers[0]
    assert isinstance(handler.formatter, serve.JsonFormatter)  # production format

    with caplog.at_level(logging.INFO, logger="observatory"):
        logging.getLogger("observatory").addHandler(caplog.handler)
        prod.post("/predict", json=body(), headers={"X-API-Key": KEY, "X-Request-ID": "req-7"})
    lines = [json.loads(serve.JsonFormatter().format(r)) for r in caplog.records]
    access = [x for x in lines if x.get("msg") == "request" and x.get("request_id") == "req-7"]
    assert access and access[0]["route"] == "/predict" and access[0]["status"] == 200
    assert KEY not in json.dumps(lines)  # the key never appears in logs

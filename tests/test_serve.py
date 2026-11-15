import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import make_data
import registry
from retrain import retrain


@pytest.fixture
def api(trained):
    import serve

    with TestClient(serve.app) as client:
        yield client


def clients(n=1, seed=0, **drift):
    b = make_data.make_batch(n, rng=np.random.default_rng(seed), **drift)
    return b.drop(columns="default").to_dict(orient="records")


def test_health_ready_model(api):
    assert api.get("/health").json() == {"status": "ok"}
    assert api.get("/ready").json()["model_version"] == "v1"
    info = api.get("/model").json()
    assert info["version"] == "v1"
    assert info["features"] == make_data.FEATURE_NAMES


def test_predict_matches_the_model(api):
    batch = clients(25)
    r = api.post("/predict", json={"clients": batch})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "v1" and len(body["predictions"]) == 25
    v1 = registry.load_model(registry.load_registry()["models"][0])
    expected = v1.predict_proba(pd.DataFrame(batch))[:, 1]
    got = [p["default_probability"] for p in body["predictions"]]
    assert np.allclose(got, expected, atol=1e-4)
    assert all(p["default"] == (p["default_probability"] >= 0.5) for p in body["predictions"])


@pytest.mark.parametrize(
    "change",
    [
        {"age": 5},  # out of range
        {"credit_limit": 0},
        {"repay_status_sep": 12},
        {"income": 1000},  # unknown field
    ],
)
def test_invalid_input_is_rejected(api, change):
    client = clients(1)[0] | change
    assert api.post("/predict", json={"clients": [client]}).status_code == 422


def test_empty_and_missing_fields_rejected(api):
    assert api.post("/predict", json={"clients": []}).status_code == 422
    partial = clients(1)[0]
    partial.pop("age")
    assert api.post("/predict", json={"clients": [partial]}).status_code == 422


def test_live_drift_from_logged_traffic(api):
    assert api.get("/monitoring/drift").json()["status"] == "insufficient_data"
    api.post("/predict", json={"clients": clients(300, seed=1)})
    calm = api.get("/monitoring/drift", params={"window": 300}).json()
    assert calm["rows"] == 300 and calm["status"] == "green"

    older = clients(300, seed=2, age_tilt=1.0, limit_tilt=-1.0)
    api.post("/predict", json={"clients": older})
    drifted = api.get("/monitoring/drift", params={"window": 300}).json()
    assert drifted["status"] == "red"
    assert drifted["worst_feature"] in {"age", "credit_limit"}


def test_picks_up_a_promoted_model_without_restart(api):
    assert api.get("/model").json()["version"] == "v1"
    api.post("/predict", json={"clients": clients(5)})  # v1 traffic
    entry = retrain(trigger="test")
    assert entry["promoted"]
    assert api.get("/model").json()["version"] == "v2"
    r = api.post("/predict", json={"clients": clients(3)})
    assert r.json()["model_version"] == "v2"
    # drift is tracked per model: only v2's 3 clients count, not v1's 5
    assert api.get("/monitoring/drift").json()["rows"] == 3

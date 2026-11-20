"""Shadow deployment: the challenger scores live traffic but never answers a client."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import make_data
import registry
import shadow
from retrain import retrain

KEY = "client-key-0123456789abcdef"
ADMIN = "admin-key-0123456789abcdef"


def clients(n, seed=0):
    b = make_data.make_batch(n, rng=np.random.default_rng(seed))
    return {"clients": b.drop(columns="default").to_dict(orient="records")}


@pytest.fixture
def shadowed(trained):
    """v1 champion, plus v2 trained on recent batches and put in shadow."""
    e = retrain(trigger="test", shadow=True)
    assert e["version"] == "v2"
    return e


@pytest.fixture
def api(shadowed):
    import serve

    with TestClient(serve.app) as client:
        yield client


# ---------- registry ----------
def test_retrain_in_shadow_never_moves_the_champion(shadowed):
    reg = registry.load_registry()
    assert reg["active"] == "v1"
    assert registry.get_challenger()["version"] == "v2"
    assert registry.get_challenger()["status"] == "shadow"
    assert shadowed["promoted"] is False and "beats_champion_offline" in shadowed


def test_promote_and_stop_move_the_aliases(shadowed):
    registry.promote("v2")
    reg = registry.load_registry()
    assert reg["active"] == "v2" and registry.get_challenger() is None
    statuses = {m["version"]: m["status"] for m in reg["models"]}
    assert statuses == {"v1": "archived", "v2": "active"}

    registry.start_shadow("v1")
    assert registry.get_challenger()["version"] == "v1"
    assert registry.stop_shadow()["status"] == "rejected"
    assert registry.get_challenger() is None and registry.stop_shadow() is None


def test_the_champion_cannot_shadow_itself(trained):
    with pytest.raises(ValueError, match="already the champion"):
        registry.start_shadow("v1")


# ---------- API ----------
def test_clients_get_the_champions_answer_and_both_are_logged(api):
    r = api.post("/predict", json=clients(40))
    assert r.status_code == 200 and r.json()["model_version"] == "v1"
    pairs = shadow.read_pairs("logs/predictions.db", "v2")
    assert len(pairs) == 40 and set(pairs["champion_version"]) == {"v1"}
    v1 = registry.load_model(registry.load_registry()["models"][0])
    X = pd.DataFrame(clients(40)["clients"])
    assert np.allclose(pairs["champion_probability"][::-1].to_numpy(), v1.predict_proba(X)[:, 1])
    served = [p["default_probability"] for p in r.json()["predictions"]]
    assert np.allclose(served, pairs["champion_probability"][::-1], atol=1e-4)

    text = api.get("/metrics").text
    assert 'observatory_challenger_model_info{version="v2"} 1.0' in text
    assert 'observatory_shadow_predictions_total{challenger_version="v2",champion_version="v1"} 40.0' in text


def test_shadow_report_waits_for_enough_traffic_then_gives_a_verdict(api):
    api.post("/predict", json=clients(100))
    s = api.get("/shadow").json()
    assert s["challenger"] == "v2" and s["rows"] == 100
    if s["offline"]["beats_champion"]:
        assert s["verdict"] == "keep_watching"
    else:
        assert s["verdict"] == "reject"
    for seed in range(1, 6):
        api.post("/predict", json=clients(100, seed=seed))
    s = api.get("/shadow").json()
    assert s["rows"] == 600 and 0 <= s["live"]["agreement"] <= 1
    assert s["verdict"] in {"promote", "review", "reject"}


def test_a_failing_challenger_never_breaks_a_request(api, monkeypatch):
    import serve

    def boom():
        raise RuntimeError("challenger exploded")

    monkeypatch.setattr(serve.app.state.challenger, "get", boom)
    r = api.post("/predict", json=clients(5))
    assert r.status_code == 200 and len(r.json()["predictions"]) == 5
    assert "observatory_shadow_errors_total 1.0" in api.get("/metrics").text


def test_promoting_over_the_api_serves_the_challenger_next(api):
    assert api.post("/shadow/promote").json() == {"champion": "v2"}
    assert api.post("/predict", json=clients(3)).json()["model_version"] == "v2"
    assert api.get("/shadow").json()["challenger"] is None
    assert api.post("/shadow/promote").status_code == 409
    assert api.post("/shadow/v1").json() == {"challenger": "v1"}
    assert api.delete("/shadow").json() == {"rejected": "v1"}
    assert api.post("/shadow/v99").status_code == 404


def test_no_challenger_means_no_shadow_work(trained):
    import serve

    with TestClient(serve.app) as client:
        client.post("/predict", json=clients(5))
        assert len(shadow.read_pairs("logs/predictions.db", "v2")) == 0
        assert "observatory_challenger_model_info{" not in client.get("/metrics").text


def production(monkeypatch, admin=None):
    import serve

    monkeypatch.setenv("OBSERVATORY_ENV", "production")
    monkeypatch.setenv("API_KEY", KEY)
    if admin:
        monkeypatch.setenv("ADMIN_API_KEY", admin)
    return TestClient(serve.app)


def test_admin_endpoints_are_off_in_production_without_an_admin_key(shadowed, monkeypatch):
    with production(monkeypatch) as c:
        assert c.post("/shadow/promote", headers={"X-API-Key": KEY}).status_code == 403
        assert c.get("/shadow", headers={"X-API-Key": KEY}).status_code == 200  # reading is fine


def test_a_scoring_key_cannot_promote(shadowed, monkeypatch):
    with production(monkeypatch, admin=ADMIN) as c:
        assert c.post("/shadow/promote", headers={"X-API-Key": KEY}).status_code == 401
        r = c.post("/shadow/promote", headers={"Authorization": f"Bearer {ADMIN}"})
        assert r.json() == {"champion": "v2"}


def test_admin_key_must_differ_from_the_scoring_key():
    import serve

    with pytest.raises(RuntimeError, match="must differ"):
        serve.Settings.from_env({"API_KEY": KEY, "ADMIN_API_KEY": KEY})


# ---------- the verdict ----------
def pairs(n, champion_flag, challenger_flag, seed=0):
    rng = np.random.default_rng(seed)
    cp = rng.uniform(0, 1, n)
    return pd.DataFrame({
        "champion_probability": cp, "challenger_probability": np.clip(cp + rng.normal(0, 0.02, n), 0, 1),
        "champion_default": rng.uniform(size=n) < champion_flag,
        "challenger_default": rng.uniform(size=n) < challenger_flag,
    })


@pytest.mark.parametrize(("beats", "n", "challenger_flag", "expected"), [
    (False, 1000, 0.30, "reject"),
    (True, 100, 0.30, "keep_watching"),
    (True, 1000, 0.30, "promote"),
    (True, 1000, 0.55, "review"),
    (None, 1000, 0.30, "review"),
])
def test_verdict(beats, n, challenger_flag, expected):
    entry = {"version": "v2", "metrics": {"auc": 0.8}, "champion_metrics_on_holdout": {"auc": 0.7},
             "beats_champion_offline": beats}
    s = shadow.summarize(pairs(n, 0.30, challenger_flag), entry, "v1")
    assert s["verdict"] == expected, s["reason"]


def test_cli_round_trip(shadowed, capsys):
    assert shadow.main(["status"]) == 0
    assert '"challenger": "v2"' in capsys.readouterr().out
    assert shadow.main(["promote"]) == 0
    assert registry.load_registry()["active"] == "v2"
    assert shadow.main(["start", "v1"]) == 0 and shadow.main(["stop"]) == 0
    assert "no challenger" in (shadow.main(["status"]), capsys.readouterr().out)[1]


def test_demo_setup_is_idempotent(trained, capsys):
    assert shadow.main(["demo"]) == 0
    assert registry.get_challenger()["version"] == "v2"
    assert shadow.main(["demo"]) == 0
    assert "leaving it" in capsys.readouterr().out
    assert len(registry.load_registry()["models"]) == 2


def test_alias_lookups_are_cached_but_promotions_over_the_api_apply_at_once(shadowed, monkeypatch):
    import serve

    monkeypatch.setenv("OBSERVATORY_ALIAS_TTL", "60")
    with TestClient(serve.app) as c:
        calls = []
        real = registry.champion_key
        monkeypatch.setattr(registry, "champion_key", lambda root=".": calls.append(1) or real(root))
        for _ in range(5):
            c.post("/predict", json=clients(1))
        assert len(calls) <= 1  # one lookup serves many requests
        c.post("/shadow/promote")
        assert c.post("/predict", json=clients(1)).json()["model_version"] == "v2"


def test_decisions_match_the_models_own_predict(shadowed):
    import model as mdl

    m = registry.load_model(registry.get_challenger())
    X = pd.DataFrame(clients(300)["clients"])
    assert ((m.predict_proba(X)[:, 1] >= mdl.threshold(m)) == m.predict(X).astype(bool)).all()

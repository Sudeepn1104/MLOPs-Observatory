import pandas as pd

import monitor
from registry import get_active, load_registry
from retrain import retrain

V1 = {"version": "v1", "metrics": {"auc": 0.76}}


def log(models, psi, auc):
    return pd.DataFrame(
        {"batch": range(len(models)), "model": models, "max_psi": psi, "auc": auc}
    )


def test_evaluate_healthy_and_firing():
    ok = log(["v1", "v1"], [0.05, 0.05], [0.76, 0.75])
    assert monitor.evaluate(ok, V1)[0] == []
    psi = monitor.evaluate(log(["v1", "v1"], [0.05, 0.3], [0.76, 0.76]), V1)[0]
    assert len(psi) == 1 and "PSI" in psi[0]
    auc = monitor.evaluate(log(["v1", "v1"], [0.05, 0.05], [0.6, 0.6]), V1)[0]
    assert len(auc) == 1 and "AUC" in auc[0]


def test_evaluate_judges_only_the_active_models_batches():
    """v1's bad batches must not make a freshly promoted v2 look broken."""
    v2 = {"version": "v2", "metrics": {"auc": 0.67}}
    history = log(["v1", "v1"], [1.5, 1.6], [0.45, 0.43])
    reasons, own = monitor.evaluate(history, v2)
    assert reasons == [] and own.empty  # waiting for data
    with_v2 = pd.concat([history, log(["v2", "v2"], [0.08, 0.09], [0.70, 0.71])])
    reasons, own = monitor.evaluate(with_v2, v2)
    assert reasons == [] and len(own) == 2


def test_auto_retrain_once_per_new_batch():
    reg = {"models": [{"version": "v1"}]}
    hist = log(["v1"] * 10, [1.0] * 10, [0.4] * 10)
    assert monitor.can_auto_retrain(reg, hist)
    reg["models"].append({"version": "v2", "status": "rejected", "data_until_batch": 9})
    assert not monitor.can_auto_retrain(reg, hist)  # no loop after a rejection
    newer = pd.concat([hist, log(["v1"], [1.0], [0.4]).assign(batch=10)])
    assert monitor.can_auto_retrain(reg, newer)


def test_full_loop_retrain_then_stream(trained):
    reg = load_registry()
    batch_log, _ = monitor.load_logs()
    assert monitor.evaluate(batch_log, get_active(reg))[0]  # v1 under drift fires

    entry = retrain(trigger="test")
    reg = load_registry()
    assert entry["data_until_batch"] == 9
    batch_log, _ = monitor.load_logs()
    assert monitor.evaluate(batch_log, get_active(reg))[1].empty  # v2 waiting

    rows = [monitor.stream_next(reg) for _ in range(2)]
    assert [r["batch"] for r in rows] == [10, 11]
    assert all(r["model"] == "v2" for r in rows)
    batch_log, feature_log = monitor.load_logs()
    assert len(batch_log) == 12
    assert set(feature_log["batch"]) == set(range(12))
    assert monitor.load_batches()["batch"].max() == 11
    # v2 learned the drifted world: its inputs look familiar again
    assert max(r["max_psi"] for r in rows) < 0.2

    monitor.reset_stream()
    assert len(monitor.load_logs()[0]) == 10


def test_streamed_batches_are_deterministic(trained):
    reg = load_registry()
    a = monitor.stream_next(reg)
    monitor.reset_stream()
    b = monitor.stream_next(reg)
    assert a == b

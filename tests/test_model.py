import numpy as np
import pandas as pd
import pytest

import make_data
import model as mdl
import registry


@pytest.fixture
def data(trained):
    df = pd.read_csv(make_data.TRAIN)
    return df.drop(columns="default"), df["default"]


@pytest.fixture
def v1(trained):
    return registry.load_model(registry.get_active(registry.load_registry()))


def test_model_is_tuned_threshold_gradient_boosting(v1):
    from sklearn.ensemble import HistGradientBoostingClassifier

    assert isinstance(v1.estimator_, HistGradientBoostingClassifier)
    assert 0.05 < mdl.threshold(v1) < 0.6  # tuned, not the implicit 0.5


def test_predict_uses_the_tuned_threshold(v1, data):
    X, _ = data
    proba = v1.predict_proba(X.head(500))[:, 1]
    assert (v1.predict(X.head(500)) == (proba >= mdl.threshold(v1))).all()


def test_probabilities_are_calibrated(v1, data):
    """Mean predicted risk matches the real default rate (the old weighted model said ~0.45)."""
    X, y = data
    m = mdl.evaluate(v1, X, y)
    assert abs(v1.predict_proba(X)[:, 1].mean() - y.mean()) < 0.03
    assert m["ece"] < 0.05
    assert set(m) >= {"auc", "pr_auc", "brier", "ece", "f1", "precision", "recall", "accuracy", "threshold"}


def test_calibration_error_detects_overconfidence():
    y = np.array([0, 1] * 500)
    assert mdl.calibration_error(y, np.full(1000, 0.5)) < 0.01
    assert mdl.calibration_error(y, np.full(1000, 0.9)) > 0.35


def test_tune_returns_best_params_and_ranked_results(data):
    X, y = data
    params, results = mdl.tune(X.head(2000), y.head(2000), n_iter=3)
    assert set(params) == set(mdl.SEARCH_SPACE)
    assert len(results) == 3 and results["rank_test_score"].iloc[0] == 1
    assert results["mean_test_score"].iloc[0] == results["mean_test_score"].max()


def test_fairness_reports_groups_and_checks(v1, data):
    X, y = data
    table, summary = mdl.fairness(v1, X, y)
    assert list(table["age_group"]) == ["under 30", "30-39", "40-49", "50+"]
    assert table["clients"].sum() == len(X)
    expected = table["selection_rate"].min() / table["selection_rate"].max()
    assert summary["selection_rate_ratio"] == pytest.approx(expected, abs=1e-3)
    assert summary["passes_four_fifths"] == (summary["selection_rate_ratio"] >= 0.8)
    assert summary["calibration_gap"] < 0.05 and summary["calibrated_within_groups"]


def test_shap_contributions_add_up_to_the_prediction(v1, data):
    X, _ = data
    sample = X.head(20)
    contrib, base = mdl.contributions(v1, sample)
    log_odds = base + contrib.sum(axis=1)
    proba = v1.estimator_.predict_proba(sample)[:, 1]
    assert np.allclose(1 / (1 + np.exp(-log_odds)), proba, atol=1e-4)
    # under the original rule, recent late payment is the strongest signal
    assert mdl.global_importance(v1, X.head(500)).index[0] == "repay_status_sep"


def test_v1_run_holds_evaluation_artifacts(trained):
    entry = registry.get_active(registry.load_registry())
    assert registry.load_artifact(entry, "missing.csv") is None


def test_training_stays_in_one_process_by_default(monkeypatch):
    """Regression: n_jobs=-1 spawned a worker per core (1.3 GB) and crashed the hosted demo."""
    import bootstrap

    assert mdl.N_JOBS == 1
    assert mdl.build_model().n_jobs == 1
    calls = []
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda cmd, **kw: calls.append(cmd[1:]))
    bootstrap.build_all()
    assert ["train.py", "--no-tune"] in calls  # no hyperparameter search at app start-up


def test_openmp_threads_are_capped_in_every_process_that_loads_models():
    """Regression: on a host showing 64 cores with ~1 usable, a retrain took 108 s instead of 4 s."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import registry\n"
        "from threadpoolctl import threadpool_info\n"
        "print(max(i['num_threads'] for i in threadpool_info() if i['user_api'] == 'openmp'))\n"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "64"}
    env.pop("OBSERVATORY_THREADS", None)
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert out.stdout.strip().splitlines()[-1] == str(mdl.MAX_THREADS) == "2", out.stderr[-500:]

import numpy as np
import pandas as pd
import pytest

from drift import check_drift, psi, status_from_psi


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.01


def test_psi_grows_with_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 5000)
    small, big = psi(ref, ref + 0.2), psi(ref, ref + 1.0)
    assert 0 < small < big
    assert big > 0.2


def test_psi_handles_constant_feature():
    # all-equal reference collapses the bin edges; must not crash or return NaN
    assert np.isfinite(psi(np.ones(100), np.ones(100) * 2))


@pytest.mark.parametrize(
    "value,expected", [(0.0, "green"), (0.099, "green"), (0.1, "yellow"), (0.2, "red"), (5, "red")]
)
def test_status_thresholds(value, expected):
    assert status_from_psi(value) == expected


def test_check_drift_one_row_per_feature():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    cur = ref.assign(b=ref["b"] + 2)
    rep = check_drift(ref, cur).set_index("feature")
    assert list(rep.index) == ["a", "b"]
    assert rep.loc["a", "status"] == "green"
    assert rep.loc["b", "status"] == "red"
    assert rep.loc["b", "ks_pvalue"] < 0.001


def test_wasserstein_norm_measures_shift_in_reference_stds():
    from drift import wasserstein_norm

    rng = np.random.default_rng(4)
    ref = rng.normal(0, 2, 20000)
    assert wasserstein_norm(ref, ref) == 0
    assert abs(wasserstein_norm(ref, ref + 1.0) - 0.5) < 0.02  # shifted by half a std
    assert wasserstein_norm(np.ones(50), np.ones(50) + 1) > 0  # constant reference: no division by zero


def test_jensen_shannon_compares_category_shares():
    from drift import jensen_shannon

    same = [1, 2, 2, 3] * 100
    assert jensen_shannon(same, same) == 0
    assert jensen_shannon(same, [1, 2, 3, 3] * 100) > 0.1
    # a category that only appears in the new data counts as drift
    assert jensen_shannon([1] * 100, [2] * 100) == pytest.approx(np.sqrt(np.log(2)), abs=1e-9)


def test_check_drift_uses_the_right_test_per_type():
    rng = np.random.default_rng(5)
    ref = pd.DataFrame({"amount": rng.normal(size=2000), "code": rng.integers(0, 3, 2000)})
    cur = ref.assign(code=np.where(rng.random(2000) < 0.5, 2, ref["code"]))  # more of code 2
    rep = check_drift(ref, cur, categorical=["code"]).set_index("feature")
    assert rep.loc["amount", "method"] == "wasserstein_norm" and not rep.loc["amount", "drift_detected"]
    assert rep.loc["code", "method"] == "jensen_shannon" and rep.loc["code", "drift_detected"]
    assert {"psi", "ks_stat", "status", "drift_score"} <= set(rep.columns)  # PSI is still reported

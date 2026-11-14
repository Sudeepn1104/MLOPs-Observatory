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

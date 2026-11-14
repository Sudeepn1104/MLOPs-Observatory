import numpy as np

from make_data import make_loans


def test_schema_and_ranges():
    df = make_loans(1000, rng=np.random.default_rng(0))
    assert list(df.columns) == [
        "age", "income", "loan_amount", "credit_score", "employment_years", "debt_ratio", "default"
    ]
    assert df["age"].between(21, 70).all()
    assert df["credit_score"].between(300, 850).all()
    assert set(df["default"].unique()) <= {0, 1}


def test_rng_makes_it_reproducible():
    a = make_loans(200, rng=np.random.default_rng(5))
    b = make_loans(200, rng=np.random.default_rng(5))
    assert a.equals(b)


def test_input_drift_moves_age():
    base = make_loans(3000, rng=np.random.default_rng(1))
    old = make_loans(3000, age_shift=10, rng=np.random.default_rng(1))
    assert old["age"].mean() - base["age"].mean() > 5

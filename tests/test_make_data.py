import gzip
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import make_data
from drift import check_drift

ROOT = Path(__file__).resolve().parent.parent
RAW_SHA256 = "a0f0ab49d6326671d6cd83be5c88dcf18007025fe9a53ecd699119c871176ca1"


def test_vendored_dataset_matches_source_checksum():
    """data/SOURCE.md records this checksum; the committed file must be unchanged."""
    raw = gzip.decompress((ROOT / "data" / "credit_default_raw.csv.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == RAW_SHA256


def test_load_raw_schema():
    df = make_data.load_raw(ROOT / make_data.RAW)
    assert len(df) == 30_000
    assert list(df.columns) == make_data.FEATURE_NAMES + ["default"]
    assert "SEX" not in df.columns and "sex" not in df.columns
    assert df.notna().all().all()
    assert 0.2 < df["default"].mean() < 0.25


def test_prepare_split_is_disjoint_and_reproducible(trained):
    train, pool = pd.read_csv(make_data.TRAIN), pd.read_csv(make_data.POOL)
    assert (len(train), len(pool)) == (20_000, 10_000)
    again_train, again_pool = make_data.prepare()
    assert again_train.reset_index(drop=True).equals(train)
    # each raw row (by its original position) lands in exactly one side
    assert not set(again_train.index) & set(again_pool.index)
    assert len(set(again_train.index) | set(again_pool.index)) == 30_000


@pytest.fixture
def pool(trained):
    return pd.read_csv(make_data.POOL)


def test_batch_rows_come_from_the_pool(pool):
    b = make_data.make_batch(300, rng=np.random.default_rng(0))
    assert list(b.columns) == list(pool.columns)
    merged = b.drop(columns="default").merge(
        pool.drop(columns="default").drop_duplicates(), how="left", indicator=True
    )
    assert (merged["_merge"] == "both").all()


def test_rng_makes_batches_reproducible(pool):
    a = make_data.make_batch(200, age_tilt=0.5, concept=0.5, rng=np.random.default_rng(5))
    b = make_data.make_batch(200, age_tilt=0.5, concept=0.5, rng=np.random.default_rng(5))
    assert a.equals(b)


def test_age_tilt_shifts_age_and_raises_psi(pool):
    rng = np.random.default_rng(1)
    base = make_data.make_batch(2000, rng=rng)
    old = make_data.make_batch(2000, age_tilt=0.8, rng=rng)
    assert old["age"].mean() > base["age"].mean() + 3
    feats = pool.drop(columns="default")
    rep = check_drift(feats, old.drop(columns="default")).set_index("feature")
    assert rep.loc["age", "status"] == "red"


def test_concept_drift_keeps_inputs_but_breaks_the_old_rule(pool):
    """The silent failure: PSI green, but a rule learned from old labels stops working."""
    rng = np.random.default_rng(2)
    clean = make_data.make_batch(3000, rng=rng)
    drifted = make_data.make_batch(3000, concept=1.0, rng=rng)
    feats = pool.drop(columns="default")
    assert check_drift(feats, drifted.drop(columns="default"))["psi"].max() < 0.1
    # v1 learned the old rule: good on recorded labels, no better than chance on the new rule
    v1 = joblib.load("models/model_v1.joblib")

    def auc(b):
        return roc_auc_score(b["default"], v1.predict_proba(b.drop(columns="default"))[:, 1])

    assert auc(clean) > 0.7
    assert auc(drifted) < 0.55

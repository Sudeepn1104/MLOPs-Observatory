import numpy as np
import pandas as pd

import make_data
import simulate


def test_concept_level_ramp():
    assert simulate.concept_level(0) == 0
    assert simulate.concept_level(simulate.CONCEPT_START - 1) == 0
    assert simulate.concept_level(simulate.N_BATCHES - 1) == 1


def test_batches_do_not_replay_training_draws():
    """Regression: batch 0 used to reuse the training data's random ages exactly."""
    train = make_data.make_loans(10000, rng=np.random.default_rng(42))
    sim = make_data.make_loans(
        simulate.BATCH_SIZE, rng=np.random.default_rng(simulate.SIM_SEED)
    )
    same = (sim["age"].values == train["age"].values[: simulate.BATCH_SIZE]).mean()
    assert same < 0.2  # ages are rounded to whole years, so a few matches are chance


def test_run_writes_logs_and_drift_grows(trained):
    log = pd.read_csv("logs/batch_log.csv")
    assert len(log) == simulate.N_BATCHES
    assert log["max_psi"].iloc[-1] > log["max_psi"].iloc[0]
    assert log["auc"].iloc[-1] < log["auc"].iloc[0]  # concept drift hurts the model
    feats = pd.read_csv("logs/feature_drift.csv")
    assert len(feats) == simulate.N_BATCHES * 6

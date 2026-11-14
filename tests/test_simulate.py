import pandas as pd

import make_data
import simulate


def test_concept_level_ramp():
    assert simulate.concept_level(0) == 0
    assert simulate.concept_level(simulate.CONCEPT_START - 1) == 0
    assert simulate.concept_level(simulate.N_BATCHES - 1) == 1


def test_simulator_seed_differs_from_split_seed():
    """Regression: the simulator once replayed the training data's random draws."""
    assert simulate.SIM_SEED != make_data.SPLIT_SEED


def test_batches_never_contain_training_rows(trained):
    """Monitoring batches must come only from rows v1 never trained on."""
    batches = pd.read_csv("logs/batches.csv").drop(columns=["default", "batch"])
    pool = pd.read_csv(make_data.POOL).drop(columns="default").drop_duplicates()
    in_pool = batches.merge(pool, how="left", indicator=True)["_merge"] == "both"
    assert in_pool.all()


def test_run_writes_logs_and_drift_grows(trained):
    log = pd.read_csv("logs/batch_log.csv")
    assert len(log) == simulate.N_BATCHES
    assert log["max_psi"].iloc[-1] > log["max_psi"].iloc[0]
    assert log["auc"].iloc[-1] < log["auc"].iloc[0]  # concept drift hurts the model
    feats = pd.read_csv("logs/feature_drift.csv")
    assert len(feats) == simulate.N_BATCHES * len(make_data.FEATURE_NAMES)

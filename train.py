"""Tunes, trains and evaluates model v1 and registers it in MLflow as the champion.

Steps:
1. Randomized search for gradient-boosting hyperparameters (5-fold CV on the
   training split, scored by ROC AUC). Each candidate is an MLflow child run.
2. Fit the final model: the best parameters plus a cross-validated decision
   threshold (see model.py).
3. Evaluate on the held-out test split: ranking, calibration, decision
   metrics, and fairness across age groups.
4. Register it as version 1 with the champion alias.
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

import model as mdl
import registry
from make_data import TARGET, TRAIN


def register_v1(fitted, X_train, metrics, params=None, root=".", artifacts=None, search=None):
    """Registers a fitted model as v1, the first champion."""
    return registry.register_version(
        fitted,
        X_train,
        {
            "trigger": "initial training",
            "train_rows": int(len(X_train)),
            "evaluated_on": "original test split",
            "metrics": metrics,
            "model_params": params or {},
            "model_format": mdl.MODEL_FORMAT,
        },
        params={
            "model_type": type(mdl.inner_model(fitted)).__name__,
            **(params or {}),
            "train_rows": len(X_train),
        },
        promote=True,
        root=root,
        run_name="v1 initial training",
        artifacts=artifacts,
        search=search,
    )


def main(tune=True, n_iter=20):
    df = pd.read_csv(TRAIN)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=mdl.SEED, stratify=y
    )

    search = None
    params = mdl.DEFAULT_PARAMS
    if tune:
        params, search = mdl.tune(X_train, y_train, n_iter=n_iter)
        print(f"Best of {n_iter} candidates: CV AUC {search.loc[0, 'mean_test_score']:.4f} with {params}")

    fitted = mdl.build_model(params).fit(X_train, y_train)
    metrics = mdl.evaluate(fitted, X_test, y_test)
    table, summary = mdl.fairness(fitted, X_test, y_test)

    registry.wipe()  # a fresh v1 means any old runs and versions are stale
    register_v1(
        fitted, X_train, {**metrics, **{f"fairness_{k}": v for k, v in summary.items()
                                         if isinstance(v, float)}},
        params, artifacts={
            "fairness.csv": table, "fairness_summary.json": summary,
            "calibration.csv": mdl.calibration_curve(fitted, X_test, y_test),
        },
        search=search,
    )
    print("Registered model v1 in MLflow:", metrics)
    print("Fairness across age groups:", summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-tune", action="store_true", help="skip the search, use model.DEFAULT_PARAMS")
    ap.add_argument("--n-iter", type=int, default=20, help="search candidates (default 20)")
    a = ap.parse_args()
    main(tune=not a.no_tune, n_iter=a.n_iter)

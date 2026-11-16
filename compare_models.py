"""Compares candidate models and logs each one to MLflow (experiment: model-selection).

    python compare_models.py

This is the evidence behind the model choice in model.py and docs/MODEL.md.
It is not part of the training pipeline; run it after changing the data or
the features to check the choice still holds. Every candidate gets the same
5-fold cross-validation, the same test split and a decision threshold tuned
the same way, so only the algorithm differs.
"""
import argparse

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import model as mdl
import registry
from make_data import TARGET, TRAIN

EXPERIMENT = "model-selection"


def candidates():
    return {
        "logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "random forest (previous model)": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=mdl.SEED, n_jobs=-1
        ),
        "random forest + isotonic calibration": CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced", random_state=mdl.SEED, n_jobs=-1
            ),
            method="isotonic", cv=mdl.CV_FOLDS,
        ),
        "gradient boosting (sklearn defaults)": mdl.base_model({}),
        "gradient boosting (tuned, current model)": mdl.base_model(),
    }


def compare(X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(mdl.CV_FOLDS, shuffle=True, random_state=mdl.SEED)
    rows = []
    for name, est in candidates().items():
        cv_auc = cross_val_score(est, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=mdl.N_JOBS)
        # as fitted ("raw": implicit 0.5 threshold) and with a tuned threshold
        raw = mdl.evaluate(est.fit(X_train, y_train), X_test, y_test)
        tuned = TunedThresholdClassifierCV(est, scoring="f1", cv=mdl.CV_FOLDS, random_state=mdl.SEED)
        dec = mdl.evaluate(tuned.fit(X_train, y_train), X_test, y_test)
        rows.append({
            "model": name,
            "cv_auc": round(float(cv_auc.mean()), 4),
            "cv_auc_std": round(float(cv_auc.std()), 4),
            "test_auc": raw["auc"],
            "test_pr_auc": raw["pr_auc"],
            "brier": raw["brier"],
            "ece": raw["ece"],
            "f1_at_0.5": raw["f1"],
            "tuned_threshold": dec["threshold"],
            "f1_at_tuned_threshold": dec["f1"],
        })
    return pd.DataFrame(rows)


def log_to_mlflow(results, root="."):
    c = registry.client(root)
    exp = c.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else c.create_experiment(EXPERIMENT)
    for _, row in results.iterrows():
        run = c.create_run(exp_id, run_name=row["model"])
        for k, v in row.items():
            if k != "model":
                c.log_metric(run.info.run_id, k, float(v))
        c.set_terminated(run.info.run_id)


def main(sample=None):
    df = pd.read_csv(TRAIN)
    if sample:
        df = df.sample(sample, random_state=mdl.SEED)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=mdl.SEED, stratify=y)
    results = compare(X_tr, y_tr, X_te, y_te)
    log_to_mlflow(results)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", type=int, help="use a random sample of N training rows (quick check)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(main(ap.parse_args().sample).to_string(index=False))

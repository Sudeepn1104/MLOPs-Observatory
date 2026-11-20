"""The model: how it is built, tuned, evaluated, explained and checked for fairness.

Design (see docs/MODEL.md for the reasoning and the measurements behind it):

- Gradient boosting (scikit-learn's HistGradientBoostingClassifier), tuned with
  a randomized cross-validated search.
- No class weights. Weighting the rare class (22% default) inflates every
  probability; unweighted boosting gives probabilities that match observed
  default rates (calibration error about 0.01 instead of 0.21).
- The yes/no decision uses an explicit threshold, tuned by cross-validation
  to maximise F1 (TunedThresholdClassifierCV), instead of the implicit 0.5.
  So `predict_proba` is an honest probability and `predict` is the decision.
- SHAP explanations per prediction, and fairness metrics across age groups.
"""
import os
import threading

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, TunedThresholdClassifierCV
from threadpoolctl import threadpool_limits

SEED = 42
CV_FOLDS = 5

# Identifies how models are built. Every registered model is stamped with it.
# A host can keep an old model store on disk across a code update; when the
# stamp of the stored v1 differs from this, the app rebuilds the store instead
# of running new code against an old model (see bootstrap.py). Change it
# whenever a model from older code would not work with this code.
MODEL_FORMAT = "hgb-tuned-threshold-1"

# Worker processes for cross-validation. Default 1: every extra process costs
# about 200 MB, and the dashboard retrains inside a memory-limited host
# (n_jobs=-1 used 1.3 GB on 4 cores and crashed the hosted demo). The fits are
# fast, so one process takes seconds. Raise it for a big search on your own
# machine: OBSERVATORY_JOBS=4 python train.py
N_JOBS = int(os.environ.get("OBSERVATORY_JOBS", "1"))

# Threads for gradient boosting (OpenMP). By default it starts one thread per
# CPU core it can see. A shared cloud host can show dozens of cores while the
# app may only use about one, and the threads then fight each other: in a
# 1-CPU container with 64 threads, a retrain took 108 s instead of 3.7 s.
#
# OpenMP's limit is per thread: setting it in one thread doesn't change
# others. Streamlit runs every page run in a new thread and the API answers
# requests in worker threads, so cap_threads() must be called in each thread
# that fits or scores a model, not once at import. It's cheap after the first
# call in a thread.
MAX_THREADS = int(os.environ.get("OBSERVATORY_THREADS", "2"))
_capped = threading.local()


def cap_threads():
    """Limits OpenMP threads to MAX_THREADS in the calling thread (idempotent per thread)."""
    if getattr(_capped, "done", False):
        return
    threadpool_limits(limits=MAX_THREADS, user_api="openmp")
    _capped.done = True


cap_threads()  # the importing thread (scripts run here)

# the best parameters tune() found on the training data (CV AUC 0.773); used
# by `python train.py --no-tune` and as a fallback (see docs/MODEL.md)
DEFAULT_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 150,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
}

SEARCH_SPACE = {
    "learning_rate": [0.02, 0.05, 0.1],
    "max_iter": [150, 300, 500],
    "max_leaf_nodes": [7, 15, 31],
    "min_samples_leaf": [20, 50, 100, 200],
    "l2_regularization": [0.0, 0.1, 1.0, 5.0],
}

AGE_BANDS = [(0, 29, "under 30"), (30, 39, "30-39"), (40, 49, "40-49"), (50, 200, "50+")]


def base_model(params=None):
    # `params={}` means scikit-learn's defaults; only None falls back to ours
    return HistGradientBoostingClassifier(
        **(DEFAULT_PARAMS if params is None else params), random_state=SEED
    )


def build_model(params=None):
    """Unfitted model: gradient boosting plus a cross-validated decision threshold."""
    cap_threads()  # fitting happens in the caller's thread
    return TunedThresholdClassifierCV(
        base_model(params), scoring="f1", cv=CV_FOLDS, random_state=SEED, n_jobs=N_JOBS
    )


def tune(X, y, n_iter=20):
    """Randomized search over SEARCH_SPACE, scored by cross-validated ROC AUC.

    Returns (best_params, cv_results DataFrame). AUC is used here because it
    judges ranking; the threshold is tuned separately by build_model().
    """
    search = RandomizedSearchCV(
        base_model(),
        SEARCH_SPACE,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED),
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_)
    keep = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time"]
    results = results[[c for c in results if c.startswith("param_")] + keep]
    results.columns = [c.removeprefix("param_") for c in results.columns]
    return search.best_params_, results.sort_values("rank_test_score").reset_index(drop=True)


def inner_model(model):
    """The boosting model inside the threshold wrapper (any other model is returned as is)."""
    return model.estimator_ if isinstance(model, TunedThresholdClassifierCV) else model


def threshold(model):
    """The decision threshold of a fitted model (0.5 for models without one)."""
    return float(getattr(model, "best_threshold_", 0.5))


def calibration_error(y, proba, bins=10):
    """Expected calibration error: the average gap between predicted and observed default rates."""
    y, proba = np.asarray(y), np.asarray(proba)
    idx = np.clip(np.digitize(proba, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(
        sum(abs(proba[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
            for b in range(bins) if (idx == b).any())
    )


def evaluate(model, X, y):
    """Ranking, calibration and decision metrics on one dataset."""
    cap_threads()
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    r = lambda v: round(float(v), 4)  # noqa: E731
    return {
        "auc": r(roc_auc_score(y, proba)),
        "pr_auc": r(average_precision_score(y, proba)),
        "brier": r(brier_score_loss(y, proba)),
        "ece": r(calibration_error(y, proba)),
        "f1": r(f1_score(y, pred, zero_division=0)),
        "precision": r(precision_score(y, pred, zero_division=0)),
        "recall": r(recall_score(y, pred, zero_division=0)),
        "accuracy": r(accuracy_score(y, pred)),
        "threshold": r(threshold(model)),
    }


def calibration_curve(model, X, y, bins=10):
    """Predicted vs observed default rate per probability bin (for a reliability plot)."""
    proba = model.predict_proba(X)[:, 1]
    df = pd.DataFrame({"proba": proba, "y": np.asarray(y)})
    df["bin"] = pd.cut(df["proba"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = df.groupby("bin", observed=True).agg(
        predicted=("proba", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    return out.reset_index(drop=True).round(4)


# ---------- fairness ----------
def age_band(age):
    for lo, hi, name in AGE_BANDS:
        if lo <= age <= hi:
            return name
    return "unknown"


def fairness(model, X, y):
    """Per-age-group outcomes, plus three summary checks.

    selection_rate_ratio  lowest / highest share of clients flagged as likely
                          to default. Below 0.8 fails the common "four-fifths"
                          rule of thumb for disparate impact.
    tpr_gap               largest difference in recall (share of real
                          defaulters caught) between groups. Above 0.1 means
                          the model protects some groups much better than
                          others ("equal opportunity").
    calibration_gap       largest gap between the mean predicted probability
                          and the real default rate within a group. Small
                          means a "30%" risk means 30% in every group.

    These checks can't all pass at once when real default rates differ
    between groups: a model that is calibrated within each group flags each
    group at its own rate. Reporting them side by side is the point; which
    one matters most is a policy decision (see docs/MODEL.md).
    """
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    df = pd.DataFrame({"band": X["age"].map(age_band), "y": np.asarray(y), "pred": pred, "proba": proba})
    rows = []
    for _, _, name in AGE_BANDS:
        g = df[df["band"] == name]
        if g.empty:
            continue
        pos, neg = g[g["y"] == 1], g[g["y"] == 0]
        rows.append({
            "age_group": name,
            "clients": int(len(g)),
            "default_rate": round(float(g["y"].mean()), 4),
            "mean_predicted": round(float(g["proba"].mean()), 4),
            "selection_rate": round(float(g["pred"].mean()), 4),
            "recall": round(float(pos["pred"].mean()), 4) if len(pos) else None,
            "false_positive_rate": round(float(neg["pred"].mean()), 4) if len(neg) else None,
            "auc": round(float(roc_auc_score(g["y"], g["proba"])), 4) if g["y"].nunique() == 2 else None,
        })
    table = pd.DataFrame(rows)
    sel = table["selection_rate"]
    tpr = table["recall"].dropna()
    summary = {
        "selection_rate_ratio": round(float(sel.min() / sel.max()), 4) if sel.max() > 0 else None,
        "tpr_gap": round(float(tpr.max() - tpr.min()), 4) if len(tpr) else None,
        "calibration_gap": round(float((table["mean_predicted"] - table["default_rate"]).abs().max()), 4),
    }
    summary["passes_four_fifths"] = bool(summary["selection_rate_ratio"] is not None
                                         and summary["selection_rate_ratio"] >= 0.8)
    summary["passes_equal_opportunity"] = bool(summary["tpr_gap"] is not None and summary["tpr_gap"] <= 0.1)
    summary["calibrated_within_groups"] = bool(summary["calibration_gap"] <= 0.05)
    return table, summary


# ---------- explanations ----------
def explainer(model):
    """A SHAP TreeExplainer for the boosting model inside the threshold wrapper."""
    import shap  # imported lazily: only explanation features need it

    return shap.TreeExplainer(inner_model(model))


def contributions(model, X, explainer_=None):
    """SHAP values per client and feature, in log-odds of default.

    Positive pushes towards default, negative away from it. For each client
    base + sum(contributions) equals the model's log-odds, so the numbers add
    up to the prediction exactly. Returns (DataFrame, base_value).
    """
    ex = explainer_ or explainer(model)
    sv = ex(X)
    values, base = np.asarray(sv.values), np.asarray(sv.base_values)
    if values.ndim == 3:  # some tree models return one set of values per class
        values, base = values[:, :, 1], base.reshape(len(values), -1)[:, -1]
    return pd.DataFrame(values, columns=X.columns, index=X.index), float(np.ravel(base)[0])


def global_importance(model, X, explainer_=None):
    """Mean absolute SHAP value per feature, largest first."""
    values, _ = contributions(model, X, explainer_)
    return values.abs().mean().sort_values(ascending=False)

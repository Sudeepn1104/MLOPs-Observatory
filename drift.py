"""Drift checks per feature.

Two families of measures, side by side:

- PSI and the KS test (written from scratch). PSI drives the retrain rule
  and the dashboard's heatmap, with thresholds 0.1 (watch) and 0.2 (drift).
- The per-type tests that Evidently uses by default for reference data of
  this size: normalised Wasserstein distance for numerical features and
  Jensen-Shannon distance for categorical ones, both flagging drift at 0.1.
  PSI bins every feature by quantiles, which suits amounts but not codes
  like education level; Jensen-Shannon compares category shares directly.
  These reproduce Evidently's numbers exactly (a test checks this against
  Evidently itself; see drift_report.py and docs/DRIFT.md).
"""
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import ks_2samp, wasserstein_distance

DRIFT_THRESHOLD = 0.1  # Evidently's default for both per-type tests


def psi(reference, current, bins=10):
    """Population Stability Index. <0.1 stable, 0.1-0.2 watch, >0.2 drift."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def wasserstein_norm(reference, current):
    """Wasserstein distance divided by the reference's standard deviation.

    How far, in reference standard deviations, the distribution has moved.
    Same as Evidently's "Wasserstein distance (normed)".
    """
    norm = max(float(np.std(reference)), 0.001)
    return float(wasserstein_distance(reference, current) / norm)


def jensen_shannon(reference, current):
    """Jensen-Shannon distance between the category shares of two samples.

    0 means identical shares, 1 (natural log: ~0.83) means no overlap.
    Same as Evidently's "Jensen-Shannon distance" for categorical features.
    """
    reference, current = pd.Series(reference), pd.Series(current)
    keys = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    ref = reference.value_counts().reindex(keys, fill_value=0).to_numpy() / len(reference)
    cur = current.value_counts().reindex(keys, fill_value=0).to_numpy() / len(current)
    return float(distance.jensenshannon(ref, cur))


def status_from_psi(value):
    if value < 0.1:
        return "green"
    if value < 0.2:
        return "yellow"
    return "red"


def check_drift(ref_df, batch_df, categorical=None):
    """One row per feature: PSI, KS, PSI status, and the per-type drift test.

    categorical: columns compared as categories (default: make_data.CATEGORICAL).
    """
    if categorical is None:
        from make_data import CATEGORICAL as categorical
    rows = []
    for col in ref_df.columns:
        ref, cur = ref_df[col].values, batch_df[col].values
        p = psi(ref, cur)
        ks = ks_2samp(ref, cur)
        if col in categorical:
            method, score = "jensen_shannon", jensen_shannon(ref, cur)
        else:
            method, score = "wasserstein_norm", wasserstein_norm(ref, cur)
        rows.append(
            {
                "feature": col,
                "psi": round(p, 4),
                "ks_stat": round(float(ks.statistic), 4),
                "ks_pvalue": round(float(ks.pvalue), 4),
                "status": status_from_psi(p),
                "method": method,
                "drift_score": round(score, 4),
                "drift_detected": bool(score >= DRIFT_THRESHOLD),
            }
        )
    return pd.DataFrame(rows)

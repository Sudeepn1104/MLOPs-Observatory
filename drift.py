"""Drift checks: PSI and KS test per feature."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(reference, current, bins=10):
    """Population Stability Index. <0.1 stable, 0.1-0.2 watch, >0.2 drift."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def status_from_psi(value):
    if value < 0.1:
        return "green"
    if value < 0.2:
        return "yellow"
    return "red"


def check_drift(ref_df, batch_df):
    """Returns one row per feature with PSI, KS stat, KS p-value, status."""
    rows = []
    for col in ref_df.columns:
        p = psi(ref_df[col].values, batch_df[col].values)
        ks = ks_2samp(ref_df[col].values, batch_df[col].values)
        rows.append(
            {
                "feature": col,
                "psi": round(p, 4),
                "ks_stat": round(float(ks.statistic), 4),
                "ks_pvalue": round(float(ks.pvalue), 4),
                "status": status_from_psi(p),
            }
        )
    return pd.DataFrame(rows)

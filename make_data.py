"""Real credit-default data and a drift simulator built on top of it.

Data: the UCI "Default of Credit Card Clients" dataset (30,000 real clients,
real default labels). See data/SOURCE.md for provenance and license.

`python make_data.py` splits it into data/train.csv (for model v1) and
data/stream_pool.csv (rows the model never trains on). `make_batch` then draws
monitoring batches from the stream pool and can add two kinds of drift:

- Input drift: rows are resampled with weights that favour, for example, older
  clients or lower credit limits. Every row is still a real client.
- Concept drift: a share of the labels is replaced by a different, simulated
  default rule. The features look the same, but what predicts default changes.
  This part is simulated: the dataset has no real concept drift to replay.
"""
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/credit_default_raw.csv.gz")
TRAIN = Path("data/train.csv")
POOL = Path("data/stream_pool.csv")
TARGET = "default"
SPLIT_SEED = 42
TRAIN_ROWS = 20_000

# raw column -> readable name. Monthly columns: 1 / _0 = September 2005, 2 = August, 3 = July.
FEATURES = {
    "LIMIT_BAL": "credit_limit",  # credit limit in NT dollars
    "AGE": "age",
    "EDUCATION": "education",  # 1 graduate school, 2 university, 3 high school, 4+/0 other
    "MARRIAGE": "marriage",  # 1 married, 2 single, 3 other, 0 unknown
    "PAY_0": "repay_status_sep",  # -2/-1/0 paid on time, 1..8 months late
    "PAY_2": "repay_status_aug",
    "PAY_3": "repay_status_jul",
    "BILL_AMT1": "bill_sep",  # bill statement amount
    "PAY_AMT1": "paid_sep",  # amount paid
    "PAY_AMT2": "paid_aug",
}
FEATURE_NAMES = list(FEATURES.values())


def load_raw(path=RAW):
    """The original dataset, with the 10 model features renamed plus the label."""
    df = pd.read_csv(path)
    out = df[list(FEATURES)].rename(columns=FEATURES)
    out[TARGET] = df["default.payment.next.month"].astype(int)
    return out.astype({c: "int64" for c in FEATURE_NAMES})


def prepare(raw_path=RAW, train_path=TRAIN, pool_path=POOL):
    """Shuffles once with a fixed seed and writes the train / stream-pool split."""
    df = load_raw(raw_path)
    idx = np.random.default_rng(SPLIT_SEED).permutation(len(df))
    train, pool = df.iloc[idx[:TRAIN_ROWS]], df.iloc[idx[TRAIN_ROWS:]]
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    pool.to_csv(pool_path, index=False)
    return train, pool


@cache
def _pool(path, mtime):
    """Stream pool plus the standardised columns the samplers need.

    Cached per absolute path and modification time, so a regenerated pool (or
    a different working directory) is never served stale.
    """
    pool = pd.read_csv(path)

    def z(s):
        return ((s - s.mean()) / s.std()).to_numpy()

    extras = {
        "age_z": z(pool["age"]),
        "limit_z": z(np.log(pool["credit_limit"])),
        "utilisation": (pool["bill_sep"] / pool["credit_limit"]).clip(-1, 2).to_numpy(),
    }
    return pool, extras


def new_rule_proba(utilisation, age_z, repay_status):
    """The changed default rule used for concept drift.

    Under the new rule, defaults follow how much of the credit line is used and
    hit younger clients, while recent late payments stop predicting default
    (they even point the other way). A model trained on the old rule leans
    heavily on repayment status, so it ranks clients worse than chance.
    """
    z = -1.2 + 3.0 * (utilisation - 0.4) - 0.9 * age_z - 1.5 * (repay_status > 0)
    return 1 / (1 + np.exp(-z))


_rng = np.random.default_rng(SPLIT_SEED)


def make_batch(n, limit_tilt=0.0, age_tilt=0.0, concept=0.0, rng=None, pool_path=POOL):
    """Draws a labelled batch of n real clients from the stream pool.

    limit_tilt: negative favours clients with lower credit limits.
    age_tilt:   positive favours older clients.
    concept:    0 = labels as recorded, 1 = every label follows the new rule.
    rng:        optional numpy Generator, so callers get their own random stream.

    A tilt of about 0.5 is a strong shift (PSI well above 0.2 on that feature).
    """
    rng = _rng if rng is None else rng
    path = Path(pool_path).resolve()
    pool, ex = _pool(str(path), path.stat().st_mtime_ns)
    logw = limit_tilt * ex["limit_z"] + age_tilt * ex["age_z"]
    w = np.exp(logw - logw.max())
    idx = rng.choice(len(pool), size=n, replace=True, p=w / w.sum())
    batch = pool.iloc[idx].reset_index(drop=True)

    c = float(np.clip(concept, 0.0, 1.0))
    if c > 0:
        flip = rng.random(n) < c
        p = new_rule_proba(
            ex["utilisation"][idx], ex["age_z"][idx], batch["repay_status_sep"].to_numpy()
        )
        new = (rng.random(n) < p).astype(int)
        batch.loc[flip, TARGET] = new[flip]
    return batch


if __name__ == "__main__":
    train, pool = prepare()
    print(
        f"Saved {TRAIN} {train.shape} and {POOL} {pool.shape}, "
        f"default rate {train[TARGET].mean():.2%}"
    )

"""Creates a synthetic loan-default dataset at data/loan.csv.

Use this to start immediately. Later, replace data/loan.csv with a real
Kaggle loan default CSV (keep the target column named 'default').
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)


def make_loans(n, income_shift=0.0, age_shift=0.0, concept=0.0):
    """concept: 0 = original default rule, 1 = fully changed rule (concept drift).

    Input drift (income_shift, age_shift) moves the feature values only.
    Concept drift changes HOW features lead to default, so a model trained
    on the old rule gets worse even if the inputs look similar.
    """
    age = rng.normal(38 + age_shift, 10, n).clip(21, 70)
    income = rng.lognormal(10.8 + income_shift, 0.5, n)
    loan_amount = income * rng.uniform(0.2, 1.5, n)
    credit_score = rng.normal(680, 70, n).clip(300, 850)
    employment_years = (age - 21).clip(0) * rng.uniform(0.1, 0.8, n)
    debt_ratio = rng.beta(2, 5, n)

    c = float(np.clip(concept, 0.0, 1.0))
    z = (
        -2.0
        + 1.6 * c  # new rule also raises the overall default rate
        + (3.0 - 6.0 * c) * debt_ratio  # debt effect flips sign at c=1
        + 1.2 * (loan_amount / income)
        - (0.006 - 0.012 * c) * (credit_score - 600)  # credit effect flips too
        - 0.02 * employment_years
        - 0.05 * c * (age - 38)  # new rule: younger borrowers default more
    )
    p = 1 / (1 + np.exp(-z))
    default = (rng.random(n) < p).astype(int)

    return pd.DataFrame(
        {
            "age": age.round(0),
            "income": income.round(0),
            "loan_amount": loan_amount.round(0),
            "credit_score": credit_score.round(0),
            "employment_years": employment_years.round(1),
            "debt_ratio": debt_ratio.round(3),
            "default": default,
        }
    )


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = make_loans(10000)
    df.to_csv("data/loan.csv", index=False)
    print(f"Saved data/loan.csv {df.shape}, default rate {df['default'].mean():.2%}")

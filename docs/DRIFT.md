# Drift detection: PSI, Evidently, and why both

The code is in [`drift.py`](../drift.py) (used by the app, the API and the scheduled check) and [`drift_report.py`](../drift_report.py) (Evidently's HTML reports).

## What drift detection is for

A model learns the distribution of its training data. Drift detection compares new data with that training data, feature by feature, **without needing labels**. That matters because in credit, the label (did the client default?) arrives months later. Drift scores are the early warning; model performance is the proof, once labels arrive.

## Two families of tests, side by side

| | PSI (plus the KS test) | Per-type tests (Evidently's defaults) |
|---|---|---|
| Numerical features | Population Stability Index on 10 quantile bins | **Wasserstein distance**, divided by the reference's standard deviation |
| Categorical features | the same PSI on quantile bins | **Jensen-Shannon distance** between category shares |
| Thresholds | < 0.1 stable, 0.1–0.2 watch, > 0.2 drift | ≥ 0.1 drift |
| Used for | the retrain rule and the dashboard's default heatmap | the heatmap's second view, the API and logs, Evidently's reports |

**Why per-type tests?** PSI bins every feature by quantiles. That suits amounts like credit limit, but not codes like education level (1 = graduate school, 2 = university, ...), where "quantiles" are meaningless and many bins collapse. Jensen-Shannon compares the share of each category directly. Five of the ten features here are codes: education, marriage and three months of repayment status.

**What the normalised Wasserstein distance means:** how far the distribution moved, measured in the reference's standard deviations. 0.5 means "shifted by about half a standard deviation". It's easy to explain and doesn't depend on bin choices.

**They behave differently.** The per-type tests react earlier: age is flagged from batch 1, while PSI stays green until batch 3. PSI still drives the retrain rule, because its thresholds were chosen for it and the rule's history is in PSI. Changing that would be a separate, deliberate decision.

## Why Evidently runs separately

Evidently is the standard open-source library for this, so using it is the natural choice. But measured before adopting it:

- **It doesn't support pandas 3.** Version 0.7.23 (the latest, Sep 2026) crashes on import with pandas 3: it imports a pandas internal that pandas 3 removed. It also requires plotly < 6.
- **It's heavy:** 38 more packages (81 → 119), including a web framework, an NLP library and statsmodels, plus about **3 s and 170 MB** just to import.

The live demo has already crashed once from memory pressure on Streamlit Cloud (see [MODEL.md](MODEL.md)), and downgrading pandas for the whole project would be a step backwards. So:

- **The app** (dashboard, API, scheduled check) uses `drift.py`, which reimplements Evidently's two tests with scipy (about 30 ms per batch).
- **Evidently** runs in its own environment ([`requirements-evidently.txt`](../requirements-evidently.txt)) to produce full interactive HTML reports.
- **A test proves they agree.** [`tests/test_evidently_agreement.py`](../tests/test_evidently_agreement.py) runs both on the same batches and requires the same scores (within 0.0001) and the same drift decisions. On the 10 monitoring batches, all 100 feature decisions match. CI runs it in a dedicated job.

When Evidently supports pandas 3, the separation can be revisited: the agreement test is what makes the switch safe either way.

## The silent failure still holds

With concept drift alone (the rule behind defaulting changes, the inputs don't), **Evidently's tests flag none of the ten features** (largest score 0.06), just like PSI, while the model's AUC falls to about 0.48. Better drift tests don't fix this, because nothing in the inputs changed. That's why the retrain rule also watches performance.

## Using it

```
pip install -r requirements-evidently.txt
```

```
python drift_report.py --batch 9 --log-mlflow
```

This writes `reports/drift_report.html` (interactive, self-contained, no internet needed) and `reports/drift_summary.json`. With `--log-mlflow` it also attaches the report to the champion's MLflow run, under `drift_reports/`. The weekly scheduled check builds a report for the latest batch and attaches it to the workflow run as the `evidently-drift-report` artifact.

## Questions an interviewer might ask

- *Why not just use Evidently everywhere?* It doesn't support our pandas version, and it adds 170 MB and 3 s to every process, on a host whose memory limit has already bitten us. I reimplemented its two default tests and proved agreement with a test, and kept Evidently for reports.
- *Why Jensen-Shannon for categories?* It compares category shares directly. PSI's quantile bins don't make sense for codes.
- *What does a Wasserstein score of 1.5 on age mean?* The age distribution moved about 1.5 standard deviations from training.
- *Can drift detection catch concept drift?* Not from inputs alone. It needs labels, through performance monitoring.

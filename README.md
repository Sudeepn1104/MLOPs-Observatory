# Tiny MLOps Observatory

[![CI](https://github.com/Sudeepn1104/MLOPs-Observatory/actions/workflows/ci.yml/badge.svg)](https://github.com/Sudeepn1104/MLOPs-Observatory/actions/workflows/ci.yml)
[![Release image](https://github.com/Sudeepn1104/MLOPs-Observatory/actions/workflows/release.yml/badge.svg)](https://github.com/Sudeepn1104/MLOPs-Observatory/pkgs/container/mlops-observatory)
[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mlops-observatory-sudeepunayak.streamlit.app/)
![Python 3.11 to 3.13](https://img.shields.io/badge/python-3.11%20to%203.13-3776AB)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

The whole life of a production ML model, end to end: a credit-default model trained on **real data** (30,000 credit card clients) is served by an API, watched for **data drift** and **performance drops**, retrained, tried in **shadow** on live traffic, and promoted only when it's safe.

Built with Python, pandas, scikit-learn, SHAP, MLflow, Evidently, FastAPI, Streamlit, plotly, Prometheus, Grafana and Docker.

**Live demo:** https://mlops-observatory-sudeepunayak.streamlit.app/ · **Live API:** [mlops-observatory-api.onrender.com/docs](https://mlops-observatory-api.onrender.com/docs) (free plan: the first request after 15 idle minutes takes about a minute) · **Deploy:** [DEPLOY.md](DEPLOY.md) · **Operate:** [RUNBOOK](docs/RUNBOOK.md) · **Changes:** [CHANGELOG](CHANGELOG.md)

![Demo: drift is detected, a silent failure is caught, v2 is retrained, compared and takes over](screenshots/demo.gif)

## At a glance

| Stage | What's here | Read more |
|---|---|---|
| Data | Real UCI credit data, split into training, test and a stream of unseen clients | [Data](#data) |
| Model | Calibrated gradient boosting, threshold tuned by cross-validation, SHAP explanations, fairness across age groups | [docs/MODEL.md](docs/MODEL.md) |
| Tracking | Every run and model version in MLflow; the `champion` alias is what production serves | [docs/MLFLOW.md](docs/MLFLOW.md) |
| Serving | FastAPI with validation, API keys, request IDs, JSON logs; 76 requests/s at 12 ms median | [Prediction API](#prediction-api) |
| Monitoring | Drift two ways: PSI/KS and Evidently's per-type tests, on batches and on live traffic; Prometheus metrics, Grafana | [docs/DRIFT.md](docs/DRIFT.md), [Grafana](#prometheus-and-grafana) |
| Alerting | 8 unit-tested Prometheus alerts with a runbook; a weekly GitHub Action that opens a drift issue | [docs/RUNBOOK.md](docs/RUNBOOK.md) |
| Release | Retrain, compare on the same holdout, run in **shadow** on live traffic, then promote; rollback in one command | [Shadow deployment](#shadow-deployment) |
| Delivery | Tests on the lowest and highest dependencies, a non-root image with pinned hashes, Trivy scan, SBOM, signed provenance, Render Blueprint | [Docker](#docker), [DEPLOY.md](DEPLOY.md) |

## Architecture

```mermaid
flowchart LR
    data[(Real credit data)] --> train["train.py / retrain.py"]
    train -->|runs, versions| mlflow[("MLflow registry<br/>champion · challenger")]

    client([Client]) -->|POST /predict| api["FastAPI<br/>serve.py"]
    mlflow -->|champion| api
    api -->|answer| client
    mlflow -.->|challenger, after the response| api
    api --> log[("Prediction log<br/>+ shadow comparisons")]
    log --> drift["Live drift<br/>PSI · per-type tests"]

    api -->|/metrics| prom["Prometheus<br/>8 alert rules"] --> grafana[Grafana]
    dash["Streamlit dashboard<br/>monitor · inject drift · retrain"] <--> mlflow
    gha["Weekly GitHub Action<br/>check.py"] -->|drift-alert issue| issue[GitHub issue]

    ci["CI: tests, audit, Evidently check"] --> image["Image on GHCR<br/>Trivy · SBOM · provenance"] --> render[Render]
```

## In simple words

The observatory is a health monitor for a machine-learning model. It watches the model after it is put to work, notices when it starts going wrong, and helps replace it with a better one.

**The model.** It predicts whether a credit card client will miss next month's payment, from their credit limit, age, education, marital status, last three months of repayment status, latest bill and last two payments. It learns from 20,000 real clients.

**The problem.** A model learns from the world as it was, and the world keeps changing. This happens in two ways:

1. **The people change (input drift).** Clients start being older or having lower credit limits than the people the model learned from.
2. **The rules change (concept drift).** Clients look the same, but what makes someone default has changed, for example after an economic shock.

Either way the model quietly gets worse, and nobody notices unless something is watching.

**What the observatory does:**

1. **Sends in new data.** 10 batches of real clients the model has never seen, each a bit more different from the training data.
2. **Checks if the data looks different.** A PSI score per feature shows how far the new data has moved: green (fine), yellow (watch), red (changed a lot).
3. **Checks if the model is still right.** It tracks accuracy and AUC (how well the model separates defaulters from non-defaulters). AUC falls from 0.81 to 0.43, worse than a coin flip.
4. **Sounds the alarm.** If PSI goes above 0.2 or AUC drops more than 0.05, it says "retrain now".
5. **Trains a replacement and keeps it only if it is better.** v2 is trained on recent data and tested against v1 on the same data. It is promoted only if it scores at least as well.
6. **Keeps watching the new model.** New batches are scored by whichever model is active, so you can watch v2 take over and recover. Each model is judged against its own baseline.
7. **Lets you break it on purpose.** Sliders in the Drift injector tab change the data or the rules live, so you can watch the alarms react.
8. **Documents itself.** The Model card tab explains the model's purpose, data and failure modes.

When only the rules change, the data looks normal and PSI stays green while the model fails. That is why the alarm uses both signals (see [Key finding](#key-finding-silent-failure)).

In one sentence: it is a smoke detector for a credit model. It spots when the model's view of the world is out of date, tells you, and safely swaps in an updated model.

## What it does

| Feature | Where |
|---|---|
| PSI heatmap per feature per batch (green / yellow / red) | Monitoring tab |
| Accuracy, AUC and F1 trend lines | Monitoring tab |
| Retrain rule: max PSI > 0.2 or AUC drop > 0.05 | Sidebar and Monitoring tab |
| Live drift injector (input drift and concept drift sliders) | Drift injector tab |
| Stream new batches scored by the active model; "v2 takes over" marker | Monitoring tab |
| Retrain button, auto-retrain, v1 vs v2 comparison, promotion check | Model registry tab |
| Experiment tracking and model registry on MLflow (`champion` alias = model in production) | MLflow UI, Model registry tab |
| Private sandbox per visitor on the live demo | Automatic on Streamlit Cloud |
| Explanations: why the model flags a client (SHAP), and what it relies on overall | Explain tab, `POST /predict?explain=true` |
| Model card with data, calibration, fairness across age groups, limits and failure modes | Model card tab |
| Downloadable drift report (CSV) | Monitoring tab |
| Shadow deployment: a challenger scores live traffic next to the champion, with a promotion verdict | `retrain.py --shadow`, `GET /shadow`, Grafana |
| Grafana dashboard and 8 unit-tested Prometheus alerts, each linked to a runbook | `docker compose --profile monitoring` |
| Backups of the model store and prediction log, rollback in one command | `backup.py`, `shadow.py promote v1` |

## Screenshots

![Overview: v1 under drift, status RED and retrain rule fired](screenshots/overview.png)
![Monitoring: drift heatmap and performance, v2 takes over at batch 10](screenshots/monitoring.png)
![Model registry: v1 vs v2 on the same recent holdout](screenshots/registry.png)
![The same registry in the MLflow UI: version 2 holds the champion alias](screenshots/mlflow_registry.png)
![Silent failure: PSI green but AUC 0.40](screenshots/injector.png)
![Explain: a client 2 months late is flagged at 70% risk; SHAP shows repayment status drives it](screenshots/explain.png)
![Model card: calibration plot and fairness across age groups](screenshots/model_card.png)
![Evidently drift report for batch 9: all 10 features drifted](screenshots/evidently_report.png)
![Grafana: live drift climbing on demo traffic, and a challenger in shadow that would flag 58% of clients where the champion flags 39%](screenshots/grafana.png)

## Key finding: silent failure

When the data **inputs** shift, PSI turns red. When the **rule** that decides default changes (concept drift) but inputs look the same, PSI stays green while AUC falls to about 0.48, worse than random. So the retrain rule uses both signals: PSI as the early warning, AUC as the proof.

## How it works

1. `make_data.py` loads the real dataset and splits it into 20,000 rows for training and 10,000 rows the model never trains on (the stream pool). `make_batch()` draws batches from the pool and can add input drift (resampling real clients, for example more older ones) and concept drift (a share of labels follows a different default rule).
2. `train.py` tunes a gradient-boosting model with a 5-fold cross-validated search, tunes its decision threshold, and evaluates it on a held-out test split: ranking, calibration and fairness across age groups. It logs everything to MLflow (each search candidate is a child run), registers the model as version 1 and points the `champion` alias at it. `model.py` holds the modelling code; [docs/MODEL.md](docs/MODEL.md) explains the choices.
3. `simulate.py` sends 10 batches of 500 clients with growing input drift. Concept drift starts at batch 4. It logs PSI, KS and metrics per batch.
4. `drift.py` computes, per feature, PSI and the KS test, plus the tests Evidently uses by default: normalised Wasserstein distance for amounts and Jensen-Shannon distance for codes such as education or repayment status. `drift_report.py` builds Evidently's interactive HTML drift report in its own environment, because Evidently doesn't support pandas 3 yet; a test proves `drift.py` gives the same scores and decisions. [docs/DRIFT.md](docs/DRIFT.md) explains the design.
5. `monitor.py` applies the retrain rule to the active model, using only the batches it scored itself and its own baseline AUC. It also streams new batches, which keep the drifted conditions of batch 9 and are scored by the active model.
6. `retrain.py` trains a challenger on the last 4 batches and tests both models on the same holdout. Every retrain is an MLflow run and a new registered version. The `champion` alias moves to it only if its AUC is at least as good; otherwise the version is recorded as rejected.
7. `registry.py` is the only module that talks to MLflow. It stores runs and model versions in a SQLite MLflow store (`models/mlflow.db`), and saves models in the skops format, which cannot run code when loaded (unlike pickle). [docs/MLFLOW.md](docs/MLFLOW.md) explains the design.
8. `workspace.py` gives each visitor of the live demo a private sandbox, so one person's Retrain or Reset doesn't change what others see. Locally everything stays in the project folder. Set `OBSERVATORY_SANDBOX=1` or `0` to force it on or off.
9. `app.py` is the dashboard.

PSI thresholds: below 0.1 stable, 0.1 to 0.2 watch, above 0.2 drift.

## Run it

```
pip install -r requirements.txt
python make_data.py
python train.py
python simulate.py
python -m streamlit run app.py
```

Run `train.py` again to reset the registry to v1.

To browse the runs and the registered model in the MLflow UI (http://localhost:5000):

```
pip install mlflow
mlflow ui --backend-store-uri sqlite:///models/mlflow.db
```

## Prediction API

`serve.py` is a FastAPI service that serves the active model:

```
pip install -r requirements-api.txt
uvicorn serve:app --port 8000
```

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness: the process is up |
| `GET /ready` | Readiness: the active model is loaded |
| `GET /version` | App version, git commit, active model |
| `GET /model` 🔑 | Active version, metrics, features, threshold |
| `POST /predict` 🔑 | Scores 1 to 1,000 clients. Inputs are validated (ranges, no unknown fields) |
| `GET /monitoring/drift` 🔑 | PSI and KS of recent requests against the active model's training data |
| `GET /metrics` 🔑 | Prometheus metrics |
| `GET /shadow` 🔑 | Champion vs the shadow challenger on live traffic, with a promotion verdict |
| `POST /shadow/{version}`, `POST /shadow/promote`, `DELETE /shadow` 🛡️ | Start a shadow deployment, promote the challenger, reject it |

🔑 needs an `X-API-Key` header (or `Authorization: Bearer`) when `API_KEY` is set. 🛡️ needs `ADMIN_API_KEY`, a separate key, so a client that can score can't change which model is served; in production these endpoints are off without it. Locally it's optional. In production (`OBSERVATORY_ENV=production`, the Docker image's default) the API refuses to start without a key.

```
curl -X POST localhost:8000/predict -H "X-API-Key: $API_KEY" -H 'content-type: application/json' -d '{"clients": [{"credit_limit": 50000, "age": 29, "education": 2, "marriage": 2, "repay_status_sep": 2, "repay_status_aug": 2, "repay_status_jul": 0, "bill_sep": 48000, "paid_sep": 0, "paid_aug": 1500}]}'
```

Add `?explain=true` to get, for every client, each feature's SHAP contribution in log-odds, largest first; they add up to the prediction. The `default` decision uses the model version's own tuned threshold, which the response reports. Interactive docs are at `/docs`. Every response carries an `X-Request-ID`, and production logs are one JSON line per request. Every scored client is logged to SQLite (`logs/predictions.db`), and `/monitoring/drift` runs the same PSI check as the dashboard on that live traffic. That needs no labels, so it's the early warning in production, where true defaults arrive months later. The API asks the MLflow registry which version holds the `champion` alias (cached for one second), so when a retrain promotes a new version it is served within a second, without a restart.

### Shadow deployment

A model that wins on an offline holdout can still behave very differently on today's traffic. So a new model can run in **shadow** first: the champion answers every request, and the challenger scores the same request after the response has gone out. Both predictions are logged side by side; no client ever sees the challenger's.

```
python retrain.py --shadow     # train v2 and put it in shadow (MLflow alias "challenger")
python shadow.py status        # agreement, share each model flags, offline AUCs, verdict
python shadow.py promote       # or: POST /shadow/promote with the admin key
```

The verdict is `promote`, `review` (it would change the share flagged by more than 10 points), `reject` (worse offline) or `keep_watching` (fewer than 500 clients compared). In the compose demo, v2 beats v1 offline (AUC 0.69 vs 0.58), but in shadow it would flag 60% of clients where v1 flags 35%, so the verdict is `review`, not an automatic promotion. The [runbook](docs/RUNBOOK.md#shipping-a-new-model-shadow-first) walks through it.

## Docker

One image runs either service. Model v1 and the monitoring history are built into the image, so a container serves predictions immediately. It runs as a non-root user, installs exact hash-checked dependencies (`requirements-lock.txt`) on a digest-pinned base image, and has a health check.

```
cp .env.example .env    # set API_KEY
docker compose up --build
```

This starts the API on http://localhost:8000/docs, the dashboard on http://localhost:8501 and the MLflow UI on http://localhost:5000. They share a volume holding the MLflow store, so a model you promote in the dashboard is served by the API on its next request and appears in the MLflow UI. To run one service on its own:

```
docker build -t observatory .
docker run -p 8000:8000 -e API_KEY=<at least 16 characters> observatory
docker run -p 8501:8501 observatory dashboard
```

## Prometheus and Grafana

```
docker compose --profile monitoring --profile demo up --build
```

This adds these services to the stack above:

- **Prometheus** on http://localhost:9090 scrapes the API's `/metrics` every 15 s, sending the API key as a Bearer token.
- **Grafana** on http://localhost:3000 opens straight on a provisioned dashboard (no login to view; admin password from `GRAFANA_ADMIN_PASSWORD`).
- **shadow-init** trains v2 on the recent batches once and puts it in shadow, so the dashboard's shadow row compares it with v1.
- **loadgen** (`loadgen.py`) sends real, unseen clients to `/predict` and ramps drift up over `RAMP_MINUTES` (default 10), following the same scenario as the simulation.

The dashboard shows the model in production, traffic, share flagged, 5xx errors, latency percentiles, the distribution of predicted risk (prediction drift, visible before any label arrives) and live drift per feature, both PSI and the per-type tests that match Evidently. The API recomputes live drift on the last 500 scored clients at most every 15 s when scraped.

A shadow row compares the challenger with the champion on the same traffic: decision agreement, the share each would flag, the mean probability difference, shadow errors and latency.

Prometheus also loads [8 alert rules](monitoring/alerts.yml) (API down, error rate, latency, live drift, PSI above the retrain threshold, a shift in predicted risk, shadow errors and disagreement). Each has a unit test (`promtool test rules`, run in CI) and links to a section of the [runbook](docs/RUNBOOK.md) that says what to do.

The dashboard is generated by `monitoring/build_dashboard.py`, and a test checks that every query uses a metric the API exports, so renaming a metric can't leave a silently empty panel. Leave out `--profile demo` to watch your own traffic.

## Scheduled checks and alerts

`check.py` applies the retrain rule to the active model and prints a report. It exits with 1 when the rule fires, so any scheduler or CI system can act on it:

```
python check.py                              # 0 healthy or waiting, 1 alert, 2 the check failed
python check.py --api-url http://localhost:8000 --json report.json --markdown report.md
```

With `--api-url` it also checks live traffic through the API's `/monitoring/drift`, sending `$API_KEY`. Add the repository secrets `OBSERVATORY_API_URL` and `OBSERVATORY_API_KEY` and the weekly run checks your deployed API too.

`.github/workflows/monitor.yml` runs the check every Monday and on demand (**Actions → Scheduled model check → Run workflow**). It keeps exactly one GitHub issue labelled `drift-alert`:

- **Rule fires:** opens the issue with the report, or updates the open one in place, so there's no comment on every run.
- **Model healthy again:** comments with the new report and closes the issue.

The run has two scenarios. `drifted` checks v1 after the 10 drifted batches, so the alert fires. `retrained` retrains, streams 2 batches to the new model and checks again, so it's healthy and closes the alert. Each run rebuilds from the committed data, so the weekly scheduled run checks the drifted scenario and keeps the alert open until you resolve it.

Command-line helpers used by the workflow, also handy locally:

```
python retrain.py               # train a challenger and promote it if better
python monitor.py stream 3      # stream 3 new batches to the active model
```

## Tests

```
pip install -r requirements-dev.txt
ruff check .
python -m pytest
```

The tests cover the prediction API (validation, logging, live drift, picking up a promoted model), the scheduled check, the MLflow registry (runs, promotion, rejection, reset, pickle-free model files), PSI and the KS check, the dataset (including its checksum) and batch sampler, the simulator, the registry and retraining, and a smoke test of the dashboard itself. GitHub Actions runs them on every push and pull request, on Python 3.11 with the oldest supported library versions and on Python 3.13 with the newest. It also audits the locked dependencies for known vulnerabilities, builds the Docker image and smoke-tests both containers. Releases are published to GitHub Container Registry with a vulnerability scan, an SBOM and build provenance (see [DEPLOY.md](DEPLOY.md)).

## Results

| | Batch 0 | Batch 9 (drifted) |
|---|---|---|
| AUC (v1) | 0.81 | 0.43 |
| F1 (v1) | 0.62 | 0.15 |
| Max PSI | 0.03 | 1.58 |

v1 scores AUC 0.77, F1 0.52 and a calibration error of 0.01 on its own test split: when it says 25%, about 25% of those clients default. Batches are only 500 rows, so batch AUC moves about ±0.03 from noise alone.

After retraining on recent data, v2 reaches AUC 0.69 on the same recent holdout where v1 scores 0.58. On the next three streamed batches v2 scores AUC 0.70 to 0.71 with every feature green again, so the alarm clears.

### Choosing the model

`python compare_models.py` runs every candidate through the same cross-validation, test split and threshold tuning, and logs them to MLflow:

| Model | CV AUC | Calibration error | F1 (tuned threshold) | Size | 1-row latency |
|---|---|---|---|---|---|
| Logistic regression | 0.717 | 0.241 | 0.503 | tiny | fast |
| RandomForest, balanced weights (previous) | 0.770 | 0.213 | 0.534 | 5.4 MB | ~47 ms |
| RandomForest + isotonic calibration | 0.771 | 0.017 | 0.534 | larger | slower |
| **Gradient boosting (current)** | **0.773** | **0.012** | 0.522 | **0.06 MB** | **~1 ms** |

All three tree models rank clients equally well: the differences are inside the cross-validation noise (±0.013), so the ceiling is the information in these 10 features. Gradient boosting was chosen because it is calibrated without an extra layer, its SHAP explanations describe the probability actually served, and it is about 90 times smaller and much faster to serve. Tuning the hyperparameters made no measurable difference; that's reported, not hidden.

### Fairness across age groups (v1, test split)

| Check | Result |
|---|---|
| Selection-rate ratio (four-fifths rule, ≥ 0.8) | 0.77, fails |
| Recall gap between groups (≤ 0.1) | 0.05, passes |
| Calibration gap within groups (≤ 0.05) | 0.01, passes |

The model flags each age group at roughly its real default rate (30–39-year-olds default least and are flagged least). When real rates differ, a model that is calibrated within every group can't also flag every group at the same rate, so these checks can't all pass. The previous model passed the four-fifths check only by over-flagging everyone (its predicted risk was about 0.45 for every group against real rates of 0.19 to 0.24). Which criterion should win is a policy and legal decision; the model card shows all three.

## Data

The UCI [Default of Credit Card Clients](https://doi.org/10.24432/C55S3H) dataset (Yeh and Lien, 2009), committed as `data/credit_default_raw.csv.gz` so nothing is downloaded at runtime. Source, license and checksum are in [`data/SOURCE.md`](data/SOURCE.md). `SEX` is in the dataset but deliberately left out of the model.

## Limits

- The clients and their default labels are real, but the drift is simulated on top: input drift by resampling real clients, concept drift by relabelling with a different rule. The dataset has no real drift over time to replay.
- The data is from Taiwan in 2005. Today's credit behaviour can differ a lot.
- Baseline AUC is about 0.77, a moderate model; different algorithms don't move it (see Choosing the model).
- The decision threshold is tuned for F1, which treats a missed defaulter and a false alarm as equally costly. A lender would set it from the real cost of each.
- Age is a feature, and the four-fifths check across age groups fails (see Fairness). Using this for real decisions would need a legal review.
- Performance metrics need true labels, which arrive late for real credit.
- Details are in the Model card tab.

## Next

- Use Evidently directly in the app once it supports pandas 3 (the agreement test makes the switch safe)

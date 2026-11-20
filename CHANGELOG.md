# Changelog

Versions follow the API's `APP_VERSION` (`GET /version`). Each tag `vX.Y.Z` publishes `ghcr.io/sudeepn1104/mlops-observatory:X.Y.Z`, scanned by Trivy, with an SBOM and signed build provenance.

## 1.2.0 (2026-11-20)

Standard tooling, a stronger and explainable model, and safe model releases.

- **Shadow deployment.** A challenger scores live traffic next to the champion without answering clients, with a promotion verdict (`GET /shadow`, `shadow.py`, `retrain.py --shadow`). Admin endpoints use a separate key. (#29)
- **Alerting.** 8 Prometheus alert rules with `promtool` unit tests in CI, each linked to a [runbook](docs/RUNBOOK.md) section. (#29)
- **Operations.** Online backups and restore (`backup.py`), one-command rollback, container memory and CPU caps, rotated logs, dropped capabilities. (#29)
- **Performance.** 34 → 76 requests/s and a 29 → 12 ms median, found by profiling: cached registry alias lookups and one model pass per request. (#29)
- **Grafana dashboard** on the API's Prometheus metrics: live drift, the predicted-risk distribution, latency, and demo traffic that drifts. (#28)
- **Evidently.** Per-type drift tests (Wasserstein, Jensen-Shannon) reproduced in scipy and checked against Evidently in CI, plus HTML drift reports. (#27)
- **Model.** Calibrated gradient boosting with a cross-validated decision threshold, SHAP explanations, and a fairness report across age groups. (#20, #23)
- **MLflow** for experiment tracking and the model registry; the model in production is the `champion` alias. (#19)
- **Incident fixes** after the new model took the hosted demo down: one process for training, a per-thread OpenMP cap, a stamped model format, and reloading changed modules under a running server. (#21, #22, #25, #26, #27)

## 1.1.0 (2026-11-14)

Deploy-ready API.

- API keys with rotation, request IDs, JSON logs, Prometheus metrics, a request size limit and `/version`. (#11)
- Image published to GHCR with Trivy scanning, an SBOM and provenance; a Render Blueprint. (#11, #14, #17, #18)
- A scheduled check that opens and closes a `drift-alert` issue. (#9)

## 1.0.0 (2026-11-14)

- Prediction API (FastAPI) with input validation, a prediction log and live drift on logged traffic, packaged as a non-root Docker image with hash-pinned dependencies. (#8)
- Real data (UCI credit card default, 30,000 clients), a private sandbox per visitor on the live demo, and monitoring of whichever model is active. (#6, #7)
- Tests, CI on the lowest and highest supported dependencies, and Dependabot. (#2 to #5)

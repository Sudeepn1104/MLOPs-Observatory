"""Prediction API: serves the active model and logs every request for monitoring.

Run:   uvicorn serve:app --port 8000
Docs:  http://localhost:8000/docs

Endpoints (* = needs the X-API-Key header when a key is configured)
  GET  /health            liveness: the process is up
  GET  /ready             readiness: the active model is loaded
  GET  /version           app version, git commit, active model
  GET  /model           * active version, metrics, features
  POST /predict         * score one or more clients (?explain=true adds SHAP reasons)
  GET  /monitoring/drift * PSI of recent requests vs the active model's training data
  GET  /metrics         * Prometheus metrics
  GET  /shadow          * champion vs the shadow challenger on live traffic
  POST /shadow/{version} # put a version in shadow   (# = needs the admin key)
  POST /shadow/promote  # make the challenger the champion
  DELETE /shadow        # reject the challenger

Configuration (environment variables)
  OBSERVATORY_ENV     "production" makes an API key mandatory and logs as JSON
  API_KEY             accepted key(s), comma-separated so keys can be rotated
  ADMIN_API_KEY       key(s) for the endpoints that change the registry; without one
                      they are disabled in production (open in local development)
  MAX_REQUEST_BYTES   largest accepted request body (default 1,000,000)
  LOG_LEVEL           default INFO
  OBSERVATORY_ALIAS_TTL  seconds a registry alias lookup is reused (default 1)
  GIT_SHA             set at image build time, reported by /version

The API always serves the MLflow registry's champion model. When a retrain promotes a
new version (from the dashboard or retrain.py), requests pick it up within
OBSERVATORY_ALIAS_TTL seconds (default 1): no restart needed. Every scored
client is written to a SQLite log, which the
drift endpoint compares with the model's reference data. That's the same PSI
check the dashboard uses, applied to live traffic, and it needs no labels.

Shadow deployment: when the registry's `challenger` alias is set, every request
is also scored by the challenger after the response is sent. Both predictions
are logged side by side (see shadow.py); the client only ever gets the champion's.
"""
import hmac
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

import model as mdl
import registry
import shadow
from drift import check_drift
from make_data import FEATURE_NAMES
from registry import get_active, load_registry

ROOT = Path(os.environ.get("OBSERVATORY_ROOT", "."))
PREDICTION_LOG = Path(os.environ.get("PREDICTION_LOG", "logs/predictions.db"))
MAX_BATCH = 1000
MIN_DRIFT_ROWS = 50
APP_VERSION = "1.2.0"

log = logging.getLogger("observatory.api")


@dataclass(frozen=True)
class Settings:
    environment: str
    api_keys: tuple
    max_request_bytes: int
    log_level: str
    git_sha: str
    admin_keys: tuple = ()

    @property
    def production(self):
        return self.environment == "production"

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        keys = tuple(k.strip() for k in env.get("API_KEY", "").split(",") if k.strip())
        admin = tuple(k.strip() for k in env.get("ADMIN_API_KEY", "").split(",") if k.strip())
        s = cls(
            environment=env.get("OBSERVATORY_ENV", "development").strip().lower(),
            api_keys=keys,
            max_request_bytes=int(env.get("MAX_REQUEST_BYTES", 1_000_000)),
            log_level=env.get("LOG_LEVEL", "INFO").upper(),
            # set at image build time; Render provides RENDER_GIT_COMMIT at runtime
            git_sha=env.get("GIT_SHA") if env.get("GIT_SHA", "unknown") != "unknown"
            else env.get("RENDER_GIT_COMMIT", "unknown"),
            admin_keys=admin,
        )
        if s.production and not s.api_keys:
            # fail fast: never serve an unprotected API in production
            raise RuntimeError("OBSERVATORY_ENV=production requires API_KEY to be set")
        if any(len(k) < 16 for k in s.api_keys):
            raise RuntimeError("every API_KEY must be at least 16 characters")
        if any(len(k) < 16 for k in s.admin_keys):
            raise RuntimeError("every ADMIN_API_KEY must be at least 16 characters")
        if set(s.admin_keys) & set(s.api_keys):
            raise RuntimeError("ADMIN_API_KEY must differ from API_KEY")
        return s


class JsonFormatter(logging.Formatter):
    def format(self, record):
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update(getattr(record, "fields", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out)


def configure_logging(settings):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if settings.production else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger("observatory")
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level)
    root.propagate = False


class Metrics:
    """Prometheus metrics. One registry per app, so tests and reloads stay isolated."""

    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "observatory_http_requests_total", "HTTP requests", ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "observatory_http_request_duration_seconds", "Request latency", ["route"],
            registry=self.registry,
        )
        self.predictions = Counter(
            "observatory_predictions_total", "Clients scored", ["model_version"],
            registry=self.registry,
        )
        self.predicted_defaults = Counter(
            "observatory_predicted_defaults_total", "Clients predicted to default", ["model_version"],
            registry=self.registry,
        )
        self.active_model = Gauge(
            "observatory_active_model_info", "Active model version (value is always 1)", ["version"],
            registry=self.registry,
        )
        self.live_max_psi = Gauge(
            "observatory_live_drift_max_psi", "Max PSI of recent traffic at the last drift check",
            ["model_version"], registry=self.registry,
        )
        self.probability = Histogram(
            "observatory_predicted_probability", "Predicted probability of default per scored client",
            ["model_version"], buckets=[i / 20 for i in range(1, 21)], registry=self.registry,
        )
        self.drift_score = Gauge(
            "observatory_live_drift_score",
            "Per-type drift score of recent traffic per feature (Wasserstein normed or Jensen-Shannon)",
            ["model_version", "feature", "method"], registry=self.registry,
        )
        self.drift_psi = Gauge(
            "observatory_live_drift_psi", "PSI of recent traffic per feature",
            ["model_version", "feature"], registry=self.registry,
        )
        self.drifted_features = Gauge(
            "observatory_live_drift_features_drifted",
            "Features whose per-type drift score is at or above the threshold",
            ["model_version"], registry=self.registry,
        )
        self.drift_rows = Gauge(
            "observatory_live_drift_window_rows", "Scored clients in the last drift check",
            ["model_version"], registry=self.registry,
        )
        self.challenger = Gauge(
            "observatory_challenger_model_info", "Model version scored in shadow (value is always 1)",
            ["version"], registry=self.registry,
        )
        self.shadow_predictions = Counter(
            "observatory_shadow_predictions_total", "Clients scored by the shadow challenger",
            ["champion_version", "challenger_version"], registry=self.registry,
        )
        self.shadow_disagreements = Counter(
            "observatory_shadow_disagreements_total",
            "Clients where the challenger's decision differs from the champion's",
            ["champion_version", "challenger_version"], registry=self.registry,
        )
        self.shadow_defaults = Counter(
            "observatory_shadow_predicted_defaults_total", "Clients the challenger would flag",
            ["challenger_version"], registry=self.registry,
        )
        self.shadow_abs_diff = Counter(
            "observatory_shadow_abs_probability_difference_total",
            "Sum over clients of |challenger - champion| probability (divide by the predictions counter)",
            ["champion_version", "challenger_version"], registry=self.registry,
        )
        self.shadow_errors = Counter(
            "observatory_shadow_errors_total", "Shadow scoring failures (never affect the response)",
            registry=self.registry,
        )
        self.shadow_latency = Histogram(
            "observatory_shadow_duration_seconds", "Time to score one request with the challenger",
            registry=self.registry,
        )
        self._drift_checked_at = 0.0

    def record_drift(self, version, rows, report=None):
        """Publishes a drift check. With no report (too little traffic) only the row count is set."""
        self.drift_rows.clear()
        self.drift_rows.labels(version).set(rows)
        self.drift_score.clear()
        self.drift_psi.clear()
        self.drifted_features.clear()
        if report is None:
            return
        for r in report.itertuples():
            self.drift_score.labels(version, r.feature, r.method).set(r.drift_score)
            self.drift_psi.labels(version, r.feature).set(r.psi)
        self.drifted_features.labels(version).set(int(report["drift_detected"].sum()))
        self.live_max_psi.clear()
        self.live_max_psi.labels(version).set(float(report["psi"].max()))

    def set_active(self, version):
        self.active_model.clear()
        self.active_model.labels(version=version).set(1)

    def set_challenger(self, version):
        self.challenger.clear()
        if version:
            self.challenger.labels(version=version).set(1)


class Client(BaseModel):
    """One credit card client. Field meanings follow the UCI dataset (see data/SOURCE.md)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "credit_limit": 50000,
                "age": 29,
                "education": 2,
                "marriage": 2,
                "repay_status_sep": 2,
                "repay_status_aug": 2,
                "repay_status_jul": 0,
                "bill_sep": 48000,
                "paid_sep": 0,
                "paid_aug": 1500,
            }
        },
    )

    credit_limit: int = Field(gt=0, le=10_000_000, description="Credit limit (NT dollars)")
    age: int = Field(ge=18, le=100)
    education: int = Field(ge=0, le=6, description="1 graduate, 2 university, 3 high school, other")
    marriage: int = Field(ge=0, le=3, description="1 married, 2 single, 3 other, 0 unknown")
    repay_status_sep: int = Field(ge=-2, le=9, description="-2..0 paid on time, 1..9 months late")
    repay_status_aug: int = Field(ge=-2, le=9)
    repay_status_jul: int = Field(ge=-2, le=9)
    bill_sep: int = Field(ge=-1_000_000, le=10_000_000, description="Latest bill (can be negative)")
    paid_sep: int = Field(ge=0, le=10_000_000)
    paid_aug: int = Field(ge=0, le=10_000_000)


assert list(Client.model_fields) == FEATURE_NAMES, "API schema must match the model features"


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clients: list[Client] = Field(min_length=1, max_length=MAX_BATCH)


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float = Field(
        description="SHAP value in log-odds: positive pushes towards default, negative away from it"
    )


class Prediction(BaseModel):
    default_probability: float = Field(description="Calibrated probability of default next month")
    default: bool = Field(description="The decision: probability at or above the model's threshold")
    reasons: list[Reason] | None = Field(
        None, description="With ?explain=true: every feature's contribution, largest effect first"
    )


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    threshold: float = Field(description="Decision threshold of this model version, tuned by CV for F1")
    base_log_odds: float | None = Field(
        None, description="With ?explain=true: the model's average log-odds, before any feature's effect"
    )
    predictions: list[Prediction]


class ModelHolder:
    """Keeps the champion model in memory and swaps it when the champion changes.

    Every call asks the MLflow registry which version the champion alias points
    at (one small query). A different version, or the same version number from
    a different run (after a reset), triggers a reload. If the registry can't
    be read for a moment, the model already in memory keeps serving.
    """

    role = "champion"

    def __init__(self, root, alias_ttl=None):
        self.root = Path(root)
        # how long a registry lookup of the alias is reused. Each lookup is a
        # database query (about 3 ms); a promotion is picked up within this time
        ttl = os.environ.get("OBSERVATORY_ALIAS_TTL", 1.0) if alias_ttl is None else alias_ttl
        self.alias_ttl = float(ttl)
        self._checked = (None, -1e9)  # (key, when)
        self._lock = threading.Lock()

        self._key_loaded = None
        self.entry = None
        self.model = None
        self.reference = None
        self._explainer = None

    def _key(self):
        return registry.champion_key(self.root)

    def _entry(self):
        return get_active(load_registry(self.root))

    def get(self):
        mdl.cap_threads()  # requests run in worker threads; OpenMP's limit is per thread
        try:
            key, when = self._checked
            if time.monotonic() - when >= self.alias_ttl:
                key = self._key()
                self._checked = (key, time.monotonic())
            entry = None if key == self._key_loaded or key is None else self._entry()
        except Exception:
            if self.model is None:
                raise
            log.exception("registry unavailable; serving the model already loaded")
            return self.entry, self.model, self.reference
        if key is None:  # only the challenger can be absent
            return None
        if key != self._key_loaded:
            with self._lock:
                if key != self._key_loaded:
                    model = registry.load_model(entry)
                    reference = registry.load_reference(entry)
                    self.entry, self.model, self.reference, self._key_loaded = entry, model, reference, key
                    log.info(f"{self.role} loaded", extra={"fields": {"model_version": entry["version"]}})
        return self.entry, self.model, self.reference

    def invalidate(self):
        """Forget the cached alias lookup, so the next get() asks the registry."""
        self._checked = (None, -1e9)

    def explainer(self):
        """SHAP explainer for the loaded model, built on first use."""
        model = self.model
        if self._explainer is None or self._explainer[0] is not model:
            self._explainer = (model, mdl.explainer(model))
        return self._explainer[1]


class ChallengerHolder(ModelHolder):
    """The shadow challenger, or None from get() when the alias isn't set."""

    role = "challenger"

    def _key(self):
        return registry.challenger_key(self.root)

    def _entry(self):
        return registry.get_challenger(self.root)

    def get(self):
        got = super().get()
        if got is None:
            self.entry = self.model = self.reference = self._key_loaded = None  # free the memory
        return got


class PredictionLog:
    """Append-only SQLite log of scored clients. Safe across threads and processes."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS predictions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, ts REAL, "
                "model_version TEXT, features TEXT, probability REAL)"
            )
            db.execute(shadow.SCHEMA)

    @contextmanager
    def _connect(self):
        # `with sqlite3.connect()` commits but never closes the connection, which
        # leaks a handle per request in a long-running server. Close explicitly.
        con = sqlite3.connect(self.path, timeout=10)
        try:
            with con:
                yield con
        finally:
            con.close()

    def write(self, request_id, version, clients, probabilities):
        now = time.time()
        rows = [
            (request_id, now, version, json.dumps(c), float(p))
            for c, p in zip(clients, probabilities, strict=True)
        ]
        with self._connect() as db:
            db.executemany(
                "INSERT INTO predictions (request_id, ts, model_version, features, probability) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def write_shadow(self, request_id, champion_version, challenger_version, champion, challenger):
        """champion, challenger: (probabilities, decisions) for the same clients."""
        now = time.time()
        rows = [
            (request_id, now, champion_version, challenger_version, float(cp), float(sp), int(cd), int(sd))
            for cp, cd, sp, sd in zip(*champion, *challenger, strict=True)
        ]
        with self._connect() as db:
            db.executemany(
                "INSERT INTO shadow_predictions (request_id, ts, champion_version, challenger_version, "
                "champion_probability, challenger_probability, champion_default, challenger_default) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def recent(self, version, limit):
        with self._connect() as db:
            rows = db.execute(
                "SELECT features FROM predictions WHERE model_version = ? "
                "ORDER BY id DESC LIMIT ?",
                (version, limit),
            ).fetchall()
        return pd.DataFrame([json.loads(r[0]) for r in rows], columns=FEATURE_NAMES)


@asynccontextmanager
async def lifespan(app):
    settings = Settings.from_env()
    configure_logging(settings)
    app.state.settings = settings
    app.state.metrics = Metrics()
    app.state.holder = ModelHolder(ROOT)
    app.state.challenger = ChallengerHolder(ROOT)
    app.state.log = PredictionLog(PREDICTION_LOG)
    try:
        entry, _, _ = app.state.holder.get()  # load once at startup so /ready is true immediately
        app.state.metrics.set_active(entry["version"])
    except Exception:  # keep serving /health; /ready reports the problem
        log.exception("model could not be loaded at startup")
    log.info(
        "api started",
        extra={"fields": {
            "environment": settings.environment, "auth": bool(settings.api_keys),
            "git_sha": settings.git_sha, "version": APP_VERSION,
        }},
    )
    yield
    log.info("api stopping")


app = FastAPI(
    title="Tiny MLOps Observatory: prediction API",
    version=APP_VERSION,
    description=(
        "Serves the active credit-default model and monitors live traffic for drift. "
        "Protected endpoints need an `X-API-Key` header: use **Authorize** above."
    ),
    lifespan=lifespan,
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    request: Request,
    key: str | None = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """Checks the key in constant time. Open when no key is configured (local dev).

    Accepts `X-API-Key: <key>` or `Authorization: Bearer <key>`; the second is
    what Prometheus sends when it scrapes /metrics.
    """
    keys = request.app.state.settings.api_keys
    if not keys:
        return
    key = key if key is not None else (bearer.credentials if bearer else None)
    if key is None or not any(hmac.compare_digest(key, k) for k in keys):
        raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})


protected = [Depends(require_api_key)]


def require_admin_key(
    request: Request,
    key: str | None = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    """For endpoints that change which model is served. A separate key, so a
    client that can score can't also promote models."""
    settings = request.app.state.settings
    if not settings.admin_keys:
        if settings.production:
            raise HTTPException(403, "admin endpoints are disabled: set ADMIN_API_KEY to enable them")
        return  # local development
    key = key if key is not None else (bearer.credentials if bearer else None)
    if key is None or not any(hmac.compare_digest(key, k) for k in settings.admin_keys):
        raise HTTPException(401, "missing or invalid admin key", headers={"WWW-Authenticate": "ApiKey"})


admin = [Depends(require_admin_key)]


@app.middleware("http")
async def observe(request: Request, call_next):
    """Request ID, body size limit, access log and latency metrics for every request."""
    request_id = (request.headers.get("x-request-id") or uuid.uuid4().hex)[:128]
    request.state.request_id = request_id  # one ID for logs, response and prediction log
    settings = request.app.state.settings
    length = request.headers.get("content-length")
    start = time.perf_counter()
    if length and length.isdigit() and int(length) > settings.max_request_bytes:
        response = JSONResponse(
            {"detail": f"request body larger than {settings.max_request_bytes} bytes"}, status_code=413
        )
    else:
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"fields": {"request_id": request_id}})
            response = JSONResponse({"detail": "internal error", "request_id": request_id}, 500)
    elapsed = time.perf_counter() - start
    route = getattr(request.scope.get("route"), "path", "unmatched")
    metrics = request.app.state.metrics
    metrics.requests.labels(request.method, route, str(response.status_code)).inc()
    metrics.latency.labels(route).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    if route not in ("/health", "/ready", "/metrics"):
        log.info(
            "request",
            extra={"fields": {
                "request_id": request_id, "method": request.method, "route": route,
                "status": response.status_code, "ms": round(elapsed * 1000, 1),
            }},
        )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        entry, _, _ = app.state.holder.get()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"model not loaded: {e}") from e
    return {"status": "ready", "model_version": entry["version"]}


@app.get("/version")
def version():
    try:
        model_version = app.state.holder.get()[0]["version"]
    except Exception:  # noqa: BLE001
        model_version = None
    return {"version": APP_VERSION, "git_sha": app.state.settings.git_sha, "model_version": model_version}


DRIFT_REFRESH_SECONDS = 15
DRIFT_WINDOW = 500


def _check_live_drift(window):
    """Drift of the active model's last `window` scored clients. Returns (entry, rows, report or None)."""
    entry, _, reference = app.state.holder.get()
    recent = app.state.log.recent(entry["version"], window)
    if len(recent) < MIN_DRIFT_ROWS:
        app.state.metrics.record_drift(entry["version"], len(recent))
        return entry, len(recent), None
    report = check_drift(reference, recent)
    app.state.metrics.record_drift(entry["version"], len(recent), report)
    return entry, len(recent), report


@app.get("/metrics", dependencies=protected, include_in_schema=False)
def metrics():
    """Prometheus metrics. Live drift is recomputed here, at most every DRIFT_REFRESH_SECONDS,
    so a scraper sees drift without anyone calling /monitoring/drift."""
    m = app.state.metrics
    now = time.monotonic()
    if now - m._drift_checked_at >= DRIFT_REFRESH_SECONDS:
        m._drift_checked_at = now
        try:
            _check_live_drift(DRIFT_WINDOW)
        except Exception:  # metrics must keep working even if the drift check can't run
            log.exception("live drift check for /metrics failed")
    return Response(generate_latest(m.registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/model", dependencies=protected)
def model_info():
    entry, _, _ = app.state.holder.get()
    return {
        "version": entry["version"],
        "trained_at": entry["trained_at"],
        "trigger": entry["trigger"],
        "mlflow_run_id": entry["run_id"],
        "registered_model": registry.MODEL_NAME,
        "alias": registry.ALIAS,
        "metrics": entry["metrics"],
        "evaluated_on": entry["evaluated_on"],
        "features": FEATURE_NAMES,
        "threshold": mdl.threshold(app.state.holder.get()[1]),
    }


@app.post("/predict", response_model=PredictResponse, dependencies=protected)
def predict(
    req: PredictRequest,
    request: Request,
    background: BackgroundTasks,
    explain: bool = Query(False, description="Add each feature's SHAP contribution to every prediction"),
):
    entry, model, _ = app.state.holder.get()
    app.state.metrics.set_active(entry["version"])
    clients = [c.model_dump() for c in req.clients]
    X = pd.DataFrame(clients, columns=FEATURE_NAMES)
    proba = model.predict_proba(X)[:, 1]
    threshold = mdl.threshold(model)
    decisions = proba >= threshold  # what model.predict does, without scoring twice
    request_id = request.state.request_id
    app.state.log.write(request_id, entry["version"], clients, proba)
    app.state.metrics.predictions.labels(entry["version"]).inc(len(clients))
    app.state.metrics.predicted_defaults.labels(entry["version"]).inc(int(decisions.sum()))
    for p in proba:
        app.state.metrics.probability.labels(entry["version"]).observe(float(p))
    # after the response is sent, so the challenger never adds latency or errors
    background.add_task(score_shadow, request_id, entry["version"], X, proba, decisions)
    reasons, base = [None] * len(clients), None
    if explain:
        contrib, base = mdl.contributions(model, X, app.state.holder.explainer())
        reasons = [
            [
                Reason(feature=f, value=float(X.iloc[i][f]), contribution=round(float(c), 4))
                for f, c in row.sort_values(key=abs, ascending=False).items()
            ]
            for i, (_, row) in enumerate(contrib.iterrows())
        ]
    return PredictResponse(
        request_id=request_id,
        model_version=entry["version"],
        threshold=round(threshold, 4),
        base_log_odds=None if base is None else round(base, 4),
        predictions=[
            Prediction(default_probability=round(float(p), 4), default=bool(d), reasons=r)
            for p, d, r in zip(proba, decisions, reasons, strict=True)
        ],
    )


def score_shadow(request_id, champion_version, X, proba, decisions):
    """Scores a request with the challenger, if there is one, and logs both predictions."""
    m = app.state.metrics
    try:
        got = app.state.challenger.get()
        if got is None or got[0]["version"] == champion_version:
            m.set_challenger(None)
            return
        entry, model, _ = got
        version = entry["version"]
        m.set_challenger(version)
        start = time.perf_counter()
        s_proba = model.predict_proba(X)[:, 1]
        s_decisions = s_proba >= mdl.threshold(model)
        m.shadow_latency.observe(time.perf_counter() - start)
        app.state.log.write_shadow(
            request_id, champion_version, version, (proba, decisions), (s_proba, s_decisions)
        )
        m.shadow_predictions.labels(champion_version, version).inc(len(X))
        m.shadow_disagreements.labels(champion_version, version).inc(int((s_decisions != decisions).sum()))
        m.shadow_defaults.labels(version).inc(int(s_decisions.sum()))
        m.shadow_abs_diff.labels(champion_version, version).inc(float(abs(s_proba - proba).sum()))
    except Exception:  # the champion's answer has already gone out; never let this fail loudly
        m.shadow_errors.inc()
        log.exception("shadow scoring failed", extra={"fields": {"request_id": request_id}})


def _invalidate():
    app.state.holder.invalidate()
    app.state.challenger.invalidate()


@app.get("/shadow", dependencies=protected)
def shadow_status():
    """Champion vs the shadow challenger on live traffic, with a promotion verdict.

    Verdicts: promote, review (behaviour changed a lot), reject (worse offline),
    keep_watching (not enough shadow traffic yet).
    """
    s = shadow.status(ROOT, app.state.log.path)
    return s if s is not None else {"challenger": None, "detail": "no challenger in shadow"}


@app.post("/shadow/promote", dependencies=admin)
def shadow_promote():
    """Makes the challenger the champion. The API serves it from the next request."""
    challenger = registry.get_challenger(ROOT)
    if challenger is None:
        raise HTTPException(409, "no challenger in shadow")
    e = registry.promote(challenger["version"], ROOT)
    _invalidate()
    log.info("challenger promoted", extra={"fields": {"model_version": e["version"]}})
    return {"champion": e["version"]}


@app.post("/shadow/{version}", dependencies=admin)
def shadow_start(version: str):
    """Puts a registered version in shadow."""
    try:
        e = registry.start_shadow(version, ROOT)
        _invalidate()
    except ValueError as err:
        raise HTTPException(409, str(err)) from err
    except Exception as err:  # noqa: BLE001 - unknown version
        raise HTTPException(404, f"no model version {version}") from err
    log.info("challenger in shadow", extra={"fields": {"model_version": e["version"]}})
    return {"challenger": e["version"]}


@app.delete("/shadow", dependencies=admin)
def shadow_stop():
    """Rejects the challenger and stops shadow scoring."""
    e = registry.stop_shadow(ROOT)
    _invalidate()
    if e is None:
        raise HTTPException(409, "no challenger in shadow")
    return {"rejected": e["version"]}


@app.get("/monitoring/drift", dependencies=protected)
def live_drift(window: int = Query(500, ge=MIN_DRIFT_ROWS, le=50_000)):
    """PSI and KS of the last `window` scored clients vs the active model's training data.

    This works without labels, so it is the early warning in production, where
    true defaults arrive months later.
    """
    entry, rows, report = _check_live_drift(window)
    if report is None:
        return {
            "model_version": entry["version"],
            "rows": rows,
            "status": "insufficient_data",
            "detail": f"need at least {MIN_DRIFT_ROWS} scored clients for this model",
        }
    worst = report.loc[report["psi"].idxmax()]
    return {
        "model_version": entry["version"],
        "rows": rows,
        "status": worst["status"],
        "max_psi": float(worst["psi"]),
        "worst_feature": worst["feature"],
        "features": report.to_dict(orient="records"),
    }

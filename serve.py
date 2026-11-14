"""Prediction API: serves the active model and logs every request for monitoring.

Run:   uvicorn serve:app --port 8000
Docs:  http://localhost:8000/docs

Endpoints (* = needs the X-API-Key header when a key is configured)
  GET  /health            liveness: the process is up
  GET  /ready             readiness: the active model is loaded
  GET  /version           app version, git commit, active model
  GET  /model           * active version, metrics, features
  POST /predict         * score one or more clients
  GET  /monitoring/drift * PSI of recent requests vs the active model's training data
  GET  /metrics         * Prometheus metrics

Configuration (environment variables)
  OBSERVATORY_ENV     "production" makes an API key mandatory and logs as JSON
  API_KEY             accepted key(s), comma-separated so keys can be rotated
  MAX_REQUEST_BYTES   largest accepted request body (default 1,000,000)
  LOG_LEVEL           default INFO
  GIT_SHA             set at image build time, reported by /version

The API always serves the registry's active model. When a retrain promotes a
new version (from the dashboard or retrain.py), the next request picks it up:
no restart needed. Every scored client is written to a SQLite log, which the
drift endpoint compares with the model's reference data. That's the same PSI
check the dashboard uses, applied to live traffic, and it needs no labels.
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

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from drift import check_drift
from make_data import FEATURE_NAMES
from registry import get_active, load_registry, registry_path

ROOT = Path(os.environ.get("OBSERVATORY_ROOT", "."))
PREDICTION_LOG = Path(os.environ.get("PREDICTION_LOG", "logs/predictions.db"))
THRESHOLD = 0.5
MAX_BATCH = 1000
MIN_DRIFT_ROWS = 50
APP_VERSION = "1.1.0"

log = logging.getLogger("observatory.api")


@dataclass(frozen=True)
class Settings:
    environment: str
    api_keys: tuple
    max_request_bytes: int
    log_level: str
    git_sha: str

    @property
    def production(self):
        return self.environment == "production"

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        keys = tuple(k.strip() for k in env.get("API_KEY", "").split(",") if k.strip())
        s = cls(
            environment=env.get("OBSERVATORY_ENV", "development").strip().lower(),
            api_keys=keys,
            max_request_bytes=int(env.get("MAX_REQUEST_BYTES", 1_000_000)),
            log_level=env.get("LOG_LEVEL", "INFO").upper(),
            # set at image build time; Render provides RENDER_GIT_COMMIT at runtime
            git_sha=env.get("GIT_SHA") if env.get("GIT_SHA", "unknown") != "unknown"
            else env.get("RENDER_GIT_COMMIT", "unknown"),
        )
        if s.production and not s.api_keys:
            # fail fast: never serve an unprotected API in production
            raise RuntimeError("OBSERVATORY_ENV=production requires API_KEY to be set")
        if any(len(k) < 16 for k in s.api_keys):
            raise RuntimeError("every API_KEY must be at least 16 characters")
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

    def set_active(self, version):
        self.active_model.clear()
        self.active_model.labels(version=version).set(1)


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


class Prediction(BaseModel):
    default_probability: float
    default: bool


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    threshold: float
    predictions: list[Prediction]


class ModelHolder:
    """Keeps the active model in memory and reloads it when the registry changes."""

    def __init__(self, root):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._stamp = None
        self.entry = None
        self.model = None
        self.reference = None

    def _registry_stamp(self):
        path = registry_path(self.root)
        return path.stat().st_mtime_ns if path.exists() else None

    def get(self):
        stamp = self._registry_stamp()
        if self.model is None or stamp != self._stamp:
            with self._lock:
                if self.model is None or stamp != self._stamp:
                    entry = get_active(load_registry(self.root))
                    self.model = joblib.load(entry["file"])
                    self.reference = pd.read_csv(entry["reference"])
                    self.entry = entry
                    self._stamp = self._registry_stamp()
        return self.entry, self.model, self.reference


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
    app.state.log = PredictionLog(PREDICTION_LOG)
    try:
        entry, _, _ = app.state.holder.get()  # load once at startup so /ready is true immediately
        app.state.metrics.set_active(entry["version"])
        log.info("model loaded", extra={"fields": {"model_version": entry["version"]}})
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


def require_api_key(request: Request, key: str | None = Security(_api_key_header)):
    """Checks X-API-Key in constant time. Open when no key is configured (local dev)."""
    keys = request.app.state.settings.api_keys
    if not keys:
        return
    if key is None or not any(hmac.compare_digest(key, k) for k in keys):
        raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})


protected = [Depends(require_api_key)]


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


@app.get("/metrics", dependencies=protected, include_in_schema=False)
def metrics():
    return Response(generate_latest(app.state.metrics.registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/model", dependencies=protected)
def model_info():
    entry, _, _ = app.state.holder.get()
    return {
        "version": entry["version"],
        "trained_at": entry["trained_at"],
        "trigger": entry["trigger"],
        "metrics": entry["metrics"],
        "evaluated_on": entry["evaluated_on"],
        "features": FEATURE_NAMES,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse, dependencies=protected)
def predict(req: PredictRequest, request: Request):
    entry, model, _ = app.state.holder.get()
    app.state.metrics.set_active(entry["version"])
    clients = [c.model_dump() for c in req.clients]
    X = pd.DataFrame(clients, columns=FEATURE_NAMES)
    proba = model.predict_proba(X)[:, 1]
    request_id = request.state.request_id
    app.state.log.write(request_id, entry["version"], clients, proba)
    app.state.metrics.predictions.labels(entry["version"]).inc(len(clients))
    app.state.metrics.predicted_defaults.labels(entry["version"]).inc(int((proba >= THRESHOLD).sum()))
    return PredictResponse(
        request_id=request_id,
        model_version=entry["version"],
        threshold=THRESHOLD,
        predictions=[
            Prediction(default_probability=round(float(p), 4), default=bool(p >= THRESHOLD))
            for p in proba
        ],
    )


@app.get("/monitoring/drift", dependencies=protected)
def live_drift(window: int = Query(500, ge=MIN_DRIFT_ROWS, le=50_000)):
    """PSI and KS of the last `window` scored clients vs the active model's training data.

    This works without labels, so it is the early warning in production, where
    true defaults arrive months later.
    """
    entry, _, reference = app.state.holder.get()
    recent = app.state.log.recent(entry["version"], window)
    if len(recent) < MIN_DRIFT_ROWS:
        return {
            "model_version": entry["version"],
            "rows": len(recent),
            "status": "insufficient_data",
            "detail": f"need at least {MIN_DRIFT_ROWS} scored clients for this model",
        }
    report = check_drift(reference, recent)
    worst = report.loc[report["psi"].idxmax()]
    app.state.metrics.live_max_psi.labels(entry["version"]).set(float(worst["psi"]))
    return {
        "model_version": entry["version"],
        "rows": len(recent),
        "status": worst["status"],
        "max_psi": float(worst["psi"]),
        "worst_feature": worst["feature"],
        "features": report.to_dict(orient="records"),
    }

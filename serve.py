"""Prediction API: serves the active model and logs every request for monitoring.

Run:   uvicorn serve:app --port 8000
Docs:  http://localhost:8000/docs

Endpoints
  GET  /health            liveness: the process is up
  GET  /ready             readiness: the active model is loaded
  GET  /model             active version, metrics, features
  POST /predict           score one or more clients
  GET  /monitoring/drift  PSI of recent requests vs the active model's training data

The API always serves the registry's active model. When a retrain promotes a
new version (from the dashboard or retrain.py), the next request picks it up:
no restart needed. Every scored client is written to a SQLite log, which the
drift endpoint compares with the model's reference data. That's the same PSI
check the dashboard uses, applied to live traffic, and it needs no labels.
"""
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from drift import check_drift
from make_data import FEATURE_NAMES
from registry import get_active, load_registry, registry_path

ROOT = Path(os.environ.get("OBSERVATORY_ROOT", "."))
PREDICTION_LOG = Path(os.environ.get("PREDICTION_LOG", "logs/predictions.db"))
THRESHOLD = 0.5
MAX_BATCH = 1000
MIN_DRIFT_ROWS = 50


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

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

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
    app.state.holder = ModelHolder(ROOT)
    app.state.log = PredictionLog(PREDICTION_LOG)
    try:
        app.state.holder.get()  # load once at startup so /ready is true immediately
    except Exception:  # noqa: BLE001 - keep serving /health; /ready reports the problem
        pass
    yield


app = FastAPI(
    title="Tiny MLOps Observatory: prediction API",
    version="1.0.0",
    description="Serves the active credit-default model and monitors live traffic for drift.",
    lifespan=lifespan,
)


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


@app.get("/model")
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


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    entry, model, _ = app.state.holder.get()
    clients = [c.model_dump() for c in req.clients]
    X = pd.DataFrame(clients, columns=FEATURE_NAMES)
    proba = model.predict_proba(X)[:, 1]
    request_id = uuid.uuid4().hex
    app.state.log.write(request_id, entry["version"], clients, proba)
    return PredictResponse(
        request_id=request_id,
        model_version=entry["version"],
        threshold=THRESHOLD,
        predictions=[
            Prediction(default_probability=round(float(p), 4), default=bool(p >= THRESHOLD))
            for p in proba
        ],
    )


@app.get("/monitoring/drift")
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
    return {
        "model_version": entry["version"],
        "rows": len(recent),
        "status": worst["status"],
        "max_psi": float(worst["psi"]),
        "worst_feature": worst["feature"],
        "features": report.to_dict(orient="records"),
    }

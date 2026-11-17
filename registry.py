"""Model registry and experiment tracking, backed by MLflow.

Every training and retraining run is an MLflow run (parameters, metrics, the
model and its drift-reference data as artifacts). Every model is a version of
the registered model `credit-default`, and the model in production is the
version the `champion` alias points at. Promotion is moving that alias.

Storage is a SQLite MLflow store per workspace `root`:

    root/models/mlflow.db        runs, metrics, registered model versions
    root/models/mlartifacts/     model files and reference data

The default root "." is the project directory, used by the scripts, the API
and a local dashboard. On the live demo each visitor gets a private root (see
workspace.py) whose store is seeded with a link to the project's v1, so one
visitor's retrains never change what another visitor sees.

Browse a store with the MLflow UI:

    pip install mlflow
    mlflow ui --backend-store-uri sqlite:///models/mlflow.db

The rest of the code talks to this module, not to MLflow directly. It only
uses explicit MlflowClient objects (never mlflow.set_tracking_uri), because
the dashboard serves several workspaces from one process.
"""
import json
import os
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

# must be set before mlflow is imported
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # no usage data sent to the MLflow project
os.environ.setdefault("DO_NOT_TRACK", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep MLflow's start-up hint out of logs

import mlflow.artifacts  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

import model as _model  # noqa: E402, F401 - importing it caps OpenMP threads in every process that loads a model

MODEL_NAME = "credit-default"
EXPERIMENT = "credit-default"
ALIAS = "champion"

# Models are saved in the skops format, which (unlike pickle) cannot run code
# when loaded. Only types from these packages are trusted inside a model file.
TRUSTED_PREFIXES = ("sklearn.", "numpy.", "scipy.")


def store_dir(root="."):
    return Path(root) / "models"


def db_path(root="."):
    return store_dir(root) / "mlflow.db"


def tracking_uri(root="."):
    return f"sqlite:///{db_path(root).resolve()}"


def client(root="."):
    uri = tracking_uri(root)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def is_project(root):
    return Path(root).resolve() == Path(".").resolve()


def exists(root="."):
    """True if this workspace has a store with a registered model."""
    if not db_path(root).exists():
        return False
    try:
        client(root).get_registered_model(MODEL_NAME)
        return True
    except MlflowException:
        return False


# ---------- tags <-> entry ----------
def _to_tags(fields):
    """Model version tags: plain strings stay readable in the MLflow UI, the rest is JSON."""
    return {
        k: (v if isinstance(v, str) else json.dumps(v)) for k, v in fields.items() if v is not None
    }


def _from_tags(tags):
    out = {}
    for k, v in tags.items():
        try:
            out[k] = json.loads(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def _entry(mv):
    e = {"train_rows": None, **_from_tags(mv.tags)}
    e.update(
        version=f"v{mv.version}",
        mlflow_version=int(mv.version),
        run_id=mv.run_id or e.get("source_run_id"),
        model_uri=mv.source,
    )
    return e


def _experiment_id(c, root):
    exp = c.get_experiment_by_name(EXPERIMENT)
    if exp is not None:
        return exp.experiment_id
    artifacts = (store_dir(root) / "mlartifacts").resolve()
    return c.create_experiment(EXPERIMENT, artifact_location=artifacts.as_uri())


def _trusted_types(model):
    """The non-default types inside the model file, if all come from trusted packages."""
    from skops.io import dumps, get_untrusted_types

    types = get_untrusted_types(data=dumps(model))
    bad = [t for t in types if not t.startswith(TRUSTED_PREFIXES)]
    if bad:
        raise ValueError(f"model contains types outside {TRUSTED_PREFIXES}: {bad}")
    return types


# ---------- writing ----------
def register_version(model, reference, fields, params=None, promote=True, root=".", run_name=None,
                     artifacts=None, search=None):
    """Logs one training run to MLflow and registers its model as a new version.

    model:     fitted scikit-learn estimator
    reference: the training features (DataFrame), kept as the drift baseline
    fields:    registry entry fields (trigger, metrics, evaluated_on, ...)
    params:    hyperparameters and settings, logged as run parameters
    promote:   True moves the champion alias to the new version
    artifacts: {"name.csv": DataFrame, "name.json": dict} logged under evaluation/
    search:    hyperparameter search results (DataFrame, one row per candidate),
               logged as child runs of this run so the MLflow UI can compare them

    Returns the registry entry of the new version.
    """
    c = client(root)
    store_dir(root).mkdir(parents=True, exist_ok=True)
    exp_id = _experiment_id(c, root)
    try:
        c.get_registered_model(MODEL_NAME)
    except MlflowException:
        c.create_registered_model(
            MODEL_NAME, description="Predicts whether a credit card client defaults next month."
        )

    run = c.create_run(exp_id, run_name=run_name, tags={"trigger": str(fields.get("trigger", ""))})
    run_id = run.info.run_id
    try:
        for k, v in (params or {}).items():
            c.log_param(run_id, k, v)
        for k, v in fields.get("metrics", {}).items():
            c.log_metric(run_id, k, v)
        for k, v in (fields.get("champion_metrics_on_holdout") or {}).items():
            c.log_metric(run_id, f"champion_{k}", v)

        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            # MLflow hints that integer columns can't hold missing values; the API
            # validates every field as a required integer, so that can't happen
            warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
            model_dir = Path(tmp) / "model"
            mlflow.sklearn.save_model(
                model,
                str(model_dir),
                serialization_format="skops",
                skops_trusted_types=_trusted_types(model),
                signature=infer_signature(reference.head(100), model.predict_proba(reference.head(100))),
                pip_requirements=["scikit-learn", "skops"],
            )
            c.log_artifacts(run_id, str(model_dir), "model")
            ref_file = Path(tmp) / "reference.csv"
            reference.to_csv(ref_file, index=False)
            c.log_artifact(run_id, str(ref_file), "reference")
            for name, obj in (artifacts or {}).items():
                path = Path(tmp) / name
                if isinstance(obj, pd.DataFrame):
                    obj.to_csv(path, index=False)
                else:
                    path.write_text(json.dumps(obj, indent=2))
                c.log_artifact(run_id, str(path), "evaluation")
        if search is not None:
            _log_search(c, exp_id, run_id, search)
        c.set_terminated(run_id, "FINISHED")
    except BaseException:
        c.set_terminated(run_id, "FAILED")
        raise

    artifact_uri = run.info.artifact_uri
    champion = _champion(c)
    tags = _to_tags(
        {
            **fields,
            "status": "active" if promote else "rejected",
            "trained_at": fields.get("trained_at") or datetime.now().isoformat(timespec="seconds"),
            "reference_uri": f"{artifact_uri}/reference/reference.csv",
        }
    )
    mv = c.create_model_version(MODEL_NAME, source=f"{artifact_uri}/model", run_id=run_id, tags=tags)
    c.set_tag(run_id, "model_version", f"v{mv.version}")
    if promote:
        if champion is not None:
            c.set_model_version_tag(MODEL_NAME, champion.version, "status", "archived")
        c.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
    return _entry(c.get_model_version(MODEL_NAME, mv.version))


def _log_search(c, exp_id, parent_id, results):
    """One child run per hyperparameter candidate, nested under the training run."""
    metric_cols = {
        "mean_test_score": "cv_auc", "std_test_score": "cv_auc_std", "mean_fit_time": "fit_seconds",
    }
    for i, row in results.iterrows():
        child = c.create_run(exp_id, run_name=f"candidate {i + 1}", tags={"mlflow.parentRunId": parent_id})
        cid = child.info.run_id
        for k, v in row.items():
            if k in metric_cols:
                c.log_metric(cid, metric_cols[k], float(v))
            elif k != "rank_test_score":
                c.log_param(cid, k, v)
        c.set_tag(cid, "rank", int(row["rank_test_score"]))
        c.set_terminated(cid, "FINISHED")


def load_artifact(entry, name):
    """An evaluation artifact of a version's run (CSV -> DataFrame, JSON -> dict), or None."""
    if not entry.get("run_id"):
        return None
    run_uri = entry["model_uri"].rsplit("/model", 1)[0]
    try:
        path = Path(_local(f"{run_uri}/evaluation/{name}"))
    except Exception:  # noqa: BLE001 - older versions have no such artifact
        return None
    return pd.read_csv(path) if name.endswith(".csv") else json.loads(path.read_text())


def _champion(c):
    try:
        return c.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except MlflowException:
        return None


def _seed_from_project(root):
    """Gives a new workspace a registry whose v1 links to the project's v1 (no copy)."""
    if is_project(root) or not exists("."):
        raise FileNotFoundError(
            "No model registry found. Run: python make_data.py, python train.py, python simulate.py"
        )
    v1 = client(".").get_model_version(MODEL_NAME, "1")
    c = client(root)
    store_dir(root).mkdir(parents=True, exist_ok=True)
    _experiment_id(c, root)
    c.create_registered_model(MODEL_NAME, description="Private workspace copy of the registry.")
    tags = {**v1.tags, "status": "active", "source_run_id": v1.run_id or ""}
    mv = c.create_model_version(MODEL_NAME, source=v1.source, tags=tags)
    c.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)


# ---------- reading ----------
def load_registry(root="."):
    """{"active": "vN", "models": [entry, ...]} with entries ordered by version."""
    if not exists(root):
        _seed_from_project(root)
    c = client(root)
    versions = sorted(c.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))
    champion = _champion(c)
    return {
        "active": f"v{champion.version}" if champion else None,
        "models": [_entry(v) for v in versions],
    }


def get_active(reg):
    return next(m for m in reg["models"] if m["version"] == reg["active"])


def champion_key(root="."):
    """Cheap check (one query) of who the champion is: ("v2", run_id).

    The run ID is part of the key because version numbers restart after a reset.
    """
    champion = _champion(client(root))
    if champion is None:
        raise LookupError("no champion model is registered")
    return f"v{champion.version}", champion.run_id or champion.tags.get("source_run_id")


def _local(uri):
    return mlflow.artifacts.download_artifacts(artifact_uri=uri)


def load_model(entry):
    """The fitted scikit-learn model of a registry entry."""
    return mlflow.sklearn.load_model(_local(entry["model_uri"]))


def load_reference(entry):
    """The training features of a registry entry: its drift baseline."""
    return pd.read_csv(_local(entry["reference_uri"]))


# ---------- reset ----------
def reset_registry(root="."):
    """Back to v1 only (handy for re-running the demo). Never deletes v1 or its run."""
    if not exists(root):
        return
    c = client(root)
    v1 = c.get_model_version(MODEL_NAME, "1")
    keep_run = v1.run_id

    # deleting single versions would leave gaps (the next model would be v3),
    # so re-create the registered model with v1 alone
    c.delete_registered_model(MODEL_NAME)
    c.create_registered_model(MODEL_NAME, description="Predicts whether a credit card client defaults.")
    mv = c.create_model_version(
        MODEL_NAME, source=v1.source, run_id=keep_run, tags={**v1.tags, "status": "active"}
    )
    c.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)

    exp = c.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return
    workspace = store_dir(root).resolve()
    for run in c.search_runs([exp.experiment_id]):
        if run.info.run_id == keep_run:
            continue
        c.delete_run(run.info.run_id)
        artifacts = Path(_strip_file_scheme(run.info.artifact_uri)).parent
        if workspace in artifacts.resolve().parents:  # only ever delete inside this workspace
            shutil.rmtree(artifacts, ignore_errors=True)


def _strip_file_scheme(uri):
    return uri[len("file://"):] if uri.startswith("file://") else uri


def reconnect():
    """Drops every open database connection held by this process.

    MLflow keeps connections open per store. When another process replaces a
    store's database file (train.py does, via wipe()), a process that was
    already running would keep reading the old, deleted file. Call this after
    such a rebuild; the next query reopens the file by path.
    """
    import gc

    from sqlalchemy.engine import Engine

    for obj in gc.get_objects():
        # type(), not isinstance(): isinstance reads obj.__class__, which runs
        # code on lazy-import proxies (SHAP's plotting module raises ImportError)
        if issubclass(type(obj), Engine):
            obj.dispose()


def wipe(root="."):
    """Deletes the whole store. Only for a fresh start in a new process (train.py)."""
    db_path(root).unlink(missing_ok=True)
    shutil.rmtree(store_dir(root) / "mlartifacts", ignore_errors=True)

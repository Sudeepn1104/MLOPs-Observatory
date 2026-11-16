# How MLflow is used here, and why

This note explains the design so you can reason about it, change it, and explain it in an interview. All MLflow code lives in one file, [`registry.py`](../registry.py).

## The two things MLflow does

MLflow has two parts that matter for this project.

**1. Tracking: a logbook of training runs.** Each time a model is trained, a *run* records:

- **parameters**: the inputs you chose (`n_estimators=200`, `max_depth=8`, how many rows)
- **metrics**: the results (`auc`, `f1`, `accuracy`, and for a retrain the champion's scores on the same data)
- **artifacts**: files, here the model itself and `reference.csv` (the training features, used later as the drift baseline)

Runs are grouped in an *experiment* called `credit-default`. The point is reproducibility: six months later you can answer "which settings and which data produced the model that was live in March?"

**2. Model registry: a catalogue of models that may go to production.** A *registered model* (`credit-default`) has numbered *versions*. Each version points at the model file of one run.

An **alias** is a movable name tag on one version. This project uses a single alias, `champion`, meaning "the version in production".

## How the app's concepts map to MLflow

| In the app | In MLflow |
|---|---|
| Train v1 (`train.py`) | A run, registered as version 1, `champion` → 1 |
| Retrain (`retrain.py`) | A new run and a new version, tagged with the comparison |
| Challenger wins | `champion` alias moves to the new version; the old one is tagged `archived` |
| Challenger loses | The version stays, tagged `rejected`; the alias doesn't move |
| "Active model" in the dashboard and API | Whatever version `champion` points at |
| Reset to v1 | The registered model is re-created with version 1 only |

**Why an alias and not "the newest version"?** Because newest is not the same as best. A rejected challenger is newer than the champion. Deployment code should ask for a *role* (`champion`), and promotion should be one small, auditable change: moving the alias. Rolling back is the same operation in reverse. (MLflow's older "stages", Staging and Production, did this job before; aliases replaced them.)

**Why keep rejected versions?** They're evidence. "We retrained on 14 March and it was worse, so we didn't ship it" is something an audit or a post-mortem needs.

## Design decisions

**SQLite as the backend.** MLflow needs a *backend store* (runs, metrics, versions) and an *artifact store* (files). Here they are `models/mlflow.db` and `models/mlartifacts/`. MLflow's plain-folder backend (`./mlruns`) is in maintenance mode, and the model registry needs a database anyway. SQLite is a database in a single file, so there's no server to run. In a company you would point the same code at a shared MLflow server backed by Postgres and S3; only the tracking URI changes.

**`mlflow-skinny`, not `mlflow`.** The app only needs the client. The full package adds the web server and UI and many more dependencies, which would make the API image bigger and slower to start. The UI runs from its own image (`Dockerfile.mlflow`) against the same database. The two must be the same MLflow version, because the server refuses a database schema from another version; a test checks that.

**Explicit clients, never global state.** Most MLflow tutorials call `mlflow.set_tracking_uri(...)` and `mlflow.start_run()`. Those set process-wide state. The dashboard serves many visitors from one process, each with their own store, so shared global state would let one visitor's run land in another visitor's store. `registry.py` always creates `MlflowClient(tracking_uri=..., registry_uri=...)` for a specific store.

**One store per visitor on the live demo.** Each visitor's workspace gets its own small MLflow store. Its version 1 is a *link* to the shared v1 model file (a registered version is just a pointer to an artifact location), so nothing is copied and nobody can change the shared model.

**Models saved as skops, not pickle.** A pickle file can run arbitrary code when it is loaded, so loading a model file from an untrusted or tampered source is a security hole. skops files list the types they contain and refuse anything not explicitly trusted. `registry.py` only trusts types from `sklearn`, `numpy` and `scipy`, and a test checks that anything else is refused.

**How the API notices a promotion.** On every request it asks the registry one small question: "which version and run does `champion` point at?" (about 2 ms). If the answer changed, it loads the new model. If the registry can't be read for a moment, it keeps serving the model it already has. The run ID is part of the check because version numbers restart after a reset.

**Everything else goes through `registry.py`.** The dashboard, API, retraining and checks call functions like `load_registry()`, `load_model()` and `register_version()`. None of them import MLflow. That keeps the MLflow-specific decisions in one place and made this migration a contained change.

## Trade-offs

- **Memory:** an API worker went from about 205 MB to about 330 MB, mostly the MLflow client and SQLAlchemy.
- **Start-up:** creating a new store runs MLflow's database migrations (about a second). A new visitor's first page load pays that once.
- **Telemetry:** MLflow sends anonymous usage data by default. This project turns it off (`MLFLOW_DISABLE_TELEMETRY`, `DO_NOT_TRACK`).
- **The MLflow UI has no login.** Docker Compose binds it to your machine only.
- **No "logged model" entities.** MLflow 3 also has a newer `LoggedModel` concept created by `mlflow.sklearn.log_model`. That function relies on the global state described above, so this project saves the model and attaches it to the run with the client instead. The registry behaves the same.

## Try it

```
python train.py
```

```
python retrain.py
```

```
pip install mlflow
```

```
mlflow ui --backend-store-uri sqlite:///models/mlflow.db
```

Then open http://localhost:5000. Under **Experiments → credit-default → Training runs** you'll see both runs with their parameters and metrics. Under **Models → credit-default** you'll see versions 1 and 2, with `@champion` on the one in production.

## Questions an interviewer might ask

- *What's the difference between tracking and the registry?* Tracking records every experiment. The registry holds the few models that are candidates for production, and says which one is live.
- *How do you roll back?* Move the `champion` alias back to the previous version. The API serves it on the next request.
- *How would this change at company scale?* A shared MLflow server with Postgres and object storage, authentication in front of it, and promotion done by a CI job or an approval step, not a button.
- *Why not pickle?* Loading a pickle can execute code. skops refuses types that aren't on an allowlist.

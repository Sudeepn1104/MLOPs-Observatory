# Runbook: operating the prediction API

What to do when an alert fires, how to ship a new model safely, and how to recover. Every alert in [`monitoring/alerts.yml`](../monitoring/alerts.yml) links to a section here.

Commands assume the project directory. With docker compose, prefix them with `docker compose exec api` (for example `docker compose exec api python shadow.py status`).

## Quick reference

| I want to… | Do this |
|---|---|
| See what's served | `GET /model`, or the Grafana dashboard's first panel |
| Check live drift | `GET /monitoring/drift` |
| Try a new model safely | `python retrain.py --shadow`, then `GET /shadow` |
| Promote the challenger | `python shadow.py promote` or `POST /shadow/promote` (admin key) |
| Roll back | `python shadow.py promote v1` (any earlier version) |
| Back up | `python backup.py --keep 7` |
| Restore | stop the API, `python backup.py restore backups/<file>.tar.gz`, start it |
| Rotate the API key | see [DEPLOY.md](../DEPLOY.md#3-operating-it) |

## Shipping a new model: shadow first

An offline holdout says how a model scores on past, labelled data. It can't say how the model behaves on the traffic you get today. A shadow deployment answers that before any client is affected.

1. **Train a challenger in shadow**: `python retrain.py --shadow`. It's registered in MLflow with the alias `challenger`; the champion keeps serving.
2. **Let it see traffic.** Every `/predict` is scored by both models. The challenger runs after the response is sent, so it adds no latency, and its failures never reach a client.
3. **Read the comparison**: `GET /shadow` (or `python shadow.py status`, or the dashboard's shadow row). It reports decision agreement, the share each model would flag, and the offline AUCs, with a verdict:

   | Verdict | Meaning | Action |
   |---|---|---|
   | `keep_watching` | fewer than 500 clients scored by both | wait |
   | `reject` | it lost to the champion on the labelled holdout | `python shadow.py stop` |
   | `review` | it would change the share flagged by more than 10 points | a person decides: is the change expected (the world drifted) or a bug? |
   | `promote` | better offline, similar behaviour live | `python shadow.py promote` |

4. **Promote**: the API serves the new champion from the next request, without a restart. The old champion is archived, not deleted, so a rollback is one command.

### Why `review` exists: an example from the demo

In the compose demo, v2 is trained on the most drifted batches. It beats v1 on that holdout (AUC 0.69 vs 0.58), so a purely offline rule would promote it. In shadow, over 8,440 live clients, it would have flagged **60%** of them where v1 flagged 35%. The two models agreed on only half the decisions, and their probabilities were almost uncorrelated (0.04).

The reason: the drifted data v2 learned from is hard to separate, so its F1-tuned threshold is low and it flags many clients to catch the defaulters. Whether that's acceptable is a business decision (more declined customers against fewer losses). The shadow run puts that number in front of a person before any client sees it, which is what the `review` verdict is for.

## Alerts

### API down
`ObservatoryApiDown`: Prometheus couldn't scrape `/metrics` for 2 minutes.

1. Is the process up? `GET /health`. On compose: `docker compose ps api` and `docker compose logs --tail 100 api`.
2. If `/health` works but `/ready` fails, the model can't be loaded: the message says why (usually the registry or a missing artifact). Restore the last backup if the store is damaged.
3. If `/health` and `/ready` work, check the scrape itself: Prometheus sends the first key of `API_KEY` as a Bearer token. A rotated key needs Prometheus restarted (`docker compose restart prometheus`).
4. Out of memory? `docker inspect -f '{{.State.OOMKilled}}' <container>`. One worker needs about 320 MB at its peak; a shadow challenger adds little (the model is 0.06 MB). Measured in the compose demo: 233 MB with a challenger in shadow.

### Errors
`ObservatoryHighErrorRate`: more than 5% of requests return 5xx for 5 minutes.

1. Find a failing request in the JSON logs: `docker compose logs api | grep '"status": 500'`. Each line has a `request_id`; the traceback is logged with the same ID.
2. Did it start with a promotion? Compare the start of the errors with `observatory_active_model_info` in Grafana. If so, **roll back**: `python shadow.py promote <previous version>`.
3. 4xx errors are the client's (validation, bad key) and don't count here.

### Slow
`ObservatorySlowPredictions`: `/predict` p95 above 500 ms for 10 minutes.

1. Large batches or `?explain=true` (SHAP) cost more: check the request rate by route.
2. CPU-starved host? The model caps its threads (`OBSERVATORY_THREADS`, default 2). See the thread-cap lesson in [MODEL.md](MODEL.md#a-production-lesson-parallelism-on-shared-hosts).
3. Scale out with more instances, not more workers per instance: metrics are per process.

### Drift
`ObservatoryLiveDrift`, `ObservatoryPsiAboveRetrainThreshold`, `ObservatoryPredictionShift`: the live inputs, or the model's outputs, no longer look like the training data.

Drift is a warning, not proof the model is wrong: labels arrive months later. In order:

1. **Which features?** The Grafana drift panels, or `GET /monitoring/drift`.
2. **Is it real?** A new client segment, a marketing campaign or a change upstream (units, a default value, a renamed category) all look like drift. A data bug needs fixing at the source, not a retrain.
3. **If the world changed**, train a challenger on recent labelled data **in shadow** (`python retrain.py --shadow`) and follow [Shipping a new model](#shipping-a-new-model-shadow-first). Don't promote blind: the example above shows why.

### Shadow
`ObservatoryShadowErrors`: the challenger fails to score. Clients are unaffected, but the comparison is incomplete. The traceback is in the logs as `shadow scoring failed`. Usually the challenger's artifact is missing or incompatible; `python shadow.py stop` removes it.

`ObservatoryShadowDisagreement` (info): the challenger disagrees on more than 20% of decisions. Expected right after a big drift; read the `/shadow` report before promoting.

## Capacity

Measured with `python loadtest.py` against the compose API: one worker, capped at 2 CPUs, with a challenger in shadow (so every request is scored twice and logged twice).

| Concurrency | Clients per request | Requests/s | Clients/s | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| 1 | 1 | 76 | 76 | 12 ms | 16 ms | 26 ms |
| 8 | 1 | 65 | 65 | 111 ms | 216 ms | 283 ms |
| 8 | 20 | 69 | 1,371 | 38 ms | 473 ms | 1,457 ms |
| 32 | 1 | 64 | 64 | 480 ms | 756 ms | 937 ms |

- One worker tops out around 65–75 requests/s; more concurrency only queues requests. Batching is the cheap win: 20 clients per request costs about the same as one.
- To scale, add instances behind a load balancer. Each needs about 250 MB.
- The first version of this API managed 34 requests/s at 29 ms. Profiling showed two registry lookups (about 3 ms each) and the model scoring every request twice (`predict_proba`, then `predict`). Now the alias lookup is cached for `OBSERVATORY_ALIAS_TTL` (1 s; promotions through the API apply at once), and the decision comes from the probability and the threshold.

## Backups and recovery

`python backup.py --keep 7` writes `backups/observatory-<time>.tar.gz` with the MLflow store (runs, versions, aliases, model files) and the prediction log. It is consistent while the API is running (SQLite's online backup). Schedule it daily: cron, a Kubernetes CronJob, or with compose:

```bash
docker compose run --rm -v "$PWD/backups:/app/backups" api python backup.py --keep 7
```

**Restore** with the API stopped, then start it again. It serves whatever the restored champion alias points at:

```bash
docker compose stop api
docker compose run --rm -v "$PWD/backups:/app/backups" api python backup.py restore backups/<file>.tar.gz
docker compose start api
```

A restore refuses archives that contain anything other than the store and the log.

## Security notes

- Two keys: `API_KEY` lets a client score and read; `ADMIN_API_KEY` is needed to change which model is served. In production the admin endpoints are off unless `ADMIN_API_KEY` is set, and the two must differ.
- Keys are compared in constant time and never logged.
- Prometheus, Grafana and the MLflow UI listen on localhost only in compose. Put them behind your SSO before exposing them.
- Containers run as non-root, with all Linux capabilities dropped, no privilege escalation, memory caps and rotated logs.
- Models are stored in the skops format, which can't run code when loaded.

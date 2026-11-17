# Deploying the Observatory

| Part | Where it runs | How it updates |
|---|---|---|
| Dashboard (`app.py`) | Streamlit Community Cloud | Redeploys on every push to `main` |
| Prediction API (`serve.py`) | Render, from `render.yaml` | Redeploys on every push to `main` |
| Docker image | GitHub Container Registry | Published after CI passes on `main`, and for `v*` tags |
| Scheduled check | GitHub Actions | Every Monday, and on demand |

## 1. Deploy the API on Render (one-time, about 5 minutes)

1. Sign in at https://dashboard.render.com with GitHub.
2. **New → Blueprint**, pick this repository, and click **Apply**. Render reads `render.yaml`, builds the `Dockerfile` and generates a random `API_KEY`.
3. When the deploy turns **Live**, open the service. Copy the URL (`https://mlops-observatory-api-XXXX.onrender.com`) and, under **Environment**, the `API_KEY` value.
4. Try it:

   ```
   curl https://YOUR-URL/ready
   ```

   ```
   curl -X POST https://YOUR-URL/predict -H "X-API-Key: YOUR-KEY" -H 'content-type: application/json' -d '{"clients": [{"credit_limit": 50000, "age": 29, "education": 2, "marriage": 2, "repay_status_sep": 2, "repay_status_aug": 2, "repay_status_jul": 0, "bill_sep": 48000, "paid_sep": 0, "paid_aug": 1500}]}'
   ```

   Interactive docs are at `https://YOUR-URL/docs` (click **Authorize** and paste the key).

From then on, every push to `main` redeploys. Render only switches traffic once `/ready` answers, so a broken build never replaces a working one.

**Free plan limits:**
- The service sleeps after 15 minutes without traffic. The first request after that takes about a minute.
- 512 MB RAM fits one worker (each holds the model and the MLflow client, about 320 MB).
- There is no persistent disk, so the prediction log and the MLflow store reset to the image's v1 on every deploy and restart. On a paid plan, uncomment the `disk:` block in `render.yaml` to keep it.

## 2. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `API_KEY` | (none) | Accepted key(s), comma-separated. Required: the image runs with `OBSERVATORY_ENV=production` and refuses to start without one. Keys must be at least 16 characters. |
| `OBSERVATORY_ENV` | `production` in the image, `development` otherwise | Production requires a key and logs as JSON |
| `PORT` | 8000 (API), 8501 (dashboard) | Port to listen on. Render sets it. |
| `API_WORKERS` | 1 | uvicorn workers. Each needs about 320 MB. Scale out with more instances rather than more workers, because metrics are per process. |
| `MAX_REQUEST_BYTES` | 1000000 | Larger request bodies get 413 before they're parsed |
| `LOG_LEVEL` | INFO | |
| `GIT_SHA` | build arg, or Render's `RENDER_GIT_COMMIT` | Reported by `/version` |

## 3. Operating it

**Health:** `/health` (process up) and `/ready` (model loaded) are open for load balancers. Everything else needs `X-API-Key`.

**Version:** `GET /version` returns the app version, the git commit and the active model.

**Logs:** one JSON line per request with `request_id`, route, status and latency. Send `X-Request-ID` to set your own ID. It comes back in the response header and body, and it's stored with each prediction. API keys are never logged.

**Metrics:** `GET /metrics` (with the key) in Prometheus format:
- request counts and latency per route
- predictions and predicted defaults per model version
- active model
- the max PSI seen by the last drift check

**Live drift:** `GET /monitoring/drift?window=500` compares the last 500 scored clients with the active model's training data. It needs no labels.

**Alerts on live traffic:** add two repository secrets (**Settings → Secrets and variables → Actions**), `OBSERVATORY_API_URL` (your Render URL) and `OBSERVATORY_API_KEY`. The weekly scheduled check then includes live drift, and a red result opens the `drift-alert` issue.

**Rotating the key:**
1. Set `API_KEY` to `new-key,old-key` and save. Render redeploys.
2. Move your clients to the new key.
3. Set `API_KEY` to `new-key` alone.

**Rolling back:** in Render, open **Events**, pick an earlier deploy and choose **Rollback**. Or pin an image tag (next section).

## 4. Running the published image anywhere

After CI passes on `main`, `.github/workflows/release.yml` publishes:

- `ghcr.io/sudeepn1104/mlops-observatory:latest` (newest `main`)
- `:sha-<7 chars>` (every `main` commit)
- `:1.2.3` and `:1.2` (when you push a tag such as `v1.2.3`)

```
docker run -p 8000:8000 -e API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))") ghcr.io/sudeepn1104/mlops-observatory:latest
```

Each image is scanned with Trivy (the release fails on fixable critical vulnerabilities) and published with an SBOM and signed build provenance. Verify it with:

```
gh attestation verify oci://ghcr.io/sudeepn1104/mlops-observatory:latest --owner Sudeepn1104
```

New GHCR packages can start private. If `docker pull` is denied, open the package on GitHub (**Packages → mlops-observatory → Package settings**) and set the visibility to public.

Locally, with the dashboard and the MLflow UI alongside:

```
cp .env.example .env    # set API_KEY
docker compose up --build
```

The MLflow UI (http://localhost:5000) has no login, so compose binds it to this machine only. Don't expose it to the internet without putting authentication in front of it.

## 5. Dashboard on Streamlit Community Cloud

It's already deployed from this repo (`app.py`, `requirements.txt`, `.streamlit/config.toml`). Each visitor gets a private sandbox, and error details are hidden from visitors. If you redeploy it, choose Python 3.12 under **Advanced settings**.

## 6. What keeps it healthy

- **CI**, on every push and pull request:
  - lint and 60+ tests on the oldest and newest supported versions
  - a dependency audit of `requirements-lock.txt`
  - a check that the lock file matches the requirements
  - an image build and smoke test, including that the image refuses to start without a key
- **Reproducible builds:** the image installs exact, hash-checked versions from `requirements-lock.txt` on a base image pinned by digest. To change dependencies, edit `requirements-api.txt` (or `requirements.txt`), then re-run the `uv pip compile` command at the top of the lock file.
- **Dependabot** proposes weekly updates for Python packages, the base image and the GitHub Actions in use.

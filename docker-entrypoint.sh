#!/bin/sh
# Starts the prediction API (default) or the dashboard. Anything else is run as a command.
#   PORT          port to listen on (hosts like Render set it; default 8000 / 8501)
#   API_WORKERS   uvicorn worker processes (default 1; each holds the model, ~200 MB)
set -e
case "$1" in
  api)
    exec uvicorn serve:app --host 0.0.0.0 --port "${PORT:-8000}" \
      --workers "${API_WORKERS:-1}" \
      --proxy-headers --forwarded-allow-ips "*" \
      --no-access-log --timeout-graceful-shutdown 20
    ;;
  dashboard)
    exec python -m streamlit run app.py --server.headless true --server.port "${PORT:-8501}" \
      --server.address 0.0.0.0 --browser.gatherUsageStats false
    ;;
  *)
    exec "$@"
    ;;
esac

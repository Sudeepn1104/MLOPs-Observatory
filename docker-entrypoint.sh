#!/bin/sh
# Starts the prediction API (default) or the dashboard. Anything else is run as a command.
set -e
case "$1" in
  api)
    exec uvicorn serve:app --host 0.0.0.0 --port 8000 --workers "${API_WORKERS:-2}"
    ;;
  dashboard)
    exec python -m streamlit run app.py --server.headless true --server.port 8501 \
      --server.address 0.0.0.0 --browser.gatherUsageStats false
    ;;
  *)
    exec "$@"
    ;;
esac

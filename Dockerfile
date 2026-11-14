# syntax=docker/dockerfile:1
# One image for both services: the prediction API (default) and the dashboard.
#   docker build -t observatory .
#   docker run -p 8000:8000 observatory                     # API at :8000/docs
#   docker run -p 8501:8501 observatory dashboard           # dashboard at :8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# dependencies first, so code changes don't reinstall them
COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements-api.txt

COPY *.py ./
COPY data/credit_default_raw.csv.gz data/SOURCE.md ./data/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# build model v1 and the monitoring history into the image, so a container
# serves predictions immediately with no download or training at start-up
RUN python make_data.py && python train.py && python simulate.py \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /app \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

USER app
EXPOSE 8000 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "healthcheck.py"]

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["api"]

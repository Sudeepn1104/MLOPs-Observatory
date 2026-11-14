# syntax=docker/dockerfile:1
# One image for both services: the prediction API (default) and the dashboard.
#   docker build -t observatory .
#   docker run -p 8000:8000 -e API_KEY=<16+ chars> observatory        # API at :8000/docs
#   docker run -p 8501:8501 observatory dashboard                       # dashboard at :8501
# The image runs in production mode: the API refuses to start without API_KEY.
# Base image pinned by digest; Dependabot proposes updates.
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

ARG GIT_SHA=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OBSERVATORY_ENV=production \
    GIT_SHA=${GIT_SHA}

LABEL org.opencontainers.image.title="mlops-observatory" \
      org.opencontainers.image.description="Credit-default model API and monitoring dashboard" \
      org.opencontainers.image.source="https://github.com/Sudeepn1104/MLOPs-Observatory" \
      org.opencontainers.image.revision=${GIT_SHA}

WORKDIR /app

# exact, hash-checked dependencies first, so code changes don't reinstall them
COPY requirements-lock.txt ./
RUN pip install --require-hashes --no-deps -r requirements-lock.txt

COPY *.py ./
COPY data/credit_default_raw.csv.gz data/SOURCE.md ./data/
COPY .streamlit ./.streamlit
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

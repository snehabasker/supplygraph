# syntax=docker/dockerfile:1
# SupplyGraph: one image for the API/UI server and the pipeline job.

FROM node:22.11.0-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.11.10-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SUPPLYGRAPH_DATA_DIR=/data \
    SUPPLYGRAPH_LOG_FORMAT=json \
    HOME=/tmp
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# No Dagster in the image: in Kubernetes the CronJob schedules the pipeline (docs/adr/0011).
RUN pip install ".[llm,ui]"
COPY seeds ./seeds
COPY eval/agent ./eval/agent
COPY --from=web /web/dist ./web/dist
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin supplygraph \
    && mkdir -p /data && chown 10001:10001 /data
USER 10001
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
ENTRYPOINT ["supplygraph"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
